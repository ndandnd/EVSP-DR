"""Exact column generation via an SOC x time expanded pricing network.

Instead of SPPRC labeling with dominance, discretized battery state lives in
the network NODES (de Vos, van Lieshout & Dollevoet, Transportation Science;
arXiv:2207.13734): trip nodes are (trip, SOC-at-trip-start) and charging nodes
are (station, time-block, SOC-before-charging). Pricing is then a plain
shortest path in a DAG, processed once in topological order per CG iteration:

  * no labels, no dominance, no queue starvation, no timeouts;
  * termination with min reduced cost >= -eps is a CERTIFICATE that no
    improving column exists in the expanded route space;
  * charging may start at ANY later block after arrival, so delayed-start
    (price-responsive) charging is native — the heuristic DP cannot express it.

Conservative rounding (SOC floored to the grid) keeps every generated duty
feasible for the continuous model. Two deliberate relaxations vs the runner's
restricted DP: recharge count is uncapped (the per-start fee discourages
excess; routes exceeding MAX_DAILY_RECHARGES are reported), and charging
energy per block is floored to the SOC grid (an effective-rate reduction).

Usage (from src/):

    python exact_pricer_expanded.py \
        --csv Practice_Custom_TwoDuty_13301_13302.csv \
        --prices_csv hourly_prices_flat.csv \
        --soc-step 15 --block-min 10 --max-iters 400
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from audit_giro_known_columns import (
    DEPOT,
    HORIZON_MIN,
    MAX_DAILY_RECHARGES,
    STATIONS,
    build_problem,
)
from config import BIG_M_PENALTY, BUS_COST_KX, CHARGE_RATE_KW, CHARGE_START_COST, CHARGING_STATIONS
from master_lp_scipy import build_route_incidence, solve_restricted_master_lp
from utils_v2 import base_station_name, load_station_hourly_prices

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
G_KWH = 300.0


def direct_singleton_seed_records(
    problem,
    *,
    g_kwh: float,
    soc_step: float,
    reserve_kwh: float,
) -> tuple[list[dict], list[int]]:
    """Build depot-trip-depot columns feasible in the expanded SOC grid.

    These deterministic one-trip routes form an integer partition whenever
    every trip can leave from and return directly to the depot.  They are a
    safe restricted-master initializer: expensive, but real model columns
    rather than BIG-M artificials.  Trips without a direct singleton are
    returned separately so callers never mistake a partial seed for a
    partition certificate.
    """

    g = float(g_kwh)
    step = float(soc_step)
    reserve = float(reserve_kwh)
    if g <= 0 or step <= 0 or reserve < 0:
        raise ValueError("g_kwh and soc_step must be positive; reserve_kwh >= 0")
    grid = [round(level * step, 6) for level in range(int(g / step) + 1)]

    def floor_soc(soc: float) -> float:
        level = min(
            max(int(math.floor((soc + 1e-9) / step)), 0),
            len(grid) - 1,
        )
        return grid[level]

    depot_trip: dict[int, tuple[float, float]] = {}
    trip_depot: dict[int, tuple[float, float]] = {}
    for node, arcs in problem.adjacency.items():
        for succ, travel_min, deadhead_kwh, arc_type in arcs:
            if arc_type == "depot_trip":
                depot_trip[succ] = (travel_min, deadhead_kwh)
            elif arc_type == "trip_depot":
                trip_depot[node] = (travel_min, deadhead_kwh)

    records: list[dict] = []
    missing: list[int] = []
    for trip in problem.trips:
        if trip not in depot_trip or trip not in trip_depot:
            missing.append(trip)
            continue
        outbound_min, outbound_kwh = depot_trip[trip]
        start_soc = floor_soc(g - outbound_kwh)
        if outbound_min > problem.start_min[trip] + 1e-9:
            missing.append(trip)
            continue
        if start_soc + 1e-9 < problem.trip_energy[trip] + reserve:
            missing.append(trip)
            continue
        return_min, return_kwh = trip_depot[trip]
        exit_soc = start_soc - problem.trip_energy[trip]
        if problem.end_min[trip] + return_min > HORIZON_MIN + 1e-9:
            missing.append(trip)
            continue
        if exit_soc - return_kwh < reserve - 1e-9:
            missing.append(trip)
            continue
        records.append({
            "trips": [trip],
            "cost": float(BUS_COST_KX),
            "route_nodes": [DEPOT, trip, DEPOT],
            "charging_stops": {
                "stations": [], "cst": [], "cet": [], "kwh": [],
            },
            "charges_started": 0,
            "found_iter": 0,
            "origin": "exact_direct_singleton_seed",
        })
    return records, missing


class ExpandedNetwork:
    """Static expanded DAG; arc costs are dual-free, trip duals applied on the fly."""

    def __init__(self, problem, station_prices, *, soc_step: float, block_min: int,
                 g_kwh: float = G_KWH, charge_kw: float = CHARGE_RATE_KW,
                 reserve_kwh: float = 0.0):
        self.problem = problem
        self.soc_step = float(soc_step)
        self.block_min = int(block_min)
        self.g = float(g_kwh)
        self.reserve = float(reserve_kwh)
        self.n_blocks = int(HORIZON_MIN) // self.block_min
        self.block_kwh = float(charge_kw) * self.block_min / 60.0
        self.prices = station_prices  # base station -> {hour: $/kWh}

        self.grid = [round(k * self.soc_step, 6)
                     for k in range(int(self.g / self.soc_step) + 1)]
        self._floor = lambda soc: min(
            max(int(math.floor((soc + 1e-9) / self.soc_step)), 0),
            len(self.grid) - 1,
        )

        # adjacency split by arc type for direct access
        self.trip_trip: dict[int, list] = {}
        self.trip_station: dict[int, list] = {}
        self.station_trip: dict[str, list] = {}
        self.depot_trip: dict[int, tuple] = {}
        self.trip_depot: dict[int, tuple] = {}
        self.station_depot: dict[str, tuple] = {}
        for node, arcs in problem.adjacency.items():
            for succ, travel_min, dh_kwh, arc_type in arcs:
                if arc_type == "trip_trip":
                    self.trip_trip.setdefault(node, []).append((succ, travel_min, dh_kwh))
                elif arc_type == "trip_station":
                    self.trip_station.setdefault(node, []).append((succ, travel_min, dh_kwh))
                elif arc_type == "station_trip":
                    self.station_trip.setdefault(node, []).append((succ, travel_min, dh_kwh))
                elif arc_type == "depot_trip":
                    self.depot_trip[succ] = (travel_min, dh_kwh)
                elif arc_type == "trip_depot":
                    self.trip_depot[node] = (travel_min, dh_kwh)
                elif arc_type == "station_depot":
                    self.station_depot[node] = (travel_min, dh_kwh)

        self._build_nodes()
        self._build_arcs()

    # ── nodes ─────────────────────────────────────────────────────────────
    def _build_nodes(self):
        p = self.problem
        self.node_meta = [("source", None, None)]  # id 0
        self.SINK = 1
        self.node_meta.append(("sink", None, None))
        self.trip_node: dict[tuple[int, int], int] = {}
        self.charge_node: dict[tuple[str, int, int], int] = {}

        order = []  # (time_key, tiebreak, node_id)
        for trip in p.trips:
            for level in range(len(self.grid)):
                if self.grid[level] + 1e-9 >= p.trip_energy[trip] + self.reserve:
                    node_id = len(self.node_meta)
                    self.node_meta.append(("trip", trip, level))
                    self.trip_node[(trip, level)] = node_id
                    order.append((p.start_min[trip], 0, node_id))
        for station in STATIONS:
            for block in range(self.n_blocks):
                for level in range(len(self.grid)):
                    node_id = len(self.node_meta)
                    self.node_meta.append(("charge", (station, block), level))
                    self.charge_node[(station, block, level)] = node_id
                    order.append((block * self.block_min, 1, node_id))
        order.sort()
        self.topo = [0] + [node_id for _, _, node_id in order] + [1]

    # ── arcs ──────────────────────────────────────────────────────────────
    def _price(self, station: str, minute: float) -> float:
        curve = self.prices[base_station_name(station)]
        hour = int(minute // 60)
        return curve.get(hour, curve[max(curve)])

    def _charge_result(self, level: int) -> float:
        return self.grid[self._floor(min(self.g, self.grid[level] + self.block_kwh))]

    def _build_arcs(self):
        p = self.problem
        grid, floor = self.grid, self._floor
        # out[node] = list of (succ_id, base_cost, trip_entered_or_-1)
        self.out: list[list] = [[] for _ in self.node_meta]
        add = lambda u, v, cost, trip=-1: self.out[u].append((v, cost, trip))

        # source -> trip
        for trip, (travel, dh) in self.depot_trip.items():
            if travel <= p.start_min[trip] + 1e-9:
                level = floor(self.g - dh)
                node = self.trip_node.get((trip, level))
                if node is not None:
                    add(0, node, BUS_COST_KX, trip)

        for (trip, level), u in self.trip_node.items():
            soc_exit = grid[level] - p.trip_energy[trip]
            depart = p.end_min[trip]
            # trip -> sink
            if trip in self.trip_depot:
                travel, dh = self.trip_depot[trip]
                if depart + travel <= HORIZON_MIN + 1e-9 and soc_exit - dh >= self.reserve - 1e-9:
                    add(u, 1, 0.0)
            # trip -> trip
            for succ, travel, dh in self.trip_trip.get(trip, ()):  # gap-filtered upstream
                nxt = floor(soc_exit - dh)
                if grid[nxt] > soc_exit - dh + 1e-9:
                    nxt -= 1
                if nxt < 0:
                    continue
                if depart + travel > p.start_min[succ] + 1e-9:
                    continue
                node = self.trip_node.get((succ, nxt))
                if node is not None:
                    add(u, node, 0.0, succ)
            # trip -> charge(station, any block starting after arrival)
            for station, travel, dh in self.trip_station.get(trip, ()):
                arrival = depart + travel
                soc_arr = soc_exit - dh
                if soc_arr < self.reserve - 1e-9:
                    continue
                lvl = floor(soc_arr)
                if grid[lvl] > soc_arr + 1e-9:
                    lvl -= 1
                if lvl < 0:
                    continue
                first_block = int(math.ceil(arrival / self.block_min - 1e-9))
                for block in range(max(first_block, 0), self.n_blocks):
                    add(u, self.charge_node[(station, block, lvl)], CHARGE_START_COST)

        for (station, block, level), u in self.charge_node.items():
            soc_after = self._charge_result(level)
            gained = soc_after - grid[level]
            cost = gained * self._price(station, block * self.block_min) if gained > 1e-9 else 0.0
            after_level = floor(soc_after)
            block_end = (block + 1) * self.block_min
            # continue charging next block (only if something was gained)
            if gained > 1e-9 and block + 1 < self.n_blocks:
                add(u, self.charge_node[(station, block + 1, after_level)], cost)
            # leave to a trip after this block
            for succ, travel, dh in self.station_trip.get(station, ()):
                if block_end + travel > p.start_min[succ] + 1e-9:
                    continue
                nxt = floor(soc_after - dh)
                if grid[nxt] > soc_after - dh + 1e-9:
                    nxt -= 1
                if nxt < 0:
                    continue
                node = self.trip_node.get((succ, nxt))
                if node is not None:
                    add(u, node, cost, succ)
            # leave to sink
            if station in self.station_depot:
                travel, dh = self.station_depot[station]
                if block_end + travel <= HORIZON_MIN + 1e-9 and soc_after - dh >= self.reserve - 1e-9:
                    add(u, 1, cost)

        self.n_arcs = sum(len(a) for a in self.out)

    # ── exact pricing pass ────────────────────────────────────────────────
    def min_reduced_cost_route(self, alpha: dict[int, float]):
        INF = float("inf")
        value = [INF] * len(self.node_meta)
        parent: list[tuple[int, int] | None] = [None] * len(self.node_meta)
        value[0] = 0.0
        for u in self.topo:
            vu = value[u]
            if vu == INF:
                continue
            for v, cost, trip in self.out[u]:
                cand = vu + cost - (alpha.get(trip, 0.0) if trip >= 0 else 0.0)
                if cand < value[v] - 1e-12:
                    value[v] = cand
                    parent[v] = (u, trip)
        if value[1] == INF:
            return None

        def _walk(from_node):
            """Reconstruct the full path: ordered trips + charging events."""
            nodes, node = [], from_node
            while node != 0:
                nodes.append(node)
                node = parent[node][0]
            nodes.reverse()  # source-side first (sink excluded when from_node=1? no: included)

            trips, stops = [], []
            run = None  # open charging run: [station, first_block, last_block, entry_level]
            for nid in nodes:
                kind, key, level = self.node_meta[nid]
                if kind == "trip":
                    if run is not None:
                        stops.append(run)
                        run = None
                    trips.append(key)
                elif kind == "charge":
                    station, block = key
                    if run is not None and run[0] == station and block == run[2] + 1:
                        run[2] = block
                    else:
                        if run is not None:
                            stops.append(run)
                        run = [station, block, block, level]
            if run is not None:
                stops.append(run)

            charging = {"stations": [], "cst": [], "cet": [], "kwh": []}
            route_nodes = [DEPOT]
            # interleave trips and stops in path order for the route node list
            seq = []
            for nid in nodes:
                kind, key, level = self.node_meta[nid]
                if kind == "trip":
                    seq.append(("t", key))
                elif kind == "charge":
                    if not seq or seq[-1] != ("s", key[0]):
                        seq.append(("s", key[0]))
            # collapse consecutive same-station markers (one per charging run)
            collapsed = []
            for item in seq:
                if not collapsed or item != collapsed[-1]:
                    collapsed.append(item)
            for tag, val in collapsed:
                route_nodes.append(val if tag == "t" else val)
            route_nodes.append(DEPOT)

            for station, b0, b1, lvl0 in stops:
                soc = self.grid[lvl0]
                for _ in range(b0, b1 + 1):
                    soc = self.grid[self._floor(min(self.g, soc + self.block_kwh))]
                charging["stations"].append(station)
                charging["cst"].append(b0 * self.block_min)
                charging["cet"].append((b1 + 1) * self.block_min)
                charging["kwh"].append(round(soc - self.grid[lvl0], 6))

            return trips, charging, route_nodes

        best_trips, best_charging, best_nodes = _walk(1)
        return {"rc": value[1], "trips": best_trips,
                "charging_stops": best_charging, "route_nodes": best_nodes,
                "charges_started": len(best_charging["stations"]),
                "_value": value, "_walk": _walk}

        best_trips, best_charges = _walk(1)
        return {"rc": value[1], "trips": best_trips, "charges_started": best_charges,
                "_value": value, "_walk": _walk}

    def k_best_routes(self, alpha: dict[int, float], k: int = 30):
        """Best route plus up to k-1 additional negative columns from the same
        pass: min-cost paths ending at the k best distinct sink-predecessors."""
        best = self.min_reduced_cost_route(alpha)
        if best is None:
            return []
        value, _walk = best.pop("_value"), best.pop("_walk")
        candidates = []
        for u in range(2, len(self.node_meta)):
            if value[u] == float("inf"):
                continue
            for v, cost, _trip in self.out[u]:
                if v == 1:
                    candidates.append((value[u] + cost, u))
        candidates.sort()
        routes, seen = [best], {frozenset(best["trips"])}
        for rc, u in candidates[: max(4 * k, 200)]:
            if len(routes) >= k or rc >= -1e-9:
                break
            trips, charging, route_nodes = _walk(u)
            key = frozenset(trips)
            if key in seen:
                continue
            seen.add(key)
            routes.append({"rc": rc, "trips": trips,
                           "charging_stops": charging, "route_nodes": route_nodes,
                           "charges_started": len(charging["stations"])})
        return routes


def _provenance(args) -> dict:
    import platform
    import scipy
    import subprocess

    def _git(*a):
        r = subprocess.run(["git", *a], cwd=Path(__file__).resolve().parent,
                           text=True, capture_output=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else None

    def _sha(path):
        import hashlib
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "scipy": scipy.__version__,
        "instance_sha256": _sha(DATA_DIR / args.csv),
        "prices_sha256": _sha(DATA_DIR / args.prices_csv),
        "rc_eps": args.rc_eps,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }


def run_cg(args) -> dict:
    t0 = time.time()
    problem = build_problem(DATA_DIR, args.csv,
                            max_station_to_trip_wait_min=HORIZON_MIN)
    prices = load_station_hourly_prices(DATA_DIR / args.prices_csv, CHARGING_STATIONS)
    net = ExpandedNetwork(problem, prices,
                          soc_step=args.soc_step, block_min=args.block_min,
                          g_kwh=args.g_kwh, charge_kw=args.charge_kw,
                          reserve_kwh=args.min_soc_frac * args.g_kwh)
    provenance = _provenance(args)
    journal_path = Path(str(args.out) + ".columns.jsonl") if args.out else None
    build_s = time.time() - t0
    print(f"[EXACT] network: {len(net.node_meta):,} nodes, {net.n_arcs:,} arcs "
          f"(soc_step={args.soc_step}, block={args.block_min}min) "
          f"built in {build_s:.1f}s", flush=True)

    trips = list(problem.trips)
    pool: dict[frozenset, dict] = {}
    history = []
    stall_hist = []  # (elapsed_s, lp_obj, min_rc) for --stall-window-min
    certified = False
    stop_reason = "max_iters"
    stall_count = 0
    method_order = ("highs-ds", "highs-ipm", "highs")

    if args.resume and journal_path and journal_path.exists():
        with open(journal_path) as fh:
            for line in fh:
                rec = json.loads(line)
                key = frozenset(rec["trips"])
                if key not in pool or rec["cost"] < pool[key]["cost"] - 1e-9:
                    pool[key] = rec
        print(f"[EXACT] resumed {len(pool)} columns from {journal_path.name}",
              flush=True)
    journal = open(journal_path, "a") if journal_path else None

    # Per-iteration stopping-rule instrumentation: append-only CSV so the
    # timing campaign can reconstruct the full LP trajectory with wall time.
    iters_csv = None
    elapsed_offset = 0.0
    if args.out:
        iters_path = Path(str(args.out) + ".iters.csv")
        fresh = not (args.resume and iters_path.exists())
        if not fresh:
            # cumulative elapsed across resumes/requeues, so stopping curves
            # stay monotone instead of restarting at zero
            try:
                last = iters_path.read_text().strip().rsplit("\n", 1)[-1]
                elapsed_offset = float(last.split(",")[0])
            except (ValueError, OSError, IndexError):
                elapsed_offset = 0.0
        iters_csv = open(iters_path, "a")
        if fresh:
            iters_csv.write("elapsed_s,iteration,lp_obj,route_weight,"
                            "artificials,min_rc,pool_columns\n")

    # Immutable timed pool snapshots (status + journal copy) for CG-vs-MIP
    # budget calibration, e.g. --snapshot-at-minutes 15,60,180,360.
    snapshot_marks = sorted(
        float(m) for m in str(args.snapshot_at_minutes or "").split(",")
        if m.strip())

    def _freeze_snapshot(mark):
        if not args.out:
            return
        import shutil
        stem = Path(str(args.out).replace(".json", ""))
        snap_json = Path(f"{stem}.m{int(mark)}.snapshot.json")
        if snap_json.exists():
            # snapshots are immutable: a requeued run must never overwrite the
            # original N-minute pool with a later one
            print(f"[EXACT] snapshot {snap_json.name} already frozen — keeping "
                  "the original", flush=True)
            return
        _write_partial(f"snapshot_m{int(mark)}")
        shutil.copyfile(args.out, snap_json)
        if journal:
            journal.flush()
        if journal_path and journal_path.exists():
            shutil.copyfile(journal_path,
                            Path(f"{stem}.m{int(mark)}.snapshot.json.columns.jsonl"))
        # snapshots need trip_ids for the MIP loader
        with open(snap_json) as fh:
            snap = json.load(fh)
        snap["trip_ids"] = trips
        snap["columns_journal"] = f"{stem}.m{int(mark)}.snapshot.json.columns.jsonl"
        with open(snap_json, "w") as fh:
            json.dump(snap, fh, indent=1)
        print(f"[EXACT] froze snapshot at {mark:g} min: {snap_json.name}",
              flush=True)

    singleton_seeds, missing_singletons = direct_singleton_seed_records(
        problem,
        g_kwh=args.g_kwh,
        soc_step=args.soc_step,
        reserve_kwh=args.min_soc_frac * args.g_kwh,
    )
    seeds_added = 0
    for record in singleton_seeds:
        key = frozenset(record["trips"])
        if key not in pool or record["cost"] < pool[key]["cost"] - 1e-9:
            pool[key] = record
            if journal:
                journal.write(json.dumps(record) + "\n")
            seeds_added += 1
    if journal and seeds_added:
        journal.flush()
    print(
        f"[EXACT] direct-singleton seed: {len(singleton_seeds)}/{len(trips)} "
        f"trips feasible ({seeds_added} added to pool)",
        flush=True,
    )
    if missing_singletons:
        print(
            "[EXACT] WARNING: direct-singleton seed is not a full partition; "
            f"missing {len(missing_singletons)} trips "
            f"({missing_singletons[:15]}).",
            flush=True,
        )

    def _write_partial(status):
        if not args.out:
            return
        partial = {
            "csv": args.csv, "prices_csv": args.prices_csv,
            "soc_step": args.soc_step, "block_min": args.block_min,
            "g_kwh": args.g_kwh, "charge_kw": args.charge_kw,
            "min_soc_frac": args.min_soc_frac,
            "master_sense": args.master_sense,
            "trip_ids": trips,
            "iterations": len(history), "certified_rc_optimal": certified,
            "final": history[-1] if history else None,
            "columns": len(pool), "wall_s": time.time() - t0,
            "stop_reason": status, "history_tail": history[-5:],
        }
        tmp = Path(str(args.out) + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(partial, fh, indent=1)
        tmp.replace(args.out)
    class _ArtificialOnlyLP:
        objective = len(trips) * BIG_M_PENALTY
        route_weight = 0.0
        artificial_total = float(len(trips))
        trip_duals = {t: float(BIG_M_PENALTY) for t in trips}

    for iteration in range(1, args.max_iters + 1):
        if args.wall_limit_s and time.time() - t0 > args.wall_limit_s:
            print(f"[EXACT] wall limit {args.wall_limit_s}s reached — stopping "
                  "gracefully (partial result saved)", flush=True)
            stop_reason = "wall_limit"
            break
        if args.out and iteration % args.checkpoint_every == 0:
            _write_partial("running")
        routes = list(pool.values())
        if routes:
            incidence = build_route_incidence(
                trip_ids=trips,
                route_trip_ids=[r["trips"] for r in routes],
            )
            lp = None
            for method in method_order:
                try:
                    lp = solve_restricted_master_lp(
                        trip_ids=trips,
                        route_incidence=incidence,
                        route_costs=[r["cost"] for r in routes],
                        artificial_penalty=BIG_M_PENALTY,
                        method=method,
                        coverage_sense=args.master_sense,
                    )
                    break
                except Exception as exc:  # HiGHS can stall on degenerate pools
                    print(f"[EXACT] master failed with {method}: {exc}; "
                          "retrying with next method", flush=True)
            if lp is None:
                print("[EXACT] all master methods failed — stopping uncertified",
                      flush=True)
                stop_reason = "master_failed"
                break
        else:
            lp = _ArtificialOnlyLP()
        batch = net.k_best_routes(lp.trip_duals, k=args.columns_per_iter)
        best = batch[0] if batch else None
        min_rc = best["rc"] if best else float("inf")
        history.append({"iter": iteration, "lp_obj": lp.objective,
                        "route_weight": lp.route_weight,
                        "artificials": lp.artificial_total, "min_rc": min_rc})
        if iters_csv:
            iters_csv.write(f"{elapsed_offset + time.time() - t0:.2f},{iteration},"
                            f"{lp.objective:.6f},{lp.route_weight:.9f},"
                            f"{lp.artificial_total:.6f},{min_rc:.6f},{len(pool)}\n")
            iters_csv.flush()
        while snapshot_marks and (elapsed_offset + time.time() - t0) >= snapshot_marks[0] * 60:
            mark = snapshot_marks.pop(0)
            _freeze_snapshot(mark)
        if iteration % 10 == 0 or min_rc >= -args.rc_eps:
            print(f"[EXACT] it {iteration:3d}: obj={lp.objective:,.2f} "
                  f"weight={lp.route_weight:.4f} art={lp.artificial_total:.2f} "
                  f"min_rc={min_rc:,.3f}", flush=True)
        if best is None or min_rc >= -args.rc_eps:
            certified = best is not None
            stop_reason = "certified" if certified else "no_path"
            break
        if args.stall_window_min and lp.artificial_total < 1e-6:
            now = elapsed_offset + time.time() - t0
            quarter = args.stall_window_min * 60.0 / 4.0
            recent = [h for h in stall_hist if h[0] >= now - quarter]
            old = [h for h in stall_hist
                   if now - 4 * quarter <= h[0] <= now - 3 * quarter]
            if recent and old:
                rc_rec = min(h[2] for h in recent)   # most negative
                rc_old = min(h[2] for h in old)
                obj_rec = min(h[1] for h in recent)
                obj_old = min(h[1] for h in old)
                rc_impr = (abs(rc_old) - abs(rc_rec)) / max(1e-9, abs(rc_old))
                obj_impr = (obj_old - obj_rec) / max(1.0, abs(obj_old))
                if rc_impr < args.stall_rc_frac and obj_impr < args.stall_obj_frac:
                    print(f"[EXACT] marginal returns stalled over "
                          f"{args.stall_window_min:g} min: |min_rc| "
                          f"{abs(rc_old):,.2f}->{abs(rc_rec):,.2f} "
                          f"({100 * rc_impr:.1f}%), obj "
                          f"{obj_old:,.2f}->{obj_rec:,.2f} "
                          f"({100 * obj_impr:.4f}%) — stopping with an "
                          "INCUMBENT (not a certificate)", flush=True)
                    stop_reason = "stalled_marginal_returns"
                    break
            stall_hist.append((now, lp.objective, min_rc))
        elif args.stall_window_min:
            stall_hist.append((elapsed_offset + time.time() - t0,
                               lp.objective, min_rc))
        added = 0
        for route in batch:
            cost = route["rc"] + sum(lp.trip_duals.get(t, 0.0) for t in route["trips"])
            key = frozenset(route["trips"])
            if key not in pool or cost < pool[key]["cost"] - 1e-9:
                record = {
                    "trips": route["trips"],           # ordered
                    "cost": cost,
                    "route_nodes": route["route_nodes"],
                    "charging_stops": route["charging_stops"],
                    "charges_started": route["charges_started"],
                    "found_iter": iteration,
                }
                pool[key] = record
                if journal:
                    journal.write(json.dumps(record) + "\n")
                added += 1
        if journal and added:
            journal.flush()
            if route["charges_started"] > MAX_DAILY_RECHARGES:
                print(f"[EXACT] note: column uses {route['charges_started']} charge "
                      f"starts (> cap {MAX_DAILY_RECHARGES}).", flush=True)
        if added == 0:
            # Every returned incidence already in the pool at equal cost: the
            # duals are frozen at a degenerate vertex. Interior-point duals
            # (analytic center of the optimal face) usually break the cycle;
            # only give up if the stall repeats under both dual sources.
            stall_count += 1
            if stall_count == 1:
                print("[EXACT] degenerate stall — switching to interior-point "
                      "duals and continuing", flush=True)
                method_order = ("highs-ipm", "highs-ds", "highs")
                continue
            print("[EXACT] stall persists under alternate duals — stopping "
                  "uncertified.", flush=True)
            stop_reason = "degenerate_stall"
            break
        stall_count = 0

    if args.diversify_rounds and pool:
        import random as _random
        rng = _random.Random(20260807)
        base_lp = None
        try:
            routes_now = list(pool.values())
            base_lp = solve_restricted_master_lp(
                trip_ids=trips,
                route_incidence=build_route_incidence(
                    trip_ids=trips,
                    route_trip_ids=[r["trips"] for r in routes_now]),
                route_costs=[r["cost"] for r in routes_now],
                artificial_penalty=BIG_M_PENALTY,
            )
        except Exception as exc:
            print(f"[EXACT] diversify: base LP failed ({exc}); skipping", flush=True)
        if base_lp is not None:
            added_div = 0
            for rnd in range(1, args.diversify_rounds + 1):
                alpha = {t_: v * (1.0 + rng.uniform(-args.diversify_delta,
                                                    args.diversify_delta))
                         for t_, v in base_lp.trip_duals.items()}
                for route in net.k_best_routes(alpha, k=args.columns_per_iter):
                    cost = route["rc"] + sum(alpha.get(t_, 0.0)
                                             for t_ in route["trips"])
                    key = frozenset(route["trips"])
                    if key not in pool or cost < pool[key]["cost"] - 1e-9:
                        record = {
                            "trips": route["trips"], "cost": cost,
                            "route_nodes": route["route_nodes"],
                            "charging_stops": route["charging_stops"],
                            "charges_started": route["charges_started"],
                            "found_iter": -rnd,
                            "origin": "diversify",
                        }
                        pool[key] = record
                        if journal:
                            journal.write(json.dumps(record) + "\n")
                        added_div += 1
            if journal:
                journal.flush()
            print(f"[EXACT] diversify: {args.diversify_rounds} rounds added "
                  f"{added_div} complementary columns", flush=True)

    if journal:
        journal.close()

    # Final LP over the persisted pool: store route values + duals so the
    # fractional solution is reconstructable without re-solving.
    final_lp_detail = None
    routes = list(pool.values())
    if routes:
        try:
            lp_final = solve_restricted_master_lp(
                trip_ids=trips,
                route_incidence=build_route_incidence(
                    trip_ids=trips,
                    route_trip_ids=[r["trips"] for r in routes]),
                route_costs=[r["cost"] for r in routes],
                artificial_penalty=BIG_M_PENALTY,
                coverage_sense=args.master_sense,
            )
            final_lp_detail = {
                "objective": lp_final.objective,
                "route_weight": lp_final.route_weight,
                "artificial_total": lp_final.artificial_total,
                "positive_routes": [
                    {"trips": routes[i]["trips"], "value": v,
                     "cost": routes[i]["cost"]}
                    for i, v in enumerate(lp_final.route_values) if v > 1e-9],
                "trip_duals": {str(k): v for k, v in lp_final.trip_duals.items()},
            }
        except Exception as exc:
            print(f"[EXACT] final LP re-solve failed: {exc}", flush=True)

    result = {
        "csv": args.csv,
        "prices_csv": args.prices_csv,
        "soc_step": args.soc_step,
        "block_min": args.block_min,
        "g_kwh": args.g_kwh,
        "charge_kw": args.charge_kw,
        "min_soc_frac": args.min_soc_frac,
        "master_sense": args.master_sense,
        "trip_ids": trips,
        "iterations": len(history),
        "certified_rc_optimal": certified,
        "final": history[-1] if history else None,
        "columns": len(pool),
        "columns_journal": str(journal_path) if journal_path else None,
        "wall_s": time.time() - t0,
        "stop_reason": stop_reason,
        "history_tail": history[-5:],
        "final_lp": final_lp_detail,
        "provenance": provenance,
    }
    print(f"[EXACT] DONE: {json.dumps(result['final'], default=float)} "
          f"certified={certified} columns={len(pool)} "
          f"wall={result['wall_s']:.0f}s", flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--prices_csv", default="hourly_prices_flat.csv")
    parser.add_argument("--soc-step", type=float, default=15.0)
    parser.add_argument("--block-min", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--columns_per_iter", type=int, default=30)
    parser.add_argument("--rc-eps", type=float, default=1e-4)
    parser.add_argument(
        "--master-sense",
        choices=("partition", "cover"),
        default="partition",
        help="Trip-row sense in the exact-CG restricted master. Partition is "
             "the operational default; cover reproduces legacy campaigns.",
    )
    parser.add_argument("--stall-window-min", type=float, default=None,
                        help="Enable marginal-returns stopping: compare the "
                             "best min_rc and LP objective of the most recent "
                             "quarter-window against the quarter-window one "
                             "full window ago; stop when both improved less "
                             "than their thresholds and no artificials "
                             "remain. Off by default.")
    parser.add_argument("--stall-rc-frac", type=float, default=0.05,
                        help="Relative |min_rc| improvement below which the "
                             "pricing signal counts as stalled.")
    parser.add_argument("--stall-obj-frac", type=float, default=1e-5,
                        help="Relative LP-objective improvement below which "
                             "the master counts as stalled.")
    parser.add_argument("--wall-limit-s", type=int, default=None,
                        help="Stop gracefully after this many seconds "
                             "(set below the Slurm limit so results get written).")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Write the partial --out JSON every N iterations.")
    parser.add_argument("--g-kwh", type=float, default=300.0,
                        help="Battery capacity. GIRO telemetry implies ~239 kWh "
                             "usable; 300 is the historical model convention.")
    parser.add_argument("--charge-kw", type=float, default=CHARGE_RATE_KW,
                        help="Charger power. GIRO telemetry implies ~220 kW; "
                             "300 is the historical model convention.")
    parser.add_argument("--min-soc-frac", type=float, default=0.0,
                        help="SOC reserve as a fraction of capacity (FDL notes "
                             "require 0.2 for duties over 20h).")
    parser.add_argument("--diversify-rounds", type=int, default=0,
                        help="After the main loop, run N extra pricing rounds "
                             "against randomly perturbed duals to harvest "
                             "complementary columns (integrality repair; "
                             "columns are journaled, certification claims "
                             "are unaffected).")
    parser.add_argument("--diversify-delta", type=float, default=0.15,
                        help="Relative dual perturbation for diversify rounds.")
    parser.add_argument("--snapshot-at-minutes", default=None,
                        help="Comma-separated elapsed-minute marks at which to "
                             "freeze immutable pool snapshots (status+journal), "
                             "e.g. 15,60,180,360 for MIP-budget calibration.")
    parser.add_argument("--resume", action="store_true",
                        help="Reload the column journal next to --out and "
                             "continue from that pool.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_cg(args)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict integer master over an exact-pricer column journal.

Differences from the historical final MIP (master.py / run_final_mip.py):
  * binary route variables from the start;
  * NO artificial variables — if some trip has no covering column the script
    says so and refuses, instead of silently paying BIG-M;
  * trip coverage is exact partitioning (== 1) by default (--cover relaxes
    to >= 1, reporting overcovered trips);
  * costs are the exact-pricer's stored route costs (bus 100k + charging),
    so "minimize buses first, charging second" holds lexicographically
    because charging never reaches 1% of one bus.

Usage (Gurobi required, e.g. on Unicorn):

    python run_exact_pool_mip.py --result results/exact_big/<...>.json \
        --timelimit 3600 --out results/exact_big/<...>_mip.json

`--validate-only` loads the pool and reports row coverage plus whether a known
singleton partition is present, without importing Gurobi. Row coverage alone
does not prove that a binary exact partition exists.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path


def resolve_pool_journal(result_path: Path, status: dict) -> Path:
    """Resolve the journal exactly as :func:`load_pool` will read it."""

    # A live status file normally records the journal path directly.  Frozen
    # snapshots, however, are often copied to a timestamped directory (or a
    # release archive) together with the journal.  In that case the recorded
    # absolute Unicorn path may no longer exist, so also look beside the
    # snapshot using the recorded basename.  The final candidate preserves the
    # exact-pricer's historical ``RESULT.json.columns.jsonl`` convention.
    recorded_journal = status.get("columns_journal")
    candidates = []
    is_snapshot = result_path.name.endswith(".snapshot.json")
    recorded_path = Path(recorded_journal) if recorded_journal else None
    if is_snapshot and recorded_path is not None:
        # Prefer the frozen sibling over a recorded live/cluster path that may
        # still exist but no longer represent this snapshot.
        candidates.append(result_path.parent / recorded_path.name)
    if is_snapshot:
        snapshot_stem = result_path.name[: -len(".snapshot.json")]
        candidates.append(result_path.with_name(f"{snapshot_stem}.columns.jsonl"))
    if recorded_journal:
        candidates.append(recorded_path)
        if not is_snapshot:
            candidates.append(result_path.parent / recorded_path.name)
    candidates.append(Path(str(result_path) + ".columns.jsonl"))

    unique_candidates = list(dict.fromkeys(candidates))
    journal_path = next((path for path in unique_candidates if path.exists()), None)
    if journal_path is None:
        tried = "\n  ".join(str(path) for path in unique_candidates)
        raise SystemExit(
            f"{result_path} has no readable column journal. Tried:\n  {tried}\n"
            "The run may predate pool persistence, or its journal was not "
            "copied with the status file.")
    return journal_path


def load_pool(result_path: Path):
    with open(result_path) as fh:
        status = json.load(fh)

    journal_path = resolve_pool_journal(result_path, status)
    pool = {}
    with open(journal_path) as fh:
        for line in fh:
            rec = json.loads(line)
            key = frozenset(rec["trips"])
            if key not in pool or rec["cost"] < pool[key]["cost"] - 1e-9:
                pool[key] = rec
    trips = status.get("trip_ids")
    if trips is None:
        raise SystemExit(f"{result_path} lacks trip_ids; rerun with current code.")
    return status, list(pool.values()), [int(t) for t in trips]


def singleton_partition_indices(routes: list[dict], trips: list[int]) -> list[int]:
    """Return one singleton-column index per trip, or an empty list."""

    by_trip = {}
    trip_set = set(trips)
    for index, route in enumerate(routes):
        route_trips = route.get("trips", [])
        if len(route_trips) == 1 and route_trips[0] in trip_set:
            by_trip[route_trips[0]] = index
    if len(by_trip) != len(trips):
        return []
    return [by_trip[trip] for trip in trips]


def greedy_partition_start_indices(
    routes: list[dict],
    trips: list[int],
    singleton_partition: list[int],
) -> list[int]:
    """Build a deterministic feasible start from disjoint priced routes.

    Start with the guaranteed singleton partition, greedily replace groups of
    singletons by saved multi-trip routes with the largest cost savings, and
    fill every remaining trip with its singleton.  This is only an incumbent
    heuristic; Gurobi remains responsible for optimizing the full pool.
    """

    if not singleton_partition:
        return []
    singleton_by_trip = {
        routes[index]["trips"][0]: index for index in singleton_partition
    }
    candidates = []
    for index, route in enumerate(routes):
        route_trips = route.get("trips", [])
        if len(route_trips) <= 1:
            continue
        singleton_cost = sum(
            routes[singleton_by_trip[trip]]["cost"] for trip in route_trips
        )
        savings = singleton_cost - route["cost"]
        if savings > 1e-9:
            candidates.append((-savings, -len(route_trips), route["cost"], index))
    candidates.sort()

    selected = []
    covered = set()
    for _negative_savings, _negative_length, _cost, index in candidates:
        route_trips = set(routes[index]["trips"])
        if covered.isdisjoint(route_trips):
            selected.append(index)
            covered.update(route_trips)
    selected.extend(singleton_by_trip[trip] for trip in trips if trip not in covered)
    return selected


def validate_injected_route(problem, record, g_kwh, charge_kw, reserve_kwh,
                            horizon_min, rate_grace_min=1.0):
    """Replay an injected route against the model graph and pool physics.

    Checks every consecutive arc exists in the restricted adjacency, times
    chain (fixed trip times; charging inside its [cst, cet] window at <= the
    pool's charger power, with a small grace for Hastus minute rounding),
    and continuous SOC never drops below the reserve. Returns None if valid,
    else a short reason.
    """
    arc = {}
    for u, arcs in problem.adjacency.items():
        for v, travel, dh, _kind in arcs:
            arc[(u, v)] = (travel, dh)

    nodes = record.get("route_nodes") or []
    if len(nodes) < 3:
        return "route_nodes missing"
    stops = record.get("charging_stops") or {}
    st_list = list(zip(stops.get("stations", []), stops.get("cst", []),
                       stops.get("cet", []), stops.get("kwh", [])))
    si = 0
    soc = float(g_kwh)
    time_now = None  # depot departure is flexible
    prev = nodes[0]
    for v in nodes[1:]:
        key = (prev, v) if not (isinstance(prev, str) and isinstance(v, str)
                                and prev == v) else None
        if key is not None and key not in arc:
            return f"missing model arc {prev}->{v}"
        travel, dh = arc.get(key, (0.0, 0.0))
        soc -= dh
        if soc < reserve_kwh - 1e-6:
            return f"SOC {soc:.1f} < reserve before {v}"
        if isinstance(v, int):
            arrive_earliest = (time_now + travel) if time_now is not None else None
            if arrive_earliest is not None and                     arrive_earliest > problem.start_min[v] + 1e-6:
                return f"arrives {arrive_earliest:.0f} after trip {v} start"
            soc -= problem.trip_energy[v]
            if soc < reserve_kwh - 1e-6:
                return f"SOC {soc:.1f} < reserve after trip {v}"
            time_now = problem.end_min[v]
        elif v != nodes[-1] or si < len(st_list):
            if si < len(st_list) and st_list[si][0] == v:
                _stn, cst, cet, kwh = st_list[si]
                si += 1
                if time_now is not None and time_now + travel > cst + rate_grace_min + 1e-6:
                    return f"reaches {v} at {time_now + travel:.0f} after cst {cst}"
                window = max(0.0, float(cet) - float(cst)) + rate_grace_min
                if kwh > window * charge_kw / 60.0 + 1e-6:
                    return (f"charge {kwh:.1f} kWh exceeds {charge_kw:.0f} kW "
                            f"in {window:.0f} min at {v}")
                soc = min(float(g_kwh), soc + float(kwh))
                time_now = float(cet)
            # station visit without a recorded stop = pure wait; allowed
        prev = v
    if time_now is not None and time_now > horizon_min + 1e-6:
        return f"ends at {time_now:.0f} past horizon"
    return None


def merge_extra_routes(routes, trips, extra_paths, prices_csv, status=None):
    """Merge runner-format route dicts (e.g. MATCHING covers) into the pool.

    Pool MIPs at k>=8 stall near singleton incumbents because CG pools lack an
    integral backbone; constructive covers supply one. Costs are recomputed
    with the exact master cost function, and EVERY candidate is replayed
    against the model graph under the POOL'S physics (g_kwh, charge_kw,
    reserve) — routes that fail are refused, not warned about.
    """

    from audit_giro_known_columns import HORIZON_MIN, build_problem
    from config import (BUS_COST_KX, CHARGE_RATE_KW, CHARGE_START_COST,
                        CHARGING_STATIONS)
    from utils_v2 import calculate_truck_route_cost_accurate, load_station_hourly_prices

    status = status or {}
    g_kwh = float(status.get("g_kwh", 300.0))
    charge_kw = float(status.get("charge_kw", CHARGE_RATE_KW))
    reserve_kwh = float(status.get("min_soc_frac", 0.0)) * g_kwh
    problem = build_problem(
        Path(__file__).resolve().parent.parent / "data", status["csv"],
        max_station_to_trip_wait_min=HORIZON_MIN)

    data_dir = Path(__file__).resolve().parent.parent / "data"
    price_name = Path(prices_csv).name if prices_csv else "hourly_prices_flat.csv"
    prices = load_station_hourly_prices(data_dir / price_name, CHARGING_STATIONS)
    depot_curve = prices.get("PARX") or next(iter(prices.values()))

    pool = {frozenset(r["trips"]): r for r in routes}
    trip_set = set(trips)
    merged = 0
    rejected = 0
    for path in extra_paths:
        with open(path) as fh:
            payload = json.load(fh)
        for route in payload.get("routes", []):
            route_trips = [n for n in route.get("route", []) if isinstance(n, int)]
            if not route_trips or not set(route_trips) <= trip_set:
                continue
            cost = calculate_truck_route_cost_accurate(
                route, BUS_COST_KX, depot_curve,
                charge_rate_kw=CHARGE_RATE_KW,
                station_hourly_prices=prices,
                charge_start_cost=CHARGE_START_COST,
            )
            key = frozenset(route_trips)
            candidate = {
                "trips": route_trips,
                "route_nodes": route.get("route", []),
                "charging_stops": route.get("charging_stops", {}),
            }
            reason = validate_injected_route(
                problem, candidate, g_kwh, charge_kw, reserve_kwh, HORIZON_MIN)
            if reason is not None:
                rejected += 1
                if rejected <= 5:
                    print(f"[MIP] REJECTED injected route ({len(route_trips)} "
                          f"trips): {reason}")
                continue
            if key not in pool or cost < pool[key]["cost"] - 1e-9:
                pool[key] = {
                    "trips": route_trips,
                    "cost": cost,
                    "route_nodes": route.get("route", []),
                    "charging_stops": route.get("charging_stops", {}),
                    "charges_started": len(
                        (route.get("charging_stops") or {}).get("stations", [])),
                    "found_iter": 0,
                    "origin": f"extra:{path.name[:40]}",
                }
                merged += 1
    print(f"[MIP] merged {merged} extra route(s), REJECTED {rejected} as "
          f"infeasible under pool physics (G={g_kwh:.0f} kWh, "
          f"{charge_kw:.0f} kW, reserve {reserve_kwh:.0f}) from "
          f"{len(extra_paths)} file(s)")
    return list(pool.values())


def finite_solver_value(value):
    """Map Gurobi infinity/sentinel values to JSON null."""

    number = float(value)
    if not math.isfinite(number) or abs(number) >= 1e100:
        return None
    return number


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True,
                        help="Exact-pricer status JSON (with columns_journal).")
    parser.add_argument("--timelimit", type=int, default=3600)
    parser.add_argument("--mipgap", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cover", action="store_true",
                        help="Relax partitioning (==1) to covering (>=1).")
    parser.add_argument(
        "--require-singleton-partition",
        action="store_true",
        help="Refuse a pool that lacks one singleton column per trip.",
    )
    parser.add_argument(
        "--extra-routes",
        type=Path,
        action="append",
        default=None,
        help="Runner-format routes JSON (e.g. a --matching run's "
             "routes_colgen_final_*.json) whose real columns are merged into "
             "the pool before solving. Costs are recomputed with the exact "
             "master cost function. Repeatable.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Lexicographic solve: stage 1 minimizes the bus count alone; "
             "stage 2 fixes that count as a budget and minimizes total cost. "
             "Splits --timelimit roughly 40/60 between the stages.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.out is not None and args.out.resolve() == args.result.resolve():
        parser.error("--out must not overwrite --result")

    status, routes, trips = load_pool(args.result)
    if args.extra_routes:
        routes = merge_extra_routes(routes, trips, args.extra_routes,
                                    status.get("prices_csv"), status)
    coverage = Counter(t for r in routes for t in r["trips"])
    uncovered = [t for t in trips if coverage[t] == 0]
    seed_partition = singleton_partition_indices(routes, trips)
    mip_start = greedy_partition_start_indices(routes, trips, seed_partition)
    print(f"[MIP] pool: {len(routes)} columns over {len(trips)} trips "
          f"(instance {status['csv']}, soc_step={status['soc_step']}, "
          f"certified={status.get('certified_rc_optimal')})")
    print(f"[MIP] coverage: min {min(coverage[t] for t in trips) if not uncovered else 0}"
          f" / median {sorted(coverage[t] for t in trips)[len(trips)//2]}"
          f" / max {max(coverage[t] for t in trips)} columns per trip")
    if uncovered:
        raise SystemExit(f"[MIP] {len(uncovered)} trips have NO covering column "
                         f"({uncovered[:15]}...): pool is MIP-infeasible without "
                         "artificials — extend the CG run first.")
    if seed_partition:
        print(f"[MIP] strict-feasibility seed: {len(seed_partition)} exact "
              "singleton columns (one per trip)")
        print(f"[MIP] greedy feasible MIP start: {len(mip_start)} buses")
    else:
        print("[MIP] WARNING: row coverage is complete, but the pool has no "
              "known integer partition seed")
        if args.require_singleton_partition:
            raise SystemExit(
                "[MIP] singleton partition required; prepare this legacy pool "
                "with prepare_exact_pool_mip.py before submission"
            )
    if args.validate_only:
        if seed_partition:
            print("[MIP] validate-only: strict partition feasibility is "
                  "guaranteed by the singleton seed. OK.")
        else:
            print("[MIP] validate-only: row coverage only; strict partition "
                  "feasibility has NOT been established.")
        return 0

    out = args.out or Path(str(args.result).replace(".json", "_mip.json"))
    out.parent.mkdir(parents=True, exist_ok=True)

    import gurobipy as gp
    from gurobipy import GRB

    t0 = time.time()
    m = gp.Model("exact_pool_mip")
    m.Params.TimeLimit = args.timelimit
    m.Params.MIPGap = args.mipgap
    m.Params.Threads = args.threads
    a = m.addVars(len(routes), vtype=GRB.BINARY, name="a")
    if mip_start:
        start_set = set(mip_start)
        for index in range(len(routes)):
            a[index].Start = 1.0 if index in start_set else 0.0
    sense = ">" if args.cover else "="
    trip_rows = {t: [] for t in trips}
    for i, r in enumerate(routes):
        for t in r["trips"]:
            trip_rows[t].append(i)
    for t in trips:
        expr = gp.quicksum(a[i] for i in trip_rows[t])
        if args.cover:
            m.addConstr(expr >= 1, name=f"cov_{t}")
        else:
            m.addConstr(expr == 1, name=f"part_{t}")
    two_stage_detail = None
    if args.two_stage:
        # Stage 1: pure fleet minimization. Charging never reaches 1% of a bus,
        # but an explicit count objective lets Gurobi prove the fleet bound
        # without dragging cost fractions through the branch-and-bound tree.
        m.Params.TimeLimit = max(60, int(args.timelimit * 0.4))
        m.setObjective(gp.quicksum(a[i] for i in range(len(routes))), GRB.MINIMIZE)
        m.optimize()
        if m.SolCount == 0:
            raise SystemExit("[MIP] two-stage: stage 1 found no feasible fleet "
                             "solution within its time slice")
        stage1_buses = int(round(m.ObjVal))
        stage1_bound = finite_solver_value(m.ObjBound)
        stage1_solution = [i for i in range(len(routes)) if a[i].X > 0.5]
        print(f"[MIP] stage 1: fleet={stage1_buses} "
              f"(bound {stage1_bound}, gap {finite_solver_value(m.MIPGap)})")
        # Stage 2: cost minimization under the proven/incumbent fleet budget.
        m.addConstr(gp.quicksum(a[i] for i in range(len(routes)))
                    <= stage1_buses, name="fleet_budget")
        for index in range(len(routes)):
            a[index].Start = 1.0 if index in set(stage1_solution) else 0.0
        m.Params.TimeLimit = max(60, int(args.timelimit * 0.6))
        two_stage_detail = {
            "stage1_buses": stage1_buses,
            "stage1_bound": stage1_bound,
            "stage1_runtime_s": time.time() - t0,
        }
    m.setObjective(gp.quicksum(routes[i]["cost"] * a[i]
                               for i in range(len(routes))), GRB.MINIMIZE)
    m.optimize()

    status_code = int(m.Status)
    # Gurobi's documented optimization status codes are stable integers.  Use
    # the numbers here so reporting also works across installations where a
    # newer symbolic constant is absent from an older gurobipy build.
    status_names = {
        1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
        5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT",
        8: "NODE_LIMIT", 9: "TIME_LIMIT", 10: "SOLUTION_LIMIT",
        11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL",
        14: "INPROGRESS", 15: "USER_OBJ_LIMIT",
    }
    status_name = status_names.get(status_code, f"UNKNOWN_{status_code}")
    chosen = [i for i in range(len(routes)) if a[i].X > 0.5] \
        if m.SolCount > 0 else []
    over = {t: c for t, c in Counter(
        t for i in chosen for t in routes[i]["trips"]).items() if c > 1}
    mip_obj = finite_solver_value(m.ObjVal) if m.SolCount > 0 else None
    mip_bound = finite_solver_value(m.ObjBound)
    mip_gap = finite_solver_value(m.MIPGap) if m.SolCount > 0 else None
    summary = {
        "source_result": str(args.result),
        "instance": status["csv"],
        "partitioning": not args.cover,
        "status": status_code,
        "status_name": status_name,
        "mip_obj": mip_obj,
        "mip_bound": mip_bound,
        "mip_gap": mip_gap,
        "buses": len(chosen),
        "charging_cost": (mip_obj - 100000.0 * len(chosen))
                         if chosen and mip_obj is not None else None,
        "overcovered_trips": len(over),
        "runtime_s": time.time() - t0,
        "pool_columns": len(routes),
        "singleton_partition_columns": len(seed_partition),
        "mip_start_used": bool(mip_start),
        "mip_start_buses": len(mip_start) if mip_start else None,
        "pool_preparation": status.get("pool_preparation"),
        "two_stage": two_stage_detail,
        "pricer_provenance": status.get("provenance"),
        "selected_routes": [routes[i] for i in chosen],
    }
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[MIP] status={status_name}({status_code}) buses={len(chosen)} "
          f"obj={summary['mip_obj']} gap={summary['mip_gap']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

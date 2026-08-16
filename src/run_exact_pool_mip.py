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
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

from durable_io import atomic_write_json, read_jsonl_records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parent,
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_result(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )


def verified_mip_code_identity() -> dict:
    """Bind a submitted solve to one clean reviewed Git commit."""

    expected = os.environ.get("EVSP_EXPECTED_COMMIT")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    require_detached = os.environ.get("EVSP_REQUIRE_DETACHED") == "1"
    enforce = bool(expected or slurm_job_id or require_detached)
    head_result = git_result("rev-parse", "--verify", "HEAD")
    observed = head_result.stdout.strip()
    tracked_result = git_result(
        "status", "--porcelain", "--untracked-files=no"
    )
    symbolic_result = git_result("symbolic-ref", "-q", "HEAD")
    if (head_result.returncode != 0 or len(observed) != 40
            or any(character not in "0123456789abcdef"
                   for character in observed)):
        raise SystemExit("[MIP] solver has no verifiable Git HEAD")
    if slurm_job_id and not expected:
        raise SystemExit("[MIP] submitted solve lacks EVSP_EXPECTED_COMMIT")
    if expected and observed != expected:
        raise SystemExit(
            f"[MIP] solver commit mismatch: expected {expected}, "
            f"found {observed}"
        )
    if tracked_result.returncode != 0:
        raise SystemExit("[MIP] could not verify solver worktree state")
    tracked_status = tracked_result.stdout.strip()
    if enforce and tracked_status:
        raise SystemExit("[MIP] solver checkout has tracked modifications")
    if symbolic_result.returncode == 0:
        branch = symbolic_result.stdout.strip()
        detached = False
    elif symbolic_result.returncode == 1:
        branch = ""
        detached = True
    else:
        raise SystemExit("[MIP] could not verify detached solver HEAD")
    if enforce and (require_detached or slurm_job_id) and not detached:
        raise SystemExit(
            f"[MIP] solver must run detached; found branch {branch}"
        )
    return {
        "expected_commit": expected,
        "observed_commit": observed,
        "branch": branch,
        "detached": detached,
        "tracked_clean": not bool(tracked_status),
        "enforced": enforce,
    }


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

    trips = status.get("trip_ids")
    if trips is None:
        raise SystemExit(f"{result_path} lacks trip_ids; rerun with current code.")
    trips = [int(trip) for trip in trips]
    if len(trips) != len(set(trips)):
        raise SystemExit(f"{result_path} repeats a trip in trip_ids")
    allowed = set(trips)
    journal_path = resolve_pool_journal(result_path, status)
    pool = {}
    records = read_jsonl_records(journal_path, repair_trailing=False)
    for ordinal, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            raise SystemExit(
                f"{journal_path} record {ordinal} is not a JSON object"
            )
        route_trips = rec.get("trips")
        if not isinstance(route_trips, list) or not route_trips:
            raise SystemExit(
                f"{journal_path} record {ordinal} has no nonempty trips list"
            )
        if any(
                not isinstance(trip, int) or isinstance(trip, bool)
                for trip in route_trips):
            raise SystemExit(
                f"{journal_path} record {ordinal} has non-integer trips"
            )
        try:
            unique_trips = set(route_trips)
        except TypeError as exc:
            raise SystemExit(
                f"{journal_path} record {ordinal} has unhashable trips"
            ) from exc
        if len(route_trips) != len(unique_trips):
            raise SystemExit(
                f"{journal_path} record {ordinal} repeats a trip"
            )
        unknown = [trip for trip in route_trips if trip not in allowed]
        if unknown:
            raise SystemExit(
                f"{journal_path} record {ordinal} contains trips outside "
                f"the snapshot: {unknown[:15]}"
            )
        try:
            cost = float(rec["cost"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"{journal_path} record {ordinal} has invalid cost"
            ) from exc
        if not math.isfinite(cost):
            raise SystemExit(
                f"{journal_path} record {ordinal} has non-finite cost"
            )
        key = frozenset(route_trips)
        if key not in pool or cost < float(pool[key]["cost"]) - 1e-9:
            pool[key] = rec
    return status, list(pool.values()), trips


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
                            horizon_min, arrival_grace_min=1.0,
                            rate_grace_min=0.0):
    """Replay an injected route against the model graph and pool physics.

    Checks every consecutive arc exists in the restricted adjacency, times
    chain (fixed trip times; charging inside its [cst, cet] window at <= the
    pool's charger power.  A small arrival-time grace accommodates Hastus
    minute rounding, but it does not create extra charging energy),
    and continuous SOC never drops below the reserve. Returns None if valid,
    else a short reason.
    """
    from audit_giro_known_columns import DEPOT

    arc = {}
    for u, arcs in problem.adjacency.items():
        for v, travel, dh, _kind in arcs:
            arc[(u, v)] = (travel, dh)

    nodes = record.get("route_nodes") or []
    if len(nodes) < 3:
        return "route_nodes missing"
    if nodes[0] != DEPOT or nodes[-1] != DEPOT:
        return "route must start and end at the depot"
    stops = record.get("charging_stops") or {}
    stop_fields = [list(stops.get(name, []))
                   for name in ("stations", "cst", "cet", "kwh")]
    if len({len(values) for values in stop_fields}) != 1:
        return "charging stop fields have different lengths"
    st_list = list(zip(*stop_fields))
    normalized_stops = []
    for station, cst, cet, kwh in st_list:
        try:
            cst_value = float(cst)
            cet_value = float(cet)
            kwh_value = float(kwh)
        except (TypeError, ValueError):
            return f"non-numeric charging stop at {station}"
        if not all(math.isfinite(value)
                   for value in (cst_value, cet_value, kwh_value)):
            return f"non-finite charging stop at {station}"
        if cet_value < cst_value - 1e-6:
            return f"charging stop at {station} ends before it starts"
        if cst_value < -1e-6:
            return f"charging stop at {station} starts before the horizon"
        if cet_value > float(horizon_min) + 1e-6:
            return f"charging stop at {station} ends past the horizon"
        if kwh_value < -1e-6:
            return f"negative charge {kwh_value:.1f} kWh at {station}"
        normalized_stops.append(
            (station, cst_value, cet_value, kwh_value)
        )
    st_list = normalized_stops
    si = 0
    soc = float(g_kwh)
    time_now = None  # depot departure is flexible
    prev = nodes[0]
    for position, v in enumerate(nodes[1:], start=1):
        is_last = position == len(nodes) - 1
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
        elif is_last:
            if time_now is not None:
                time_now += travel
        else:
            if si < len(st_list) and st_list[si][0] == v:
                _stn, cst, cet, kwh = st_list[si]
                si += 1
                if (time_now is not None
                        and time_now + travel
                        > float(cst) + arrival_grace_min + 1e-6):
                    return f"reaches {v} at {time_now + travel:.0f} after cst {cst}"
                window = max(0.0, float(cet) - float(cst)) + rate_grace_min
                if kwh > window * charge_kw / 60.0 + 1e-6:
                    return (f"charge {kwh:.1f} kWh exceeds {charge_kw:.0f} kW "
                            f"in {window:.0f} min at {v}")
                if soc + float(kwh) > float(g_kwh) + 1e-6:
                    return (f"charge at {v} raises SOC to "
                            f"{soc + float(kwh):.1f} > capacity {g_kwh:.1f}")
                soc += float(kwh)
                time_now = float(cet)
            elif time_now is not None:
                # A station visit without a recorded stop is a pure-wait
                # waypoint.  At minimum its inbound travel must advance time.
                time_now += travel
        prev = v
    if si != len(st_list):
        return f"{len(st_list) - si} charging stop record(s) were not consumed"
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
                charge_rate_kw=charge_kw,
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


def merge_validated_partition_start(
    routes,
    trips,
    partition_path,
    prices_csv,
    status=None,
    *,
    data_dir=None,
):
    """Merge and select one explicitly supplied exact-partition start.

    Unlike ``--extra-routes``, this path is fail-closed: every supplied route
    must be a real route under the pool's physics, and the supplied routes
    together must cover every pool trip exactly once.  The selected start uses
    the cheapest pool realization for each supplied trip incidence after
    merging, so an existing cheaper duplicate never weakens the start.
    """

    from audit_giro_known_columns import HORIZON_MIN, build_problem
    from config import (BUS_COST_KX, CHARGE_START_COST,
                        CHARGING_STATIONS)
    from utils_v2 import (
        calculate_truck_route_cost_accurate,
        load_station_hourly_prices,
    )

    path = Path(partition_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"[MIP] initial partition source is missing: {path}")
    raw = path.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"[MIP] initial partition is not valid JSON: {path}"
        ) from exc
    supplied = payload.get("routes") if isinstance(payload, dict) else None
    if not isinstance(supplied, list) or not supplied:
        raise SystemExit(
            f"[MIP] initial partition must contain a nonempty routes list: {path}"
        )

    status = status or {}
    required = (
        "csv", "prices_csv", "soc_step", "block_min",
        "g_kwh", "charge_kw", "min_soc_frac",
    )
    missing = [field for field in required if status.get(field) is None]
    if missing:
        raise SystemExit(
            f"[MIP] pool snapshot lacks required start-validation fields: "
            f"{missing}"
        )
    if prices_csv is not None and str(prices_csv) != str(status["prices_csv"]):
        raise SystemExit(
            "[MIP] requested start-validation prices differ from the pool "
            "snapshot"
        )
    try:
        g_kwh = float(status["g_kwh"])
        charge_kw = float(status["charge_kw"])
        reserve_frac = float(status["min_soc_frac"])
        soc_step = float(status["soc_step"])
        block_min = float(status["block_min"])
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "[MIP] pool snapshot has non-numeric physics"
        ) from exc
    if (not all(math.isfinite(value) for value in (
            g_kwh, charge_kw, reserve_frac, soc_step, block_min))
            or g_kwh <= 0.0 or charge_kw <= 0.0
            or soc_step <= 0.0 or block_min <= 0.0
            or not 0.0 <= reserve_frac <= 1.0):
        raise SystemExit("[MIP] pool snapshot has invalid or non-finite physics")
    reserve_kwh = reserve_frac * g_kwh

    provenance = status.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit(
            "[MIP] pool snapshot lacks hashed input provenance"
        )
    data_dir = Path(
        data_dir
        if data_dir is not None
        else Path(__file__).resolve().parent.parent / "data"
    ).resolve()

    def verified_data_file(relative, hash_field, label):
        expected_hash = provenance.get(hash_field)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise SystemExit(
                f"[MIP] pool snapshot lacks {hash_field} provenance"
            )
        candidate = (data_dir / str(relative)).resolve()
        try:
            candidate.relative_to(data_dir)
        except ValueError as exc:
            raise SystemExit(
                f"[MIP] pool {label} escapes the data directory: {relative}"
            ) from exc
        if not candidate.is_file():
            raise SystemExit(
                f"[MIP] pool {label} is missing: {candidate}"
            )
        actual_hash = file_sha256(candidate)
        if actual_hash != expected_hash:
            raise SystemExit(
                f"[MIP] pool {label} hash mismatch: expected "
                f"{expected_hash}, found {actual_hash}"
            )
        return candidate

    instance_path = verified_data_file(
        status["csv"], "instance_sha256", "instance"
    )
    prices_path = verified_data_file(
        status["prices_csv"], "prices_sha256", "prices"
    )
    problem = build_problem(
        data_dir,
        str(instance_path.relative_to(data_dir)),
        max_station_to_trip_wait_min=HORIZON_MIN,
    )
    trip_set = set(trips)
    if list(problem.trips) != list(trips):
        raise SystemExit(
            "[MIP] initial partition validation reconstructed a different "
            "ordered trip set from the pool snapshot"
        )
    for trip in problem.trips:
        values = (
            float(problem.start_min[trip]),
            float(problem.end_min[trip]),
            float(problem.trip_energy[trip]),
        )
        if (not all(math.isfinite(value) for value in values)
                or values[2] < 0.0 or values[1] < values[0]):
            raise SystemExit(
                f"[MIP] reconstructed instance has invalid trip data for {trip}"
            )
    prices = load_station_hourly_prices(
        prices_path, CHARGING_STATIONS
    )
    depot_curve = prices.get("PARX") or next(iter(prices.values()))
    arc_deadhead = {}
    for node, arcs in problem.adjacency.items():
        for successor, travel, deadhead_kwh, _kind in arcs:
            travel = float(travel)
            deadhead_kwh = float(deadhead_kwh)
            if (not math.isfinite(travel)
                    or not math.isfinite(deadhead_kwh)
                    or travel < 0.0 or deadhead_kwh < 0.0):
                raise SystemExit(
                    f"[MIP] reconstructed instance has invalid arc data "
                    f"{node}->{successor}"
                )
            arc_deadhead[(node, successor)] = deadhead_kwh

    validated = []
    counts = Counter()
    for ordinal, route in enumerate(supplied, start=1):
        if not isinstance(route, dict):
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} is not a JSON object"
            )
        nodes = route.get("route")
        if nodes is None:
            nodes = route.get("route_nodes")
        if not isinstance(nodes, list):
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} has no route nodes"
            )
        route_trips = [
            node for node in nodes
            if isinstance(node, int) and not isinstance(node, bool)
        ]
        if not route_trips:
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} contains no trips"
            )
        if len(route_trips) != len(set(route_trips)):
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} repeats a trip"
            )
        unknown = sorted(set(route_trips) - trip_set)
        if unknown:
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} contains trips "
                f"outside the pool instance: {unknown[:15]}"
            )

        candidate = {
            "trips": route_trips,
            "route_nodes": nodes,
            "charging_stops": route.get("charging_stops", {}),
        }
        reason = validate_injected_route(
            problem,
            candidate,
            g_kwh,
            charge_kw,
            reserve_kwh,
            HORIZON_MIN,
        )
        if reason is not None:
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} failed physical "
                f"validation: {reason}"
            )
        recomputed_deadhead_kwh = 0.0
        for left, right in zip(nodes, nodes[1:]):
            if isinstance(left, str) and isinstance(right, str) and left == right:
                continue
            try:
                recomputed_deadhead_kwh += arc_deadhead[(left, right)]
            except KeyError as exc:
                # ``validate_injected_route`` should already have refused it;
                # keep pricing independently fail-closed.
                raise SystemExit(
                    f"[MIP] initial partition route {ordinal} has no "
                    f"deadhead-energy arc {left}->{right}"
                ) from exc
        pricing_route = dict(route)
        pricing_route["route"] = nodes
        pricing_route["route_nodes"] = nodes
        pricing_route["deadhead_kwh"] = recomputed_deadhead_kwh
        cost = calculate_truck_route_cost_accurate(
            pricing_route,
            BUS_COST_KX,
            depot_curve,
            charge_rate_kw=charge_kw,
            station_hourly_prices=prices,
            charge_start_cost=CHARGE_START_COST,
        )
        if not math.isfinite(float(cost)):
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} has non-finite cost"
            )
        counts.update(route_trips)
        validated.append({
            **candidate,
            "cost": float(cost),
            "deadhead_kwh": recomputed_deadhead_kwh,
            "charges_started": len(
                (route.get("charging_stops") or {}).get("stations", [])
            ),
            "found_iter": 0,
            "origin": f"initial_partition:{path.name[:40]}",
        })

    missing = [trip for trip in trips if counts[trip] == 0]
    repeated = {trip: counts[trip] for trip in trips if counts[trip] > 1}
    if missing or repeated:
        raise SystemExit(
            "[MIP] supplied initial routes are not an exact partition: "
            f"missing={missing[:15]}, repeated={list(repeated.items())[:15]}"
        )

    existing_keys = {frozenset(route["trips"]) for route in routes}
    merged = list(routes)
    start_indices = []
    added = preserved_duplicates = 0
    for record in validated:
        key = frozenset(record["trips"])
        if key not in existing_keys:
            added += 1
        else:
            preserved_duplicates += 1
        start_indices.append(len(merged))
        merged.append(record)
        existing_keys.add(key)
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_sha256:
        raise SystemExit(
            f"[MIP] initial partition source changed while loading: {path}"
        )

    detail = {
        "kind": "validated_exact_partition",
        "source": str(path),
        "source_sha256": source_sha256,
        "validated": True,
        "validated_bus_count": len(start_indices),
        "expected_full_objective": float(
            sum(merged[index]["cost"] for index in start_indices)
        ),
        "pool_columns_added": added,
        "pool_columns_replaced": 0,
        "pool_columns_reused": 0,
        "pool_duplicate_incidences_preserved": preserved_duplicates,
        "actual_start_column_hashes": [
            hashlib.sha256(json.dumps(
                {
                    "trips": merged[index]["trips"],
                    "route_nodes": merged[index]["route_nodes"],
                    "charging_stops": merged[index]["charging_stops"],
                    "cost": merged[index]["cost"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()).hexdigest()
            for index in start_indices
        ],
    }
    print(
        f"[MIP] validated exact-partition start: {len(start_indices)} buses "
        f"from {path} (added {added}, preserved duplicate incidences "
        f"{preserved_duplicates})"
    )
    return merged, start_indices, detail


def optimize_with_start_audit(
    model,
    GRB,
    *,
    start_supplied: bool,
    progress_observer=None,
) -> dict:
    """Optimize with one composed start-audit/progress callback."""

    audit = {
        "status": "not_observed" if start_supplied else "not_supplied",
        "accepted": None,
        "rejection_observed": False,
        "messages": [],
    }
    callback_api = getattr(GRB, "Callback", None)
    can_capture = (
        callback_api is not None
        and hasattr(callback_api, "MESSAGE")
        and hasattr(callback_api, "MSG_STRING")
        and callable(getattr(model, "cbGet", None))
    )
    if not can_capture and progress_observer is None:
        model.optimize()
        return audit

    def callback(callback_model, where):
        if progress_observer is not None:
            progress_observer(callback_model, where)
        if (
            not start_supplied
            or not can_capture
            or where != callback_api.MESSAGE
        ):
            return
        message = str(
            callback_model.cbGet(callback_api.MSG_STRING)
        ).strip()
        if message and "MIP start" in message:
            audit["messages"].append(message)
            if (
                "Loaded user MIP start with objective" in message
                or "User MIP start produced solution with objective" in message
            ):
                audit["status"] = "accepted"
                audit["accepted"] = True
            elif "User MIP start violates constraint" in message:
                audit["rejection_observed"] = True
                if audit["accepted"] is not True:
                    audit["status"] = "rejected_infeasible"
                    audit["accepted"] = False
            elif "User MIP start did not produce a new incumbent" in message:
                audit["rejection_observed"] = True
                if audit["accepted"] is not True:
                    audit["status"] = "not_loaded_as_incumbent"
                    audit["accepted"] = False

    model.optimize(callback)
    return audit


def finite_solver_value(value):
    """Map Gurobi infinity/sentinel values to JSON null."""

    number = float(value)
    if not math.isfinite(number) or abs(number) >= 1e100:
        return None
    return number


def fleet_bound_proves_incumbent(
    incumbent_buses: int,
    lower_bound: float | None,
    _status_code: int,
) -> bool:
    """Whether a minimization bound certifies the integer fleet incumbent."""

    return (
        lower_bound is not None
        and math.isfinite(lower_bound)
        and math.ceil(lower_bound - 1e-6) >= int(incumbent_buses)
    )


def optimal_scope(*, two_stage: bool, fleet_proven: bool,
                  cost_stage_executed: bool, final_status: int) -> str:
    """Name exactly what, if anything, the final status proves."""

    if not two_stage:
        return "full_pool_objective" if final_status == 2 else "none"
    if not fleet_proven:
        return "none"
    if cost_stage_executed and final_status == 2:
        return "full_pool_lexicographic"
    return "fleet_only"


def validate_final_selected_routes(
    status, trips, selected_routes, *, data_dir=None
) -> None:
    """Rebuild the instance and physically replay every final selected route."""

    from audit_giro_known_columns import HORIZON_MIN, build_problem

    provenance = status.get("provenance") or {}
    data_dir = Path(
        data_dir
        if data_dir is not None
        else Path(__file__).resolve().parent.parent / "data"
    ).resolve()
    instance_path = (data_dir / str(status.get("csv"))).resolve()
    try:
        instance_path.relative_to(data_dir)
    except ValueError as exc:
        raise SystemExit("[MIP] final replay instance escapes data/") from exc
    expected_hash = provenance.get("instance_sha256")
    if (
        not instance_path.is_file()
        or not isinstance(expected_hash, str)
        or file_sha256(instance_path) != expected_hash
    ):
        raise SystemExit("[MIP] final replay instance hash mismatch")
    problem = build_problem(
        data_dir,
        str(instance_path.relative_to(data_dir)),
        max_station_to_trip_wait_min=HORIZON_MIN,
    )
    if list(problem.trips) != list(trips):
        raise SystemExit("[MIP] final replay reconstructed a different trip set")
    try:
        g_kwh = float(status["g_kwh"])
        charge_kw = float(status["charge_kw"])
        reserve_frac = float(status["min_soc_frac"])
        reserve_kwh = reserve_frac * g_kwh
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("[MIP] final replay physics are invalid") from exc
    if (
        not all(math.isfinite(value) for value in (
            g_kwh, charge_kw, reserve_frac, reserve_kwh
        ))
        or g_kwh <= 0.0
        or charge_kw <= 0.0
        or not 0.0 <= reserve_frac <= 1.0
    ):
        raise SystemExit("[MIP] final replay physics are invalid/non-finite")
    counts = Counter()
    for ordinal, route in enumerate(selected_routes, start=1):
        route_trips = list(route.get("trips") or [])
        counts.update(route_trips)
        route_nodes = route.get(
            "route_nodes", route.get("route", [])
        )
        node_trips = [
            node for node in route_nodes
            if isinstance(node, int) and not isinstance(node, bool)
        ]
        if node_trips != route_trips:
            raise SystemExit(
                f"[MIP] final selected route {ordinal} trip incidence "
                "differs from its replayed route nodes"
            )
        candidate = {
            "trips": route_trips,
            "route_nodes": route_nodes,
            "charging_stops": route.get("charging_stops", {}),
        }
        reason = validate_injected_route(
            problem,
            candidate,
            g_kwh,
            charge_kw,
            reserve_kwh,
            HORIZON_MIN,
        )
        if reason is not None:
            raise SystemExit(
                f"[MIP] final selected route {ordinal} failed physical "
                f"replay: {reason}"
            )
    if any(counts[trip] != 1 for trip in trips):
        raise SystemExit(
            "[MIP] final selected routes do not cover every trip exactly once"
        )


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
    parser.add_argument(
        "--initial-partition-routes",
        type=Path,
        default=None,
        help="Runner-format routes JSON that must validate as one exact "
             "partition under the pool physics. Its routes are merged into "
             "the pool and used explicitly as the complete MIP start.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Lexicographic solve: stage 1 minimizes the bus count alone; "
             "stage 1 may use the full --timelimit. Stage 2 minimizes route "
             "cost only if stage 1 proves the fleet early and time remains.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--progress-dir",
        type=Path,
        default=None,
        help="Opt-in directory for atomic observational convergence "
             "checkpoints (not a Gurobi tree restart).",
    )
    args = parser.parse_args(argv)

    if args.out is not None and args.out.resolve() == args.result.resolve():
        parser.error("--out must not overwrite --result")
    code_identity = verified_mip_code_identity()

    # Bind the solve to immutable bytes.  If a caller mistakenly gives a live
    # journal and it changes while being loaded, refuse the ambiguous result.
    with open(args.result) as fh:
        source_status = json.load(fh)
    source_journal = resolve_pool_journal(args.result, source_status)
    source_result_sha256 = file_sha256(args.result)
    source_journal_sha256 = file_sha256(source_journal)
    expected_result_sha256 = os.environ.get(
        "EVSP_MIP_EXPECTED_RESULT_SHA256"
    )
    expected_journal_sha256 = os.environ.get(
        "EVSP_MIP_EXPECTED_JOURNAL_SHA256"
    )
    if os.environ.get("SLURM_JOB_ID"):
        if not expected_result_sha256 or not expected_journal_sha256:
            raise SystemExit(
                "[MIP] submitted solve lacks required input hashes"
            )
    if (expected_result_sha256
            and source_result_sha256 != expected_result_sha256):
        raise SystemExit(
            "[MIP] staged result does not match its submission-manifest hash"
        )
    if (expected_journal_sha256
            and source_journal_sha256 != expected_journal_sha256):
        raise SystemExit(
            "[MIP] staged journal does not match its submission-manifest hash"
        )
    expected_initial_partition_sha256 = os.environ.get(
        "EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256"
    )
    if (os.environ.get("SLURM_JOB_ID")
            and args.initial_partition_routes is not None
            and not expected_initial_partition_sha256):
        raise SystemExit(
            "[MIP] submitted initial partition lacks required hash"
        )
    if expected_initial_partition_sha256:
        if args.initial_partition_routes is None:
            raise SystemExit(
                "[MIP] submission expects an initial partition but none was "
                "supplied"
            )
        if (file_sha256(args.initial_partition_routes)
                != expected_initial_partition_sha256):
            raise SystemExit(
                "[MIP] staged initial partition does not match its "
                "submission-manifest hash"
            )
    extra_route_sources = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in (args.extra_routes or [])
    ]
    initial_partition_start = None

    status, routes, trips = load_pool(args.result)
    if (file_sha256(args.result) != source_result_sha256
            or file_sha256(source_journal) != source_journal_sha256):
        raise SystemExit(
            "[MIP] source status or column journal changed while the pool was "
            "being loaded; use an immutable snapshot and retry"
        )
    if args.extra_routes:
        routes = merge_extra_routes(routes, trips, args.extra_routes,
                                    status.get("prices_csv"), status)
        for recorded, path in zip(extra_route_sources, args.extra_routes):
            if file_sha256(path) != recorded["sha256"]:
                raise SystemExit(
                    f"[MIP] extra route source changed while loading: {path}"
                )
    if args.initial_partition_routes is not None:
        routes, mip_start, initial_partition_start = (
            merge_validated_partition_start(
                routes,
                trips,
                args.initial_partition_routes,
                status.get("prices_csv"),
                status,
            )
        )
    coverage = Counter(t for r in routes for t in r["trips"])
    uncovered = [t for t in trips if coverage[t] == 0]
    seed_partition = singleton_partition_indices(routes, trips)
    if initial_partition_start is None:
        mip_start = greedy_partition_start_indices(
            routes, trips, seed_partition
        )
        initial_partition_start = {
            "kind": (
                "greedy_pool_partition" if mip_start else "none"
            ),
            "source": None,
            "source_sha256": None,
            "validated": bool(mip_start),
            "validated_bus_count": len(mip_start) if mip_start else None,
            "expected_full_objective": (
                float(sum(routes[index]["cost"] for index in mip_start))
                if mip_start else None
            ),
        }
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
        if initial_partition_start["kind"] == "greedy_pool_partition":
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
        if initial_partition_start["kind"] == "validated_exact_partition":
            print("[MIP] validate-only: supplied start is a physically valid "
                  "exact partition. OK.")
        elif seed_partition:
            print("[MIP] validate-only: strict partition feasibility is "
                  "guaranteed by the singleton seed. OK.")
        else:
            print("[MIP] validate-only: row coverage only; strict partition "
                  "feasibility has NOT been established.")
        return 0

    out = args.out or Path(str(args.result).replace(".json", "_mip.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    progress_path = (
        args.progress_dir.expanduser().resolve()
        if args.progress_dir is not None else None
    )
    if progress_path is not None:
        protected = {
            args.result.resolve(),
            source_journal.resolve(),
            out.resolve(),
            *(
                path.resolve()
                for path in (args.extra_routes or [])
            ),
        }
        if args.initial_partition_routes is not None:
            protected.add(args.initial_partition_routes.resolve())
        if progress_path in protected:
            parser.error("--progress-dir aliases a protected input/output")
        if progress_path.exists():
            parser.error("--progress-dir already exists; choose a new path")

    import gurobipy as gp
    from gurobipy import GRB
    from config import BUS_COST_KX
    from mip_convergence import (
        GurobiProgressObserver,
        MIPProgressRecorder,
        TerminationRequest,
    )

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

    t0 = time.time()
    m = gp.Model("exact_pool_mip")
    m.Params.TimeLimit = args.timelimit
    m.Params.MIPGap = args.mipgap
    m.Params.Threads = args.threads
    experiment_arm = {
        (False, False): "A",
        (True, False): "B",
        (False, True): "C",
        (True, True): "D",
    }[(args.two_stage, args.initial_partition_routes is not None)]
    progress = None
    termination = None
    if progress_path is not None:
        progress = MIPProgressRecorder(
            progress_path,
            time_limit_s=args.timelimit,
            metadata={
                "source_result_sha256": source_result_sha256,
                "source_journal_sha256": source_journal_sha256,
                "source_initial_partition_sha256": (
                    expected_initial_partition_sha256
                    or initial_partition_start.get("source_sha256")
                ),
                "extra_route_sources": extra_route_sources,
                "gurobi_version": ".".join(
                    str(value) for value in gp.gurobi.version()
                ),
                "parameters": {
                    "time_limit_s": args.timelimit,
                    "mip_gap": args.mipgap,
                    "threads": args.threads,
                    "two_stage": args.two_stage,
                    "cover": args.cover,
                },
                "git_commit": code_identity["observed_commit"],
                "expected_git_commit": code_identity["expected_commit"],
                "experiment_arm": experiment_arm,
            },
        )
        termination = TerminationRequest()
        termination.install()
    a = m.addVars(len(routes), vtype=GRB.BINARY, name="a")
    if mip_start:
        start_set = set(mip_start)
        for index in range(len(routes)):
            a[index].Start = 1.0 if index in start_set else 0.0
    initial_partition_start["assignment_complete"] = bool(mip_start)
    initial_partition_start["assigned_variable_count"] = (
        len(routes) if mip_start else 0
    )
    initial_partition_start["selected_variable_count"] = (
        len(mip_start) if mip_start else 0
    )
    if progress is not None:
        progress.transition_stage(
            "fleet" if args.two_stage else "single",
            elapsed_s=0.0,
        )
        if mip_start:
            progress.record_initial_incumbent(
                list(mip_start),
                objective=float(sum(
                    routes[index]["cost"] for index in mip_start
                )),
                fleet=len(mip_start),
                kind=(
                    "validated_partition_at_t0"
                    if initial_partition_start["kind"]
                    == "validated_exact_partition"
                    else "initial_mip_start_at_t0"
                ),
            )
        progress.emit_zero()
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

    def progress_observer(stage, fixed_fleet=None):
        if progress is None:
            return None
        return GurobiProgressObserver(
            progress,
            GRB=GRB,
            variables=a,
            routes=routes,
            bus_cost=BUS_COST_KX,
            stage=stage,
            fixed_fleet=fixed_fleet,
            termination=termination,
        )

    two_stage_detail = None
    cost_stage_executed = False
    cost_stage_has_solution = False
    validated_start_available = (
        initial_partition_start["kind"] == "validated_exact_partition"
        and bool(mip_start)
    )
    if args.two_stage:
        # Fleet recovery is the primary experiment.  Stage 1 may consume the
        # complete budget.  Cost optimization is allowed only after the
        # integer fleet count has been proved, using whatever time remains.
        m.Params.TimeLimit = args.timelimit
        m.setObjective(
            gp.quicksum(a[i] for i in range(len(routes))), GRB.MINIMIZE
        )
        initial_partition_start["solver_acceptance"] = (
            optimize_with_start_audit(
                m,
                GRB,
                start_supplied=bool(mip_start),
                progress_observer=progress_observer("fleet"),
            )
        )
        stage1_has_solution = m.SolCount > 0
        validated_start_fallback = (
            not stage1_has_solution
            and validated_start_available
        )
        stage1_buses = (
            int(round(m.ObjVal))
            if stage1_has_solution
            else (len(mip_start) if validated_start_fallback else None)
        )
        stage1_bound = finite_solver_value(m.ObjBound)
        stage1_status = int(m.Status)
        stage1_gap = (
            finite_solver_value(m.MIPGap)
            if stage1_has_solution else None
        )
        fleet_proven = (
            fleet_bound_proves_incumbent(
                stage1_buses, stage1_bound, stage1_status
            )
            if (stage1_has_solution or validated_start_fallback) else False
        )
        stage1_solution = (
            [i for i in range(len(routes)) if a[i].X > 0.5]
            if stage1_has_solution
            else (list(mip_start) if validated_start_fallback else [])
        )
        if (
            validated_start_fallback
            and stage1_bound is not None
            and stage1_buses is not None
        ):
            stage1_gap = (
                max(0.0, stage1_buses - stage1_bound)
                / max(1.0, stage1_buses)
            )
        stage1_runtime_s = (
            progress.elapsed_s() if progress is not None
            else time.time() - t0
        )
        remaining_s = max(0.0, float(args.timelimit) - stage1_runtime_s)
        print(
            f"[MIP] stage 1: fleet={stage1_buses} "
            f"(bound {stage1_bound}, gap {stage1_gap}, "
            f"proven={fleet_proven}, remaining={remaining_s:.1f}s)"
        )
        two_stage_detail = {
            "stage1_buses": stage1_buses,
            "stage1_solver_has_solution": stage1_has_solution,
            "stage1_incumbent_source": (
                "solver" if stage1_has_solution
                else (
                    "validated_start_fallback"
                    if validated_start_fallback else None
                )
            ),
            "stage1_bound": stage1_bound,
            "stage1_status": stage1_status,
            "stage1_status_name": status_names.get(
                stage1_status, f"UNKNOWN_{stage1_status}"
            ),
            "stage1_gap": stage1_gap,
            "fleet_proven": fleet_proven,
            "stage1_runtime_s": stage1_runtime_s,
            "stage2_executed": False,
            "stage2_has_solution": False,
            "stage2_skip_reason": None,
        }
        if (
            stage1_has_solution
            and fleet_proven
            and remaining_s >= 1.0
            and not (termination and termination.requested)
        ):
            m.addConstr(
                gp.quicksum(a[i] for i in range(len(routes))) == stage1_buses,
                name="fleet_budget",
            )
            stage1_set = set(stage1_solution)
            for index in range(len(routes)):
                a[index].Start = 1.0 if index in stage1_set else 0.0
            variable_costs = [
                route["cost"] - BUS_COST_KX for route in routes
            ]
            m.setObjective(
                gp.quicksum(variable_costs[i] * a[i]
                            for i in range(len(routes))),
                GRB.MINIMIZE,
            )
            m.Params.TimeLimit = remaining_s
            if progress is not None:
                progress.transition_stage(
                    "cost", elapsed_s=progress.elapsed_s()
                )
            stage2_start_acceptance = optimize_with_start_audit(
                m,
                GRB,
                start_supplied=True,
                progress_observer=progress_observer(
                    "cost", fixed_fleet=stage1_buses
                ),
            )
            cost_stage_executed = True
            cost_stage_has_solution = m.SolCount > 0
            two_stage_detail["stage2_executed"] = True
            two_stage_detail["stage2_has_solution"] = cost_stage_has_solution
            two_stage_detail["stage2_start_acceptance"] = (
                stage2_start_acceptance
            )
        elif termination and termination.requested:
            two_stage_detail["stage2_skip_reason"] = (
                "termination_signal_requested"
            )
        elif not stage1_has_solution:
            two_stage_detail["stage2_skip_reason"] = "no_fleet_incumbent"
        elif not fleet_proven:
            two_stage_detail["stage2_skip_reason"] = "fleet_not_proven"
        else:
            two_stage_detail["stage2_skip_reason"] = "no_time_remaining"
    else:
        m.setObjective(
            gp.quicksum(routes[i]["cost"] * a[i]
                        for i in range(len(routes))),
            GRB.MINIMIZE,
        )
        initial_partition_start["solver_acceptance"] = (
            optimize_with_start_audit(
                m,
                GRB,
                start_supplied=bool(mip_start),
                progress_observer=progress_observer("single"),
            )
        )
        fleet_proven = int(m.Status) == 2

    if args.two_stage and not stage1_has_solution and validated_start_fallback:
        chosen = list(stage1_solution)
        status_code = stage1_status
        solver_obj = float(stage1_buses)
        solver_bound = stage1_bound
        mip_gap = stage1_gap
    elif args.two_stage and not stage1_has_solution:
        chosen = []
        status_code = stage1_status
        solver_obj = None
        solver_bound = stage1_bound
        mip_gap = None
    elif args.two_stage and not cost_stage_executed:
        chosen = list(stage1_solution)
        status_code = stage1_status
        solver_obj = float(stage1_buses)
        solver_bound = stage1_bound
        mip_gap = stage1_gap
    elif args.two_stage and not cost_stage_has_solution:
        # The proved fleet incumbent is still a valid deliverable even if the
        # second optimizer fails to accept its warm start before interruption.
        chosen = list(stage1_solution)
        status_code = int(m.Status)
        solver_obj = float(sum(
            routes[i]["cost"] - BUS_COST_KX for i in chosen
        ))
        solver_bound = finite_solver_value(m.ObjBound)
        mip_gap = (
            max(0.0, solver_obj - solver_bound) / max(1.0, abs(solver_obj))
            if solver_bound is not None else None
        )
    elif (
        not args.two_stage
        and m.SolCount == 0
        and validated_start_available
    ):
        chosen = list(mip_start)
        status_code = int(m.Status)
        solver_obj = float(sum(routes[index]["cost"] for index in chosen))
        solver_bound = finite_solver_value(m.ObjBound)
        mip_gap = (
            max(0.0, solver_obj - solver_bound)
            / max(1.0, abs(solver_obj))
            if solver_bound is not None else None
        )
    else:
        status_code = int(m.Status)
        chosen = [i for i in range(len(routes)) if a[i].X > 0.5] \
            if m.SolCount > 0 else []
        solver_obj = finite_solver_value(m.ObjVal) if m.SolCount > 0 else None
        solver_bound = finite_solver_value(m.ObjBound)
        mip_gap = finite_solver_value(m.MIPGap) if m.SolCount > 0 else None

    status_name = status_names.get(status_code, f"UNKNOWN_{status_code}")
    selected_routes = [routes[i] for i in chosen]
    if selected_routes and not args.cover:
        validate_final_selected_routes(status, trips, selected_routes)
    over = {t: c for t, c in Counter(
        t for i in chosen for t in routes[i]["trips"]).items() if c > 1}
    has_incumbent = bool(chosen)
    solver_incumbent_found = (
        bool(stage1_has_solution)
        if args.two_stage and not cost_stage_executed
        else bool(m.SolCount > 0)
    )
    incumbent_source = (
        "validated_start_fallback"
        if validated_start_available
        and not solver_incumbent_found
        and has_incumbent
        else ("solver" if has_incumbent else None)
    )
    mip_obj = (float(sum(routes[i]["cost"] for i in chosen))
               if chosen else None)
    if cost_stage_executed and solver_bound is not None and two_stage_detail:
        mip_bound = (BUS_COST_KX * two_stage_detail["stage1_buses"]
                     + solver_bound)
        mip_bound_scope = "fixed_proven_fleet_variable_cost"
    elif args.two_stage and solver_bound is not None:
        # Route variable costs are nonnegative, so this is a valid but coarse
        # lower bound on the full lexicographic objective.
        mip_bound = BUS_COST_KX * solver_bound
        mip_bound_scope = "fleet_count_only_coarse_cost_bound"
    else:
        mip_bound = solver_bound
        mip_bound_scope = "full_pool_objective"
    stage2_absolute_gap = (
        max(0.0, solver_obj - solver_bound)
        if (cost_stage_executed and solver_obj is not None
            and solver_bound is not None) else None
    )
    if two_stage_detail is not None:
        two_stage_detail.update({
            "stage2_status": status_code if cost_stage_executed else None,
            "stage2_status_name": status_name if cost_stage_executed else None,
            "stage2_variable_obj": solver_obj if cost_stage_executed else None,
            "stage2_variable_bound": (solver_bound
                                      if cost_stage_executed else None),
            "stage2_absolute_gap": stage2_absolute_gap,
            "stage2_reported_incumbent_source": (
                "stage2_solver" if cost_stage_has_solution
                else ("stage1_fallback" if cost_stage_executed else None)
            ),
        })
    final_code_identity = verified_mip_code_identity()
    if (final_code_identity["observed_commit"]
            != code_identity["observed_commit"]):
        raise SystemExit("[MIP] solver commit changed during optimization")
    code_identity["final_observed_commit"] = final_code_identity[
        "observed_commit"
    ]
    code_identity["final_tracked_clean"] = final_code_identity[
        "tracked_clean"
    ]
    summary = {
        "source_result": str(args.result),
        "instance": status["csv"],
        "partitioning": not args.cover,
        "experiment_arm": experiment_arm,
        "status": status_code,
        "status_name": status_name,
        "optimal_scope": optimal_scope(
            two_stage=args.two_stage,
            fleet_proven=fleet_proven,
            cost_stage_executed=cost_stage_executed,
            final_status=status_code,
        ),
        "mip_obj": mip_obj,
        "mip_bound": mip_bound,
        "mip_bound_scope": mip_bound_scope,
        "requested_mip_gap": args.mipgap,
        "mip_gap": mip_gap,
        "absolute_cost_gap": stage2_absolute_gap,
        "buses": len(chosen) if has_incumbent else None,
        "incumbent_found": has_incumbent,
        "solver_incumbent_found": solver_incumbent_found,
        "incumbent_source": incumbent_source,
        "fleet_proven": fleet_proven,
        "fleet_bound": (two_stage_detail.get("stage1_bound")
                        if two_stage_detail else None),
        "charging_cost": (mip_obj - BUS_COST_KX * len(chosen))
                         if chosen and mip_obj is not None else None,
        "variable_route_cost": (mip_obj - BUS_COST_KX * len(chosen))
                               if chosen and mip_obj is not None else None,
        "overcovered_trips": len(over),
        "runtime_s": (
            progress.elapsed_s() if progress is not None
            else time.time() - t0
        ),
        "pool_columns": len(routes),
        "singleton_partition_columns": len(seed_partition),
        "mip_start_assigned": bool(mip_start),
        "mip_start_used": (
            (initial_partition_start.get("solver_acceptance") or {})
            .get("accepted") is True
        ),
        "mip_start_buses": len(mip_start) if mip_start else None,
        "mip_start": initial_partition_start,
        "pool_preparation": status.get("pool_preparation"),
        "source_cg_wall_s": status.get("wall_s"),
        "source_cg_iterations": status.get("iterations"),
        "source_snapshot_mark_minutes": status.get(
            "snapshot_mark_minutes"
        ),
        "two_stage": two_stage_detail,
        "pricer_provenance": status.get("provenance"),
        "physics": {
            "soc_step": status.get("soc_step"),
            "block_min": status.get("block_min"),
            "g_kwh": status.get("g_kwh"),
            "charge_kw": status.get("charge_kw"),
            "min_soc_frac": status.get("min_soc_frac"),
            "prices_csv": status.get("prices_csv"),
        },
        "source_result_sha256": source_result_sha256,
        "source_journal": str(source_journal),
        "source_journal_sha256": source_journal_sha256,
        "extra_route_sources": extra_route_sources,
        "mip_provenance": {
            "git_commit": code_identity["observed_commit"],
            "expected_git_commit": code_identity["expected_commit"],
            "observed_git_commit": code_identity["observed_commit"],
            "final_observed_git_commit": code_identity[
                "final_observed_commit"
            ],
            "git_branch": code_identity["branch"],
            "git_detached": code_identity["detached"],
            "git_dirty": not code_identity["tracked_clean"],
            "tracked_clean_at_end": code_identity["final_tracked_clean"],
            "python": platform.python_version(),
            "gurobi": ".".join(str(value) for value in gp.gurobi.version()),
            "arguments": {
                "timelimit": args.timelimit,
                "mipgap": args.mipgap,
                "threads": args.threads,
                "two_stage": args.two_stage,
                "cover": args.cover,
                "initial_partition_routes": (
                    str(args.initial_partition_routes)
                    if args.initial_partition_routes is not None else None
                ),
            },
        },
        "selected_routes": selected_routes,
        "progress": (
            {
                "directory": str(progress_path),
                "checkpoint_schedule_s": progress.schedule,
                "observational_only": True,
                "gurobi_tree_restart_supported": False,
                "disabled_reason": progress.disabled_reason,
                "termination_signal": (
                    termination.signal_name if termination else None
                ),
            }
            if progress is not None else None
        ),
    }
    atomic_write_json(out, summary)
    if progress is not None:
        progress.finalize(
            elapsed_s=summary["runtime_s"],
            final={
                "status": status_code,
                "status_name": status_name,
                "incumbent_found": has_incumbent,
                "buses": summary["buses"],
                "mip_obj": mip_obj,
                "mip_bound": mip_bound,
                "mip_gap": mip_gap,
                "fleet_proven": fleet_proven,
                "optimal_scope": summary["optimal_scope"],
                "selected_route_indices": list(chosen),
                "route_vector_sha256": (
                    hashlib.sha256(json.dumps(
                        sorted(chosen), separators=(",", ":")
                    ).encode()).hexdigest()
                    if chosen else None
                ),
                "termination_signal": (
                    termination.signal_name if termination else None
                ),
            },
        )
    if termination is not None:
        termination.restore()
    print(f"[MIP] status={status_name}({status_code}) buses={summary['buses']} "
          f"obj={summary['mip_obj']} gap={summary['mip_gap']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

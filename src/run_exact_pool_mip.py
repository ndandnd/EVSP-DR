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


def load_pool(result_path: Path):
    with open(result_path) as fh:
        status = json.load(fh)

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
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.out is not None and args.out.resolve() == args.result.resolve():
        parser.error("--out must not overwrite --result")

    status, routes, trips = load_pool(args.result)
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

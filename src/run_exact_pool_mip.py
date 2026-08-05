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

`--validate-only` loads the pool, checks coverage and prints statistics
without importing Gurobi (usable on machines without a license).
"""

from __future__ import annotations

import argparse
import json
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True,
                        help="Exact-pricer status JSON (with columns_journal).")
    parser.add_argument("--timelimit", type=int, default=3600)
    parser.add_argument("--mipgap", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cover", action="store_true",
                        help="Relax partitioning (==1) to covering (>=1).")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.out is not None and args.out.resolve() == args.result.resolve():
        parser.error("--out must not overwrite --result")

    status, routes, trips = load_pool(args.result)
    coverage = Counter(t for r in routes for t in r["trips"])
    uncovered = [t for t in trips if coverage[t] == 0]
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
    if args.validate_only:
        print("[MIP] validate-only: pool is coverage-complete. OK.")
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

    chosen = [i for i in range(len(routes)) if a[i].X > 0.5] \
        if m.SolCount > 0 else []
    over = {t: c for t, c in Counter(
        t for i in chosen for t in routes[i]["trips"]).items() if c > 1}
    summary = {
        "source_result": str(args.result),
        "instance": status["csv"],
        "partitioning": not args.cover,
        "status": int(m.Status),
        "mip_obj": m.ObjVal if m.SolCount > 0 else None,
        "mip_bound": m.ObjBound,
        "mip_gap": m.MIPGap if m.SolCount > 0 else None,
        "buses": len(chosen),
        "charging_cost": (m.ObjVal - 100000.0 * len(chosen)) if chosen else None,
        "overcovered_trips": len(over),
        "runtime_s": time.time() - t0,
        "pricer_provenance": status.get("provenance"),
        "selected_routes": [routes[i] for i in chosen],
    }
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[MIP] status={m.Status} buses={len(chosen)} "
          f"obj={summary['mip_obj']} gap={summary['mip_gap']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

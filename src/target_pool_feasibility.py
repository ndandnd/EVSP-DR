#!/usr/bin/env python3
"""Decide whether a finite exact-CG pool contains a target-size partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

from run_exact_pool_mip import (
    deduplicate_pool,
    file_sha256,
    ordered_pool_sha256,
    prepare_strict_partition_pool,
    resolve_pool_journal,
    validate_final_selected_routes,
    verified_mip_code_identity,
    write_new_json,
)


SCHEMA = "evsp-dr-target-pool-feasibility-v1"
OUTCOMES = {"INFEASIBLE", "FEASIBLE", "TIME_LIMIT"}


def classify_outcome(status, solution_count, GRB):
    """Map a completed feasibility search to one of three honest outcomes."""

    if int(solution_count) > 0:
        return "FEASIBLE"
    if status == GRB.INFEASIBLE:
        return "INFEASIBLE"
    if status == GRB.TIME_LIMIT:
        return "TIME_LIMIT"
    raise RuntimeError(
        f"target-feasibility solver ended without a classified outcome: {status}"
    )


def _trip_rows(routes, trips):
    rows = {trip: [] for trip in trips}
    for index, route in enumerate(routes):
        for trip in route["trips"]:
            if trip not in rows:
                raise ValueError(f"route {index} contains foreign trip {trip}")
            rows[trip].append(index)
    missing = [trip for trip, indices in rows.items() if not indices]
    if missing:
        raise ValueError(f"finite pool omits trips: {missing[:15]}")
    return rows


def load_bound_pool_bytes(status, journal_bytes):
    """Parse exactly the immutable journal bytes whose digest is recorded."""

    trips = status.get("trip_ids")
    if (
        not isinstance(trips, list)
        or any(not isinstance(trip, int) or isinstance(trip, bool)
               for trip in trips)
        or len(trips) != len(set(trips))
    ):
        raise ValueError("source status has invalid trip_ids")
    allowed = set(trips)
    routes = []
    for ordinal, line in enumerate(journal_bytes.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            routes.append(json.loads(line))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                f"journal route {ordinal} is malformed"
            ) from exc
    for ordinal, route in enumerate(routes, start=1):
        route_trips = route.get("trips") if isinstance(route, dict) else None
        if (
            not isinstance(route_trips, list)
            or not route_trips
            or any(not isinstance(trip, int) or isinstance(trip, bool)
                   for trip in route_trips)
            or len(route_trips) != len(set(route_trips))
            or not set(route_trips) <= allowed
        ):
            raise ValueError(f"journal route {ordinal} has invalid trips")
        try:
            cost = float(route["cost"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"journal route {ordinal} has invalid cost"
            ) from exc
        if not math.isfinite(cost):
            raise ValueError(f"journal route {ordinal} has non-finite cost")
    return routes, list(trips)


def load_bound_pool(status, journal_path):
    """Compatibility wrapper that reads one already canonical path once."""

    return load_bound_pool_bytes(status, Path(journal_path).read_bytes())


def solve_target_feasibility(
    routes,
    trips,
    target,
    *,
    timelimit,
    threads,
    seed=0,
):
    """Run the constant-objective target-constrained partition MIP."""

    if not isinstance(target, int) or isinstance(target, bool) or target < 1:
        raise ValueError("target must be a positive integer")
    if not math.isfinite(float(timelimit)) or float(timelimit) <= 0:
        raise ValueError("timelimit must be positive and finite")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    import gurobipy as gp
    from gurobipy import GRB

    rows = _trip_rows(routes, trips)
    model = gp.Model("target_pool_feasibility")
    model.Params.TimeLimit = float(timelimit)
    model.Params.Threads = threads
    model.Params.Seed = int(seed)
    model.Params.SolutionLimit = 1
    variables = model.addVars(len(routes), vtype=GRB.BINARY, name="route")
    for trip in trips:
        model.addConstr(
            gp.quicksum(variables[index] for index in rows[trip]) == 1,
            name=f"partition_{trip}",
        )
    model.addConstr(
        gp.quicksum(variables[index] for index in range(len(routes)))
        <= target,
        name="target_fleet",
    )
    model.setObjective(0.0, GRB.MINIMIZE)
    started = time.perf_counter()
    model.optimize()
    if model.Status == GRB.INF_OR_UNBD:
        model.Params.DualReductions = 0
        model.reset()
        model.optimize()
    runtime_s = time.perf_counter() - started
    outcome = classify_outcome(model.Status, model.SolCount, GRB)
    selected = (
        [
            index for index in range(len(routes))
            if float(variables[index].X) > 0.5
        ]
        if outcome == "FEASIBLE" else []
    )
    if outcome == "FEASIBLE":
        if len(selected) > target:
            raise RuntimeError("solver witness exceeds target fleet")
        counts = {trip: 0 for trip in trips}
        for index in selected:
            for trip in routes[index]["trips"]:
                counts[trip] += 1
        if any(value != 1 for value in counts.values()):
            raise RuntimeError("solver witness is not an exact partition")
    return {
        "outcome": outcome,
        "selected_indices": selected,
        "runtime_s": runtime_s,
        "solver_status": int(model.Status),
        "solver_status_name": {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        }.get(model.Status, f"STATUS_{model.Status}"),
        "solution_count": int(model.SolCount),
        "node_count": float(model.NodeCount),
        "parameters": {
            "timelimit": float(timelimit),
            "threads": threads,
            "seed": int(seed),
            "solution_limit": 1,
            "objective": "constant_zero_pure_feasibility",
        },
    }


def evaluate(args):
    requested_output = Path(args.out)
    if os.path.lexists(requested_output):
        raise FileExistsError(requested_output)
    output_path = requested_output.resolve()
    result_path = Path(args.result).resolve(strict=True)
    code_identity = verified_mip_code_identity()
    source_status_bytes = result_path.read_bytes()
    source_status = json.loads(source_status_bytes)
    source_journal = resolve_pool_journal(
        result_path, source_status
    ).resolve(strict=True)
    source_journal_bytes = source_journal.read_bytes()
    source_result_sha256 = hashlib.sha256(
        source_status_bytes
    ).hexdigest()
    source_journal_sha256 = hashlib.sha256(
        source_journal_bytes
    ).hexdigest()
    routes, trips = load_bound_pool_bytes(
        source_status, source_journal_bytes
    )
    if (
        file_sha256(result_path) != source_result_sha256
        or file_sha256(source_journal) != source_journal_sha256
    ):
        raise RuntimeError("target-feasibility source changed while loading")
    routes, physical_audit = prepare_strict_partition_pool(
        source_status,
        routes,
        data_dir=args.data_dir,
        reference_data_dir=args.reference_data_dir,
    )
    routes = deduplicate_pool(routes)
    pool_sha256 = ordered_pool_sha256(routes)
    solved = solve_target_feasibility(
        routes,
        trips,
        args.target,
        timelimit=args.timelimit,
        threads=args.threads,
        seed=args.seed,
    )
    selected_routes = [
        routes[index] for index in solved.pop("selected_indices")
    ]
    if solved["outcome"] == "FEASIBLE":
        validate_final_selected_routes(
            source_status,
            trips,
            selected_routes,
            data_dir=args.data_dir,
            reference_data_dir=args.reference_data_dir,
            physical_pool_audit=physical_audit,
        )
    if (
        file_sha256(result_path) != source_result_sha256
        or file_sha256(source_journal) != source_journal_sha256
    ):
        raise RuntimeError("target-feasibility source changed during solve")
    final_code_identity = verified_mip_code_identity()
    if final_code_identity != code_identity:
        raise RuntimeError("target-feasibility code identity changed during solve")
    payload = {
        "schema": SCHEMA,
        "outcome": solved["outcome"],
        "conclusion": (
            "target_partition_exists"
            if solved["outcome"] == "FEASIBLE"
            else "target_partition_absent_from_finite_pool"
            if solved["outcome"] == "INFEASIBLE"
            else None
        ),
        "censored": solved["outcome"] == "TIME_LIMIT",
        "target_fleet": args.target,
        "witness_route_count": (
            len(selected_routes) if solved["outcome"] == "FEASIBLE" else None
        ),
        "witness_routes": (
            selected_routes if solved["outcome"] == "FEASIBLE" else []
        ),
        "finite_pool_scope_only": True,
        "pricing_or_global_infeasibility_certified": False,
        "source": {
            "result": str(result_path),
            "result_sha256": source_result_sha256,
            "journal": str(source_journal),
            "journal_sha256": source_journal_sha256,
            "instance_sha256": (source_status.get("provenance") or {}).get(
                "instance_sha256"
            ),
            "pool_columns": len(routes),
            "pool_ordered_sha256": pool_sha256,
        },
        "physics": {
            key: source_status.get(key)
            for key in (
                "g_kwh", "charge_kw", "min_soc_frac",
                "soc_step", "block_min",
            )
        },
        "physical_pool_audit": physical_audit,
        "solver": solved,
        "code_identity": code_identity,
    }
    if payload["outcome"] not in OUTCOMES:
        raise RuntimeError("internal outcome classification error")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(output_path, payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--timelimit", type=float, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--reference-data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = evaluate(args)
    print(json.dumps({
        "outcome": payload["outcome"],
        "target_fleet": payload["target_fleet"],
        "witness_route_count": payload["witness_route_count"],
        "censored": payload["censored"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

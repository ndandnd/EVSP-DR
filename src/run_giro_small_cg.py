#!/usr/bin/env python3
"""Bounded nonlinear-physics CG prototype for Partille k2/k3 cohorts.

This experiment generates columns from trip-coverage and conservative
station-time capacity duals. Pricing enumerates the direct/charge transitions
in a bounded hold-until-departure policy; early-disconnect schedules are added
only as replay variants. Consequently the reported LPs are restricted-pool
values, never full-model lower bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from audit_giro_duty_recovery import (
    DEFAULT_HORIZON_MIN,
    fixed_sequence_recovery,
)
from audit_giro_known_columns import build_problem
from giro_partille_physics import PARTILLE_PROFILES
from giro_weighted_pricing import action_capacity_rows, weighted_price_route


SCHEMA = "evsp-dr-giro-small-cg-v2-capacity-dual-pricing"
CHARGER_COUNTS = {
    "2190L": 1, "4808": 1, "3127L": 2, "7880C": 1, "JON_A": 1,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_pool_atomic(path: Path, routes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for route in routes:
            handle.write(json.dumps(route, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def route_key(route: dict) -> str:
    occupancy = sorted(
        (
            action["station"],
            round(float(action["setup_start_min"]), 6),
            round(float(action["connection_end_min"]), 6),
        )
        for action in route["actions"]
        if action.get("kind") == "charge" and action["station"] != "PARX"
    )
    return canonical_sha({"trips": route["trips"], "occupancy": occupancy})


def capacity_incidence(route: dict) -> set[tuple[str, int]]:
    return {
        row
        for action in route["actions"]
        for row in action_capacity_rows(action)
        if row[0] in CHARGER_COUNTS
    }


def _master(
    routes,
    trips,
    *,
    sense,
    binary,
    capacity,
    artificial_penalty=1000.0,
    time_limit_s=None,
    threads=1,
    log_path=None,
):
    model = gp.Model(f"giro_small_{sense}_{'mip' if binary else 'lp'}")
    model.Params.OutputFlag = 1 if log_path else 0
    model.Params.Threads = int(threads)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        model.Params.LogFile = str(log_path)
    if time_limit_s is not None:
        model.Params.TimeLimit = float(time_limit_s)
    variables = model.addVars(
        len(routes), lb=0.0, ub=1.0 if binary else GRB.INFINITY,
        vtype=GRB.BINARY if binary else GRB.CONTINUOUS, name="route",
    )
    artificials = {}
    if not binary:
        artificials = model.addVars(len(trips), lb=0.0, name="artificial")
    trip_rows = {}
    for row, trip in enumerate(trips):
        expression = gp.quicksum(
            variables[index] for index, route in enumerate(routes)
            if trip in route["trips"]
        )
        if not binary:
            expression += artificials[row]
        trip_rows[trip] = (
            model.addConstr(expression >= 1.0, name=f"cover_{trip}")
            if sense == "cover"
            else model.addConstr(expression == 1.0, name=f"partition_{trip}")
        )
    capacity_rows = {}
    if capacity:
        memberships = [capacity_incidence(route) for route in routes]
        keys = sorted(set().union(*memberships)) if memberships else []
        for site, minute in keys:
            capacity_rows[site, minute] = model.addConstr(
                gp.quicksum(
                    variables[index] for index, rows in enumerate(memberships)
                    if (site, minute) in rows
                ) <= CHARGER_COUNTS[site],
                name=f"charger_{site}_{minute}",
            )
    objective = gp.quicksum(variables.values())
    if not binary:
        objective += artificial_penalty * gp.quicksum(artificials.values())
    model.setObjective(objective, GRB.MINIMIZE)
    started = time.perf_counter()
    model.optimize()
    runtime = time.perf_counter() - started
    has_solution = model.SolCount > 0
    result = {
        "status_code": int(model.Status),
        "status": {
            GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE", GRB.INF_OR_UNBD: "INF_OR_UNBD",
        }.get(model.Status, f"STATUS_{model.Status}"),
        "runtime_s": runtime,
        "has_solution": has_solution,
        "objective": float(model.ObjVal) if has_solution else None,
        "selected_indices": (
            [index for index in range(len(routes)) if variables[index].X > 0.5]
            if binary and has_solution else []
        ),
        "route_values": (
            [float(variables[index].X) for index in range(len(routes))]
            if not binary and has_solution else None
        ),
        "artificial_total": (
            sum(float(artificials[row].X) for row in range(len(trips)))
            if not binary and has_solution else None
        ),
        "capacity_row_count": len(capacity_rows),
        "capacity_time_grid_min": 1 if capacity else None,
    }
    if binary and has_solution:
        result.update({
            "objective_bound": float(model.ObjBound),
            "mip_gap": float(model.MIPGap),
            "node_count": float(model.NodeCount),
        })
    if not binary and model.Status == GRB.OPTIMAL:
        route_weight = sum(result["route_values"])
        result["route_weight"] = route_weight
        result["phase1_penalized"] = result["artificial_total"] > 1e-7
        result["restricted_pool_fleet_lp"] = (
            route_weight if result["artificial_total"] <= 1e-7 else None
        )
        result["trip_duals"] = {
            trip: float(constraint.Pi) for trip, constraint in trip_rows.items()
        }
        result["capacity_duals"] = {
            (site, minute): float(constraint.Pi)
            for (site, minute), constraint in capacity_rows.items()
        }
        result["nonzero_capacity_duals"] = {
            f"{site}@{minute}": float(constraint.Pi)
            for (site, minute), constraint in capacity_rows.items()
            if abs(float(constraint.Pi)) > 1e-10
        }
    return result


def _route(problem, trips, profile, source):
    result = fixed_sequence_recovery(
        problem, tuple(trips), profile, horizon_min=DEFAULT_HORIZON_MIN
    )
    if not result["feasible"]:
        return None
    return {
        "trips": list(trips),
        "actions": list(result["actions"]),
        "profile": profile.name,
        "cost": 1.0,
        "source": source,
        "charge_policy": "early_disconnect",
    }


def seed_routes(problem, frame, profile, mode):
    routes = []
    for trip in problem.trips:
        route = _route(problem, [trip], profile, "raw_singleton")
        if route is None:
            raise ValueError(f"no feasible RAW singleton for trip {trip}")
        routes.append(route)
    if mode == "known":
        if "Source_Duty" not in frame:
            raise ValueError("known seed mode requires Source_Duty column")
        for duty in frame["Source_Duty"].drop_duplicates():
            trips = tuple(int(index) for index in frame.index[frame["Source_Duty"] == duty])
            route = _route(problem, trips, profile, f"known_duty:{duty}")
            if route is None:
                raise ValueError(f"known duty {duty} is infeasible")
            routes.append(route)
    return routes


def run_cg_arm(
    *,
    sense,
    seeds,
    union_routes,
    union_keys,
    problem,
    profile,
    cg_wall_s,
    pricing_wall_s,
    pricing_label_limit,
    cg_max_iters,
    threads,
    pool_out,
):
    routes = list(seeds)
    keys = {route_key(route) for route in routes}
    started = time.perf_counter()
    iterations = []
    stop_reason = None
    any_pricing_guard = False
    capacity_active = True
    for iteration in range(int(cg_max_iters)):
        if time.perf_counter() - started >= cg_wall_s:
            stop_reason = "cg_wall_limit"
            break
        lp = _master(
            routes, list(problem.trips), sense=sense, binary=False,
            capacity=capacity_active, threads=threads,
        )
        if lp["status"] != "OPTIMAL":
            stop_reason = f"restricted_master_{lp['status'].lower()}"
            iterations.append({"iteration": iteration, "lp": lp})
            break
        priced = weighted_price_route(
            problem, profile, lp["trip_duals"],
            capacity_duals=lp["capacity_duals"],
            horizon_min=DEFAULT_HORIZON_MIN,
            wall_limit_s=min(
                pricing_wall_s,
                max(0.01, cg_wall_s - (time.perf_counter() - started)),
            ),
            label_limit=pricing_label_limit,
        )
        any_pricing_guard = any_pricing_guard or priced.get("guard") is not None
        added = []
        candidate = priced.get("route")
        if candidate is not None and priced["reduced_cost_with_capacity_duals"] < -1e-8:
            candidate["source"] = f"weighted_pricing:{sense}:{iteration}"
            variants = [candidate]
            early = _route(
                problem, candidate["trips"], profile,
                f"early_replay:{sense}:{iteration}",
            )
            if early is not None:
                variants.append(early)
            for route in variants:
                key = route_key(route)
                if key not in keys:
                    keys.add(key)
                    routes.append(route)
                    added.append(key)
                if key not in union_keys:
                    union_keys.add(key)
                    union_routes.append(route)
        iterations.append({
            "iteration": iteration,
            "pricing_master_capacity_rows_enabled": capacity_active,
            "lp": {
                key: value for key, value in lp.items()
                if key not in {"trip_duals", "capacity_duals"}
            },
            "pricing": {key: value for key, value in priced.items() if key != "route"},
            "added_route_keys": added,
            "arm_pool_size": len(routes),
            "union_pool_size": len(union_routes),
        })
        write_pool_atomic(pool_out, union_routes)
        if candidate is None:
            stop_reason = "no_feasible_pricing_route"
            break
        if priced["reduced_cost_with_capacity_duals"] >= -1e-8:
            stop_reason = "no_negative_route_in_hold_policy_pricing"
            break
        if not added:
            stop_reason = "duplicate_best_capacity_aware_route"
            break
    else:
        stop_reason = "cg_iteration_limit"
    return {
        "sense": sense,
        "seed_pool_size": len(seeds),
        "final_arm_pool_size": len(routes),
        "runtime_s": time.perf_counter() - started,
        "stop_reason": stop_reason,
        "pricing_guard_encountered": any_pricing_guard,
        "full_model_lp_bound_certified": False,
        "iterations": iterations,
    }


def cover_repair(routes, selected, trips):
    occurrences = {trip: [] for trip in trips}
    for index in selected:
        for trip in routes[index]["trips"]:
            occurrences[trip].append(index)
    duplicates = {str(trip): rows for trip, rows in occurrences.items() if len(rows) > 1}
    return {
        "duplicate_trip_count": len(duplicates),
        "duplicate_occurrence_count": sum(len(rows) - 1 for rows in duplicates.values()),
        "duplicate_routes": duplicates,
        "repair": (
            "assign each passenger trip to one selected route and reclassify "
            "duplicate traversals as nonrevenue movements with unchanged time/energy"
        ),
        "repaired_passenger_partition_claimed": False,
    }


def execute(args):
    instance = args.instance.resolve()
    if args.expected_instance_sha256 and file_sha256(instance) != args.expected_instance_sha256:
        raise ValueError("instance SHA-256 mismatch")
    frame = pd.read_csv(instance).reset_index(drop=True)
    problem = build_problem(
        instance.parent, instance.name,
        reference_data_dir=args.reference_data_dir.resolve(),
        horizon_min=DEFAULT_HORIZON_MIN,
        max_trip2trip_min=DEFAULT_HORIZON_MIN,
        max_trip_to_station_min=DEFAULT_HORIZON_MIN,
        max_station_to_trip_wait_min=DEFAULT_HORIZON_MIN,
    )
    profile = PARTILLE_PROFILES[args.vehicle_profile]
    seeds = seed_routes(problem, frame, profile, args.seed_mode)
    union_routes = []
    union_keys = set()
    for route in seeds:
        key = route_key(route)
        if key not in union_keys:
            union_keys.add(key)
            union_routes.append(route)
    write_pool_atomic(args.pool_out.resolve(), union_routes)
    arms = []
    for sense in ("cover", "partition"):
        arms.append(run_cg_arm(
            sense=sense, seeds=seeds, union_routes=union_routes,
            union_keys=union_keys, problem=problem, profile=profile,
            cg_wall_s=args.cg_wall_s, pricing_wall_s=args.pricing_wall_s,
            pricing_label_limit=args.pricing_label_limit,
            cg_max_iters=args.cg_max_iters, threads=args.threads,
            pool_out=args.pool_out.resolve(),
        ))
    pool_sha = canonical_sha(union_routes)
    log_dir = args.log_dir.resolve()
    final = {}
    for capacity_name, capacity, limit in (
        ("constrained", True, args.mip_wall_s),
        ("no_capacity_relaxation", False, args.relaxed_mip_wall_s),
    ):
        final[capacity_name] = {}
        for sense in ("cover", "partition"):
            lp = _master(
                union_routes, list(problem.trips), sense=sense, binary=False,
                capacity=capacity, threads=args.threads,
                log_path=log_dir / f"{capacity_name}_{sense}_lp.log",
            )
            mip = _master(
                union_routes, list(problem.trips), sense=sense, binary=True,
                capacity=capacity, time_limit_s=limit, threads=args.threads,
                log_path=log_dir / f"{capacity_name}_{sense}_mip.log",
            )
            if sense == "cover" and mip["has_solution"]:
                mip["cover_repair_scope"] = cover_repair(
                    union_routes, mip["selected_indices"], list(problem.trips)
                )
            lp.pop("trip_duals", None)
            lp.pop("capacity_duals", None)
            final[capacity_name][sense] = {"lp": lp, "mip": mip}
    payload = {
        "schema": SCHEMA,
        "instance": str(instance),
        "instance_sha256": file_sha256(instance),
        "vehicle_profile": args.vehicle_profile,
        "seed_mode": args.seed_mode,
        "trip_count": len(problem.trips),
        "shared_union_pool_size": len(union_routes),
        "shared_union_pool_sha256": pool_sha,
        "pool_path": str(args.pool_out.resolve()),
        "parameters": {
            "cg_wall_s_per_arm": args.cg_wall_s,
            "pricing_wall_s_per_iteration": args.pricing_wall_s,
            "pricing_label_limit": args.pricing_label_limit,
            "cg_max_iters": args.cg_max_iters,
            "mip_wall_s": args.mip_wall_s,
            "relaxed_mip_wall_s": args.relaxed_mip_wall_s,
            "threads": args.threads,
        },
        "cg_arms": arms,
        "final_same_pool_comparisons": final,
        "reporting_scope": {
            "full_model_lp_bound_certified": False,
            "reason": (
                "pricing includes conservative station-capacity duals but is "
                "guard-limited and searches only the hold-until-departure "
                "charging policy; early-disconnect variants are replayed only "
                "after a trip sequence is selected"
            ),
            "capacity_discretization": (
                "plug occupancy is conservatively rounded onto one-minute rows; "
                "this is not exact continuous-event capacity"
            ),
            "cover_interpretation": (
                "duplicate passenger coverage may be reassigned, but physical "
                "nonrevenue repair is not replayed or certified"
            ),
            "pricing_physics": (
                "documented nonlinear single-vehicle Partille physics on static "
                "symmetric reference deadheads"
            ),
        },
    }
    write_pool_atomic(args.pool_out.resolve(), union_routes)
    write_json_atomic(args.out.resolve(), payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parent.parent
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--expected-instance-sha256")
    parser.add_argument("--vehicle-profile", choices=sorted(PARTILLE_PROFILES), required=True)
    parser.add_argument("--seed-mode", choices=("raw", "known"), default="raw")
    parser.add_argument("--cg-wall-s", type=float, default=600.0)
    parser.add_argument("--pricing-wall-s", type=float, default=30.0)
    parser.add_argument("--pricing-label-limit", type=int, default=250000)
    parser.add_argument("--cg-max-iters", type=int, default=200)
    parser.add_argument("--mip-wall-s", type=float, default=300.0)
    parser.add_argument("--relaxed-mip-wall-s", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reference-data-dir", type=Path, default=repo / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pool-out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.cg_wall_s, args.pricing_wall_s, args.mip_wall_s) <= 0:
        raise SystemExit("time limits must be positive")
    if min(args.pricing_label_limit, args.cg_max_iters, args.threads) < 1:
        raise SystemExit("label/iteration/thread limits must be positive")
    payload = execute(args)
    print(json.dumps({
        "trip_count": payload["trip_count"],
        "shared_union_pool_size": payload["shared_union_pool_size"],
        "result": str(args.out.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

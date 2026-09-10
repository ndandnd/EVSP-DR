#!/usr/bin/env python3
"""Matched exact-event pilot for charger inventory and PARX power.

The LP contains every documented finite-capacity station/minute row before
pricing starts.  Event pricing subtracts both trip and capacity duals and
stops only on the exact shortest-path reduced cost.  The resulting certificate
is for the conservative event/SOC graph encoded here, not a continuous-SOC or
nonlinear charging model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from audit_giro_known_columns import HORIZON_MIN, STATIONS, build_problem
from config import BIG_M_PENALTY, BUS_COST_KX
from event_pricer_network import (
    EventExpandedNetwork,
    conservative_capacity_rows,
)
from utils_v2 import base_station_name, load_station_hourly_prices


SCHEMA = "evsp-dr-capacity-speed-exact-event-pilot-v1"
MIP_SCHEMA = "evsp-dr-capacity-speed-two-stage-cover-mip-v2"
CHARGER_COUNTS = {
    "2190L": 1,
    "4808": 1,
    "3127L": 2,
    "7880C": 1,
    "JON_A": 1,
}
ARMS = {
    "baseline": {"capacity": False, "parx_kw": 240.0},
    "capacity": {"capacity": True, "parx_kw": 240.0},
    "parx60": {"capacity": False, "parx_kw": 60.0},
    "combined": {"capacity": True, "parx_kw": 60.0},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_pool(path: Path, routes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for route in routes:
            handle.write(json.dumps(route, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def route_key(route: dict) -> str:
    stops = route.get("expanded_grid_charging_stops") or {}
    return canonical_sha({
        "trips": route["trips"],
        "stations": stops.get("stations", []),
        "cst": stops.get("cst", []),
        "cet": stops.get("cet", []),
        "kwh": stops.get("kwh", []),
    })


def route_capacity_rows(route: dict) -> frozenset[tuple[str, int]]:
    stops = route.get("expanded_grid_charging_stops") or {}
    rows = set()
    for station, start, end in zip(
        stops.get("stations", []), stops.get("cst", []), stops.get("cet", []),
    ):
        rows.update(conservative_capacity_rows({
            "kind": "charge", "station": station, "cst": start, "cet": end,
        }, sites=CHARGER_COUNTS, grid_min=1))
    return frozenset(rows)


def station_power(arm: str) -> dict[str, float]:
    return {"PARX": float(ARMS[arm]["parx_kw"])}


def provenance(args, instance: Path, prices: Path, reference: Path) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    if args.expected_commit and commit != args.expected_commit:
        raise ValueError(f"commit mismatch: {commit} != {args.expected_commit}")
    if args.require_clean and dirty:
        raise ValueError("tracked worktree is dirty")
    return {
        "git_commit": commit,
        "git_tracked_dirty": dirty,
        "instance": str(instance),
        "instance_sha256": sha256_file(instance),
        "prices": str(prices),
        "prices_sha256": sha256_file(prices),
        "reference_data_dir": str(reference),
        "reference_sha256": sha256_file(reference / "Ref_dict.csv"),
        "deadhead_sha256": sha256_file(reference / "par_ref_dhd.csv"),
        "gurobi_version": ".".join(map(str, gp.gurobi.version())),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": os.uname().nodename,
    }


def build_network(args, problem, prices):
    return EventExpandedNetwork(
        problem,
        prices,
        soc_step=args.soc_step,
        block_min=args.block_min,
        g_kwh=args.battery_kwh,
        charge_kw=args.non_parx_kw,
        reserve_kwh=args.reserve_kwh,
        strict_tariff_coverage=False,
        arc_mode="explicit",
        station_charge_kw=station_power(args.arm),
    )


class ExactCapacityMaster:
    def __init__(self, trips, *, capacity: bool, threads: int, log_path=None):
        self.trips = tuple(trips)
        self.capacity = bool(capacity)
        self.model = gp.Model("capacity_speed_exact_event_cover_lp")
        self.model.Params.OutputFlag = 1 if log_path else 0
        self.model.Params.Threads = int(threads)
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.Params.LogFile = str(log_path)
        self.model.ModelSense = GRB.MINIMIZE
        self.artificial = {}
        self.trip_rows = {}
        for trip in self.trips:
            artificial = self.model.addVar(
                lb=0.0, obj=BIG_M_PENALTY, name=f"artificial_{trip}",
            )
            self.artificial[trip] = artificial
            self.trip_rows[trip] = self.model.addConstr(
                artificial >= 1.0, name=f"cover_{trip}",
            )
        self.capacity_rows = {}
        if self.capacity:
            for site, count in sorted(CHARGER_COUNTS.items()):
                for minute in range(int(HORIZON_MIN)):
                    self.capacity_rows[site, minute] = self.model.addConstr(
                        gp.LinExpr() <= count,
                        name=f"capacity_{site}_{minute}",
                    )
        self.variables = []
        self.routes = []
        self.model.update()

    def add_route(self, route):
        constraints = [self.trip_rows[trip] for trip in route["trips"]]
        coefficients = [1.0] * len(constraints)
        if self.capacity:
            for row in sorted(route_capacity_rows(route)):
                constraints.append(self.capacity_rows[row])
                coefficients.append(1.0)
        variable = self.model.addVar(
            lb=0.0,
            obj=float(route["cost"]),
            column=gp.Column(coefficients, constraints),
            name=f"route_{len(self.routes)}",
        )
        self.routes.append(route)
        self.variables.append(variable)
        self.model.update()

    def solve(self):
        started = time.perf_counter()
        self.model.optimize()
        runtime = time.perf_counter() - started
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"restricted LP status {self.model.Status}")
        return {
            "objective": float(self.model.ObjVal),
            "runtime_s": runtime,
            "rows": int(self.model.NumConstrs),
            "columns": int(self.model.NumVars),
            "nonzeros": int(self.model.NumNZs),
            "artificial_total": sum(value.X for value in self.artificial.values()),
            "route_weight": sum(value.X for value in self.variables),
            "trip_duals": {
                trip: float(row.Pi) for trip, row in self.trip_rows.items()
            },
            "capacity_duals": {
                key: float(row.Pi) for key, row in self.capacity_rows.items()
                if abs(float(row.Pi)) > 1e-10
            },
        }


def run_cg(args, problem, prices, prov, out: Path, pool_out: Path):
    if out.exists() or pool_out.exists():
        raise FileExistsError("refusing to overwrite CG output")
    started = time.perf_counter()
    network_started = time.perf_counter()
    network = build_network(args, problem, prices)
    network_build_s = time.perf_counter() - network_started
    log_path = out.with_suffix(out.suffix + ".gurobi.log")
    master = ExactCapacityMaster(
        problem.trips, capacity=ARMS[args.arm]["capacity"], threads=args.threads,
        log_path=log_path,
    )
    keys = set()
    for trip in problem.trips:
        route = network.fixed_sequence_record((trip,))
        if route is None:
            continue
        route.update({"origin": "exact_event_singleton", "found_iter": 0})
        key = route_key(route)
        if key not in keys:
            keys.add(key)
            master.add_route(route)
    iterations = []
    terminal_rc = None
    stop_reason = None
    for iteration in range(1, args.max_iters + 1):
        if time.perf_counter() - started >= args.cg_wall_s:
            stop_reason = "cg_wall_limit"
            break
        lp = master.solve()
        pricing_started = time.perf_counter()
        candidate = network.min_reduced_cost_route(
            lp["trip_duals"],
            capacity_duals=lp["capacity_duals"],
            capacity_sites=set(CHARGER_COUNTS) if ARMS[args.arm]["capacity"] else None,
            capacity_grid_min=1,
        )
        pricing_s = time.perf_counter() - pricing_started
        if candidate is None:
            stop_reason = "pricing_no_path"
            break
        terminal_rc = float(candidate["rc"])
        record = candidate["_event_record"]
        independent_rc = (
            float(record["cost"])
            - sum(lp["trip_duals"][trip] for trip in record["trips"])
            - sum(lp["capacity_duals"].get(row, 0.0)
                  for row in route_capacity_rows(record))
        )
        if not math.isclose(terminal_rc, independent_rc, abs_tol=1e-5):
            raise RuntimeError(
                f"pricing RC mismatch {terminal_rc} vs {independent_rc}"
            )
        iterations.append({
            "iteration": iteration,
            "lp_objective": lp["objective"],
            "lp_solve_s": lp["runtime_s"],
            "lp_rows": lp["rows"],
            "lp_columns": lp["columns"],
            "lp_nonzeros": lp["nonzeros"],
            "pricing_s": pricing_s,
            "artificial_total": lp["artificial_total"],
            "route_weight": lp["route_weight"],
            "nonzero_capacity_duals": len(lp["capacity_duals"]),
            "min_reduced_cost": terminal_rc,
            "priced_trip_count": len(record["trips"]),
            "pool_columns": len(master.routes),
        })
        if terminal_rc >= -args.rc_eps:
            stop_reason = "exact_nonnegative_reduced_cost"
            break
        record.update({"origin": "exact_capacity_dual_event_pricing", "found_iter": iteration})
        key = route_key(record)
        if key in keys:
            raise RuntimeError("negative exact priced route already exists in master")
        keys.add(key)
        master.add_route(record)
    final_lp = master.solve()
    certified = (
        stop_reason == "exact_nonnegative_reduced_cost"
        and final_lp["artificial_total"] <= 1e-7
    )
    atomic_pool(pool_out, master.routes)
    payload = {
        "schema": SCHEMA,
        "mode": "cg",
        "arm": args.arm,
        "status": "complete" if certified else "incomplete",
        "stop_reason": stop_reason,
        "certified_rc_optimal": certified,
        "pricing_certificate_scope": (
            "full conservative event-time/SOC covering LP with documented "
            "one-minute station-capacity rows"
            if ARMS[args.arm]["capacity"] else
            "full conservative event-time/SOC covering LP without shared capacity"
        ),
        "terminal_exact_min_reduced_cost": terminal_rc,
        "final": {
            "objective": final_lp["objective"],
            "route_weight": final_lp["route_weight"],
            "artificial_total": final_lp["artificial_total"],
            "pool_columns": len(master.routes),
        },
        "physics": {
            "battery_kwh": args.battery_kwh,
            "initial_soc_kwh": args.battery_kwh,
            "reserve_kwh": args.reserve_kwh,
            "terminal_soc_constraint": "reserve_only",
            "soc_step_kwh": args.soc_step,
            "event_block_min": args.block_min,
            "non_parx_kw": args.non_parx_kw,
            "parx_kw": ARMS[args.arm]["parx_kw"],
            "capacity_enforced": ARMS[args.arm]["capacity"],
            "charger_counts": CHARGER_COUNTS if ARMS[args.arm]["capacity"] else {},
            "parx_capacity": "unlimited",
        },
        "network": network.metrics(),
        "network_build_s": network_build_s,
        "gurobi_log": str(log_path),
        "iterations": iterations,
        "runtime_s": time.perf_counter() - started,
        "pool": str(pool_out),
        "pool_sha256": sha256_file(pool_out),
        "provenance": prov,
    }
    atomic_json(out, payload)
    return payload


def load_pool(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def physical_capacity_audit(routes, selected):
    events = {site: [] for site in CHARGER_COUNTS}
    for index in selected:
        stops = routes[index].get("expanded_grid_charging_stops") or {}
        for station, start, end in zip(
            stops.get("stations", []), stops.get("cst", []), stops.get("cet", []),
        ):
            site = base_station_name(station)
            if site in events:
                events[site].append((float(start), 1, index))
                events[site].append((float(end), -1, index))
    detail = {}
    valid = True
    for site, rows in events.items():
        active = 0
        peak = 0
        peak_time = None
        for minute, delta, _route in sorted(rows, key=lambda row: (row[0], row[1])):
            active += delta
            if active > peak:
                peak, peak_time = active, minute
        count = CHARGER_COUNTS[site]
        detail[site] = {
            "documented_chargers": count,
            "peak_simultaneous_connections": peak,
            "peak_time_min": peak_time,
            "valid": peak <= count,
        }
        valid = valid and peak <= count
    return {"valid": valid, "half_open_intervals": True, "stations": detail}


def duplicate_service_audit(routes, selected, trips):
    counts = {trip: 0 for trip in trips}
    for index in selected:
        for trip in routes[index]["trips"]:
            counts[trip] += 1
    return {
        "all_trips_covered": all(count >= 1 for count in counts.values()),
        "overcovered_trip_count": sum(count > 1 for count in counts.values()),
        "extra_trip_assignments": sum(max(0, count - 1) for count in counts.values()),
        "overcovered_trips": {
            str(trip): count for trip, count in counts.items() if count > 1
        },
    }


def classify_saved_pool(cg_status, routes, trips):
    """Classify a hash-bound saved pool and require usable trip coverage."""

    if not routes:
        raise ValueError("no usable saved pool: pool contains no routes")
    expected = set(trips)
    covered = {
        trip for route in routes for trip in route.get("trips", [])
        if trip in expected
    }
    missing = sorted(expected - covered)
    if missing:
        raise ValueError(
            "no usable saved pool: missing trip coverage for "
            f"{len(missing)} trip(s), sample={missing[:10]}"
        )
    certified = cg_status.get("certified_rc_optimal") is True
    return {
        "usable": True,
        "cg_pricing_certified": certified,
        "classification": (
            "certified_exact_event_cg_pool"
            if certified else "timed_uncertified_exact_event_cg_pool"
        ),
        "cg_status": cg_status.get("status"),
        "cg_stop_reason": cg_status.get("stop_reason"),
        "route_count": len(routes),
        "covered_trip_count": len(covered),
    }


def fleet_cap_from_stage1(stage1):
    """Return the authorized stage-two cap from any feasible stage-one incumbent."""

    if not stage1.get("has_solution"):
        raise ValueError("no usable stage-one fleet incumbent")
    return {
        "sense": "<=",
        "rhs": int(stage1["incumbent_fleet"]),
        "source": "best_validated_stage1_incumbent",
        "source_fleet_proven": bool(stage1["fleet_proven"]),
    }


def add_fleet_incumbent_cap(model, fleet_expression, stage1):
    """Add the nonbinding-direction cap that allows a smaller stage-two fleet."""

    cap = fleet_cap_from_stage1(stage1)
    constraint = model.addConstr(
        fleet_expression <= cap["rhs"], name="stage2_fleet_incumbent_cap",
    )
    return constraint, cap


def _status_name(status):
    return {
        GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE", GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }.get(status, f"STATUS_{status}")


def solve_mip(args, problem, routes, capacity, log_path):
    model = gp.Model("capacity_speed_exact_event_cover_mip")
    model.Params.OutputFlag = 1
    model.Params.LogFile = str(log_path)
    model.Params.Threads = int(args.threads)
    model.Params.MIPGap = float(args.mip_gap)
    x = model.addVars(len(routes), vtype=GRB.BINARY, name="route")
    for trip in problem.trips:
        model.addConstr(
            gp.quicksum(x[index] for index, route in enumerate(routes)
                        if trip in route["trips"]) >= 1.0,
            name=f"cover_{trip}",
        )
    if capacity:
        memberships = [route_capacity_rows(route) for route in routes]
        for site, count in sorted(CHARGER_COUNTS.items()):
            for minute in range(int(HORIZON_MIN)):
                row = (site, minute)
                members = [x[index] for index, rows in enumerate(memberships) if row in rows]
                if members:
                    model.addConstr(gp.quicksum(members) <= count,
                                    name=f"capacity_{site}_{minute}")
    fleet_expression = gp.quicksum(x.values())
    charging_expression = gp.quicksum(
        (float(route["cost"]) - BUS_COST_KX) * x[index]
        for index, route in enumerate(routes)
    )
    total_budget_s = float(args.mip_wall_s)
    stage1_budget_s = total_budget_s / 2.0
    model.Params.TimeLimit = stage1_budget_s
    model.setObjective(fleet_expression, GRB.MINIMIZE)
    total_started = time.perf_counter()
    stage1_started = time.perf_counter()
    model.optimize()
    stage1_runtime_s = time.perf_counter() - stage1_started
    stage1_has_solution = model.SolCount > 0
    stage1_selected = [
        index for index in range(len(routes))
        if stage1_has_solution and x[index].X > 0.5
    ]
    stage1_incumbent = len(stage1_selected) if stage1_has_solution else None
    stage1_bound_raw = float(model.ObjBound)
    stage1_integer_bound = (
        math.ceil(stage1_bound_raw - 1e-7)
        if math.isfinite(stage1_bound_raw) else None
    )
    fleet_proven = bool(
        stage1_has_solution and stage1_integer_bound is not None
        and stage1_integer_bound >= stage1_incumbent
    )
    stage1_audit = (
        physical_capacity_audit(routes, stage1_selected)
        if stage1_has_solution else None
    )
    stage1_cover = (
        duplicate_service_audit(routes, stage1_selected, problem.trips)
        if stage1_has_solution else None
    )
    stage1_validated = bool(
        stage1_has_solution
        and stage1_cover["all_trips_covered"]
        and (not capacity or stage1_audit["valid"])
    )
    if stage1_has_solution and not stage1_validated:
        raise RuntimeError("stage-one solver incumbent failed physical validation")
    stage1 = {
        "status_code": int(model.Status),
        "status": _status_name(model.Status),
        "has_solution": stage1_has_solution,
        "validated_incumbent": stage1_validated,
        "incumbent_fleet": stage1_incumbent,
        "fleet_bound_raw": stage1_bound_raw,
        "fleet_integer_lower_bound": stage1_integer_bound,
        "fleet_proven": fleet_proven,
        "mip_gap": float(model.MIPGap) if stage1_has_solution else None,
        "node_count": float(model.NodeCount),
        "runtime_s": stage1_runtime_s,
        "time_limit_s": stage1_budget_s,
        "selected_indices": stage1_selected,
    }
    remaining_s = max(0.0, total_budget_s - (time.perf_counter() - total_started))
    stage2 = {
        "executed": False,
        "skip_reason": None,
        "time_limit_s": remaining_s,
        "has_solution": False,
    }
    selected = stage1_selected
    if not stage1_has_solution:
        stage2["skip_reason"] = "no_usable_stage1_fleet_incumbent"
    elif remaining_s <= 1e-3:
        stage2["skip_reason"] = "no_remaining_mip_budget"
    else:
        _constraint, cap = add_fleet_incumbent_cap(
            model, fleet_expression, stage1,
        )
        for index in range(len(routes)):
            x[index].Start = 1.0 if index in stage1_selected else 0.0
        model.setObjective(charging_expression, GRB.MINIMIZE)
        model.Params.TimeLimit = remaining_s
        stage2_started = time.perf_counter()
        model.optimize()
        stage2_runtime_s = time.perf_counter() - stage2_started
        stage2_has_solution = model.SolCount > 0
        stage2_selected = [
            index for index in range(len(routes))
            if stage2_has_solution and x[index].X > 0.5
        ]
        if stage2_has_solution:
            selected = stage2_selected
        stage2 = {
            "executed": True,
            "skip_reason": None,
            "status_code": int(model.Status),
            "status": _status_name(model.Status),
            "has_solution": stage2_has_solution,
            "fleet_cap": cap,
            "incumbent_fleet": (
                len(stage2_selected) if stage2_has_solution else None
            ),
            "charging_cost": (
                float(model.ObjVal) if stage2_has_solution else None
            ),
            "charging_cost_bound": float(model.ObjBound),
            "charging_cost_gap": (
                float(model.MIPGap) if stage2_has_solution else None
            ),
            "node_count": float(model.NodeCount),
            "runtime_s": stage2_runtime_s,
            "time_limit_s": remaining_s,
            "selected_indices": stage2_selected,
        }
    has_solution = bool(selected)
    charging_cost = sum(
        float(routes[index]["cost"]) - BUS_COST_KX for index in selected
    ) if has_solution else None
    return {
        "method": "two_stage_fleet_then_charging_cost",
        "status": (
            stage2.get("status") if stage2["executed"] else stage1["status"]
        ),
        "has_solution": has_solution,
        "selected_indices": selected,
        "fleet": len(selected) if has_solution else None,
        "charging_related_cost": charging_cost,
        "reconstructed_weighted_cost": (
            BUS_COST_KX * len(selected) + charging_cost
            if has_solution else None
        ),
        "total_budget_s": total_budget_s,
        "stage1": stage1,
        "stage2": stage2,
    }


def run_mip(args, problem, prov, out: Path, pool: Path, cg_status: Path):
    if out.exists():
        raise FileExistsError(out)
    status = json.loads(cg_status.read_text())
    if status.get("arm") != args.arm:
        raise ValueError("CG arm mismatch")
    if status.get("pool_sha256") != sha256_file(pool):
        raise ValueError("CG pool hash mismatch")
    routes = load_pool(pool)
    pool_acceptance = classify_saved_pool(status, routes, problem.trips)
    log_path = out.with_suffix(out.suffix + ".gurobi.log")
    result = solve_mip(
        args, problem, routes, ARMS[args.arm]["capacity"], log_path,
    )
    audit = physical_capacity_audit(routes, result["selected_indices"])
    duplicate_audit = duplicate_service_audit(
        routes, result["selected_indices"], problem.trips,
    )
    if ARMS[args.arm]["capacity"] and result["has_solution"] and not audit["valid"]:
        raise RuntimeError("selected MIP solution fails continuous capacity audit")
    payload = {
        "schema": MIP_SCHEMA,
        "mode": "mip",
        "arm": args.arm,
        "cg_status": str(cg_status),
        "cg_status_sha256": sha256_file(cg_status),
        "pool": str(pool),
        "pool_sha256": sha256_file(pool),
        "pool_acceptance": pool_acceptance,
        "result": result,
        "physical_station_capacity_audit": audit,
        "duplicate_service_audit": duplicate_audit,
        "capacity_enforced_in_mip": ARMS[args.arm]["capacity"],
        "provenance": prov,
    }
    atomic_json(out, payload)
    return payload


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--mode", choices=("cg", "mip"), required=True)
    value.add_argument("--arm", choices=tuple(ARMS), required=True)
    value.add_argument("--instance", type=Path, required=True)
    value.add_argument("--prices", type=Path, required=True)
    value.add_argument("--reference-data-dir", type=Path, required=True)
    value.add_argument("--out", type=Path, required=True)
    value.add_argument("--pool-out", type=Path)
    value.add_argument("--pool", type=Path)
    value.add_argument("--cg-status", type=Path)
    value.add_argument("--battery-kwh", type=float, default=240.0)
    value.add_argument("--non-parx-kw", type=float, default=240.0)
    value.add_argument("--reserve-kwh", type=float, default=0.0)
    value.add_argument("--soc-step", type=float, default=2.5)
    value.add_argument("--block-min", type=int, default=5)
    value.add_argument("--rc-eps", type=float, default=1e-5)
    value.add_argument("--max-iters", type=int, default=10000)
    value.add_argument("--cg-wall-s", type=float, default=3600.0)
    value.add_argument("--mip-wall-s", type=float, default=1800.0)
    value.add_argument("--mip-gap", type=float, default=1e-4)
    value.add_argument("--threads", type=int, default=1)
    value.add_argument("--expected-commit")
    value.add_argument("--require-clean", action="store_true")
    return value


def main():
    args = parser().parse_args()
    instance = args.instance.expanduser().resolve(strict=True)
    prices_path = args.prices.expanduser().resolve(strict=True)
    reference = args.reference_data_dir.expanduser().resolve(strict=True)
    out = args.out.expanduser().resolve()
    problem = build_problem(
        instance.parent, instance.name, reference_data_dir=reference,
    )
    prices = load_station_hourly_prices(
        prices_path, sorted({base_station_name(station) for station in STATIONS}),
    )
    prov = provenance(args, instance, prices_path, reference)
    if args.mode == "cg":
        if args.pool_out is None:
            raise ValueError("--pool-out is required for CG")
        payload = run_cg(
            args, problem, prices, prov, out, args.pool_out.expanduser().resolve(),
        )
    else:
        if args.pool is None or args.cg_status is None:
            raise ValueError("--pool and --cg-status are required for MIP")
        payload = run_mip(
            args, problem, prov, out,
            args.pool.expanduser().resolve(strict=True),
            args.cg_status.expanduser().resolve(strict=True),
        )
    print(json.dumps({
        "out": str(out), "mode": args.mode, "arm": args.arm,
        "status": payload.get("status", payload.get("result", {}).get("status")),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Matched terminal-energy pilot on the five-duty, 62-trip cohort.

The fixed-duty arm enumerates conservative terminal-SOC frontiers on the event
graph.  The joint arm reoptimizes an immutable saved CG pool augmented with
those frontier routes.  Both masters enforce the same aggregate return energy;
no duty-specific terminal target is attached to a joint route.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TARGET_KWH = 280.7833253
FLEET_CAP = 5
BUS_COST = 100000.0
TOL = 1e-7


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, sort_keys=True, indent=2,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True,
    ).strip()


def check_identity(commit):
    if git("rev-parse", "HEAD") != commit:
        raise ValueError("execution commit mismatch")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked execution checkout is dirty")
    detached = subprocess.run(
        ["git", "-C", str(ROOT), "symbolic-ref", "-q", "HEAD"],
        capture_output=True,
    )
    if detached.returncode != 1:
        raise ValueError("clean detached execution checkout required")


def terminal(route, kind="grid"):
    key = (
        "expanded_grid_terminal_soc_kwh"
        if kind == "grid" else "continuous_terminal_soc_kwh"
    )
    value = float((route.get("continuous_realization") or {})[key])
    if not math.isfinite(value) or value < -TOL:
        raise ValueError(f"invalid {kind} terminal energy")
    return value


def variable_cost(route, kind="grid"):
    value = float(
        route["cost"] if kind == "grid"
        else route["continuous_realized_cost"]
    ) - BUS_COST
    if not math.isfinite(value):
        raise ValueError("non-finite route cost")
    return value


def route_signature(route):
    return canonical({key: route[key] for key in (
        "trips", "route_nodes", "charging_stops",
        "expanded_grid_charging_stops", "cost",
    )})


def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize_routes(routes):
    coverage = Counter(trip for route in routes for trip in route["trips"])
    return {
        "fleet": len(routes),
        "expanded_grid_charging_cost": sum(
            variable_cost(route, "grid") for route in routes
        ),
        "physical_charging_cost": sum(
            variable_cost(route, "continuous") for route in routes
        ),
        "expanded_grid_terminal_energy_kwh": sum(
            terminal(route, "grid") for route in routes
        ),
        "continuous_terminal_energy_kwh": sum(
            terminal(route, "continuous") for route in routes
        ),
        "overcovered_trip_count": sum(value > 1 for value in coverage.values()),
        "maximum_trip_multiplicity": max(coverage.values(), default=0),
        "selected_routes": routes,
    }


def validate_saved_status(status, cell, unique_trip_sets):
    provenance = status.get("provenance") or {}
    expected = {
        "time_model": "event", "g_kwh": 240.0, "charge_kw": 350.0,
        "soc_step": 2.5, "block_min": 5, "min_soc_frac": 0.0,
        "strict_tariff_coverage": True,
    }
    if any(status.get(key) != value for key, value in expected.items()):
        raise ValueError("saved CG physics/configuration mismatch")
    if (
        provenance.get("instance_sha256") != cell["instance_sha256"]
        or provenance.get("prices_sha256") != cell["tariff_sha256"]
        or int(status.get("columns", -1)) != unique_trip_sets
    ):
        raise ValueError("saved CG identity or column count mismatch")


def replay_pool_routes(problem, routes, prices):
    """Recompute every terminal coefficient and grid cost used by the MIP."""
    from audit_giro_known_columns import HORIZON_MIN
    from expanded_path_realization import realize_expanded_path, realized_costs
    from run_exact_pool_mip import validate_injected_route

    validated = []
    for ordinal, route in enumerate(routes, start=1):
        realized, detail = realize_expanded_path(
            problem, route, g_kwh=240.0, charge_kw=350.0,
            reserve_kwh=0.0, soc_step=2.5, block_min=5,
            time_model="event",
        )
        if realized is None:
            raise ValueError(
                f"pool route {ordinal} cannot be replayed: {detail.get('reason')}"
            )
        reason = validate_injected_route(
            problem, realized, 240.0, 350.0, 0.0, HORIZON_MIN,
            arrival_grace_min=0.0,
        )
        if reason is not None:
            raise ValueError(f"pool route {ordinal} is invalid: {reason}")
        costs = realized_costs(
            realized, detail["mapping"], station_prices=prices,
        )
        checks = (
            (float(route["cost"]),
             float(costs["recomputed_expanded_grid_cost"]), "grid cost"),
            (terminal(route, "grid"),
             float(detail["mapping"]["expanded_grid_terminal_soc_kwh"]),
             "grid terminal energy"),
            (terminal(route, "continuous"),
             float(detail["mapping"]["continuous_terminal_soc_kwh"]),
             "continuous terminal energy"),
        )
        for recorded, recomputed, label in checks:
            if not math.isclose(
                recorded, recomputed, abs_tol=1e-6, rel_tol=1e-10,
            ):
                raise ValueError(
                    f"pool route {ordinal} {label} metadata mismatch"
                )
        if terminal(route, "continuous") + TOL < terminal(route, "grid"):
            raise ValueError(
                f"pool route {ordinal} grid terminal energy is not conservative"
            )
        validated.append(route)
    return validated


def gurobi_model(name, log_path, threads):
    import gurobipy as gp
    model = gp.Model(name)
    model.Params.OutputFlag = 1
    model.Params.LogFile = str(log_path)
    model.Params.Threads = threads
    model.Params.MIPGap = 1e-4
    return gp, model


def solve_fixed(frontiers, target_kwh, log_path):
    gp, model = gurobi_model("fixed_terminal_frontier", log_path, 1)
    model.Params.MIPGap = 0.0
    options = [route for duty in frontiers for route in duty["routes"]]
    variables = model.addVars(len(options), vtype=gp.GRB.BINARY, name="route")
    offset = 0
    for duty in frontiers:
        indices = range(offset, offset + len(duty["routes"]))
        model.addConstr(gp.quicksum(variables[i] for i in indices) == 1,
                        name=f"duty_{duty['duty_id']}")
        offset += len(duty["routes"])
    model.addConstr(gp.quicksum(
        terminal(route, "grid") * variables[i]
        for i, route in enumerate(options)
    ) >= target_kwh, name="aggregate_terminal_energy")
    model.setObjective(gp.quicksum(
        variable_cost(route, "grid") * variables[i]
        for i, route in enumerate(options)
    ), gp.GRB.MINIMIZE)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"fixed frontier master status {model.Status}")
    chosen = [options[i] for i in range(len(options)) if variables[i].X > .5]
    if len(chosen) != FLEET_CAP:
        raise RuntimeError("fixed frontier master did not select five duties")
    return chosen, {
        "status": int(model.Status), "objective": float(model.ObjVal),
        "bound": finite(model.ObjBound), "gap": finite(model.MIPGap),
        "rows": int(model.NumConstrs), "columns": int(model.NumVars),
        "nonzeros": int(model.NumNZs),
    }


def pareto_pool(routes):
    by_trips = {}
    for route in routes:
        trips = tuple(sorted(route["trips"]))
        if not trips or len(trips) != len(set(trips)):
            raise ValueError("pool route has empty or repeated trips")
        terminal(route, "grid")
        variable_cost(route, "grid")
        by_trips.setdefault(trips, {})[route_signature(route)] = route
    retained = []
    for group in by_trips.values():
        values = list(group.values())
        for candidate in values:
            dominated = any(
                other is not candidate
                and variable_cost(other, "grid")
                    <= variable_cost(candidate, "grid") + TOL
                and terminal(other, "grid")
                    >= terminal(candidate, "grid") - TOL
                and (
                    variable_cost(other, "grid")
                        < variable_cost(candidate, "grid") - TOL
                    or terminal(other, "grid")
                        > terminal(candidate, "grid") + TOL
                )
                for other in values
            )
            if not dominated:
                retained.append(candidate)
    return retained


def preserve_start_routes(routes, fixed_start):
    """Keep the exact validated start, even when another route dominates it."""
    result = list(routes)
    signatures = {route_signature(route) for route in result}
    for route in fixed_start:
        signature = route_signature(route)
        if signature not in signatures:
            result.append(route)
            signatures.add(signature)
    return result


def solve_joint(routes, trips, fixed_start, target_kwh, log_path,
                total_seconds=3600, stage1_seconds=1800):
    gp, model = gurobi_model("joint_terminal_pool", log_path, 8)
    variables = model.addVars(len(routes), vtype=gp.GRB.BINARY, name="route")
    for trip in trips:
        indices = [i for i, route in enumerate(routes) if trip in route["trips"]]
        if not indices:
            raise RuntimeError(f"trip {trip} has no covering route")
        model.addConstr(gp.quicksum(variables[i] for i in indices) >= 1,
                        name=f"cover_{trip}")
    terminal_row = model.addConstr(gp.quicksum(
        terminal(route, "grid") * variables[i]
        for i, route in enumerate(routes)
    ) >= target_kwh, name="aggregate_terminal_energy")
    model.addConstr(gp.quicksum(variables.values()) <= FLEET_CAP,
                    name="seed_fleet_cap")
    index_by_signature = {
        route_signature(route): index for index, route in enumerate(routes)
    }
    start_audit = summarize_routes(fixed_start)
    start_coverage = Counter(
        trip for route in fixed_start for trip in route["trips"]
    )
    if (
        start_audit["fleet"] != FLEET_CAP
        or start_audit["expanded_grid_terminal_energy_kwh"] + TOL
            < target_kwh
        or any(start_coverage[trip] < 1 for trip in trips)
    ):
        raise ValueError("fixed-duty MIP start is not cover/energy/fleet feasible")
    start_indices = [index_by_signature[route_signature(route)]
                     for route in fixed_start]
    for i in range(len(routes)):
        variables[i].Start = 1.0 if i in start_indices else 0.0
    model.setObjective(gp.quicksum(variables.values()), gp.GRB.MINIMIZE)
    model.Params.TimeLimit = stage1_seconds
    started = time.monotonic()
    model.optimize()
    elapsed = time.monotonic() - started
    if model.SolCount == 0:
        raise RuntimeError(f"stage 1 has no incumbent; status {model.Status}")
    stage1_buses = int(round(model.ObjVal))
    stage1_chosen = [i for i in range(len(routes)) if variables[i].X > .5]
    stage1_routes = [routes[i] for i in stage1_chosen]
    stage1_audit = summarize_routes(stage1_routes)
    stage1_coverage = Counter(
        trip for route in stage1_routes for trip in route["trips"]
    )
    if (
        stage1_audit["fleet"] != stage1_buses
        or stage1_buses > FLEET_CAP
        or any(stage1_coverage[trip] < 1 for trip in trips)
        or stage1_audit["expanded_grid_terminal_energy_kwh"] + TOL
            < target_kwh
        or stage1_audit["continuous_terminal_energy_kwh"] + TOL
            < target_kwh
    ):
        raise RuntimeError("stage 1 incumbent failed fleet/cover/energy audit")
    stage1 = {
        "status": int(model.Status), "buses": stage1_buses,
        "bound": finite(model.ObjBound), "gap": finite(model.MIPGap),
        "runtime_s": elapsed, "proven": (
            model.Status == gp.GRB.OPTIMAL
            or (finite(model.ObjBound) is not None
                and math.ceil(float(model.ObjBound) - TOL) >= stage1_buses)
        ), "incumbent_validated": True,
    }
    model.addConstr(gp.quicksum(variables.values()) <= stage1_buses,
                    name="stage2_fleet_at_most_incumbent")
    model.setObjective(gp.quicksum(
        variable_cost(route, "grid") * variables[i]
        for i, route in enumerate(routes)
    ), gp.GRB.MINIMIZE)
    for i in range(len(routes)):
        variables[i].Start = 1.0 if i in stage1_chosen else 0.0
    remaining = max(1.0, total_seconds - elapsed)
    model.Params.TimeLimit = remaining
    stage2_started = time.monotonic()
    model.optimize()
    stage2_elapsed = time.monotonic() - stage2_started
    if model.SolCount == 0:
        raise RuntimeError(f"stage 2 has no incumbent; status {model.Status}")
    selected = [routes[i] for i in range(len(routes)) if variables[i].X > .5]
    audit = summarize_routes(selected)
    if audit["expanded_grid_terminal_energy_kwh"] + TOL < target_kwh:
        raise RuntimeError("selected routes violate terminal energy")
    if audit["fleet"] > stage1_buses or audit["fleet"] > FLEET_CAP:
        raise RuntimeError("selected routes violate fleet cap")
    selected_coverage = Counter(
        trip for route in selected for trip in route["trips"]
    )
    if any(selected_coverage[trip] < 1 for trip in trips):
        raise RuntimeError("selected routes do not cover every trip")
    if audit["continuous_terminal_energy_kwh"] + TOL < target_kwh:
        raise RuntimeError("continuous replay violates terminal energy")
    detail = {
        "stage1": stage1,
        "stage2": {
            "status": int(model.Status), "objective": float(model.ObjVal),
            "bound": finite(model.ObjBound), "gap": finite(model.MIPGap),
            "runtime_s": stage2_elapsed,
            "fleet_cap_relation": "selected_fleet <= best_stage1_incumbent",
        },
        "rows": int(model.NumConstrs), "columns": int(model.NumVars),
        "nonzeros": int(model.NumNZs),
        "aggregate_terminal_constraint_slack_kwh": float(terminal_row.Slack),
    }
    return selected, detail


def parse_sources(values):
    result = {}
    for value in values:
        tariff, separator, root = value.partition("=")
        if not separator or tariff in result:
            raise ValueError("sources must be unique TARIFF=ROOT pairs")
        result[tariff] = Path(root).expanduser().resolve()
    if set(result) != {"peak08", "peak12", "peak18"}:
        raise ValueError("peak08, peak12, and peak18 sources are required")
    return result


def prepare(args):
    check_identity(args.commit)
    if args.root.exists():
        raise FileExistsError(args.root)
    sources = parse_sources(args.source)
    instance = ROOT / ("data/scale_ladder/instances/"
        "original_replay_eligible_20260908/"
        "Practice_Custom_DutyUnion_original_eligible_k05_20260908.csv")
    cells = []
    for index, tariff in enumerate(("peak08", "peak12", "peak18")):
        source = sources[tariff]
        paths = {
            name: source / name for name in (
                "snapshot.json", "snapshot.json.columns.jsonl", "original.json",
            )
        }
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"incomplete saved source for {tariff}")
        original = json.loads(paths["original.json"].read_text())
        observed = float(original["summary"][0]["terminal_surplus_total_kwh"])
        if not math.isclose(observed, TARGET_KWH, abs_tol=1e-7, rel_tol=0):
            raise ValueError(f"unexpected GIRO terminal energy for {tariff}")
        tariff_path = ROOT / f"data/tariff_response/{tariff}_h26.csv"
        cells.append({
            "index": index, "tariff": tariff,
            "instance": str(instance), "instance_sha256": sha(instance),
            "tariff_path": str(tariff_path),
            "tariff_sha256": sha(tariff_path),
            "source_root": str(source),
            "source_hashes": {name: sha(path) for name, path in paths.items()},
        })
    plan = {
        "schema": "evsp-dr-terminal-energy-fair-pilot-v1",
        "commit": args.commit, "cells": cells,
        "target_physical_terminal_energy_kwh": TARGET_KWH,
        "terminal_row_coefficient": "expanded_grid_terminal_soc_kwh",
        "terminal_coefficient_note": (
            "The exact RHS is 280.7833253 kWh. Terminal coefficients retain "
            "post-deadhead residuals and therefore need not be 2.5-kWh multiples."
        ),
        "per_bus_terminal_floor_kwh": 0.0,
        "fleet_cap": FLEET_CAP, "initial_energy_per_bus_kwh": 240.0,
        "charge_power_kw": 350.0, "soc_step_kwh": 2.5,
        "block_minutes": 5,
        "master_sense": "cover",
        "joint_scope": "saved_finite_CG_pool_plus_fixed_duty_frontiers",
        "original_giro_terminal_values_are_not_mapped_to_joint_routes": True,
        "stage_resources": {
            "frontier": {"partition": "default_partition", "cpus": 1,
                         "memory": "24G", "exclude": ["scaglione-compute-01"]},
            "mip": {"partition": "scaglione", "cpus": 8, "memory": "48G",
                    "exclude": ["scaglione-compute-01", "scaglione-cpu-04"]},
        },
    }
    args.root.mkdir(parents=True)
    (args.root / "logs").mkdir()
    atomic_json(args.root / "plan.json", plan)
    print(json.dumps({"root": str(args.root), "cells": len(cells),
                      "plan_sha256": sha(args.root / "plan.json")}))


def load_plan(args):
    path = args.root / "plan.json"
    if sha(path) != os.environ["EVSP_PLAN_SHA256"]:
        raise ValueError("plan hash mismatch")
    plan = json.loads(path.read_text())
    check_identity(plan["commit"])
    cell = plan["cells"][args.index]
    for key, digest_key in (("instance", "instance_sha256"),
                            ("tariff_path", "tariff_sha256")):
        if sha(cell[key]) != cell[digest_key]:
            raise ValueError(f"{key} hash mismatch")
    for name, digest in cell["source_hashes"].items():
        if sha(Path(cell["source_root"]) / name) != digest:
            raise ValueError(f"saved source changed: {name}")
    return plan, cell


def write_allocation(folder, stage):
    atomic_json(folder / f"{stage}.allocation.json", {
        "hostname": platform.node(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "stage": stage,
    })


def frontier_worker(args, plan, cell, folder):
    from audit_giro_known_columns import HORIZON_MIN, build_problem
    from compare_original_giro_charging import extract_original, read_rows
    from config import CHARGING_STATIONS
    from event_pricer_network import EventExpandedNetwork
    from utils_v2 import load_station_hourly_prices

    source_routes = extract_original(
        read_rows(ROOT / "data/Par_VehicleDetails_Updated.csv"),
        read_rows(cell["instance"]),
    )
    if len(source_routes) != FLEET_CAP:
        raise ValueError("source duty count differs from fleet cap")
    problem = build_problem(
        Path(cell["instance"]).parent, Path(cell["instance"]).name,
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=ROOT / "data",
    )
    prices = load_station_hourly_prices(cell["tariff_path"], CHARGING_STATIONS)
    started = time.monotonic()
    network = EventExpandedNetwork(
        problem, prices, soc_step=plan["soc_step_kwh"],
        block_min=plan["block_minutes"], g_kwh=240.0, charge_kw=350.0,
        reserve_kwh=0.0, strict_tariff_coverage=True, arc_mode="lazy",
    )
    build_s = time.monotonic() - started
    frontiers = []
    for source in source_routes:
        distinct = {}
        for step in range(int(round(240.0 / plan["soc_step_kwh"])) + 1):
            threshold = step * plan["soc_step_kwh"]
            route = network.fixed_sequence_record(
                source["trips"], min_terminal_soc_kwh=threshold,
            )
            if route is None:
                continue
            route.update({
                "duty_id": source["duty_id"],
                "source_ordered_trip_ids": source["source_ordered_trip_ids"],
                "origin": "terminal_soc_frontier_fixed_duty",
                "frontier_threshold_kwh": threshold,
                "cost_tariff_sha256": cell["tariff_sha256"],
            })
            distinct[route_signature(route)] = route
        routes = pareto_pool(list(distinct.values()))
        if not routes:
            raise RuntimeError(f"empty terminal frontier for duty {source['duty_id']}")
        frontiers.append({"duty_id": source["duty_id"], "routes": routes})
    fixed_routes, master = solve_fixed(
        frontiers, plan["target_physical_terminal_energy_kwh"],
        folder / "fixed.gurobi.log",
    )
    result = {
        "schema": "evsp-dr-terminal-energy-frontiers-v1",
        "cell": cell, "network_build_s": build_s,
        "network_metrics": network.metrics(), "frontiers": frontiers,
        "fixed_master": master, "fixed_solution": summarize_routes(fixed_routes),
        "certificate_scope": (
            "Each frontier point is the cheapest event-graph route for its "
            "fixed ordered trip sequence and terminal threshold. The group "
            "master is solved to zero reported MIP gap over the enumerated "
            "2.5-kWh threshold frontier."
        ),
    }
    atomic_json(folder / "frontier.json", result)
    atomic_json(folder / "FRONTIER_COMPLETE.json", {
        "frontier_sha256": sha(folder / "frontier.json"),
    })


def mip_worker(args, plan, cell, folder):
    from audit_giro_known_columns import HORIZON_MIN, build_problem
    from config import CHARGING_STATIONS
    from utils_v2 import load_station_hourly_prices

    frontier_path = folder / "frontier.json"
    marker = json.loads((folder / "FRONTIER_COMPLETE.json").read_text())
    if sha(frontier_path) != marker["frontier_sha256"]:
        raise ValueError("frontier completion hash mismatch")
    frontier = json.loads(frontier_path.read_text())
    fixed_routes = frontier["fixed_solution"]["selected_routes"]
    source_journal = Path(cell["source_root"]) / "snapshot.json.columns.jsonl"
    routes = []
    with source_journal.open() as handle:
        for line in handle:
            if not line.endswith("\n"):
                raise ValueError("incomplete saved-pool journal")
            routes.append(json.loads(line))
    source_records = len(routes)
    status = json.loads(
        (Path(cell["source_root"]) / "snapshot.json").read_text()
    )
    validate_saved_status(
        status, cell, len({frozenset(route["trips"]) for route in routes}),
    )
    routes.extend(
        route for duty in frontier["frontiers"] for route in duty["routes"]
    )
    problem = build_problem(
        Path(cell["instance"]).parent, Path(cell["instance"]).name,
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=ROOT / "data",
    )
    prices = load_station_hourly_prices(cell["tariff_path"], CHARGING_STATIONS)
    replay_started = time.monotonic()
    routes = replay_pool_routes(problem, routes, prices)
    replay_s = time.monotonic() - replay_started
    routes = pareto_pool(routes)
    routes = preserve_start_routes(routes, fixed_routes)
    import pandas as pd
    trips = [int(value) for value in
             pd.read_csv(cell["instance"])["count_trip_id"].tolist()]
    selected, solver = solve_joint(
        routes, trips, fixed_routes,
        plan["target_physical_terminal_energy_kwh"],
        folder / "mip.gurobi.log",
    )
    joint = summarize_routes(selected)
    fixed = summarize_routes(fixed_routes)
    original = json.loads(
        (Path(cell["source_root"]) / "original.json").read_text()
    )
    comparison = {
        "schema": "evsp-dr-terminal-energy-fair-comparison-v1",
        "cell": cell, "target_terminal_energy_kwh": TARGET_KWH,
        "terminal_constraint_semantics": (
            "One aggregate expanded-grid terminal-energy row shared by fixed "
            "and joint optimization; no heterogeneous original-duty targets "
            "are assigned to joint routes."
        ),
        "original_giro": original["summary"][0],
        "original_giro_per_duty_terminal_kwh": [
            {"duty_id": route["duty_id"],
             "terminal_soc_kwh": route["production_model_check"]["terminal_soc_kwh"]}
            for route in original["routes"]
        ],
        "fixed_duties_optimized": fixed, "joint_pool_optimized": joint,
        "joint_solver": solver,
        "saved_pool_records": source_records,
        "pareto_augmented_pool_routes": len(routes),
        "all_pool_routes_replayed_before_optimization": True,
        "pool_replay_s": replay_s,
        "proof_scope": {
            "physical": (
                "Every selected route is an event-graph witness with a valid "
                "continuous replay; aggregate grid terminal energy is at least "
                "the GIRO physical total and is conservative for replay."
            ),
            "fixed": frontier["certificate_scope"],
            "joint": (
                "Two-stage set-covering MIP result over the immutable saved CG "
                "pool plus fixed-duty terminal frontiers. It is not a full-model "
                "pricing certificate for the new terminal-energy dual."
            ),
        },
        "unchanged_conditions": {
            "fleet_cap": 5, "initial_energy_per_bus_kwh": 240.0,
            "charge_power_kw": 350.0, "station_capacity_modeled": False,
            "charge_start_fee": 5.0, "tariff": cell["tariff"],
        },
    }
    atomic_json(folder / "mip.json", {
        "solver": solver, "joint_solution": joint,
        "pool_routes": len(routes), "saved_pool_records": source_records,
    })
    atomic_json(folder / "comparison.json", comparison)
    atomic_json(folder / "COMPLETE.json", {
        "mip_sha256": sha(folder / "mip.json"),
        "comparison_sha256": sha(folder / "comparison.json"),
    })


def worker(args):
    plan, cell = load_plan(args)
    folder = args.root / cell["tariff"]
    if args.stage == "frontier":
        folder.mkdir()
    elif not folder.is_dir():
        raise FileNotFoundError("frontier stage directory is missing")
    write_allocation(folder, args.stage)
    if args.stage == "frontier":
        frontier_worker(args, plan, cell, folder)
    else:
        mip_worker(args, plan, cell, folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--commit", required=True)
    prepare_parser.add_argument("--source", action="append", required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--root", type=Path, required=True)
    worker_parser.add_argument("--index", type=int, choices=range(3), required=True)
    worker_parser.add_argument("--stage", choices=("frontier", "mip"), required=True)
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    (prepare if args.mode == "prepare" else worker)(args)


if __name__ == "__main__":
    main()

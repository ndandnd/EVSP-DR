"""Execute the public synthetic EVSP structural regression family."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from arcflow_oracle import (
    build_model,
    build_network_from_problem,
    gate_g1,
    gate_g4,
    index_active_arcs,
    solve,
)
from convert_utrecht_evsp import load_problem
from utils_v2 import base_station_name


def flat_zero_prices(problem) -> dict:
    hours = range(int(math.ceil(problem.horizon_min / 60.0)))
    return {
        base_station_name(station): {hour: 0.0 for hour in hours}
        for station in problem.stations
    }


def solve_grid(payload: dict, *, soc_step: float, block_min: int) -> dict:
    problem = load_problem(payload)
    vehicle = payload["vehicle"]
    data = build_network_from_problem(
        payload["name"], problem, flat_zero_prices(problem),
        prices_csv="public_flat_zero_tariff",
        soc_step=soc_step, block_min=block_min,
        g_kwh=float(vehicle["battery_kwh"]),
        charge_kw=60.0 * float(vehicle["max_charge_kwh_per_min"]),
        min_soc_frac=0.0,
    )
    arcs = index_active_arcs(data.network)
    gates = [gate_g1(data, arcs)]
    model = build_model(data, arcs)
    lp, _ = solve(model, objective_kind="fleet", integrality="none")
    if lp.status != "optimal" or lp.vehicles is None:
        return {"lp": asdict(lp), "integer": None, "gates": gates}
    fleet = int(math.ceil(lp.vehicles - 1e-7))
    integer, primal = solve(
        model, objective_kind="combined", integrality="all",
        fixed_fleet=fleet, time_limit_s=300,
    )
    if primal is not None and integer.all_arcs_integral:
        gate, routes = gate_g4(model, primal)
        gates.append(gate)
    else:
        routes = []
    return {
        "grid": {"soc_step": soc_step, "block_min": block_min},
        "network": {
            "nodes": arcs.full_nodes, "arcs": arcs.full_arcs,
            "active_arcs": arcs.size,
        },
        "lp": asdict(lp),
        "integer": {**asdict(integer), "routes": routes},
        "fleet_proven": bool(
            integer.all_arcs_integral
            and integer.vehicles is not None
            and round(integer.vehicles) == fleet
        ),
        "gates": gates,
    }


def solve_pair_pool(pool: dict, n_trips: int) -> dict:
    routes = pool["routes"]
    rows, columns = [], []
    for column, route in enumerate(routes):
        for trip in route["trips"]:
            rows.append(int(trip))
            columns.append(column)
    matrix = csc_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(n_trips, len(routes)),
    )
    result = milp(
        c=np.ones(len(routes)),
        integrality=np.ones(len(routes), dtype=np.uint8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(
            matrix, np.ones(n_trips), np.ones(n_trips)
        ),
    )
    return {
        "status": int(result.status),
        "message": str(result.message),
        "fleet": float(result.fun) if result.fun is not None else None,
        "selected_routes": [
            routes[index] for index, value in enumerate(
                result.x if result.x is not None else []
            )
            if value > 0.5
        ] if result.x is not None else [],
        "columns": len(routes),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    problem_payload = json.loads(args.problem.read_text())
    pool_payload = json.loads(args.pool.read_text())
    features = problem_payload["features"]
    fine = solve_grid(problem_payload, **features["fine_grid"])
    coarse = solve_grid(problem_payload, **features["coarse_grid"])
    pair_pool = solve_pair_pool(pool_payload, len(problem_payload["trips"]))
    fine_fleet = (
        fine["integer"]["vehicles"] if fine.get("integer") else None
    )
    coarse_fleet = (
        coarse["integer"]["vehicles"] if coarse.get("integer") else None
    )
    result = {
        "schema": "evsp-dr-public-synthetic-result-v1",
        "instance": problem_payload["name"],
        "source": problem_payload["source"],
        "features": features,
        "feature_summary": {
            "trip_count": len(problem_payload["trips"]),
            "time_span_min": (
                max(t["end_min"] for t in problem_payload["trips"])
                - min(t["start_min"] for t in problem_payload["trips"])
            ),
            "simultaneous_trip_lower_bound":
                features["simultaneous_trip_lower_bound"],
            "coarse_minus_fine_fleet": (
                coarse_fleet - fine_fleet
                if coarse_fleet is not None and fine_fleet is not None else None
            ),
            "pair_pool_minus_full_fine_fleet": (
                pair_pool["fleet"] - fine_fleet
                if pair_pool["fleet"] is not None and fine_fleet is not None
                else None
            ),
        },
        "fine": fine,
        "coarse": coarse,
        "restricted_pair_pool": pair_pool,
    }
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "fine_lp": fine["lp"].get("vehicles"),
        "fine_integer": fine_fleet,
        "coarse_lp": coarse["lp"].get("vehicles"),
        "coarse_integer": coarse_fleet,
        "pair_pool_integer": pair_pool["fleet"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run EVSP-DR's arc-flow method on a converted public E-VSP benchmark."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

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


def flat_prices(problem, value: float = 0.0) -> dict:
    hours = range(int(math.ceil(problem.horizon_min / 60.0)))
    return {
        base_station_name(station): {hour: float(value) for hour in hours}
        for station in problem.stations
    }


def run_benchmark(
    problem_path: Path,
    *,
    soc_step: float,
    block_min: int,
    g_kwh: float,
    charge_kw: float,
    min_soc_frac: float,
    published_fleet: int,
    published_cg_lp_score: float | None,
    published_integer_score: float | None,
    integrality: str,
    time_limit_s: float,
    max_fleet: int,
) -> dict:
    payload = json.loads(problem_path.read_text())
    problem = load_problem(payload)
    data = build_network_from_problem(
        payload["name"], problem, flat_prices(problem),
        prices_csv="public_flat_zero_tariff",
        soc_step=soc_step, block_min=block_min, g_kwh=g_kwh,
        charge_kw=charge_kw, min_soc_frac=min_soc_frac,
    )
    arcs = index_active_arcs(data.network)
    gates = [gate_g1(data, arcs)]
    model = build_model(data, arcs)
    lp, _lp_primal = solve(
        model, objective_kind="fleet", integrality="none",
        time_limit_s=time_limit_s,
    )
    if lp.status != "optimal" or lp.vehicles is None:
        return {
            "schema": "evsp-dr-public-benchmark-result-v1",
            "instance": payload["name"],
            "source": payload["source"],
            "network": {
                "nodes": arcs.full_nodes, "arcs": arcs.full_arcs,
                "active_arcs": arcs.size,
            },
            "lp": asdict(lp),
            "integer": None,
            "gates": gates,
            "status": "lp_not_optimal",
        }

    first_candidate = int(math.ceil(lp.vehicles - 1e-7))
    integer_result = None
    integer_routes = []
    for fleet in range(first_candidate, max_fleet + 1):
        candidate, primal = solve(
            model, objective_kind="combined", integrality=integrality,
            fixed_fleet=fleet, time_limit_s=time_limit_s,
        )
        if primal is not None and candidate.all_arcs_integral:
            gate, integer_routes = gate_g4(model, primal)
            gates.append(gate)
            integer_result = candidate
            break
        if candidate.status == "infeasible":
            continue
        integer_result = candidate
        break

    proven_fleet = (
        integer_result is not None
        and integer_result.all_arcs_integral
        and integer_result.vehicles is not None
        and round(integer_result.vehicles) == first_candidate
    )
    our_fleet = (
        int(round(integer_result.vehicles))
        if integer_result is not None
        and integer_result.all_arcs_integral
        and integer_result.vehicles is not None
        else None
    )
    return {
        "schema": "evsp-dr-public-benchmark-result-v1",
        "instance": payload["name"],
        "source": payload["source"],
        "physics": {
            "g_kwh": g_kwh, "charge_kw": charge_kw,
            "min_soc_frac": min_soc_frac,
            "soc_step": soc_step, "block_min": block_min,
            "tariff": "flat_zero",
        },
        "network": {
            "nodes": arcs.full_nodes, "arcs": arcs.full_arcs,
            "active_arcs": arcs.size,
        },
        "lp": asdict(lp),
        "integer": (
            {**asdict(integer_result), "routes": integer_routes}
            if integer_result is not None else None
        ),
        "fleet_proven": proven_fleet,
        "published": {
            "fleet": published_fleet,
            "cg_lp_score": published_cg_lp_score,
            "integer_score": published_integer_score,
        },
        "comparison": {
            "fleet_difference": (
                our_fleet - published_fleet
                if our_fleet is not None else None
            ),
            "fleet_gap_percent": (
                100.0 * (our_fleet - published_fleet) / published_fleet
                if our_fleet is not None else None
            ),
            "objective_gap_percent": None,
            "objective_gap_reason":
                "objectives differ; EVSP-DR omits nonlinear charging and "
                "battery-degradation costs and uses a different bus cost",
        },
        "gates": gates,
        "status": "fleet_proven" if proven_fleet else "not_proven",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--soc-step", type=float, required=True)
    parser.add_argument("--block-min", type=int, required=True)
    parser.add_argument("--g-kwh", type=float, required=True)
    parser.add_argument("--charge-kw", type=float, required=True)
    parser.add_argument("--min-soc-frac", type=float, default=0.0)
    parser.add_argument("--published-fleet", type=int, required=True)
    parser.add_argument("--published-cg-lp-score", type=float)
    parser.add_argument("--published-integer-score", type=float)
    parser.add_argument(
        "--integrality", choices=("all", "service"), default="service"
    )
    parser.add_argument("--time-limit-s", type=float, default=600.0)
    parser.add_argument("--max-fleet", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_benchmark(
        args.problem, soc_step=args.soc_step, block_min=args.block_min,
        g_kwh=args.g_kwh, charge_kw=args.charge_kw,
        min_soc_frac=args.min_soc_frac,
        published_fleet=args.published_fleet,
        published_cg_lp_score=args.published_cg_lp_score,
        published_integer_score=args.published_integer_score,
        integrality=args.integrality, time_limit_s=args.time_limit_s,
        max_fleet=args.max_fleet,
    )
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "instance": result["instance"],
        "lp_fleet": result["lp"].get("vehicles"),
        "integer_fleet": (
            result["integer"].get("vehicles")
            if result.get("integer") else None
        ),
        "published_fleet": result.get("published", {}).get("fleet"),
        "status": result["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


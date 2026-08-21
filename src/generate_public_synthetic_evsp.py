"""Generate a public, seeded EVSP family with known structural stressors."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

from convert_utrecht_evsp import SCHEMA


DEFAULT_SEED = 20260821
GENERATOR = "evsp-dr-public-structural-v1"


def generate_family(seed: int = DEFAULT_SEED) -> tuple[dict, dict]:
    rng = random.Random(seed)
    first_start = 10 * rng.randrange(24, 37)
    starts = (first_start, first_start + 35, first_start + 70)
    trips = []
    for wave, start in enumerate(starts):
        for stream in range(2):
            trip_id = 2 * wave + stream
            trips.append({
                "id": trip_id,
                "source_ordinal": trip_id,
                "external_id": f"wave{wave + 1}-stream{stream + 1}",
                "line": f"S{stream + 1}",
                "trip_number": str(wave + 1),
                "from": "HUB",
                "start_min": start,
                "end_min": start + 20,
                "to": "HUB",
                "distance_km": 0.0,
                "energy_kwh": 30.0,
            })
    horizon = 10 * ((max(trip["end_min"] for trip in trips) + 69) // 10)
    payload = {
        "schema": SCHEMA,
        "name": f"public_structural_s{seed}",
        "horizon_min": horizon,
        "depot": "DEPOT",
        "stations": ["HUB"],
        "vehicle": {
            "battery_kwh": 60.0,
            "energy_kwh_per_km": 0.0,
            "max_charge_kwh_per_min": 1.0,
        },
        "station_parameters": {
            "HUB": {
                "vehicle_capacity": 99,
                "setup_min": 0.0,
                "infrastructure_charge_kwh_per_min": 1.0,
                "max_bus_soc_fraction": 1.0,
            }
        },
        "time_windows": [{
            "version": 0, "start_min": 0, "end_min": horizon - 1,
        }],
        "deadheads": [
            {
                "from": "DEPOT", "to": "HUB",
                "travel_min_by_version": [0, 0, 0, 0],
                "distance_km": 0.0, "energy_kwh": 0.0,
            },
            {
                "from": "HUB", "to": "DEPOT",
                "travel_min_by_version": [0, 0, 0, 0],
                "distance_km": 0.0, "energy_kwh": 0.0,
            },
        ],
        "trips": trips,
        "source": {
            "generator": GENERATOR,
            "seed": seed,
            "proprietary_input": False,
        },
        "conversion": {
            "trip_energy": "engineered explicit kWh",
            "deadhead_energy": "zero-distance public synthetic geometry",
            "deadhead_time": "static",
            "charging_model_changed": False,
        },
        "features": {
            "simultaneous_trip_lower_bound": 2,
            "fine_grid": {"soc_step": 5.0, "block_min": 5},
            "coarse_grid": {"soc_step": 5.0, "block_min": 10},
            "mechanism":
                "two 15-minute layovers; fine grid charges 15+15 kWh, "
                "coarse grid charges only 10+10 kWh",
            "expected_fine_fleet": 2,
            "expected_coarse_fleet": 3,
        },
    }
    pair_routes = [
        {"trips": [trip["id"]], "origin": "singleton"}
        for trip in trips
    ]
    for left in trips:
        for right in trips:
            if left["id"] < right["id"] and left["end_min"] <= right["start_min"]:
                pair_routes.append({
                    "trips": [left["id"], right["id"]],
                    "origin": "all_feasible_pairs_only",
                })
    pool = {
        "schema": "evsp-dr-public-restricted-pool-v1",
        "instance": payload["name"],
        "seed": seed,
        "construction":
            "all singleton and temporally feasible two-trip routes; "
            "routes of length three are deliberately absent",
        "routes": pair_routes,
        "expected_pool_fleet": 3,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["generator_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload, pool


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    problem, pool = generate_family(args.seed)
    problem_path = args.output_dir / f"{problem['name']}.json"
    pool_path = args.output_dir / f"{problem['name']}.pair_pool.json"
    write_new(problem_path, problem)
    write_new(pool_path, pool)
    print(json.dumps({
        "seed": args.seed,
        "problem": str(problem_path.resolve()),
        "pool": str(pool_path.resolve()),
        "trips": len(problem["trips"]),
        "routes": len(pool["routes"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

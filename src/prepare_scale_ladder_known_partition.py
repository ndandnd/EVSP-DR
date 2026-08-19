#!/usr/bin/env python3
"""Prepare one verified known-partition diagnostic start for the ladder."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

from audit_giro_known_columns import HORIZON_MIN, build_problem
from build_tariff_response_manifest import REPO_ROOT, sha256_file
from config import CHARGING_STATIONS
from config import BUS_COST_KX, CHARGE_START_COST
from expanded_path_realization import (
    BLOCK_SCHEDULE_SCHEMA,
    blocks_from_continuous_stops,
    validate_continuous_charging_blocks,
)
from rerealize_routes import _arc_map, rerealize_route
from run_exact_pool_mip import (
    charging_stop_arrivals,
    validate_injected_route,
)
from scale_ladder_trip_identity import identity
from tariff_response_core import PHYSICS, giro_routes_for_instance, route_identity
from utils_v2 import (
    calculate_truck_route_cost_accurate,
    load_station_hourly_prices,
)


SCHEMA = "evsp-dr-scale-ladder-known-partition-v1"
HISTORICAL_FLAT_SHA256 = (
    "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200"
)
COVERAGE_POLICY = "historical_last_hour_extension_verified_constant"


def _extended_constant_prices(path):
    if sha256_file(path) != HISTORICAL_FLAT_SHA256:
        raise ValueError("historical flat tariff hash mismatch")
    prices = load_station_hourly_prices(path, CHARGING_STATIONS)
    required_hour = int(math.ceil(HORIZON_MIN / 60.0)) - 1
    for station, curve in prices.items():
        if len(set(curve.values())) != 1:
            raise ValueError("historical flat tariff is not constant")
        last = curve[max(curve)]
        for hour in range(required_hour + 1):
            curve.setdefault(hour, last)
    return prices


def prepare(instance_path, expected_sha256, output_path):
    started = time.perf_counter()
    instance_path = Path(instance_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    identities = identity(instance_path)
    if identities["instance_file_sha256"] != expected_sha256:
        raise ValueError("instance file hash mismatch")
    master = REPO_ROOT / "data/Par_VehicleDetails_Updated.csv"
    routes = giro_routes_for_instance(master, instance_path)
    problem = build_problem(
        instance_path.parent,
        instance_path.name,
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=REPO_ROOT / "data",
    )
    tariff_path = REPO_ROOT / "data/hourly_prices_flat.csv"
    prices = _extended_constant_prices(tariff_path)
    preprocessing_s = time.perf_counter() - started
    pricing_started = time.perf_counter()
    optimized = []
    arc = _arc_map(problem)
    depot_curve = prices.get("PARX") or next(iter(prices.values()))
    for route in routes:
        record, _planned_cost, reason = rerealize_route(
            route["trips"],
            problem,
            arc,
            prices,
            300.0,
            300.0,
            0.0,
        )
        if record is None:
            raise ValueError(
                f"known duty {route['duty_id']} infeasible: {reason}"
            )
        record["trips"] = list(route["trips"])
        record["route_nodes"] = list(record["route"])
        replay = validate_injected_route(
            problem, record, 300.0, 300.0, 0.0, HORIZON_MIN,
            arc_map=arc,
        )
        if replay is not None:
            raise ValueError(
                f"known duty {route['duty_id']} replay failed: {replay}"
            )
        cost = calculate_truck_route_cost_accurate(
            record,
            BUS_COST_KX,
            depot_curve,
            charge_rate_kw=300.0,
            station_hourly_prices=prices,
            charge_start_cost=CHARGE_START_COST,
        )
        blocks = blocks_from_continuous_stops(
            record,
            station_prices=prices,
            charge_kw=300.0,
            earliest_start_by_stop=charging_stop_arrivals(
                problem, record
            ),
        )
        validation = validate_continuous_charging_blocks(
            record,
            blocks,
            station_prices=prices,
            charge_kw=300.0,
            expected_continuous_cost=cost,
        )
        record.update({
            "duty_id": route["duty_id"],
            "source_ordered_trip_ids": route["source_ordered_trip_ids"],
            "cost": float(cost),
            "continuous_realized_cost": float(cost),
            "master_cost_semantics": "continuous_realized_cost",
            "continuous_realized_charging_blocks": blocks,
            "physical_realization": {
                "status": "validated_continuous_known_partition",
                "continuous_realized_charging_blocks_schema":
                    BLOCK_SCHEDULE_SCHEMA,
                "continuous_realized_charging_blocks_sha256":
                    validation["block_schedule_sha256"],
                "continuous_cost_pricing_certified": False,
            },
        })
        optimized.append(record)
    pricing_s = time.perf_counter() - pricing_started
    payload = {
        "schema": SCHEMA,
        "source": "KNOWN-PARTITION",
        "scientific_role":
            "feasibility_integral_assembly_diagnostic_not_algorithmic_recovery",
        "routes": optimized,
        "route_count": len(optimized),
        "trip_count": len(problem.trips),
        "route_identity_sha256": route_identity(optimized),
        "instance": instance_path.name,
        "instance_sha256": expected_sha256,
        "trip_identity": identities,
        "tariff": {
            "tariff_id": "historical_flat",
            "relative_path": "data/hourly_prices_flat.csv",
            "sha256": HISTORICAL_FLAT_SHA256,
            "coverage_policy": COVERAGE_POLICY,
        },
        "physics": PHYSICS,
        "discretized_fixed_duty_certified": False,
        "continuous_cost_pricing_certified": False,
        "certificate_scope":
            "none_feasibility_diagnostic_only",
        "runtime": {
            "preprocessing_s": preprocessing_s,
            "master_s": 0.0,
            "pricing_s": pricing_s,
            "postprocessing_s": 0.0,
        },
    }
    payload["partition_sha256"] = _canonical_sha(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.tmp.",
        mode="w",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _canonical_sha(payload):
    import hashlib
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--instance-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = prepare(args.instance, args.instance_sha256, args.out)
    print(json.dumps({
        "routes": payload["route_count"],
        "trips": payload["trip_count"],
        "partition_sha256": payload["partition_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit known-duty membership in the exact expanded pricing route space."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from audit_giro_known_columns import HORIZON_MIN, build_problem
from build_tariff_response_manifest import REPO_ROOT, sha256_file
from config import CHARGING_STATIONS
from fixed_duty_expanded_optimizer import optimize_fixed_duty
from rerealize_routes import _arc_map, rerealize_route
from scale_ladder_trip_identity import identity
from tariff_response_core import giro_routes_for_instance
from utils_v2 import load_station_hourly_prices


SCHEMA = "evsp-dr-scale-ladder-known-membership-v1"
FLAT_SHA256 = (
    "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200"
)
FIELDS = (
    "cell_id", "scale", "selection_replicate", "duty_id",
    "trip_count", "known_partition_continuously_feasible",
    "known_partition_in_primary_expanded_space",
    "fixed_sequence_pricing_certified", "first_feasible_soc_step",
    "first_feasible_block_min", "nonrepresentability_reason",
    "primary_soc_step", "primary_block_min", "adaptive_sensitivity_run",
    "instance_file_sha256", "ordered_trip_id_set_sha256",
    "solver_local_trip_index_sha256", "ordered_trip_sequence_sha256",
    "trip_identity_schema",
)


def _prices():
    path = REPO_ROOT / "data/hourly_prices_flat.csv"
    if sha256_file(path) != FLAT_SHA256:
        raise ValueError("historical flat tariff hash mismatch")
    prices = load_station_hourly_prices(path, CHARGING_STATIONS)
    required = int(math.ceil(HORIZON_MIN / 60.0)) - 1
    for curve in prices.values():
        if len(set(curve.values())) != 1:
            raise ValueError("historical flat tariff is not constant")
        value = curve[max(curve)]
        for hour in range(required + 1):
            curve.setdefault(hour, value)
    return prices


def audit(instance_path, expected_sha256, scale, selection_replicate):
    instance_path = Path(instance_path).resolve()
    identities = identity(instance_path)
    if identities["instance_file_sha256"] != expected_sha256:
        raise ValueError("instance file hash mismatch")
    routes = giro_routes_for_instance(
        REPO_ROOT / "data/Par_VehicleDetails_Updated.csv",
        instance_path,
    )
    problem = build_problem(
        instance_path.parent,
        instance_path.name,
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=REPO_ROOT / "data",
    )
    prices = _prices()
    arc = _arc_map(problem)
    rows = []
    adaptive = int(scale) <= 5
    for route in routes:
        continuous, _cost, continuous_reason = rerealize_route(
            route["trips"], problem, arc, prices, 300.0, 300.0, 0.0
        )
        continuously_feasible = continuous is not None
        grids = [(15.0, 10)]
        if adaptive:
            grids.extend([(5.0, 10), (2.5, 10), (1.0, 10)])
        outcomes = []
        for soc_step, block_min in grids:
            result = optimize_fixed_duty(
                problem,
                route["trips"],
                prices,
                g_kwh=300.0,
                charge_kw=300.0,
                reserve_kwh=0.0,
                soc_step=soc_step,
                block_min=block_min,
                tariff_id="historical_flat",
                tariff_sha256=FLAT_SHA256,
                instance_sha256=expected_sha256,
                allow_diagnostic_grid=True,
            )
            outcomes.append((soc_step, block_min, result))
            if result["feasible"]:
                break
        if (
            adaptive
            and outcomes
            and outcomes[-1][0:2] == (1.0, 10)
            and not outcomes[-1][2]["feasible"]
        ):
            result = optimize_fixed_duty(
                problem,
                route["trips"],
                prices,
                g_kwh=300.0,
                charge_kw=300.0,
                reserve_kwh=0.0,
                soc_step=1.0,
                block_min=5,
                tariff_id="historical_flat",
                tariff_sha256=FLAT_SHA256,
                instance_sha256=expected_sha256,
                allow_diagnostic_grid=True,
            )
            outcomes.append((1.0, 5, result))
        primary = outcomes[0][2]
        first = next((
            (soc_step, block_min, result)
            for soc_step, block_min, result in outcomes
            if result["feasible"]
        ), None)
        reason = None
        if not primary["feasible"]:
            reason = primary.get("reason") or "not_representable"
            if not adaptive:
                reason += ";adaptive_sensitivity_not_run_scale_gt5"
            elif first is None:
                reason += ";blocked_through_1kwh_5min"
        rows.append({
            "cell_id": f"k{int(scale):02d}_s{int(selection_replicate)}",
            "scale": int(scale),
            "selection_replicate": int(selection_replicate),
            "duty_id": route["duty_id"],
            "trip_count": len(route["trips"]),
            "known_partition_continuously_feasible": continuously_feasible,
            "known_partition_in_primary_expanded_space":
                primary["feasible"],
            "fixed_sequence_pricing_certified": (
                first is not None
                and first[2].get("certificate", {}).get("certified") is True
            ),
            "first_feasible_soc_step": first[0] if first else None,
            "first_feasible_block_min": first[1] if first else None,
            "nonrepresentability_reason": reason or continuous_reason,
            "primary_soc_step": 15.0,
            "primary_block_min": 10,
            "adaptive_sensitivity_run": adaptive,
            **{
                field: identities[field] for field in (
                    "instance_file_sha256",
                    "ordered_trip_id_set_sha256",
                    "solver_local_trip_index_sha256",
                    "ordered_trip_sequence_sha256",
                    "trip_identity_schema",
                )
            },
        })
    grid_order = [
        (15.0, 10), (5.0, 10), (2.5, 10), (1.0, 10), (1.0, 5)
    ]
    grid_rank = {grid: index for index, grid in enumerate(grid_order)}
    aggregate_first = next((
        grid for grid in grid_order
        if all(
            row["first_feasible_soc_step"] is not None
            and grid_rank[(
                float(row["first_feasible_soc_step"]),
                int(row["first_feasible_block_min"]),
            )] <= grid_rank[grid]
            for row in rows
        )
    ), None)
    aggregate = {
        "schema": SCHEMA,
        "cell_id": f"k{int(scale):02d}_s{int(selection_replicate)}",
        "scale": int(scale),
        "selection_replicate": int(selection_replicate),
        "known_partition_continuously_feasible": all(
            row["known_partition_continuously_feasible"] for row in rows
        ),
        "known_partition_in_primary_expanded_space": all(
            row["known_partition_in_primary_expanded_space"] for row in rows
        ),
        "fixed_sequence_pricing_certified": all(
            row["fixed_sequence_pricing_certified"] for row in rows
        ),
        "first_feasible_soc_step":
            aggregate_first[0] if aggregate_first else None,
        "first_feasible_block_min":
            aggregate_first[1] if aggregate_first else None,
        "nonrepresentability_reason": ";".join(sorted({
            row["nonrepresentability_reason"]
            for row in rows if row["nonrepresentability_reason"]
        })),
        "duties": rows,
        "trip_identity": identities,
        "tariff_sha256": FLAT_SHA256,
        "diagnostic_only": True,
    }
    return aggregate


def write_outputs(payload, json_path, csv_path):
    for path in (json_path, csv_path):
        if Path(path).exists():
            raise FileExistsError(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(json_path).with_name(
        f".{Path(json_path).name}.tmp.{os.getpid()}"
    )
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, json_path)
    finally:
        temporary.unlink(missing_ok=True)
    with Path(csv_path).open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(payload["duties"])
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--instance-sha256", required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--selection-replicate", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit(
        args.instance, args.instance_sha256,
        args.scale, args.selection_replicate,
    )
    write_outputs(payload, args.out, args.csv_out)
    print(json.dumps({
        key: payload[key] for key in (
            "cell_id", "known_partition_continuously_feasible",
            "known_partition_in_primary_expanded_space",
            "first_feasible_soc_step", "first_feasible_block_min",
        )
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

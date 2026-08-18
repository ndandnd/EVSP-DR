#!/usr/bin/env python3
"""Prepare one tariff-specific, certified fixed-duty seed partition."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from audit_giro_known_columns import HORIZON_MIN, build_problem
from build_tariff_response_manifest import REPO_ROOT, sha256_file
from fixed_duty_expanded_optimizer import optimize_fixed_duty
from tariff_response_core import (
    PHYSICS,
    giro_routes_for_instance,
    load_tariff_manifest,
    route_identity,
    tariff_prices,
)


SCHEMA = "evsp-dr-tier1-fixed-duty-partition-v1"


def prepare(
    *,
    instance_path: Path,
    instance_sha256: str,
    master_path: Path,
    tariff_manifest: Path,
    tariff_id: str,
    reference_data_dir: Path,
    output_path: Path,
) -> dict:
    instance_path = instance_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if os.path.lexists(output_path):
        raise FileExistsError(output_path)
    if sha256_file(instance_path) != instance_sha256:
        raise ValueError("instance SHA-256 mismatch")
    tariffs = {
        row["tariff_id"]: row
        for row in load_tariff_manifest(tariff_manifest)
    }
    tariff = tariffs.get(tariff_id)
    if tariff is None:
        raise ValueError("tariff is not in the reviewed manifest")
    routes = giro_routes_for_instance(master_path, instance_path)
    problem = build_problem(
        instance_path.parent,
        instance_path.name,
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=reference_data_dir,
    )
    prices = tariff_prices(tariff)
    optimized = []
    certificates = []
    for route in routes:
        result = optimize_fixed_duty(
            problem,
            route["trips"],
            prices,
            tariff_id=tariff_id,
            tariff_sha256=tariff["sha256"],
            **{
                key: PHYSICS[key]
                for key in (
                    "g_kwh", "charge_kw", "reserve_kwh",
                    "soc_step", "block_min",
                )
            },
        )
        if not result["feasible"]:
            raise ValueError(
                f"fixed duty {route['duty_id']} infeasible: "
                f"{result['reason']}"
            )
        record = result["route"]
        record.update({
            "duty_id": route["duty_id"],
            "base_duty_id": route["base_duty_id"],
            "source_ordered_trip_ids":
                route["source_ordered_trip_ids"],
        })
        optimized.append(record)
        certificates.append({
            "duty_id": route["duty_id"],
            **result["certificate"],
        })
    payload = {
        "schema": SCHEMA,
        "source": (
            "GIRO40-AUGMENTED" if len(routes) == 40
            else "GIRO-AUGMENTED"
        ),
        "routes": optimized,
        "route_count": len(optimized),
        "trip_count": len(problem.trips),
        "route_identity_sha256": route_identity(optimized),
        "instance": instance_path.name,
        "instance_sha256": instance_sha256,
        "tariff": tariff,
        "physics": PHYSICS,
        "certificates": certificates,
        "discretized_fixed_duty_certified": True,
        "continuous_cost_pricing_certified": False,
        "certificate_scope":
            "optimal_discretized_charging_for_each_fixed_duty",
        "provenance": {
            "master": str(master_path.resolve()),
            "master_sha256": sha256_file(master_path),
            "tariff_manifest": str(tariff_manifest.resolve()),
            "tariff_manifest_sha256": sha256_file(tariff_manifest),
            "reference_sha256": sha256_file(
                reference_data_dir / "Ref_dict.csv"
            ),
            "deadhead_sha256": sha256_file(
                reference_data_dir / "par_ref_dhd.csv"
            ),
        },
    }
    payload["partition_sha256"] = hashlib_sha(payload)
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


def hashlib_sha(payload):
    import hashlib
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--instance-sha256", required=True)
    parser.add_argument(
        "--master", type=Path,
        default=REPO_ROOT / "data/Par_VehicleDetails_Updated.csv",
    )
    parser.add_argument(
        "--tariff-manifest", type=Path,
        default=REPO_ROOT / "data/tariff_response/tariff_manifest.csv",
    )
    parser.add_argument("--tariff-id", required=True)
    parser.add_argument(
        "--reference-data-dir", type=Path,
        default=REPO_ROOT / "data",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = prepare(
        instance_path=args.instance,
        instance_sha256=args.instance_sha256,
        master_path=args.master,
        tariff_manifest=args.tariff_manifest,
        tariff_id=args.tariff_id,
        reference_data_dir=args.reference_data_dir,
        output_path=args.out,
    )
    print(json.dumps({
        "route_count": payload["route_count"],
        "trip_count": payload["trip_count"],
        "partition_sha256": payload["partition_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

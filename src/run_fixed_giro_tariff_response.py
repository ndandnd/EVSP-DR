#!/usr/bin/env python3
"""Run Tier 0/1 tariff response for the frozen GIRO40 duty partition."""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

from audit_giro_known_columns import HORIZON_MIN, build_problem
from fixed_duty_expanded_optimizer import optimize_fixed_duty
from tariff_response_core import (
    GIRO40_PARTITION_PATH,
    PHYSICS,
    evaluate_giro_original,
    load_tariff_manifest,
    reconstruct_giro40_original,
    route_identity,
    tariff_prices,
)
from build_tariff_response_manifest import REPO_ROOT, sha256_file


SUMMARY_FIELDS = (
    "tariff_id", "tier", "buses", "grid_model_objective",
    "continuous_replay_objective", "charging_cost",
    "continuous_charging_cost", "total_charged_kwh",
    "peak_window_kwh", "charging_kwh_by_hour_json",
    "charging_kwh_by_station_json", "charging_starts_by_hour_json",
    "waiting_min", "deadhead_min",
    "deadhead_kwh", "charging_stops", "discretized_certification_status",
    "physical_replay_status", "terminal_soc_policy",
    "runtime_preprocessing_s", "runtime_master_s", "runtime_pricing_s",
    "runtime_postprocessing_s", "scalar_cost_availability",
    "availability_reason", "certificate_scope",
    "continuous_cost_pricing_certified", "route_identity_sha256",
)
BLOCK_FIELDS = (
    "tariff_id", "tier", "duty_id", "route_order", "stop_index",
    "block_index", "station", "start_min", "end_min", "realized_kwh",
    "expanded_grid_kwh", "price_per_kwh", "tariff_hour", "tariff_key",
    "availability", "availability_reason", "source_row",
)
CERT_FIELDS = (
    "tariff_id", "duty_id", "certified", "scope",
    "certificate_sha256", "expanded_grid_objective",
    "continuous_replay_objective", "labels_accepted",
    "transitions_evaluated", "physical_replay_status",
    "continuous_cost_optimality_certified",
)
DUTY_FIELDS = (
    "duty_id", "base_duty_id", "included_variant",
    "excluded_variant_id", "trip_count", "local_trip_ids_json",
    "source_ordered_trip_ids_json", "route_incidence_sha256",
    "recorded_charge_count", "recorded_charge_kwh",
)


def _write_csv(path, fields, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _peak_kwh(blocks, tariff_row):
    if tariff_row["peak_window_start_hour"] == "":
        return None
    start = int(tariff_row["peak_window_start_hour"]) * 60
    end = int(tariff_row["peak_window_end_hour"]) * 60
    return sum(
        float(block["realized_kwh"])
        for block in blocks
        if float(block["start_min"]) >= start
        and float(block["end_min"]) <= end
    )


def run(
    *,
    instance_path: Path,
    expected_instance_sha256: str,
    master_path: Path,
    tariff_manifest: Path,
    output_dir: Path,
    reference_data_dir: Path,
) -> dict:
    started = time.perf_counter()
    instance_path = instance_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output exists: {output_dir}")
    if sha256_file(instance_path) != expected_instance_sha256:
        raise ValueError("instance SHA-256 mismatch")
    tariffs = load_tariff_manifest(tariff_manifest)
    original = reconstruct_giro40_original(master_path)
    data_dir = instance_path.parent
    relative = instance_path.name
    problem = build_problem(
        data_dir,
        relative,
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=reference_data_dir,
    )
    if len(problem.trips) != 947:
        raise ValueError("Tier 0/1 requires the frozen 947-trip instance")
    preprocessing_s = time.perf_counter() - started

    staging = Path(tempfile.mkdtemp(
        dir=output_dir.parent, prefix=f".{output_dir.name}.tmp."
    ))
    summaries = []
    block_rows = []
    certificates = []
    tier1_payloads = {}
    try:
        _write_csv(
            staging / "giro40_duty_manifest.csv",
            DUTY_FIELDS,
            original["duties"],
        )
        for tariff in tariffs:
            tier0, events = evaluate_giro_original(original, tariff)
            summaries.append({
                **tier0,
                "continuous_charging_cost": None,
                "peak_window_kwh": None,
                "charging_kwh_by_hour_json": None,
                "charging_kwh_by_station_json": None,
                "charging_starts_by_hour_json": None,
                "waiting_min": None,
                "deadhead_min": None,
                "deadhead_kwh": None,
                "charging_stops": len(events),
                "runtime_preprocessing_s": preprocessing_s,
                "runtime_master_s": 0.0,
                "runtime_pricing_s": 0.0,
                "runtime_postprocessing_s": 0.0,
                "route_identity_sha256": route_identity(
                    original["routes"]
                ),
            })
            for event in events:
                block_rows.append({
                    "tariff_id": tariff["tariff_id"],
                    "tier": "TIER0_GIRO_ORIGINAL",
                    "duty_id": event["duty_id"],
                    "station": event["station"],
                    "start_min": event["start_min"],
                    "end_min": event["end_min"],
                    "realized_kwh": event["kwh"],
                    "expanded_grid_kwh": None,
                    "price_per_kwh": event["price_per_kwh"],
                    "availability": event["availability"],
                    "availability_reason": event[
                        "availability_reason"
                    ],
                    "source_row": event["source_row"],
                })

            prices = tariff_prices(tariff)
            pricing_started = time.perf_counter()
            route_results = []
            for route_order, route in enumerate(original["routes"]):
                result = optimize_fixed_duty(
                    problem,
                    route["trips"],
                    prices,
                    tariff_id=tariff["tariff_id"],
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
                        f"{tariff['tariff_id']} duty "
                        f"{route['duty_id']} infeasible: {result['reason']}"
                    )
                result["duty_id"] = route["duty_id"]
                result["route_order"] = route_order
                route_results.append(result)
            pricing_s = time.perf_counter() - pricing_started
            postprocessing_started = time.perf_counter()
            routes = [result["route"] for result in route_results]
            coverage = {
                trip
                for route in routes
                for trip in route["trips"]
            }
            if (
                len(routes) != 40
                or coverage != set(range(947))
                or sum(len(route["trips"]) for route in routes) != 947
            ):
                raise ValueError("Tier 1 is not an exact 40-duty partition")
            all_blocks = []
            for result in route_results:
                route = result["route"]
                blocks = route["continuous_realized_charging_blocks"]
                all_blocks.extend(blocks)
                certificate = result["certificate"]
                certificates.append({
                    "tariff_id": tariff["tariff_id"],
                    "duty_id": result["duty_id"],
                    "certified": certificate["certified"],
                    "scope": certificate["scope"],
                    "certificate_sha256":
                        certificate["certificate_sha256"],
                    "expanded_grid_objective":
                        result["expanded_grid_objective"],
                    "continuous_replay_objective":
                        result["continuous_replay_objective"],
                    "labels_accepted": certificate["labels_accepted"],
                    "transitions_evaluated":
                        certificate["transitions_evaluated"],
                    "physical_replay_status":
                        result["physical_replay_status"],
                    "continuous_cost_optimality_certified":
                        certificate[
                            "continuous_cost_optimality_certified"
                        ],
                })
                for block in blocks:
                    block_rows.append({
                        **block,
                        "tariff_id": tariff["tariff_id"],
                        "tier":
                            "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
                        "duty_id": result["duty_id"],
                        "route_order": result["route_order"],
                        "availability": "available",
                        "availability_reason": "",
                        "source_row": "",
                    })
            grid_objective = sum(
                result["expanded_grid_objective"]
                for result in route_results
            )
            continuous_objective = sum(
                result["continuous_replay_objective"]
                for result in route_results
            )
            total_kwh = sum(
                float(block["realized_kwh"]) for block in all_blocks
            )
            by_hour = {}
            by_station = {}
            for block in all_blocks:
                hour = str(int(block["tariff_hour"]))
                station = str(block["station"])
                by_hour[hour] = by_hour.get(hour, 0.0) + float(
                    block["realized_kwh"]
                )
                by_station[station] = by_station.get(
                    station, 0.0
                ) + float(block["realized_kwh"])
            starts_by_hour = {}
            for route in routes:
                for start in (
                    route.get("charging_stops") or {}
                ).get("cst", []):
                    hour = str(int(float(start) // 60))
                    starts_by_hour[hour] = starts_by_hour.get(hour, 0) + 1
            summaries.append({
                "tariff_id": tariff["tariff_id"],
                "tier": "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
                "buses": 40,
                "grid_model_objective": grid_objective,
                "continuous_replay_objective": continuous_objective,
                "charging_cost": grid_objective - BUS_COST_KX * 40,
                "continuous_charging_cost":
                    continuous_objective - BUS_COST_KX * 40,
                "total_charged_kwh": total_kwh,
                "peak_window_kwh": _peak_kwh(all_blocks, tariff),
                "charging_kwh_by_hour_json": json.dumps(
                    by_hour, sort_keys=True, separators=(",", ":")
                ),
                "charging_kwh_by_station_json": json.dumps(
                    by_station, sort_keys=True, separators=(",", ":")
                ),
                "charging_starts_by_hour_json": json.dumps(
                    starts_by_hour, sort_keys=True, separators=(",", ":")
                ),
                "waiting_min": sum(
                    result["waiting_min"] for result in route_results
                ),
                "deadhead_min": sum(
                    result["deadhead_min"] for result in route_results
                ),
                "deadhead_kwh": sum(
                    result["deadhead_kwh"] for result in route_results
                ),
                "charging_stops": sum(
                    len(route["charging_stops"]["stations"])
                    for route in routes
                ),
                "discretized_certification_status":
                    "certified_all_40_fixed_duties",
                "physical_replay_status": "validated_all_routes",
                "terminal_soc_policy": PHYSICS[
                    "terminal_soc_policy"
                ],
                "runtime_preprocessing_s": preprocessing_s,
                "runtime_master_s": 0.0,
                "runtime_pricing_s": pricing_s,
                "runtime_postprocessing_s":
                    time.perf_counter() - postprocessing_started,
                "scalar_cost_availability": "available",
                "availability_reason": "",
                "certificate_scope":
                    "optimal_discretized_charging_fixed_sequences",
                "continuous_cost_pricing_certified": False,
                "route_identity_sha256": route_identity(routes),
            })
            payload = {
                "schema": "evsp-dr-tier1-giro40-tariff-v1",
                "tariff": tariff,
                "instance_sha256": expected_instance_sha256,
                "physics": PHYSICS,
                "routes": routes,
                "route_identity_sha256": route_identity(routes),
                "certificates": [
                    result["certificate"] for result in route_results
                ],
                "continuous_cost_pricing_certified": False,
            }
            tier1_path = staging / (
                f"tier1_{tariff['tariff_id']}.json"
            )
            tier1_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
            tier1_payloads[tariff["tariff_id"]] = tier1_path

        _write_csv(
            staging / "tariff_response_summary.csv",
            SUMMARY_FIELDS,
            sorted(summaries, key=lambda row: (
                row["tariff_id"], row["tier"]
            )),
        )
        _write_csv(
            staging / "charging_blocks_long.csv",
            BLOCK_FIELDS,
            sorted(block_rows, key=lambda row: (
                row["tariff_id"], row["tier"],
                str(row.get("duty_id", "")),
                float(row.get("start_min", 0)),
                int(row.get("block_index") or 0),
            )),
        )
        _write_csv(
            staging / "fixed_duty_certificate_summary.csv",
            CERT_FIELDS,
            sorted(certificates, key=lambda row: (
                row["tariff_id"], row["duty_id"]
            )),
        )
        provenance = {
            "schema": "evsp-dr-fixed-giro40-provenance-v1",
            "python": platform.python_version(),
            "instance": str(instance_path),
            "instance_sha256": expected_instance_sha256,
            "master": str(master_path.resolve()),
            "master_sha256": sha256_file(master_path),
            "partition": str(GIRO40_PARTITION_PATH),
            "partition_sha256": sha256_file(GIRO40_PARTITION_PATH),
            "tariff_manifest": str(tariff_manifest.resolve()),
            "tariff_manifest_sha256": sha256_file(tariff_manifest),
            "physics": PHYSICS,
            "continuous_cost_pricing_certified": False,
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        _rename_noreplace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "output_dir": str(output_dir),
        "tariffs": len(tariffs),
        "summary_rows": len(summaries),
        "certificate_rows": len(certificates),
    }


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        -100, os.fsencode(source), -100, os.fsencode(target), 1
    ) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(target)
        raise OSError(error, os.strerror(error), target)


def main(argv=None) -> int:
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
    parser.add_argument(
        "--reference-data-dir", type=Path,
        default=REPO_ROOT / "data",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        instance_path=args.instance,
        expected_instance_sha256=args.instance_sha256,
        master_path=args.master,
        tariff_manifest=args.tariff_manifest,
        output_dir=args.out_dir,
        reference_data_dir=args.reference_data_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

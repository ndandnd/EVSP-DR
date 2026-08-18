#!/usr/bin/env python3
"""Generate the tracked synthetic-only tariff-response example package."""

from __future__ import annotations

import json
from pathlib import Path

from build_tariff_response_evidence import SCHEMA, build
from build_tariff_response_manifest import REPO_ROOT, sha256_file
from expanded_path_realization import charging_block_schedule_sha256
from tariff_response_core import PHYSICS, load_tariff_manifest, tariff_prices


ROOT = REPO_ROOT / "analysis/tariff_response_synthetic_reviewed_20260818"
ALPHA_IDS = (
    "peak12_alpha_0p0",
    "peak12_alpha_0p25",
    "peak12_alpha_0p5",
    "peak12_alpha_1p0",
    "peak12_alpha_2p0",
)


def cell(tariff, tier, treatment, *, flexible=False):
    price = tariff_prices(tariff)["PARX"][12]
    factor = (
        1.0 if tier == "TIER0_GIRO_ORIGINAL"
        else 0.8 if tier == "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING"
        else 0.6
    )
    energy = 10.0 * factor
    block = {
        "stop_index": 0,
        "block_index": 0,
        "station": "PARX_1",
        "start_min": 720.0,
        "end_min": 730.0,
        "realized_kwh": energy,
        "expanded_grid_kwh": energy,
        "tariff_hour": 12,
        "tariff_key": "PARX:12",
        "price_per_kwh": price,
    }
    groups = [[0, 1]] if flexible else [[0], [1]]
    routes = []
    for index, trips in enumerate(groups):
        blocks = [block] if index == 0 else []
        routes.append({
            "route_id": f"{tier}-{index}",
            "trips": trips,
            "trip_blocks": [{
                "trip_id": trip,
                "start_min": 100 + 60 * trip,
                "end_min": 125 + 60 * trip,
            } for trip in trips],
            "charging_stops": {
                "stations": ["PARX_1"] if blocks else [],
                "cst": [720] if blocks else [],
                "cet": [730] if blocks else [],
                "kwh": [energy] if blocks else [],
            },
            "continuous_realized_charging_blocks": blocks,
            "recorded_charging_blocks": blocks,
            "continuous_realized_charging_blocks_sha256":
                charging_block_schedule_sha256(blocks),
            "cost_tariff_sha256": (
                None if tier == "TIER0_GIRO_ORIGINAL"
                else tariff["sha256"]
            ),
            "expanded_grid_cost":
                100000 + (5 + energy * price if blocks else 0),
            "continuous_realized_cost":
                100000 + (5 + energy * price if blocks else 0),
            "waiting_min": 20.0 * factor if index == 0 else 0.0,
            "deadhead_min": 5.0 * factor if index == 0 else 0.0,
            "deadhead_kwh": 2.0 * factor if index == 0 else 0.0,
            "continuous_terminal_soc_kwh": 50.0,
        })
    buses = len(routes)
    charging = 5 + energy * price
    certificates = []
    if tier == "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING":
        import hashlib
        for route in routes:
            payload = {
                "certified": True,
                "scope": "synthetic_discretized_fixed_duty",
                "continuous_cost_optimality_certified": False,
            }
            digest = hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            route["fixed_duty_certificate_sha256"] = digest
            certificates.append({
                "route_id": route["route_id"],
                **payload,
                "certificate_sha256": digest,
            })
    return {
        "cell_id": f"{tier}-{tariff['tariff_id']}",
        "instance_id": "SYNTHETIC-k2-NOT-SCIENTIFIC",
        "scale": 2,
        "tariff_id": tariff["tariff_id"],
        "tariff_sha256": tariff["sha256"],
        "tier": tier,
        "treatment": treatment,
        "trip_ids": [0, 1],
        "routes": routes,
        "metrics": {
            "buses": buses,
            "grid_model_objective": buses * 100000 + charging,
            "continuous_replay_objective": buses * 100000 + charging,
            "charging_cost": charging,
            "continuous_charging_cost": charging,
            "total_charged_kwh": energy,
            "peak_window_kwh": energy,
            "charging_kwh_by_hour_json": json.dumps({"12": energy}),
            "charging_kwh_by_station_json": json.dumps(
                {"PARX_1": energy}
            ),
            "charging_starts_by_hour_json": json.dumps({"12": 1}),
            "terminal_soc_min_kwh": 50.0,
            "terminal_soc_max_kwh": 50.0,
            "waiting_min": 20.0 * factor,
            "deadhead_min": 5.0 * factor,
            "deadhead_kwh": 2.0 * factor,
            "charging_stops": 1,
            "discretized_certification_status": (
                "not_applicable_synthetic"
                if tier == "TIER0_GIRO_ORIGINAL"
                else "synthetic_certificate_fixture"
            ),
            "runtime_preprocessing_s": 0.1,
            "runtime_master_s": 0.2 if tier.startswith("TIER2") else 0.0,
            "runtime_pricing_s": 0.3,
            "runtime_postprocessing_s": 0.1,
        },
        "physical_replay_status": (
            "unavailable_synthetic_recorded"
            if tier == "TIER0_GIRO_ORIGINAL"
            else "validated_all_routes"
        ),
        "terminal_soc_policy": PHYSICS["terminal_soc_policy"],
        "continuous_cost_pricing_certified": False,
        "certificate_scope": (
            "none_synthetic_recorded"
            if tier == "TIER0_GIRO_ORIGINAL"
            else "synthetic_discretized_fixed_duty"
            if tier.startswith("TIER1")
            else "synthetic_finite_augmented_pool"
        ),
        "fixed_duty_certificates": certificates,
        "cg_iterations": ([{
            "iteration": 1,
            "elapsed_s": 1.0,
            "lp_obj": float(buses),
            "route_weight": float(buses),
            "artificials": 0.0,
            "min_rc": -0.1,
            "pool_columns": 3,
        }] if tier.startswith("TIER2") else []),
        "mip_checkpoints": ([{
            "checkpoint_elapsed_s": 60,
            "incumbent_fleet": buses,
            "fleet_bound": buses,
            "fleet_gap": 0.0,
            "node_count": 1,
            "solution_count": 1,
            "route_vector_sha256": "d" * 64,
            "solver_ended_before_checkpoint": False,
        }] if tier.startswith("TIER2") else []),
        "source_artifacts": [],
    }


def main():
    if ROOT.exists() and any(
        path.is_file() for path in ROOT.rglob("*")
    ):
        raise FileExistsError(ROOT)
    input_dir = ROOT / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    duty = input_dir / "giro40_duty_manifest.synthetic.csv"
    duty.write_text(
        "duty_id,warning\n"
        "synthetic-duty-0,SYNTHETIC ONLY\n"
        "synthetic-duty-1,SYNTHETIC ONLY\n"
    )
    tariff_manifest = (
        REPO_ROOT / "data/tariff_response/tariff_manifest.csv"
    )
    tariffs = {
        row["tariff_id"]: row
        for row in load_tariff_manifest(tariff_manifest)
    }
    cells = []
    for tariff_id in ALPHA_IDS:
        tariff = tariffs[tariff_id]
        cells.extend([
            cell(
                tariff, "TIER0_GIRO_ORIGINAL", "GIRO_ORIGINAL"
            ),
            cell(
                tariff,
                "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
                "FIXED_GIRO",
            ),
            cell(
                tariff,
                "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING",
                "GIRO40-AUGMENTED",
                flexible=True,
            ),
        ])
    manifest = {
        "schema": SCHEMA,
        "synthetic": True,
        "warning": "SYNTHETIC FIXTURE — NOT EXPERIMENTAL EVIDENCE",
        "physics": PHYSICS,
        "tariff_manifest": "data/tariff_response/tariff_manifest.csv",
        "tariff_manifest_sha256": sha256_file(tariff_manifest),
        "giro40_duty_manifest": str(duty.relative_to(REPO_ROOT)),
        "giro40_duty_manifest_sha256": sha256_file(duty),
        "cells": cells,
    }
    manifest_path = input_dir / "experiment_manifest.synthetic.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    build(manifest_path, ROOT / "example_output")
    (ROOT / "README.md").write_text(
        "# Synthetic tariff-response example\n\n"
        "**SYNTHETIC FIXTURE — NOT EXPERIMENTAL EVIDENCE.**\n\n"
        "This two-trip package only demonstrates schemas, deterministic "
        "tables, Gantt rendering, response curves, and fail-closed "
        "provenance. It must not be cited as an EVSP-DR result.\n"
    )
    print(ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

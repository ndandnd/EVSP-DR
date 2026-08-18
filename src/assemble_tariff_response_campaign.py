#!/usr/bin/env python3
"""Assemble completed pilot outputs into a verified evidence manifest/package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from audit_giro_known_columns import HORIZON_MIN, build_problem
from build_tariff_response_evidence import SCHEMA, build
from build_tariff_response_manifest import REPO_ROOT, sha256_file
from config import BUS_COST_KX
from tariff_response_core import (
    PHYSICS,
    evaluate_giro_original,
    giro_routes_for_instance,
    load_tariff_manifest,
    tariff_prices,
)
from make_giro_seed_routes import _minutes, _station_node


def _trip_blocks(problem, trips):
    return [{
        "trip_id": trip,
        "start_min": float(problem.start_min[trip]),
        "end_min": float(problem.end_min[trip]),
    } for trip in trips]


def _timeline_metrics(problem, route):
    arc = {
        (source, target): (float(travel), float(energy))
        for source, entries in problem.adjacency.items()
        for target, travel, energy, _kind in entries
    }
    nodes = route.get("route_nodes", route.get("route", []))
    stops = route.get("charging_stops") or {}
    station_index = 0
    clock = 0.0
    waiting = 0.0
    deadhead_min = 0.0
    deadhead_kwh = 0.0
    for left, right in zip(nodes, nodes[1:]):
        travel, energy = arc[(left, right)]
        clock += travel
        deadhead_min += travel
        deadhead_kwh += energy
        if isinstance(right, int):
            waiting += max(0.0, problem.start_min[right] - clock)
            clock = float(problem.end_min[right])
        elif right != "PARX_0":
            start = float(stops["cst"][station_index])
            end = float(stops["cet"][station_index])
            waiting += max(0.0, start - clock)
            clock = end
            station_index += 1
    return waiting, deadhead_min, deadhead_kwh


def _aggregate(routes, tariff, problem):
    prices = tariff_prices(tariff)
    grid = continuous = total_kwh = 0.0
    waiting = deadhead_min = deadhead_kwh = 0.0
    stops = 0
    by_hour = defaultdict(float)
    by_station = defaultdict(float)
    starts = defaultdict(int)
    peak = (
        0.0 if tariff["peak_window_start_hour"] != "" else None
    )
    peak_start = (
        int(tariff["peak_window_start_hour"]) * 60
        if peak is not None else None
    )
    peak_end = (
        int(tariff["peak_window_end_hour"]) * 60
        if peak is not None else None
    )
    terminal = []
    normalized = []
    for index, source in enumerate(routes):
        route = dict(source)
        route["route_id"] = route.get(
            "duty_id", f"route-{index:04d}"
        )
        route["trip_blocks"] = _trip_blocks(problem, route["trips"])
        timeline = _timeline_metrics(problem, route)
        route.setdefault("waiting_min", timeline[0])
        route.setdefault("deadhead_min", timeline[1])
        route.setdefault("deadhead_kwh", timeline[2])
        route.setdefault(
            "continuous_realized_charging_blocks_sha256",
            (route.get("physical_realization") or {}).get(
                "continuous_realized_charging_blocks_sha256"
            ),
        )
        blocks = route["continuous_realized_charging_blocks"]
        for block in blocks:
            kwh = float(block["realized_kwh"])
            total_kwh += kwh
            hour = str(int(block["tariff_hour"]))
            by_hour[hour] += kwh
            by_station[block["station"]] += kwh
            if (
                peak is not None
                and float(block["start_min"]) >= peak_start
                and float(block["end_min"]) <= peak_end
            ):
                peak += kwh
        for start in (route.get("charging_stops") or {}).get("cst", []):
            starts[str(int(float(start) // 60))] += 1
        stops += len(
            (route.get("charging_stops") or {}).get("stations", [])
        )
        grid += float(route["expanded_grid_cost"])
        continuous += float(route["continuous_realized_cost"])
        waiting += float(route["waiting_min"])
        deadhead_min += float(route["deadhead_min"])
        deadhead_kwh += float(route["deadhead_kwh"])
        terminal.append(float(route["continuous_terminal_soc_kwh"]))
        normalized.append(route)
    return normalized, {
        "buses": len(routes),
        "grid_model_objective": grid,
        "continuous_replay_objective": continuous,
        "charging_cost": grid - BUS_COST_KX * len(routes),
        "continuous_charging_cost":
            continuous - BUS_COST_KX * len(routes),
        "total_charged_kwh": total_kwh,
        "peak_window_kwh": peak,
        "charging_kwh_by_hour_json": json.dumps(
            dict(by_hour), sort_keys=True, separators=(",", ":")
        ),
        "charging_kwh_by_station_json": json.dumps(
            dict(by_station), sort_keys=True, separators=(",", ":")
        ),
        "charging_starts_by_hour_json": json.dumps(
            dict(starts), sort_keys=True, separators=(",", ":")
        ),
        "terminal_soc_min_kwh": min(terminal),
        "terminal_soc_max_kwh": max(terminal),
        "waiting_min": waiting,
        "deadhead_min": deadhead_min,
        "deadhead_kwh": deadhead_kwh,
        "charging_stops": stops,
    }


def _recorded_subset(master_path, duties, routes, tariff):
    frame = pd.read_csv(master_path)
    frame["VehicleTask"] = frame["VehicleTask"].astype(str)
    events = []
    for index, row in frame[
        frame["VehicleTask"].isin(duties)
        & (frame["Identifier"] == "Recharge")
    ].iterrows():
        events.append({
            "source_row": int(index + 2),
            "duty_id": str(row["VehicleTask"]),
            "station": _station_node(str(row["From1"])),
            "station_base": _station_node(
                str(row["From1"])
            ).rsplit("_", 1)[0],
            "raw_from": str(row["From1"]),
            "raw_to": str(row["To1"]),
            "raw_start": str(row["Start1"]),
            "raw_end": str(row["End1"]),
            "start_min": _minutes(str(row["Start1"])),
            "end_min": _minutes(str(row["End1"])),
            "kwh": float(row["Recharge kWh"]),
        })
    original = {
        "events": events,
        "routes": routes,
        "recorded_terminal_soc_policy":
            "unavailable_no_declared_operator_policy",
    }
    return evaluate_giro_original(original, tariff)


def _cg_iterations(job):
    path = Path(str(job["output"]) + ".iters.csv")
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "iteration": int(float(row["iteration"])),
                "elapsed_s": float(row["elapsed_s"]),
                "lp_obj": float(row["lp_obj"]),
                "route_weight": float(row["route_weight"]),
                "artificials": float(row["artificials"]),
                "min_rc": float(row["min_rc"]),
                "pool_columns": int(float(row["pool_columns"])),
            })
    return rows


def _mip_checkpoints(job):
    rows = []
    for path in sorted(Path(job["progress_dir"]).glob("checkpoint_*.json")):
        payload = json.loads(path.read_text())
        incumbent = payload.get("incumbent") or {}
        statistics = payload.get("latest_statistics") or {}
        rows.append({
            "checkpoint_elapsed_s": payload["checkpoint_elapsed_s"],
            "incumbent_fleet": incumbent.get("fleet"),
            "fleet_bound": statistics.get("fleet_bound"),
            "fleet_gap": statistics.get("fleet_gap"),
            "node_count": statistics.get("node_count"),
            "solution_count": statistics.get("solution_count"),
            "route_vector_sha256": incumbent.get("route_vector_sha256"),
            "solver_ended_before_checkpoint":
                payload.get("solver_ended_before_checkpoint"),
        })
    return rows


def assemble(campaign_root, output_manifest, evidence_output):
    campaign_root = campaign_root.resolve()
    plan_raw = (campaign_root / "approved-plan.json").read_bytes()
    plan = json.loads(plan_raw)
    manifest = json.loads((campaign_root / "campaign.json").read_text())
    if manifest.get("approval_sha256") != hashlib.sha256(
        plan_raw
    ).hexdigest():
        raise ValueError("campaign approval hash mismatch")
    submitted = {
        item["job_key"] for item in manifest.get("submitted_jobs") or []
    }
    main_jobs = {
        job["job_key"] for job in plan["jobs"]
        if not job["separate_k40_gate"]
    }
    if submitted != main_jobs:
        raise ValueError("main pilot submission is incomplete")
    for job in plan["jobs"]:
        if job["job_key"] not in submitted:
            continue
        completion_path = Path(
            str(job["output"]) + ".worker-completion.json"
        )
        completion = json.loads(completion_path.read_text())
        if (
            completion.get("schema")
            != "evsp-dr-tariff-response-worker-completion-v1"
            or completion.get("plan_sha256")
            != manifest["approval_sha256"]
        ):
            raise ValueError("worker completion provenance mismatch")
        for artifact, digest in (
            completion.get("artifact_sha256") or {}
        ).items():
            path = Path(artifact)
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError("worker artifact changed after validation")
    jobs = {job["job_key"]: job for job in plan["jobs"]}
    tariffs = {
        row["tariff_id"]: row
        for row in load_tariff_manifest(Path(plan["tariff_manifest"]))
    }
    master = REPO_ROOT / "data/Par_VehicleDetails_Updated.csv"
    problems = {}
    giro_routes = {}
    for scale in (5, 8, 40):
        instance = Path(next(
            job["instance"]["path"] for job in plan["jobs"]
            if not job["separate_k40_gate"] and job["scale"] == scale
        ))
        problems[scale] = build_problem(
            instance.parent, instance.name,
            max_station_to_trip_wait_min=HORIZON_MIN,
            reference_data_dir=REPO_ROOT / "data",
        )
        giro_routes[scale] = giro_routes_for_instance(master, instance)
    cells = []
    artifacts = []
    fixed_root = (
        Path(plan["campaign_root"])
        / "outputs/fixed_full_giro40_all_tariffs"
    )
    with (fixed_root / "tariff_response_summary.csv").open(
        newline=""
    ) as handle:
        fixed_summary = {
            (row["tariff_id"], row["tier"]): row
            for row in csv.DictReader(handle)
        }
    for scale in (5, 8, 40):
        problem = problems[scale]
        for tariff_id, tariff in tariffs.items():
            seed_path = (
                fixed_root / f"tier1_{tariff_id}.json"
                if scale == 40
                else Path(next(
                    job["output"] for job in plan["jobs"]
                    if job["phase"] == "SEED"
                    and job["scale"] == scale
                    and job["tariff_id"] == tariff_id
                ))
            )
            seed = json.loads(seed_path.read_text())
            tier1_routes, tier1_metrics = _aggregate(
                seed["routes"], tariff, problem
            )
            for route in tier1_routes:
                route["cost_tariff_sha256"] = tariff["sha256"]
            duty_routes = [{
                "route_id": route["duty_id"],
                "duty_id": route["duty_id"],
                "trips": route["trips"],
                "trip_blocks": _trip_blocks(problem, route["trips"]),
                "recorded_charging_blocks": [],
                "charging_stops": {
                    "stations": [], "cst": [], "cet": [], "kwh": [],
                },
            } for route in giro_routes[scale]]
            tier0_summary, tier0_events = _recorded_subset(
                master,
                {route["duty_id"] for route in giro_routes[scale]},
                duty_routes,
                tariff,
            )
            by_duty = defaultdict(list)
            for event in tier0_events:
                by_duty[event["duty_id"]].append({
                    "station": event["station"],
                    "start_min": event["start_min"],
                    "end_min": event["end_min"],
                    "realized_kwh": event["kwh"],
                    "expanded_grid_kwh": None,
                    "price_per_kwh": event["price_per_kwh"],
                    "tariff_hour": int(event["start_min"] // 60),
                    "tariff_key": None,
                })
            for route in duty_routes:
                route["recorded_charging_blocks"] = by_duty[
                    route["duty_id"]
                ]
            tier0_metrics = {
                "buses": len(duty_routes),
                "grid_model_objective":
                    tier0_summary["grid_model_objective"],
                "continuous_replay_objective": None,
                "charging_cost": tier0_summary["charging_cost"],
                "continuous_charging_cost": None,
                "total_charged_kwh":
                    tier0_summary["total_charged_kwh"],
                "peak_window_kwh": None,
                "charging_kwh_by_hour_json": None,
                "charging_kwh_by_station_json": None,
                "charging_starts_by_hour_json": None,
                "terminal_soc_min_kwh": None,
                "terminal_soc_max_kwh": None,
                "waiting_min": None,
                "deadhead_min": None,
                "deadhead_kwh": None,
                "charging_stops": tier0_summary["charging_starts"],
                "discretized_certification_status":
                    "not_applicable_recorded",
                "runtime_preprocessing_s": 0.0,
                "runtime_master_s": 0.0,
                "runtime_pricing_s": 0.0,
                "runtime_postprocessing_s": 0.0,
            }
            base = {
                "instance_id":
                    f"k{scale}-{plan['instances'][f'k{scale}']['sha256'][:12]}",
                "scale": scale,
                "tariff_id": tariff_id,
                "tariff_sha256": tariff["sha256"],
                "trip_ids": list(problem.trips),
                "terminal_soc_policy": PHYSICS["terminal_soc_policy"],
                "continuous_cost_pricing_certified": False,
            }
            cells.append({
                **base,
                "cell_id": f"k{scale}-{tariff_id}-tier0",
                "tier": "TIER0_GIRO_ORIGINAL",
                "treatment": "GIRO_ORIGINAL",
                "routes": duty_routes,
                "metrics": tier0_metrics,
                "terminal_soc_policy":
                    "unavailable_no_declared_operator_policy",
                "physical_replay_status":
                    "unavailable_recorded_power_profile_ambiguous",
                "certificate_scope": "none_recorded_schedule",
                "availability": tier0_summary[
                    "scalar_cost_availability"
                ],
                "availability_reason":
                    tier0_summary["availability_reason"],
                "source_artifacts": [
                    {
                        "role": "giro_master",
                        "path": str(master),
                        "sha256": sha256_file(master),
                    },
                    {
                        "role": "instance",
                        "path": str(next(
                            job["instance"]["path"] for job in plan["jobs"]
                            if not job["separate_k40_gate"]
                            and job["scale"] == scale
                        )),
                        "sha256":
                            plan["instances"][f"k{scale}"]["sha256"],
                    },
                ],
            })
            cells.append({
                **base,
                "cell_id": f"k{scale}-{tariff_id}-tier1",
                "tier": "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
                "treatment": "FIXED_GIRO",
                "routes": tier1_routes,
                "metrics": {
                    **tier1_metrics,
                    "discretized_certification_status":
                        "certified_all_fixed_duties",
                    "runtime_preprocessing_s": (
                        float(fixed_summary[(
                            tariff_id,
                            "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
                        )]["runtime_preprocessing_s"])
                        if scale == 40
                        else seed["runtime"]["preprocessing_s"]
                    ),
                    "runtime_master_s": (
                        float(fixed_summary[(
                            tariff_id,
                            "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
                        )]["runtime_master_s"])
                        if scale == 40 else seed["runtime"]["master_s"]
                    ),
                    "runtime_pricing_s": (
                        float(fixed_summary[(
                            tariff_id,
                            "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
                        )]["runtime_pricing_s"])
                        if scale == 40 else seed["runtime"]["pricing_s"]
                    ),
                    "runtime_postprocessing_s": (
                        float(fixed_summary[(
                            tariff_id,
                            "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
                        )]["runtime_postprocessing_s"])
                        if scale == 40
                        else seed["runtime"]["postprocessing_s"]
                    ),
                },
                "physical_replay_status": "validated_all_routes",
                "certificate_scope":
                    "optimal_discretized_charging_fixed_sequences",
                "fixed_duty_certificates": seed["certificates"],
                "source_artifacts": [{
                    "role": "tier1_seed",
                    "path": str(seed_path),
                    "sha256": sha256_file(seed_path),
                }],
            })
            if scale == 40:
                continue
            for treatment, tier in (
                ("RAW", "TIER2_RAW_ROUTE_CHARGING"),
                (
                    "GIRO-AUGMENTED",
                    "TIER2_GIRO_AUGMENTED_ROUTE_CHARGING",
                ),
            ):
                mip_job = next(
                    job for job in plan["jobs"]
                    if job["phase"] == "MIP"
                    and job["scale"] == scale
                    and job["tariff_id"] == tariff_id
                    and job["treatment"] == treatment
                )
                result = json.loads(Path(mip_job["output"]).read_text())
                if not result.get("incumbent_found"):
                    raise ValueError(f"MIP has no incumbent: {mip_job['job_key']}")
                selected, metrics = _aggregate(
                    result["selected_routes"], tariff, problem
                )
                cg_job = jobs[mip_job["dependency_key"]]
                cells.append({
                    **base,
                    "cell_id": f"k{scale}-{tariff_id}-{treatment}",
                    "tier": tier,
                    "treatment": treatment,
                    "routes": selected,
                    "metrics": {
                        **metrics,
                        "discretized_certification_status":
                            "finite_pool_" + str(result["optimal_scope"]),
                        "runtime_preprocessing_s": float(
                            result["physical_pool_preparation_wall_s"]
                        ),
                        "runtime_master_s": float(
                            result["gurobi_optimize_wall_s"]
                        ),
                        "runtime_pricing_s": float(
                            json.loads(Path(cg_job["output"]).read_text())[
                                "wall_s"
                            ]
                        ),
                        "runtime_postprocessing_s": max(
                            0.0,
                            float(result["end_to_end_before_publication_s"])
                            - float(result["gurobi_optimize_wall_s"])
                            - float(result[
                                "physical_pool_preparation_wall_s"
                            ]),
                        ),
                    },
                    "physical_replay_status": "validated_all_routes",
                    "certificate_scope":
                        "finite_augmented_pool"
                        if treatment != "RAW" else "finite_raw_pool",
                    "cg_iterations": _cg_iterations(cg_job),
                    "mip_checkpoints": _mip_checkpoints(mip_job),
                    "source_artifacts": [
                        {
                            "role": "cg_status",
                            "path": cg_job["output"],
                            "sha256": sha256_file(Path(cg_job["output"])),
                        },
                        {
                            "role": "mip_result",
                            "path": mip_job["output"],
                            "sha256": sha256_file(Path(mip_job["output"])),
                        },
                    ],
                })
            artifacts.append({
                "role": "tier1_seed",
                "path": str(seed_path),
                "sha256": sha256_file(seed_path),
            })
    experiment = {
        "schema": SCHEMA,
        "synthetic": False,
        "physics": PHYSICS,
        "tariff_manifest": plan["tariff_manifest"],
        "tariff_manifest_sha256": plan["tariff_manifest_sha256"],
        "giro40_duty_manifest": str(
            Path(plan["campaign_root"])
            / "outputs/fixed_full_giro40_all_tariffs"
            / "giro40_duty_manifest.csv"
        ),
        "giro40_duty_manifest_sha256": sha256_file(
            Path(plan["campaign_root"])
            / "outputs/fixed_full_giro40_all_tariffs"
            / "giro40_duty_manifest.csv"
        ),
        "cells": cells,
    }
    if output_manifest.exists():
        raise FileExistsError(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(experiment, indent=2, sort_keys=True) + "\n"
    )
    return build(output_manifest, evidence_output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(assemble(
        args.campaign_root, args.manifest_out, args.evidence_out
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

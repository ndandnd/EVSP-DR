#!/usr/bin/env python3
"""Normalize completed scale-ladder CG/MIP artifacts and deterministic plots."""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from build_tariff_response_manifest import sha256_file
from scale_ladder_trip_identity import (
    classify_legacy_trip_hash,
    identity,
)
from expanded_path_realization import (
    BLOCK_SCHEDULE_SCHEMA,
    charging_block_schedule_sha256,
)


CG_FIELDS = (
    "cell_id", "scale", "selection_replicate", "cg_replicate",
    "campaign_role", "soc_step", "block_min",
    "target_fleet", "elapsed_s", "iteration", "lp_obj",
    "route_weight", "artificial_mass", "min_reduced_cost",
    "pool_columns", "columns_added", "master_time_s", "pricing_time_s",
    "phase_timing_available", "phase_timing_unavailable_reason",
    "target_route_weight_observed", "grid_interpretation",
    "pricing_certified", "stopping_reason", "censored",
    "instance_file_sha256", "ordered_trip_id_set_sha256",
    "solver_local_trip_index_sha256", "ordered_trip_sequence_sha256",
    "trip_identity_schema", "legacy_trip_hash",
    "legacy_trip_hash_schema", "legacy_trip_hash_field",
)
CG_SUMMARY_FIELDS = (
    "cell_id", "scale", "selection_replicate", "cg_replicate",
    "campaign_role", "soc_step", "block_min",
    "target_fleet", "budget_s", "elapsed_s", "iterations",
    "final_lp_obj", "final_route_weight", "final_artificial_mass",
    "final_min_reduced_cost", "pool_columns", "pricing_certified",
    "stopping_reason", "time_to_target_route_weight_s",
    "time_to_zero_artificials_s", "censored",
    "phase_timing_available", "phase_timing_unavailable_reason",
    "target_route_weight_observed", "grid_interpretation",
    "instance_file_sha256", "ordered_trip_id_set_sha256",
    "solver_local_trip_index_sha256", "ordered_trip_sequence_sha256",
    "trip_identity_schema",
)
MIP_FIELDS = (
    "cell_id", "scale", "selection_replicate", "cg_replicate",
    "arm", "scientific_role", "checkpoint_elapsed_s",
    "incumbent_fleet", "statistics_incumbent_fleet", "fleet_bound",
    "fleet_gap", "node_count", "solution_count",
    "route_vector_sha256", "solver_ended_before_checkpoint",
    "instance_file_sha256", "trip_identity_schema",
)
MIP_SUMMARY_FIELDS = (
    "cell_id", "scale", "selection_replicate", "cg_replicate",
    "arm", "scientific_role", "budget_s", "output_available",
    "status_name", "incumbent_found", "buses", "fleet_bound",
    "mip_gap", "fleet_proven", "runtime_s", "optimal_scope",
    "physical_replay_status", "censored", "missing_reason",
    "source_result_sha256", "source_journal_sha256",
    "instance_file_sha256", "trip_identity_schema",
)
PROGRESS_FIELDS = (
    "scale", "target_fleet", "selection_replicate", "cg_replicate",
    "instance_file_sha256", "ordered_trip_id_set_sha256",
    "solver_local_trip_index_sha256", "ordered_trip_sequence_sha256",
    "trip_identity_schema", "cg_elapsed_s", "cg_iteration",
    "restricted_master_route_weight", "artificial_mass",
    "pricing_certified", "cg_stopping_reason", "cg_censored",
    "raw_mip_incumbent", "raw_finite_pool_bound", "raw_gap",
    "known_mip_incumbent", "known_finite_pool_bound", "known_gap",
    "known_arm_role", "physical_validation_status", "missing_reason",
    "known_partition_continuously_feasible",
    "known_partition_in_primary_expanded_space",
    "fixed_sequence_pricing_certified", "first_feasible_soc_step",
    "first_feasible_block_min", "nonrepresentability_reason",
    "target_gap", "target_gap_scope", "target_gap_interpretation",
)
INVENTORY_FIELDS = (
    "cell_id", "role", "path", "sha256", "size_bytes",
)
MEMBERSHIP_FIELDS = (
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


def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _phase_times(path):
    values = defaultdict(lambda: defaultdict(float))
    if not path.is_file():
        return values
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") != "phase":
                continue
            iteration = row.get("iteration")
            if iteration is None:
                continue
            phase = row.get("phase")
            if phase == "master_attempt":
                values[int(iteration)]["master"] += float(row["duration_s"])
            elif str(phase).startswith("pricing_"):
                values[int(iteration)]["pricing"] += float(row["duration_s"])
    return values


def _checkpoint_rows(job, identity_fields):
    rows = []
    progress = Path(job["progress_dir"])
    for path in sorted(progress.glob("checkpoint_*.json")):
        payload = json.loads(path.read_text())
        incumbent = payload.get("incumbent") or {}
        stats = payload.get("latest_statistics") or {}
        rows.append({
            "cell_id": job["cell_id"],
            "scale": job["scale"],
            "selection_replicate": job["selection_replicate"],
            "cg_replicate": job["cg_replicate"],
            "arm": job["arm"],
            "scientific_role": job.get("scientific_role"),
            "checkpoint_elapsed_s": payload.get("checkpoint_elapsed_s"),
            "incumbent_fleet": incumbent.get("fleet"),
            "statistics_incumbent_fleet":
                stats.get("statistics_incumbent_fleet"),
            "fleet_bound": stats.get("fleet_bound"),
            "fleet_gap": stats.get("fleet_gap"),
            "node_count": stats.get("node_count"),
            "solution_count": stats.get("solution_count"),
            "route_vector_sha256": incumbent.get("route_vector_sha256"),
            "solver_ended_before_checkpoint":
                payload.get("solver_ended_before_checkpoint"),
            "instance_file_sha256":
                identity_fields["instance_file_sha256"],
            "trip_identity_schema":
                identity_fields["trip_identity_schema"],
        })
    return rows


SCIENCE_GROUPS = (
    "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
    "MIP_RAW", "MIP_KNOWN",
)
TERMINAL_STATES = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED",
    "SPECIAL_EXIT", "TIMEOUT",
}


def _require_scale_ladder_scheduler_evidence(plan, manifest, plan_sha):
    gate = str(manifest.get("gate_job_id") or "")
    arrays = manifest.get("submitted_arrays") or {}
    if (
        manifest.get("submitted") is not True
        or manifest.get("gate_state") != "released_reconciled"
        or not gate.isdigit()
        or set(arrays) != set(SCIENCE_GROUPS)
        or any(not str(value).isdigit() for value in arrays.values())
    ):
        raise ValueError(
            "campaign lacks a completed verified scheduler contract"
        )
    expected_gate = {
        "job_id": gate,
        "job_name": f"LDG{plan_sha[:5]}",
        "partition": "default_partition",
        "comment": f"SLADG:{plan_sha[:20]}",
    }

    def require_gate_observation(observation, *, terminal):
        if not isinstance(observation, dict) or any(
            str(observation.get(field) or "") != expected
            for field, expected in expected_gate.items()
        ):
            raise ValueError("scientific gate observation identity mismatch")
        state = str(observation.get("state") or "")
        if terminal:
            if (
                state != "COMPLETED"
                or observation.get("exit_code") != "0:0"
                or observation.get("source") not in {"scontrol", "sacct"}
            ):
                raise ValueError(
                    "scientific gate lacks exact COMPLETED/0:0 evidence"
                )
        elif (
            state == "PENDING"
            and observation.get("reason") == "JobHeldUser"
        ):
            raise ValueError(
                "scientific gate release observation remains user-held"
            )
        elif state not in (
            {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING"}
            | TERMINAL_STATES
        ):
            raise ValueError("scientific gate release state is invalid")

    release = manifest.get("gate_release_verification")
    if (
        not isinstance(release, dict)
        or release.get("verified") is not True
        or str(release.get("job_id") or "") != gate
    ):
        raise ValueError("scientific gate release evidence is missing")
    require_gate_observation(release.get("observation"), terminal=False)
    reconciliation = manifest.get("gate_reconciliation")
    if (
        not isinstance(reconciliation, dict)
        or reconciliation.get("verified") is not True
        or str(reconciliation.get("gate_job_id") or "") != gate
    ):
        raise ValueError("scientific gate completion evidence is missing")
    require_gate_observation(
        reconciliation.get("observation"), terminal=True
    )

    verifications = manifest.get("array_submission_verifications") or {}
    if set(verifications) != set(SCIENCE_GROUPS):
        raise ValueError("scientific array receipt set is incomplete")
    user = str((plan.get("runtime_environment") or {}).get("USER") or "")
    if not user:
        raise ValueError("approved scheduler user is missing")
    name_prefixes = {
        "PREFLIGHT": "LDPF",
        "SEED": "LDSD",
        "CG": "LDCG",
        "CG_SENSITIVITY": "LDCS",
        "MIP_RAW": "LDMR",
        "MIP_KNOWN": "LDMK",
    }
    for group in SCIENCE_GROUPS:
        verification = verifications[group]
        expected_partition = (
            "scaglione" if group.startswith("MIP")
            else "default_partition"
        )
        expected_dependencies = {"afterok": [gate]}
        if group in {"CG", "CG_SENSITIVITY"}:
            expected_dependencies["afterok"].append(
                f"{arrays['PREFLIGHT']}_*"
            )
        elif group == "MIP_RAW":
            expected_dependencies["aftercorr"] = [
                f"{arrays['CG']}_*"
            ]
        elif group == "MIP_KNOWN":
            expected_dependencies["aftercorr"] = [
                f"{arrays['CG']}_*", f"{arrays['SEED']}_*",
            ]
        expected_dependencies = {
            kind: sorted(values)
            for kind, values in expected_dependencies.items()
        }
        if (
            not isinstance(verification, dict)
            or verification.get("verified") is not True
            or verification.get("group") != group
            or str(verification.get("parent_job_id") or "")
            != str(arrays[group])
            or str(verification.get("gate_job_id") or "") != gate
            or verification.get("user") != user
            or verification.get("job_name")
            != name_prefixes[group] + plan_sha[:4]
            or verification.get("partition") != expected_partition
            or verification.get("comment")
            != f"SLAD:{plan_sha[:20]}:{group}"
            or verification.get("state") != "PENDING"
            or verification.get("reason") != "Dependency"
            or verification.get("task_ids")
            != list(range(len(plan["task_groups"][group])))
            or verification.get("dependency_semantics")
            != expected_dependencies
        ):
            raise ValueError(
                f"scientific array receipt is invalid: {group}"
            )


def summarize(campaign_root, output_dir, k40_reuse_manifest=None):
    campaign_root = Path(campaign_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    plan_raw = (campaign_root / "approved-plan.json").read_bytes()
    plan = json.loads(plan_raw)
    manifest = json.loads((campaign_root / "campaign.json").read_text())
    if manifest.get("approval_sha256") != hashlib.sha256(
        plan_raw
    ).hexdigest():
        raise ValueError("campaign approval hash mismatch")
    local_diagnostic = plan.get("execution_mode") == "local_diagnostic"
    if local_diagnostic:
        if (
            manifest.get("execution_mode") != "local_diagnostic"
            or not isinstance(manifest.get("diagnostic_only"), bool)
            or manifest.get("submitted") is not False
        ):
            raise ValueError("local diagnostic manifest is inconsistent")
    else:
        _require_scale_ladder_scheduler_evidence(
            plan,
            manifest,
            hashlib.sha256(plan_raw).hexdigest(),
        )
    if not local_diagnostic:
        probes = manifest.get("probe_results") or {}
        if (
            manifest.get("probe_state") != "passed"
            or set(probes) != {"default_partition", "scaglione"}
        ):
            raise ValueError("infrastructure probes are incomplete")
        for probe_id, result in probes.items():
            path = Path(result.get("output") or "")
            payload = json.loads(path.read_text())
            sidecar = Path(str(path) + ".sha256")
            sidecar_hash = (
                sidecar.read_text().split()[0]
                if sidecar.is_file() and sidecar.read_text().split()
                else None
            )
            if (
                result.get("compatible") is not True
                or payload.get("compatible") is not True
                or payload.get("plan_sha256")
                != manifest["approval_sha256"]
                or payload.get("probe_id") not in {
                    "default", "scaglione"
                }
                or sha256_file(path) != sidecar_hash
                or payload.get("observed_portable_identity_sha256")
                != plan["python_identity"][
                    "portable_identity_sha256"
                ]
            ):
                raise ValueError(
                    f"infrastructure probe invalid: {probe_id}"
                )
    jobs = {job["job_key"]: job for job in plan["jobs"]}
    for job in plan["jobs"]:
        _validate_completion(job, manifest["approval_sha256"])
    membership_rows = []
    membership_by_instance = {}
    for job in plan["jobs"]:
        membership_path = (
            Path(job["output"])
            if job["phase"] == "PREFLIGHT"
            else Path(job["membership_output"])
            if local_diagnostic
            and job["phase"] == "SEED"
            and job.get("membership_output")
            else None
        )
        if membership_path is None:
            continue
        payload = json.loads(membership_path.read_text())
        if payload.get("schema") != (
            "evsp-dr-scale-ladder-known-membership-v1"
        ):
            raise ValueError("membership preflight schema mismatch")
        membership_by_instance[(
            int(payload["scale"]), int(payload["selection_replicate"])
        )] = payload
        membership_rows.extend(payload["duties"])
    cg_jobs = [
        job for job in plan["jobs"]
        if job["phase"] in {"CG", "CG_SENSITIVITY"}
    ]
    cg_rows = []
    cg_summary = []
    mip_rows = []
    mip_summary = []
    inventory = []
    if not local_diagnostic:
        for probe_id, result in sorted(
            (manifest.get("probe_results") or {}).items()
        ):
            path = Path(result["output"])
            inventory.append(_inventory(
                "", f"environment_probe_{probe_id}", path
            ))
            inventory.append(_inventory(
                "", f"environment_probe_{probe_id}_sha256",
                Path(str(path) + ".sha256"),
            ))
    for job in plan["jobs"]:
        membership_path = (
            Path(job["output"])
            if job["phase"] == "PREFLIGHT"
            else Path(job["membership_output"])
            if local_diagnostic
            and job["phase"] == "SEED"
            and job.get("membership_output")
            else None
        )
        if membership_path is not None:
            inventory.append(_inventory(
                job["cell_id"], "known_membership", membership_path
            ))
            inventory.append(_inventory(
                job["cell_id"], "known_membership_csv",
                membership_path.with_suffix(".csv"),
            ))
    cg_by_cell = {}
    for job in cg_jobs:
        status_path = Path(job["output"])
        if not status_path.is_file():
            raise ValueError(f"CG output missing: {job['cell_id']}")
        status = json.loads(status_path.read_text())
        identities = identity(Path(job["instance"]["path"]))
        for field in identities:
            if field in job["instance"] and str(
                job["instance"][field]
            ) != str(identities[field]):
                raise ValueError(f"CG instance identity differs: {field}")
        provenance = status.get("provenance") or {}
        if (
            provenance.get("instance_sha256")
            != identities["instance_file_sha256"]
            or provenance.get("prices_sha256")
            != plan["tariff"]["primary_tariff_sha256"]
            or float(status.get("soc_step", math.nan))
            != float(job["soc_step"])
            or int(status.get("block_min", -1))
            != int(job["block_min"])
            or status.get("g_kwh") != 300.0
            or status.get("charge_kw") != 300.0
            or status.get("min_soc_frac") != 0.0
        ):
            raise ValueError(f"CG source/model identity differs: {job['cell_id']}")
        legacy = classify_legacy_trip_hash(
            status.get("trip_set_sha256"), identities
        )
        phases = (
            _phase_times(Path(job["telemetry"]))
            if job.get("telemetry") else defaultdict(lambda: defaultdict(float))
        )
        iters_path = Path(str(status_path) + ".iters.csv")
        with iters_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        previous_columns = 0
        normalized = []
        for raw in rows:
            iteration = int(float(raw["iteration"]))
            pool_columns = int(float(raw["pool_columns"]))
            timing_available = (
                job.get("telemetry") is not None
                and phases[iteration]["master"] > 0.0
                and phases[iteration]["pricing"] > 0.0
            )
            normalized.append({
                "cell_id": job["cell_id"],
                "scale": job["scale"],
                "selection_replicate": job["selection_replicate"],
                "cg_replicate": job["cg_replicate"],
                "campaign_role": (
                    "primary" if job["phase"] == "CG"
                    else "small_grid_sensitivity"
                ),
                "soc_step": job["soc_step"],
                "block_min": job["block_min"],
                "target_fleet": job["target_fleet"],
                "elapsed_s": float(raw["elapsed_s"]),
                "iteration": iteration,
                "lp_obj": float(raw["lp_obj"]),
                "route_weight": float(raw["route_weight"]),
                "artificial_mass": float(raw["artificials"]),
                "min_reduced_cost": float(raw["min_rc"]),
                "pool_columns": pool_columns,
                "columns_added": max(0, pool_columns - previous_columns),
                "master_time_s": phases[iteration]["master"] or None,
                "pricing_time_s": phases[iteration]["pricing"] or None,
                "phase_timing_available": timing_available,
                "phase_timing_unavailable_reason": (
                    None if timing_available
                    else "telemetry_disabled_for_long_run"
                    if job.get("telemetry") is None
                    else "phase_event_missing"
                ),
                "target_route_weight_observed": (
                    float(raw["artificials"]) <= 1e-9
                    and float(raw["route_weight"])
                    <= float(job["target_fleet"]) + 1e-6
                ),
                "grid_interpretation": (
                    "primary_discretized_route_space"
                    if job["phase"] == "CG"
                    else
                    "known_duties_contained_fallback_grid"
                    if job["soc_step"] == 1.0
                    and job["block_min"] == 5
                    else "route_space_sensitivity_diagnostic"
                ),
                "pricing_certified": (
                    status.get("certified_rc_optimal") is True
                    and raw is rows[-1]
                ),
                "stopping_reason": status.get("stop_reason"),
                "censored": status.get("certified_rc_optimal") is not True,
                **{
                    field: identities[field] for field in identities
                    if field != "trip_count"
                },
                **legacy,
            })
            previous_columns = pool_columns
        cg_rows.extend(normalized)
        if job.get("telemetry") and any(
            row["phase_timing_available"] is not True
            for row in normalized
        ):
            raise ValueError(
                f"configured phase telemetry is incomplete: {job['cell_id']}"
            )
        final = normalized[-1] if normalized else None
        time_target = next((
            row["elapsed_s"] for row in normalized
            if row["artificial_mass"] <= 1e-9
            and row["route_weight"] <= job["target_fleet"] + 1e-6
        ), None)
        time_zero = next((
            row["elapsed_s"] for row in normalized
            if row["artificial_mass"] <= 1e-9
        ), None)
        summary = {
            "cell_id": job["cell_id"],
            "scale": job["scale"],
            "selection_replicate": job["selection_replicate"],
            "cg_replicate": job["cg_replicate"],
            "campaign_role": (
                "primary" if job["phase"] == "CG"
                else "small_grid_sensitivity"
            ),
            "soc_step": job["soc_step"],
            "block_min": job["block_min"],
            "target_fleet": job["target_fleet"],
            "budget_s": job["budget_s"],
            "elapsed_s": status.get("wall_s"),
            "iterations": status.get("iterations"),
            "final_lp_obj": final["lp_obj"] if final else None,
            "final_route_weight": final["route_weight"] if final else None,
            "final_artificial_mass":
                final["artificial_mass"] if final else None,
            "final_min_reduced_cost":
                final["min_reduced_cost"] if final else None,
            "pool_columns": status.get("columns"),
            "pricing_certified": status.get("certified_rc_optimal"),
            "stopping_reason": status.get("stop_reason"),
            "time_to_target_route_weight_s": time_target,
            "time_to_zero_artificials_s": time_zero,
            "censored": status.get("certified_rc_optimal") is not True,
            "phase_timing_available": (
                bool(normalized)
                and all(row["phase_timing_available"] for row in normalized)
            ),
            "phase_timing_unavailable_reason": (
                None if normalized and all(
                    row["phase_timing_available"] for row in normalized
                )
                else "telemetry_disabled_for_long_run"
                if job.get("telemetry") is None
                else "phase_event_missing"
            ),
            "target_route_weight_observed": (
                final is not None
                and final["artificial_mass"] <= 1e-9
                and final["route_weight"]
                <= float(job["target_fleet"]) + 1e-6
            ),
            "grid_interpretation": (
                "primary_discretized_route_space"
                if job["phase"] == "CG"
                else
                "known_duties_contained_fallback_grid"
                if job["soc_step"] == 1.0
                and job["block_min"] == 5
                else "route_space_sensitivity_diagnostic"
            ),
            **{
                field: identities[field] for field in identities
                if field != "trip_count"
            },
        }
        cg_summary.append(summary)
        if job["phase"] == "CG":
            cg_by_cell[job["cell_id"]] = summary
        for role, path in (
            ("cg_status", status_path),
            ("cg_journal", Path(status["columns_journal"])),
            ("cg_iterations", iters_path),
        ):
            inventory.append(_inventory(job["cell_id"], role, path))
        if job.get("telemetry"):
            inventory.append(_inventory(
                job["cell_id"], "cg_telemetry", Path(job["telemetry"])
            ))
        for snapshot in sorted(status_path.parent.glob(
            f"{status_path.stem}.m*.snapshot.json"
        )):
            inventory.append(_inventory(
                job["cell_id"], "cg_snapshot_status", snapshot
            ))
            inventory.append(_inventory(
                job["cell_id"], "cg_snapshot_journal",
                Path(str(snapshot) + ".columns.jsonl"),
            ))
    for job in plan["jobs"]:
        if job["phase"] != "MIP":
            continue
        identities = identity(Path(job["instance"]["path"]))
        result_path = Path(job["output"])
        result = json.loads(result_path.read_text())
        dependency = jobs[job["dependency_cg"]]
        cg_status_path = Path(dependency["output"])
        cg_status = json.loads(cg_status_path.read_text())
        cg_journal = Path(cg_status["columns_journal"])
        if (
            result.get("source_result_sha256")
            != sha256_file(cg_status_path)
            or result.get("source_journal_sha256")
            != sha256_file(cg_journal)
            or (result.get("physics") or {}).get("g_kwh") != 300.0
            or (result.get("physics") or {}).get("charge_kw") != 300.0
            or (result.get("physics") or {}).get("min_soc_frac") != 0.0
        ):
            raise ValueError(f"MIP source/model identity differs: {job['job_key']}")
        start = result.get("mip_start") or {}
        expected_experiment_arm = (
            "B" if job["arm"] == "RAW" else "D"
        )
        if (
            result.get("experiment_arm") != expected_experiment_arm
            or (
                job["arm"] == "RAW"
                and (
                    start.get("source") is not None
                    or start.get("kind") == "validated_exact_partition"
                )
            )
            or (
                job["arm"] == "KNOWN-PARTITION"
                and (
                    start.get("kind") != "validated_exact_partition"
                    or start.get("source_sha256")
                    != sha256_file(Path(
                        jobs[job["dependency_seed"]]["output"]
                    ))
                )
            )
        ):
            raise ValueError(f"MIP arm/start identity differs: {job['job_key']}")
        mip_rows.extend(_checkpoint_rows(job, identities))
        mip_summary.append({
            "cell_id": job["cell_id"],
            "scale": job["scale"],
            "selection_replicate": job["selection_replicate"],
            "cg_replicate": job["cg_replicate"],
            "arm": job["arm"],
            "scientific_role": job.get("scientific_role"),
            "budget_s": job["budget_s"],
            "output_available": True,
            "status_name": result.get("status_name"),
            "incumbent_found": result.get("incumbent_found"),
            "buses": result.get("buses"),
            "fleet_bound": result.get("fleet_bound"),
            "mip_gap": result.get("mip_gap"),
            "fleet_proven": result.get("fleet_proven"),
            "runtime_s": result.get("runtime_s"),
            "optimal_scope": result.get("optimal_scope"),
            "physical_replay_status": (
                "validated" if result.get("incumbent_found") else None
            ),
            "censored": result.get("fleet_proven") is not True,
            "missing_reason": None,
            "source_result_sha256": result.get("source_result_sha256"),
            "source_journal_sha256": result.get("source_journal_sha256"),
            "instance_file_sha256":
                identities["instance_file_sha256"],
            "trip_identity_schema": identities["trip_identity_schema"],
        })
        inventory.append(_inventory(job["cell_id"], "mip_result", result_path))
        for path in sorted(Path(job["progress_dir"]).glob("*.json")):
            inventory.append(_inventory(
                job["cell_id"], "mip_checkpoint", path
            ))
    _append_k40_rows(
        plan, manifest["approval_sha256"], k40_reuse_manifest,
        mip_summary, mip_rows, inventory
    )
    progress = _progress_rows(
        plan, cg_by_cell, mip_summary, membership_by_instance
    )
    staging = Path(tempfile.mkdtemp(
        dir=output_dir.parent, prefix=f".{output_dir.name}.tmp."
    ))
    try:
        _write_csv(staging / "cg_iteration_long.csv", CG_FIELDS, cg_rows)
        _write_csv(staging / "cg_run_summary.csv",
                   CG_SUMMARY_FIELDS, cg_summary)
        _write_csv(staging / "mip_checkpoint_long.csv",
                   MIP_FIELDS, mip_rows)
        _write_csv(staging / "mip_run_summary.csv",
                   MIP_SUMMARY_FIELDS, mip_summary)
        _write_csv(staging / "artifact_inventory.csv",
                   INVENTORY_FIELDS, inventory)
        _write_csv(staging / "scale_progress_summary.csv",
                   PROGRESS_FIELDS, progress)
        _write_csv(
            staging / "known_route_membership_long.csv",
            MEMBERSHIP_FIELDS,
            sorted(membership_rows, key=lambda row: (
                int(row["scale"]), int(row["selection_replicate"]),
                row["duty_id"],
            )),
        )
        _plots(staging, cg_rows, mip_rows)
        provenance = {
            "schema": "evsp-dr-scale-ladder-summary-v1",
            "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
            "git_commit": plan["checkout_identity"]["commit"],
            "tariff": plan["tariff"],
            "physics": plan["physics"],
            "trip_identity_schema": plan["trip_identity_schema"],
            "code_hashes": plan["code_hashes"],
            "python_identity": plan["python_identity"],
            "execution_mode": plan.get("execution_mode", "slurm_campaign"),
            "diagnostic_only": manifest.get("diagnostic_only", False),
            "resource_groups": {
                key: len(value)
                for key, value in plan["task_groups"].items()
            },
            "output_sha256": {
                path.name: sha256_file(path)
                for path in staging.iterdir() if path.is_file()
            },
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        _rename_noreplace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "cg_cells": len(cg_summary),
        "mip_rows": len(mip_summary),
        "output": str(output_dir),
    }


def _append_k40_rows(
    plan, plan_sha, reuse_manifest, summaries, checkpoints, inventory
):
    supplied = {}
    if reuse_manifest:
        path = Path(reuse_manifest).resolve()
        payload = json.loads(path.read_text())
        if payload.get("schema") != "evsp-dr-scale-ladder-k40-reuse-v1":
            raise ValueError("k40 reuse manifest schema mismatch")
        if payload.get("approved_plan_sha256") != plan_sha:
            raise ValueError("k40 reuse manifest belongs to another plan")
        rows = payload.get("slots") or []
        supplied = {
            (row["cell_id"], row["arm"]): row for row in rows
        }
        result_paths = [str(row.get("result_path")) for row in rows]
        if (
            len(supplied) != len(rows)
            or len(set(result_paths)) != len(result_paths)
        ):
            raise ValueError("k40 reuse slots/results are duplicated")
        inventory.append(_inventory("", "k40_reuse_manifest", path))
    jobs_by_key = {job["job_key"]: job for job in plan["jobs"]}
    for slot in plan["k40_reuse_slots"]:
        key = (slot["cell_id"], slot["arm"])
        row = supplied.get(key)
        missing = "k40_reuse_not_supplied"
        result = None
        if row:
            result_path = Path(row["result_path"])
            try:
                if (
                    not result_path.is_file()
                    or sha256_file(result_path) != row["result_sha256"]
                ):
                    raise ValueError("result path/hash mismatch")
                candidate = json.loads(result_path.read_text())
                completion_path = Path(row["producer_completion_path"])
                if (
                    not completion_path.is_file()
                    or sha256_file(completion_path)
                    != row["producer_completion_sha256"]
                ):
                    raise ValueError("producer completion mismatch")
                completion = json.loads(completion_path.read_text())
                if (
                    completion.get("schema")
                    not in {
                        "evsp-dr-mip-worker-completion-v2",
                        "evsp-dr-scale-ladder-worker-completion-v1",
                    }
                    or completion.get("phase") not in {None, "MIP"}
                ):
                    raise ValueError("producer completion schema mismatch")
                if completion["schema"] == (
                    "evsp-dr-mip-worker-completion-v2"
                ):
                    progress_hashes = completion.get(
                        "progress_artifact_sha256"
                    ) or {}
                    expected_completion_arm = (
                        "RAW" if slot["arm"] == "RAW"
                        else "GIRO40-AUGMENTED"
                    )
                    if (
                        completion.get("result_sha256")
                        != row["result_sha256"]
                        or completion.get("source_result_sha256")
                        != candidate.get("source_result_sha256")
                        or completion.get("source_journal_sha256")
                        != candidate.get("source_journal_sha256")
                        or completion.get("arm") != expected_completion_arm
                        or completion.get(
                            "result_and_progress_validation_passed"
                        ) is not True
                        or not progress_hashes
                    ):
                        raise ValueError("producer completion omits result")
                    for artifact, digest in progress_hashes.items():
                        artifact_path = Path(artifact)
                        if not artifact_path.is_absolute():
                            artifact_path = completion_path.parent / artifact_path
                        if (
                            not artifact_path.is_file()
                            or sha256_file(artifact_path) != digest
                        ):
                            raise ValueError(
                                "producer progress artifact changed"
                            )
                    progress_set = hashlib.sha256(json.dumps(
                        progress_hashes,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()).hexdigest()
                    if completion.get("progress_set_sha256") != progress_set:
                        raise ValueError("producer progress set mismatch")
                else:
                    completion_hashes = completion.get(
                        "artifact_sha256"
                    ) or {}
                    if completion_hashes.get(
                        str(result_path.resolve())
                    ) != row["result_sha256"]:
                        raise ValueError("producer completion omits result")
                cg_job = jobs_by_key[slot["required_cg_job_key"]]
                cg_status_path = Path(cg_job["output"])
                cg_status = json.loads(cg_status_path.read_text())
                cg_journal = Path(cg_status["columns_journal"])
                expected_experiment_arm = (
                    "B" if slot["arm"] == "RAW" else "D"
                )
                start = candidate.get("mip_start") or {}
                physical = candidate.get("physical_pool_audit") or {}
                provenance = candidate.get("pricer_provenance") or {}
                mip_provenance = candidate.get("mip_provenance") or {}
                arguments = mip_provenance.get("arguments") or {}
                gurobi_parameters = mip_provenance.get(
                    "gurobi_parameters"
                ) or {}
                known_partition_sha = None
                if slot["arm"] == "KNOWN-PARTITION":
                    known_path = Path(row["known_partition_path"])
                    if (
                        not known_path.is_file()
                        or sha256_file(known_path)
                        != row["known_partition_sha256"]
                        or row["known_partition_sha256"]
                        != slot["required_known_partition_sha256"]
                    ):
                        raise ValueError("known partition artifact mismatch")
                    known_partition_sha = row["known_partition_sha256"]
                selected_routes = candidate.get("selected_routes") or []
                selected_trip_counts = defaultdict(int)
                for selected_route in selected_routes:
                    for trip in selected_route.get("trips") or []:
                        selected_trip_counts[int(trip)] += 1
                valid_selected_partition = (
                    candidate.get("incumbent_found") is not True
                    or (
                        len(selected_routes) == candidate.get("buses")
                        and set(selected_trip_counts) == set(range(947))
                        and all(
                            value == 1
                            for value in selected_trip_counts.values()
                        )
                    )
                )
                valid_selected_physical = all(
                    isinstance(
                        route.get("continuous_realized_charging_blocks"),
                        list,
                    )
                    and charging_block_schedule_sha256(
                        route["continuous_realized_charging_blocks"]
                    ) == (
                        route.get("physical_realization") or {}
                    ).get(
                        "continuous_realized_charging_blocks_sha256"
                    )
                    and (
                        route.get("physical_realization") or {}
                    ).get(
                        "continuous_realized_charging_blocks_schema"
                    ) == BLOCK_SCHEDULE_SCHEMA
                    for route in selected_routes
                )
                if (
                    candidate.get("partitioning") is not True
                    or candidate.get("experiment_arm")
                    != expected_experiment_arm
                    or candidate.get("source_result_sha256")
                    != sha256_file(cg_status_path)
                    or candidate.get("source_journal_sha256")
                    != sha256_file(cg_journal)
                    or provenance.get("instance_sha256")
                    != slot["required_instance_file_sha256"]
                    or provenance.get("prices_sha256")
                    != slot["required_tariff_sha256"]
                    or (physical.get("input_hashes") or {}).get(
                        "instance_sha256"
                    ) != slot["required_instance_file_sha256"]
                    or (physical.get("input_hashes") or {}).get(
                        "prices_sha256"
                    ) != slot["required_tariff_sha256"]
                    or (physical.get("input_hashes") or {}).get(
                        "reference_sha256"
                    ) != slot["required_reference_sha256"]
                    or (physical.get("input_hashes") or {}).get(
                        "deadhead_sha256"
                    ) != slot["required_deadhead_sha256"]
                    or int(physical.get("rejected_columns", -1)) != 0
                    or int(physical.get("accepted_columns", -1))
                    != int(physical.get("total_columns", -2))
                    or (candidate.get("physics") or {}).get("g_kwh")
                    != 300.0
                    or (candidate.get("physics") or {}).get("charge_kw")
                    != 300.0
                    or (candidate.get("physics") or {}).get("min_soc_frac")
                    != 0.0
                    or mip_provenance.get("observed_git_commit")
                    != row["producer_commit"]
                    or mip_provenance.get("expected_git_commit")
                    != row["producer_commit"]
                    or mip_provenance.get("git_dirty") is not False
                    or mip_provenance.get("tracked_clean_at_end") is not True
                    or mip_provenance.get("final_observed_git_commit")
                    != row["producer_commit"]
                    or row["producer_commit"]
                    not in slot["accepted_producer_commits"]
                    or arguments.get("two_stage") is not True
                    or arguments.get("cover") is not False
                    or int(arguments.get("threads", -1))
                    != slot["required_threads"]
                    or int(arguments.get("timelimit", -1))
                    != slot["required_time_limit_s"]
                    or not math.isclose(
                        float(arguments.get("mipgap", math.nan)),
                        slot["required_mip_gap"],
                        rel_tol=0.0, abs_tol=1e-12,
                    )
                    or int(gurobi_parameters.get("Seed", -1))
                    != slot["required_gurobi_seed"]
                    or not valid_selected_partition
                    or not valid_selected_physical
                    or (
                        slot["arm"] == "RAW"
                        and (
                            start.get("source") is not None
                            or start.get("kind")
                            == "validated_exact_partition"
                        )
                    )
                    or (
                        slot["arm"] == "KNOWN-PARTITION"
                        and (
                            start.get("kind")
                            != "validated_exact_partition"
                            or start.get("source_sha256")
                            != known_partition_sha
                            or start.get("validated_bus_count") != 40
                            or (start.get("solver_acceptance") or {}).get(
                                "accepted"
                            ) is not True
                        )
                    )
                    or (
                        slot["arm"] == "RAW"
                        and (
                            candidate.get("extra_route_sources") != []
                            or int(physical.get(
                                "added_giro_route_count", -1
                            )) != 0
                            or physical.get("base_pool_column_count")
                            != physical.get("augmented_pool_column_count")
                        )
                    )
                    or (
                        slot["arm"] == "KNOWN-PARTITION"
                        and (
                            int(physical.get(
                                "added_giro_route_count", -1
                            )) != 40
                            or int(physical.get(
                                "assigned_mip_start_route_count", -1
                            )) != 40
                            or int(physical.get(
                                "augmented_pool_column_count", -1
                            )) != int(physical.get(
                                "base_pool_column_count", -2
                            )) + 40
                        )
                    )
                    or (
                        candidate.get("incumbent_found") is True
                        and any(
                            not isinstance(
                                route.get(
                                    "continuous_realized_charging_blocks"
                                ), list
                            )
                            for route in candidate.get("selected_routes") or []
                        )
                    )
                ):
                    raise ValueError("result provenance/arm mismatch")
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                missing = "k40_reuse_identity_mismatch"
            else:
                result = candidate
                missing = None
                inventory.append(_inventory(
                    slot["cell_id"], "k40_reused_mip", result_path
                ))
        summaries.append({
            "cell_id": slot["cell_id"],
            "scale": 40,
            "selection_replicate": 2,
            "cg_replicate": slot["cg_replicate"],
            "arm": slot["arm"],
            "scientific_role": (
                "feasibility_integral_assembly_diagnostic_not_algorithmic_recovery"
                if slot["arm"] == "KNOWN-PARTITION" else None
            ),
            "budget_s": None,
            "output_available": result is not None,
            "status_name": (result or {}).get("status_name"),
            "incumbent_found": (result or {}).get("incumbent_found"),
            "buses": (result or {}).get("buses"),
            "fleet_bound": (result or {}).get("fleet_bound"),
            "mip_gap": (result or {}).get("mip_gap"),
            "fleet_proven": (result or {}).get("fleet_proven"),
            "runtime_s": (result or {}).get("runtime_s"),
            "optimal_scope": (result or {}).get("optimal_scope"),
            "physical_replay_status": (
                "validated" if (result or {}).get("incumbent_found")
                else None
            ),
            "censored": True,
            "missing_reason": missing,
            "source_result_sha256":
                (result or {}).get("source_result_sha256"),
            "source_journal_sha256":
                (result or {}).get("source_journal_sha256"),
            "instance_file_sha256":
                slot["required_instance_file_sha256"],
            "trip_identity_schema": plan["trip_identity_schema"],
        })


def _progress_rows(plan, cg_by_cell, mip_summary, membership_by_instance):
    mip = {
        (row["cell_id"], row["arm"]): row for row in mip_summary
    }
    rows = []
    for job in plan["jobs"]:
        if job["phase"] != "CG":
            continue
        cg = cg_by_cell[job["cell_id"]]
        raw = mip.get((job["cell_id"], "RAW"), {})
        known = mip.get((job["cell_id"], "KNOWN-PARTITION"), {})
        membership = membership_by_instance[(
            int(job["scale"]), int(job["selection_replicate"])
        )]
        reasons = sorted({
            value for value in (
                raw.get("missing_reason"), known.get("missing_reason")
            ) if value
        })
        rows.append({
            "scale": job["scale"],
            "target_fleet": job["target_fleet"],
            "selection_replicate": job["selection_replicate"],
            "cg_replicate": job["cg_replicate"],
            "instance_file_sha256":
                job["instance"]["instance_file_sha256"],
            "ordered_trip_id_set_sha256":
                job["instance"]["ordered_trip_id_set_sha256"],
            "solver_local_trip_index_sha256":
                job["instance"]["solver_local_trip_index_sha256"],
            "ordered_trip_sequence_sha256":
                job["instance"]["ordered_trip_sequence_sha256"],
            "trip_identity_schema": job["instance"]["trip_identity_schema"],
            "cg_elapsed_s": cg["elapsed_s"],
            "cg_iteration": cg["iterations"],
            "restricted_master_route_weight": cg["final_route_weight"],
            "artificial_mass": cg["final_artificial_mass"],
            "pricing_certified": cg["pricing_certified"],
            "cg_stopping_reason": cg["stopping_reason"],
            "cg_censored": cg["censored"],
            "raw_mip_incumbent": raw.get("buses"),
            "raw_finite_pool_bound": raw.get("fleet_bound"),
            "raw_gap": raw.get("mip_gap"),
            "known_mip_incumbent": known.get("buses"),
            "known_finite_pool_bound": known.get("fleet_bound"),
            "known_gap": known.get("mip_gap"),
            "known_arm_role":
                "feasibility_integral_assembly_diagnostic_not_algorithmic_recovery",
            "physical_validation_status": (
                raw.get("physical_replay_status")
                or known.get("physical_replay_status")
            ),
            "missing_reason": ";".join(reasons),
            "known_partition_continuously_feasible":
                membership["known_partition_continuously_feasible"],
            "known_partition_in_primary_expanded_space":
                membership["known_partition_in_primary_expanded_space"],
            "fixed_sequence_pricing_certified":
                membership["fixed_sequence_pricing_certified"],
            "first_feasible_soc_step":
                membership["first_feasible_soc_step"],
            "first_feasible_block_min":
                membership["first_feasible_block_min"],
            "nonrepresentability_reason":
                membership["nonrepresentability_reason"],
            "target_gap": (
                max(0.0, float(cg["final_route_weight"])
                    - float(job["target_fleet"]))
                if cg["final_route_weight"] is not None
                and cg["final_artificial_mass"] is not None
                and float(cg["final_artificial_mass"]) <= 1e-9
                else None
            ),
            "target_gap_scope": "primary_discretized_route_space",
            "target_gap_interpretation": target_gap_interpretation(
                cg["final_route_weight"],
                job["target_fleet"],
                membership[
                    "known_partition_in_primary_expanded_space"
                ],
                cg["final_artificial_mass"],
            ),
        })
    return rows


def target_gap_interpretation(
    route_weight, target_fleet, known_partition_in_primary_space,
    artificial_mass=0.0,
):
    if route_weight is None:
        return "target_not_comparable_missing_lp_value"
    if artificial_mass is None or float(artificial_mass) > 1e-9:
        return "target_not_comparable_positive_or_missing_artificial_mass"
    if float(route_weight) <= float(target_fleet) + 1e-6:
        return "target_route_weight_observed_combined_cost_master"
    if known_partition_in_primary_space is not True:
        return "known_comparator_invalid_scaling_unresolved"
    return "within_primary_space_target_route_weight_not_observed"


def _inventory(cell, role, path):
    path = Path(path).resolve()
    return {
        "cell_id": cell,
        "role": role,
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _stream_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_completion(job, plan_sha):
    output = Path(job["output"])
    completion_path = Path(str(output) + ".worker-completion.json")
    completion = json.loads(completion_path.read_text())
    hashes = completion.get("artifact_sha256")
    if (
        completion.get("schema")
        != "evsp-dr-scale-ladder-worker-completion-v1"
        or completion.get("phase") != job["phase"]
        or completion.get("plan_sha256") != plan_sha
        or completion.get("instance_file_sha256")
        != job["instance"]["instance_file_sha256"]
        or completion.get("job_key") != job["job_key"]
        or completion.get("arm") != job.get("arm")
        or not isinstance(hashes, dict)
        or not hashes
    ):
        raise ValueError(f"worker completion differs: {job['job_key']}")
    output = Path(job["output"])
    expected = {output}
    if job["phase"] in {"CG", "CG_SENSITIVITY"}:
        expected.update({
            Path(str(output) + ".columns.jsonl"),
            Path(str(output) + ".iters.csv"),
        })
        if job.get("telemetry"):
            expected.add(Path(job["telemetry"]))
        for snapshot in output.parent.glob(
            f"{output.stem}.m*.snapshot.json"
        ):
            expected.add(snapshot)
            expected.add(Path(str(snapshot) + ".columns.jsonl"))
        for mark in job["snapshot_minutes"]:
            snapshot = output.parent / (
                f"{output.stem}.m{int(mark)}.snapshot.json"
            )
            availability = (
                completion.get("snapshot_availability") or {}
            ).get(str(mark))
            if availability == "available":
                if (
                    not snapshot.is_file()
                    or not Path(str(snapshot) + ".columns.jsonl").is_file()
                ):
                    raise ValueError(
                        f"available CG snapshot missing: {job['job_key']} m{mark}"
                    )
            elif availability in {
                "censored_solver_terminated_before_mark",
                "missed_in_prior_allocation",
            }:
                status = json.loads(output.read_text())
                if (
                    (status.get("snapshot_availability") or {}).get(
                        str(int(mark))
                    ) != availability
                    or snapshot.exists()
                    or Path(str(snapshot) + ".columns.jsonl").exists()
                ):
                    raise ValueError(
                        f"CG snapshot censoring mismatch: {job['job_key']}"
                    )
            else:
                raise ValueError(
                    f"CG snapshot availability missing: {job['job_key']} m{mark}"
                )
    elif job["phase"] == "PREFLIGHT":
        expected.add(output.with_suffix(".csv"))
    elif job["phase"] == "SEED" and job.get("membership_output"):
        membership = Path(job["membership_output"])
        expected.update({membership, membership.with_suffix(".csv")})
    elif job["phase"] == "MIP":
        progress = Path(job["progress_dir"])
        final = progress / "final.json"
        if not final.is_file():
            raise ValueError(f"MIP final checkpoint missing: {job['job_key']}")
        result = json.loads(output.read_text())
        schedule = (result.get("progress") or {}).get(
            "checkpoint_schedule_s"
        )
        if not isinstance(schedule, list):
            raise ValueError(f"MIP checkpoint schedule missing: {job['job_key']}")
        for mark in schedule:
            checkpoint = progress / (
                f"checkpoint_{int(round(float(mark)/60)):04d}m.json"
            )
            if not checkpoint.is_file():
                raise ValueError(
                    f"planned MIP checkpoint missing: {job['job_key']}"
                )
        expected.update(
            path for path in progress.rglob("*")
            if path.is_file()
        )
    if {str(path.resolve()) for path in expected} != set(hashes):
        raise ValueError(f"worker completion artifact set differs: {job['job_key']}")
    for path, digest in hashes.items():
        artifact = Path(path)
        if not artifact.is_file() or _stream_sha(artifact) != digest:
            raise ValueError(f"worker artifact changed: {job['job_key']}")


def _plots(staging, cg_rows, mip_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    groups = defaultdict(list)
    for row in cg_rows:
        if row.get("campaign_role") != "primary":
            continue
        groups[row["cell_id"]].append(row)
    for cell, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["elapsed_s"])
        axes[0].plot(
            [row["elapsed_s"] / 3600 for row in rows],
            [row["route_weight"] for row in rows],
            label=cell, linewidth=0.8,
        )
        axes[1].plot(
            [row["iteration"] for row in rows],
            [row["route_weight"] for row in rows],
            label=cell, linewidth=0.8,
        )
    axes[0].set(xlabel="CG elapsed hours", ylabel="LP route weight")
    axes[1].set(xlabel="CG iteration", ylabel="LP route weight")
    if groups:
        axes[0].legend(fontsize=4, ncol=3)
        axes[1].legend(fontsize=4, ncol=3)
    fig.tight_layout()
    fig.savefig(
        staging / "cg_route_weight.png", dpi=170,
        metadata={"Software": "EVSP-DR"},
    )
    fig.savefig(
        staging / "cg_route_weight.pdf",
        metadata={
            "Creator": "EVSP-DR", "CreationDate": None, "ModDate": None,
        },
    )
    plt.close(fig)
    sensitivity_groups = defaultdict(list)
    for row in cg_rows:
        if row.get("campaign_role") != "small_grid_sensitivity":
            continue
        sensitivity_groups[(
            row["cell_id"], row["soc_step"], row["block_min"]
        )].append(row)
    if sensitivity_groups:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for key, rows in sorted(sensitivity_groups.items()):
            rows.sort(key=lambda row: row["elapsed_s"])
            ax.plot(
                [row["elapsed_s"] / 3600 for row in rows],
                [row["route_weight"] for row in rows],
                label=f"{key[0]} g{key[1]} b{key[2]}",
                linewidth=0.8,
            )
        ax.set(
            xlabel="Diagnostic CG elapsed hours",
            ylabel="LP route weight",
            title=(
                "Route-space sensitivity diagnostics "
                "(excluded from primary-grid comparison)"
            ),
        )
        ax.legend(fontsize=5, ncol=3)
        fig.tight_layout()
        fig.savefig(
            staging / "cg_sensitivity_route_weight.png",
            dpi=170, metadata={"Software": "EVSP-DR"},
        )
        fig.savefig(
            staging / "cg_sensitivity_route_weight.pdf",
            metadata={
                "Creator": "EVSP-DR",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = defaultdict(list)
    for row in mip_rows:
        groups[(row["cell_id"], row["arm"])].append(row)
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["checkpoint_elapsed_s"])
        x = [row["checkpoint_elapsed_s"] / 3600 for row in rows]
        y = [
            row["statistics_incumbent_fleet"]
            if row["statistics_incumbent_fleet"] is not None
            else row["incumbent_fleet"]
            for row in rows
        ]
        line = ax.step(
            x, y, where="post", label=f"{key[0]} {key[1]} incumbent"
        )[0]
        if any(row["fleet_bound"] is not None for row in rows):
            ax.step(
                x, [row["fleet_bound"] for row in rows],
                where="post", linestyle="--",
                color=line.get_color(),
                label=f"{key[0]} {key[1]} bound",
            )
    ax.set(xlabel="MIP elapsed hours", ylabel="Fleet / finite-pool bound")
    if groups:
        ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(
        staging / "mip_incumbent_bound.png", dpi=170,
        metadata={"Software": "EVSP-DR"},
    )
    fig.savefig(
        staging / "mip_incumbent_bound.pdf",
        metadata={
            "Creator": "EVSP-DR", "CreationDate": None, "ModDate": None,
        },
    )
    plt.close(fig)


def _rename_noreplace(source, target, *, platform=None, libc=None):
    """Atomically rename without replacement on Linux or Darwin."""

    platform = platform or sys.platform
    libc = libc or ctypes.CDLL(None, use_errno=True)
    if platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        if function is None:
            raise OSError("renameat2(RENAME_NOREPLACE) unavailable")
        function.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            -100, os.fsencode(source), -100, os.fsencode(target), 1
        )
    elif platform == "darwin":
        function = getattr(libc, "renamex_np", None)
        if function is None:
            raise OSError("renamex_np(RENAME_EXCL) unavailable")
        function.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            os.fsencode(source), os.fsencode(target), 0x00000004
        )
    else:
        raise OSError(f"atomic no-clobber rename unsupported: {platform}")
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(target)
        raise OSError(error, os.strerror(error), target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--k40-reuse-manifest", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(
        args.campaign_root, args.out_dir, args.k40_reuse_manifest
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

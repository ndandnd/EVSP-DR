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
import tempfile
from collections import defaultdict
from pathlib import Path

from build_tariff_response_manifest import sha256_file
from scale_ladder_trip_identity import (
    classify_legacy_trip_hash,
    identity,
)


CG_FIELDS = (
    "cell_id", "scale", "selection_replicate", "cg_replicate",
    "target_fleet", "elapsed_s", "iteration", "lp_obj",
    "route_weight", "artificial_mass", "min_reduced_cost",
    "pool_columns", "columns_added", "master_time_s", "pricing_time_s",
    "pricing_certified", "stopping_reason", "censored",
    "instance_file_sha256", "ordered_trip_id_set_sha256",
    "solver_local_trip_index_sha256", "ordered_trip_sequence_sha256",
    "trip_identity_schema", "legacy_trip_hash",
    "legacy_trip_hash_schema", "legacy_trip_hash_field",
)
CG_SUMMARY_FIELDS = (
    "cell_id", "scale", "selection_replicate", "cg_replicate",
    "target_fleet", "budget_s", "elapsed_s", "iterations",
    "final_lp_obj", "final_route_weight", "final_artificial_mass",
    "final_min_reduced_cost", "pool_columns", "pricing_certified",
    "stopping_reason", "time_to_target_route_weight_s",
    "time_to_zero_artificials_s", "censored",
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
)
INVENTORY_FIELDS = (
    "cell_id", "role", "path", "sha256", "size_bytes",
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
    if (
        manifest.get("submitted") is not True
        or manifest.get("gate_state")
        not in {"released", "released_reconciled"}
        or set((manifest.get("submitted_arrays") or {}))
        != {"SEED", "CG", "MIP_RAW", "MIP_KNOWN"}
        or any(
            not str(value).isdigit()
            for value in (manifest.get("submitted_arrays") or {}).values()
        )
    ):
        raise ValueError("campaign submission is incomplete")
    jobs = {job["job_key"]: job for job in plan["jobs"]}
    for job in plan["jobs"]:
        _validate_completion(job, manifest["approval_sha256"])
    cg_jobs = [job for job in plan["jobs"] if job["phase"] == "CG"]
    cg_rows = []
    cg_summary = []
    mip_rows = []
    mip_summary = []
    inventory = []
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
            or status.get("soc_step") != 15.0
            or status.get("block_min") != 10
            or status.get("g_kwh") != 300.0
            or status.get("charge_kw") != 300.0
            or status.get("min_soc_frac") != 0.0
        ):
            raise ValueError(f"CG source/model identity differs: {job['cell_id']}")
        legacy = classify_legacy_trip_hash(
            status.get("trip_set_sha256"), identities
        )
        phases = _phase_times(Path(job["telemetry"]))
        iters_path = Path(str(status_path) + ".iters.csv")
        with iters_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        previous_columns = 0
        normalized = []
        for raw in rows:
            iteration = int(float(raw["iteration"]))
            pool_columns = int(float(raw["pool_columns"]))
            normalized.append({
                "cell_id": job["cell_id"],
                "scale": job["scale"],
                "selection_replicate": job["selection_replicate"],
                "cg_replicate": job["cg_replicate"],
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
            **{
                field: identities[field] for field in identities
                if field != "trip_count"
            },
        }
        cg_summary.append(summary)
        cg_by_cell[job["cell_id"]] = summary
        for role, path in (
            ("cg_status", status_path),
            ("cg_journal", Path(status["columns_journal"])),
            ("cg_iterations", iters_path),
            ("cg_telemetry", Path(job["telemetry"])),
        ):
            inventory.append(_inventory(job["cell_id"], role, path))
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
        plan, k40_reuse_manifest, mip_summary, mip_rows, inventory
    )
    progress = _progress_rows(plan, cg_by_cell, mip_summary)
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


def _append_k40_rows(plan, reuse_manifest, summaries, checkpoints, inventory):
    supplied = {}
    if reuse_manifest:
        path = Path(reuse_manifest).resolve()
        payload = json.loads(path.read_text())
        if payload.get("schema") != "evsp-dr-scale-ladder-k40-reuse-v1":
            raise ValueError("k40 reuse manifest schema mismatch")
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
                    or (candidate.get("physics") or {}).get("g_kwh")
                    != 300.0
                    or (candidate.get("physics") or {}).get("charge_kw")
                    != 300.0
                    or (candidate.get("physics") or {}).get("min_soc_frac")
                    != 0.0
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
                        and start.get("kind")
                        != "validated_exact_partition"
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


def _progress_rows(plan, cg_by_cell, mip_summary):
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
        })
    return rows


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
    if job["phase"] == "CG":
        expected.update({
            Path(str(output) + ".columns.jsonl"),
            Path(str(output) + ".iters.csv"),
            Path(job["telemetry"]),
        })
        for snapshot in output.parent.glob(
            f"{output.stem}.m*.snapshot.json"
        ):
            expected.add(snapshot)
            expected.add(Path(str(snapshot) + ".columns.jsonl"))
    elif job["phase"] == "MIP":
        expected.update(
            path for path in Path(job["progress_dir"]).rglob("*")
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


def _rename_noreplace(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise OSError("renameat2 unavailable")
    function.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    if function(
        -100, os.fsencode(source), -100, os.fsencode(target), 1
    ) != 0:
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

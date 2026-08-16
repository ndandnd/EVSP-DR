#!/usr/bin/env python3
"""Normalize heterogeneous EVSP-DR evidence and audit rerun requirements."""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import stat
import tempfile
from collections import defaultdict
from pathlib import Path

from cross_generation_schema import (
    normalize_termination,
    parse_artifact,
    sha256_bytes,
)


OUTPUT_SCHEMA = "evsp-dr-cross-generation-evidence-v1"
K40_INSTANCE_SHA256 = (
    "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd"
)
K40_TRIP_SET_SHA256 = (
    "35604b22facf1646963e85eb98a858906f0dd7dbebd86ea0d3ac7b797de62ed0"
)
INVENTORY_FIELDS = (
    "artifact_id", "run_id", "path", "artifact_type", "schema_family",
    "algorithm_family", "implementation", "treatment",
    "scale_family", "scale",
    "replicate", "seed", "expected_sha256", "observed_sha256",
    "available", "validation_status", "reason", "tail_dropped",
    "tail_reason", "endpoint_only", "instance_sha256", "trip_set_sha256",
    "trip_count",
    "tariff_sha256", "git_commit", "git_dirty",
    "field_availability_json",
)
CG_FIELDS = (
    "artifact_id", "run_id", "schema_family", "algorithm_family",
    "implementation", "scale_family", "scale", "replicate", "seed",
    "iteration", "legacy_master_objective", "legacy_master_improvement",
    "master_objective_before_add", "master_improvement_before_add",
    "lp_objective", "lp_route_weight", "peak_trip_concurrency",
    "artificial_trips",
    "artificial_total", "best_reduced_cost", "best_reduced_cost_reason",
    "columns_added", "pool_columns", "pool_columns_delta",
    "wall_time_s", "master_time_s",
    "pricing_time_s", "cumulative_master_time_s",
    "cumulative_pricing_time_s", "master_pricing_split_available",
    "timed_out", "deepest_tier_timed_out", "label_cap_evictions",
    "pricing_labels_used", "pricing_label_cap_configured",
    "pricing_completed_routes", "pricing_negative_completed",
    "pricing_eligible_negative_incidences",
    "pricing_returned_trip_count_min", "pricing_returned_trip_count_mean",
    "pricing_returned_trip_count_max", "pricing_exhaustive",
    "pricing_queue_order", "pricing_output_selection",
    "pricing_dominance_mode", "highest_tier_reached",
    "recent_window_sum", "stagnant_counter",
    "pricing_time_limit_used_s", "tier_statistics_json",
    "availability_reason",
)
CG_SUMMARY_FIELDS = (
    "run_id", "algorithm_family", "implementation", "schema_family",
    "scale_family", "scale", "replicate", "seed", "iteration_count",
    "final_wall_time_s", "final_legacy_master_objective",
    "final_master_objective_before_add", "final_lp_objective",
    "final_lp_route_weight", "final_artificial_total",
    "final_best_reduced_cost", "final_pool_columns",
    "termination_category", "termination_raw", "termination_reason",
    "pricing_certified", "zero_artificials_reached",
    "zero_artificials_observable",
    "time_to_zero_artificials_s", "zero_artificials_censored",
    "target_lp_weight", "target_lp_weight_reached",
    "target_lp_weight_observable",
    "time_to_target_lp_weight_s", "target_lp_weight_censored",
    "certification_observable", "time_to_certification_s",
    "certification_censored", "event_clock",
    "censor_time_s", "master_time_share", "pricing_time_share",
    "timing_split_available", "phase_telemetry_available",
    "phase_telemetry_reason", "source_artifact_ids",
)
MIP_CHECKPOINT_FIELDS = (
    "artifact_id", "run_id", "algorithm_family", "implementation",
    "scale_family", "scale", "replicate", "treatment",
    "checkpoint_elapsed_s", "observed_total_elapsed_s",
    "statistics_observed_s", "stage", "incumbent_state",
    "incumbent_fleet", "incumbent_objective", "fleet_bound",
    "objective_bound", "fleet_gap", "node_count", "solution_count",
    "route_vector_sha256", "first_feasible_s",
    "solver_ended_before_checkpoint", "source_result_sha256",
    "source_journal_sha256", "source_start_sha256", "experiment_arm",
    "git_commit", "observational_only",
)
MIP_SUMMARY_FIELDS = (
    "run_id", "algorithm_family", "implementation", "scale_family",
    "scale", "replicate", "treatment", "incumbent_found",
    "integer_fleet", "objective", "objective_bound",
    "objective_bound_scope", "gap",
    "fleet_bound", "fleet_proven", "status_name", "optimal_scope",
    "runtime_s", "partitioning", "physically_validated_schedule",
    "giro_columns_added", "pool_scope", "first_feasible_s",
    "time_to_fleet_proof_s", "proof_time_observable",
    "proof_censored", "censor_time_s", "summary_availability_reason",
    "source_result_sha256", "source_journal_sha256",
    "source_start_sha256",
    "source_status_artifact_id", "source_journal_artifact_id",
    "physical_replay_artifact_id",
    "physical_replay_route_vector_sha256",
    "final_artifact_id", "checkpoint_artifact_ids",
    "git_commit",
)
COVERAGE_FIELDS = (
    "algorithm_family", "implementation", "treatment",
    "scale_family", "scale",
    "size_class", "comparison_group", "model_difference", "required_evidence",
    "minimum_replicates", "trajectory_replicates", "endpoint_replicates",
    "mip_checkpoint_replicates", "mip_final_replicates",
    "verified_trip_sets", "replicate_semantics",
    "coverage_status", "availability_reason",
)
RERUN_FIELDS = (
    *COVERAGE_FIELDS,
    "rerun_required", "additional_replicates", "recommended_budget_s",
    "recommended_stopping_rule", "common_model_assumptions",
    "dry_run_only", "command_available", "command_blocked_reason",
    "command_argv_json",
)
TELEMETRY_FIELDS = (
    "artifact_id", "run_id", "session", "phase", "duration_s",
    "elapsed_session_s", "iteration", "attempt", "pool_columns",
    "incidence_nnz", "network_nodes", "network_arcs",
    "telemetry_overhead_before_s", "peak_rss_bytes",
    "outcome", "identity_sha256",
)


def _canonical(payload) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _read_regular_no_follow(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"artifact is not a regular file: {path}")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _read_manifest(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("input manifest is not a JSON object")
    if value.get("schema") != "evsp-dr-cross-generation-input-manifest-v1":
        raise ValueError("unexpected input manifest schema")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("input manifest artifacts is not a list")
    return value, raw


def _resolve_path(spec: dict, manifest: dict, manifest_path: Path,
                  repo_root: Path) -> Path:
    raw = os.path.expandvars(str(spec.get("path") or ""))
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.absolute()
    mode = manifest.get("relative_paths", "repository")
    base = repo_root if mode == "repository" else manifest_path.parent
    return (base / path).absolute()


def _valid_sha(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _flatten_inventory(spec, path, observed, status, reason, parsed=None):
    metadata = spec.get("metadata") or {}
    tail = (parsed or {}).get("tail") or {}
    provenance_fields = (
        "git_commit", "git_dirty", "instance_sha256",
        "trip_set_sha256", "tariff_sha256", "seed",
        "solver_backend", "python_environment_identity",
        "time_limit_s", "memory_limit_bytes", "threads",
    )
    field_availability = {
        key: {
            "available": metadata.get(key) is not None,
            "reason": (
                None if metadata.get(key) is not None
                else metadata.get("provenance_availability_reason")
                or "not_recorded_in_source_or_reviewed_manifest"
            ),
        }
        for key in provenance_fields
    }
    return {
        "artifact_id": spec.get("artifact_id"),
        "run_id": spec.get("run_id"),
        "path": str(spec.get("path")),
        "artifact_type": spec.get("artifact_type"),
        "schema_family": (parsed or {}).get("schema"),
        "algorithm_family": metadata.get("algorithm_family"),
        "implementation": metadata.get("implementation"),
        "treatment": metadata.get("treatment"),
        "scale_family": metadata.get("scale_family"),
        "scale": metadata.get("scale"),
        "replicate": metadata.get("replicate"),
        "seed": metadata.get("seed"),
        "expected_sha256": spec.get("expected_sha256"),
        "observed_sha256": observed,
        "available": path.is_file(),
        "validation_status": status,
        "reason": reason,
        "tail_dropped": tail.get("tail_dropped"),
        "tail_reason": tail.get("tail_reason"),
        "endpoint_only": spec.get("artifact_type") in {
            "endpoint_json", "artifact_manifest_json"
        },
        "instance_sha256": metadata.get("instance_sha256"),
        "trip_set_sha256": metadata.get("trip_set_sha256"),
        "trip_count": metadata.get("trip_count"),
        "tariff_sha256": metadata.get("tariff_sha256"),
        "git_commit": metadata.get("git_commit"),
        "git_dirty": metadata.get("git_dirty"),
        "field_availability_json": json.dumps(
            field_availability, sort_keys=True, separators=(",", ":")
        ),
    }


def _validate_specs(manifest: dict):
    seen_artifacts = set()
    seen_roles = set()
    seen_trajectory_hashes = {}
    for spec in manifest["artifacts"]:
        if not isinstance(spec, dict):
            raise ValueError("artifact specification is not an object")
        artifact_id = spec.get("artifact_id")
        run_id = spec.get("run_id")
        artifact_type = spec.get("artifact_type")
        if (
            not isinstance(artifact_id, str) or not artifact_id
            or not isinstance(run_id, str) or not run_id
            or not isinstance(artifact_type, str)
            or not _valid_sha(spec.get("expected_sha256"))
            or type(spec.get("required", False)) is not bool
        ):
            raise ValueError("artifact identity/hash is malformed")
        if artifact_id in seen_artifacts:
            raise ValueError(f"duplicate artifact_id {artifact_id}")
        role = (
            run_id,
            artifact_type,
            spec.get("artifact_role", artifact_type),
        )
        if role in seen_roles:
            raise ValueError(f"duplicate run/schema role {role}")
        seen_artifacts.add(artifact_id)
        seen_roles.add(role)
        metadata = spec.get("metadata") or {}
        if artifact_type in {
            "heuristic_dp_historical_csv",
            "heuristic_dp_current_csv",
            "exact_cg_iterations_csv",
            "mip_checkpoint", "mip_final",
        }:
            for key in (
                "algorithm_family", "implementation",
                "scale_family", "scale", "replicate",
            ):
                if metadata.get(key) is None:
                    raise ValueError(
                        f"artifact {artifact_id} lacks reviewed metadata {key}"
                    )
        if artifact_type in {
            "heuristic_dp_historical_csv",
            "heuristic_dp_current_csv",
            "exact_cg_iterations_csv",
        }:
            duplicate = seen_trajectory_hashes.get(spec["expected_sha256"])
            if duplicate is not None:
                raise ValueError(
                    f"trajectory bytes are duplicated as {duplicate} and "
                    f"{artifact_id}"
                )
            seen_trajectory_hashes[spec["expected_sha256"]] = artifact_id
        if (
            metadata.get("scale_family") == "union"
            and str(metadata.get("scale")) == "40"
            and (
                metadata.get("trip_count") != 947
                or metadata.get("instance_sha256") != K40_INSTANCE_SHA256
                or metadata.get("trip_set_sha256")
                != K40_TRIP_SET_SHA256
            )
        ):
            raise ValueError(
                f"k40 artifact {artifact_id} is not the fixed 947-trip case"
            )


def _validate_run_consistency(manifest: dict):
    fields = (
        "instance_sha256", "trip_set_sha256", "tariff_sha256",
        "scale_family", "scale", "replicate", "seed", "treatment",
    )
    grouped = defaultdict(list)
    for spec in manifest["artifacts"]:
        grouped[spec["run_id"]].append(spec)
    for run_id, specs in grouped.items():
        trajectory_specs = [
            spec for spec in specs
            if spec["artifact_type"] in {
                "heuristic_dp_historical_csv",
                "heuristic_dp_current_csv",
                "exact_cg_iterations_csv",
            }
        ]
        endpoint_specs = [
            spec for spec in specs if spec["artifact_type"] == "endpoint_json"
        ]
        telemetry_specs = [
            spec for spec in specs
            if spec["artifact_type"] == "exact_cg_phase_telemetry_jsonl"
        ]
        journal_specs = [
            spec for spec in specs
            if spec["artifact_type"] == "exact_cg_column_journal_jsonl"
        ]
        mip_final_specs = [
            spec for spec in specs if spec["artifact_type"] == "mip_final"
        ]
        if (
            len(trajectory_specs) > 1
            or len(endpoint_specs) > 1
            or len(telemetry_specs) > 1
            or len(journal_specs) > 1
            or len(mip_final_specs) > 1
        ):
            raise ValueError(f"run {run_id} has duplicate evidence roles")
        trajectory_types = {
            spec["artifact_type"] for spec in trajectory_specs
        }
        if len(trajectory_types) > 1:
            raise ValueError(f"run {run_id} mixes trajectory schemas")
        algorithms = {
            (spec.get("metadata") or {}).get("algorithm_family")
            for spec in specs
            if (spec.get("metadata") or {}).get("algorithm_family")
            not in (None, "artifact_manifest")
        }
        if len(algorithms) > 1:
            raise ValueError(f"run {run_id} mixes algorithm families")
        if trajectory_specs:
            trajectory_implementation = (
                trajectory_specs[0].get("metadata") or {}
            ).get("implementation")
            for companion in endpoint_specs:
                implementation = (
                    companion.get("metadata") or {}
                ).get("implementation")
                if (
                    implementation is not None
                    and implementation != trajectory_implementation
                ):
                    raise ValueError(
                        f"run {run_id} endpoint implementation mismatch"
                    )
            for telemetry in telemetry_specs:
                implementation = (
                    telemetry.get("metadata") or {}
                ).get("implementation")
                if implementation not in {
                    trajectory_implementation,
                    f"{trajectory_implementation}_phase_telemetry",
                }:
                    raise ValueError(
                        f"run {run_id} telemetry implementation mismatch"
                    )
        mip_specs = [
            spec for spec in specs
            if spec["artifact_type"] in {"mip_checkpoint", "mip_final"}
        ]
        if len({
            (spec.get("metadata") or {}).get("implementation")
            for spec in mip_specs
        }) > 1:
            raise ValueError(f"run {run_id} mixes MIP implementations")
        for field in fields:
            values = {
                (spec.get("metadata") or {}).get(field)
                for spec in specs
                if (spec.get("metadata") or {}).get(field) is not None
            }
            if len(values) > 1:
                raise ValueError(
                    f"run {run_id} has mixed {field}: {sorted(values)}"
                )


def _embedded_provenance_mismatches(parsed: dict, spec: dict) -> list[str]:
    endpoint = parsed.get("endpoint")
    if endpoint is None and parsed.get("schema") == "exact_cg_snapshot_json":
        endpoint = parsed.get("checkpoint")
    if not isinstance(endpoint, dict):
        return []
    metadata = spec.get("metadata") or {}
    provenance = endpoint.get("provenance") or {}
    observed = {
        "instance_sha256": (
            provenance.get("instance_sha256")
            or endpoint.get("Instance_SHA256")
        ),
        "tariff_sha256": (
            provenance.get("prices_sha256")
            or endpoint.get("Price_SHA256")
            or endpoint.get("Tariff_SHA256")
        ),
        "git_commit": (
            provenance.get("git_commit")
            or (endpoint.get("Git") or {}).get("commit")
        ),
        "git_dirty": (
            provenance.get("git_dirty")
            if "git_dirty" in provenance
            else (endpoint.get("Git") or {}).get("dirty")
        ),
    }
    trip_ids = endpoint.get("trip_ids")
    if isinstance(trip_ids, list):
        observed["trip_set_sha256"] = hashlib.sha256(
            json.dumps(trip_ids, separators=(",", ":")).encode()
        ).hexdigest()
        observed["trip_count"] = len(trip_ids)
    mismatches = []
    for key, value in observed.items():
        expected = metadata.get(key)
        if (
            expected is not None
            and not (
                key in {"trip_set_sha256", "trip_count"}
                and not isinstance(trip_ids, list)
            )
            and expected != value
        ):
            mismatches.append(f"{key}: manifest={expected} artifact={value}")
    return mismatches


def _termination_from_endpoint(endpoint):
    if not isinstance(endpoint, dict):
        return None
    return (
        endpoint.get("stop_reason")
        or endpoint.get("Termination_Reason")
        or endpoint.get("termination_reason")
    )


def _certification_from_endpoint(endpoint):
    if not isinstance(endpoint, dict):
        return False
    return (
        endpoint.get("certified_rc_optimal") is True
        or endpoint.get("Termination_Reason") == "rc_optimal_restricted"
        or endpoint.get("termination_reason") == "rc_optimal_restricted"
    )


def _endpoint_horizon(endpoint):
    if not isinstance(endpoint, dict):
        return None, None
    for key, clock in (
        ("wall_s", "wall_time"),
        ("Total_Time_s", "wall_time"),
        ("Total_Runtime_s", "wall_time"),
        ("active_time_s", "active_time"),
    ):
        value = endpoint.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number >= 0:
            return number, clock
    return None, None


def _cg_summaries(
    cg_rows, endpoints, telemetry_by_run, specs_by_run,
    parsed_artifact_ids,
):
    grouped = defaultdict(list)
    for row in cg_rows:
        grouped[row["run_id"]].append(row)
    summaries = []
    for run_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda value: (value["iteration"], value["artifact_id"]))
        final = rows[-1]
        endpoint = endpoints.get(run_id)
        termination_raw = _termination_from_endpoint(endpoint)
        termination, termination_reason = normalize_termination(termination_raw)
        certified = _certification_from_endpoint(endpoint)
        endpoint_horizon, endpoint_clock = _endpoint_horizon(endpoint)
        metadata = (specs_by_run[run_id][0].get("metadata") or {})
        tolerance = float(metadata.get("artificial_tolerance", 1e-6))
        target = metadata.get("target_lp_weight")
        zero_times = [
            row["wall_time_s"] for row in rows
            if row["artificial_total"] is not None
            and row["artificial_total"] <= tolerance
            and row["wall_time_s"] is not None
        ]
        target_times = [
            row["wall_time_s"] for row in rows
            if target is not None
            and row["lp_route_weight"] is not None
            and row["lp_route_weight"] <= float(target)
            and row["artificial_total"] is not None
            and row["artificial_total"] <= tolerance
            and row["wall_time_s"] is not None
        ]
        censor = max(
            value for value in (final["wall_time_s"], endpoint_horizon)
            if value is not None
        )
        zero_observable = any(
            row["artificial_total"] is not None for row in rows
        )
        target_observable = (
            target is not None and zero_observable
            and any(row["lp_route_weight"] is not None for row in rows)
        )
        certification_observable = (
            isinstance(endpoint, dict)
            and final["algorithm_family"] in {
                "exact_expanded_network", "heuristic_dp_current"
            }
        )
        cum_master = final["cumulative_master_time_s"]
        cum_pricing = final["cumulative_pricing_time_s"]
        total_active = (
            cum_master + cum_pricing
            if cum_master is not None and cum_pricing is not None
            else None
        )
        telemetry = telemetry_by_run.get(run_id) or []
        summaries.append({
            "run_id": run_id,
            "algorithm_family": final["algorithm_family"],
            "implementation": final["implementation"],
            "schema_family": final["schema_family"],
            "scale_family": final["scale_family"],
            "scale": final["scale"],
            "replicate": final["replicate"],
            "seed": final["seed"],
            "iteration_count": len(rows),
            "final_wall_time_s": censor,
            "final_legacy_master_objective": final[
                "legacy_master_objective"
            ],
            "final_master_objective_before_add": final[
                "master_objective_before_add"
            ],
            "final_lp_objective": final["lp_objective"],
            "final_lp_route_weight": final["lp_route_weight"],
            "final_artificial_total": final["artificial_total"],
            "final_best_reduced_cost": final["best_reduced_cost"],
            "final_pool_columns": final["pool_columns"],
            "termination_category": termination,
            "termination_raw": termination_raw,
            "termination_reason": termination_reason,
            "pricing_certified": certified,
            "zero_artificials_reached": bool(zero_times),
            "zero_artificials_observable": zero_observable,
            "time_to_zero_artificials_s": min(zero_times) if zero_times else None,
            "zero_artificials_censored": (
                not bool(zero_times) if zero_observable else None
            ),
            "target_lp_weight": target,
            "target_lp_weight_reached": bool(target_times),
            "target_lp_weight_observable": target_observable,
            "time_to_target_lp_weight_s": (
                min(target_times) if target_times else None
            ),
            "target_lp_weight_censored": (
                not bool(target_times) if target_observable else None
            ),
            "certification_observable": certification_observable,
            "time_to_certification_s": (
                endpoint_horizon if certified else None
            ),
            "certification_censored": (
                not certified if certification_observable else None
            ),
            "event_clock": endpoint_clock or "trajectory_wall_time",
            "censor_time_s": censor,
            "master_time_share": (
                cum_master / total_active if total_active else None
            ),
            "pricing_time_share": (
                cum_pricing / total_active if total_active else None
            ),
            "timing_split_available": (
                final["master_pricing_split_available"] is True
            ),
            "phase_telemetry_available": bool(telemetry),
            "phase_telemetry_reason": (
                None if telemetry
                else "telemetry_disabled_or_not_supplied"
            ),
            "source_artifact_ids": " | ".join(sorted({
                spec["artifact_id"] for spec in specs_by_run[run_id]
                if spec["artifact_id"] in parsed_artifact_ids
            })),
        })
    for run_id, endpoint in sorted(endpoints.items()):
        if run_id in grouped:
            continue
        specs = specs_by_run[run_id]
        metadata = specs[0].get("metadata") or {}
        horizon, clock = _endpoint_horizon(endpoint)
        termination_raw = _termination_from_endpoint(endpoint)
        termination, reason = normalize_termination(termination_raw)
        certified = _certification_from_endpoint(endpoint)
        summaries.append({
            "run_id": run_id,
            "algorithm_family": metadata.get("algorithm_family"),
            "implementation": metadata.get("implementation"),
            "schema_family": "endpoint_only",
            "scale_family": metadata.get("scale_family"),
            "scale": metadata.get("scale"),
            "replicate": metadata.get("replicate"),
            "seed": metadata.get("seed"),
            "iteration_count": 0,
            "final_wall_time_s": horizon,
            "final_legacy_master_objective": None,
            "final_master_objective_before_add": None,
            "final_lp_objective": None,
            "final_lp_route_weight": None,
            "final_artificial_total": None,
            "final_best_reduced_cost": None,
            "final_pool_columns": None,
            "termination_category": termination,
            "termination_raw": termination_raw,
            "termination_reason": reason,
            "pricing_certified": certified,
            "zero_artificials_reached": None,
            "zero_artificials_observable": False,
            "time_to_zero_artificials_s": None,
            "zero_artificials_censored": None,
            "target_lp_weight": metadata.get("target_lp_weight"),
            "target_lp_weight_reached": None,
            "target_lp_weight_observable": False,
            "time_to_target_lp_weight_s": None,
            "target_lp_weight_censored": None,
            "certification_observable": (
                isinstance(endpoint, dict)
                and metadata.get("algorithm_family") in {
                    "exact_expanded_network", "heuristic_dp_current"
                }
            ),
            "time_to_certification_s": horizon if certified else None,
            "certification_censored": (
                not certified
                if isinstance(endpoint, dict)
                and metadata.get("algorithm_family") in {
                    "exact_expanded_network", "heuristic_dp_current"
                }
                else None
            ),
            "event_clock": clock,
            "censor_time_s": horizon,
            "master_time_share": None,
            "pricing_time_share": None,
            "timing_split_available": False,
            "phase_telemetry_available": False,
            "phase_telemetry_reason": "no_trajectory_or_telemetry",
            "source_artifact_ids": " | ".join(sorted(
                spec["artifact_id"] for spec in specs
                if spec["artifact_id"] in parsed_artifact_ids
            )),
        })
    return summaries


def _mip_summaries(mip_rows, mip_finals, hash_to_artifact_ids):
    checkpoints = defaultdict(list)
    for row in mip_rows:
        checkpoints[row["run_id"]].append(row)
    summaries = []
    for final in sorted(mip_finals, key=lambda value: value["run_id"]):
        rows = sorted(
            checkpoints.get(final["run_id"], []),
            key=lambda value: (
                value["checkpoint_elapsed_s"]
                if value["checkpoint_elapsed_s"] is not None else math.inf
            ),
        )
        if any(
            row["treatment"] != final["treatment"]
            or row["source_result_sha256"]
            != final["source_result_sha256"]
            or row["source_journal_sha256"]
            != final["source_journal_sha256"]
            or row["source_start_sha256"]
            != final["source_start_sha256"]
            or row["git_commit"] != final["git_commit"]
            for row in rows
        ):
            raise ValueError(
                f"MIP checkpoint/final identity mismatch: {final['run_id']}"
            )
        proof_times = [
            row["statistics_observed_s"] for row in rows
            if row["statistics_observed_s"] is not None
            and row["incumbent_fleet"] is not None
            and row["fleet_bound"] is not None
            and math.ceil(float(row["fleet_bound"]) - 1e-6)
            >= int(row["incumbent_fleet"])
            and row["solver_ended_before_checkpoint"] is False
        ]
        if proof_times and final.get("fleet_proven") is not True:
            raise ValueError(
                f"MIP checkpoint/final proof mismatch: {final['run_id']}"
            )
        first_feasible = [
            row["first_feasible_s"] for row in rows
            if row["first_feasible_s"] is not None
        ]
        runtime = final.get("runtime_s")
        summaries.append({
            **final,
            "pool_scope": (
                "finite_giro_augmented_pool"
                if final.get("giro_columns_added")
                else "finite_raw_pool"
            ),
            "first_feasible_s": (
                min(first_feasible) if first_feasible else None
            ),
            "time_to_fleet_proof_s": (
                min(proof_times)
                if proof_times else None
            ),
            "proof_time_observable": bool(proof_times),
            "proof_censored": final.get("fleet_proven") is not True,
            "censor_time_s": runtime,
            "summary_availability_reason": (
                None if proof_times
                else "proof_status_known_but_exact_proof_time_unobserved"
                if final.get("fleet_proven") is True
                else None
            ),
            "final_artifact_id": final["artifact_id"],
            "checkpoint_artifact_ids": " | ".join(sorted({
                row["artifact_id"] for row in rows
            })),
            "source_status_artifact_id": " | ".join(
                hash_to_artifact_ids.get(
                    final.get("source_result_sha256"), []
                )
            ),
            "source_journal_artifact_id": " | ".join(
                hash_to_artifact_ids.get(
                    final.get("source_journal_sha256"), []
                )
            ),
            "physical_replay_artifact_id": " | ".join(
                hash_to_artifact_ids.get(
                    final.get("physical_replay_artifact_sha256"), []
                )
            ),
        })
    final_run_ids = {row["run_id"] for row in mip_finals}
    for run_id, rows in sorted(checkpoints.items()):
        if run_id in final_run_ids:
            continue
        rows = sorted(rows, key=lambda row: (
            row["checkpoint_elapsed_s"]
            if row["checkpoint_elapsed_s"] is not None else math.inf
        ))
        exemplar = rows[-1]
        summaries.append({
            "run_id": run_id,
            "algorithm_family": exemplar["algorithm_family"],
            "implementation": exemplar["implementation"],
            "scale_family": exemplar["scale_family"],
            "scale": exemplar["scale"],
            "replicate": exemplar["replicate"],
            "treatment": exemplar["treatment"],
            "incumbent_found": None,
            "integer_fleet": None,
            "objective": None,
            "objective_bound": None,
            "objective_bound_scope": None,
            "gap": None,
            "fleet_bound": None,
            "fleet_proven": None,
            "status_name": None,
            "optimal_scope": None,
            "runtime_s": max(
                row["observed_total_elapsed_s"] for row in rows
            ),
            "partitioning": None,
            "physically_validated_schedule": None,
            "giro_columns_added": (
                True if exemplar["treatment"] == "GIRO"
                else False if exemplar["treatment"] == "RAW" else None
            ),
            "pool_scope": (
                "finite_giro_augmented_pool"
                if exemplar["treatment"] == "GIRO"
                else "finite_raw_pool"
                if exemplar["treatment"] == "RAW" else None
            ),
            "first_feasible_s": min(
                (
                    row["first_feasible_s"] for row in rows
                    if row["first_feasible_s"] is not None
                ),
                default=None,
            ),
            "time_to_fleet_proof_s": None,
            "proof_time_observable": False,
            "proof_censored": None,
            "censor_time_s": max(
                row["observed_total_elapsed_s"] for row in rows
            ),
            "summary_availability_reason": (
                "checkpoint_only_without_validated_final_result"
            ),
            "source_result_sha256": exemplar["source_result_sha256"],
            "source_journal_sha256": exemplar["source_journal_sha256"],
            "source_start_sha256": exemplar["source_start_sha256"],
            "source_status_artifact_id": "",
            "source_journal_artifact_id": "",
            "physical_replay_artifact_id": "",
            "final_artifact_id": None,
            "checkpoint_artifact_ids": " | ".join(sorted({
                row["artifact_id"] for row in rows
            })),
            "git_commit": exemplar["git_commit"],
        })
    summaries.sort(key=lambda row: row["run_id"])
    return summaries


def _default_coverage_expectations():
    scales = (
        ("pair", "pairs"), ("single", "single"),
        *[("union", value) for value in (5, 8, 10, 13, 15, 20, 30, 40)],
    )
    algorithms = (
        ("heuristic_dp_historical", "legacy_dp", None, "trajectory"),
        ("heuristic_dp_current", "instrumented_dp", None, "trajectory"),
        ("exact_expanded_network", "exact_pricer", None, "trajectory"),
        (
            "mip_finite_pool", "two_stage_pool_mip",
            "RAW", "mip_checkpoint_and_final"
        ),
        (
            "mip_finite_pool", "two_stage_pool_mip",
            "GIRO", "mip_checkpoint_and_final"
        ),
    )
    def size_class(scale_family, scale):
        if str(scale) == "40":
            return "k40"
        if scale_family in {"pair", "single"}:
            return "small"
        try:
            numeric = int(scale)
        except (TypeError, ValueError):
            return "unclassified"
        return "small" if numeric <= 13 else "medium"

    return [{
        "algorithm_family": family,
        "implementation": implementation,
        "treatment": treatment,
        "scale_family": scale_family,
        "scale": scale,
        "size_class": size_class(scale_family, scale),
        "comparison_group": f"{scale_family}-{scale}",
        "model_difference": (
            "historical/current DP and exact expanded-network pricing are "
            "different implementations; compare trajectories without "
            "claiming identical pricing graphs"
        ),
        "required_evidence": required,
        "minimum_replicates": 3,
    } for scale_family, scale in scales
      for family, implementation, treatment, required in algorithms]


def _coverage_and_reruns(manifest, inventory, cg_rows, mip_rows, mip_finals):
    expectations = list(
        manifest.get("coverage_expectations")
        or _default_coverage_expectations()
    )
    verified = [
        row for row in inventory if row["validation_status"] == "verified"
    ]
    expected_keys = {
        (
            row["algorithm_family"], row["implementation"],
            row.get("treatment"), row["scale_family"], str(row["scale"]),
        )
        for row in expectations
    }
    for observed in verified:
        key = (
            observed.get("algorithm_family"),
            observed.get("implementation"),
            observed.get("treatment"),
            observed.get("scale_family"),
            str(observed.get("scale")),
        )
        if (
            key in expected_keys
            or observed.get("algorithm_family") == "artifact_manifest"
            or observed.get("scale") is None
            or observed.get("artifact_type") in {
                "endpoint_json",
                "artifact_manifest_json",
                "exact_cg_phase_telemetry_jsonl",
                "exact_cg_column_journal_jsonl",
                "exact_cg_snapshot_json",
                "run_checkpoint_json",
                "route_validation_json",
            }
        ):
            continue
        expectations.append({
            "algorithm_family": key[0],
            "implementation": key[1],
            "treatment": key[2],
            "scale_family": key[3],
            "scale": observed.get("scale"),
            "size_class": "observed_outside_design",
            "comparison_group": (
                f"observed:{key[0]}:{key[1]}:{key[3]}:{key[4]}"
            ),
            "model_difference": "observed artifact outside reviewed design matrix",
            "required_evidence": (
                "mip_checkpoint_and_final"
                if key[0] == "mip_finite_pool" else "trajectory"
            ),
            "minimum_replicates": 1,
        })
        expected_keys.add(key)
    coverage = []
    reruns = []
    observed_runtime = defaultdict(list)
    for row in cg_rows:
        if row["wall_time_s"] is not None:
            observed_runtime[(
                row["algorithm_family"], row["implementation"],
                row.get("treatment"), row["scale_family"], row["scale"]
            )].append(row["wall_time_s"])
    for row in mip_finals:
        if row["runtime_s"] is not None:
            observed_runtime[(
                row["algorithm_family"], row["implementation"],
                row.get("treatment"), row["scale_family"], row["scale"]
            )].append(row["runtime_s"])
    for expected in expectations:
        def matches(row):
            return (
                row.get("algorithm_family") == expected["algorithm_family"]
                and row.get("implementation") == expected["implementation"]
                and row.get("treatment") == expected.get("treatment")
                and row.get("scale_family") == expected["scale_family"]
                and str(row.get("scale")) == str(expected["scale"])
            )
        trajectory_run_ids = {
            row["run_id"] for row in cg_rows if matches(row)
        }
        trajectory_identities = [
            row for row in verified
            if matches(row)
            and row["run_id"] in trajectory_run_ids
            and row.get("trip_set_sha256")
            and row.get("replicate") is not None
        ]
        trajectory_by_trip_set = defaultdict(set)
        for row in trajectory_identities:
            trajectory_by_trip_set[row["trip_set_sha256"]].add(
                row["replicate"]
            )
        trajectory_count = max(
            (len(values) for values in trajectory_by_trip_set.values()),
            default=0,
        )
        endpoint_by_trip_set = defaultdict(set)
        for row in verified:
            if (
                matches(row) and row["endpoint_only"]
                and row.get("replicate") is not None
                and row.get("trip_set_sha256")
            ):
                endpoint_by_trip_set[row["trip_set_sha256"]].add(
                    row["replicate"]
                )
        endpoint_count = max(
            (len(values) for values in endpoint_by_trip_set.values()),
            default=0,
        )
        checkpoint_runs = {
            row["run_id"] for row in mip_rows if matches(row)
        }
        final_runs = {
            row["run_id"] for row in mip_finals if matches(row)
        }
        paired_mip_runs = checkpoint_runs & final_runs
        paired_valid_runs = set()
        for run_id in paired_mip_runs:
            companion_rows = [
                row for row in verified
                if matches(row) and row["run_id"] == run_id
                and row["artifact_type"] in {
                    "mip_checkpoint", "mip_final"
                }
            ]
            roles = {row["artifact_type"] for row in companion_rows}
            trip_identities = {
                row["trip_set_sha256"] for row in companion_rows
                if row.get("trip_set_sha256")
            }
            if roles == {"mip_checkpoint", "mip_final"} and len(
                    trip_identities) == 1:
                paired_valid_runs.add(run_id)
        paired_mip_inventory = [
            row for row in verified
            if matches(row) and row["run_id"] in paired_valid_runs
            and row.get("trip_set_sha256")
            and row.get("replicate") is not None
        ]
        mip_by_trip_set = defaultdict(set)
        for row in paired_mip_inventory:
            mip_by_trip_set[row["trip_set_sha256"]].add(row["replicate"])
        mip_pair_count = max(
            (len(values) for values in mip_by_trip_set.values()), default=0
        )
        matching_inventory = [row for row in verified if matches(row)]
        trip_sets = {
            row["trip_set_sha256"] for row in matching_inventory
            if row.get("trip_set_sha256")
        }
        if len(trip_sets) == 1:
            replicate_semantics = "same_trip_set_algorithm_replicates"
        elif len(trip_sets) > 1:
            replicate_semantics = (
                "different_trip_sets_not_interchangeable_algorithm_replicates"
            )
        else:
            replicate_semantics = "trip_set_identity_unavailable"
        required = expected["required_evidence"]
        minimum = int(expected.get("minimum_replicates", 3))
        if required == "trajectory":
            count = trajectory_count
            endpoint_only = endpoint_count > 0 and count == 0
        else:
            count = mip_pair_count
            endpoint_only = bool(final_runs) and not checkpoint_runs
        if count >= minimum:
            status = "available"
            reason = None
        elif endpoint_only:
            status = "endpoint_only_rerun_required"
            reason = "endpoint exists but time-resolved evidence is missing"
        elif count:
            status = "insufficient_replicates"
            reason = f"{count} verified replicates; {minimum} required"
        else:
            status = "not_available"
            matching_present = [
                row for row in verified if matches(row)
            ]
            reason = (
                "verified artifact exists but trip-set identity is unavailable"
                if matching_present and not trip_sets
                else "no verified artifact for the exact algorithm/scale slot"
            )
        row = {
            **expected,
            "size_class": expected.get("size_class") or (
                "k40" if str(expected["scale"]) == "40"
                else "small" if str(expected["scale"]) in {
                    "5", "8", "10", "13", "pairs", "single"
                } else "medium"
            ),
            "minimum_replicates": minimum,
            "trajectory_replicates": trajectory_count,
            "endpoint_replicates": endpoint_count,
            "mip_checkpoint_replicates": len(checkpoint_runs),
            "mip_final_replicates": len(final_runs),
            "verified_trip_sets": len(trip_sets),
            "replicate_semantics": replicate_semantics,
            "coverage_status": status,
            "availability_reason": reason,
        }
        coverage.append(row)
        runtimes = observed_runtime.get((
            expected["algorithm_family"],
            expected["implementation"],
            expected.get("treatment"),
            expected["scale_family"],
            expected["scale"],
        ), [])
        default_budget = {
            "5": 1800, "8": 3600, "10": 7200, "13": 10800,
            "15": 14400, "20": 21600, "30": 43200, "40": 82800,
        }.get(str(expected["scale"]), 7200)
        recommended = (
            max(default_budget, int(math.ceil(max(runtimes) * 1.25 / 300) * 300))
            if runtimes else default_budget
        )
        if expected["algorithm_family"] == "exact_expanded_network":
            stopping = (
                "pricing certification, explicit wall limit, and marginal-"
                "return stall retained as separate terminal outcomes"
            )
        elif expected["algorithm_family"] == "mip_finite_pool":
            stopping = (
                "two-stage finite-pool proof or explicit time limit; retain "
                "incumbent and bound checkpoints"
            )
        else:
            stopping = (
                "active-time budget plus declared marginal-return/label-cap "
                "rules; restricted-pricing certification labeled separately"
            )
        additional = max(0, minimum - count)
        if expected["algorithm_family"] == "exact_expanded_network":
            command = [
                "python", "-u", "src/exact_pricer_expanded.py",
                "--csv", "<hash-bound-instance.csv>",
                "--prices_csv", "hourly_prices_flat.csv",
                "--soc-step", "15", "--block-min", "10",
                "--g-kwh", "300", "--charge-kw", "300",
                "--min-soc-frac", "0",
                "--stall-window-min", "30",
                "--stall-rc-frac", "0.05",
                "--stall-obj-frac", "1e-5",
                "--wall-limit-s", str(recommended),
                "--out", "<new-exact-output.json>",
            ]
            command_available = False
            command_blocked_reason = (
                "resolve hash-bound instance/output paths before execution"
            )
        elif expected["algorithm_family"] == "heuristic_dp_current":
            command = [
                "python", "-u", "src/run_ex_unicorn.py",
                "--csv", "<hash-bound-instance.csv>",
                "--G", "300",
                "--prices_csv", "hourly_prices_flat.csv",
                "--queue_order", "<declared>",
                "--dominance_mode", "<declared>",
                "--active_time_limit_hours",
                f"{recommended / 3600:g}",
                "--skip_final_mip",
                "--results_root", "<new-results-root>",
            ]
            command_available = False
            command_blocked_reason = (
                "resolve declared queue/dominance and hash-bound paths"
            )
        elif expected["algorithm_family"] == "mip_finite_pool":
            command = [
                "python", "-u",
                "src/launch_mip_statistics_campaign.py",
                "--mode", "pilot",
                "--campaign", "<new-review-campaign>",
                "--plan-out", "<new-plan.json>",
            ]
            command_available = False
            command_blocked_reason = (
                "regenerate an unblocked reviewed plan with exact scale, "
                "treatment, starts and approved budget"
            )
        else:
            command = [
                "UNAVAILABLE",
                "historical implementation/model must be isolated or archived",
            ]
            command_available = False
            command_blocked_reason = (
                "historical implementation/model is not available"
            )
        reruns.append({
            **row,
            "rerun_required": status != "available",
            "additional_replicates": additional,
            "recommended_budget_s": recommended,
            "recommended_stopping_rule": stopping,
            "common_model_assumptions": (
                "300 kWh, 300 kW, zero reserve, 15 kWh SOC grid, "
                "10-minute blocks, flat tariff where supported; any pricing-"
                "graph/model difference remains an isolated method factor"
            ),
            "dry_run_only": True,
            "command_available": command_available,
            "command_blocked_reason": command_blocked_reason,
            "command_argv_json": json.dumps(command, separators=(",", ":")),
        })
    return coverage, reruns


def _write_csv(path: Path, fields, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _save_figures(staging, cg_rows, mip_rows, mip_summaries, coverage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pdf_metadata = {
        "Creator": "EVSP-DR cross-generation evidence",
        "CreationDate": None,
        "ModDate": None,
    }

    def save(fig, stem):
        fig.tight_layout()
        fig.savefig(
            staging / f"{stem}.png", dpi=160,
            metadata={"Software": "EVSP-DR"},
        )
        fig.savefig(staging / f"{stem}.pdf", metadata=pdf_metadata)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for run_id, rows in sorted(_group(cg_rows, "run_id").items()):
        values = [row for row in rows if row["wall_time_s"] is not None]
        values.sort(key=lambda row: row["wall_time_s"])
        if any(row["lp_route_weight"] is not None for row in values):
            axes[0].plot(
                [row["wall_time_s"] / 3600 for row in values],
                [row["lp_route_weight"] for row in values],
                marker=".", label=run_id,
            )
        if any(row["artificial_total"] is not None for row in values):
            axes[1].plot(
                [row["wall_time_s"] / 3600 for row in values],
                [row["artificial_total"] for row in values],
                marker=".", label=run_id,
            )
    axes[0].set_ylabel("LP route weight (not integer fleet)")
    axes[1].set_ylabel("Artificial mass")
    axes[1].set_xlabel("Wall time (hours)")
    axes[0].set_title("LP route weight and artificials remain separate")
    if axes[0].lines:
        axes[0].legend(fontsize=6)
    save(fig, "cg_route_weight_artificials")

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for run_id, rows in sorted(_group(cg_rows, "run_id").items()):
        values = [row for row in rows if row["wall_time_s"] is not None]
        values.sort(key=lambda row: row["wall_time_s"])
        if any(row["best_reduced_cost"] is not None for row in values):
            axes[0].plot(
                [row["wall_time_s"] / 3600 for row in values],
                [row["best_reduced_cost"] for row in values],
                marker=".", label=run_id,
            )
        if any(row["columns_added"] is not None for row in values):
            axes[1].step(
                [row["wall_time_s"] / 3600 for row in values],
                [row["columns_added"] for row in values],
                where="post", label=run_id,
            )
        if any(row["pool_columns_delta"] is not None for row in values):
            axes[1].step(
                [row["wall_time_s"] / 3600 for row in values],
                [row["pool_columns_delta"] for row in values],
                where="post", linestyle=":",
                label=f"{run_id} pool-size delta",
            )
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set_ylabel("Best reduced cost")
    axes[1].set_ylabel("Reported columns added / pool-size delta")
    axes[1].set_xlabel("Wall time (hours)")
    axes[0].set_title("Pricing progress")
    save(fig, "cg_reduced_cost_columns")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    measured = [
        row for row in cg_rows
        if row["master_time_s"] is not None
        and row["pricing_time_s"] is not None
    ]
    by_run = _group(measured, "run_id")
    labels, master, pricing = [], [], []
    for run_id, rows in sorted(by_run.items()):
        m = sum(row["master_time_s"] for row in rows)
        p = sum(row["pricing_time_s"] for row in rows)
        if m + p:
            labels.append(run_id)
            master.append(m / (m + p))
            pricing.append(p / (m + p))
    if labels:
        x = np.arange(len(labels))
        ax.bar(x, master, label="Measured master")
        ax.bar(x, pricing, bottom=master, label="Measured pricing")
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No measured master/pricing splits",
                transform=ax.transAxes, ha="center")
    ax.set_ylabel("Measured active-time share")
    ax.set_title("Master/pricing shares (measured schemas only)")
    save(fig, "cg_master_pricing_time_shares")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run_id, rows in sorted(_group(mip_rows, "run_id").items()):
        rows = [
            row for row in rows
            if row["solver_ended_before_checkpoint"] is False
        ]
        rows = sorted(rows, key=lambda row: (
            row["statistics_observed_s"]
            if row["statistics_observed_s"] is not None
            else row["observed_total_elapsed_s"]
        ))
        x = [
            (
                row["statistics_observed_s"]
                if row["statistics_observed_s"] is not None
                else row["observed_total_elapsed_s"]
            ) / 3600
            for row in rows
        ]
        ax.step(x, [row["incumbent_fleet"] for row in rows],
                where="post", label=f"{run_id} incumbent")
        ax.step(x, [row["fleet_bound"] for row in rows],
                where="post", linestyle="--", label=f"{run_id} bound")
    if mip_rows:
        ax.legend(fontsize=6)
    else:
        ax.text(0.5, 0.5, "No MIP checkpoints",
                transform=ax.transAxes, ha="center")
    ax.set(xlabel="MIP time (hours)", ylabel="Integer fleet / finite-pool bound",
           title="MIP incumbent and finite-pool fleet bound")
    save(fig, "mip_incumbent_fleet_bound")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    valid = [
        row for row in mip_summaries if row["integer_fleet"] is not None
    ]
    if valid:
        labels = [
            f"{row['scale']}:{row['implementation']}:{row['treatment']}"
            for row in valid
        ]
        x = np.arange(len(valid))
        ax.bar(x, [row["integer_fleet"] for row in valid])
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No verified final MIP fleets",
                transform=ax.transAxes, ha="center")
    ax.set(ylabel="Integer fleet", title="Final finite-pool MIP fleet by method")
    save(fig, "mip_final_fleet_by_scale_method")

    algorithms = sorted({
        f"{row['algorithm_family']}:{row['implementation']}:"
        f"{row.get('treatment') or '-'}"
        for row in coverage
    })
    scales = sorted({
        f"{row['scale_family']}:{row['scale']}" for row in coverage
    })
    matrix = np.full((len(algorithms), len(scales)), np.nan)
    status_values = {
        "not_available": 0,
        "endpoint_only_rerun_required": 1,
        "insufficient_replicates": 2,
        "available": 3,
    }
    for row in coverage:
        i = algorithms.index(
            f"{row['algorithm_family']}:{row['implementation']}:"
            f"{row.get('treatment') or '-'}"
        )
        j = scales.index(f"{row['scale_family']}:{row['scale']}")
        matrix[i, j] = status_values.get(row["coverage_status"], 0)
    fig, ax = plt.subplots(figsize=(max(8, len(scales) * 0.7), 5))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="lightgray")
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=3, cmap=cmap)
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels([
        "missing", "endpoint only", "insufficient reps", "available"
    ])
    ax.set_xticks(range(len(scales)), scales, rotation=45, ha="right")
    ax.set_yticks(range(len(algorithms)), algorithms)
    ax.set_title("Explicit artifact/missingness coverage matrix")
    save(fig, "artifact_coverage_matrix")


def _group(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[field]].append(row)
    return grouped


def _environment_identity():
    packages = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    executable = Path(sys.executable).resolve()
    return {
        "python_executable": str(executable),
        "python_executable_sha256": hashlib.sha256(
            executable.read_bytes()
        ).hexdigest(),
        "python_version": platform.python_version(),
        "packages": packages,
    }


def _git_identity(repo_root: Path):
    def run(*args):
        result = subprocess.run(
            ["git", *args], cwd=repo_root, text=True,
            capture_output=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _data_dictionary():
    definitions = {
        "legacy_master_objective": ("cg_iteration_long.csv", "number", None,
            "Historical heuristic-DP Master_Obj. Preserved separately; not "
            "silently equated to Master_Obj_Before_Add."
        ),
        "master_objective_before_add": ("cg_iteration_long.csv", "number", None,
            "Current heuristic-DP master objective before newly priced columns "
            "are added."
        ),
        "lp_route_weight": ("cg_iteration_long.csv", "number", "routes",
            "Restricted-master LP route-variable sum; not an integer fleet."
        ),
        "artificial_total": ("cg_iteration_long.csv", "number", "artificial mass",
            "Artificial mass. Zero is separate from pricing certification."
        ),
        "pricing_certified": ("cg_run_summary.csv", "boolean", None,
            "Explicit terminal pricing certificate only; never inferred from "
            "a missing/timeout row."
        ),
        "fleet_proven": ("mip_run_summary.csv", "boolean", None,
            "Integer fleet proven only over the named finite MIP pool."
        ),
        "physically_validated_schedule": ("mip_run_summary.csv", "boolean", None,
            "Selected integer routes include physically replayed schedules; "
            "separate from incidence partition feasibility."
        ),
        "master_time_share": ("cg_run_summary.csv", "number", "fraction",
            "Computed only from explicitly measured master/pricing fields."
        ),
        "pool_columns_delta": ("cg_iteration_long.csv", "number", "columns",
            "Change in recorded exact-pool size between adjacent rows; not "
            "the producer's per-iteration insertion count."
        ),
        "objective_bound_scope": ("mip_run_summary.csv", "string", None,
            "Scope of the finite-pool MIP objective bound."
        ),
        "censored": ("*_run_summary.csv", "boolean", None,
            "Target event was not observed by the verified run endpoint."
        ),
        "instance_sha256": ("artifact_inventory.csv", "string", None,
                            "Hash-bound instance identity; null when unavailable."),
        "trip_set_sha256": ("artifact_inventory.csv", "string", None,
                            "Ordered trip-set identity; distinct trip sets are not replicates."),
        "tariff_sha256": ("artifact_inventory.csv", "string", None,
                          "Hash-bound tariff identity; null when unavailable."),
        "git_dirty": ("artifact_inventory.csv", "boolean", None,
                      "Recorded source worktree state; null for legacy artifacts."),
        "termination_category": ("cg_run_summary.csv", "string", None,
                                 "Normalized terminal category with raw token retained separately."),
        "pool_scope": ("mip_run_summary.csv", "string", None,
                       "Finite RAW or GIRO-augmented pool; never global route space."),
    }
    return [{
        "field": field,
        "table": values[0],
        "type": values[1],
        "units": values[2],
        "definition": values[3],
        "availability": "null unless explicitly measured/recorded",
    } for field, values in sorted(definitions.items())]


def _publish_staging(staging: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    members = {}
    for source in sorted(staging.iterdir()):
        if not source.is_file():
            continue
        with source.open("rb") as handle:
            os.fsync(handle.fileno())
        members[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    completion = {
        "schema": "evsp-dr-cross-generation-output-completion-v1",
        "members": members,
    }
    completion_path = staging / "completion.json"
    with completion_path.open("x") as handle:
        json.dump(completion, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    staging_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(staging_fd)
    os.close(staging_fd)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication unavailable")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        -100, os.fsencode(staging),
        -100, os.fsencode(output),
        1,
    ) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"output directory exists: {output}")
        raise OSError(error, os.strerror(error), str(output))
    parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(parent_fd)
    os.close(parent_fd)


def build(input_manifest: Path, output_dir: Path, *, repo_root: Path,
          command: list[str], approved_manifest_sha256: str) -> dict:
    manifest_path = input_manifest.expanduser().absolute()
    repo = repo_root.expanduser().absolute()
    output = output_dir.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest, manifest_raw = _read_manifest(manifest_path)
    observed_manifest_sha = sha256_bytes(manifest_raw)
    if (
        not _valid_sha(approved_manifest_sha256)
        or observed_manifest_sha != approved_manifest_sha256
    ):
        raise ValueError("input manifest differs from approved SHA-256")
    _validate_specs(manifest)
    _validate_run_consistency(manifest)
    inventory = []
    cg_rows = []
    mip_rows = []
    mip_finals = []
    telemetry_rows = []
    endpoints = {}
    specs_by_run = defaultdict(list)
    artifact_hashes = {}
    parsed_by_artifact = {}
    for spec in sorted(manifest["artifacts"], key=lambda value: value["artifact_id"]):
        specs_by_run[spec["run_id"]].append(spec)
        path = _resolve_path(spec, manifest, manifest_path, repo)
        if not os.path.lexists(path):
            inventory.append(_flatten_inventory(
                spec, path, None, "missing",
                spec.get("availability_reason") or "artifact_not_found",
            ))
            continue
        mode = os.lstat(path).st_mode
        if (
            not stat.S_ISREG(mode)
            or stat.S_ISLNK(mode)
            or path.resolve() != path
        ):
            inventory.append(_flatten_inventory(
                spec, path, None, "rejected",
                "symlinked/non-canonical artifact path",
            ))
            continue
        raw = _read_regular_no_follow(path)
        observed = sha256_bytes(raw)
        artifact_hashes[spec["artifact_id"]] = observed
        if observed != spec["expected_sha256"]:
            inventory.append(_flatten_inventory(
                spec, path, observed, "rejected", "sha256_mismatch",
            ))
            continue
        try:
            parsed = parse_artifact(raw, spec)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            inventory.append(_flatten_inventory(
                spec, path, observed, "rejected", str(exc),
            ))
            continue
        mismatches = _embedded_provenance_mismatches(parsed, spec)
        if mismatches:
            inventory.append(_flatten_inventory(
                spec, path, observed, "rejected",
                "semantic provenance mismatch: " + " | ".join(mismatches),
            ))
            continue
        parsed_by_artifact[spec["artifact_id"]] = parsed
        inventory.append(_flatten_inventory(
            spec, path, observed, "verified", None, parsed
        ))
        cg_rows.extend(parsed.get("cg_rows") or [])
        mip_rows.extend(parsed.get("mip_rows") or [])
        mip_finals.extend(parsed.get("mip_finals") or [])
        telemetry_rows.extend(parsed.get("telemetry_rows") or [])
        if parsed.get("endpoint") is not None:
            endpoints[spec["run_id"]] = parsed["endpoint"]
    required_ids = {
        spec["artifact_id"] for spec in manifest["artifacts"]
        if spec.get("required") is True
    }
    failed_required = [
        row for row in inventory
        if row["artifact_id"] in required_ids
        and row["validation_status"] != "verified"
    ]
    rejected_present = [
        row for row in inventory
        if row["validation_status"] == "rejected"
    ]
    if failed_required or rejected_present:
        failures = failed_required or rejected_present
        raise ValueError(
            "listed artifacts failed validation: "
            + " | ".join(
                f"{row['artifact_id']}: {row['reason']}"
                for row in failures
            )
        )
    verified_by_hash = defaultdict(list)
    for row in inventory:
        if (
            row["validation_status"] == "verified"
            and row["observed_sha256"]
        ):
            verified_by_hash[row["observed_sha256"]].append(row)
    for final in mip_finals:
        status_rows = verified_by_hash.get(
            final.get("source_result_sha256"), []
        )
        journal_rows = verified_by_hash.get(
            final.get("source_journal_sha256"), []
        )
        if (
            not any(row["artifact_type"] in {
                "endpoint_json", "exact_cg_snapshot_json"
            } for row in status_rows)
            or not any(
                row["artifact_type"] == "exact_cg_column_journal_jsonl"
                for row in journal_rows
            )
        ):
            raise ValueError(
                f"MIP source pool artifacts are not verified: "
                f"{final['artifact_id']}"
            )
        if final.get("physically_validated_schedule"):
            spec = next(
                value for value in manifest["artifacts"]
                if value["artifact_id"] == final["artifact_id"]
            )
            replay_hash = (spec.get("metadata") or {}).get(
                "physical_replay_artifact_sha256"
            )
            replay_rows = [
                row
                for row in verified_by_hash.get(replay_hash, [])
                if row["artifact_type"] == "route_validation_json"
            ]
            if not replay_rows:
                raise ValueError(
                    f"MIP physical replay artifact is not verified: "
                    f"{final['artifact_id']}"
                )
            replay_vectors = {
                (parsed_by_artifact[row["artifact_id"]].get("checkpoint") or {})
                .get("route_vector_sha256")
                for row in replay_rows
            }
            if (
                final.get("physical_replay_route_vector_sha256")
                not in replay_vectors
            ):
                raise ValueError(
                    f"MIP physical replay route vector is not verified: "
                    f"{final['artifact_id']}"
                )
    telemetry_by_run = _group(telemetry_rows, "run_id")
    cg_summaries = _cg_summaries(
        cg_rows, endpoints, telemetry_by_run, specs_by_run,
        set(parsed_by_artifact),
    )
    hash_to_artifact_ids = defaultdict(list)
    for inventory_row in inventory:
        if (
            inventory_row["validation_status"] == "verified"
            and inventory_row["observed_sha256"]
        ):
            hash_to_artifact_ids[
                inventory_row["observed_sha256"]
            ].append(inventory_row["artifact_id"])
    for values in hash_to_artifact_ids.values():
        values.sort()
    mip_summaries = _mip_summaries(
        mip_rows, mip_finals, hash_to_artifact_ids
    )
    coverage, reruns = _coverage_and_reruns(
        manifest, inventory, cg_rows, mip_rows, mip_finals
    )
    staging = Path(tempfile.mkdtemp(
        dir=output.parent, prefix=f".{output.name}.tmp."
    ))
    try:
        _write_csv(staging / "artifact_inventory.csv", INVENTORY_FIELDS,
                   sorted(inventory, key=lambda row: row["artifact_id"]))
        _write_csv(staging / "cg_iteration_long.csv", CG_FIELDS,
                   sorted(cg_rows, key=lambda row: (
                       row["run_id"], row["iteration"], row["artifact_id"]
                   )))
        _write_csv(staging / "cg_run_summary.csv", CG_SUMMARY_FIELDS,
                   cg_summaries)
        _write_csv(staging / "mip_checkpoint_long.csv", MIP_CHECKPOINT_FIELDS,
                   sorted(mip_rows, key=lambda row: (
                       row["run_id"],
                       row["checkpoint_elapsed_s"]
                       if row["checkpoint_elapsed_s"] is not None else math.inf,
                   )))
        _write_csv(staging / "mip_run_summary.csv", MIP_SUMMARY_FIELDS,
                   mip_summaries)
        _write_csv(staging / "phase_telemetry_long.csv", TELEMETRY_FIELDS,
                   sorted(telemetry_rows, key=lambda row: (
                       row["run_id"], row["elapsed_session_s"]
                   )))
        _write_csv(staging / "data_dictionary.csv",
                   ("field", "table", "type", "units", "definition",
                    "availability"), _data_dictionary())
        _write_csv(staging / "artifact_coverage_matrix.csv", COVERAGE_FIELDS,
                   coverage)
        _write_csv(staging / "missing_data_and_rerun_plan.csv", RERUN_FIELDS,
                   reruns)
        _write_csv(staging / "coauthor_cg_endpoints.csv", CG_SUMMARY_FIELDS,
                   cg_summaries)
        _write_csv(staging / "coauthor_mip_endpoints.csv", MIP_SUMMARY_FIELDS,
                   mip_summaries)
        (staging / "input_manifest.json").write_bytes(manifest_raw)
        _save_figures(staging, cg_rows, mip_rows, mip_summaries, coverage)
        schema_text = """# Cross-generation normalized schema

- `legacy_master_objective` preserves historical `Master_Obj`.
- `master_objective_before_add` preserves current
  `Master_Obj_Before_Add`; the two are not silently equated.
- LP route weight is not an integer fleet.
- Zero artificials, pricing certification, physical schedule validation,
  finite-pool fleet proof and global route-space optimality are distinct.
- Master/pricing shares are populated only for schemas that measured both.
- Missing instrumentation remains null with an availability reason.
- RAW and GIRO-augmented pools are separate treatments.
- Time-to-event columns include explicit right-censoring fields.
"""
        (staging / "SCHEMA.md").write_text(schema_text)
        rerun_plan = {
            "schema": "evsp-dr-cross-generation-rerun-plan-v1",
            "dry_run_only": True,
            "input_manifest_sha256": observed_manifest_sha,
            "approved_input_manifest_sha256": approved_manifest_sha256,
            "jobs": reruns,
            "submits_jobs": False,
        }
        (staging / "benchmark_rerun_plan.json").write_text(
            json.dumps(rerun_plan, indent=2, sort_keys=True) + "\n"
        )
        provenance = {
            "schema": OUTPUT_SCHEMA,
            "input_manifest": "input_manifest.json",
            "input_manifest_sha256": sha256_bytes(manifest_raw),
            "source_artifact_hashes": artifact_hashes,
            "run_provenance": {
                run_id: {
                    key: (verified_specs[0].get("metadata") or {}).get(key)
                    for key in (
                        "algorithm_family", "implementation",
                        "git_commit", "git_dirty", "instance_sha256",
                        "trip_set_sha256", "tariff_sha256",
                        "model", "charging_discretization",
                        "battery_kwh", "charge_kw", "reserve_fraction",
                        "tariff", "master_sense", "initializer",
                        "replicate", "seed", "solver_backend",
                        "solver_versions", "python_environment_identity",
                        "time_limit_s", "memory_limit_bytes", "threads",
                        "stopping_rules", "tolerances",
                        "pool_status_sha256", "pool_journal_sha256",
                        "treatment", "giro_columns_added",
                    )
                }
                for run_id, specs in sorted(specs_by_run.items())
                if (
                    verified_specs := [
                        spec for spec in specs
                        if spec["artifact_id"] in parsed_by_artifact
                    ]
                )
            },
            "git": _git_identity(repo),
            "environment": _environment_identity(),
            "generation_command_argv": command,
            "model_semantic_guards": {
                "lp_route_weight_is_integer_fleet": False,
                "zero_artificials_equals_pricing_certification": False,
                "pricing_certification_equals_finite_pool_proof": False,
                "finite_pool_proof_equals_global_optimality": False,
                "incidence_partition_equals_physical_schedule": False,
                "raw_equals_giro_augmented_pool": False,
            },
            "outputs": {},
        }
        for path in sorted(staging.iterdir()):
            if path.name == "provenance.json":
                continue
            provenance["outputs"][path.name] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        _publish_staging(staging, output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "schema": OUTPUT_SCHEMA,
        "output_dir": str(output),
        "verified_artifacts": sum(
            row["validation_status"] == "verified" for row in inventory
        ),
        "rejected_artifacts": sum(
            row["validation_status"] == "rejected" for row in inventory
        ),
        "missing_artifacts": sum(
            row["validation_status"] == "missing" for row in inventory
        ),
        "cg_iterations": len(cg_rows),
        "mip_checkpoints": len(mip_rows),
        "rerun_slots": sum(row["rerun_required"] for row in reruns),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--approved-input-manifest-sha256", required=True
    )
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args(argv)
    command = [
        "python", "-u", "src/build_cross_generation_evidence.py",
        "--input-manifest", "<INPUT_MANIFEST>",
        "--approved-input-manifest-sha256",
        args.approved_input_manifest_sha256,
        "--out-dir", "<OUTPUT_DIR>",
        "--repo-root", "<REPO_ROOT>",
    ]
    result = build(
        args.input_manifest,
        args.out_dir,
        repo_root=args.repo_root,
        command=command,
        approved_manifest_sha256=args.approved_input_manifest_sha256,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed validation for the controlled RAW/GIRO40 k40 campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from launch_mip_statistics_campaign import (
    GIRO40_AUGMENTED,
    GIRO40_PARTITION_FILE_SHA256,
    K40_CS_FROZEN_HASHES,
    K40_CS_GIRO40_BUDGET_HOURS,
    K40_CS_LABELS,
    K40_CS_OVERNIGHT_MODE,
    K40_CS_PACKAGING_BASE_COMMIT,
    K40_CS_RAW_BUDGET_HOURS,
    RAW_K40_INSTANCE_SHA256,
    RAW_K40_PHYSICAL_CODE_HASHES,
    RAW_K40_PHYSICAL_COMMIT,
    RAW_K40_SOURCE_COMMIT,
    RAW_K40_TARIFF_SHA256,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_plan(plan: dict, *, expected_commit: str) -> list[dict]:
    _require(
        plan.get("schema") == "evsp-dr-mip-statistics-approved-plan-v1",
        "unexpected plan schema",
    )
    _require(
        plan.get("mode") == K40_CS_OVERNIGHT_MODE,
        "plan is not the controlled overnight mode",
    )
    _require(plan.get("blocked") is False, "plan is blocked")
    identity = plan.get("checkout_identity") or {}
    _require(
        identity.get("expected_commit") == expected_commit,
        "plan commit differs from reviewed commit",
    )
    _require(identity.get("detached") is True, "checkout is not detached")
    _require(
        identity.get("tracked_clean") is True,
        "checkout is not tracked-clean",
    )
    _require(
        identity.get("runtime_artifacts_absent") is True,
        "checkout has unreviewed runtime artifacts",
    )
    _require(
        plan.get("fresh_exact_cg_preparations") == [],
        "plan contains a CG preparation",
    )
    resources = plan.get("resources") or {}
    _require(resources.get("partition") == "scaglione", "wrong partition")
    _require(resources.get("threads") == 8, "wrong thread count")
    _require(resources.get("requeue") is False, "requeue enabled")
    _require(
        resources.get("submission_release")
        == "single_held_four_task_array",
        "jobs are not held for one atomic release",
    )
    _require(
        resources.get("array_tasks") == 4
        and resources.get("array_slurm_wall_time") == "08:10:00",
        "Slurm array shape/wall guard changed",
    )
    _require(
        resources.get("signal") == "B:USR1@180",
        "graceful signal guard changed",
    )
    _require(
        plan.get("physical_realization_review") == {
            "semantics_base_commit": RAW_K40_PHYSICAL_COMMIT,
            "packaging_base_commit": K40_CS_PACKAGING_BASE_COMMIT,
            "runtime_code_hashes": {
                path: (plan.get("code_hashes") or {}).get(path)
                for path in RAW_K40_PHYSICAL_CODE_HASHES
            },
        },
        "corrected physical gate is not hash-bound",
    )
    _require(
        plan.get("k40_cs_overnight_guards") == {
            "raw_budget_seconds": 28800,
            "giro40_augmented_budget_seconds": 7200,
            "raw_external_routes_allowed": False,
            "giro40_partition_route_count": 40,
            "strict_partitioning": True,
            "expected_trip_count": 947,
            "expected_snapshot_minutes": 1440,
            "continuous_cost_pricing_certified": False,
            "ca_jobs_included": False,
        },
        "overnight guard block changed",
    )
    _require(
        set(plan.get("selected_candidates") or {}) == set(K40_CS_LABELS),
        "selected sources are not R1_CS/R2_CS",
    )
    jobs = plan.get("jobs") or []
    _require(len(jobs) == 4, "overnight plan must contain four jobs")
    seen = set()
    summaries = []
    start_hashes = set()
    for job in jobs:
        source = job.get("source") or {}
        execution = job.get("execution") or {}
        label = source.get("raw_k40_label")
        arm = job.get("arm")
        key = (label, arm)
        _require(label in K40_CS_LABELS, "CA or unknown source included")
        _require(
            arm in {"RAW", GIRO40_AUGMENTED},
            "scientific treatment label changed",
        )
        _require(key not in seen, f"duplicate campaign cell: {key}")
        seen.add(key)
        _require(
            source.get("status_sha256")
            == K40_CS_FROZEN_HASHES[label]["status"],
            f"{label}: frozen status hash changed",
        )
        _require(
            source.get("journal_sha256")
            == K40_CS_FROZEN_HASHES[label]["journal"],
            f"{label}: frozen journal hash changed",
        )
        _require(
            source.get("instance_sha256") == RAW_K40_INSTANCE_SHA256
            and source.get("tariff_sha256") == RAW_K40_TARIFF_SHA256,
            f"{label}: frozen data/tariff hash changed",
        )
        _require(
            source.get("source_commit") == RAW_K40_SOURCE_COMMIT,
            f"{label}: source CG commit changed",
        )
        _require(
            source.get("trip_count") == 947
            and source.get("snapshot_mark_minutes") == 1440,
            f"{label}: source scale/snapshot changed",
        )
        treatment = source.get("treatment") or {}
        _require(
            treatment.get("master_sense") == "cover"
            and treatment.get("initial_pool") == "singletons",
            f"{label}: source treatment changed",
        )
        expected_seconds = int((
            K40_CS_RAW_BUDGET_HOURS
            if arm == "RAW" else K40_CS_GIRO40_BUDGET_HOURS
        ) * 3600)
        _require(
            job.get("time_limit_s") == expected_seconds
            and execution.get("time_limit_s") == expected_seconds,
            f"{key}: wrong Gurobi budget",
        )
        _require(
            job.get("threads") == 8
            and execution.get("threads") == 8,
            f"{key}: wrong thread count",
        )
        _require(
            job.get("partitioning") == "strict_exact_once"
            and job.get("two_stage") is True,
            f"{key}: solver formulation changed",
        )
        _require(
            job.get("cost_stage_policy") == (
                "run_only_after_finite_pool_fleet_proof"
                if arm == "RAW"
                else "disabled_for_mixed_augmented_cost_semantics"
            ),
            f"{key}: route-cost stage policy changed",
        )
        _require(
            len(str(job.get("job_name") or "")) <= 15,
            f"{key}: Slurm name exceeds 15 characters",
        )
        if arm == "RAW":
            _require(
                job.get("augmentation_changes_column_set") is False
                and job.get("validated_start") is None
                and execution.get("validated_start") is None,
                f"{label}: RAW received GIRO columns",
            )
        else:
            start = job.get("validated_start") or {}
            _require(
                job.get("augmentation_changes_column_set") is True,
                f"{label}: GIRO40 is not labeled AUGMENTED",
            )
            _require(
                start.get("route_count") == 40
                and start.get("validated_bus_count") == 40
                and start.get("physical_replay_validated") is True,
                f"{label}: GIRO40 partition is not fully validated",
            )
            _require(
                start.get("sha256") == GIRO40_PARTITION_FILE_SHA256,
                f"{label}: GIRO40 partition hash changed",
            )
            _require(
                execution.get("validated_start_sha256")
                == start.get("sha256"),
                f"{label}: staged start identity differs",
            )
            start_hashes.add(start.get("sha256"))
        summaries.append({
            "label": label,
            "treatment": arm,
            "cell_id": job.get("cell_id"),
            "job_name": job.get("job_name"),
            "time_limit_s": expected_seconds,
            "status_sha256": source.get("status_sha256"),
            "journal_sha256": source.get("journal_sha256"),
            "validated_start_sha256": execution.get(
                "validated_start_sha256"
            ),
        })
    _require(
        seen == {
            (label, arm)
            for label in K40_CS_LABELS
            for arm in ("RAW", GIRO40_AUGMENTED)
        },
        "four-cell treatment matrix is incomplete",
    )
    _require(
        len(start_hashes) == 1,
        "R1/R2 GIRO40 cells must use the same frozen partition bytes",
    )
    return sorted(
        summaries, key=lambda row: (row["label"], row["treatment"])
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    summaries = validate_plan(
        json.loads(args.plan.read_text()),
        expected_commit=args.expected_commit,
    )
    print(json.dumps({"validated": True, "jobs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

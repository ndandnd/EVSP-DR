#!/usr/bin/env python3
"""Fail-closed validation for the four-cell RAW k40 MIP campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from launch_mip_statistics_campaign import (
    RAW_K40_BUDGET_HOURS,
    RAW_K40_INSTANCE_SHA256,
    RAW_K40_SOURCE_COMMIT,
    RAW_K40_SPECS,
    RAW_K40_TARIFF_SHA256,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_plan(plan: dict, *, expected_commit: str) -> list[dict]:
    """Return a concise cell summary only when every launch guard holds."""

    _require(
        plan.get("schema") == "evsp-dr-mip-statistics-approved-plan-v1",
        "unexpected plan schema",
    )
    _require(plan.get("mode") == "raw_k40", "plan is not raw_k40")
    _require(plan.get("blocked") is False, "plan is blocked")
    _require(
        plan.get("fresh_exact_cg_preparations") == [],
        "raw campaign contains fresh preparation commands",
    )
    identity = plan.get("checkout_identity") or {}
    _require(
        identity.get("expected_commit") == expected_commit,
        "plan commit differs from requested commit",
    )
    _require(identity.get("detached") is True, "checkout is not detached")
    _require(
        identity.get("tracked_clean") is True,
        "checkout is not tracked-clean",
    )
    resources = plan.get("resources") or {}
    _require(resources.get("partition") == "scaglione", "wrong partition")
    _require(resources.get("threads") == 8, "wrong thread count")
    _require(resources.get("requeue") is False, "requeue is enabled")
    guards = plan.get("raw_k40_guards") or {}
    _require(
        guards
        == {
            "giro_columns_allowed": False,
            "extra_routes_allowed": False,
            "initial_partition_allowed": False,
            "strict_partitioning": True,
            "budget_seconds": RAW_K40_BUDGET_HOURS * 3600,
            "expected_trip_count": 947,
            "expected_snapshot_minutes": 1440,
        },
        "raw-k40 guard block changed",
    )
    _require(
        set(plan.get("selected_candidates") or {}) == set(RAW_K40_SPECS),
        "selected raw-k40 cells differ from the four approved cells",
    )

    jobs = plan.get("jobs") or []
    _require(len(jobs) == 4, "raw-k40 plan must contain four jobs")
    summaries = []
    observed_labels = set()
    for job in jobs:
        source = job.get("source") or {}
        execution = job.get("execution") or {}
        label = source.get("raw_k40_label")
        _require(label in RAW_K40_SPECS, "unknown raw-k40 source label")
        _require(label not in observed_labels, f"duplicate source label: {label}")
        observed_labels.add(label)
        spec = RAW_K40_SPECS[label]
        status = Path(str(source.get("status_path") or ""))
        _require(
            status.name == spec["filename"]
            and status.parent.name == spec["campaign"],
            f"{label}: source path differs from frozen campaign cell",
        )
        _require(job.get("matrix") == "raw_k40", f"{label}: wrong matrix")
        _require(job.get("arm") == "RAW", f"{label}: not a RAW arm")
        _require(
            job.get("augmentation_changes_column_set") is False,
            f"{label}: column augmentation is enabled",
        )
        _require(
            job.get("partitioning") == "strict_exact_once",
            f"{label}: partitioning is not strict",
        )
        _require(job.get("two_stage") is True, f"{label}: two-stage disabled")
        _require(
            job.get("time_limit_s") == RAW_K40_BUDGET_HOURS * 3600,
            f"{label}: wrong time limit",
        )
        _require(job.get("threads") == 8, f"{label}: wrong threads")
        _require(job.get("mip_gap") == 1e-4, f"{label}: wrong MIP gap")
        _require(
            job.get("validated_start") is None
            and job.get("staged_start") is None,
            f"{label}: external partition start is present",
        )
        _require(job.get("blocked_reasons") == [], f"{label}: cell is blocked")
        _require(source.get("scale") == 40, f"{label}: wrong scale")
        _require(source.get("trip_count") == 947, f"{label}: wrong trip count")
        _require(
            source.get("instance_sha256") == RAW_K40_INSTANCE_SHA256,
            f"{label}: wrong instance hash",
        )
        _require(
            source.get("tariff_sha256") == RAW_K40_TARIFF_SHA256,
            f"{label}: wrong tariff hash",
        )
        _require(
            source.get("source_commit") == RAW_K40_SOURCE_COMMIT,
            f"{label}: wrong pricing source commit",
        )
        _require(
            source.get("snapshot_mark_minutes") == 1440,
            f"{label}: wrong snapshot age",
        )
        _require(
            (source.get("treatment") or {}).get("master_sense") == "cover"
            and (source.get("treatment") or {}).get("initial_pool")
            == spec["initial_pool"],
            f"{label}: wrong source treatment",
        )
        _require(execution.get("arm") == "RAW", f"{label}: execution not RAW")
        _require(
            execution.get("validated_start") is None
            and execution.get("validated_start_sha256") is None,
            f"{label}: execution contains an external start",
        )
        _require(
            execution.get("time_limit_s") == RAW_K40_BUDGET_HOURS * 3600,
            f"{label}: execution time limit differs",
        )
        _require(
            execution.get("source_label") == label
            and execution.get("source_master_sense") == "cover"
            and execution.get("source_initial_pool") == spec["initial_pool"],
            f"{label}: execution source metadata differs",
        )
        summaries.append({
            "label": label,
            "job_name": job.get("job_name"),
            "cell_id": job.get("cell_id"),
            "initial_pool": spec["initial_pool"],
            "lp_weight": source.get("route_weight"),
            "columns": source.get("columns"),
            "time_limit_s": execution.get("time_limit_s"),
        })
    _require(observed_labels == set(RAW_K40_SPECS), "raw-k40 cells are incomplete")
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = json.loads(args.plan.read_text())
    summaries = validate_plan(plan, expected_commit=args.expected_commit)
    print(json.dumps({"validated": True, "jobs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

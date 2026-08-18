#!/usr/bin/env python3
"""Validate one completed controlled k40 RAW/GIRO40 MIP result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from launch_mip_statistics_campaign import GIRO40_AUGMENTED


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} is missing")


def validate_result(
    result: dict,
    *,
    progress_dir: Path,
    arm: str,
    time_limit_s: int,
    source_label: str,
) -> None:
    if source_label not in {"R1_CS", "R2_CS"}:
        raise ValueError("source is not an approved CS replicate")
    expected_limit = 28800 if arm == "RAW" else 7200
    if arm not in {"RAW", GIRO40_AUGMENTED}:
        raise ValueError("scientific treatment label is invalid")
    if time_limit_s != expected_limit:
        raise ValueError("8h/2h treatment budget changed")
    mip_provenance = result.get("mip_provenance") or {}
    arguments = mip_provenance.get("arguments") or {}
    if (
        arguments.get("cover") is not False
        or arguments.get("two_stage") is not True
        or int(arguments.get("timelimit", -1)) != time_limit_s
        or int(arguments.get("threads", -1)) != 8
        or float(arguments.get("mipgap", math.nan)) != 1e-4
        or int(result.get("source_snapshot_mark_minutes", -1)) != 1440
    ):
        raise ValueError("solver/source arguments changed")
    parameters = mip_provenance.get("gurobi_parameters") or {}
    if (
        not mip_provenance.get("python")
        or not mip_provenance.get("gurobi")
        or not mip_provenance.get("host")
        or parameters.get("Seed") is None
        or parameters.get("seed_source") != "gurobi_default"
        or parameters.get("seed_explicitly_set") is not False
    ):
        raise ValueError("solver environment/parameter provenance is missing")
    try:
        node_count = float(result.get("node_count"))
        solution_count = int(result.get("solution_count"))
    except (TypeError, ValueError):
        raise ValueError("node/solution statistics are missing") from None
    if (
        not math.isfinite(node_count)
        or node_count < 0
        or solution_count < 1
    ):
        raise ValueError("node/solution statistics are invalid")
    physical = result.get("physical_pool_audit") or {}
    if (
        int(physical.get("rejected_columns", -1)) != 0
        or int(physical.get("accepted_columns", -1))
        != int(physical.get("total_columns", -2))
    ):
        raise ValueError("physical pool gate rejected columns")
    for field in (
        "physical_pool_preparation_wall_s",
        "source_hashing_wall_s",
        "gurobi_optimize_wall_s",
        "end_to_end_before_publication_s",
    ):
        try:
            finite = math.isfinite(float(result.get(field)))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(f"timing missing: {field}")
    if result.get("continuous_cost_pricing_certified") is not False:
        raise ValueError("continuous-cost pricing certificate was claimed")
    if result.get("incumbent_found") is not True:
        raise ValueError("CS result lacks a physical incumbent")
    start = result.get("mip_start") or {}
    base_count = int(physical.get("base_pool_column_count", -1))
    augmented_count = int(
        physical.get("augmented_pool_column_count", -1)
    )
    _sha(physical.get("base_pool_ordered_sha256"), "base pool identity")
    _sha(
        physical.get("augmented_pool_ordered_sha256"),
        "augmented pool identity",
    )
    if arm == "RAW":
        if (
            start.get("source") is not None
            or arguments.get("initial_partition_routes") is not None
            or result.get("extra_route_sources") != []
            or int(physical.get("added_giro_route_count", -1)) != 0
            or base_count != augmented_count
        ):
            raise ValueError("RAW cell received GIRO columns")
    else:
        if (
            start.get("kind") != "validated_exact_partition"
            or start.get("validated_bus_count") != 40
            or start.get("assigned_mip_start_route_count") != 40
            or int(physical.get("added_giro_route_count", -1)) != 40
            or int(physical.get("assigned_mip_start_route_count", -1)) != 40
            or augmented_count != base_count + 40
            or (start.get("solver_acceptance") or {}).get("accepted")
            is not True
        ):
            raise ValueError("GIRO40 augmentation/start was not accepted")
        _sha(
            physical.get("added_giro_route_set_sha256"),
            "GIRO40 route identity",
        )
    required = {60, 300, 900, 1800, 3600, 7200}
    if time_limit_s == 28800:
        required.update(range(10800, 28801, 3600))
    missing = [
        mark for mark in sorted(required)
        if not (
            progress_dir / f"checkpoint_{mark // 60:04d}m.json"
        ).is_file()
    ]
    if missing:
        raise ValueError(f"checkpoint cadence is incomplete: {missing}")
    if not (progress_dir / "final.json").is_file():
        raise ValueError("progress final.json is missing")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("RAW", GIRO40_AUGMENTED), required=True
    )
    parser.add_argument("--time-limit-s", type=int, required=True)
    parser.add_argument(
        "--source-label", choices=("R1_CS", "R2_CS"), required=True
    )
    args = parser.parse_args(argv)
    validate_result(
        json.loads(args.result.read_text()),
        progress_dir=args.progress_dir,
        arm=args.arm,
        time_limit_s=args.time_limit_s,
        source_label=args.source_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

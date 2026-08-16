#!/usr/bin/env python3
"""Read-only validated summary of repeated k40 factorial campaigns."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from k40_factorial_artifacts import (
    ARMS,
    HISTORICAL_WEIGHT,
    MARKS,
    validate_campaign,
    validate_historical,
)


FIELDS = (
    "row_type", "replicate", "campaign", "arm", "master_sense",
    "initial_pool", "checkpoint", "nominal_minutes", "actual_wall_s",
    "actual_hours", "iterations", "columns", "objective", "route_weight",
    "artificials", "min_reduced_cost", "real_lp_feasible",
    "feasible_route_weight", "certified", "stop_reason",
    "final_lp_source",
    "historical_delta",
    "actual_wall_s_n", "objective_n", "route_weight_n",
    "artificials_n", "min_reduced_cost_n",
    "feasible_route_weight_n", "historical_delta_n",
    "actual_wall_s_min", "actual_wall_s_max",
    "objective_min", "objective_max",
    "route_weight_min", "route_weight_max",
    "artificials_min", "artificials_max",
    "min_reduced_cost_min", "min_reduced_cost_max",
    "feasible_route_weight_min", "feasible_route_weight_max",
    "historical_delta_min", "historical_delta_max",
)


def _number_summary(values, *, expected_count: int):
    values = [float(value) for value in values if value is not None]
    if len(values) != expected_count:
        return None, None, None, len(values)
    return statistics.mean(values), min(values), max(values), len(values)


def summarize(campaign_dirs: list[Path], historical_path: Path) -> dict:
    if len(campaign_dirs) != 2:
        raise ValueError("exactly two factorial campaign directories required")
    resolved = [path.expanduser().resolve() for path in campaign_dirs]
    if len(resolved) != len(set(resolved)):
        raise ValueError("campaign directories must be distinct")
    historical = validate_historical(historical_path)
    campaigns = [
        validate_campaign(path, replicate=f"R{index}")
        for index, path in enumerate(resolved, start=1)
    ]
    identities = {
        campaign["trip_set_sha256"] for campaign in campaigns
    } | {historical["trip_set_sha256"]}
    if len(identities) != 1:
        raise ValueError(
            "factorial replicates and historical comparator have mixed trip sets"
        )
    if set(campaigns[0]["job_ids"]) & set(campaigns[1]["job_ids"]):
        raise ValueError("factorial replicates reuse Slurm job IDs")
    rows = [row for campaign in campaigns for row in campaign["rows"]]
    for row in rows:
        row["historical_delta"] = (
            row["feasible_route_weight"] - HISTORICAL_WEIGHT
            if row["checkpoint"] == "m1320"
            and row["feasible_route_weight"] is not None
            else None
        )

    aggregates = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["checkpoint"])].append(row)
    for (arm, checkpoint), group in sorted(grouped.items()):
        aggregate = {
            "row_type": "aggregate",
            "replicate": "mean",
            "campaign": "all",
            "arm": arm,
            "master_sense": group[0]["master_sense"],
            "initial_pool": group[0]["initial_pool"],
            "checkpoint": checkpoint,
            "nominal_minutes": group[0]["nominal_minutes"],
            "real_lp_feasible": all(
                row["real_lp_feasible"] for row in group
            ),
            "certified": all(row["certified"] for row in group),
            "stop_reason": "replicate_aggregate",
        }
        for field in (
            "actual_wall_s", "actual_hours", "iterations", "columns",
            "objective", "route_weight", "artificials",
            "min_reduced_cost", "feasible_route_weight", "historical_delta",
        ):
            mean, minimum, maximum, count = _number_summary(
                (row.get(field) for row in group),
                expected_count=len(campaigns),
            )
            aggregate[field] = mean
            aggregate[f"{field}_min"] = minimum
            aggregate[f"{field}_max"] = maximum
            aggregate[f"{field}_n"] = count
        aggregates.append(aggregate)

    certified_rows = sum(row["certified"] for row in rows)
    conclusions = [
        (
            f"{certified_rows}/{len(rows)} replicate/checkpoint rows explicitly "
            "record certified_rc_optimal=true."
        ),
        (
            "Route weight is reported independently of artificials; "
            "feasible_route_weight is null whenever artificials exceed 1e-6."
        ),
        (
            "Deltas from 39.252026205592166 are reported only for real "
            "artificial-free m1320 LP rows using recorded actual wall_s."
        ),
        (
            "These are restricted-master LP diagnostics, not integer bus "
            "schedules or global route-space optimality claims."
        ),
    ]
    for arm in ARMS:
        arm_rows = [
            row for row in rows
            if row["arm"] == arm and row["checkpoint"] == "m1320"
        ]
        feasible = [
            row for row in arm_rows
            if row["feasible_route_weight"] is not None
        ]
        if len(feasible) == len(arm_rows):
            weights = [row["feasible_route_weight"] for row in feasible]
            deltas = [row["historical_delta"] for row in feasible]
            hours = [row["actual_hours"] for row in feasible]
            conclusions.append(
                f"At m1320, {arm} has two artificial-free restricted-master "
                f"LP observations: route-weight mean "
                f"{statistics.mean(weights):.9f}, range "
                f"[{min(weights):.9f}, {max(weights):.9f}], historical-delta "
                f"mean {statistics.mean(deltas):+.9f}, at recorded-hour range "
                f"[{min(hours):.4f}, {max(hours):.4f}]."
            )
        else:
            conclusions.append(
                f"At m1320, {arm} is artificial-free in "
                f"{len(feasible)}/{len(arm_rows)} replicates; the remaining "
                "route weights do not support a feasible fleet comparison."
            )
    if any(
            row["arm"] == "PA"
            and row["route_weight"] is not None
            and not row["real_lp_feasible"]
            for row in rows):
        conclusions.append(
            "PA has route-weight observations while artificials remain; those "
            "values are not presented as feasible LP fleet weights."
        )
    return {
        "schema": "evsp-dr-k40-factorial-summary-v1",
        "historical": historical,
        "campaigns": campaigns,
        "replicate_rows": rows,
        "aggregate_rows": aggregates,
        "conclusions": conclusions,
    }


def _csv_bytes(payload: dict) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(payload["replicate_rows"])
    writer.writerows(payload["aggregate_rows"])
    return stream.getvalue().encode()


def _markdown_bytes(payload: dict) -> bytes:
    lines = [
        "# Validated k40 factorial summary",
        "",
        (
            f"Historical comparator: route weight "
            f"{payload['historical']['route_weight']:.15f} at "
            f"{payload['historical']['actual_hours']:.3f} recorded hours."
        ),
        "",
        "## Replicate m1320 rows",
        "",
        "| Replicate | Arm | Actual h | Route weight | Artificials | "
        "Feasible weight | Min RC | Certified | Delta |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in payload["replicate_rows"]:
        if row["checkpoint"] != "m1320":
            continue
        def show(value):
            return "" if value is None else f"{value:.9f}"
        lines.append(
            f"| {row['replicate']} | {row['arm']} | "
            f"{row['actual_hours']:.4f} | {show(row['route_weight'])} | "
            f"{show(row['artificials'])} | "
            f"{show(row['feasible_route_weight'])} | "
            f"{show(row['min_reduced_cost'])} | {row['certified']} | "
            f"{show(row['historical_delta'])} |"
        )
    lines.extend(["", "## Supported conclusions", ""])
    lines.extend(f"- {conclusion}" for conclusion in payload["conclusions"])
    lines.append("")
    return "\n".join(lines).encode()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.tmp.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite output: {path}") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def publish(payload: dict, output_prefix: Path) -> dict:
    prefix = output_prefix.expanduser().resolve()
    bundle = Path(str(prefix) + ".bundle")
    outputs = {
        "json": bundle / "summary.json",
        "csv": bundle / "summary.csv",
        "markdown": bundle / "summary.md",
    }
    lock = Path(str(bundle) + ".lock")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(bundle) or os.path.lexists(lock):
        raise FileExistsError(f"output bundle/lock already exists: {bundle}")
    descriptor = os.open(
        lock,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0),
        0o600,
    )
    os.close(descriptor)
    staging = Path(tempfile.mkdtemp(
        dir=bundle.parent, prefix=f".{bundle.name}.tmp."
    ))
    try:
        _write_new(
            staging / "summary.json",
            (json.dumps(payload, indent=2) + "\n").encode(),
        )
        _write_new(staging / "summary.csv", _csv_bytes(payload))
        _write_new(staging / "summary.md", _markdown_bytes(payload))
        directory = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if os.path.lexists(bundle):
            raise FileExistsError(f"refusing to overwrite bundle: {bundle}")
        os.rename(staging, bundle)
        parent = os.open(bundle.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        # The lock intentionally remains: an interrupted attempt needs a new
        # prefix rather than an ambiguous in-place retry.
        raise
    return {key: str(path) for key, path in outputs.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = summarize(args.campaign_dir, args.historical)
    outputs = publish(payload, args.out_prefix)
    print(json.dumps({"outputs": outputs, **payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

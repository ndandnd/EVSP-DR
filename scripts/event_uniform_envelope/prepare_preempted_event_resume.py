#!/usr/bin/env python3
"""Stage one immutable continuation of a preempted medium event-CG cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--solver-commit", required=True)
    parser.add_argument("--wall-limit-s", type=float, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    out_root = args.out_root.resolve()
    if out_root.exists():
        raise SystemExit(f"resume root already exists: {out_root}")

    source_plan_path = required_file(
        source_root / "execution_plan.json", "source execution plan"
    )
    source_plan = json.loads(source_plan_path.read_text())
    if source_plan.get("solver_commit") != args.solver_commit:
        raise SystemExit("source solver commit does not match launcher")
    source_cap = float(source_plan.get("wall_limit_s", 0.0))
    if abs(source_cap - args.wall_limit_s) > 1e-9:
        raise SystemExit(
            f"source wall cap {source_cap} does not match {args.wall_limit_s}"
        )

    matrix_path = required_file(source_root / "matrix.tsv", "source matrix")
    with matrix_path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not (0 <= args.source_index < len(rows)):
        raise SystemExit(f"source index outside matrix: {args.source_index}")
    fields = rows[args.source_index]
    if len(fields) != 11:
        raise SystemExit(f"unexpected source matrix width: {len(fields)}")
    (
        index_text,
        cell,
        scale,
        _replicate,
        _trips,
        instance_csv,
        instance_sha256,
        representation,
        soc_step,
        block_min,
        row_wall_limit,
    ) = fields
    if int(index_text) != args.source_index:
        raise SystemExit("source matrix index mismatch")
    instance_path = required_file(Path(instance_csv), "instance CSV")
    if sha256(instance_path) != instance_sha256:
        raise SystemExit("instance CSV hash changed")
    if abs(float(row_wall_limit) - args.wall_limit_s) > 1e-9:
        raise SystemExit("source row wall cap mismatch")

    source_status = required_file(
        source_root / "cg" / f"M__{cell}__{representation}.json",
        "preempted status",
    )
    status = json.loads(source_status.read_text())
    if status.get("certified_rc_optimal") is True:
        raise SystemExit("source cell is already certified")
    if status.get("stop_reason") != "external_signal":
        raise SystemExit(
            "source cell is not a clean signal checkpoint: "
            f"{status.get('stop_reason')!r}"
        )
    wall_s = float(status.get("wall_s", 0.0))
    if not (0.0 < wall_s < args.wall_limit_s - 120.0):
        raise SystemExit(
            f"source wall time {wall_s} is not an incomplete capped run"
        )
    if status.get("csv") != str(instance_path):
        raise SystemExit("source status instance path mismatch")
    if status.get("time_model") != "event":
        raise SystemExit("source status is not the event model")

    source_journal_text = status.get("columns_journal")
    if not source_journal_text:
        raise SystemExit("source status lacks columns_journal")
    source_journal = required_file(
        Path(source_journal_text), "preempted columns journal"
    )
    source_iters = required_file(
        Path(str(source_status) + ".iters.csv"), "preempted iteration log"
    )
    source_telemetry = required_file(
        Path(str(source_status) + ".phase-telemetry.jsonl"),
        "preempted phase telemetry",
    )

    (out_root / "cg").mkdir(parents=True)
    (out_root / "logs").mkdir()
    destination = out_root / "cg" / source_status.name
    destination_journal = Path(str(destination) + ".columns.jsonl")
    destination_iters = Path(str(destination) + ".iters.csv")
    archived_telemetry = Path(
        str(destination) + ".source-phase-telemetry.jsonl"
    )
    for source, target in (
        (source_status, destination),
        (source_journal, destination_journal),
        (source_iters, destination_iters),
        (source_telemetry, archived_telemetry),
    ):
        shutil.copyfile(source, target)

    manifest_rows = [{
        "local_index": 0,
        "source_panel_index": args.source_index,
        "cell": cell,
        "target_fleet": scale,
        "instance_csv": str(instance_path),
        "representation_id": representation,
        "soc_step": soc_step,
        "block_min": block_min,
        "source_status": str(source_status),
        "source_status_sha256": sha256(source_status),
        "source_journal": str(source_journal),
        "source_journal_sha256": sha256(source_journal),
        "resume_status": str(destination),
        "resume_journal": str(destination_journal),
        "staged_status_sha256": sha256(destination),
        "staged_journal_sha256": sha256(destination_journal),
    }]
    manifest_path = out_root / "matrix.tsv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(manifest_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    (out_root / "execution_plan.json").write_text(
        json.dumps({
            "schema": "evsp-dr-preempted-event-resume-v1",
            "source_root": str(source_root),
            "source_execution_plan_sha256": sha256(source_plan_path),
            "source_matrix_sha256": sha256(matrix_path),
            "source_index": args.source_index,
            "source_stop_reason": "external_signal",
            "source_wall_s": wall_s,
            "solver_commit": args.solver_commit,
            "cells": 1,
            "cumulative_scientific_wall_limit_s": args.wall_limit_s,
            "max_iters": 10000,
            "columns_per_iter": 30,
            "telemetry_policy": (
                "archive the source telemetry and start a fresh "
                "identity-bound continuation stream"
            ),
            "preserves_original_artifacts": True,
        }, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"staged preempted event continuation: index={args.source_index} "
        f"cell={cell} wall_s={wall_s:.3f}/{args.wall_limit_s:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage a new immutable CG continuation from capped resume artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-resume-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--panel", choices=("A", "B"), required=True)
    parser.add_argument("--solver-commit", required=True)
    parser.add_argument("--wall-limit-s", type=float, required=True)
    parser.add_argument("--expected-cells", type=int, required=True)
    parser.add_argument("--representation")
    parser.add_argument("--max-iters", type=int, default=10000)
    args = parser.parse_args()
    source_root = args.source_resume_root.resolve()
    out_root = args.out_root.resolve()
    if out_root.exists():
        raise SystemExit(f"resume root already exists: {out_root}")
    parent_plan_path = source_root / "execution_plan.json"
    parent_plan = json.loads(parent_plan_path.read_text())
    if parent_plan.get("panel") != args.panel:
        raise SystemExit("parent resume panel mismatch")
    if parent_plan.get("solver_commit") != args.solver_commit:
        raise SystemExit("parent resume solver commit mismatch")
    parent_cap = float(parent_plan["cumulative_scientific_wall_limit_s"])
    if args.wall_limit_s <= parent_cap:
        raise SystemExit("child cumulative wall limit must exceed parent cap")
    selected = []
    with (source_root / "matrix.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            original = Path(row["source_status"])
            original_journal = Path(row["source_journal"])
            source = Path(row["resume_status"])
            source_journal = Path(row["resume_journal"])
            if (
                not original.is_file()
                or sha256(original) != row["source_status_sha256"]
                or not original_journal.is_file()
                or sha256(original_journal) != row["source_journal_sha256"]
            ):
                raise SystemExit(
                    f"parent source identity changed at {row['local_index']}"
                )
            if not source.is_file() or not source_journal.is_file():
                raise SystemExit(
                    f"missing parent resume artifact at {row['local_index']}"
                )
            status = json.loads(source.read_text())
            wall_s = float(status.get("wall_s", 0.0))
            if status.get("certified_rc_optimal") is True:
                continue
            if not (
                status.get("stop_reason") == "wall_limit"
                and wall_s >= parent_cap - 120.0
            ):
                raise SystemExit(
                    f"parent row {row['local_index']} is neither certified "
                    "nor at its cumulative wall cap"
                )
            if (
                args.representation
                and row["representation_id"] != args.representation
            ):
                continue
            selected.append((row, source, source_journal))
    if len(selected) != args.expected_cells:
        raise SystemExit(
            f"expected {args.expected_cells} capped parent cells, "
            f"found {len(selected)}"
        )
    (out_root / "cg").mkdir(parents=True)
    (out_root / "logs").mkdir()
    manifest_rows = []
    for local_index, (parent, source, source_journal) in enumerate(selected):
        destination = out_root / "cg" / source.name
        destination_journal = Path(str(destination) + ".columns.jsonl")
        companions = [
            (source, destination),
            (source_journal, destination_journal),
            (
                Path(str(source) + ".iters.csv"),
                Path(str(destination) + ".iters.csv"),
            ),
        ]
        source_telemetry = Path(str(source) + ".phase-telemetry.jsonl")
        if source_telemetry.is_file():
            companions.append((
                source_telemetry,
                Path(str(destination) + ".source-phase-telemetry.jsonl"),
            ))
        for source_path, destination_path in companions:
            if not source_path.is_file():
                raise SystemExit(f"missing continuation companion: {source_path}")
            shutil.copyfile(source_path, destination_path)
        manifest_rows.append({
            "local_index": local_index,
            "source_panel_index": parent["source_panel_index"],
            "cell": parent["cell"],
            "target_fleet": parent["target_fleet"],
            "instance_csv": parent["instance_csv"],
            "representation_id": parent["representation_id"],
            "soc_step": parent["soc_step"],
            "block_min": parent["block_min"],
            "source_status": str(source),
            "source_status_sha256": sha256(source),
            "source_journal": str(source_journal),
            "source_journal_sha256": sha256(source_journal),
            "resume_status": str(destination),
            "resume_journal": str(destination_journal),
            "staged_status_sha256": sha256(destination),
            "staged_journal_sha256": sha256(destination_journal),
        })
    with (out_root / "matrix.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(manifest_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    (out_root / "execution_plan.json").write_text(json.dumps({
        "schema": "evsp-dr-extended-cg-resume-v2",
        "panel": args.panel,
        "source_root": str(source_root),
        "parent_execution_plan_sha256": sha256(parent_plan_path),
        "parent_cumulative_wall_limit_s": parent_cap,
        "solver_commit": args.solver_commit,
        "cells": len(manifest_rows),
        "selection_stop_reason": "wall_limit_at_parent_cap",
        "selection_representation": args.representation,
        "cumulative_scientific_wall_limit_s": args.wall_limit_s,
        "max_iters": args.max_iters,
        "columns_per_iter": 30,
        "telemetry_policy": (
            "archive parent live telemetry separately; start a fresh "
            "identity-bound stream for the child continuation"
        ),
        "preserves_original_artifacts": True,
    }, indent=2, sort_keys=True) + "\n")
    print(
        f"staged {args.panel} continuation cells: "
        + ", ".join(
            row["cell"] + "/" + row["representation_id"]
            for row in manifest_rows
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

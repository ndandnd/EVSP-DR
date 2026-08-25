#!/usr/bin/env python3
"""Stage immutable CG copies for a distinct extended-resume campaign."""

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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--panel", choices=("A", "B"), required=True)
    parser.add_argument("--solver-commit", required=True)
    parser.add_argument("--wall-limit-s", type=float, required=True)
    parser.add_argument("--expected-cells", type=int, required=True)
    parser.add_argument("--representation")
    args = parser.parse_args()
    root = args.root.resolve()
    out_root = args.out_root.resolve()
    if out_root.exists():
        raise SystemExit(f"resume root already exists: {out_root}")
    rows = list(csv.reader((root / "matrix.tsv").open(), delimiter="\t"))
    selected = []
    for fields in rows:
        index, cell, target, instance_csv, rep = fields[:5]
        if args.panel == "A":
            soc, block = fields[6:8]
        else:
            soc, block = fields[5:7]
        source = root / "cg" / f"{args.panel}__{cell}__{rep}.json"
        status = json.loads(source.read_text())
        if status.get("stop_reason") != "wall_limit":
            continue
        if args.representation and rep != args.representation:
            continue
        selected.append((fields, soc, block, source, status))
    if len(selected) != args.expected_cells:
        raise SystemExit(
            f"expected {args.expected_cells} selected censored cells, "
            f"found {len(selected)}"
        )
    (out_root / "cg").mkdir(parents=True)
    (out_root / "logs").mkdir()
    manifest_rows = []
    for local_index, (fields, soc, block, source, status) in enumerate(selected):
        source_index, cell, target, instance_csv, rep = fields[:5]
        destination = out_root / "cg" / source.name
        source_journal = Path(status["columns_journal"]).resolve(strict=True)
        destination_journal = Path(str(destination) + ".columns.jsonl")
        companions = [
            (source, destination),
            (source_journal, destination_journal),
            (Path(str(source) + ".iters.csv"), Path(str(destination) + ".iters.csv")),
        ]
        source_telemetry = Path(str(source) + ".phase-telemetry.jsonl")
        if source_telemetry.is_file():
            companions.append((
                source_telemetry,
                Path(str(destination) + ".phase-telemetry.jsonl"),
            ))
        for source_path, destination_path in companions:
            if not source_path.is_file():
                raise SystemExit(f"missing resume companion: {source_path}")
            shutil.copyfile(source_path, destination_path)
        manifest_rows.append({
            "local_index": local_index,
            "source_panel_index": source_index,
            "cell": cell,
            "target_fleet": target,
            "instance_csv": instance_csv,
            "representation_id": rep,
            "soc_step": soc,
            "block_min": block,
            "source_status": str(source),
            "source_status_sha256": sha256(source),
            "source_journal": str(source_journal),
            "source_journal_sha256": sha256(source_journal),
            "resume_status": str(destination),
            "resume_journal": str(destination_journal),
            "staged_status_sha256": sha256(destination),
            "staged_journal_sha256": sha256(destination_journal),
        })
    manifest = out_root / "matrix.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(manifest_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    (out_root / "execution_plan.json").write_text(json.dumps({
        "schema": "evsp-dr-extended-cg-resume-v1",
        "panel": args.panel,
        "source_root": str(root),
        "solver_commit": args.solver_commit,
        "cells": len(manifest_rows),
        "selection_stop_reason": "wall_limit",
        "selection_representation": args.representation,
        "cumulative_scientific_wall_limit_s": args.wall_limit_s,
        "max_iters": 10000,
        "columns_per_iter": 30,
        "preserves_original_artifacts": True,
    }, indent=2, sort_keys=True) + "\n")
    print(
        f"staged {args.panel} resume cells: "
        + ", ".join(row["cell"] + "/" + row["representation_id"]
                    for row in manifest_rows)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

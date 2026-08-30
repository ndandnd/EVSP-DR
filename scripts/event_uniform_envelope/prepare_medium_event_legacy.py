#!/usr/bin/env python3
"""Freeze the 18 existing k8/k13/k20 duty unions for event-CG probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo", type=Path, required=True)
    parser.add_argument("--solver-repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-commit", required=True)
    parser.add_argument("--solver-commit", required=True)
    parser.add_argument("--wrapper-commit", required=True)
    args = parser.parse_args()
    input_repo = args.input_repo.resolve()
    solver_repo = args.solver_repo.resolve()
    root = args.root.resolve()
    if root.exists():
        raise SystemExit(f"campaign root already exists: {root}")
    manifest = input_repo / "data/scale_ladder/instances/scale_ladder_instance_manifest_6sel_seed20260803.csv"
    with manifest.open(newline="") as handle:
        selected = [
            row for row in csv.DictReader(handle)
            if int(row["scale"]) in {8, 13, 20}
        ]
    selected.sort(key=lambda row: (int(row["scale"]), int(row["selection_replicate"])))
    if len(selected) != 18:
        raise SystemExit(f"expected 18 medium legacy rows, found {len(selected)}")
    if {int(row["scale"]) for row in selected} != {8, 13, 20}:
        raise SystemExit("medium scale set mismatch")
    root.mkdir(parents=True)
    for name in ("cg", "logs"):
        (root / name).mkdir()
    matrix = root / "matrix.tsv"
    with matrix.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for index, row in enumerate(selected):
            path = input_repo / row["relative_path"]
            observed = sha256(path)
            if observed != row["instance_file_sha256"]:
                raise SystemExit(f"input hash mismatch: {path}")
            scale = int(row["scale"])
            replicate = int(row["selection_replicate"])
            writer.writerow([
                index,
                f"k{scale:02d}_s{replicate}",
                scale,
                replicate,
                row["trip_count"],
                str(path),
                observed,
                "event_2p5_event5",
                "2.5",
                "5",
                "43200",
            ])
    code_paths = (
        "src/exact_pricer_expanded.py",
        "src/event_pricer_network.py",
        "src/exact_cg_telemetry.py",
    )
    plan = {
        "schema": "evsp-dr-medium-event-corrected-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wrapper_commit": args.wrapper_commit,
        "solver_commit": args.solver_commit,
        "input_commit": args.input_commit,
        "input_manifest": str(manifest),
        "input_manifest_sha256": sha256(manifest),
        "cells": 18,
        "scales": [8, 13, 20],
        "replicates_per_scale": 6,
        "physics": {
            "battery_kwh": 240,
            "charge_kw": 240,
            "min_soc_fraction": 0,
            "tariff": "flat",
        },
        "representation": "event_2p5_event5",
        "time_model": "event",
        "event_arc_mode": "lazy",
        "launch_contract": (
            "medium_event_cg.sub requires EVSP_TIME_MODEL=event and "
            "EVSP_EVENT_ARC_MODE=lazy"
        ),
        "wall_limit_s": 43200,
        "code_sha256": {path: sha256(solver_repo / path) for path in code_paths},
    }
    (root / "execution_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    print(f"prepared {len(selected)} immutable medium event rows under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

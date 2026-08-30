#!/usr/bin/env python3
"""Freeze nine new small cells and four upper-scale event-CG probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_row(row: dict[str, str]) -> bool:
    scale = int(row["scale"])
    replicate = int(row["selection_replicate"])
    return (
        (scale in {2, 3, 5} and replicate in {4, 5, 6})
        or scale in {30, 40}
    )


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
        selected = [row for row in csv.DictReader(handle) if selected_row(row)]
    selected.sort(key=lambda row: (int(row["scale"]), int(row["selection_replicate"])))
    counts = Counter(int(row["scale"]) for row in selected)
    if counts != Counter({2: 3, 3: 3, 5: 3, 30: 3, 40: 1}):
        raise SystemExit(f"unexpected extension rows: {dict(sorted(counts.items()))}")
    root.mkdir(parents=True)
    for name in ("cg", "logs"):
        (root / name).mkdir()
    with (root / "matrix.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for index, row in enumerate(selected):
            path = input_repo / row["relative_path"]
            observed = sha256(path)
            if observed != row["instance_file_sha256"]:
                raise SystemExit(f"input hash mismatch: {path}")
            scale = int(row["scale"])
            replicate = int(row["selection_replicate"])
            writer.writerow([
                index, f"k{scale:02d}_s{replicate}", scale, replicate,
                row["trip_count"], str(path), observed,
                "event_2p5_event5", "2.5", "5", "43200",
            ])
    code_paths = (
        "src/exact_pricer_expanded.py",
        "src/event_pricer_network.py",
        "src/exact_cg_telemetry.py",
    )
    plan = {
        "schema": "evsp-dr-event-extension-corrected-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wrapper_commit": args.wrapper_commit,
        "solver_commit": args.solver_commit,
        "input_commit": args.input_commit,
        "input_manifest": str(manifest),
        "input_manifest_sha256": sha256(manifest),
        "cells": len(selected),
        "selection": {
            "small_extension": "targets 2/3/5, replicates 4/5/6",
            "upper_boundary": "all three target-30 plus target-40",
        },
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
    print(f"prepared {len(selected)} immutable extension rows under {root}")
    print(f"scale counts: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

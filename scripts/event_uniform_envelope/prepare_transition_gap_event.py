#!/usr/bin/env python3
"""Freeze the k3/k4/k6/k7 transition-gap event-CG campaign."""

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


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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

    source = (
        input_repo / "data" / "scale_ladder" / "instances"
        / "transition_gap_20260902"
    )
    manifest = source / "selection_manifest.csv"
    candidate_universe = source / "candidate_universe.csv"
    duty_certificates = source / "known_duty_continuous_240_240.csv"
    exclusion_manifest = source / "excluded_existing_scale_ladder_manifest.csv"
    input_plan_path = source / "input_plan.json"
    input_plan = json.loads(input_plan_path.read_text(encoding="utf-8"))
    expected_files = {
        "selection_manifest.csv": manifest,
        "candidate_universe.csv": candidate_universe,
        "known_duty_continuous_240_240.csv": duty_certificates,
        "excluded_existing_scale_ladder_manifest.csv": exclusion_manifest,
    }
    for name, path in expected_files.items():
        if sha256(path) != input_plan["files"][name]:
            raise SystemExit(f"transition input hash mismatch: {name}")

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (
        int(row["scale"]), int(row["selection_replicate"])
    ))
    counts = Counter((int(row["scale"]), row["sample_family"]) for row in rows)
    expected_counts = Counter({
        **{(scale, "probability"): 6 for scale in (3, 4, 6, 7)},
        **{(scale, "stress"): 3 for scale in (3, 4, 6, 7)},
    })
    if len(rows) != 36 or counts != expected_counts:
        raise SystemExit(f"unexpected transition selection: {counts}")
    if len({row["duty_set_sha256"] for row in rows}) != len(rows):
        raise SystemExit("duplicate selected duty set")
    if not all(
        truth(row["known_partition_continuous_physical_upper_bound"])
        for row in rows
    ):
        raise SystemExit("selected row lacks current-physics k-route upper bound")

    root.mkdir(parents=True)
    for name in ("cg", "logs"):
        (root / name).mkdir()
    matrix = root / "matrix.tsv"
    with matrix.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for index, row in enumerate(rows):
            instance = input_repo / row["relative_path"]
            if sha256(instance) != row["instance_file_sha256"]:
                raise SystemExit(f"instance hash mismatch: {instance}")
            writer.writerow([
                index,
                row["cell_id"],
                row["scale"],
                row["selection_replicate"],
                row["trip_count"],
                str(instance),
                row["instance_file_sha256"],
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
        "schema": "evsp-dr-transition-gap-event-cg-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wrapper_commit": args.wrapper_commit,
        "solver_commit": args.solver_commit,
        "input_commit": args.input_commit,
        "input_plan_sha256": sha256(input_plan_path),
        "selection_manifest_sha256": sha256(manifest),
        "candidate_universe_sha256": sha256(candidate_universe),
        "known_duty_continuous_240_240_sha256": sha256(duty_certificates),
        "cells": len(rows),
        "scales": [3, 4, 6, 7],
        "sample_counts_per_scale": {"probability": 6, "stress": 3},
        "stress_roles": ["trip_heavy", "energy_heavy", "tight_gap"],
        "selection_uses_solver_outcomes": False,
        "raw_cg_receives_known_partition_routes": False,
        "known_partition_scope": input_plan["known_partition_scope"],
        "known_partition_caveat": input_plan["known_partition_caveat"],
        "physics": {
            "battery_kwh": 240.0,
            "charge_kw": 240.0,
            "min_soc_fraction": 0.0,
            "tariff": "flat",
        },
        "representation": "event_2p5_event5",
        "time_model": "event",
        "event_arc_mode": "lazy",
        "initial_pool": "singletons",
        "wall_limit_s": 43200,
        "code_sha256": {
            path: sha256(solver_repo / path) for path in code_paths
        },
    }
    (root / "execution_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {len(rows)} immutable transition-gap rows under {root}")
    print(f"scale/family counts: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

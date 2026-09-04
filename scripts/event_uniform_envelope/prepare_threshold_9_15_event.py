#!/usr/bin/env python3
"""Freeze the pre-outcome k9--k15 event-CG threshold campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SCALES = (9, 10, 11, 12, 13, 14, 15)
ARM = {
    "arm": "b030_reduced",
    "columns_per_iter": 30,
    "selection": "reduced_cost",
    "diversity_weight": 0.0,
}


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
        / "threshold_9_15_20260904"
    )
    manifest = source / "selection_manifest.csv"
    input_plan_path = source / "input_plan.json"
    input_plan = json.loads(input_plan_path.read_text(encoding="utf-8"))
    if (
        input_plan.get("schema") != "evsp-dr-threshold-9-15-inputs-v1"
        or input_plan.get("scales") != list(SCALES)
        or input_plan.get("selected_rows") != 70
        or sha256(manifest)
        != input_plan["files"]["selection_manifest.csv"]
    ):
        raise SystemExit("k9--k15 input-plan identity mismatch")

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (
        int(row["scale"]), int(row["selection_replicate"])
    ))
    counts = Counter((int(row["scale"]), row["sample_family"]) for row in rows)
    expected_counts = Counter({
        **{(scale, "probability"): 6 for scale in SCALES},
        **{(scale, "structural"): 4 for scale in SCALES},
    })
    if len(rows) != 70 or counts != expected_counts:
        raise SystemExit(f"unexpected k9--k15 selection: {counts}")
    if len({row["duty_set_sha256"] for row in rows}) != len(rows):
        raise SystemExit("duplicate selected duty set")
    if not all(
        truth(row["known_partition_continuous_physical_upper_bound"])
        for row in rows
    ):
        raise SystemExit("selected row lacks current-physics k-route upper bound")

    root.mkdir(parents=True)
    for name in ("network_cache", "logs/cache", "logs/cg"):
        (root / name).mkdir(parents=True)
    (root / "cg" / ARM["arm"]).mkdir(parents=True)
    frozen_manifest = root / "input_selection_manifest.csv"
    frozen_manifest.write_bytes(manifest.read_bytes())

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
        "schema": "evsp-dr-threshold-9-15-event-cg-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wrapper_commit": args.wrapper_commit,
        "solver_commit": args.solver_commit,
        "input_commit": args.input_commit,
        "input_plan_sha256": sha256(input_plan_path),
        "input_selection_manifest_sha256": sha256(frozen_manifest),
        "cells": len(rows),
        "scales": list(SCALES),
        "sample_counts_per_scale": {"probability": 6, "structural": 4},
        "structural_roles": [
            "trip_light", "trip_heavy", "energy_heavy", "tight_gap"
        ],
        "selection_uses_solver_outcomes": False,
        "raw_cg_receives_known_partition_routes": False,
        "known_partition_scope": input_plan["known_partition_scope"],
        "known_partition_caveat": input_plan["known_partition_caveat"],
        "arms": [ARM],
        "cache_policy": (
            "one immutable hash-validated event network per instance; "
            "the baseline CG task starts only after its matching cache succeeds"
        ),
        "network_cache_slurm_limit_s": 86400,
        "wall_limit_s_per_cg_arm": 43200,
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
        "code_sha256": {
            path: sha256(solver_repo / path) for path in code_paths
        },
    }
    (root / "execution_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {len(rows)} immutable k9--k15 rows under {root}")
    print(f"scale/family counts: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

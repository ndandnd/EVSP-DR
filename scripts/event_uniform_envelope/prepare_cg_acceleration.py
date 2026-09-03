#!/usr/bin/env python3
"""Freeze the k13/k20 inputs and controlled CG-acceleration arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ARMS = (
    ("b030_reduced", 30, "reduced_cost", 0.0),
    ("b060_reduced", 60, "reduced_cost", 0.0),
    ("b120_reduced", 120, "reduced_cost", 0.0),
    ("b200_reduced", 200, "reduced_cost", 0.0),
    ("b120_complementary", 120, "complementary", 0.5),
    ("b200_complementary", 200, "complementary", 0.5),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    manifest = input_repo / (
        "data/scale_ladder/instances/"
        "scale_ladder_instance_manifest_6sel_seed20260803.csv"
    )
    with manifest.open(newline="", encoding="utf-8") as handle:
        selected = [
            row for row in csv.DictReader(handle)
            if int(row["scale"]) in {13, 20}
        ]
    selected.sort(key=lambda row: (
        int(row["scale"]), int(row["selection_replicate"])
    ))
    if len(selected) != 12:
        raise SystemExit(f"expected 12 k13/k20 rows, found {len(selected)}")

    root.mkdir(parents=True)
    for name in ("network_cache", "logs/cache", "logs/cg"):
        (root / name).mkdir(parents=True)
    for arm, _columns, _selection, _weight in ARMS:
        (root / "cg" / arm).mkdir(parents=True)

    matrix = root / "matrix.tsv"
    with matrix.open("w", newline="", encoding="utf-8") as handle:
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
        "schema": "evsp-dr-cg-acceleration-factorial-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wrapper_commit": args.wrapper_commit,
        "solver_commit": args.solver_commit,
        "input_commit": args.input_commit,
        "input_manifest": str(manifest),
        "input_manifest_sha256": sha256(manifest),
        "scales": [13, 20],
        "replicates_per_scale": 6,
        "arms": [
            {
                "arm": arm,
                "columns_per_iter": columns,
                "selection": selection,
                "diversity_weight": weight,
            }
            for arm, columns, selection, weight in ARMS
        ],
        "cache_policy": (
            "one immutable hash-validated event network per instance; "
            "all six CG arms require the same completed cache"
        ),
        "wall_limit_s_per_cg_arm": 43200,
        "physics": {
            "battery_kwh": 240,
            "charge_kw": 240,
            "min_soc_fraction": 0,
            "tariff": "flat",
        },
        "representation": "event_2p5_event5",
        "code_sha256": {
            path: sha256(solver_repo / path) for path in code_paths
        },
    }
    (root / "execution_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"prepared 12 inputs and {len(ARMS)} arms under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

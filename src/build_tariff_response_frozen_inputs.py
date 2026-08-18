#!/usr/bin/env python3
"""Build the exact frozen k5/k8/k40 tariff-response input manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

from build_tariff_response_manifest import REPO_ROOT, sha256_file
from make_duty_pair_instances import (
    _base_task,
    load_duty_frames,
    merge_duties,
)


SEED = 20260803
REPLICATE = 2
MASTER_SHA256 = (
    "6b46acce8b0870aff967c73aac372b90873ed32a6e424e55b851e4b8676ab57f"
)
K40_SHA256 = (
    "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd"
)
OUTPUT_DIR = REPO_ROOT / "data/tariff_response/frozen_instances"
FIELDS = (
    "scale", "replicate", "relative_path", "file_sha256",
    "trip_count", "trip_set_sha256", "ordered_trip_sequence_sha256",
    "duty_count", "duties_json", "duty_set_sha256",
    "source_master_sha256", "generator_seed", "generator_family",
)


def canonical_sha(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def generated_replicates(frames, duties, sizes, *, per_size=6):
    rng = random.Random(SEED)
    selected = {}
    for scale in sizes:
        made = attempts = 0
        seen = set()
        while made < per_size and attempts < 20000:
            attempts += 1
            sample = rng.sample(duties, scale)
            if len({_base_task(duty) for duty in sample}) < scale:
                continue
            key = frozenset(sample)
            if key in seen:
                continue
            seen.add(key)
            made += 1
            if made == REPLICATE:
                selected[scale] = sorted(sample)
    if set(selected) != set(sizes):
        raise ValueError("could not reproduce requested deterministic cells")
    return selected


def build(output_dir: Path = OUTPUT_DIR):
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    master = REPO_ROOT / "data/Par_VehicleDetails_Updated.csv"
    if sha256_file(master) != MASTER_SHA256:
        raise ValueError("GIRO master identity changed")
    frames = load_duty_frames()
    duties = sorted(frames)
    small = generated_replicates(frames, duties, (3, 5, 8, 13))
    large = generated_replicates(frames, duties, (15, 20, 30, 40))
    selections = {5: small[5], 8: small[8], 40: large[40]}
    output_dir.mkdir(parents=True)
    rows = []
    for scale in (5, 8, 40):
        selected = selections[scale]
        if len({_base_task(duty) for duty in selected}) != scale:
            raise ValueError("frozen input contains weekday siblings")
        frame = merge_duties(frames, selected)
        path = output_dir / (
            f"Practice_Custom_DutyUnion_k{scale:02d}_r{REPLICATE}.csv"
        )
        frame.to_csv(path, index=False, lineterminator="\n")
        file_sha = sha256_file(path)
        if scale == 40 and file_sha != K40_SHA256:
            raise ValueError(
                f"k40 regeneration differs: {file_sha} != {K40_SHA256}"
            )
        source_trips = [
            int(value) for value in frame["Ordered_Trip_ID"]
        ]
        rows.append({
            "scale": scale,
            "replicate": REPLICATE,
            "relative_path": str(path.relative_to(REPO_ROOT)),
            "file_sha256": file_sha,
            "trip_count": len(frame),
            "trip_set_sha256": canonical_sha(sorted(source_trips)),
            "ordered_trip_sequence_sha256": canonical_sha(source_trips),
            "duty_count": len(selected),
            "duties_json": json.dumps(
                selected, separators=(",", ":")
            ),
            "duty_set_sha256": canonical_sha(selected),
            "source_master_sha256": MASTER_SHA256,
            "generator_seed": SEED,
            "generator_family": (
                "small_3_5_8_13_per6"
                if scale in {5, 8}
                else "large_15_20_30_40_per6"
            ),
        })
    manifest = output_dir / "frozen_input_manifest.csv"
    with manifest.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return manifest, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest, rows = build(args.out_dir)
    print(json.dumps({
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "instances": rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

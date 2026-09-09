#!/usr/bin/env python3
"""Build deterministic same-type Partille k2/k3 CG cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from audit_giro_duty_recovery import load_duty_frames_from
from giro_partille_physics import profile_for_duty


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_with_sources(frames, duties):
    parts = []
    for duty in duties:
        part = frames[duty].copy()
        part["Source_Duty"] = duty
        parts.append(part)
    merged = pd.concat(parts, ignore_index=True)

    def minutes(value):
        hour, minute = str(value).split(":")
        return int(hour) * 60 + int(minute)

    merged = (
        merged.assign(_sort=merged["Start1"].map(minutes))
        .sort_values(["_sort", "Ordered_Trip_ID"])
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    merged["count_trip_id"] = range(len(merged))
    if not merged["Ordered_Trip_ID"].is_unique:
        duplicates = merged.loc[
            merged["Ordered_Trip_ID"].duplicated(), "Ordered_Trip_ID"
        ].tolist()
        raise ValueError(f"cohort repeats Ordered_Trip_ID values: {duplicates[:10]}")
    return merged


def build(master: Path, output_dir: Path) -> dict:
    frames = load_duty_frames_from(master)
    by_profile = {
        profile: sorted(
            (duty for duty in frames if profile_for_duty(duty).name == profile),
            key=lambda duty: (len(frames[duty]), duty),
        )
        for profile in ("18E1", "18E2")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for profile, duties in by_profile.items():
        tag = f"e{profile[3:]}"
        for side, ordered in (("short", duties), ("long", list(reversed(duties)))):
            for k in (2, 3):
                selected = ordered[:k]
                if len({str(duty).rstrip("muwt") for duty in selected}) != k:
                    raise ValueError("cohort contains variants of one base duty")
                cell_id = f"{tag}_{side}_k{k}"
                frame = merge_with_sources(frames, selected)
                path = output_dir / f"{cell_id}.csv"
                frame.to_csv(path, index=False)
                cells.append({
                    "cell_id": cell_id,
                    "k": k,
                    "vehicle_profile": profile,
                    "cohort": side,
                    "duties": selected,
                    "trip_count": len(frame),
                    "instance": path.name,
                    "instance_sha256": sha256(path),
                })
    manifest = {
        "schema": "evsp-dr-giro-small-cg-cohorts-v1",
        "master": str(master.resolve()),
        "master_sha256": sha256(master),
        "selection": (
            "shortest/longest literal duties within each Partille vehicle group; "
            "no mixed type and no weekday-variant siblings"
        ),
        "cells": cells,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--master", type=Path,
        default=repo / "data/Par_VehicleDetails_Updated.csv",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(args.master.resolve(), args.out_dir.resolve())
    print(json.dumps({
        "cell_count": len(manifest["cells"]),
        "cells": [row["cell_id"] for row in manifest["cells"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

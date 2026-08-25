#!/usr/bin/env python3
"""Emit manifest indices whose integer-stage artifact is missing or invalid."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


TARGET_OUTCOMES = {"FEASIBLE", "INFEASIBLE", "TIME_LIMIT"}


def classify(path: Path, stage: str) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        return "missing"
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return "malformed"
    if not isinstance(payload, dict):
        return "malformed"
    if stage == "mip":
        return "valid" if payload.get("status_name") else "invalid_schema"
    return (
        "valid"
        if payload.get("outcome") in TARGET_OUTCOMES
        else "invalid_schema"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", choices=("A", "B"), required=True)
    parser.add_argument("--stage", choices=("mip", "target"), required=True)
    args = parser.parse_args()
    counts = Counter()
    missing = []
    with args.manifest.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            stem = (
                f"{args.panel}__{row['cell']}__"
                f"{row['representation_id']}.json"
            )
            state = classify(args.root / args.stage / stem, args.stage)
            counts[state] += 1
            if state != "valid":
                missing.append(row["index"])
    print(
        f"{args.panel} {args.stage} artifact validation: "
        + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)),
        file=sys.stderr,
    )
    print("\n".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

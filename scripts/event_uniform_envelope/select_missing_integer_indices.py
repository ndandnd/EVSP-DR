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


def classify(path: Path, stage: str, row: dict) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        return "missing"
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return "malformed"
    if not isinstance(payload, dict):
        return "malformed"
    if stage == "mip":
        if not payload.get("status_name"):
            return "invalid_schema"
        observed_result = payload.get("source_result_sha256")
        observed_journal = payload.get("source_journal_sha256")
    else:
        if payload.get("outcome") not in TARGET_OUTCOMES:
            return "invalid_schema"
        source = payload.get("source") or {}
        observed_result = source.get("result_sha256")
        observed_journal = source.get("journal_sha256")
    if (
        observed_result != row["source_status_sha256"]
        or observed_journal != row["source_journal_sha256"]
    ):
        return "identity_mismatch"
    return "valid"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", choices=("A", "B"), required=True)
    parser.add_argument("--stage", choices=("mip", "target"), required=True)
    args = parser.parse_args()
    counts = Counter()
    missing = []
    invalid = []
    with args.manifest.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            stem = (
                f"{args.panel}__{row['cell']}__"
                f"{row['representation_id']}.json"
            )
            state = classify(
                args.root / args.stage / stem, args.stage, row
            )
            counts[state] += 1
            if state == "missing":
                missing.append(row["index"])
            elif state != "valid":
                invalid.append((row["index"], state))
    print(
        f"{args.panel} {args.stage} artifact validation: "
        + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)),
        file=sys.stderr,
    )
    if invalid:
        print(
            "invalid existing integer artifacts: "
            + ", ".join(f"{index}:{state}" for index, state in invalid),
            file=sys.stderr,
        )
        return 2
    if missing:
        sys.stdout.write("\n".join(missing) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

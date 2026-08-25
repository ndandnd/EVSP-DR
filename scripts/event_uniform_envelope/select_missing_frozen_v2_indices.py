#!/usr/bin/env python3
"""Validate matched-wall v2 snapshot identities and emit missing indices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


SCHEMA = "evsp-dr-exact-cg-matched-wall-snapshot-v2"
BOUNDARY = "include_columns_only_through_last_durably_completed_iteration"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_journal(status: dict, source: Path) -> Path:
    recorded = status.get("columns_journal")
    candidates = [Path(recorded)] if recorded else []
    if recorded:
        candidates.append(source.parent / Path(recorded).name)
    candidates.append(Path(str(source) + ".columns.jsonl"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"no source journal for {source}")


def classify(source: Path, output: Path) -> str:
    if not output.is_file() or output.stat().st_size == 0:
        return "missing"
    try:
        source_payload = json.loads(source.read_text())
        payload = json.loads(output.read_text())
    except Exception:
        return "malformed"
    if not isinstance(payload, dict):
        return "malformed"
    matched = payload.get("matched_wall_snapshot") or {}
    if (
        matched.get("schema") != SCHEMA
        or matched.get("conservative_boundary") != BOUNDARY
        or payload.get("stop_reason") != "matched_wall_snapshot"
    ):
        return "invalid_schema"
    try:
        journal = source_journal(source_payload, source)
    except (FileNotFoundError, TypeError):
        return "source_missing"
    parent = payload.get("resume_parent") or {}
    if (
        parent.get("source_status_sha256") != sha256(source)
        or parent.get("source_journal_sha256") != sha256(journal)
    ):
        return "identity_mismatch"
    frozen_journal = Path(payload.get("columns_journal", ""))
    frozen_iterations = Path(str(output) + ".iters.csv")
    if (
        not frozen_journal.is_file()
        or frozen_journal.stat().st_size == 0
        or not frozen_iterations.is_file()
        or frozen_iterations.stat().st_size == 0
    ):
        return "incomplete"
    records = sum(
        bool(line.strip())
        for line in frozen_journal.read_bytes().splitlines()
    )
    if records != matched.get("journal_record_count"):
        return "identity_mismatch"
    return "valid"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    counts = Counter()
    missing = []
    invalid = []
    with (args.root / "matrix.tsv").open(newline="") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            index, cell, _, _, representation = fields[:5]
            stem = f"B__{cell}__{representation}.json"
            state = classify(
                args.root / "cg" / stem,
                args.output_dir / stem,
            )
            counts[state] += 1
            if state == "missing":
                missing.append(index)
            elif state != "valid":
                invalid.append((index, state))
    print(
        "Panel B frozen_v2 artifact validation: "
        + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)),
        file=sys.stderr,
    )
    if invalid:
        print(
            "invalid existing frozen_v2 artifacts: "
            + ", ".join(f"{index}:{state}" for index, state in invalid),
            file=sys.stderr,
        )
        return 2
    if missing:
        sys.stdout.write("\n".join(missing) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

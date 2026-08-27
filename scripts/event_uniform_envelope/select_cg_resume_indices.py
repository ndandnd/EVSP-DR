#!/usr/bin/env python3
"""Emit extended-resume indices that have not certified or reached their cap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--expected-panel", choices=("A", "B"))
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-wall-limit-s", type=float)
    args = parser.parse_args()
    root = args.resume_root.resolve()
    plan = json.loads((root / "execution_plan.json").read_text())
    limit = float(plan["cumulative_scientific_wall_limit_s"])
    if args.expected_panel and plan.get("panel") != args.expected_panel:
        raise SystemExit("resume plan panel does not match launcher")
    if args.expected_commit and plan.get("solver_commit") != args.expected_commit:
        raise SystemExit("resume plan solver commit does not match launcher")
    if (
        args.expected_wall_limit_s is not None
        and abs(limit - args.expected_wall_limit_s) > 1e-9
    ):
        raise SystemExit("resume plan wall limit does not match launcher")
    counts = Counter()
    pending = []
    errors = []
    with (root / "matrix.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            source = Path(row["source_status"])
            source_journal = Path(row["source_journal"])
            resume = Path(row["resume_status"])
            resume_journal = Path(row["resume_journal"])
            if (
                not source.is_file()
                or sha256(source) != row["source_status_sha256"]
                or not source_journal.is_file()
                or sha256(source_journal) != row["source_journal_sha256"]
            ):
                errors.append(f"{row['local_index']}:source_identity_changed")
                continue
            if not resume.is_file() or not resume_journal.is_file():
                errors.append(f"{row['local_index']}:missing_staged_artifact")
                continue
            try:
                status = json.loads(resume.read_text())
                wall_s = float(status.get("wall_s", 0.0))
            except Exception:
                errors.append(f"{row['local_index']}:malformed_resume_status")
                continue
            if status.get("certified_rc_optimal") is True:
                counts["certified"] += 1
            elif (
                status.get("stop_reason") == "wall_limit"
                and wall_s >= limit - 120.0
            ):
                counts["wall_cap"] += 1
            else:
                counts["pending"] += 1
                pending.append(row["local_index"])
    print(
        "CG resume validation: "
        + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)),
        file=sys.stderr,
    )
    if errors:
        print("invalid resume artifacts: " + ", ".join(errors), file=sys.stderr)
        return 2
    if pending:
        sys.stdout.write("\n".join(pending) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

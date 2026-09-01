#!/usr/bin/env python3
"""Select only validated 48-hour HiGHS rows censored by Slurm OOM."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", choices=("A", "B"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit_path = root / "backend_retry172800.csv"
    with audit_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row["index"] for row in rows}) != len(rows):
        raise SystemExit(f"invalid 48-hour audit rows: {audit_path}")

    selected = []
    execution_errors = []
    for row in rows:
        if row.get("classification") != "slurm_execution_error":
            continue
        execution_errors.append(row["index"])
        valid = (
            row.get("prior_validated_stage") in {"24h", "8h_fallback"}
            and len(row.get("source_status_sha256", "")) == 64
            and len(row.get("source_journal_sha256", "")) == 64
            and row.get("highs48_present") == "False"
            and not row.get("highs48_artifact_sha256")
            and row.get("highs48_slurm_state") == "OUT_OF_MEMORY"
            and row.get("highs48_slurm_exit") == "0:125"
        )
        if not valid:
            raise SystemExit(
                f"unsafe Panel {args.panel} execution-error row "
                f"{row['index']}"
            )
        selected.append(row["index"])
    if selected != execution_errors:
        raise SystemExit("not every execution error is a validated OOM")
    sys.stdout.write("\n".join(selected) + ("\n" if selected else ""))
    print(
        f"Panel {args.panel} 48-hour OOM selection: {len(selected)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

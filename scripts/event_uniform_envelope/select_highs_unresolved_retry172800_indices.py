#!/usr/bin/env python3
"""Select validated 24-hour rows that still warrant a 48-hour HiGHS solve."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


RETRYABLE = {
    "fleet_agreement_highs_unproven",
    "fleet_agreement_both_unproven",
    "highs_unproven",
    "both_unproven",
}
RESOLVED = {
    "proven_fleet_agreement",
    "fleet_agreement_gurobi_unproven",
    "gurobi_unproven",
}


def yes(row: dict, key: str) -> bool:
    return row.get(key) == "True"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", choices=("A", "B"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "highs_unresolved_retry86400_jobs.tsv").open(newline="") as handle:
        jobs = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row.get("panel") == args.panel
        ]
    if len(jobs) != 1:
        raise SystemExit(f"expected one Panel {args.panel} 24-hour job")
    expected = jobs[0]["indices"].split(",")
    with (root / "backend_retry86400.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (root / "backend_retry28800.csv").open(newline="") as handle:
        prior = {row["index"]: row for row in csv.DictReader(handle)}
    if [row["index"] for row in rows] != expected:
        raise SystemExit(f"Panel {args.panel} 24-hour CSV/job indices differ")
    retry = []
    for row in rows:
        outcome = row["classification"]
        if outcome in RESOLVED:
            continue
        if outcome in {"missing_or_invalid_artifact", "slurm_execution_error"}:
            old = prior.get(row["index"], {})
            required_prior = (
                "highs8_present",
                "highs8_physical_witness_valid", "highs8_source_hash_match",
                "highs8_configuration_match",
            )
            if (
                old.get("classification") not in RETRYABLE
                or any(not yes(old, key) for key in required_prior)
            ):
                raise SystemExit(
                    f"unsafe Panel {args.panel} fallback row "
                    f"{row['index']}: invalid eight-hour evidence"
                )
            print(
                f"Panel {args.panel} row {row['index']}: selecting 48h "
                f"from validated 8h evidence after {outcome}",
                file=sys.stderr,
            )
            retry.append(row["index"])
            continue
        if outcome not in RETRYABLE:
            raise SystemExit(
                f"unsafe Panel {args.panel} row {row['index']}: {outcome}"
            )
        required = (
            "highs24_present", "highs24_physical_witness_valid",
            "highs24_source_hash_match", "highs24_configuration_match",
        )
        failed = [key for key in required if not yes(row, key)]
        if failed:
            raise SystemExit(
                f"unsafe Panel {args.panel} row {row['index']}: "
                f"failed checks {','.join(failed)}"
            )
        if row["highs24_slurm_state"] != "COMPLETED" or row["highs24_slurm_exit"] != "0:0":
            raise SystemExit(
                f"unsafe Panel {args.panel} row {row['index']}: Slurm mismatch"
            )
        retry.append(row["index"])
    sys.stdout.write("\n".join(retry) + ("\n" if retry else ""))
    print(
        f"Panel {args.panel} 48h selection: "
        f"retry={len(retry)} resolved={len(rows) - len(retry)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

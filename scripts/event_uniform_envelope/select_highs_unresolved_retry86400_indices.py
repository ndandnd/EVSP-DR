#!/usr/bin/env python3
"""Select safe eight-hour rows that still warrant a 24-hour HiGHS solve."""

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
    with (root / "highs_unresolved_retry28800_jobs.tsv").open(
        newline=""
    ) as handle:
        jobs = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row.get("panel") == args.panel
        ]
    if len(jobs) != 1:
        raise SystemExit(f"expected one Panel {args.panel} eight-hour job")
    expected = jobs[0]["indices"].split(",")
    with (root / "backend_retry28800.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed = [row["index"] for row in rows]
    if observed != expected:
        raise SystemExit(
            f"Panel {args.panel} eight-hour CSV/job indices differ: "
            f"csv={observed} jobs={expected}"
        )
    retry = []
    for row in rows:
        outcome = row["classification"]
        if outcome not in RETRYABLE | RESOLVED:
            raise SystemExit(
                f"unsafe Panel {args.panel} row {row['index']}: {outcome}"
            )
        required = (
            "highs8_present", "highs8_physical_witness_valid",
            "highs8_source_hash_match", "highs8_configuration_match",
        )
        failed = [key for key in required if not yes(row, key)]
        if failed:
            raise SystemExit(
                f"unsafe Panel {args.panel} row {row['index']}: "
                f"failed checks {','.join(failed)}"
            )
        if (
            row["highs8_slurm_state"] != "COMPLETED"
            or row["highs8_slurm_exit"] != "0:0"
        ):
            raise SystemExit(
                f"unsafe Panel {args.panel} row {row['index']}: "
                "eight-hour Slurm execution mismatch"
            )
        if outcome in RETRYABLE:
            retry.append(row["index"])
    sys.stdout.write("\n".join(retry) + ("\n" if retry else ""))
    print(
        f"Panel {args.panel} 24h selection: "
        f"retry={len(retry)} resolved={len(rows) - len(retry)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

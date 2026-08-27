#!/usr/bin/env python3
"""Select only safe, still-unproven rows for the eight-hour HiGHS retry."""

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
SOLVER_COMMIT = "44b6d5030a78ddca9c74f582d70ad87572e61794"


def true(row: dict, key: str) -> bool:
    return row.get(key) == "True"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", choices=("A", "B"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    jobs_path = root / "highs_disagreement_retry_jobs.tsv"
    audit_path = root / "backend_retry7200.csv"
    with jobs_path.open(newline="") as handle:
        job_rows = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row.get("panel") == args.panel
        ]
    if len(job_rows) != 1:
        raise SystemExit(
            f"expected exactly one Panel {args.panel} two-hour job record"
        )
    expected = job_rows[0]["indices"].split(",")
    with audit_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed = [row["index"] for row in rows]
    if observed != expected:
        raise SystemExit(
            f"Panel {args.panel} audit indices differ from immutable job record: "
            f"expected={expected} observed={observed}"
        )
    retry = []
    for row in rows:
        classification = row["classification"]
        if classification not in RETRYABLE | RESOLVED:
            raise SystemExit(
                f"unsafe Panel {args.panel} row {row['index']}: "
                f"{classification}"
            )
        required_true = (
            "gurobi_present",
            "gurobi_source_hash_match",
            "gurobi_physical_witness_valid",
            "highs30_present",
            "highs30_source_hash_match",
            "highs30_physical_witness_valid",
            "highs2_present",
            "highs2_source_hash_match",
            "highs2_physical_witness_valid",
            "highs2_configuration_match",
        )
        failed = [key for key in required_true if not true(row, key)]
        if failed:
            raise SystemExit(
                f"unsafe Panel {args.panel} row {row['index']}: "
                f"failed checks {','.join(failed)}"
            )
        if (
            row["highs2_expected_solver_commit"] != SOLVER_COMMIT
            or row["highs2_observed_solver_commit"] != SOLVER_COMMIT
            or float(row["highs2_requested_timelimit_s"]) != 7200.0
            or int(row["highs2_threads_requested"]) != 8
            or row["highs2_slurm_state"] != "COMPLETED"
            or row["highs2_slurm_exit"] != "0:0"
        ):
            raise SystemExit(
                f"unsafe Panel {args.panel} row {row['index']}: "
                "two-hour execution provenance mismatch"
            )
        if classification in RETRYABLE:
            retry.append(row["index"])
    sys.stdout.write("\n".join(retry) + ("\n" if retry else ""))
    print(
        f"Panel {args.panel} eight-hour selection: "
        f"retry={len(retry)} resolved={len(rows) - len(retry)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

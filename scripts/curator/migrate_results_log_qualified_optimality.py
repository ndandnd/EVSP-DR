#!/usr/bin/env python3
"""Migrate records/RESULTS_LOG.csv to the qualified-optimality schema (2026-08-21).

Operator-directed schema revision: an unqualified proven/optimal flag conflates
"optimal over the finite column pool Gurobi received" with "optimal for the
discretized model". This holds even when the source CG LP is certified, because
an integer-useful route can carry zero or positive reduced cost at the LP
optimum and never enter the pool. Nine columns are appended:

    source_cg_certified, source_cg_stop_reason, source_cg_iterations,
    pool_fleet_proven, pool_mip_bound, model_fleet_proven,
    model_optimality_method (sandwich | arcflow | branch_and_price),
    optimality_scope (finite_pool | discrete_model), physical_witness_valid

Existing rows are preserved byte-for-byte in their original columns; the new
columns are empty for CG rows, which make no integer-optimality claim.
Idempotent: refuses to run if the header already carries the new columns.
"""
import csv
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TARGET = REPO / "records/RESULTS_LOG.csv"
NEW_COLUMNS = [
    "source_cg_certified", "source_cg_stop_reason", "source_cg_iterations",
    "pool_fleet_proven", "pool_mip_bound", "model_fleet_proven",
    "model_optimality_method", "optimality_scope", "physical_witness_valid",
]


def main() -> int:
    with TARGET.open(newline="") as h:
        reader = csv.DictReader(h)
        fields = list(reader.fieldnames)
        if NEW_COLUMNS[0] in fields:
            print("schema already migrated; nothing to do")
            return 1
        rows = list(reader)
    fields += NEW_COLUMNS
    with TARGET.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            for c in NEW_COLUMNS:
                r[c] = ""
            w.writerow(r)
    print(f"migrated header to {len(fields)} columns; rewrote {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

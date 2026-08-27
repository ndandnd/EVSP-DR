#!/usr/bin/env python3
"""Print final CG tails and classify cross-backend fleet disagreements."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def yes(row: dict, key: str) -> bool:
    return row.get(key) == "True"


def disagreement_class(row: dict) -> str:
    if not yes(row, "gurobi_present") or not yes(row, "highs_present"):
        return "missing_artifact"
    if not yes(row, "gurobi_source_hash_match") or not yes(
        row, "highs_source_hash_match"
    ):
        return "source_identity_error"
    if not yes(row, "gurobi_physical_witness_valid") or not yes(
        row, "highs_physical_witness_valid"
    ):
        return "invalid_physical_witness"
    gurobi_proven = yes(row, "gurobi_fleet_proven")
    highs_proven = yes(row, "highs_fleet_proven")
    if gurobi_proven and highs_proven:
        return "contradictory_proven_fleets"
    if gurobi_proven:
        return "highs_unproven"
    if highs_proven:
        return "gurobi_unproven"
    return "both_unproven"


def triage_resume(root: Path, relative: str, panel: str) -> None:
    path = root / relative / "resume_summary.csv"
    rows = list(csv.DictReader(path.open(newline="")))
    print(f"Panel {panel} CG outcomes: {dict(sorted(Counter(row['outcome'] for row in rows).items()))}")
    for row in rows:
        if row["outcome"] != "certified":
            print(
                f"{panel} CG {row['cell']} {row['representation_id']} "
                f"outcome={row['outcome']} stop={row['resume_stop_reason']} "
                f"wall={row['resume_wall_s']}/{row['cumulative_wall_cap_s']} "
                f"slurm={row['slurm_state']}/{row['slurm_exit']}"
            )


def triage_backend(root: Path, panel: str) -> list[dict]:
    source = root / "backend_reproduction.csv"
    rows = []
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["fleet_agreement"] == "True":
                continue
            rows.append({
                "panel": panel,
                "index": row["index"],
                "cell": row["cell"],
                "representation_id": row["representation_id"],
                "classification": disagreement_class(row),
                "gurobi_status": row["gurobi_status"],
                "gurobi_buses": row["gurobi_buses"],
                "gurobi_bound": row["gurobi_bound"],
                "gurobi_proven": row["gurobi_fleet_proven"],
                "highs_status": row["highs_status"],
                "highs_buses": row["highs_buses"],
                "highs_bound": row["highs_bound"],
                "highs_proven": row["highs_fleet_proven"],
            })
    output = root / "backend_disagreements.csv"
    with output.open("w", newline="") as handle:
        fields = list(rows[0]) if rows else [
            "panel", "index", "cell", "representation_id", "classification",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["classification"] for row in rows)
    print(f"Panel {panel} disagreement classes: {dict(sorted(counts.items()))}")
    for row in rows:
        print(
            f"{panel} MIP {row['index']} {row['cell']} {row['representation_id']} "
            f"{row['classification']} | G={row['gurobi_buses']}/{row['gurobi_bound']} "
            f"proven={row['gurobi_proven']} | H={row['highs_buses']}/"
            f"{row['highs_bound']} proven={row['highs_proven']}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-a", type=Path, required=True)
    parser.add_argument("--panel-b", type=Path, required=True)
    args = parser.parse_args()
    a_root = args.panel_a.resolve()
    b_root = args.panel_b.resolve()
    triage_resume(a_root, "cg_resume24h_2dd2b4c", "A")
    triage_resume(b_root, "cg_certification6h_13596d0", "B")
    rows = triage_backend(a_root, "A") + triage_backend(b_root, "B")
    unsafe = {
        "missing_artifact", "source_identity_error",
        "invalid_physical_witness", "contradictory_proven_fleets",
    }
    blocked = any(row["classification"] in unsafe for row in rows)
    print(f"longer-retry gate: {'BLOCKED' if blocked else 'ELIGIBLE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

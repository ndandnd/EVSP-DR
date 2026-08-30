#!/usr/bin/env python3
"""Validate the static research-control-tower ledgers before publication."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis" / "research_control_tower_20260830"


def read(name: str) -> list[dict[str, str]]:
    path = DATA / name
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty ledger: {path}")
    return rows


def require_unique(rows: list[dict[str, str]], key: str, label: str) -> None:
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise SystemExit(f"duplicate {key} in {label}")


def main() -> int:
    goals = read("goal_status.csv")
    stages = read("stage_definitions.csv")
    proofs = read("proof_snapshot.csv")
    queue = read("current_queue_snapshot.csv")
    campaigns = read("campaign_register.csv")
    actions = read("next_actions.csv")
    scale = read("large_scale_state.csv")

    require_unique(goals, "goal_id", "goals")
    require_unique(stages, "stage_id", "stages")
    require_unique(campaigns, "campaign_id", "campaigns")
    require_unique(actions, "priority", "actions")

    if [row["stage_id"] for row in stages] != [f"S{i}" for i in range(8)]:
        raise SystemExit("stage ledger must contain S0 through S7 in order")

    totals = [row for row in queue if row["campaign"] == "TOTAL"]
    if len(totals) != 1:
        raise SystemExit("queue snapshot must contain exactly one TOTAL row")
    detail = [row for row in queue if row["campaign"] != "TOTAL"]
    for field in ("total_tasks", "active_tasks", "not_in_queue_unaudited"):
        observed = sum(int(row[field]) for row in detail)
        expected = int(totals[0][field])
        if observed != expected:
            raise SystemExit(f"queue {field}: detail={observed}, total={expected}")

    for row in detail:
        total = int(row["total_tasks"])
        active = int(row["active_tasks"])
        absent = int(row["not_in_queue_unaudited"])
        if active + absent != total:
            raise SystemExit(f"queue partition mismatch for {row['campaign']}")

    committed_core = [
        row
        for row in proofs
        if row["evidence_class"] == "committed"
        and row["model_scope"] == "current_core"
    ]
    if len(committed_core) != 1:
        raise SystemExit("exactly one committed current_core proof row is required")
    core = committed_core[0]
    if (core["cells"], core["l_model_proved"], core["i_model_proved"]) != (
        "9",
        "9",
        "9",
    ):
        raise SystemExit("current-core 9/9 proof invariant changed; review manually")

    if {row["target_k"] for row in scale if row["evidence_era"] == "current"} != {
        "8",
        "13",
        "20",
        "30",
        "40",
    }:
        raise SystemExit("current large-scale rows must cover k=8,13,20,30,40")

    print(
        "validated research control tower: "
        f"goals={len(goals)} stages={len(stages)} proofs={len(proofs)} "
        f"campaigns={len(campaigns)} actions={len(actions)} scale_rows={len(scale)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

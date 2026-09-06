#!/usr/bin/env python3
"""Snapshot a live cumulative-48h CG campaign without finalizing its audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from audit_cg_resume import jobs, load_slurm, task


ACTIVE_STATES = {
    "CONFIGURING", "COMPLETING", "PENDING", "REQUEUED", "REQUEUE_FED",
    "RUNNING", "SUSPENDED",
}


def load_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def outcome(status: dict | None, state: str, cap: float) -> str:
    state = state.split()[0] if state else ""
    if state in ACTIVE_STATES:
        return state.lower()
    if status is None:
        return "missing"
    if status.get("certified_rc_optimal") is True:
        return "certified"
    stop = str(status.get("stop_reason") or "uncertified")
    wall = float(status.get("wall_s") or 0.0)
    if stop == "wall_limit" and wall >= cap - 120.0:
        return "wall_cap"
    return stop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.resume_root.resolve()
    plan = json.loads((root / "execution_plan.json").read_text())
    cap = float(plan["cumulative_scientific_wall_limit_s"])
    accounting = load_slurm(args.sacct)
    job_ids = jobs(root)
    rows: list[dict] = []

    with (root / "matrix.tsv").open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle, delimiter="\t"):
            status_path = Path(item["resume_status"])
            status = load_json(status_path)
            slurm = task(accounting, job_ids, item["local_index"])
            state = str(slurm.get("state") or "")
            current = outcome(status, state, cap)
            rows.append({
                "local_index": item["local_index"],
                "source_panel_index": item["source_panel_index"],
                "target_fleet": item["target_fleet"],
                "cell": item["cell"],
                "representation_id": item["representation_id"],
                "outcome": current,
                "certified": (status or {}).get("certified_rc_optimal", ""),
                "stop_reason": (status or {}).get("stop_reason", ""),
                "cumulative_wall_h": (
                    round(float((status or {}).get("wall_s") or 0.0) / 3600, 3)
                    if status else ""
                ),
                "iterations": (status or {}).get("iterations", ""),
                "columns": (status or {}).get("columns", ""),
                "final_min_rc": ((status or {}).get("final") or {}).get("min_rc", ""),
                "slurm_state": state,
                "slurm_exit": slurm.get("exit", ""),
                "slurm_elapsed": slurm.get("elapsed", ""),
                "slurm_max_rss": slurm.get("rss", ""),
                "slurm_node": slurm.get("node", ""),
                "status_path": str(status_path),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Cumulative-48h outcomes:", dict(sorted(Counter(
        row["outcome"] for row in rows
    ).items())))
    print("index|k|cell|outcome|wall_h|iterations|columns|min_rc|slurm")
    for row in rows:
        if row["outcome"] in {"pending", "running", "requeued", "requeue_fed"}:
            continue
        print(
            row["source_panel_index"], row["target_fleet"], row["cell"],
            row["outcome"], row["cumulative_wall_h"], row["iterations"],
            row["columns"], row["final_min_rc"],
            f"{row['slurm_state']}/{row['slurm_exit']}", sep="|",
        )
    print(f"Snapshot CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

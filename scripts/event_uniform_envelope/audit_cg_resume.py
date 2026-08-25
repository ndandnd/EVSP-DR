#!/usr/bin/env python3
"""Normalize an immutable extended-CG resume campaign to CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_slurm(path: Path) -> dict[str, dict]:
    result = {}
    if not path.is_file():
        return result
    with path.open(newline="") as handle:
        for fields in csv.reader(handle, delimiter="|"):
            if len(fields) < 9:
                continue
            job, _, state, exit_code, elapsed, rss, _, _, node = fields[:9]
            result[job] = {
                "state": state, "exit": exit_code, "elapsed": elapsed,
                "rss": rss, "node": node,
            }
    return result


def jobs(root: Path) -> list[str]:
    result = []
    for path in sorted(root.glob("jobs_*.tsv")):
        with path.open(newline="") as handle:
            result.extend(row["array_job_id"] for row in csv.DictReader(
                handle, delimiter="\t"
            ))
    return result


def task(slurm: dict, job_ids: list[str], index: str) -> dict:
    for job in reversed(job_ids):
        task_id = f"{job}_{index}"
        if task_id not in slurm and f"{task_id}.batch" not in slurm:
            continue
        row = dict(slurm.get(task_id, {}))
        batch = slurm.get(f"{task_id}.batch", {})
        if batch.get("rss"):
            row["rss"] = batch["rss"]
        return row
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    args = parser.parse_args()
    root = args.resume_root.resolve()
    plan = json.loads((root / "execution_plan.json").read_text())
    cap = float(plan["cumulative_scientific_wall_limit_s"])
    slurm = load_slurm(args.sacct)
    job_ids = jobs(root)
    rows = []
    with (root / "matrix.tsv").open(newline="") as handle:
        for item in csv.DictReader(handle, delimiter="\t"):
            source = json.loads(Path(item["source_status"]).read_text())
            resumed = json.loads(Path(item["resume_status"]).read_text())
            original_wall = float(source.get("wall_s", 0.0))
            resumed_wall = float(resumed.get("wall_s", 0.0))
            slurm_row = task(slurm, job_ids, item["local_index"])
            rows.append({
                "local_index": item["local_index"],
                "source_panel_index": item["source_panel_index"],
                "cell": item["cell"],
                "target_fleet": item["target_fleet"],
                "representation_id": item["representation_id"],
                "soc_step": item["soc_step"],
                "block_min": item["block_min"],
                "source_stop_reason": source.get("stop_reason"),
                "source_certified": source.get("certified_rc_optimal"),
                "source_wall_s": original_wall,
                "source_iterations": source.get("iterations"),
                "source_columns": source.get("columns"),
                "resume_stop_reason": resumed.get("stop_reason"),
                "resume_certified": resumed.get("certified_rc_optimal"),
                "resume_wall_s": resumed_wall,
                "resume_added_wall_s": resumed_wall - original_wall,
                "resume_attempt_wall_s": resumed.get("attempt_wall_s"),
                "resume_iterations": resumed.get("iterations"),
                "resume_columns": resumed.get("columns"),
                "resume_peak_rss_mb": resumed.get("peak_rss_mb"),
                "cumulative_wall_cap_s": cap,
                "outcome": (
                    "certified" if resumed.get("certified_rc_optimal") is True
                    else "wall_cap" if resumed_wall >= cap - 1.0
                    else "incomplete"
                ),
                "slurm_state": slurm_row.get("state"),
                "slurm_exit": slurm_row.get("exit"),
                "slurm_elapsed": slurm_row.get("elapsed"),
                "slurm_max_rss": slurm_row.get("rss"),
                "slurm_node": slurm_row.get("node"),
            })
    output = root / "resume_summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("CG resume outcomes:", dict(sorted(Counter(row["outcome"] for row in rows).items())))
    print(f"CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

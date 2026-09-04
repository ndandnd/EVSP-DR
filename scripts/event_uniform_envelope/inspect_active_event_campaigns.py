#!/usr/bin/env python3
"""Snapshot live acceleration and small-threshold campaigns without auditing them final."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from audit_highs_disagreement_retry import load_accounting, slurm_task


ACTIVE = {"PENDING", "RUNNING", "REQUEUED", "REQUEUE_FED"}
EXECUTION = {
    "BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT",
}


def load_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError):
        return None


def classify(status: dict | None, state: str) -> str:
    if state in ACTIVE:
        return state.lower()
    if state in EXECUTION:
        return f"execution_{state.lower()}"
    if status is None:
        return "missing"
    if status.get("certified_rc_optimal") is True:
        return "certified"
    return str(status.get("stop_reason") or "uncertified")


def expand_indices(text: str) -> list[int]:
    result = []
    for token in text.split(","):
        if "-" in token:
            lo, hi = (int(value) for value in token.split("-", 1))
            result.extend(range(lo, hi + 1))
        elif token:
            result.append(int(token))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def acceleration_rows(root: Path, accounting: dict) -> list[dict]:
    with (root / "matrix.tsv").open(newline="", encoding="utf-8") as handle:
        matrix = list(csv.reader(handle, delimiter="\t"))
    plan = json.loads((root / "execution_plan.json").read_text())
    jobs = {}
    for path in [root / "jobs.tsv", *sorted(root.glob("jobs_recovery_*.tsv"))]:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["stage"] == "cg":
                    jobs[row["arm"]] = row["array_job_id"]
    rows = []
    for fields in matrix:
        index, cell, scale, replicate, trips = fields[:5]
        representation = fields[7]
        for arm in plan["arms"]:
            task = slurm_task(accounting, jobs[arm["arm"]], index)
            state = str(task.get("state") or "")
            path = root / "cg" / arm["arm"] / (
                f"M__{cell}__{representation}.json"
            )
            status = load_json(path)
            rows.append({
                "campaign": "acceleration", "index": index,
                "cell_id": cell, "scale": scale, "replicate": replicate,
                "trip_count": trips, "arm": arm["arm"],
                "array_job_id": jobs[arm["arm"]],
                "slurm_task": f"{jobs[arm['arm']]}_{index}",
                "slurm_state": state, "slurm_exit": task.get("exit", ""),
                "slurm_elapsed": task.get("elapsed", ""),
                "slurm_max_rss": task.get("max_rss", ""),
                "result_present": path.is_file(),
                "outcome": classify(status, state),
                "certified": (status or {}).get("certified_rc_optimal", ""),
                "stop_reason": (status or {}).get("stop_reason", ""),
                "wall_s": (status or {}).get("wall_s", ""),
                "iterations": (status or {}).get("iterations", ""),
                "columns": (status or {}).get("columns", ""),
            })
    return rows


def small_rows(root: Path, accounting: dict) -> list[dict]:
    with (root / "matrix.tsv").open(newline="", encoding="utf-8") as handle:
        matrix = list(csv.reader(handle, delimiter="\t"))
    jobs = {}
    for path in [root / "jobs.tsv", *sorted(root.glob("jobs_recovery_*.tsv"))]:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                for index in expand_indices(row["indices"]):
                    jobs[index] = row["array_job_id"]
    rows = []
    for fields in matrix:
        index, cell, scale, replicate, trips = fields[:5]
        representation = fields[7]
        job = jobs[int(index)]
        task = slurm_task(accounting, job, index)
        state = str(task.get("state") or "")
        path = root / "cg" / f"M__{cell}__{representation}.json"
        status = load_json(path)
        rows.append({
            "campaign": "small_threshold", "index": index,
            "cell_id": cell, "scale": scale, "replicate": replicate,
            "trip_count": trips, "arm": "baseline",
            "array_job_id": job, "slurm_task": f"{job}_{index}",
            "slurm_state": state, "slurm_exit": task.get("exit", ""),
            "slurm_elapsed": task.get("elapsed", ""),
            "slurm_max_rss": task.get("max_rss", ""),
            "result_present": path.is_file(),
            "outcome": classify(status, state),
            "certified": (status or {}).get("certified_rc_optimal", ""),
            "stop_reason": (status or {}).get("stop_reason", ""),
            "wall_s": (status or {}).get("wall_s", ""),
            "iterations": (status or {}).get("iterations", ""),
            "columns": (status or {}).get("columns", ""),
        })
    return rows


def report(label: str, rows: list[dict], group_fields: tuple[str, ...]) -> None:
    print(f"=== {label} ===")
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    for key in sorted(groups):
        counts = Counter(row["outcome"] for row in groups[key])
        name = " ".join(
            f"{field}={value}" for field, value in zip(group_fields, key)
        )
        print(name, dict(sorted(counts.items())))
    abnormal = [
        row for row in rows
        if str(row["outcome"]).startswith("execution_")
    ]
    for row in abnormal:
        print(
            "ATTENTION", row["slurm_task"], row["cell_id"],
            row["outcome"], row["slurm_exit"], row["slurm_elapsed"],
            row["slurm_max_rss"],
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceleration-root", type=Path, required=True)
    parser.add_argument("--small-root", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    accounting = load_accounting(args.sacct)
    acceleration = acceleration_rows(args.acceleration_root.resolve(), accounting)
    small = small_rows(args.small_root.resolve(), accounting)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "acceleration_progress.csv", acceleration)
    write_csv(args.output_dir / "small_threshold_progress.csv", small)
    report("ACCELERATION", acceleration, ("arm", "scale"))
    report("SMALL THRESHOLD", small, ("scale",))
    print(f"CSV snapshot directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

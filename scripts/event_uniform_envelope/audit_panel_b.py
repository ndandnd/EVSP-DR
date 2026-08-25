#!/usr/bin/env python3
"""Normalize matched-wall Panel B CG and frozen-pool statistics to CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> tuple[dict, str]:
    try:
        return json.loads(path.read_text()), ""
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def load_jobs(root: Path) -> dict[str, str]:
    jobs = {}
    paths = [root / "jobs.tsv"]
    paths.extend(sorted(root.glob("refreeze_v2*_jobs.tsv")))
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                jobs[row["stage"]] = row["array_job_id"]
    return jobs


def load_slurm(path: Path) -> dict[str, dict]:
    result = {}
    with path.open(newline="") as handle:
        for fields in csv.reader(handle, delimiter="|"):
            if len(fields) < 9:
                continue
            job_id, _, state, exit_code, elapsed, max_rss, max_vm, req_mem, node = fields[:9]
            result[job_id] = {
                "state": state, "exit": exit_code, "elapsed": elapsed,
                "max_rss": max_rss, "max_vm": max_vm,
                "req_mem": req_mem, "node": node,
            }
    return result


def task(slurm: dict, array_id: str | None, index: str) -> dict:
    if not array_id:
        return {}
    task_id = f"{array_id}_{index}"
    row = dict(slurm.get(task_id, {}))
    batch = slurm.get(f"{task_id}.batch", {})
    if batch.get("max_rss"):
        row["max_rss"] = batch["max_rss"]
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    jobs = load_jobs(root)
    slurm = load_slurm(args.sacct)
    rows = []
    for fields in csv.reader((root / "matrix.tsv").open(), delimiter="\t"):
        index, cell, target, instance_csv, rep, soc, block, budget, runner_limit = fields
        stem = f"B__{cell}__{rep}"
        cg_path = root / "cg" / f"{stem}.json"
        frozen_path = root / "frozen" / f"{stem}.json"
        frozen_v2_path = root / "frozen_v2" / f"{stem}.json"
        cg, cg_error = load(cg_path) if cg_path.exists() else ({}, "missing")
        frozen, frozen_error = (
            load(frozen_path) if frozen_path.exists() else ({}, "missing")
        )
        frozen_v2, frozen_v2_error = (
            load(frozen_v2_path)
            if frozen_v2_path.exists() else ({}, "missing")
        )
        final = cg.get("final") or {}
        frozen_final = frozen.get("final") or {}
        frozen_v2_final = frozen_v2.get("final") or {}
        network = cg.get("network_metrics") or {}
        cg_slurm = task(slurm, jobs.get("cg"), index)
        freeze_slurm = task(slurm, jobs.get("freeze"), index)
        freeze_v2_slurm = task(slurm, jobs.get("freeze_v2"), index)
        rows.append({
            "index": index,
            "cell": cell,
            "target_fleet": target,
            "instance_csv": instance_csv,
            "representation_id": rep,
            "soc_step": soc,
            "block_min": block,
            "paired_event_budget_s": budget,
            "runner_limit_s": runner_limit,
            "cg_present": cg_path.exists(),
            "cg_stop_reason": cg.get("stop_reason"),
            "cg_certified": cg.get("certified_rc_optimal"),
            "cg_iterations": cg.get("iterations"),
            "cg_columns": cg.get("columns"),
            "cg_wall_s": cg.get("wall_s"),
            "cg_peak_rss_mb": cg.get("peak_rss_mb"),
            "cg_route_weight": final.get("route_weight"),
            "cg_min_rc": final.get("min_rc"),
            "dag_nodes": network.get("dag_nodes"),
            "dag_arcs": network.get("dag_arcs"),
            "cg_error": cg_error,
            "frozen_present": frozen_path.exists(),
            "frozen_iteration": frozen.get("iterations"),
            "frozen_columns": frozen.get("columns"),
            "frozen_wall_s": frozen.get("wall_s"),
            "frozen_peak_rss_mb": frozen.get("peak_rss_mb"),
            "frozen_route_weight": frozen_final.get("route_weight"),
            "frozen_min_rc": frozen_final.get("min_rc"),
            "frozen_error": frozen_error,
            "frozen_v2_present": frozen_v2_path.exists(),
            "frozen_v2_iteration": frozen_v2.get("iterations"),
            "frozen_v2_columns": frozen_v2.get("columns"),
            "frozen_v2_wall_s": frozen_v2.get("wall_s"),
            "frozen_v2_peak_rss_mb": frozen_v2.get("peak_rss_mb"),
            "frozen_v2_route_weight": frozen_v2_final.get("route_weight"),
            "frozen_v2_min_rc": frozen_v2_final.get("min_rc"),
            "frozen_v2_column_delta": (
                frozen_v2.get("columns") - frozen.get("columns")
                if isinstance(frozen_v2.get("columns"), int)
                and isinstance(frozen.get("columns"), int)
                else None
            ),
            "frozen_v2_error": frozen_v2_error,
            "cg_slurm_state": cg_slurm.get("state"),
            "cg_slurm_exit": cg_slurm.get("exit"),
            "cg_slurm_max_rss": cg_slurm.get("max_rss"),
            "freeze_slurm_state": freeze_slurm.get("state"),
            "freeze_slurm_exit": freeze_slurm.get("exit"),
            "freeze_slurm_max_rss": freeze_slurm.get("max_rss"),
            "freeze_v2_slurm_state": freeze_v2_slurm.get("state"),
            "freeze_v2_slurm_exit": freeze_v2_slurm.get("exit"),
            "freeze_v2_slurm_max_rss": freeze_v2_slurm.get("max_rss"),
        })
    with (root / "panel_b_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Panel B CG stops:", dict(sorted(Counter(str(row["cg_stop_reason"]) for row in rows).items())))
    print("Panel B frozen rows:", sum(row["frozen_present"] for row in rows), "/", len(rows))
    if (root / "frozen_v2").exists():
        print("Panel B v2 frozen rows:", sum(row["frozen_v2_present"] for row in rows), "/", len(rows))
        print("Panel B v2 added columns:", sum(
            row["frozen_v2_column_delta"] or 0 for row in rows
        ))
    print(f"CSV: {root / 'panel_b_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

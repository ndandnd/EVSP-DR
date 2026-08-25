#!/usr/bin/env python3
"""Normalize matched-wall Panel B CG and frozen-pool statistics to CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    paths.extend(sorted(root.glob("integer_v2*_jobs.tsv")))
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
        mip_path = root / "mip" / f"{stem}.json"
        target_path = root / "target" / f"{stem}.json"
        cg, cg_error = load(cg_path) if cg_path.exists() else ({}, "missing")
        frozen, frozen_error = (
            load(frozen_path) if frozen_path.exists() else ({}, "missing")
        )
        frozen_v2, frozen_v2_error = (
            load(frozen_v2_path)
            if frozen_v2_path.exists() else ({}, "missing")
        )
        mip, mip_error = load(mip_path) if mip_path.exists() else ({}, "missing")
        target_payload, target_error = (
            load(target_path) if target_path.exists() else ({}, "missing")
        )
        final = cg.get("final") or {}
        frozen_final = frozen.get("final") or {}
        frozen_v2_final = frozen_v2.get("final") or {}
        network = cg.get("network_metrics") or {}
        cg_slurm = task(slurm, jobs.get("cg"), index)
        freeze_slurm = task(slurm, jobs.get("freeze"), index)
        freeze_v2_slurm = task(slurm, jobs.get("freeze_v2"), index)
        mip_slurm = task(slurm, jobs.get("mip_v2"), index)
        target_slurm = task(slurm, jobs.get("target_v2"), index)
        target_solver = target_payload.get("solver") or {}
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
            "mip_present": mip_path.exists(),
            "mip_status": mip.get("status_name"),
            "mip_buses": mip.get("buses"),
            "mip_bound": mip.get("mip_bound", mip.get("fleet_bound")),
            "mip_gap": mip.get("mip_gap"),
            "finite_pool_proven": mip.get("fleet_proven"),
            "mip_optimality_scope": mip.get("optimality_scope"),
            "mip_runtime_s": mip.get("runtime_s"),
            "mip_peak_rss_mb": mip.get("peak_rss_mb"),
            "mip_physical_witness_valid": mip.get("physical_witness_valid"),
            "mip_source_result_sha256": mip.get("source_result_sha256"),
            "mip_source_journal_sha256": mip.get("source_journal_sha256"),
            "mip_error": mip_error,
            "target_present": target_path.exists(),
            "target_outcome": target_payload.get("outcome"),
            "target_runtime_s": target_solver.get("runtime_s"),
            "target_solver": target_solver.get("backend"),
            "target_source_result_sha256":
                (target_payload.get("source") or {}).get("result_sha256"),
            "target_source_journal_sha256":
                (target_payload.get("source") or {}).get("journal_sha256"),
            "target_error": target_error,
            "cg_slurm_state": cg_slurm.get("state"),
            "cg_slurm_exit": cg_slurm.get("exit"),
            "cg_slurm_max_rss": cg_slurm.get("max_rss"),
            "freeze_slurm_state": freeze_slurm.get("state"),
            "freeze_slurm_exit": freeze_slurm.get("exit"),
            "freeze_slurm_max_rss": freeze_slurm.get("max_rss"),
            "freeze_v2_slurm_state": freeze_v2_slurm.get("state"),
            "freeze_v2_slurm_exit": freeze_v2_slurm.get("exit"),
            "freeze_v2_slurm_max_rss": freeze_v2_slurm.get("max_rss"),
            "mip_slurm_state": mip_slurm.get("state"),
            "mip_slurm_exit": mip_slurm.get("exit"),
            "mip_slurm_max_rss": mip_slurm.get("max_rss"),
            "target_slurm_state": target_slurm.get("state"),
            "target_slurm_exit": target_slurm.get("exit"),
            "target_slurm_max_rss": target_slurm.get("max_rss"),
        })
    with (root / "panel_b_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stage_rows = [
        {
            "stage": stage,
            "expected_rows": len(rows),
            "artifact_rows": sum(row[present] for row in rows),
            "slurm_completed": sum(
                row[state] == "COMPLETED" for row in rows
            ),
            "slurm_failed": sum(row[state] == "FAILED" for row in rows),
        }
        for stage, present, state in (
            ("cg", "cg_present", "cg_slurm_state"),
            ("freeze_v1", "frozen_present", "freeze_slurm_state"),
            ("freeze_v2", "frozen_v2_present", "freeze_v2_slurm_state"),
            ("mip_v2", "mip_present", "mip_slurm_state"),
            ("target_v2", "target_present", "target_slurm_state"),
        )
    ]
    with (root / "panel_b_stage_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)
    error_rows = []
    signatures = Counter()
    signature_examples = {}
    for path in sorted((root / "logs").glob("*.err")):
        if path.stat().st_size == 0:
            continue
        payload = path.read_bytes()
        lines = [
            line.strip() for line in payload.decode(errors="replace").splitlines()
            if line.strip()
        ]
        stage = path.name.split("_", 1)[0]
        last_line = lines[-1] if lines else ""
        signatures[(stage, last_line)] += 1
        signature_examples.setdefault((stage, last_line), str(path))
        error_rows.append({
            "path": str(path),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "first_line": lines[0] if lines else "",
            "last_line": last_line,
        })
    with (root / "stderr_inventory.csv").open("w", newline="") as handle:
        fields = ["path", "size", "sha256", "first_line", "last_line"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(error_rows)
    signature_rows = [
        {
            "stage": stage,
            "count": count,
            "last_line": last_line,
            "example_path": signature_examples[(stage, last_line)],
        }
        for (stage, last_line), count in sorted(
            signatures.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
        )
    ]
    with (root / "stderr_signatures.csv").open("w", newline="") as handle:
        fields = ["stage", "count", "last_line", "example_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(signature_rows)
    print("Panel B CG stops:", dict(sorted(Counter(str(row["cg_stop_reason"]) for row in rows).items())))
    print("Panel B frozen rows:", sum(row["frozen_present"] for row in rows), "/", len(rows))
    if (root / "frozen_v2").exists():
        print("Panel B v2 frozen rows:", sum(row["frozen_v2_present"] for row in rows), "/", len(rows))
        print("Panel B v2 added columns:", sum(
            row["frozen_v2_column_delta"] or 0 for row in rows
        ))
    print("Panel B MIP:", dict(sorted(Counter(
        str(row["mip_status"]) for row in rows
    ).items())))
    print("Panel B target:", dict(sorted(Counter(
        str(row["target_outcome"]) for row in rows
    ).items())))
    print("stderr signatures:")
    if signature_rows:
        for row in signature_rows:
            print(f"  {row['count']:>3} {row['stage']:<5} | {row['last_line']}")
    else:
        print("    0 (all stderr files empty)")
    print(f"CSV: {root / 'panel_b_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

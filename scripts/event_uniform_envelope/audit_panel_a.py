#!/usr/bin/env python3
"""Normalize Panel A artifacts and scheduler evidence into accessible CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


STAGES = ("cg", "fleet_lp", "mip", "target")


def load_json(path: Path) -> tuple[dict, str]:
    try:
        return json.loads(path.read_text()), ""
    except Exception as exc:  # Preserve malformed/missing evidence as a row.
        return {}, f"{type(exc).__name__}: {exc}"


def load_jobs(root: Path) -> dict[str, str]:
    jobs = {}
    aliases = {
        "combined_cost_cg": "cg",
        "fleet_lp_phase2": "fleet_lp",
        "raw_pool_mip": "mip",
        "target_feasibility": "target",
        "mip_recovery": "mip",
        "target_recovery": "target",
    }
    paths = [root / "jobs.tsv"]
    paths.extend(sorted(root.glob("integer_recovery*_jobs.tsv")))
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                stage = aliases.get(row.get("stage", ""), row.get("stage", ""))
                job_id = row.get("array_job_id", "")
                if stage and job_id:
                    jobs[stage] = job_id
    return jobs


def load_slurm(path: Path) -> dict[str, dict]:
    slurm = {}
    if not path.exists():
        return slurm
    with path.open(newline="") as handle:
        for fields in csv.reader(handle, delimiter="|"):
            if len(fields) < 9:
                continue
            job_id, _, state, exit_code, elapsed, max_rss, max_vm, req_mem, node = fields[:9]
            slurm[job_id] = {
                "state": state,
                "exit_code": exit_code,
                "elapsed": elapsed,
                "max_rss": max_rss,
                "max_vm": max_vm,
                "req_mem": req_mem,
                "node": node,
            }
    return slurm


def task_slurm(slurm: dict, array_id: str | None, index: str) -> dict:
    if not array_id:
        return {}
    task_id = f"{array_id}_{index}"
    task = dict(slurm.get(task_id, {}))
    batch = slurm.get(f"{task_id}.batch", {})
    for key in ("max_rss", "max_vm"):
        if batch.get(key):
            task[key] = batch[key]
    return task


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    jobs = load_jobs(root)
    slurm = load_slurm(args.sacct)

    matrix = list(csv.reader((root / "matrix.tsv").open(), delimiter="\t"))
    rows = []
    for fields in matrix:
        index, cell, target, instance_csv, rep, time_model, soc, block = fields
        stem = f"A__{cell}__{rep}"
        paths = {
            "cg": root / "cg" / f"{stem}.json",
            "fleet_lp": root / "fleet_lp" / f"{stem}.json",
            "mip": root / "mip" / f"{stem}.json",
            "target": root / "target" / f"{stem}.json",
        }
        payloads = {}
        errors = {}
        for stage, path in paths.items():
            payloads[stage], errors[stage] = (
                load_json(path) if path.exists() else ({}, "missing")
            )

        cg = payloads["cg"]
        l2 = payloads["fleet_lp"]
        mip = payloads["mip"]
        target_payload = payloads["target"]
        target_solver = target_payload.get("solver") or {}
        final = cg.get("final") or {}
        network = cg.get("network_metrics") or {}
        certificate = l2.get("certificate") or {}
        stage_slurm = {
            stage: task_slurm(slurm, jobs.get(stage), index)
            for stage in STAGES
        }

        row = {
            "index": index,
            "cell": cell,
            "target_fleet": target,
            "instance_csv": instance_csv,
            "representation_id": rep,
            "time_model": time_model,
            "soc_step": soc,
            "block_min": block,
            "cg_present": paths["cg"].exists(),
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
            "cg_error": errors["cg"],
            "fleet_lp_present": paths["fleet_lp"].exists(),
            "fleet_lp_certified": certificate.get("certified"),
            "fleet_lp_stop_reason": certificate.get("stop_reason"),
            "fleet_lp_lower_bound": l2.get("fleet_lp_lower_bound"),
            "fleet_lp_iterations": certificate.get("iterations"),
            "fleet_lp_wall_s": l2.get("wall_s"),
            "fleet_lp_peak_rss_mb": l2.get("peak_rss_mb"),
            "fleet_lp_min_rc": certificate.get("minimum_reduced_cost"),
            "fleet_lp_error": errors["fleet_lp"],
            "mip_present": paths["mip"].exists(),
            "mip_status": mip.get("status_name"),
            "mip_buses": mip.get("buses"),
            "mip_bound": mip.get("mip_bound"),
            "mip_gap": mip.get("mip_gap"),
            "finite_pool_proven": mip.get("fleet_proven"),
            "mip_runtime_s": mip.get("runtime_s"),
            "mip_error": errors["mip"],
            "target_present": paths["target"].exists(),
            "target_outcome": target_payload.get("outcome"),
            "target_runtime_s": target_solver.get("runtime_s"),
            "target_solver": target_solver.get("backend"),
            "target_error": errors["target"],
        }
        for short, stage in (("cg", "cg"), ("l2", "fleet_lp"), ("mip", "mip"), ("tf", "target")):
            row[f"{short}_slurm_state"] = stage_slurm[stage].get("state")
            row[f"{short}_slurm_exit"] = stage_slurm[stage].get("exit_code")
            row[f"{short}_slurm_max_rss"] = stage_slurm[stage].get("max_rss")
        rows.append(row)

    summary = root / "panel_a_summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    stage_rows = []
    for stage in STAGES:
        stage_rows.append({
            "stage": stage,
            "expected_rows": len(rows),
            "artifact_rows": sum(str(row[f"{stage}_present"]).lower() == "true" for row in rows),
            "slurm_completed": sum(row[("l2" if stage == "fleet_lp" else "tf" if stage == "target" else stage) + "_slurm_state"] == "COMPLETED" for row in rows),
            "slurm_failed": sum(row[("l2" if stage == "fleet_lp" else "tf" if stage == "target" else stage) + "_slurm_state"] == "FAILED" for row in rows),
        })
    with (root / "panel_a_stage_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)

    error_rows = []
    signature_counts = Counter()
    signature_sizes = Counter()
    signature_examples = {}
    for path in sorted((root / "logs").glob("*.err")):
        if path.stat().st_size == 0:
            continue
        payload = path.read_bytes()
        lines = [line.strip() for line in payload.decode(errors="replace").splitlines() if line.strip()]
        stage = path.name.split("_", 1)[0]
        last_line = lines[-1] if lines else ""
        signature = (stage, last_line)
        signature_counts[signature] += 1
        signature_sizes[signature] += len(payload)
        signature_examples.setdefault(signature, str(path))
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
            "total_bytes": signature_sizes[(stage, last_line)],
            "last_line": last_line,
            "example_path": signature_examples[(stage, last_line)],
        }
        for (stage, last_line), count in sorted(
            signature_counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
        )
    ]
    with (root / "stderr_signatures.csv").open("w", newline="") as handle:
        fields = ["stage", "count", "total_bytes", "last_line", "example_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(signature_rows)

    event = [row for row in rows if row["time_model"] == "event"]
    eligible = [
        row for row in event
        if row["cg_certified"] is True
        and row["cg_stop_reason"] == "certified"
        and row["cg_wall_s"] is not None
    ]
    (root / "panel_b_gate.json").write_text(json.dumps({
        "eligible": len(eligible) == 9,
        "event_cells": len(event),
        "certified_event_cells": len(eligible),
        "eligible_cells": [row["cell"] for row in eligible],
    }, indent=2, sort_keys=True) + "\n")

    print(f"wrote {summary}")
    print("artifact counts:", {stage: sum(row[f"{stage}_present"] for row in rows) for stage in STAGES})
    print("CG stops:", dict(sorted(Counter(str(row["cg_stop_reason"]) for row in rows).items())))
    print("fleet LP:", dict(sorted(Counter(str(row["fleet_lp_stop_reason"]) for row in rows).items())))
    print("MIP:", dict(sorted(Counter(str(row["mip_status"]) for row in rows).items())))
    print("target:", dict(sorted(Counter(str(row["target_outcome"]) for row in rows).items())))
    print(f"Panel B event gate: {len(eligible)}/9 certified")
    print("stderr signatures:")
    if signature_rows:
        for row in signature_rows:
            print(f"  {row['count']:>3} {row['stage']:<5} | {row['last_line']}")
    else:
        print("    0 (all stderr files empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

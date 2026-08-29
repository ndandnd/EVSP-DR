#!/usr/bin/env python3
"""Normalize medium event-CG artifacts, telemetry, and Slurm accounting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_highs_disagreement_retry import load_accounting, slurm_task


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def load_json(path: Path) -> tuple[dict, str]:
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return {}, "not_an_object"
        return payload, ""
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def telemetry(path: Path) -> tuple[dict[str, float], int, int]:
    totals: dict[str, float] = defaultdict(float)
    rows = 0
    malformed = 0
    if not path.is_file():
        return totals, rows, malformed
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if record.get("record_type") == "phase":
                totals[str(record.get("phase"))] += float(record.get("duration_s", 0))
                rows += 1
        except Exception:
            malformed += 1
    return totals, rows, malformed


def expand_indices(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        if "-" in item:
            lo, hi = (int(value) for value in item.split("-", 1))
            values.extend(range(lo, hi + 1))
        elif item:
            values.append(int(item))
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "matrix.tsv").open(newline="") as handle:
        matrix = list(csv.reader(handle, delimiter="\t"))
    with (root / "jobs.tsv").open(newline="") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    index_job: dict[int, str] = {}
    for job in jobs:
        for index in expand_indices(job["indices"]):
            if index in index_job:
                raise SystemExit(f"duplicate task index in jobs.tsv: {index}")
            index_job[index] = job["array_job_id"]
    if set(index_job) != set(range(len(matrix))):
        raise SystemExit(f"job index set mismatch: {sorted(index_job)}")
    accounting = load_accounting(args.sacct)
    output_rows = []
    for fields in matrix:
        (
            index_text, cell, scale, replicate, trips, csv_path, csv_sha,
            representation, soc, block, wall_limit,
        ) = fields
        index = int(index_text)
        result_path = root / "cg" / f"M__{cell}__{representation}.json"
        telemetry_path = Path(f"{result_path}.phase-telemetry.jsonl")
        result, error = load_json(result_path)
        phases, phase_rows, malformed = telemetry(telemetry_path)
        final = result.get("final") or {}
        metrics = result.get("network_metrics") or {}
        slurm = slurm_task(accounting, index_job[index], index)
        config_match = bool(result) and (
            result.get("csv") == csv_path
            and float(result.get("soc_step")) == float(soc)
            and int(result.get("block_min")) == int(block)
            and float(result.get("g_kwh")) == 240.0
            and float(result.get("charge_kw")) == 240.0
            and float(result.get("min_soc_frac")) == 0.0
            and result.get("time_model") == "event"
        )
        output_rows.append({
            "index": index,
            "cell_id": cell,
            "scale": scale,
            "selection_replicate": replicate,
            "trip_count": trips,
            "instance_path": csv_path,
            "instance_sha256": csv_sha,
            "representation_id": representation,
            "soc_step_kwh": soc,
            "block_min": block,
            "wall_limit_s": wall_limit,
            "result_present": result_path.is_file(),
            "result_error": error,
            "result_sha256": sha256(result_path),
            "configuration_match": config_match,
            "certified_rc_optimal": result.get("certified_rc_optimal"),
            "stop_reason": result.get("stop_reason"),
            "L_model": final.get("route_weight"),
            "lp_objective": final.get("lp_obj"),
            "artificials": final.get("artificials"),
            "min_reduced_cost": final.get("min_rc"),
            "iterations": result.get("iterations"),
            "pool_columns": result.get("columns"),
            "cg_wall_s": result.get("wall_s"),
            "peak_rss_mb": result.get("peak_rss_mb"),
            "dag_nodes": metrics.get("dag_nodes"),
            "dag_arcs": metrics.get("dag_arcs"),
            "packed_arc_bytes": metrics.get("packed_arc_bytes"),
            "telemetry_present": telemetry_path.is_file(),
            "telemetry_sha256": sha256(telemetry_path),
            "telemetry_phase_rows": phase_rows,
            "telemetry_malformed_rows": malformed,
            "network_build_s": phases.get("network_build", 0.0),
            "master_lp_s": phases.get("master_attempt", 0.0),
            "incidence_construction_s": phases.get("incidence_construction", 0.0),
            "pricing_shortest_path_s": phases.get("pricing_shortest_path", 0.0),
            "pricing_batch_s": phases.get("pricing_extra_columns", 0.0),
            "route_insertion_s": phases.get("route_insertion", 0.0),
            "journal_fsync_s": phases.get("journal_fsync", 0.0),
            "status_checkpoint_s": phases.get("status_checkpoint", 0.0),
            "array_job_id": index_job[index],
            "slurm_task": f"{index_job[index]}_{index}",
            "slurm_job_id_raw": slurm.get("job_id_raw"),
            "slurm_state": slurm.get("state"),
            "slurm_exit": slurm.get("exit"),
            "slurm_elapsed": slurm.get("elapsed"),
            "slurm_total_cpu": slurm.get("total_cpu"),
            "slurm_max_rss": slurm.get("max_rss"),
            "slurm_max_vm_size": slurm.get("max_vm_size"),
            "slurm_node": slurm.get("node"),
        })
    output = root / "medium_event_summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print("Medium event result presence:", dict(Counter(row["result_present"] for row in output_rows)))
    print("Medium event CG stops:", dict(Counter(str(row["stop_reason"]) for row in output_rows)))
    print("Medium event Slurm states:", dict(Counter(str(row["slurm_state"]) for row in output_rows)))
    print(f"CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

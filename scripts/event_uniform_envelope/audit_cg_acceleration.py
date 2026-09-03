#!/usr/bin/env python3
"""Write row-level and arm/scale CSVs for the CG acceleration study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from audit_highs_disagreement_retry import load_accounting, slurm_task


def finite(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def median(values):
    return statistics.median(values) if values else None


def read_matrix(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            rows.append({
                "index": int(fields[0]),
                "cell": fields[1],
                "scale": int(fields[2]),
                "replicate": int(fields[3]),
                "trip_count": int(fields[4]),
                "representation": fields[7],
            })
    return rows


def read_arms(path: Path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    return plan["arms"]


def allocation_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return max(0, len(path.read_text(encoding="utf-8").splitlines()) - 1)


def telemetry(path: Path):
    totals = defaultdict(float)
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
                totals[str(record.get("phase"))] += float(
                    record.get("duration_s", 0.0)
                )
                rows += 1
        except Exception:
            malformed += 1
    return totals, rows, malformed


def arm_jobs(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        row["arm"]: row["array_job_id"]
        for row in rows if row["stage"] == "cg"
    }


def outcome(status, slurm_state=""):
    if slurm_state in {
        "FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "PREEMPTED"
    }:
        return f"execution_{slurm_state.lower()}"
    if status is None:
        return "missing"
    if status.get("certified_rc_optimal") is True:
        return "certified"
    return str(status.get("stop_reason") or "uncertified")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--sacct", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    matrix = read_matrix(root / "matrix.tsv")
    arms = read_arms(root / "execution_plan.json")
    jobs = arm_jobs(root / "jobs.tsv")
    accounting = load_accounting(args.sacct)
    rows = []
    for source in matrix:
        cache = root / "network_cache" / (
            f"M__{source['cell']}__{source['representation']}.pkl"
        )
        manifest_path = Path(str(cache) + ".manifest.json")
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else {}
        )
        build_s = finite(manifest.get("original_build_s"))
        for arm in arms:
            status_path = root / "cg" / arm["arm"] / (
                f"M__{source['cell']}__{source['representation']}.json"
            )
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                status = None
            metrics = (status or {}).get("network_metrics") or {}
            final = (status or {}).get("final") or {}
            cg_wall_s = finite((status or {}).get("wall_s"))
            telemetry_path = Path(str(status_path) + ".phase-telemetry.jsonl")
            phases, phase_rows, malformed = telemetry(telemetry_path)
            slurm = slurm_task(
                accounting, jobs[arm["arm"]], str(source["index"])
            )
            row_outcome = outcome(status, slurm.get("state", ""))
            iterations = (status or {}).get("iterations", "")
            pool_columns = (status or {}).get("columns", "")
            retained_growth = ""
            if iterations not in ("", 0) and pool_columns != "":
                retained_growth = max(
                    0, int(pool_columns) - int(source["trip_count"])
                ) / int(iterations)
            rows.append({
                **source,
                "arm": arm["arm"],
                "columns_per_iter": arm["columns_per_iter"],
                "selection": arm["selection"],
                "diversity_weight": arm["diversity_weight"],
                "outcome": row_outcome,
                "certified": row_outcome == "certified",
                "stop_reason": (status or {}).get("stop_reason", ""),
                "cg_wall_s": cg_wall_s if cg_wall_s is not None else "",
                "network_original_build_s": (
                    build_s if build_s is not None else ""
                ),
                "end_to_end_s": (
                    build_s + cg_wall_s
                    if build_s is not None and cg_wall_s is not None else ""
                ),
                "network_cache_hit": metrics.get("cache_hit", ""),
                "network_cache_load_s": metrics.get("cache_io_s", ""),
                "iterations": iterations,
                "pool_columns": pool_columns,
                "retained_pool_growth_per_iteration": retained_growth,
                "L_model": final.get("route_weight", ""),
                "lp_objective": final.get("lp_obj", ""),
                "artificials": final.get("artificials", ""),
                "minimum_reduced_cost": final.get("min_rc", ""),
                "peak_rss_mb": (status or {}).get("peak_rss_mb", ""),
                "telemetry_rows": phase_rows,
                "telemetry_malformed_rows": malformed,
                "master_lp_s": phases.get("master_attempt", 0.0),
                "incidence_construction_s": phases.get(
                    "incidence_construction", 0.0
                ),
                "pricing_batch_s": phases.get(
                    "pricing_extra_columns", 0.0
                ),
                "pricing_exact_best_s": phases.get(
                    "pricing_shortest_path", 0.0
                ),
                "route_insertion_s": phases.get("route_insertion", 0.0),
                "allocations": allocation_count(
                    Path(str(status_path) + ".allocations.tsv")
                ),
                "slurm_job_id_raw": slurm.get("job_id_raw", ""),
                "slurm_state": slurm.get("state", ""),
                "slurm_exit": slurm.get("exit", ""),
                "slurm_elapsed": slurm.get("elapsed", ""),
                "slurm_total_cpu": slurm.get("total_cpu", ""),
                "slurm_max_rss": slurm.get("max_rss", ""),
                "slurm_node": slurm.get("node", ""),
                "status_path": str(status_path),
                "cache_path": str(cache),
            })

    row_path = root / "cg_acceleration_rows.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for arm in arms:
        for scale in (13, 20):
            group = [
                row for row in rows
                if row["arm"] == arm["arm"] and row["scale"] == scale
            ]
            if not group:
                continue
            counts = Counter(row["outcome"] for row in group)
            certified = [row for row in group if row["certified"]]
            observed_wall = [
                float(row["cg_wall_s"]) for row in group
                if row["cg_wall_s"] != ""
            ]
            total_wall = sum(observed_wall)
            summary.append({
                "arm": arm["arm"],
                "scale": scale,
                "rows": len(group),
                "certified": len(certified),
                "wall_limit": counts["wall_limit"],
                "external_signal": counts["external_signal"],
                "missing": counts["missing"],
                "execution_failure": sum(
                    count for label, count in counts.items()
                    if label.startswith("execution_")
                ),
                "certification_rate": len(certified) / len(group),
                "certified_cg_wall_h_median": (
                    median([
                        float(row["cg_wall_s"]) for row in certified
                        if row["cg_wall_s"] != ""
                    ]) / 3600 if certified else ""
                ),
                "iterations_median": median([
                    float(row["iterations"]) for row in group
                    if row["iterations"] != ""
                ]),
                "pool_columns_median": median([
                    float(row["pool_columns"]) for row in group
                    if row["pool_columns"] != ""
                ]),
                "retained_pool_growth_per_iteration_median": median([
                    float(row["retained_pool_growth_per_iteration"])
                    for row in group
                    if row["retained_pool_growth_per_iteration"] != ""
                ]),
                "cache_load_s_median": median([
                    float(row["network_cache_load_s"]) for row in group
                    if row["network_cache_load_s"] != ""
                ]),
                "pricing_share_aggregate": (
                    sum(float(row["pricing_batch_s"]) for row in group)
                    / total_wall if total_wall else ""
                ),
                "master_share_aggregate": (
                    sum(float(row["master_lp_s"]) for row in group)
                    / total_wall if total_wall else ""
                ),
                "allocations_total": sum(int(row["allocations"]) for row in group),
            })
    summary_path = root / "cg_acceleration_by_arm_scale.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(
            row["arm"], f"k{row['scale']}",
            f"certified={row['certified']}/{row['rows']}",
            f"wall={row['wall_limit']}",
            f"median_iter={row['iterations_median']}",
            f"median_pool={row['pool_columns_median']}",
        )
    print(f"Row CSV: {row_path}")
    print(f"Summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

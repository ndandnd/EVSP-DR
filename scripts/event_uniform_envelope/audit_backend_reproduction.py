#!/usr/bin/env python3
"""Compare Gurobi and native-HiGHS solves of the same immutable RAW pools."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        return {}, "missing"
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            return {}, "not_object"
        return value, ""
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def load_slurm(path: Path) -> dict[str, dict]:
    result = {}
    if not path.is_file():
        return result
    with path.open(newline="") as handle:
        for fields in csv.reader(handle, delimiter="|"):
            if len(fields) < 9:
                continue
            job_id, _, state, exit_code, elapsed, rss, vm, req, node = fields[:9]
            result[job_id] = {
                "state": state, "exit": exit_code, "elapsed": elapsed,
                "rss": rss, "vm": vm, "req": req, "node": node,
            }
    return result


def job_ids(root: Path) -> list[str]:
    values = []
    for path in sorted(root.glob("highs_native*_jobs.tsv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row.get("stage") == "mip_highs_native":
                    values.append(row["array_job_id"])
    return values


def task(slurm: dict, jobs: list[str], index: str) -> dict:
    for job in reversed(jobs):
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", choices=("A", "B"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    slurm = load_slurm(args.sacct)
    jobs = job_ids(root)
    rows = []
    with args.manifest.open(newline="") as handle:
        for source in csv.DictReader(handle, delimiter="\t"):
            stem = f"{args.panel}__{source['cell']}__{source['representation_id']}.json"
            gurobi, gurobi_error = load(root / "mip" / stem)
            highs, highs_error = load(root / "mip_highs_native" / stem)
            slurm_row = task(slurm, jobs, source["index"])
            gb = gurobi.get("buses")
            hb = highs.get("buses")
            gr = gurobi.get("runtime_s")
            hr = highs.get("runtime_s")
            rows.append({
                "index": source["index"],
                "cell": source["cell"],
                "target_fleet": source["target_fleet"],
                "representation_id": source["representation_id"],
                "source_status_sha256": source["source_status_sha256"],
                "source_journal_sha256": source["source_journal_sha256"],
                "gurobi_present": not bool(gurobi_error),
                "gurobi_error": gurobi_error,
                "gurobi_status": gurobi.get("status_name"),
                "gurobi_buses": gb,
                "gurobi_bound": gurobi.get("mip_bound", gurobi.get("fleet_bound")),
                "gurobi_gap": gurobi.get("mip_gap"),
                "gurobi_fleet_proven": gurobi.get("fleet_proven"),
                "gurobi_optimality_scope": gurobi.get("optimality_scope"),
                "gurobi_runtime_s": gr,
                "gurobi_peak_rss_mb": gurobi.get("peak_rss_mb"),
                "gurobi_physical_witness_valid": gurobi.get("physical_witness_valid"),
                "gurobi_source_hash_match": (
                    gurobi.get("source_result_sha256") == source["source_status_sha256"]
                    and gurobi.get("source_journal_sha256") == source["source_journal_sha256"]
                ),
                "highs_present": not bool(highs_error),
                "highs_error": highs_error,
                "highs_backend": highs.get("backend"),
                "highs_status": highs.get("status_name"),
                "highs_buses": hb,
                "highs_bound": highs.get("fleet_bound"),
                "highs_gap": highs.get("mip_gap"),
                "highs_fleet_proven": highs.get("fleet_proven"),
                "highs_optimality_scope": highs.get("optimality_scope"),
                "highs_runtime_s": hr,
                "highs_peak_rss_mb": highs.get("peak_rss_mb"),
                "highs_physical_witness_valid": highs.get("physical_witness_valid"),
                "highs_source_hash_match": (
                    highs.get("source_result_sha256") == source["source_status_sha256"]
                    and highs.get("source_journal_sha256") == source["source_journal_sha256"]
                ),
                "fleet_agreement": gb == hb if gb is not None and hb is not None else None,
                "fleet_difference_highs_minus_gurobi": (
                    hb - gb if isinstance(gb, int) and isinstance(hb, int) else None
                ),
                "proof_agreement": (
                    gurobi.get("fleet_proven") == highs.get("fleet_proven")
                    if gurobi and highs else None
                ),
                "runtime_ratio_highs_over_gurobi": (
                    float(hr) / float(gr) if gr and hr else None
                ),
                "highs_slurm_state": slurm_row.get("state"),
                "highs_slurm_exit": slurm_row.get("exit"),
                "highs_slurm_elapsed": slurm_row.get("elapsed"),
                "highs_slurm_max_rss": slurm_row.get("rss"),
                "highs_slurm_node": slurm_row.get("node"),
            })
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(
        "missing" if not row["highs_present"]
        else "fleet_agree" if row["fleet_agreement"]
        else "fleet_disagree"
        for row in rows
    )
    print(f"{args.panel} native-HiGHS reproduction: {dict(sorted(counts.items()))}")
    print(f"CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

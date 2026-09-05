#!/usr/bin/env python3
"""Normalize the four 36-hour RAW-pool k=5 Gurobi results into CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_accounting(path: Path) -> dict[str, dict[str, str]]:
    fields = (
        "job_id_raw", "job_name", "state", "exit", "elapsed",
        "total_cpu", "max_rss", "max_vm_size", "node",
    )
    records = {}
    with path.open(newline="") as handle:
        for values in csv.reader(handle, delimiter="|"):
            if len(values) < len(fields):
                continue
            record = dict(zip(fields, values))
            raw = record["job_id_raw"]
            if "." not in raw:
                records[raw] = record
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "snapshot_manifest.csv").open(newline="") as handle:
        sources = list(csv.DictReader(handle))
    if len(sources) != 4:
        raise SystemExit(f"expected four source rows, found {len(sources)}")
    jobs = sorted(root.glob("jobs_*.tsv"))
    if len(jobs) != 1:
        raise SystemExit(f"expected one job record, found {len(jobs)}")
    with jobs[0].open(newline="") as handle:
        job_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(job_rows) != 1:
        raise SystemExit("expected one MIP array job record")
    job = job_rows[0]
    accounting = load_accounting(args.sacct)
    rows = []
    for source in sources:
        index = source["local_index"]
        task = f"{job['array_job_id']}_{index}"
        slurm = accounting.get(task, {})
        result_path = (
            root / "mip"
            / f"M__{source['cell']}__{source['representation_id']}.raw_pool_mip36h.json"
        )
        result = {}
        error = None
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                error = f"invalid_result_json:{exc}"
        else:
            error = "missing_result"
        physical = result.get("physical_pool_audit") or {}
        selected = result.get("selected_routes")
        selected_physics = {
            (route.get("physical_realization") or {}).get("status")
            for route in selected or []
            if isinstance(route, dict)
        }
        experiment_configuration_valid = bool(
            result.get("experiment_arm") == "B"
            and result.get("extra_route_sources") == []
            and physical.get("added_giro_route_count") == 0
            and physical.get("post_augmentation_columns")
            == result.get("pool_columns")
        )
        physical_witness_valid = bool(
            result.get("incumbent_found") is True
            and result.get("partitioning") is True
            and isinstance(selected, list)
            and len(selected) == result.get("buses")
            and result.get("overcovered_trips") == 0
            and physical.get("rejected_columns") is not None
            and selected_physics <= {
                "valid_as_recorded_mapped", "deterministically_repaired",
            }
            and None not in selected_physics
            and experiment_configuration_valid
        )
        source_match = (
            result.get("source_result_sha256") == source["snapshot_sha256"]
            and result.get("source_journal_sha256") == source["journal_sha256"]
        )
        code = result.get("mip_provenance") or {}
        code_match = (
            code.get("expected_git_commit") == job["wrapper_commit"]
            and code.get("observed_git_commit") == job["wrapper_commit"]
            and code.get("final_observed_git_commit") == job["wrapper_commit"]
            and code.get("git_detached") is True
            and code.get("git_dirty") is False
        )
        buses = result.get("buses")
        rows.append({
            **source,
            "result_present": result_path.is_file(),
            "result_error": error,
            "result": str(result_path),
            "result_sha256": sha256(result_path),
            "status_name": result.get("status_name"),
            "optimal_scope": result.get("optimal_scope"),
            "buses": buses,
            "fleet_bound": result.get("fleet_bound"),
            "fleet_proven_over_pool": result.get("fleet_proven"),
            "target_fleet_recovered": (
                buses is not None and int(buses) <= int(source["target_fleet"])
            ),
            "incumbent_found": result.get("incumbent_found"),
            "physical_witness_valid": physical_witness_valid,
            "experiment_configuration_valid": experiment_configuration_valid,
            "source_hash_match": source_match,
            "code_identity_match": code_match,
            "runtime_s": result.get("runtime_s"),
            "physical_pool_preparation_wall_s": result.get(
                "physical_pool_preparation_wall_s"
            ),
            "mip_gap": result.get("mip_gap"),
            "node_count": result.get("node_count"),
            "pool_columns_after_physical_gate": result.get("pool_columns"),
            "physical_columns_accepted": physical.get("accepted_columns"),
            "physical_columns_rejected": physical.get("rejected_columns"),
            "slurm_task": task,
            "slurm_state": slurm.get("state"),
            "slurm_exit": slurm.get("exit"),
            "slurm_elapsed": slurm.get("elapsed"),
            "slurm_total_cpu": slurm.get("total_cpu"),
            "slurm_max_rss": slurm.get("max_rss"),
            "slurm_max_vm_size": slurm.get("max_vm_size"),
            "slurm_node": slurm.get("node"),
        })
    output = root / "k5_raw_mip36h_summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['cell']}: buses={row['buses']} bound={row['fleet_bound']} "
            f"pool_proven={row['fleet_proven_over_pool']} "
            f"target_recovered={row['target_fleet_recovered']} "
            f"physical={row['physical_witness_valid']} "
            f"slurm={row['slurm_state']}/{row['slurm_exit']}"
        )
    unsafe = [
        row for row in rows
        if row["slurm_state"] != "COMPLETED"
        or row["slurm_exit"] != "0:0"
        or not row["source_hash_match"]
        or not row["code_identity_match"]
        or not row["experiment_configuration_valid"]
        or (row["incumbent_found"] and not row["physical_witness_valid"])
    ]
    print(f"gate: {'BLOCKED_UNSAFE' if unsafe else 'REVIEW_RESULTS'}")
    print(f"CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit eight-hour HiGHS retries against Gurobi and the two-hour retry."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from audit_backend_reproduction import load
from audit_highs_disagreement_retry import (
    as_float,
    classification,
    load_accounting,
    progress,
    sha256,
    slurm_task,
)


RETRYABLE = {
    "fleet_agreement_highs_unproven",
    "fleet_agreement_both_unproven",
    "highs_unproven",
    "both_unproven",
}
RESOLVED = {
    "proven_fleet_agreement",
    "fleet_agreement_gurobi_unproven",
    "gurobi_unproven",
}
UNSAFE = {
    "missing_or_invalid_artifact",
    "source_identity_error",
    "retry_configuration_error",
    "missing_incumbent",
    "invalid_or_unverified_physical_witness",
    "contradictory_proven_fleets",
    "prior_audit_error",
    "slurm_execution_error",
}


def yes(row: dict, key: str) -> bool:
    return row.get(key) == "True"


def integer(value) -> int | None:
    number = as_float(value)
    if number is None or abs(number - round(number)) > 1e-7:
        return None
    return int(round(number))


def job_record(root: Path, panel: str) -> tuple[dict, list[str]]:
    with (root / "highs_unresolved_retry28800_jobs.tsv").open(
        newline=""
    ) as handle:
        rows = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row.get("panel") == panel
        ]
    if len(rows) != 1:
        raise SystemExit(f"expected one Panel {panel} eight-hour job record")
    indices = [item for item in rows[0]["indices"].split(",") if item]
    if not indices or len(indices) != len(set(indices)):
        raise SystemExit(f"invalid Panel {panel} eight-hour indices")
    return rows[0], indices


def prior_valid(row: dict) -> bool:
    return (
        row.get("classification") in RETRYABLE
        and all(yes(row, key) for key in (
            "gurobi_present", "gurobi_source_hash_match",
            "gurobi_physical_witness_valid", "highs2_present",
            "highs2_source_hash_match", "highs2_physical_witness_valid",
            "highs2_configuration_match",
        ))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", choices=("A", "B"), required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "backend_retry7200.csv").open(newline="") as handle:
        prior = {row["index"]: row for row in csv.DictReader(handle)}
    record, indices = job_record(root, args.panel)
    expected = [
        index for index, row in prior.items()
        if row.get("classification") in RETRYABLE
    ]
    if indices != expected:
        raise SystemExit(
            f"Panel {args.panel} job indices differ from audited selection: "
            f"jobs={indices} audit={expected}"
        )
    accounting = load_accounting(args.sacct)
    rows = []
    for index in indices:
        old = prior[index]
        stem = (
            f"{args.panel}__{old['cell']}__"
            f"{old['representation_id']}.json"
        )
        path = root / "mip_highs_native_retry28800" / stem
        retry, error = load(path)
        code = retry.get("code_identity") or {}
        config = (
            retry.get("backend") == record["backend"]
            and as_float(retry.get("requested_timelimit_s"))
            == as_float(record["timelimit_s"])
            and str(retry.get("threads_requested")) == record["threads"]
            and code.get("expected_commit") == record["solver_commit"]
            and code.get("observed_commit") == record["solver_commit"]
        )
        source_match = (
            retry.get("source_result_sha256") == old["source_status_sha256"]
            and retry.get("source_journal_sha256")
            == old["source_journal_sha256"]
        )
        slurm = slurm_task(accounting, record["array_job_id"], index)
        if not prior_valid(old):
            outcome = "prior_audit_error"
        elif slurm.get("state") != "COMPLETED" or slurm.get("exit") != "0:0":
            outcome = "slurm_execution_error"
        else:
            gurobi = {
                "buses": integer(old.get("gurobi_buses")),
                "fleet_proven": yes(old, "gurobi_fleet_proven"),
            }
            outcome = classification(
                errors=[error],
                identities=[source_match],
                gurobi_payload=gurobi,
                retry_payload=retry,
                gurobi_physical=True,
                retry_physical=retry.get("physical_witness_valid"),
                retry_configuration_match=config,
            )
        old_highs = {
            "buses": integer(old.get("highs2_buses")),
            "fleet_bound": as_float(old.get("highs2_bound")),
            "fleet_proven": yes(old, "highs2_fleet_proven"),
        }
        change = progress(old_highs, retry)
        rows.append({
            "panel": args.panel,
            "index": index,
            "cell": old["cell"],
            "target_fleet": old["target_fleet"],
            "representation_id": old["representation_id"],
            "source_status_sha256": old["source_status_sha256"],
            "source_journal_sha256": old["source_journal_sha256"],
            "prior_2h_classification": old["classification"],
            "classification": outcome,
            "retry_progress_from_2h": change,
            "gurobi_buses": old["gurobi_buses"],
            "gurobi_bound": old["gurobi_bound"],
            "gurobi_fleet_proven": old["gurobi_fleet_proven"],
            "highs30_buses": old["highs30_buses"],
            "highs30_bound": old["highs30_bound"],
            "highs30_fleet_proven": old["highs30_fleet_proven"],
            "highs2_buses": old["highs2_buses"],
            "highs2_bound": old["highs2_bound"],
            "highs2_fleet_proven": old["highs2_fleet_proven"],
            "highs2_runtime_s": old["highs2_runtime_s"],
            "highs8_present": not bool(error),
            "highs8_error": error,
            "highs8_status": retry.get("status_name"),
            "highs8_buses": retry.get("buses"),
            "highs8_bound": retry.get("fleet_bound"),
            "highs8_gap": retry.get("mip_gap"),
            "highs8_fleet_proven": retry.get("fleet_proven"),
            "highs8_optimality_scope": retry.get("optimality_scope"),
            "highs8_runtime_s": retry.get("runtime_s"),
            "highs8_peak_rss_mb": retry.get("peak_rss_mb"),
            "highs8_physical_witness_valid": retry.get(
                "physical_witness_valid"
            ),
            "highs8_source_hash_match": source_match,
            "highs8_configuration_match": config,
            "highs8_artifact_sha256": sha256(path),
            "highs8_array_job_id": record["array_job_id"],
            "highs8_slurm_task": f"{record['array_job_id']}_{index}",
            "highs8_slurm_job_id_raw": slurm.get("job_id_raw"),
            "highs8_slurm_state": slurm.get("state"),
            "highs8_slurm_exit": slurm.get("exit"),
            "highs8_slurm_elapsed": slurm.get("elapsed"),
            "highs8_slurm_total_cpu": slurm.get("total_cpu"),
            "highs8_slurm_max_rss": slurm.get("max_rss"),
            "highs8_slurm_max_vm_size": slurm.get("max_vm_size"),
            "highs8_slurm_node": slurm.get("node"),
        })
    output = root / "backend_retry28800.csv"
    unresolved = root / "backend_retry28800_unresolved.csv"
    unresolved_rows = [
        row for row in rows if row["classification"] not in RESOLVED
    ]
    for path, values in ((output, rows), (unresolved, unresolved_rows)):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(values)
    counts = Counter(row["classification"] for row in rows)
    changes = Counter(row["retry_progress_from_2h"] for row in rows)
    print(f"Panel {args.panel} 8h classifications: {dict(sorted(counts.items()))}")
    print(f"Panel {args.panel} 2h->8h progress: {dict(sorted(changes.items()))}")
    for row in rows:
        print(
            f"{args.panel} {row['index']} {row['cell']} "
            f"{row['representation_id']} {row['classification']} / "
            f"{row['retry_progress_from_2h']} | "
            f"G={row['gurobi_buses']}/{row['gurobi_bound']} "
            f"proven={row['gurobi_fleet_proven']} | "
            f"H2={row['highs2_buses']}/{row['highs2_bound']} "
            f"proven={row['highs2_fleet_proven']} | "
            f"H8={row['highs8_buses']}/{row['highs8_bound']} "
            f"proven={row['highs8_fleet_proven']}"
        )
    gate = (
        "BLOCKED_UNSAFE" if any(
            row["classification"] in UNSAFE for row in rows
        ) else "REVIEW_UNRESOLVED" if unresolved_rows else "RESOLVED"
    )
    print(f"Panel {args.panel} 8h gate: {gate}")
    print(f"CSV: {output}")
    print(f"Unresolved CSV: {unresolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

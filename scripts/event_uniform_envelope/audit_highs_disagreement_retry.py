#!/usr/bin/env python3
"""Audit the two-hour native-HiGHS retries against both prior backends."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from audit_backend_reproduction import gurobi_physical_witness, load


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_accounting(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    with path.open(newline="") as handle:
        for fields in csv.reader(handle, delimiter="|"):
            if len(fields) < 13:
                continue
            (
                job_id, job_id_raw, job_name, state, exit_code, elapsed,
                time_limit, total_cpu, alloc_cpus, max_rss, max_vm_size,
                req_mem, node,
            ) = (field.strip() for field in fields[:13])
            rows[job_id] = {
                "job_id_raw": job_id_raw,
                "job_name": job_name,
                "state": state,
                "exit": exit_code,
                "elapsed": elapsed,
                "time_limit": time_limit,
                "total_cpu": total_cpu,
                "alloc_cpus": alloc_cpus,
                "max_rss": max_rss,
                "max_vm_size": max_vm_size,
                "req_mem": req_mem,
                "node": node,
            }
    return rows


def retry_job(root: Path, panel: str) -> tuple[dict, list[str]]:
    path = root / "highs_disagreement_retry_jobs.tsv"
    with path.open(newline="") as handle:
        selected = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row.get("panel") == panel
        ]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one {panel} retry job in {path}")
    row = selected[0]
    indices = [value for value in row["indices"].split(",") if value]
    if not indices:
        raise ValueError(f"empty retry index list for Panel {panel}")
    return row, indices


def slurm_task(accounting: dict[str, dict], job: str, index: str) -> dict:
    task_id = f"{job}_{index}"
    result = dict(accounting.get(task_id, {}))
    batch = accounting.get(f"{task_id}.batch", {})
    for key in ("max_rss", "max_vm_size", "total_cpu"):
        if batch.get(key):
            result[key] = batch[key]
    return result


def source_matches(payload: dict, manifest_row: dict) -> bool:
    return (
        payload.get("source_result_sha256")
        == manifest_row["source_status_sha256"]
        and payload.get("source_journal_sha256")
        == manifest_row["source_journal_sha256"]
    )


def valid_witness(payload: dict, *, gurobi: bool = False) -> tuple[bool | None, str]:
    if gurobi:
        return gurobi_physical_witness(payload)
    if "physical_witness_valid" in payload:
        return payload.get("physical_witness_valid"), "explicit"
    return None, "unavailable"


def as_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def classification(
    *,
    errors: list[str],
    identities: list[bool],
    gurobi_payload: dict,
    retry_payload: dict,
    gurobi_physical: bool | None,
    retry_physical: bool | None,
    retry_configuration_match: bool,
) -> str:
    if any(errors):
        return "missing_or_invalid_artifact"
    if not all(identities):
        return "source_identity_error"
    if not retry_configuration_match:
        return "retry_configuration_error"
    if gurobi_payload.get("buses") is None or retry_payload.get("buses") is None:
        return "missing_incumbent"
    if gurobi_physical is not True or retry_physical is not True:
        return "invalid_or_unverified_physical_witness"
    same = gurobi_payload["buses"] == retry_payload["buses"]
    gurobi_proven = gurobi_payload.get("fleet_proven") is True
    retry_proven = retry_payload.get("fleet_proven") is True
    if same and gurobi_proven and retry_proven:
        return "proven_fleet_agreement"
    if same and gurobi_proven:
        return "fleet_agreement_highs_unproven"
    if same and retry_proven:
        return "fleet_agreement_gurobi_unproven"
    if same:
        return "fleet_agreement_both_unproven"
    if gurobi_proven and retry_proven:
        return "contradictory_proven_fleets"
    if gurobi_proven:
        return "highs_unproven"
    if retry_proven:
        return "gurobi_unproven"
    return "both_unproven"


def progress(short: dict, retry: dict) -> str:
    if not short or not retry:
        return "not_comparable"
    if short.get("fleet_proven") is not True and retry.get("fleet_proven") is True:
        return "became_proven"
    if short.get("buses") != retry.get("buses"):
        return "incumbent_changed"
    old_bound = as_float(short.get("fleet_bound"))
    new_bound = as_float(retry.get("fleet_bound"))
    if old_bound is not None and new_bound is not None and new_bound > old_bound + 1e-7:
        return "bound_improved"
    if retry.get("fleet_proven") is True:
        return "remained_proven"
    return "no_proof_change"


def normalized_gurobi(payload: dict) -> dict:
    stage1 = payload.get("two_stage") or {}
    return {
        "status": stage1.get("stage1_status_name", payload.get("status_name")),
        "final_status": payload.get("status_name"),
        "buses": payload.get("buses"),
        "bound": payload.get("fleet_bound", stage1.get("stage1_bound")),
        "gap": stage1.get("stage1_gap", payload.get("mip_gap")),
        "proven": payload.get("fleet_proven"),
        "scope": payload.get("optimal_scope", payload.get("optimality_scope")),
        "runtime_s": payload.get("runtime_s"),
        "peak_rss_mb": payload.get("peak_rss_mb"),
    }


def normalized_highs(payload: dict) -> dict:
    return {
        "backend": payload.get("backend"),
        "status": payload.get("status_name"),
        "buses": payload.get("buses"),
        "bound": payload.get("fleet_bound"),
        "gap": payload.get("mip_gap"),
        "proven": payload.get("fleet_proven"),
        "scope": payload.get("optimality_scope"),
        "runtime_s": payload.get("runtime_s"),
        "peak_rss_mb": payload.get("peak_rss_mb"),
        "requested_timelimit_s": payload.get("requested_timelimit_s"),
        "threads_requested": payload.get("threads_requested"),
    }


def audit_panel(
    root: Path, panel: str, accounting: dict[str, dict]
) -> tuple[list[dict], Counter, Counter]:
    manifest_name = (
        "panel_a_highs_inputs.tsv" if panel == "A"
        else "panel_b_highs_inputs.tsv"
    )
    manifest_path = root / manifest_name
    with manifest_path.open(newline="") as handle:
        manifest = {
            row["index"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }
    job_record, indices = retry_job(root, panel)
    job = job_record["array_job_id"]
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate retry indices for Panel {panel}")
    rows = []
    for index in indices:
        if index not in manifest:
            raise ValueError(f"Panel {panel} retry index {index} absent from manifest")
        source = manifest[index]
        stem = f"{panel}__{source['cell']}__{source['representation_id']}.json"
        gurobi_path = root / "mip" / stem
        short_path = root / "mip_highs_native" / stem
        retry_path = root / "mip_highs_native_retry7200" / stem
        gurobi, gurobi_error = load(gurobi_path)
        short, short_error = load(short_path)
        retry, retry_error = load(retry_path)
        gurobi_physical, gurobi_physical_source = valid_witness(
            gurobi, gurobi=True
        )
        short_physical, short_physical_source = valid_witness(short)
        retry_physical, retry_physical_source = valid_witness(retry)
        retry_code_identity = retry.get("code_identity") or {}
        retry_configuration_match = (
            retry.get("backend") == job_record["backend"]
            and as_float(retry.get("requested_timelimit_s"))
            == as_float(job_record["timelimit_s"])
            and str(retry.get("threads_requested")) == job_record["threads"]
            and retry_code_identity.get("expected_commit")
            == job_record["solver_commit"]
            and retry_code_identity.get("observed_commit")
            == job_record["solver_commit"]
        )
        identities = [
            source_matches(payload, source) if payload else False
            for payload in (gurobi, short, retry)
        ]
        cls = classification(
            errors=[gurobi_error, short_error, retry_error],
            identities=identities,
            gurobi_payload=gurobi,
            retry_payload=retry,
            gurobi_physical=gurobi_physical,
            retry_physical=retry_physical,
            retry_configuration_match=retry_configuration_match,
        )
        retry_progress = progress(short, retry)
        g = normalized_gurobi(gurobi)
        h30 = normalized_highs(short)
        h2 = normalized_highs(retry)
        slurm = slurm_task(accounting, job, index)
        row = {
            "panel": panel,
            "index": index,
            "cell": source["cell"],
            "target_fleet": source["target_fleet"],
            "representation_id": source["representation_id"],
            "source_status_sha256": source["source_status_sha256"],
            "source_journal_sha256": source["source_journal_sha256"],
            "classification": cls,
            "retry_progress": retry_progress,
            "gurobi_present": not bool(gurobi_error),
            "gurobi_error": gurobi_error,
            "gurobi_status": g["status"],
            "gurobi_final_status": g["final_status"],
            "gurobi_buses": g["buses"],
            "gurobi_bound": g["bound"],
            "gurobi_gap": g["gap"],
            "gurobi_fleet_proven": g["proven"],
            "gurobi_optimality_scope": g["scope"],
            "gurobi_runtime_s": g["runtime_s"],
            "gurobi_peak_rss_mb": g["peak_rss_mb"],
            "gurobi_physical_witness_valid": gurobi_physical,
            "gurobi_physical_witness_source": gurobi_physical_source,
            "gurobi_source_hash_match": identities[0],
            "gurobi_artifact_sha256": sha256(gurobi_path),
            "highs30_present": not bool(short_error),
            "highs30_error": short_error,
            "highs30_backend": h30["backend"],
            "highs30_status": h30["status"],
            "highs30_buses": h30["buses"],
            "highs30_bound": h30["bound"],
            "highs30_gap": h30["gap"],
            "highs30_fleet_proven": h30["proven"],
            "highs30_optimality_scope": h30["scope"],
            "highs30_runtime_s": h30["runtime_s"],
            "highs30_peak_rss_mb": h30["peak_rss_mb"],
            "highs30_physical_witness_valid": short_physical,
            "highs30_physical_witness_source": short_physical_source,
            "highs30_source_hash_match": identities[1],
            "highs30_artifact_sha256": sha256(short_path),
            "highs2_present": not bool(retry_error),
            "highs2_error": retry_error,
            "highs2_backend": h2["backend"],
            "highs2_status": h2["status"],
            "highs2_buses": h2["buses"],
            "highs2_bound": h2["bound"],
            "highs2_gap": h2["gap"],
            "highs2_fleet_proven": h2["proven"],
            "highs2_optimality_scope": h2["scope"],
            "highs2_runtime_s": h2["runtime_s"],
            "highs2_peak_rss_mb": h2["peak_rss_mb"],
            "highs2_requested_timelimit_s": h2["requested_timelimit_s"],
            "highs2_threads_requested": h2["threads_requested"],
            "highs2_physical_witness_valid": retry_physical,
            "highs2_physical_witness_source": retry_physical_source,
            "highs2_source_hash_match": identities[2],
            "highs2_artifact_sha256": sha256(retry_path),
            "highs2_configuration_match": retry_configuration_match,
            "highs2_expected_solver_commit": job_record["solver_commit"],
            "highs2_observed_solver_commit": retry_code_identity.get(
                "observed_commit"
            ),
            "highs2_recorded_backend": job_record["backend"],
            "highs2_recorded_timelimit_s": job_record["timelimit_s"],
            "highs2_recorded_threads": job_record["threads"],
            "highs2_recorded_partition": job_record["partition"],
            "highs2_incumbent_changed": (
                retry.get("buses") != short.get("buses")
            ),
            "highs2_became_proven": (
                short.get("fleet_proven") is not True
                and retry.get("fleet_proven") is True
            ),
            "highs2_bound_improved": (
                as_float(retry.get("fleet_bound")) is not None
                and as_float(short.get("fleet_bound")) is not None
                and as_float(retry.get("fleet_bound"))
                > as_float(short.get("fleet_bound")) + 1e-7
            ),
            "highs30_difference_from_gurobi": (
                short.get("buses") - gurobi.get("buses")
                if isinstance(short.get("buses"), int)
                and isinstance(gurobi.get("buses"), int) else None
            ),
            "highs2_difference_from_gurobi": (
                retry.get("buses") - gurobi.get("buses")
                if isinstance(retry.get("buses"), int)
                and isinstance(gurobi.get("buses"), int) else None
            ),
            "highs2_difference_from_highs30": (
                retry.get("buses") - short.get("buses")
                if isinstance(retry.get("buses"), int)
                and isinstance(short.get("buses"), int) else None
            ),
            "highs2_array_job_id": job,
            "highs2_slurm_task": f"{job}_{index}",
            "highs2_slurm_job_id_raw": slurm.get("job_id_raw"),
            "highs2_slurm_state": slurm.get("state"),
            "highs2_slurm_exit": slurm.get("exit"),
            "highs2_slurm_elapsed": slurm.get("elapsed"),
            "highs2_slurm_timelimit": slurm.get("time_limit"),
            "highs2_slurm_total_cpu": slurm.get("total_cpu"),
            "highs2_slurm_alloc_cpus": slurm.get("alloc_cpus"),
            "highs2_slurm_max_rss": slurm.get("max_rss"),
            "highs2_slurm_max_vm_size": slurm.get("max_vm_size"),
            "highs2_slurm_req_mem": slurm.get("req_mem"),
            "highs2_slurm_node": slurm.get("node"),
        }
        rows.append(row)
    output = root / "backend_retry7200.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    unresolved = root / "backend_retry7200_unresolved.csv"
    resolved_classes = {
        "proven_fleet_agreement",
        "fleet_agreement_gurobi_unproven",
        "gurobi_unproven",
    }
    unresolved_rows = [
        row for row in rows if row["classification"] not in resolved_classes
    ]
    with unresolved.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(unresolved_rows)
    counts = Counter(row["classification"] for row in rows)
    progress_counts = Counter(row["retry_progress"] for row in rows)
    print(f"Panel {panel} 2h classifications: {dict(sorted(counts.items()))}")
    print(f"Panel {panel} retry progress: {dict(sorted(progress_counts.items()))}")
    for row in rows:
        print(
            f"{panel} {row['index']} {row['cell']} {row['representation_id']} "
            f"{row['classification']} / {row['retry_progress']} | "
            f"G={row['gurobi_buses']}/{row['gurobi_bound']} "
            f"proven={row['gurobi_fleet_proven']} | "
            f"H30={row['highs30_buses']}/{row['highs30_bound']} "
            f"proven={row['highs30_fleet_proven']} | "
            f"H2={row['highs2_buses']}/{row['highs2_bound']} "
            f"proven={row['highs2_fleet_proven']}"
        )
    print(f"CSV: {output}")
    print(f"Unresolved CSV: {unresolved}")
    return rows, counts, progress_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-a", type=Path, required=True)
    parser.add_argument("--panel-b", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    args = parser.parse_args()
    accounting = load_accounting(args.sacct)
    a_rows, _, _ = audit_panel(args.panel_a.resolve(), "A", accounting)
    b_rows, _, _ = audit_panel(args.panel_b.resolve(), "B", accounting)
    rows = a_rows + b_rows
    unsafe = {
        "missing_or_invalid_artifact",
        "source_identity_error",
        "retry_configuration_error",
        "missing_incumbent",
        "invalid_or_unverified_physical_witness",
        "contradictory_proven_fleets",
    }
    unsafe_rows = [row for row in rows if row["classification"] in unsafe]
    unresolved = [
        row for row in rows
        if row["classification"] in {
            "fleet_agreement_highs_unproven",
            "fleet_agreement_both_unproven",
            "highs_unproven",
            "both_unproven",
        }
    ]
    print(
        f"Combined retry audit: rows={len(rows)} "
        f"unsafe={len(unsafe_rows)} unresolved={len(unresolved)}"
    )
    print(
        "next-run gate: "
        + ("BLOCKED_UNSAFE" if unsafe_rows else "REVIEW_UNRESOLVED" if unresolved else "RESOLVED")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

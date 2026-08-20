#!/usr/bin/env python3
"""Read-only monitor/summary for a k40 factorial strict-MIP campaign."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import subprocess
from pathlib import Path

from k40_factorial_mip_result import validate_scientific_result


FIELDS = (
    "label", "replicate", "treatment", "snapshot_mark_minutes",
    "budget_seconds", "job_id", "slurm_state", "elapsed", "exit_code",
    "output_exists", "output_valid", "validation_error",
    "status_name", "buses", "fleet_bound", "fleet_proven",
    "mip_obj", "mip_bound", "mip_gap", "runtime_s", "start_accepted",
    "start_column_hashes", "optimal_scope", "route_space_scope",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_result(
    output_bundle: Path,
    job: dict,
    manifest: dict,
) -> dict:
    if not output_bundle.is_dir():
        raise ValueError("output bundle is missing")
    result_path = output_bundle / "result.json"
    completion_path = output_bundle / "completion.json"
    result = json.loads(result_path.read_text())
    completion = json.loads(completion_path.read_text())
    if not isinstance(result, dict) or not isinstance(completion, dict):
        raise ValueError("result/completion member is not a JSON object")
    spec_path = Path(job["spec_path"])
    if _sha256(spec_path) != job["spec_sha256"]:
        raise ValueError("persisted job spec hash mismatch")
    spec = json.loads(spec_path.read_text())
    if spec != job["spec"]:
        raise ValueError("persisted job spec content mismatch")
    for path_key, hash_key in (
        ("staged_result", "staged_result_sha256"),
        ("staged_journal", "staged_journal_sha256"),
        ("staged_instance", "staged_instance_sha256"),
        ("staged_prices", "staged_prices_sha256"),
        ("staged_start", "staged_start_sha256"),
    ):
        if _sha256(Path(spec[path_key])) != spec[hash_key]:
            raise ValueError(f"staged {path_key} hash mismatch")
    if (
        completion.get("schema")
        != "evsp-dr-k40-factorial-mip-completion-v1"
        or completion.get("output_bundle") != str(output_bundle)
        or completion.get("result_member") != "result.json"
        or completion.get("output_sha256") != _sha256(result_path)
        or completion.get("job_spec_sha256") != job["spec_sha256"]
        or completion.get("worker_sha256") != manifest["worker_sha256"]
    ):
        raise ValueError("completion attestation mismatch")
    attestation = result.get("completion_attestation")
    if (
        not isinstance(attestation, dict)
        or attestation.get("schema")
        != "evsp-dr-k40-factorial-mip-result-v1"
        or attestation.get("job_spec_sha256") != job["spec_sha256"]
        or attestation.get("worker_sha256") != manifest["worker_sha256"]
        or attestation.get("runner_sha256") != spec["runner_sha256"]
        or attestation.get("validated_start_sha256")
        != spec["staged_start_sha256"]
    ):
        raise ValueError("result attestation mismatch")
    expected_cell = {
        key: spec[key] for key in (
            "label", "replicate", "treatment", "snapshot_mark_minutes",
            "time_limit_s", "threads", "mip_gap",
        )
    }
    if result.get("campaign_cell") != expected_cell:
        raise ValueError("result campaign-cell identity mismatch")
    if (
        result.get("partitioning") is not True
        or result.get("route_space_scope")
        != "finite_augmented_snapshot_pool_only"
        or result.get("source_result_sha256")
        != spec["staged_result_sha256"]
        or result.get("source_journal_sha256")
        != spec["staged_journal_sha256"]
    ):
        raise ValueError("result model/source identity mismatch")
    start = result.get("mip_start")
    acceptance = (start or {}).get("solver_acceptance")
    columns = (start or {}).get("actual_start_columns")
    if (
        not isinstance(start, dict)
        or start.get("kind") != "validated_exact_partition"
        or start.get("source_sha256") != spec["staged_start_sha256"]
        or not isinstance(acceptance, dict)
        or acceptance.get("accepted") is not True
        or not isinstance(columns, list)
        or len(columns) != 40
        or any(
            not isinstance(column, dict)
            or not isinstance(column.get("index"), int)
            or len(str(column.get("sha256") or "")) != 64
            for column in columns
        )
        or len({column["index"] for column in columns}) != 40
    ):
        raise ValueError("result MIP-start evidence is incomplete")
    source = json.loads(Path(spec["staged_result"]).read_text())
    validate_scientific_result(result, spec, source)
    selected = result.get("selected_routes")
    if not isinstance(selected, list) or not selected:
        raise ValueError("result selected routes are missing")
    counts = collections.Counter(
        trip for route in selected for trip in route.get("trips", [])
    )
    if (
        set(counts) != set(source["trip_ids"])
        or any(counts[trip] != 1 for trip in source["trip_ids"])
        or any(
            not isinstance(route.get("charging_stops"), dict)
            for route in selected
        )
        or result.get("buses") != len(selected)
    ):
        raise ValueError("result is not an exact scheduled partition")
    for key in ("mip_obj", "runtime_s"):
        value = result.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"result {key} is invalid")
    for key in (
        "fleet_bound", "fleet_proven", "mip_bound", "mip_gap", "two_stage",
    ):
        if key not in result:
            raise ValueError(f"result omits {key}")
    provenance = result.get("mip_provenance") or {}
    if (
        provenance.get("expected_git_commit")
        != manifest["checkout_identity"]["expected_commit"]
        or provenance.get("observed_git_commit")
        != provenance.get("expected_git_commit")
        or provenance.get("final_observed_git_commit")
        != provenance.get("observed_git_commit")
        or provenance.get("tracked_clean_at_end") is not True
    ):
        raise ValueError("result Git provenance mismatch")
    return result


def _sacct(job_ids):
    if not job_ids:
        return {}
    result = subprocess.run(
        [
            "sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
            "--format=JobID,State,Elapsed,ExitCode",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    records = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and "." not in fields[0]:
            records[fields[0]] = {
                "state": fields[1],
                "elapsed": fields[2],
                "exit_code": fields[3],
            }
    return records


def rows(campaign_root: Path, *, query_slurm=True) -> list[dict]:
    root = campaign_root.expanduser().resolve()
    manifest = json.loads((root / "campaign.json").read_text())
    jobs = manifest.get("jobs") or []
    accounting = _sacct([
        str(job["job_id"]) for job in jobs if job.get("job_id")
    ]) if query_slurm else {}
    output_rows = []
    for job in jobs:
        job_id = str(job.get("job_id") or "")
        slurm = accounting.get(job_id, {})
        output = Path(job["output"])
        result = {}
        output_valid = False
        validation_error = None
        if output.exists():
            try:
                result = _validated_result(output, job, manifest)
                output_valid = True
            except (OSError, ValueError, KeyError, TypeError) as exc:
                validation_error = str(exc)
        acceptance = (
            (result.get("mip_start") or {}).get("solver_acceptance") or {}
        )
        start_columns = (
            (result.get("mip_start") or {}).get("actual_start_columns") or []
        )
        output_rows.append({
            "label": job["label"],
            "replicate": job["replicate"],
            "treatment": job["treatment"],
            "snapshot_mark_minutes": job["snapshot_mark_minutes"],
            "budget_seconds": manifest["budget_seconds"],
            "job_id": job_id or None,
            "slurm_state": slurm.get("state"),
            "elapsed": slurm.get("elapsed"),
            "exit_code": slurm.get("exit_code"),
            "output_exists": output.exists(),
            "output_valid": output_valid,
            "validation_error": validation_error,
            "status_name": result.get("status_name"),
            "buses": result.get("buses"),
            "fleet_bound": result.get("fleet_bound"),
            "fleet_proven": result.get("fleet_proven"),
            "mip_obj": result.get("mip_obj"),
            "mip_bound": result.get("mip_bound"),
            "mip_gap": result.get("mip_gap"),
            "runtime_s": result.get("runtime_s"),
            "start_accepted": acceptance.get("accepted"),
            "start_column_hashes": len(start_columns) or None,
            "optimal_scope": result.get("optimal_scope"),
            "route_space_scope": result.get("route_space_scope"),
        })
    return output_rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--no-sacct", action="store_true")
    parser.add_argument("--format", choices=("tsv", "json"), default="tsv")
    args = parser.parse_args(argv)
    output_rows = rows(
        args.campaign_root, query_slurm=not args.no_sacct
    )
    if args.format == "json":
        print(json.dumps(output_rows, indent=2))
        return 0
    print("\t".join(FIELDS))
    for row in output_rows:
        print("\t".join(
            "" if row.get(field) is None else str(row[field])
            for field in FIELDS
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

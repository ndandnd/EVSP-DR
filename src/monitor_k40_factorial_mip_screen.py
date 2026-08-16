#!/usr/bin/env python3
"""Read-only monitor/summary for a k40 factorial strict-MIP campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


FIELDS = (
    "label", "replicate", "treatment", "snapshot_mark_minutes",
    "budget_seconds", "job_id", "slurm_state", "elapsed", "exit_code",
    "output_exists", "output_valid", "validation_error",
    "status_name", "buses", "fleet_bound", "fleet_proven",
    "mip_obj", "mip_bound", "mip_gap", "runtime_s", "start_accepted",
    "start_column_hashes", "optimal_scope", "route_space_scope",
)


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
        if output.is_file():
            try:
                loaded = json.loads(output.read_text())
                completion_path = Path(str(output) + ".complete.json")
                completion = json.loads(completion_path.read_text())
                output_sha = hashlib.sha256(output.read_bytes()).hexdigest()
                if not isinstance(loaded, dict):
                    raise ValueError("MIP result is not a JSON object")
                if not isinstance(completion, dict):
                    raise ValueError("completion sidecar is not a JSON object")
                attestation = loaded.get("completion_attestation")
                if (
                    completion.get("schema")
                    != "evsp-dr-k40-factorial-mip-completion-v1"
                    or completion.get("output_sha256") != output_sha
                    or not isinstance(attestation, dict)
                    or attestation.get("job_spec_sha256")
                    != job["spec_sha256"]
                    or loaded.get("route_space_scope")
                    != "finite_augmented_snapshot_pool_only"
                ):
                    raise ValueError("completion attestation mismatch")
                result = loaded
                output_valid = True
            except (OSError, ValueError, KeyError) as exc:
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
            "output_exists": output.is_file(),
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

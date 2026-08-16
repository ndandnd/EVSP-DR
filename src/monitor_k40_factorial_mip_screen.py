#!/usr/bin/env python3
"""Read-only publication/recovery monitor for k40 factorial MIP campaigns."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from recover_k40_factorial_mip_campaign import build_recovery_plan


def _accounting(job_ids: list[str]) -> dict:
    if not job_ids:
        return {}
    result = subprocess.run(
        [
            "sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
            "--format=JobIDRaw,JobName,State,ExitCode,Comment",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {"_error": result.stderr.strip()}
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 5 and "." not in fields[0]:
            rows[fields[0]] = {
                "job_name": fields[1],
                "state": fields[2],
                "exit_code": fields[3],
                "comment": fields[4],
            }
    return rows


def monitor(
    campaign_root: Path,
    *,
    source_campaign_sha256: str,
    query_slurm: bool = False,
) -> list[dict]:
    plan, _prepared = build_recovery_plan(
        campaign_root,
        source_campaign_sha256=source_campaign_sha256,
    )
    rows = []
    accounting = _accounting([
        row["job_id"] for row in plan["jobs"] if row["job_id"]
    ]) if query_slurm else {}
    manifest = json.loads(
        (campaign_root.expanduser().resolve() / "campaign.json").read_text()
    )
    jobs_by_label = {
        job["label"]: job for job in manifest.get("jobs") or []
    }
    for job in plan["jobs"]:
        state = job["publication_state"]
        if state == "complete_valid":
            outcome = "complete_valid_output"
        elif job["recoverable"]:
            outcome = "recoverable_validated_raw"
        elif state == "incomplete_publication":
            outcome = "incomplete_publication"
        else:
            outcome = "missing_or_invalid_result"
        source_job = jobs_by_label.get(job["label"]) or {}
        slurm = accounting.get(job["job_id"])
        rows.append({
            **job,
            "outcome": outcome,
            "recovery_commit": plan["recovery_commit"],
            "slurm": slurm,
            "slurm_identity_valid": (
                None if slurm is None
                else (
                    slurm.get("job_name") == source_job.get("job_name")
                    and slurm.get("comment")
                    == source_job.get("slurm_comment")
                )
            ),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--source-campaign-sha256", required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--no-slurm", action="store_true")
    parser.add_argument("--format", choices=("json", "tsv"), default="tsv")
    args = parser.parse_args(argv)
    rows = monitor(
        args.campaign_root,
        source_campaign_sha256=args.source_campaign_sha256,
        query_slurm=not args.no_slurm,
    )
    allowed = (
        {"complete_valid_output"}
        if args.require_complete
        else {"complete_valid_output", "recoverable_validated_raw"}
    )
    exit_code = 0 if all(
        row["outcome"] in allowed for row in rows
    ) else 2
    if args.format == "json":
        print(json.dumps(rows, indent=2))
        return exit_code
    fields = (
        "label", "job_id", "outcome", "publication_state",
        "recoverable", "recovery_method", "candidate_path",
        "raw_sha256", "recovered_result_sha256", "errors",
    )
    print("\t".join(fields))
    for row in rows:
        print("\t".join(
            " | ".join(row[field]) if field == "errors"
            else ("" if row.get(field) is None else str(row[field]))
            for field in fields
        ))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

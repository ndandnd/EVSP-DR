#!/usr/bin/env python3
"""Find accepted-but-unrecorded profile jobs by campaign-unique Slurm name."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from durable_io import flush_and_fsync


def _query(
    job_name: str, start_date: str, expected_comment: str,
) -> list[dict]:
    result = subprocess.run(
        [
            "sacct", "-X", "-n", "-P",
            "--name", job_name,
            "--starttime", start_date,
            "--format=JobID,JobName,State,Submit,Start,Elapsed,ExitCode,Comment",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "sacct failed")
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if (len(fields) < 8 or "." in fields[0]
                or fields[1] != job_name
                or fields[7] != expected_comment):
            continue
        rows.append({
            "job_id": fields[0],
            "job_name": fields[1],
            "state": fields[2],
            "submit": fields[3],
            "start": fields[4],
            "elapsed": fields[5],
            "exit_code": fields[6],
            "comment": fields[7],
        })
    return rows


def reconcile(campaign_root: Path, *, apply: bool = False) -> dict:
    root = campaign_root.expanduser().resolve()
    manifest_path = root / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    start_date = str(manifest.get("created_at", ""))[:10]
    recovered = []
    unresolved = []
    recorded = []
    for job in manifest.get("jobs") or []:
        if job.get("job_id"):
            recorded.append({
                "label": job.get("label"),
                "job_id": str(job["job_id"]),
                "submission_state": job.get("submission_state"),
            })
            continue
        attempt_marker = root / f".{job.get('label')}.attempt.json"
        needs_query = (
            job.get("submission_state") in {"attempting", "failed"}
            or attempt_marker.exists()
        )
        if not needs_query:
            continue
        matches = _query(
            job["job_name"], start_date, job["slurm_comment"]
        )
        if len(matches) != 1:
            unresolved.append({
                "label": job["label"],
                "job_name": job["job_name"],
                "matches": matches,
            })
            continue
        match = matches[0]
        recovered.append({"label": job["label"], **match})
        if apply:
            job["job_id"] = match["job_id"]
            job["submission_state"] = "submitted_reconciled"
            job["reconciled_slurm_state"] = match["state"]
    if apply and recovered:
        manifest["submitted"] = all(
            job.get("job_id") and job.get("submission_state") in {
                "submitted", "submitted_reconciled",
            }
            for job in manifest.get("jobs") or []
        )
        temporary = manifest_path.with_name(
            f".{manifest_path.name}.tmp.{os.getpid()}"
        )
        with temporary.open("w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
            flush_and_fsync(handle)
        os.replace(temporary, manifest_path)
    return {
        "campaign": manifest.get("campaign"),
        "apply": apply,
        "recorded": recorded,
        "recovered": recovered,
        "unresolved": unresolved,
        "safe_to_retry": (
            not recorded
            and not recovered
            and not unresolved
            and all(
                not job.get("job_id")
                and job.get("submission_state") == "planned"
                and not (root / f".{job.get('label')}.attempt.json").exists()
                for job in manifest.get("jobs") or []
            )
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    result = reconcile(args.campaign_root, apply=args.apply)
    print(json.dumps(result, indent=2))
    return 0 if not result["unresolved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

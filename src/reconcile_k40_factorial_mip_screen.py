#!/usr/bin/env python3
"""Reconcile ambiguous k40 MIP submissions using unique names and accounting."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from durable_io import flush_and_fsync


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Slurm query failed")
    return completed.stdout


def _query(
    job_name: str,
    start_date: str,
    expected_comment: str,
    expected_user: str,
) -> list[dict]:
    matches = {}
    for line in _run([
        "squeue", "-h", "--name", job_name, "-o", "%i|%j|%T|%u|%k",
    ]).splitlines():
        fields = line.split("|")
        if (
            len(fields) < 5
            or fields[1] != job_name
            or fields[3] != expected_user
        ):
            continue
        comment = fields[4]
        if comment != expected_comment:
            continue
        matches[fields[0]] = {
            "job_id": fields[0],
            "job_name": fields[1],
            "state": fields[2],
            "comment": comment,
            "source": "squeue",
        }
    for line in _run([
        "sacct", "-X", "-n", "-P", "--name", job_name,
        "--starttime", start_date,
        "--format=JobIDRaw,JobName,State,Submit,Start,Elapsed,ExitCode,User,Comment",
    ]).splitlines():
        fields = line.split("|")
        if (
            len(fields) < 9
            or "." in fields[0]
            or fields[1] != job_name
            or fields[7] != expected_user
        ):
            continue
        comment = fields[8]
        if comment != expected_comment:
            continue
        matches[fields[0]] = {
            "job_id": fields[0],
            "job_name": fields[1],
            "state": fields[2],
            "submit": fields[3],
            "start": fields[4],
            "elapsed": fields[5],
            "exit_code": fields[6],
            "user": fields[7],
            "comment": comment,
            "source": "sacct",
        }
    return sorted(matches.values(), key=lambda row: row["job_id"])


def _validate_manifest(manifest: dict) -> list[dict]:
    if manifest.get("schema") != "evsp-dr-k40-factorial-mip-campaign-v1":
        raise ValueError("not a k40 factorial MIP campaign manifest")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("MIP campaign has no jobs")
    mode = manifest.get("mode")
    if mode == "screen" and len(jobs) != 12:
        raise ValueError("screen campaign must contain exactly 12 jobs")
    if mode == "escalation" and not 1 <= len(jobs) <= 12:
        raise ValueError("escalation campaign job count is invalid")
    if mode not in {"screen", "escalation"}:
        raise ValueError("MIP campaign mode is invalid")
    if not isinstance(manifest.get("submission_user"), str) \
            or not manifest["submission_user"]:
        raise ValueError("MIP campaign submission user is invalid")
    for field in ("label", "job_name", "slurm_comment"):
        values = [job.get(field) for job in jobs]
        if (
            any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"MIP campaign {field} values are invalid")
    if any(len(job["job_name"]) > 15 for job in jobs):
        raise ValueError("MIP campaign has an overlong Slurm job name")
    expected_cells = {
        (rep, treatment, mark)
        for rep in ("R1", "R2")
        for treatment in ("CA", "CS")
        for mark in (360, 720, 1440)
    }
    actual_cells = {
        (
            job.get("replicate"),
            job.get("treatment"),
            job.get("snapshot_mark_minutes"),
        )
        for job in jobs
    }
    if len(actual_cells) != len(jobs) or not actual_cells <= expected_cells:
        raise ValueError("MIP campaign contains invalid/duplicate cells")
    if mode == "screen" and actual_cells != expected_cells:
        raise ValueError("screen campaign does not contain the exact 12 cells")
    for job in jobs:
        expected_label = (
            f"{job['replicate']}_{job['treatment']}_"
            f"m{job['snapshot_mark_minutes']}"
        )
        if job["label"] != expected_label:
            raise ValueError("MIP campaign label/cell mismatch")
    return jobs


def reconcile(campaign_root: Path, *, apply: bool = False) -> dict:
    root = campaign_root.expanduser().resolve()
    manifest_path = root / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    jobs = _validate_manifest(manifest)
    start_date = str(manifest.get("created_at") or "")[:10]
    if len(start_date) != 10:
        raise ValueError("campaign created_at is invalid")
    recovered, unresolved, recorded, pending = [], [], [], []
    for job in jobs:
        if job.get("job_id"):
            recorded.append({
                "label": job["label"],
                "job_id": str(job["job_id"]),
                "submission_state": job.get("submission_state"),
            })
            continue
        attempt = root / f".{job['label']}.attempt.json"
        attempted = (
            job.get("submission_state") in {"attempting", "failed"}
            or attempt.exists()
        )
        if not attempted:
            if job.get("submission_state") != "planned":
                raise ValueError(
                    f"{job['label']} has an invalid submission state"
                )
            pending.append({"label": job["label"]})
            continue
        matches = _query(
            job["job_name"],
            start_date,
            job["slurm_comment"],
            manifest["submission_user"],
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
        manifest["submitted"] = all(job.get("job_id") for job in jobs)
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
        "pending": pending,
        "unresolved": unresolved,
        "safe_to_submit_pending": not unresolved,
        "safe_to_retry_as_new_campaign": (
            not recorded and not recovered and not unresolved
            and len(pending) == len(jobs)
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

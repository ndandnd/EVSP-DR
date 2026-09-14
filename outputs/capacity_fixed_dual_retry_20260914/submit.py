#!/usr/bin/env python3
"""Submit the four independent frozen fixed-dual production cases."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
import campaign

SBATCH = "/usr/local/slurm/slurm-25.05.5/bin/sbatch"


def main(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    freeze = json.loads((root / "freeze.json").read_text())
    manifest_sha = campaign.sha256_file(manifest_path)
    if freeze["manifest_sha256"] != manifest_sha:
        raise ValueError("manifest freeze mismatch")
    for name, expected in manifest["tooling_sha256"].items():
        if campaign.sha256_file(root / name) != expected:
            raise ValueError(f"tooling hash mismatch: {name}")
    jobs_path = root / "jobs.json"
    if jobs_path.exists():
        raise FileExistsError("jobs.json exists; refusing duplicate submission")
    jobs = {"schema": "evsp-dr-fixed-capacity-jobs-v1",
            "manifest_sha256": manifest_sha, "jobs": {}}
    for case_id, case in manifest["cases"].items():
        command = [
            SBATCH, "--parsable", "--no-requeue", "--partition", "default_partition",
            "--exclude", "scaglione-compute-01", "--cpus-per-task", "1",
            "--mem", "24G", "--time", "04:15:00",
            "--job-name", "cfd_" + case_id.replace("capacity_", "cap_")[:55],
            "--output", str(root / "logs" / f"{case_id}_%j.out"),
            "--error", str(root / "logs" / f"{case_id}_%j.err"),
            str(root / "worker.sh"), case_id,
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(f"sbatch failed for {case_id}: {completed.stderr}")
        job_id = completed.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"ambiguous sbatch response for {case_id}: {completed.stdout!r}")
        jobs["jobs"][case_id] = {
            "job_id": job_id, "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "argv": command, "stdout": completed.stdout.strip(),
        }
        campaign.atomic_json(jobs_path, jobs)
        print(case_id, job_id, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve())

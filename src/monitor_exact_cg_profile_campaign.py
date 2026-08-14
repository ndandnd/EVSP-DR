#!/usr/bin/env python3
"""Read-only Slurm/artifact monitor for one exact-CG profile campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _accounting(job_ids: list[str]) -> dict[str, dict]:
    if not job_ids:
        return {}
    result = subprocess.run(
        [
            "sacct", "-X", "-n", "-P",
            "-j", ",".join(job_ids),
            "--format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {"_error": {"error": result.stderr.strip()}}
    records = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 6 or "." in fields[0]:
            continue
        records[fields[0]] = {
            "job_name": fields[1],
            "state": fields[2],
            "elapsed": fields[3],
            "exit_code": fields[4],
            "max_rss": fields[5],
        }
    return records


def monitor(campaign_root: Path, *, query_slurm: bool = True) -> list[dict]:
    root = campaign_root.expanduser().resolve()
    manifest = json.loads((root / "campaign.json").read_text())
    jobs = manifest.get("jobs") or []
    ids = [str(job["job_id"]) for job in jobs if job.get("job_id")]
    accounting = _accounting(ids) if query_slurm else {}
    rows = []
    for job in jobs:
        job_id = str(job.get("job_id") or "")
        output = Path(job["output"])
        artifact = None
        if output.is_file():
            try:
                payload = json.loads(output.read_text())
                artifact = {
                    "valid_json": isinstance(payload, dict),
                    "schema": payload.get("schema"),
                    "source_unchanged": payload.get("source_unchanged"),
                }
            except (OSError, ValueError):
                artifact = {"valid_json": False}
        row = {
            "label": job["label"],
            "job_id": job_id or None,
            "job_name": job.get("job_name"),
            "submission_state": job.get("submission_state"),
            "slurm": accounting.get(job_id),
            "output": str(output),
            "output_exists": output.is_file(),
            "artifact": artifact,
        }
        if "_error" in accounting:
            row["accounting_error"] = accounting["_error"]["error"]
        rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--no-sacct", action="store_true")
    parser.add_argument("--format", choices=("tsv", "json"), default="tsv")
    args = parser.parse_args(argv)
    rows = monitor(args.campaign_root, query_slurm=not args.no_sacct)
    if args.format == "json":
        print(json.dumps(rows, indent=2))
        return 0
    fields = (
        "label", "job_id", "job_name", "submission_state", "state",
        "elapsed", "exit_code", "max_rss", "output_exists",
    )
    print("\t".join(fields))
    for row in rows:
        slurm = row.get("slurm") or {}
        values = {
            **row,
            "state": slurm.get("state"),
            "elapsed": slurm.get("elapsed"),
            "exit_code": slurm.get("exit_code"),
            "max_rss": slurm.get("max_rss"),
        }
        print("\t".join(
            "" if values.get(field) is None else str(values[field])
            for field in fields
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

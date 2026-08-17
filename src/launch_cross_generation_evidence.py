#!/usr/bin/env python3
"""Dry-run-first Slurm launcher for cross-generation evidence jobs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from collect_cross_generation_inputs import _assignments
from run_cross_generation_evidence_job import (
    REQUIRED_ROOT_ALIASES,
    parse_campaign_assignments,
)


def _git(repo: Path, *args) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True,
        capture_output=True, check=True,
    )
    return result.stdout.strip()


def _campaign_values(args) -> list[str]:
    values = list(getattr(args, "campaign", []) or [])
    current = getattr(args, "current_mip_campaign_root", None)
    current_mode = getattr(args, "current_mip_mode", None)
    raw = getattr(args, "raw_k40_campaign_root", None)
    if current is not None:
        if current_mode is None:
            raise ValueError(
                "compatibility current campaign requires current mode"
            )
        values.append(f"{current_mode}={current.expanduser().absolute()}")
    if raw is not None:
        values.append(f"raw_k40={raw.expanduser().absolute()}")
    return values


def _dependency_job_ids(args) -> list[str]:
    values = list(getattr(args, "after_job_id", []) or [])
    if any(not re.fullmatch(r"[0-9]+", value) for value in values):
        raise ValueError("Slurm dependency job IDs must be decimal integers")
    if len(values) != len(set(values)):
        raise ValueError("Slurm dependency job IDs must be unique")
    return values


def build_sbatch_command(args) -> tuple[list[str], dict]:
    repo = args.repo_root.expanduser().resolve()
    roots = _assignments(args.root)
    missing = sorted(REQUIRED_ROOT_ALIASES - set(roots))
    if missing:
        raise ValueError(f"explicit source roots missing: {missing}")
    campaigns = parse_campaign_assignments(_campaign_values(args))
    dependency_ids = _dependency_job_ids(args)
    wait_timeout_s = getattr(args, "wait_timeout_s", None)
    if wait_timeout_s is None:
        wait_timeout_s = 0.0 if dependency_ids else 86400.0
    if args.phase == "build" and (
        not args.approved_manifest_sha256
        or args.build_out is None
        or args.archive_out is None
    ):
        raise ValueError(
            "build phase requires approved manifest SHA, build output, archive"
        )
    worker = repo / "src/run_cross_generation_evidence_job.py"
    worker_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
    python = Path(sys.executable).resolve()
    worker_args = [
        str(python), "-u", str(worker),
        "--phase", args.phase,
        "--template", str(args.template.expanduser().resolve()),
        "--manifest", str(args.manifest.expanduser().resolve()),
        "--wait-timeout-s", str(wait_timeout_s),
        "--poll-s", str(args.poll_s),
        "--repo-root", str(repo),
        "--expected-commit", args.expected_commit,
    ]
    for campaign, mode in campaigns:
        worker_args.extend(["--campaign", f"{mode}={campaign}"])
    for alias, path in sorted(roots.items()):
        worker_args.extend(["--root", f"{alias}={path}"])
    if args.phase == "build":
        worker_args.extend([
            "--approved-manifest-sha256",
            args.approved_manifest_sha256,
            "--build-out", str(args.build_out.expanduser().absolute()),
            "--archive-out", str(args.archive_out.expanduser().absolute()),
        ])
    args.log_dir.mkdir(parents=True, exist_ok=True)
    job_name = "EVXBUILD" if args.phase == "build" else "EVXCOLLECT"
    sbatch = [
        "sbatch", "--parsable", "--no-requeue",
        f"--job-name={job_name}",
        f"--partition={args.partition}",
        "--nodes=1", "--ntasks=1",
        f"--cpus-per-task={args.cpus}",
        f"--mem={args.memory}",
        f"--time={args.time_limit}",
        f"--output={args.log_dir.resolve()}/{job_name}-%j.out",
        f"--error={args.log_dir.resolve()}/{job_name}-%j.err",
        "--export=NONE",
    ]
    if dependency_ids:
        sbatch.append(
            f"--dependency=afterany:{':'.join(dependency_ids)}"
        )
    sbatch.append(f"--wrap={shlex.join(worker_args)}")
    plan = {
        "schema": "evsp-dr-cross-generation-slurm-plan-v1",
        "phase": args.phase,
        "submits_cg_or_mip_solves": False,
        "worker": str(worker),
        "worker_sha256": worker_sha,
        "python_executable": str(python),
        "python_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
        "expected_commit": args.expected_commit,
        "roots": {
            key: str(value) for key, value in sorted(roots.items())
        },
        "campaigns": [
            {"mode": mode, "path": str(campaign)}
            for campaign, mode in campaigns
        ],
        "after_job_ids": dependency_ids,
        "worker_wait_timeout_s": wait_timeout_s,
        "sbatch_argv": sbatch,
    }
    return sbatch, plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("collect", "build"), required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approved-manifest-sha256")
    parser.add_argument("--build-out", type=Path)
    parser.add_argument("--archive-out", type=Path)
    parser.add_argument(
        "--campaign", action="append", default=[],
        help="Repeatable MODE=ABSOLUTE_PATH; MODE is pilot, secondary, raw_k40.",
    )
    parser.add_argument("--after-job-id", action="append", default=[])
    parser.add_argument(
        "--current-mip-campaign-root", type=Path
    )
    parser.add_argument(
        "--raw-k40-campaign-root", type=Path
    )
    parser.add_argument(
        "--current-mip-mode",
        choices=("pilot", "secondary"),
    )
    parser.add_argument("--wait-timeout-s", type=float)
    parser.add_argument("--poll-s", type=float, default=300.0)
    parser.add_argument("--partition", default="scaglione")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory", default="32G")
    parser.add_argument("--time-limit", default="24:00:00")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--expected-commit")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo_root.expanduser().resolve()
    observed_commit = _git(repo, "rev-parse", "HEAD")
    if args.expected_commit is None:
        args.expected_commit = observed_commit
    if observed_commit != args.expected_commit:
        raise ValueError("launcher checkout differs from expected commit")
    if _git(repo, "status", "--porcelain"):
        raise ValueError("launcher checkout is not tracked-clean")
    sbatch, plan = build_sbatch_command(args)
    canonical = json.dumps(
        plan, sort_keys=True, separators=(",", ":")
    ).encode()
    plan["plan_sha256"] = hashlib.sha256(canonical).hexdigest()
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.submit:
        print("[dry-run] evidence job not submitted")
        return 0
    result = subprocess.run(
        sbatch, text=True, capture_output=True, check=True
    )
    job_id = result.stdout.strip()
    if not job_id.isdigit():
        raise RuntimeError(f"unexpected sbatch response: {job_id!r}")
    print(json.dumps({
        "submitted_evidence_job_id": job_id,
        "phase": args.phase,
        "submits_cg_or_mip_solves": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

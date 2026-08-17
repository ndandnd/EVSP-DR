#!/usr/bin/env python3
"""Dry-run-first Slurm launcher for cross-generation evidence jobs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

from collect_cross_generation_inputs import _assignments
from run_cross_generation_evidence_job import REQUIRED_ROOT_ALIASES


def _git(repo: Path, *args) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True,
        capture_output=True, check=True,
    )
    return result.stdout.strip()


def build_sbatch_command(args) -> tuple[list[str], dict]:
    repo = args.repo_root.expanduser().resolve()
    roots = _assignments(args.root)
    missing = sorted(REQUIRED_ROOT_ALIASES - set(roots))
    if missing:
        raise ValueError(f"explicit source roots missing: {missing}")
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
        "--current-mip-campaign-root",
        str(args.current_mip_campaign_root.expanduser().resolve()),
        "--raw-k40-campaign-root",
        str(args.raw_k40_campaign_root.expanduser().resolve()),
        "--current-mip-mode", args.current_mip_mode,
        "--wait-timeout-s", str(args.wait_timeout_s),
        "--poll-s", str(args.poll_s),
        "--repo-root", str(repo),
        "--expected-commit", args.expected_commit,
    ]
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
        f"--wrap={shlex.join(worker_args)}",
    ]
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
        "current_mip_campaign_root": str(
            args.current_mip_campaign_root.expanduser().resolve()
        ),
        "raw_k40_campaign_root": str(
            args.raw_k40_campaign_root.expanduser().resolve()
        ),
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
        "--current-mip-campaign-root", type=Path, required=True
    )
    parser.add_argument(
        "--raw-k40-campaign-root", type=Path, required=True
    )
    parser.add_argument(
        "--current-mip-mode",
        choices=("pilot", "secondary"),
        required=True,
    )
    parser.add_argument("--wait-timeout-s", type=float, default=86400.0)
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

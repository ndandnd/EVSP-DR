#!/usr/bin/env python3
"""Compute-node worker for collecting/building cross-generation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from archive_cross_generation_evidence import archive_evidence
from build_cross_generation_evidence import build
from collect_cross_generation_inputs import _assignments, collect
from summarize_mip_statistics import _load_campaign


REQUIRED_ROOT_ALIASES = {
    "current_heuristic",
    "repool_small",
    "exact_big",
    "k40_factorial",
    "mip_campaign",
    "releases",
}


def _campaign_ready(root: Path) -> tuple[bool, str]:
    campaign = root.expanduser().resolve()
    manifest_path = campaign / "campaign.json"
    plan_path = campaign / "approved-plan.json"
    if not manifest_path.is_file() or not plan_path.is_file():
        return False, "campaign_or_approved_plan_missing"
    try:
        manifest = json.loads(manifest_path.read_text())
        plan_raw = plan_path.read_bytes()
    except (OSError, json.JSONDecodeError):
        return False, "campaign_metadata_unreadable"
    if hashlib.sha256(plan_raw).hexdigest() != manifest.get(
            "approval_sha256"):
        return False, "campaign_approval_sha_mismatch"
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        return False, "campaign_jobs_missing"
    for job in jobs:
        output = Path(str(job.get("output") or ""))
        progress = Path(str(job.get("progress_dir") or ""))
        if (
            not output.is_file()
            or not (progress / "final.json").is_file()
        ):
            return False, f"incomplete_cell:{job.get('cell_id')}"
    try:
        _load_campaign(campaign)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"campaign_validation_failed:{type(exc).__name__}:{exc}"
    return True, "complete_valid"


def wait_for_campaigns(
    campaign_roots: list[Path],
    *,
    timeout_s: float,
    poll_s: float,
) -> None:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        states = [
            (root, *_campaign_ready(root)) for root in campaign_roots
        ]
        if states and all(ready for _root, ready, _reason in states):
            return
        if time.monotonic() >= deadline:
            detail = " | ".join(
                f"{root}:{reason}"
                for root, ready, reason in states if not ready
            )
            raise RuntimeError(
                f"MIP campaign incomplete after wait: {detail}"
            )
        time.sleep(max(1.0, float(poll_s)))


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(parent_fd)
    os.close(parent_fd)


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
    parser.add_argument("--wait-timeout-s", type=float, default=0.0)
    parser.add_argument("--poll-s", type=float, default=300.0)
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "cross-generation collection/build must run in a Slurm job"
        )
    repo = args.repo_root.expanduser().resolve()
    observed_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if observed_commit != args.expected_commit or dirty:
        raise RuntimeError(
            "evidence worker requires reviewed commit and clean checkout"
        )
    roots = _assignments(args.root)
    missing_aliases = sorted(REQUIRED_ROOT_ALIASES - set(roots))
    if missing_aliases:
        raise ValueError(
            f"explicit source roots missing: {missing_aliases}"
        )
    wait_for_campaigns(
        [
            args.current_mip_campaign_root,
            args.raw_k40_campaign_root,
        ],
        timeout_s=args.wait_timeout_s,
        poll_s=args.poll_s,
    )
    if args.phase == "collect":
        payload = collect(args.template, roots)
        _write_manifest(args.manifest, payload)
        print(json.dumps({
            "phase": "collect",
            "manifest": str(args.manifest),
            "manifest_sha256": hashlib.sha256(
                args.manifest.read_bytes()
            ).hexdigest(),
            "artifacts": len(payload.get("artifacts") or []),
        }, indent=2, sort_keys=True))
        return 0
    if (
        not args.approved_manifest_sha256
        or args.build_out is None
        or args.archive_out is None
    ):
        parser.error(
            "build requires --approved-manifest-sha256, "
            "--build-out, and --archive-out"
        )
    result = build(
        args.manifest,
        args.build_out,
        repo_root=args.repo_root,
        command=[
            "python", "-u", "src/run_cross_generation_evidence_job.py",
            "--phase", "build",
            "--manifest", "<REVIEWED_MANIFEST>",
            "--approved-manifest-sha256",
            args.approved_manifest_sha256,
            "--build-out", "<NEW_BUILD_DIR>",
            "--archive-out", "<NEW_ARCHIVE_DIR>",
        ],
        approved_manifest_sha256=args.approved_manifest_sha256,
    )
    archive = archive_evidence(
        args.build_out, args.manifest, args.archive_out
    )
    print(json.dumps({
        "phase": "build",
        "build": result,
        "archive": archive,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

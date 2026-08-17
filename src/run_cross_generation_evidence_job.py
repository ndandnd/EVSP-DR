#!/usr/bin/env python3
"""Compute-node worker for collecting/building cross-generation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from archive_cross_generation_evidence import archive_evidence
from build_cross_generation_evidence import build
from collect_cross_generation_inputs import _assignments, collect
from mip_convergence import checkpoint_schedule_s
from summarize_mip_statistics import _load_campaign
from validate_raw_k40_mip_plan import validate_plan as validate_raw_k40_plan


REQUIRED_ROOT_ALIASES = {
    "current_heuristic",
    "repool_small",
    "exact_big",
    "k40_factorial",
    "mip_campaign",
    "releases",
}


def _assert_slurm_compute_node() -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError(
            "evidence worker requires a Slurm batch allocation"
        )
    job = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        text=True, capture_output=True, check=True,
    )
    fields = {
        token.split("=", 1)[0]: token.split("=", 1)[1]
        for token in job.stdout.split()
        if "=" in token
    }
    if fields.get("JobId") != job_id or not fields.get("NodeList"):
        raise RuntimeError("SLURM_JOB_ID is not a readable allocation")
    node_list = fields["NodeList"]
    result = subprocess.run(
        ["scontrol", "show", "hostnames", node_list],
        text=True, capture_output=True, check=True,
    )
    allocated = {
        line.strip().split(".", 1)[0]
        for line in result.stdout.splitlines() if line.strip()
    }
    host = platform.node().split(".", 1)[0]
    if host not in allocated:
        raise RuntimeError(
            f"host {host} is not in Slurm allocation {job_id}"
        )


def _campaign_ready(
    root: Path, *, expected_mode: str
) -> tuple[bool, str]:
    campaign = root.expanduser().resolve()
    manifest_path = campaign / "campaign.json"
    plan_path = campaign / "approved-plan.json"
    if not manifest_path.is_file() or not plan_path.is_file():
        return False, "campaign_or_approved_plan_missing"
    try:
        manifest = json.loads(manifest_path.read_text())
        plan_raw = plan_path.read_bytes()
        approved_plan = json.loads(plan_raw)
    except (OSError, json.JSONDecodeError):
        return False, "campaign_metadata_unreadable"
    if hashlib.sha256(plan_raw).hexdigest() != manifest.get(
            "approval_sha256"):
        return False, "campaign_approval_sha_mismatch"
    if approved_plan.get("mode") != expected_mode:
        return False, (
            f"campaign_mode_mismatch:{approved_plan.get('mode')}"
            f"!={expected_mode}"
        )
    if expected_mode == "raw_k40":
        try:
            validate_raw_k40_plan(
                approved_plan,
                expected_commit=approved_plan["checkout_identity"][
                    "expected_commit"
                ],
            )
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"raw_k40_validation_failed:{exc}"
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        return False, "campaign_jobs_missing"
    for job in jobs:
        output = Path(str(job.get("output") or ""))
        progress = Path(str(job.get("progress_dir") or ""))
        budget_s = job.get("time_limit_s")
        if budget_s is None and job.get("budget_hours") is not None:
            budget_s = float(job["budget_hours"]) * 3600.0
        expected_checkpoints = {
            f"checkpoint_{int(round(mark / 60)):04d}m.json"
            for mark in checkpoint_schedule_s(float(budget_s or 0.0))
        }
        observed_checkpoints = {
            path.name for path in progress.glob("checkpoint_*.json")
            if path.is_file()
        }
        if (
            not output.is_file()
            or not (progress / "final.json").is_file()
            or not expected_checkpoints.issubset(observed_checkpoints)
        ):
            return False, f"incomplete_cell:{job.get('cell_id')}"
    try:
        _load_campaign(campaign)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"campaign_validation_failed:{type(exc).__name__}:{exc}"
    return True, "complete_valid"


def wait_for_campaigns(
    campaigns: list[tuple[Path, str]],
    *,
    timeout_s: float,
    poll_s: float,
) -> None:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        states = [
            (
                root,
                *_campaign_ready(root, expected_mode=expected_mode),
            )
            for root, expected_mode in campaigns
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


def _require_campaign_artifacts(payload: dict, campaign: Path) -> None:
    manifest = json.loads((campaign / "campaign.json").read_text())
    artifacts = payload.get("artifacts") or []
    by_type = {}
    for artifact in artifacts:
        raw = str(artifact.get("path") or "")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            continue
        by_type.setdefault(artifact.get("artifact_type"), []).append(
            path.resolve()
        )
    for job in manifest.get("jobs") or []:
        cell = str(job.get("cell_id"))
        output = Path(str(job.get("output") or "")).resolve()
        progress = Path(str(job.get("progress_dir") or "")).resolve()
        input_root = campaign / "input" / cell
        final_artifacts = [
            artifact for artifact in artifacts
            if artifact.get("artifact_type") == "mip_final"
            and Path(str(artifact.get("path"))).expanduser().resolve()
            == output
        ]
        if not final_artifacts:
            raise ValueError(
                f"reviewed manifest omits final output for {cell}"
            )
        arm = job.get("arm")
        expected_augmentation = {
            "RAW": "none",
            "MATCHING": "matching_cover",
            "GIRO": "giro_partition",
        }.get(arm)
        if expected_augmentation is None or any(
            (artifact.get("metadata") or {}).get("treatment") != arm
            or (artifact.get("metadata") or {}).get("augmentation_kind")
            != expected_augmentation
            for artifact in final_artifacts
        ):
            raise ValueError(
                f"reviewed manifest treatment differs from approved job {cell}"
            )
        if not any(
            progress in path.parents
            for path in by_type.get("mip_checkpoint", [])
        ):
            raise ValueError(
                f"reviewed manifest omits checkpoints for {cell}"
            )
        if not any(
            input_root in path.parents
            for path in by_type.get("mip_pool_status_json", [])
        ):
            raise ValueError(
                f"reviewed manifest omits source status for {cell}"
            )
        if not any(
            input_root in path.parents
            for path in by_type.get(
                "exact_cg_column_journal_jsonl", []
            )
        ):
            raise ValueError(
                f"reviewed manifest omits source journal for {cell}"
            )


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
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
    parser.add_argument(
        "--current-mip-mode",
        choices=("pilot", "secondary"),
        required=True,
    )
    parser.add_argument("--wait-timeout-s", type=float, default=0.0)
    parser.add_argument("--poll-s", type=float, default=300.0)
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    _assert_slurm_compute_node()
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
    current_campaign = args.current_mip_campaign_root.expanduser().resolve()
    raw_campaign = args.raw_k40_campaign_root.expanduser().resolve()
    if current_campaign == raw_campaign:
        raise ValueError("current and RAW-k40 campaigns must be distinct")
    mip_root = roots["mip_campaign"].resolve()
    if (
        mip_root not in current_campaign.parents
        or mip_root not in raw_campaign.parents
    ):
        raise ValueError(
            "named campaigns must be contained by mip_campaign source root"
        )
    wait_for_campaigns(
        [
            (current_campaign, args.current_mip_mode),
            (raw_campaign, "raw_k40"),
        ],
        timeout_s=args.wait_timeout_s,
        poll_s=args.poll_s,
    )
    campaign_hashes = {
        str(campaign): {
            name: hashlib.sha256(
                (campaign / name).read_bytes()
            ).hexdigest()
            for name in ("campaign.json", "approved-plan.json")
        }
        for campaign in (current_campaign, raw_campaign)
    }
    if args.phase == "collect":
        payload = collect(args.template, roots)
        for campaign in (current_campaign, raw_campaign):
            _require_campaign_artifacts(payload, campaign)
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
    reviewed_payload = json.loads(args.manifest.read_text())
    for campaign in (current_campaign, raw_campaign):
        _require_campaign_artifacts(reviewed_payload, campaign)
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
        args.build_out,
        args.manifest,
        args.archive_out,
        (current_campaign, raw_campaign),
        campaign_hashes,
    )
    print(json.dumps({
        "phase": "build",
        "build": result,
        "archive": archive,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

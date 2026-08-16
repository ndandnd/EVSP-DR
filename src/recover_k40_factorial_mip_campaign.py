#!/usr/bin/env python3
"""Dry-run-first recovery of validated k40 MIP raw results; never solves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from durable_io import flush_and_fsync
from k40_factorial_mip_result import (
    enrich_result,
    publish_result_bundle,
    result_bundle_material,
    sha256_file,
    validate_scientific_result,
)
from portable_bundle import atomic_write_new_file, inspect_bundle

SOURCE_CAMPAIGN_COMMIT = "f40b1206b244cbc9accad272ac852837c8debdb3"
EXPECTED_CELLS = {
    f"{rep}_{treatment}_m{mark}"
    for rep in ("R1", "R2")
    for treatment in ("CA", "CS")
    for mark in (360, 720, 1440)
}

def _canonical(payload: dict) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()


def _approval_payload(manifest: dict) -> dict:
    return {
        **{
            key: value for key, value in manifest.items()
            if key not in {
                "created_at", "submitted", "jobs", "approval_sha256"
            }
        },
        "jobs": [{
            key: value for key, value in job.items()
            if key not in {
                "staged_result_bytes", "spec_bytes", "job_id",
                "submission_state", "submission_error",
                "reconciled_slurm_state",
                "pre_submission_observed_git_commit",
            }
        } for job in manifest.get("jobs") or []],
    }


def _git_commit(*, require_clean: bool) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40:
        raise RuntimeError("recovery checkout has no valid Git commit")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0 or (
        require_clean and status.stdout.strip()
    ):
        raise RuntimeError("recovery checkout has tracked modifications")
    return commit


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _verify_file(path_value, expected, label) -> Path:
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} hash mismatch: {path}")
    return path


def _preserved_inventory(paths: list[Path]) -> list[dict]:
    inventory = []
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"preserved source is symlinked: {path}")
        if path.is_file():
            inventory.append({
                "path": str(path),
                "kind": "file",
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            })
        elif path.is_dir():
            members = {}
            for member in sorted(path.rglob("*")):
                if member.is_symlink():
                    raise ValueError(
                        f"preserved staging member is symlinked: {member}"
                    )
                if member.is_file():
                    members[str(member.relative_to(path))] = {
                        "sha256": sha256_file(member),
                        "size": member.stat().st_size,
                        "type": "file",
                    }
                elif member.is_dir():
                    members[str(member.relative_to(path))] = {
                        "type": "directory"
                    }
            inventory.append({
                "path": str(path),
                "kind": "directory",
                "members": members,
            })
    return inventory


def _candidate_result(
    job: dict, spec: dict
) -> tuple[Path | None, str | None, list[dict]]:
    output = Path(job["output"]).expanduser().resolve()
    job_id = str(job.get("job_id") or "")
    if not job_id.isdigit():
        raise ValueError("job ID is missing/non-numeric")
    raw = Path(str(output) + f".raw.{job_id}")
    temporary_dirs = sorted(output.parent.glob(f".{output.name}.tmp.*"))
    preserved = _preserved_inventory(
        [
            path for path in [raw, *temporary_dirs]
            if os.path.lexists(path)
        ]
    )
    if raw.is_file():
        return raw, "raw_result", preserved
    result_candidates = [
        directory / "result.json"
        for directory in temporary_dirs
        if directory.is_dir() and (directory / "result.json").is_file()
    ]
    if len(result_candidates) == 1:
        return result_candidates[0], "failed_temporary_bundle", preserved
    if len(result_candidates) > 1:
        raise ValueError("multiple ambiguous temporary result bundles")
    return None, None, preserved


def _prepare_job(
    root: Path,
    manifest: dict,
    job: dict,
    recovery_commit: str,
) -> tuple[dict, dict | None]:
    errors = []
    prepared = None
    output = Path(job["output"]).expanduser().resolve()
    state = inspect_bundle(output, required_members=("result.json",))
    preserved = []
    candidate = None
    method = None
    raw_sha = None
    verified_hashes = {}
    try:
        if state["state"] == "invalid":
            raise ValueError(
                "existing destination is invalid: "
                + " | ".join(state.get("errors") or [])
            )
        spec_path = _verify_file(
            job["spec_path"], job["spec_sha256"], "job spec"
        )
        spec = _load_object(spec_path)
        if spec != job["spec"]:
            raise ValueError("job spec differs from campaign manifest")
        for path_key, hash_key, label in (
            ("staged_result", "staged_result_sha256", "status"),
            ("staged_journal", "staged_journal_sha256", "journal"),
            ("staged_instance", "staged_instance_sha256", "instance"),
            ("staged_prices", "staged_prices_sha256", "tariff"),
            ("staged_start", "staged_start_sha256", "validated start"),
        ):
            _verify_file(spec[path_key], spec[hash_key], label)
            verified_hashes[label] = spec[hash_key]
        worker = _verify_file(
            manifest["worker"],
            manifest["worker_sha256"],
            "reviewed worker",
        )
        checkout_root = root.parents[3]
        runner = _verify_file(
            checkout_root / "src/run_exact_pool_mip.py",
            spec["runner_sha256"],
            "reviewed runner",
        )
        del worker, runner
        verified_hashes.update({
            "job_spec": job["spec_sha256"],
            "worker": manifest["worker_sha256"],
            "runner": spec["runner_sha256"],
        })
        candidate, method, preserved = _candidate_result(job, spec)
        if candidate is None:
            raise ValueError("no raw result or temporary result bundle found")
        raw_sha = sha256_file(candidate)
        raw = _load_object(candidate)
        recovery = {
            "job_spec_sha256": job["spec_sha256"],
            "worker_sha256": manifest["worker_sha256"],
            "original_job_id": str(job.get("job_id") or ""),
            "raw_sha256": raw_sha,
            "recovery_commit": recovery_commit,
            "recovery_method": method,
        }
        result = enrich_result(raw, spec=spec, recovery=recovery)
        source = _load_object(Path(spec["staged_result"]))
        validate_scientific_result(result, spec, source)
        result_raw = (
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        ).encode()
        prepared = {
            "result": result,
            "result_sha256": hashlib.sha256(result_raw).hexdigest(),
            "spec": spec,
            "source": source,
            "recovery": recovery,
            "preserved_sources": preserved,
        }
        _members, expected_metadata, expected_completion_sha = (
            result_bundle_material(
                result=result,
                spec=spec,
                recovery=recovery,
            )
        )
        prepared["expected_completion_metadata"] = expected_metadata
        prepared["expected_completion_sha256"] = expected_completion_sha
        existing_result = output / "result.json"
        if (
            os.path.lexists(existing_result)
            and (
                not existing_result.is_file()
                or existing_result.is_symlink()
                or sha256_file(existing_result)
                != prepared["result_sha256"]
            )
        ):
            raise ValueError(
                "existing partial result differs from approved recovery result"
            )
        publication_state = state["state"]
        if publication_state == "complete_valid":
            committed = _load_object(output / "result.json")
            if committed != result:
                raise ValueError(
                    "complete bundle result differs from approved recovery result"
                )
            completion = state.get("completion") or {}
            metadata = completion.get("metadata") or {}
            committed_members = completion.get("members") or {}
            attestation = committed.get("completion_attestation") or {}
            if (
                set(committed_members) != {"result.json"}
                or metadata != expected_metadata
                or state.get("completion_sha256")
                != expected_completion_sha
                or attestation.get("job_spec_sha256")
                != job["spec_sha256"]
                or attestation.get("raw_sha256") != raw_sha
            ):
                raise ValueError(
                    "complete bundle metadata/attestation mismatch"
                )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
        publication_state = (
            "invalid"
            if (
                state["state"] == "complete_valid"
                or os.path.lexists(output / "result.json")
            )
            else state["state"]
        )
        prepared = None
    return ({
        "label": job.get("label"),
        "job_id": str(job.get("job_id") or ""),
        "destination": str(output),
        "publication_state": publication_state,
        "recoverable": prepared is not None,
        "recovery_method": method,
        "candidate_path": str(candidate) if candidate else None,
        "raw_sha256": raw_sha,
        "recovered_result_sha256": (
            prepared["result_sha256"] if prepared else None
        ),
        "expected_completion_sha256": (
            prepared["expected_completion_sha256"] if prepared else None
        ),
        "preserved_sources": preserved,
        "verified_hashes": verified_hashes,
        "errors": errors,
    }, prepared)


def build_recovery_plan(
    campaign_root: Path,
    *,
    source_campaign_sha256: str,
    require_clean=False,
) -> tuple[dict, dict]:
    root = campaign_root.expanduser().resolve()
    manifest_path = root / "campaign.json"
    if sha256_file(manifest_path) != source_campaign_sha256:
        raise ValueError("campaign.json differs from out-of-band approved SHA")
    manifest = _load_object(manifest_path)
    if manifest.get("schema") != "evsp-dr-k40-factorial-mip-campaign-v1":
        raise ValueError("unexpected k40 factorial campaign schema")
    approval = hashlib.sha256(
        _canonical(_approval_payload(manifest))
    ).hexdigest()
    if approval != manifest.get("approval_sha256"):
        raise ValueError("campaign differs from its approval SHA-256")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 12:
        raise ValueError("recovery requires the exact 12-cell campaign")
    labels = [job.get("label") for job in jobs if isinstance(job, dict)]
    job_ids = [str(job.get("job_id") or "") for job in jobs if isinstance(job, dict)]
    if (
        set(labels) != EXPECTED_CELLS
        or len(labels) != len(set(labels))
        or any(not job_id.isdigit() for job_id in job_ids)
        or len(job_ids) != len(set(job_ids))
        or manifest.get("checkout_identity", {}).get("expected_commit")
        != SOURCE_CAMPAIGN_COMMIT
    ):
        raise ValueError("campaign cell/job/source identity mismatch")
    for job in jobs:
        output = Path(str(job.get("output") or "")).expanduser().resolve()
        if root not in output.parents:
            raise ValueError(f"job output escapes campaign root: {output}")
        match = re.fullmatch(r"(R[12])_(CA|CS)_m(360|720|1440)", job["label"])
        if match is None:
            raise ValueError(f"invalid cell label: {job['label']}")
        replicate, treatment, mark = (
            match.group(1), match.group(2), int(match.group(3))
        )
        spec = job.get("spec")
        if (
            not isinstance(spec, dict)
            or job.get("replicate") != replicate
            or job.get("treatment") != treatment
            or int(job.get("snapshot_mark_minutes", -1)) != mark
            or spec.get("label") != job["label"]
            or spec.get("replicate") != replicate
            or spec.get("treatment") != treatment
            or int(spec.get("snapshot_mark_minutes", -1)) != mark
            or int(spec.get("time_limit_s", -1)) != 7200
            or int(spec.get("threads", -1)) != 8
            or Path(str(spec.get("output") or "")).resolve() != output
            or output.parent != root / "outputs"
            or output.name != f"{job['label']}.mip.bundle"
        ):
            raise ValueError(f"cell manifest/spec identity mismatch: {job['label']}")
    recovery_commit = _git_commit(require_clean=require_clean)
    rows = []
    prepared = {}
    for job in sorted(jobs, key=lambda value: value["label"]):
        row, payload = _prepare_job(
            root, manifest, job, recovery_commit
        )
        rows.append(row)
        if payload is not None:
            prepared[job["label"]] = payload
    plan = {
        "schema": "evsp-dr-k40-factorial-mip-recovery-plan-v1",
        "campaign": manifest["campaign"],
        "campaign_root": str(root),
        "source_campaign_approval_sha256": approval,
        "source_campaign_sha256": source_campaign_sha256,
        "source_commit": manifest.get("checkout_identity", {}).get(
            "expected_commit"
        ),
        "recovery_commit": recovery_commit,
        "jobs": rows,
        "recoverable_count": sum(row["recoverable"] for row in rows),
        "complete_count": sum(
            row["publication_state"] == "complete_valid" for row in rows
        ),
        "invalid_or_missing_count": sum(
            not row["recoverable"]
            and row["publication_state"] != "complete_valid"
            for row in rows
        ),
        "reruns_gurobi": False,
        "preserves_raw_and_staging": True,
    }
    return plan, prepared


def _write_new(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir():
        raise ValueError(f"recovery record parent is missing: {path.parent}")
    atomic_write_new_file(path, payload)


def _recovery_intent(plan: dict) -> dict:
    return {
        "schema": "evsp-dr-k40-factorial-mip-recovery-intent-v1",
        "campaign": plan["campaign"],
        "campaign_root": plan["campaign_root"],
        "source_campaign_sha256": plan["source_campaign_sha256"],
        "source_campaign_approval_sha256": plan[
            "source_campaign_approval_sha256"
        ],
        "source_commit": plan["source_commit"],
        "recovery_commit": plan["recovery_commit"],
        "jobs": [{
            key: row.get(key) for key in (
                "label", "job_id", "destination", "recovery_method",
                "candidate_path", "raw_sha256", "recovered_result_sha256",
                "expected_completion_sha256", "preserved_sources",
                "verified_hashes",
            )
        } for row in plan["jobs"]],
        "reruns_gurobi": False,
        "preserves_raw_and_staging": True,
    }


def apply_recovery(
    campaign_root: Path,
    *,
    approved_plan_sha256: str,
    source_campaign_sha256: str,
) -> dict:
    root = campaign_root.expanduser().resolve()
    plan, prepared = build_recovery_plan(
        campaign_root,
        source_campaign_sha256=source_campaign_sha256,
        require_clean=True,
    )
    plan_raw = _canonical(_recovery_intent(plan))
    observed_sha = hashlib.sha256(plan_raw).hexdigest()
    if observed_sha != approved_plan_sha256:
        raise ValueError("current recovery plan differs from approved SHA-256")
    for row in plan["jobs"]:
        if row["publication_state"] == "invalid":
            raise ValueError(
                f"{row['label']} has an invalid existing destination"
            )
        if row["label"] not in prepared:
            raise ValueError(
                f"{row['label']} lacks a validated preserved result"
            )
    existing_record = root / "recovery" / f"{approved_plan_sha256}.json"
    if existing_record.is_file():
        record = _load_object(existing_record)
        expected_receipts = []
        for row in plan["jobs"]:
            destination = Path(row["destination"])
            actual_result = sha256_file(destination / "result.json")
            actual_completion = sha256_file(
                destination / "completion.json"
            )
            if (
                actual_result != row["recovered_result_sha256"]
                or actual_completion
                != row["expected_completion_sha256"]
            ):
                raise ValueError("completed bundle differs from recovery intent")
            expected_receipts.append({
                "label": row["label"],
                "original_job_id": row["job_id"],
                "raw_sha256": row["raw_sha256"],
                "recovery_method": row["recovery_method"],
                "destination": row["destination"],
                "result_sha256": actual_result,
                "completion_sha256": actual_completion,
            })
        if (
            set(record) != {
                "schema", "recovery_plan_sha256", "recovery_commit",
                "completed_labels",
                "raw_and_staging_preserved", "reran_gurobi", "receipts",
            }
            or record.get("schema")
            != "evsp-dr-k40-factorial-mip-recovery-record-v1"
            or record.get("recovery_commit") != plan["recovery_commit"]
            or record.get("recovery_plan_sha256")
            != approved_plan_sha256
            or record.get("raw_and_staging_preserved") is not True
            or record.get("reran_gurobi") is not False
            or record.get("receipts") != expected_receipts
            or record.get("completed_labels")
            != [row["label"] for row in plan["jobs"]]
            or any(
                inspect_bundle(
                    Path(row["destination"]),
                    required_members=("result.json",),
                )["state"] != "complete_valid"
                for row in plan["jobs"]
            )
        ):
            raise ValueError("existing recovery record/bundles are invalid")
        for row in plan["jobs"]:
            payload = prepared[row["label"]]
            if _preserved_inventory([
                Path(item["path"])
                for item in payload["preserved_sources"]
            ]) != payload["preserved_sources"]:
                raise ValueError("preserved raw/staging source changed")
        return record
    for row in plan["jobs"]:
        if row["publication_state"] == "complete_valid":
            continue
        if not row["recoverable"]:
            raise ValueError(
                f"{row['label']} is not recoverable: {row['errors']}"
            )
        payload = prepared[row["label"]]
        before = payload["preserved_sources"]
        publish_result_bundle(
            Path(row["destination"]),
            result=payload["result"],
            spec=payload["spec"],
            source_status=payload["source"],
            recovery=payload["recovery"],
            allow_existing_incomplete=True,
        )
        if (
            sha256_file(Path(row["destination"]) / "result.json")
            != row["recovered_result_sha256"]
            or sha256_file(Path(row["destination"]) / "completion.json")
            != row["expected_completion_sha256"]
        ):
            raise ValueError(
                f"{row['label']} published bytes differ from recovery intent"
            )
        after = _preserved_inventory([
            Path(item["path"]) for item in before
        ])
        if after != before:
            raise ValueError(
                f"{row['label']} raw/staging sources changed during recovery"
            )
    record = {
        "schema": "evsp-dr-k40-factorial-mip-recovery-record-v1",
        "recovery_plan_sha256": observed_sha,
        "recovery_commit": plan["recovery_commit"],
        "completed_labels": [row["label"] for row in plan["jobs"]],
        "raw_and_staging_preserved": True,
        "reran_gurobi": False,
        "receipts": [{
            "label": row["label"],
            "original_job_id": row["job_id"],
            "raw_sha256": row["raw_sha256"],
            "recovery_method": row["recovery_method"],
            "destination": row["destination"],
            "result_sha256": sha256_file(
                Path(row["destination"]) / "result.json"
            ),
            "completion_sha256": sha256_file(
                Path(row["destination"]) / "completion.json"
            ),
        } for row in plan["jobs"]],
    }
    record_path = root / "recovery" / f"{observed_sha}.json"
    if not record_path.parent.exists():
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.mkdir("recovery", 0o700, dir_fd=root_fd)
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
    if record_path.exists():
        existing = _load_object(record_path)
        if existing != record:
            raise ValueError("existing recovery record differs")
    else:
        _write_new(
            record_path,
            (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
        )
    return record


def validate_existing_recovery(
    campaign_root: Path,
    *,
    source_campaign_sha256: str,
    approved_plan_sha256: str,
) -> dict:
    root = campaign_root.expanduser().resolve()
    plan, prepared = build_recovery_plan(
        root,
        source_campaign_sha256=source_campaign_sha256,
    )
    if hashlib.sha256(
        _canonical(_recovery_intent(plan))
    ).hexdigest() != approved_plan_sha256:
        raise ValueError("current recovery intent differs from approved SHA")
    record_path = root / "recovery" / f"{approved_plan_sha256}.json"
    record = _load_object(record_path)
    receipts = []
    for row in plan["jobs"]:
        destination = Path(row["destination"])
        inspection = inspect_bundle(
            destination, required_members=("result.json",)
        )
        if (
            inspection["state"] != "complete_valid"
            or row["label"] not in prepared
            or sha256_file(destination / "result.json")
            != row["recovered_result_sha256"]
            or sha256_file(destination / "completion.json")
            != row["expected_completion_sha256"]
        ):
            raise ValueError(f"{row['label']} complete bundle is invalid")
        receipts.append({
            "label": row["label"],
            "original_job_id": row["job_id"],
            "raw_sha256": row["raw_sha256"],
            "recovery_method": row["recovery_method"],
            "destination": row["destination"],
            "result_sha256": row["recovered_result_sha256"],
            "completion_sha256": row["expected_completion_sha256"],
        })
        if _preserved_inventory([
            Path(item["path"])
            for item in prepared[row["label"]]["preserved_sources"]
        ]) != prepared[row["label"]]["preserved_sources"]:
            raise ValueError(f"{row['label']} preserved inputs changed")
    expected = {
        "schema": "evsp-dr-k40-factorial-mip-recovery-record-v1",
        "recovery_plan_sha256": approved_plan_sha256,
        "recovery_commit": plan["recovery_commit"],
        "completed_labels": [row["label"] for row in plan["jobs"]],
        "raw_and_staging_preserved": True,
        "reran_gurobi": False,
        "receipts": receipts,
    }
    if record != expected:
        raise ValueError("recovery record differs from exact expected record")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--source-campaign-sha256",
        required=True,
        help="Out-of-band approved SHA-256 of campaign.json.",
    )
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--approved-plan-sha256")
    args = parser.parse_args(argv)
    plan, _prepared = build_recovery_plan(
        args.campaign_root,
        source_campaign_sha256=args.source_campaign_sha256,
    )
    intent = _recovery_intent(plan)
    plan_raw = _canonical(intent)
    plan_sha = hashlib.sha256(plan_raw).hexdigest()
    print(json.dumps(plan, indent=2))
    print(f"[recovery-plan-sha256] {plan_sha}")
    if args.plan_out:
        _write_new(args.plan_out, plan_raw)
    if not args.apply:
        print("[dry-run] no bundles published; no Gurobi solve invoked")
        return 0 if plan["invalid_or_missing_count"] == 0 else 2
    if not args.approved_plan_sha256:
        parser.error("--apply requires --approved-plan-sha256")
    record = apply_recovery(
        args.campaign_root,
        approved_plan_sha256=args.approved_plan_sha256,
        source_campaign_sha256=args.source_campaign_sha256,
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

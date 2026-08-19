#!/usr/bin/env python3
"""Fail-closed validation of one staged tariff-response archive tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import stat
from pathlib import Path

from launch_tariff_response_pilot import tariff_gate_spec
from reconcile_tariff_response_gate import _submitted_jobs_are_complete
from slurm_state_contract import verified_gate_evidence
from tariff_response_completion import validate_completion_identity


def sha(path):
    if path.is_symlink() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise ValueError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relocated(path, *, declared_root, staged_root):
    path = Path(path).resolve()
    try:
        relative = path.relative_to(declared_root)
    except ValueError as exc:
        raise ValueError(f"artifact escapes campaign root: {path}") from exc
    return staged_root / relative


def validate_reservations(files, reservations, selected, plan_sha):
    if (
        len(reservations) != len(selected)
        or len(files) != len(selected)
        or {Path(path).name for path in reservations}
        != {path.name for path in files}
    ):
        raise ValueError("reservation count/names differ from selected jobs")
    expected_reservations = {
        f"{job['execution_digest']}.json": job
        for job in selected.values()
    }
    reservation_jobs = set()
    for path in files:
        payload = json.loads(path.read_text())
        expected_job = expected_reservations.get(path.name)
        if (
            payload.get("schema")
            != "evsp-dr-tariff-response-reservation-v1"
            or payload.get("plan_sha256") != plan_sha
            or expected_job is None
            or payload.get("job_key") != expected_job["job_key"]
            or payload.get("execution_digest")
            != expected_job["execution_digest"]
        ):
            raise ValueError("staged reservation content is invalid")
        reservation_jobs.add(payload["job_key"])
    if reservation_jobs != set(selected):
        raise ValueError("staged reservations do not cover selected jobs")


def validate(root: Path, expected_commit: str, expected_scope: str):
    if root.is_symlink():
        raise ValueError("staged archive root is a symlink")
    root = root.resolve()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"staged archive contains a symlink: {path}")
    if expected_scope not in {"main", "k40-preparation"}:
        raise ValueError("unknown archive scope")
    plan_raw = (root / "approved-plan.json").read_bytes()
    plan = json.loads(plan_raw)
    manifest = json.loads((root / "campaign.json").read_text())
    plan_sha = hashlib.sha256(plan_raw).hexdigest()
    scope = (
        "k40_preparation_only"
        if expected_scope == "k40-preparation"
        else "main_k5_k8_pilot"
    )
    if (
        plan["checkout_identity"]["commit"] != expected_commit
        or manifest.get("approval_sha256") != plan_sha
        or manifest.get("submission_scope") != scope
        or manifest.get("submitted") is not True
    ):
        raise ValueError("campaign approval/commit/scope is invalid")
    verified_gate_evidence(
        manifest,
        tariff_gate_spec(
            plan, plan_sha, str(manifest.get("gate_job_id") or "")
        ),
    )
    selected = {
        job["job_key"]: job for job in plan["jobs"]
        if bool(job["separate_k40_gate"])
        == (expected_scope == "k40-preparation")
    }
    submitted_rows = manifest.get("submitted_jobs") or []
    submitted = {
        item["job_key"]: item for item in submitted_rows
    }
    if (
        len(submitted_rows) != len(selected)
        or len(submitted) != len(submitted_rows)
        or len({
            str(item.get("job_id")) for item in submitted_rows
        }) != len(submitted_rows)
        or any(
            not str(item.get("job_id") or "").isdigit()
            for item in submitted_rows
        )
    ):
        raise ValueError("submitted job rows are duplicated or malformed")
    if set(submitted) != set(selected):
        raise ValueError("submitted job set is incomplete")
    if not _submitted_jobs_are_complete(plan, manifest):
        raise ValueError("submitted job scheduler receipts are unverified")
    declared_root = Path(
        plan["k40_campaign_root"]
        if expected_scope == "k40-preparation"
        else plan["campaign_root"]
    ).resolve()

    reservations = manifest.get("reservations") or []
    staged_reservations = root / "input/reservations"
    transaction_path = staged_reservations / "transaction.json"
    transaction = json.loads(transaction_path.read_text())
    if (
        transaction.get("schema")
        != "evsp-dr-tariff-response-reservation-transaction-v1"
        or transaction.get("plan_sha256") != plan_sha
        or transaction.get("campaign") != plan["campaign"]
        or transaction.get("jobs") != [{
            "job_key": job["job_key"],
            "execution_digest": job["execution_digest"],
        } for job in sorted(
            selected.values(), key=lambda item: item["job_key"]
        )]
    ):
        raise ValueError("staged reservation transaction is invalid")
    files = sorted(
        path for path in staged_reservations.glob("*.json")
        if path.name != "transaction.json"
    )
    validate_reservations(files, reservations, selected, plan_sha)

    staged_tariff_manifest = root / "input/tariffs/tariff_manifest.csv"
    if sha(staged_tariff_manifest) != plan["tariff_manifest_sha256"]:
        raise ValueError("staged tariff manifest hash mismatch")
    staged_master = root / "input/source/Par_VehicleDetails_Updated.csv"
    if sha(staged_master) != plan["giro_master"]["sha256"]:
        raise ValueError("staged GIRO master hash mismatch")
    staged_duties = root / "input/source/giro40_duty_manifest.csv"
    if sha(staged_duties) != plan["giro40_duty_manifest"]["sha256"]:
        raise ValueError("staged GIRO40 duty manifest hash mismatch")
    staged_frozen = root / "input/source/frozen_input_manifest.csv"
    if sha(staged_frozen) != plan["frozen_input_manifest"]["sha256"]:
        raise ValueError("staged frozen input manifest hash mismatch")

    for job in selected.values():
        instance = relocated(
            job["instance"]["path"],
            declared_root=declared_root,
            staged_root=root,
        )
        if sha(instance) != job["instance"]["sha256"]:
            raise ValueError(f"instance hash mismatch: {job['job_key']}")
        if job.get("tariff_sha256"):
            tariff = root / "input/tariffs" / Path(
                job["tariff_relative_path"]
            ).name
            if sha(tariff) != job["tariff_sha256"]:
                raise ValueError(f"tariff hash mismatch: {job['job_key']}")
        output = relocated(
            job["output"],
            declared_root=declared_root,
            staged_root=root,
        )
        completion_path = Path(str(output) + ".worker-completion.json")
        completion = json.loads(completion_path.read_text())
        hashes = validate_completion_identity(
            completion, job, plan_sha
        )
        mapped = {}
        for declared, digest in hashes.items():
            staged = relocated(
                declared,
                declared_root=declared_root,
                staged_root=root,
            )
            if not staged.is_file() or sha(staged) != digest:
                raise ValueError(f"worker artifact changed: {job['job_key']}")
            mapped[str(staged)] = digest
        required = set()
        if output.is_dir():
            required.update(
                path for path in output.rglob("*") if path.is_file()
            )
        elif output.is_file():
            required.add(output)
        if job["phase"] == "CG":
            required.update([
                Path(str(output) + ".columns.jsonl"),
                Path(str(output) + ".iters.csv"),
                relocated(
                    job["phase_telemetry"],
                    declared_root=declared_root,
                    staged_root=root,
                ),
            ])
            status = json.loads(output.read_text())
            if status.get("stop_reason") in {None, "resume_starting"}:
                raise ValueError(f"CG status is live: {job['job_key']}")
        elif job["phase"] == "MIP":
            progress = relocated(
                job["progress_dir"],
                declared_root=declared_root,
                staged_root=root,
            )
            required.update(
                path for path in progress.rglob("*") if path.is_file()
            )
        if set(mapped) != {str(path) for path in required}:
            raise ValueError(f"completion artifact set incomplete: {job['job_key']}")

    if expected_scope == "main":
        evidence = root / "evidence/normalized"
        experiment = root / "evidence/experiment-manifest.json"
        provenance = json.loads((evidence / "provenance.json").read_text())
        output_hashes = provenance.get("output_sha256")
        campaign_provenance = provenance.get("campaign_provenance") or {}
        if (
            not isinstance(output_hashes, dict)
            or not output_hashes
            or sha(experiment)
            != provenance.get("experiment_manifest_sha256")
            or campaign_provenance.get("approved_plan_sha256") != plan_sha
            or campaign_provenance.get("git_commit") != expected_commit
            or campaign_provenance.get("approved_environment")
            != plan.get("environment_identity")
        ):
            raise ValueError("normalized evidence provenance is invalid")
        for name, digest in output_hashes.items():
            path = evidence / name
            if not path.is_file() or sha(path) != digest:
                raise ValueError(f"normalized evidence changed: {name}")
        normalized_files = {
            path.name for path in evidence.iterdir() if path.is_file()
        }
        if normalized_files != set(output_hashes) | {"provenance.json"}:
            raise ValueError("normalized evidence contains unindexed files")
        inventory = evidence / "artifact_inventory.csv"
        with inventory.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("artifact inventory is empty")
        for row in rows:
            path = Path(row["path"])
            if not (declared_root == path or declared_root in path.parents):
                raise ValueError("artifact inventory escapes staged campaign")
            path = relocated(
                path, declared_root=declared_root, staged_root=root
            )
            if not path.is_file() or sha(path) != row["sha256"]:
                raise ValueError("artifact inventory hash mismatch")
    return {
        "validated": True,
        "scope": expected_scope,
        "jobs": len(selected),
        "plan_sha256": plan_sha,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--scope", choices=("main", "k40-preparation"), required=True
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
        parser.error("--expected-commit must be exact")
    print(json.dumps(validate(
        args.campaign_root, args.expected_commit, args.scope
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

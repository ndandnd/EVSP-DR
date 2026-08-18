#!/usr/bin/env python3
"""Fail-closed validation of one staged tariff-response archive tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relocated(path, *, declared_root, staged_root):
    path = Path(path).resolve()
    try:
        relative = path.relative_to(declared_root)
    except ValueError as exc:
        raise ValueError(f"artifact escapes campaign root: {path}") from exc
    return staged_root / relative


def validate(root: Path, expected_commit: str, expected_scope: str):
    root = root.resolve()
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
        or manifest.get("gate_state")
        not in {"released", "released_reconciled"}
    ):
        raise ValueError("campaign approval/commit/scope is invalid")
    selected = {
        job["job_key"]: job for job in plan["jobs"]
        if bool(job["separate_k40_gate"])
        == (expected_scope == "k40-preparation")
    }
    submitted = {
        item["job_key"]: item for item in manifest.get("submitted_jobs") or []
    }
    if set(submitted) != set(selected):
        raise ValueError("submitted job set is incomplete")
    declared_root = Path(
        plan["k40_campaign_root"]
        if expected_scope == "k40-preparation"
        else plan["campaign_root"]
    ).resolve()

    reservations = manifest.get("reservations") or []
    staged_reservations = root / "input/reservations"
    files = sorted(staged_reservations.glob("*.json"))
    if len(reservations) != len(selected) or len(files) != len(selected):
        raise ValueError("reservation count differs from selected jobs")
    reservation_jobs = set()
    for path in files:
        payload = json.loads(path.read_text())
        if (
            payload.get("schema")
            != "evsp-dr-tariff-response-reservation-v1"
            or payload.get("plan_sha256") != plan_sha
            or payload.get("job_key") not in selected
        ):
            raise ValueError("staged reservation content is invalid")
        reservation_jobs.add(payload["job_key"])
    if reservation_jobs != set(selected):
        raise ValueError("staged reservations do not cover selected jobs")

    staged_tariff_manifest = root / "input/tariffs/tariff_manifest.csv"
    if sha(staged_tariff_manifest) != plan["tariff_manifest_sha256"]:
        raise ValueError("staged tariff manifest hash mismatch")

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
        hashes = completion.get("artifact_sha256")
        if (
            completion.get("schema")
            != "evsp-dr-tariff-response-worker-completion-v1"
            or completion.get("phase") != job["phase"]
            or completion.get("plan_sha256") != plan_sha
            or completion.get("instance_sha256")
            != job["instance"]["sha256"]
            or completion.get("tariff_sha256")
            != job.get("tariff_sha256")
            or not isinstance(hashes, dict)
            or not hashes
        ):
            raise ValueError(f"completion mismatch: {job['job_key']}")
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
        required = [output] if output.is_file() else []
        if job["phase"] == "CG":
            required.extend([
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
            required.append(
                relocated(
                    job["progress_dir"],
                    declared_root=declared_root,
                    staged_root=root,
                ) / "final.json"
            )
        if any(str(path) not in mapped for path in required):
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
        inventory = evidence / "artifact_inventory.csv"
        with inventory.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("artifact inventory is empty")
        for row in rows:
            path = Path(row["path"])
            if declared_root == path or declared_root in path.parents:
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

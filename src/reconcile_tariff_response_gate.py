#!/usr/bin/env python3
"""Reconcile an ambiguous tariff-response gate release without resubmission."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from launch_tariff_response_pilot import (
    TARIFF_CHILD_ROLE,
    TARIFF_GATE_ROLE,
    _tariff_campaign_lock,
    _write_manifest,
    tariff_child_spec,
    tariff_gate_spec,
)
from slurm_state_contract import (
    SlurmContractError,
    TERMINAL_STATES,
    discover_live_job_by_identity,
    release_with_postcondition,
    resolve_exact_job,
    verified_dependency_evidence,
    verify_dependency_receipt,
)


def _selected_job_keys(plan, manifest):
    scope = manifest.get("submission_scope")
    if scope == "main_k5_k8_pilot":
        k40 = False
    elif scope == "k40_preparation_only":
        k40 = True
    else:
        raise ValueError("campaign submission scope is invalid")
    return {
        job["job_key"] for job in plan["jobs"]
        if bool(job["separate_k40_gate"]) == k40
    }


def _selected_jobs(plan, manifest):
    keys = _selected_job_keys(plan, manifest)
    return [job for job in plan["jobs"] if job["job_key"] in keys]


def _submitted_jobs_are_complete(plan, manifest):
    rows = manifest.get("submitted_jobs") or []
    expected_jobs = _selected_jobs(plan, manifest)
    expected = {job["job_key"] for job in expected_jobs}
    keys = [str(row.get("job_key") or "") for row in rows]
    ids = [str(row.get("job_id") or "") for row in rows]
    if not (
        len(rows) == len(expected)
        and set(keys) == expected
        and len(set(keys)) == len(keys)
        and all(job_id.isdigit() for job_id in ids)
        and len(set(ids)) == len(ids)
    ):
        return False
    by_key = {row["job_key"]: row for row in rows}
    gate = str(manifest.get("gate_job_id") or "")
    try:
        for job in expected_jobs:
            row = by_key[job["job_key"]]
            dependencies = [gate]
            if job.get("dependency_key"):
                dependencies.append(
                    str(by_key[job["dependency_key"]]["job_id"])
                )
            dependency = "afterok:" + ":".join(dependencies)
            spec = tariff_child_spec(
                plan, job, dependency, row["job_id"]
            )
            if any(
                str(row.get(field) or "") != str(spec[field])
                for field in (
                    "job_id", "user", "job_name", "partition",
                    "comment", "dependency", "role",
                )
            ):
                return False
            verified_dependency_evidence(
                row.get("submission_receipt"), spec
            )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _record_terminal_failure(manifest, gate, observation, message):
    manifest["submitted"] = False
    manifest["gate_state"] = "terminal_failed"
    manifest["gate_terminal_failure"] = {
        "verified": True,
        "role": TARIFF_GATE_ROLE,
        "job_id": str(gate),
        "observation": observation,
        "state": observation.get("state"),
        "exit_code": observation.get("exit_code"),
        "source": observation.get("source"),
        "message": message,
    }


def _record_unverified(manifest, state, exc):
    manifest["submitted"] = False
    manifest["gate_state"] = state
    manifest["gate_reconciliation_error"] = {
        "message": str(exc),
        "observation": getattr(exc, "observation", None),
        "diagnostics": getattr(exc, "diagnostics", []),
    }


def _legacy_unverified(manifest, message):
    prior = manifest.get("gate_state")
    manifest["legacy_gate_state"] = prior
    manifest["gate_state"] = "legacy_unverified"
    manifest["submitted"] = False
    manifest["gate_release_verification"] = {
        "verified": False,
        "reason": message,
        "legacy_state": prior,
    }


def _resolve_bounded(spec, runner, sleeper, attempts=5):
    diagnostics = []
    last_error = None
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            sleeper(1.0)
        try:
            return resolve_exact_job(spec, runner=runner)
        except SlurmContractError as exc:
            if exc.observation is not None and not exc.retriable:
                raise
            last_error = exc
            diagnostics.append({
                "attempt": attempt,
                "message": str(exc),
                "diagnostics": exc.diagnostics,
            })
    raise SlurmContractError(
        "tariff gate state remained unobservable",
        observation=(
            last_error.observation if last_error is not None else None
        ),
        diagnostics=diagnostics,
    )


def _recover_child_intents(
    plan, manifest, manifest_path, runner, sleeper,
):
    intents = dict(manifest.get("job_submission_intents") or {})
    if not intents:
        return
    rows = manifest.setdefault("submitted_jobs", [])
    by_key = {row["job_key"]: row for row in rows}
    selected = _selected_jobs(plan, manifest)
    selected_by_key = {job["job_key"]: job for job in selected}
    unknown = set(intents) - set(selected_by_key)
    if unknown:
        raise ValueError("campaign contains unknown child submission intents")
    gate = str(manifest.get("gate_job_id") or "")
    for job in selected:
        key = job["job_key"]
        if key not in intents:
            continue
        dependencies = [gate]
        if job.get("dependency_key"):
            dependency_row = by_key.get(job["dependency_key"])
            if dependency_row is None:
                raise ValueError(
                    "child intent dependency has no exact submitted receipt"
                )
            dependencies.append(str(dependency_row["job_id"]))
        dependency = "afterok:" + ":".join(dependencies)
        expected_intent = tariff_child_spec(
            plan, job, dependency
        )
        intent = intents[key]
        if any(
            str(intent.get(field) or "")
            != str(expected_intent.get(field) or "")
            for field in (
                "user", "job_name", "partition", "comment",
                "role", "dependency",
            )
        ):
            raise ValueError("child submission intent identity mismatch")
        discovered = None
        for attempt in range(1, 6):
            if attempt > 1:
                sleeper(1.0)
            discovered = discover_live_job_by_identity(
                expected_intent, runner=runner
            )
            if discovered is not None:
                break
        if discovered is None:
            raise RuntimeError(
                "child submission intent remains ambiguous; replacement "
                "is forbidden"
            )
        job_id = str(discovered["job_id"])
        spec = tariff_child_spec(
            plan, job, dependency, job_id
        )
        receipt = verify_dependency_receipt(
            spec, runner=runner, sleeper=sleeper
        )
        row = {
            "job_key": key,
            **{
                field: spec[field] for field in (
                    "job_id", "user", "job_name", "partition",
                    "comment", "dependency", "role",
                )
            },
            "submission_receipt": receipt,
        }
        rows.append(row)
        by_key[key] = row
        intents.pop(key)
        manifest["job_submission_intents"] = dict(intents)
        _write_manifest(manifest_path, manifest)


def _reconcile_locked(
    root: Path,
    expected_plan_sha256: str,
    *,
    runner=None,
    sleeper=None,
):
    root = root.resolve()
    plan_raw = (root / "approved-plan.json").read_bytes()
    observed = hashlib.sha256(plan_raw).hexdigest()
    if observed != expected_plan_sha256:
        raise ValueError("approved plan SHA-256 mismatch")
    manifest_path = root / "campaign.json"
    plan = json.loads(plan_raw)
    manifest = json.loads(manifest_path.read_text())
    runner = subprocess.run if runner is None else runner
    sleeper = time.sleep if sleeper is None else sleeper
    if manifest.get("approval_sha256") != observed:
        raise ValueError("campaign approval hash differs from approved plan")

    scheduler = plan.get("scheduler_identity") or {}
    recorded_spec = manifest.get("gate_spec")
    if not scheduler.get("user"):
        _legacy_unverified(
            manifest,
            "legacy campaign lacks the immutable scheduler user required "
            "for exact reconciliation",
        )
        _write_manifest(manifest_path, manifest)
        raise ValueError("legacy tariff gate evidence is labeled unverified")

    gate = str(
        manifest.get("gate_job_id")
        or (
            recorded_spec.get("job_id")
            if isinstance(recorded_spec, dict) else ""
        )
        or ""
    )
    if gate.isdigit() and not isinstance(recorded_spec, dict):
        _legacy_unverified(
            manifest,
            "legacy campaign has a gate ID but no immutable gate "
            "specification",
        )
        _write_manifest(manifest_path, manifest)
        raise ValueError("legacy tariff gate evidence is labeled unverified")
    if not gate.isdigit():
        intent = manifest.get("gate_submission_intent")
        expected_intent = tariff_gate_spec(plan, observed)
        if not isinstance(intent, dict) or any(
            str(intent.get(field) or "")
            != str(expected_intent.get(field) or "")
            for field in (
                "user", "job_name", "partition", "comment", "role",
            )
        ):
            _record_unverified(
                manifest,
                "ambiguous_gate_receipt",
                SlurmContractError(
                    "unrecorded gate lacks an exact submission intent"
                ),
            )
            _write_manifest(manifest_path, manifest)
            raise ValueError("unrecorded tariff gate identity is ambiguous")
        discovered = None
        discovery_errors = []
        for attempt in range(1, 6):
            if attempt > 1:
                sleeper(1.0)
            try:
                discovered = discover_live_job_by_identity(
                    expected_intent, runner=runner
                )
            except SlurmContractError as exc:
                discovery_errors.append({
                    "attempt": attempt,
                    "message": str(exc),
                    "diagnostics": exc.diagnostics,
                })
                continue
            if discovered is not None:
                break
        if discovered is None:
            error = SlurmContractError(
                "prior gate submission remains ambiguous; replacement "
                "submission is forbidden",
                diagnostics=discovery_errors,
            )
            _record_unverified(manifest, "ambiguous_gate_receipt", error)
            _write_manifest(manifest_path, manifest)
            raise RuntimeError(str(error))
        gate = str(discovered["job_id"])
        manifest["gate_job_id"] = gate
        recorded_spec = tariff_gate_spec(plan, observed, gate)
        manifest["gate_spec"] = recorded_spec
        manifest.pop("gate_submission_intent", None)
        _write_manifest(manifest_path, manifest)

    expected_spec = tariff_gate_spec(plan, observed, gate)
    if any(
        str(recorded_spec.get(field) or "")
        != str(expected_spec.get(field) or "")
        for field in (
            "job_id", "user", "job_name", "partition", "comment", "role",
        )
    ):
        error = SlurmContractError(
            "recorded tariff gate specification differs from the plan"
        )
        _record_unverified(manifest, "gate_identity_mismatch", error)
        _write_manifest(manifest_path, manifest)
        raise ValueError(str(error))

    try:
        _recover_child_intents(
            plan, manifest, manifest_path, runner, sleeper
        )
    except (RuntimeError, ValueError, SlurmContractError) as exc:
        manifest["submitted"] = False
        manifest["gate_state"] = "child_submission_reconciliation_failed"
        manifest["child_reconciliation_error"] = {
            "message": str(exc),
            "observation": getattr(exc, "observation", None),
            "diagnostics": getattr(exc, "diagnostics", []),
        }
        _write_manifest(manifest_path, manifest)
        raise RuntimeError(str(exc)) from exc

    try:
        current = _resolve_bounded(
            expected_spec, runner, sleeper
        )
    except SlurmContractError as exc:
        if (
            isinstance(exc.observation, dict)
            and exc.observation.get("state") in TERMINAL_STATES
        ):
            _record_terminal_failure(
                manifest, gate, exc.observation, str(exc)
            )
        else:
            _record_unverified(manifest, "reconciliation_unverified", exc)
        _write_manifest(manifest_path, manifest)
        raise RuntimeError(str(exc)) from exc

    if current["state"] in TERMINAL_STATES:
        if (
            current["state"] != "COMPLETED"
            or current.get("exit_code") != "0:0"
        ):
            _record_terminal_failure(
                manifest,
                gate,
                current,
                "tariff gate reached a terminal non-success state",
            )
            _write_manifest(manifest_path, manifest)
            raise ValueError(
                "tariff gate is terminal without COMPLETED/0:0"
            )

    if not _submitted_jobs_are_complete(plan, manifest):
        manifest["submitted"] = False
        manifest["gate_state"] = "incomplete_submission"
        manifest["gate_reconciliation_observation"] = current
        _write_manifest(manifest_path, manifest)
        raise ValueError(
            "campaign job set is incomplete; gate mutation is refused"
        )

    if current["state"] in TERMINAL_STATES:
        release_verification = {
            "verified": True,
            "role": TARIFF_GATE_ROLE,
            "job_id": gate,
            "command_attempts": 0,
            "observation": current,
            "command_diagnostics": [],
        }
    else:
        manifest["gate_state"] = "release_reconciling"
        manifest["submitted"] = False
        _write_manifest(manifest_path, manifest)
        try:
            release_verification = release_with_postcondition(
                expected_spec,
                runner=runner,
                sleeper=sleeper,
            )
        except SlurmContractError as exc:
            if (
                isinstance(exc.observation, dict)
                and exc.observation.get("state") in TERMINAL_STATES
            ):
                _record_terminal_failure(
                    manifest, gate, exc.observation, str(exc)
                )
            else:
                _record_unverified(
                    manifest, "held_release_failed", exc
                )
            _write_manifest(manifest_path, manifest)
            raise RuntimeError(str(exc)) from exc

    manifest["gate_release_verification"] = release_verification
    _write_manifest(manifest_path, manifest)
    final_observation = release_verification["observation"]
    if (
        final_observation.get("state") == "COMPLETED"
        and final_observation.get("exit_code") == "0:0"
    ):
        manifest["gate_terminal_verification"] = {
            "verified": True,
            "role": TARIFF_GATE_ROLE,
            "job_id": gate,
            "observation": final_observation,
        }
        manifest["gate_state"] = "completed_verified"
    else:
        manifest["gate_state"] = "released_verified"
    manifest["submitted"] = True
    manifest["gate_reconciliation"] = {
        "verified": True,
        "role": TARIFF_GATE_ROLE,
        "job_id": gate,
        "observation": final_observation,
    }
    _write_manifest(manifest_path, manifest)
    return manifest


def reconcile(
    root: Path,
    expected_plan_sha256: str,
    *,
    runner=None,
    sleeper=None,
):
    root = root.resolve()
    with _tariff_campaign_lock(root):
        return _reconcile_locked(
            root,
            expected_plan_sha256,
            runner=runner,
            sleeper=sleeper,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--approved-plan-sha256", required=True)
    args = parser.parse_args(argv)
    reconcile(args.campaign_root, args.approved_plan_sha256)
    print("GATE RECONCILED WITH EXACT SCHEDULER EVIDENCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

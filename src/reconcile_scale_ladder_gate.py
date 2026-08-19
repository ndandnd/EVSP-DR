#!/usr/bin/env python3
"""Reconcile an ambiguous ladder gate release from Slurm accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from launch_scale_ladder import (
    PROBE_PARTITIONS,
    PROBE_RETRYABLE_STATES,
    _probes_compatible,
    _probes_waiting,
    _probe_spec,
    _replace_json,
    _sbatch,
    _submit_probe,
    _submit_array,
    _wait_for_probes,
)

SLURM_QUERY_TIMEOUT_S = 10.0


def _approved_tool_path(plan, name):
    spec = plan.get(name) or {}
    path = Path(str(spec.get("path") or ""))
    if (
        spec.get("available") is not True
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest()
        != spec.get("sha256")
    ):
        raise ValueError(f"approved {name} unavailable/changed")
    return path


def _normalized_state(value):
    words = str(value or "").strip().split()
    return words[0].split("+", 1)[0].upper() if words else ""


def _gate_fingerprint(expected_plan_sha, gate):
    return {
        "job_id": str(gate),
        "job_name": f"LDG{expected_plan_sha[:5]}",
        "partition": "default_partition",
        "comment": f"SLADG:{expected_plan_sha[:20]}",
    }


def _gate_fingerprint_errors(expected, row):
    return [
        {
            "field": field,
            "expected": expected[field],
            "observed": str(row.get(field) or ""),
        }
        for field in ("job_id", "job_name", "partition", "comment")
        if str(row.get(field) or "") != expected[field]
    ]


def _bounded_query(runner, command):
    try:
        return runner(
            command, text=True, capture_output=True, check=False,
            timeout=SLURM_QUERY_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Slurm query timed out after {exc.timeout} seconds"
        ) from exc


def _resolve_gate_state(
    plan, gate, expected_plan_sha, *, runner=None,
):
    """Resolve the exact gate, preferring live controller state to sacct."""
    runner = subprocess.run if runner is None else runner
    gate = str(gate)
    if not gate.isdigit():
        raise ValueError("gate job ID is invalid")
    expected = _gate_fingerprint(expected_plan_sha, gate)
    user = str((plan.get("runtime_environment") or {}).get("USER") or "")
    if not user:
        raise ValueError("approved runtime user is missing")

    squeue_path = _approved_tool_path(plan, "squeue")
    listed = _bounded_query(runner, [
        str(squeue_path), "-h", "-u", user,
        "-o", "%i|%j|%T|%P|%R|%k",
    ])
    if listed.returncode != 0:
        raise RuntimeError("cannot query live gate state with squeue")
    live_rows = []
    for line in listed.stdout.splitlines():
        fields = [field.strip() for field in line.split("|", 5)]
        if len(fields) == 6 and fields[0] == gate:
            live_rows.append({
                "job_id": fields[0], "job_name": fields[1],
                "state": _normalized_state(fields[2]),
                "partition": fields[3], "reason": fields[4],
                "comment": fields[5],
            })
    if len(live_rows) > 1:
        raise ValueError("multiple live rows match the gate job ID")
    if live_rows:
        errors = _gate_fingerprint_errors(expected, live_rows[0])
        if errors:
            raise ValueError(f"live gate fingerprint mismatch: {errors}")
        return {**live_rows[0], "source": "squeue", "live": True}

    scontrol_path = _approved_tool_path(plan, "scontrol")
    shown = _bounded_query(
        runner, [str(scontrol_path), "show", "job", gate, "-o"]
    )
    if shown.returncode == 0 and shown.stdout.strip():
        values = {}
        for token in shown.stdout.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            values[key] = value
        controller_row = {
            "job_id": values.get("JobId", ""),
            "job_name": values.get("JobName", ""),
            "state": _normalized_state(values.get("JobState")),
            "partition": values.get("Partition", ""),
            "reason": values.get("Reason", ""),
            "comment": values.get("Comment", ""),
            "exit_code": values.get("ExitCode"),
        }
        errors = _gate_fingerprint_errors(expected, controller_row)
        if errors:
            raise ValueError(
                f"controller gate fingerprint mismatch: {errors}"
            )
        return {**controller_row, "source": "scontrol", "live": True}
    absent_message = (shown.stderr or shown.stdout).lower()
    if shown.returncode != 0 and "invalid job id" not in absent_message:
        raise RuntimeError("cannot prove the gate absent from scontrol")

    sacct_path = _approved_tool_path(plan, "sacct")
    completed = _bounded_query(runner, [
        str(sacct_path), "-X", "-n", "-P", "-j", gate,
        "--format=JobIDRaw,JobName%64,State,Partition%64,"
        "Comment%256,ExitCode",
    ])
    if completed.returncode != 0:
        raise RuntimeError("cannot query gate accounting")
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split("|", 5)]
        if len(fields) == 6 and fields[0] == gate:
            rows.append({
                "job_id": fields[0], "job_name": fields[1],
                "state": _normalized_state(fields[2]),
                "partition": fields[3], "comment": fields[4],
                "exit_code": fields[5],
            })
    if len(rows) != 1:
        raise ValueError("gate accounting has no unique exact job row")
    errors = _gate_fingerprint_errors(expected, rows[0])
    if errors:
        raise ValueError(f"accounting gate fingerprint mismatch: {errors}")
    if rows[0]["state"] == "COMPLETED" and rows[0]["exit_code"] != "0:0":
        raise ValueError("completed gate accounting has nonzero exit code")
    return {**rows[0], "source": "sacct", "live": False}


def _require_gate_held(plan, gate, expected_plan_sha):
    observation = _resolve_gate_state(plan, gate, expected_plan_sha)
    if (
        observation.get("live") is not True
        or observation.get("state") != "PENDING"
        or observation.get("reason") != "JobHeldUser"
    ):
        raise ValueError("gate is not proven held by the user")
    return _approved_tool_path(plan, "scontrol")


def _discover_probe_job(plan, expected_plan_sha, partition, spec):
    """Recover a probe accepted by Slurm before its job ID was recorded."""
    recorded = str(spec.get("job_id") or "")
    if recorded.isdigit():
        return recorded
    output = Path(str(spec.get("output") or ""))
    if output.is_file():
        payload = json.loads(output.read_text())
        recovered = str(payload.get("slurm_job_id") or "")
        if (
            not recovered.isdigit()
            or payload.get("plan_sha256") != expected_plan_sha
            or payload.get("probe_id") != spec.get("probe_id")
            or payload.get("probe_attempt") != spec.get("attempt")
            or payload.get("slurm_partition") != partition
        ):
            raise ValueError("unrecorded probe artifact identity mismatch")
        return recovered
    squeue = plan.get("squeue") or {}
    squeue_path = Path(str(squeue.get("path") or ""))
    if (
        squeue.get("available") is not True
        or not squeue_path.is_file()
        or hashlib.sha256(squeue_path.read_bytes()).hexdigest()
        != squeue.get("sha256")
    ):
        raise ValueError("approved squeue unavailable/changed")
    listed = _bounded_query(
        subprocess.run,
        [str(squeue_path), "-h", "-o", "%i|%k"],
    )
    if listed.returncode != 0:
        raise RuntimeError("cannot query squeue for unrecorded probe")
    matches = {
        fields[0]
        for line in listed.stdout.splitlines()
        if len(fields := line.split("|", 1)) == 2
        and fields[1] == spec.get("comment")
        and fields[0].isdigit()
    }
    if len(matches) > 1:
        raise ValueError("multiple probes share one attempt comment")
    return next(iter(matches), None)


def _hard_probe_mismatch(result):
    path = Path(str(result.get("output") or ""))
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("compatible") is False
        and bool(payload.get("differences"))
    )


def reconcile(
    root, expected_plan_sha, *,
    release_held_gate=False,
    resume_missing_arrays=False,
    retry_failed_probes=False,
):
    root = Path(root).resolve()
    plan_raw = (root / "approved-plan.json").read_bytes()
    if hashlib.sha256(plan_raw).hexdigest() != expected_plan_sha:
        raise ValueError("approved plan hash mismatch")
    manifest_path = root / "campaign.json"
    plan = json.loads(plan_raw)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("gate_state") in {
        "creating", "held", "held_after_partial_submission",
        "held_probe_failure", "held_probe_waiting",
    }:
        squeue = plan.get("squeue") or {}
        squeue_path = Path(str(squeue.get("path") or ""))
        if (
            squeue.get("available") is not True
            or not squeue_path.is_file()
            or hashlib.sha256(squeue_path.read_bytes()).hexdigest()
            != squeue.get("sha256")
        ):
            raise ValueError("approved squeue unavailable/changed")
        listed = _bounded_query(
            subprocess.run,
            [str(squeue_path), "-h", "-o", "%F|%k"],
        )
        if listed.returncode != 0:
            raise RuntimeError("cannot query squeue")
        prefix = f"SLAD:{expected_plan_sha[:20]}:"
        gate_comment = f"SLADG:{expected_plan_sha[:20]}"
        discovered = {}
        discovered_gate = None
        for line in listed.stdout.splitlines():
            fields = line.split("|", 1)
            if len(fields) != 2:
                continue
            if fields[1] == gate_comment:
                if discovered_gate and discovered_gate != fields[0]:
                    raise ValueError("multiple gates share one plan comment")
                discovered_gate = fields[0]
                continue
            if not fields[1].startswith(prefix):
                continue
            group = fields[1][len(prefix):]
            if group in discovered and discovered[group] != fields[0]:
                raise ValueError("multiple arrays share one plan/group comment")
            discovered[group] = fields[0]
        recorded = manifest.get("submitted_arrays") or {}
        for group, job_id in discovered.items():
            if group in recorded and str(recorded[group]) != str(job_id):
                raise ValueError("recorded/discovered array ID conflict")
        combined = {**discovered, **recorded}
        if any(not str(value).isdigit() for value in combined.values()):
            raise ValueError("reconstructed array ID is invalid")
        manifest["submitted_arrays"] = combined
        if discovered_gate:
            recorded_gate = manifest.get("gate_job_id")
            if recorded_gate and str(recorded_gate) != str(discovered_gate):
                raise ValueError("recorded/discovered gate ID conflict")
            manifest["gate_job_id"] = discovered_gate
        gate_for_resume = manifest.get("gate_job_id")
        required_groups = (
            "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
            "MIP_RAW", "MIP_KNOWN",
        )
        missing_groups = [
            group for group in required_groups if group not in combined
        ]
        if missing_groups and resume_missing_arrays:
            if not str(gate_for_resume or "").isdigit():
                gate_for_resume = _sbatch(plan, [
                    "--hold", "--partition=default_partition",
                    "--time=00:05:00",
                    f"--job-name=LDG{expected_plan_sha[:5]}",
                    f"--comment=SLADG:{expected_plan_sha[:20]}",
                    f"--output={root / 'logs'}/gate_%j.out",
                    f"--error={root / 'logs'}/gate_%j.err",
                    "--export=NONE", "--wrap=/bin/true",
                ])
                manifest["gate_job_id"] = str(gate_for_resume)
                manifest["gate_state"] = "held"
                _replace_json(manifest_path, manifest)
            if not str(gate_for_resume or "").isdigit():
                raise ValueError("cannot resume arrays without a proven gate")
            _require_gate_held(
                plan, str(gate_for_resume), expected_plan_sha
            )
            logs = root / "logs"
            for group in required_groups:
                if group in combined:
                    continue
                dependency = None
                if group == "CG":
                    dependency = f"afterok:{combined['PREFLIGHT']}"
                elif group == "CG_SENSITIVITY":
                    dependency = f"afterok:{combined['PREFLIGHT']}"
                elif group == "MIP_RAW":
                    dependency = f"aftercorr:{combined['CG']}"
                elif group == "MIP_KNOWN":
                    dependency = (
                        f"aftercorr:{combined['CG']}:{combined['SEED']}"
                    )
                combined[group] = _submit_array(
                    plan,
                    root / "approved-plan.json",
                    expected_plan_sha,
                    group,
                    str(gate_for_resume),
                    logs,
                    dependency=dependency,
                )
                manifest["submitted_arrays"] = dict(combined)
                manifest["gate_job_id"] = str(gate_for_resume)
                manifest["gate_state"] = "held"
                _replace_json(manifest_path, manifest)
            missing_groups = []
        manifest["submitted_arrays"] = combined
        manifest["gate_state"] = (
            "held" if not missing_groups
            else "held_after_partial_submission"
        )
        _replace_json(manifest_path, manifest)
        if missing_groups:
            raise ValueError(
                "accepted arrays reconstructed; rerun with "
                "--resume-missing-arrays"
            )
    if (
        manifest.get("approval_sha256") != expected_plan_sha
        or manifest.get("gate_state")
        not in {
            "held", "release_attempting", "held_release_failed",
            "release_retry_attempting", "release_retry_requested",
            "held_probe_passed", "held_probe_waiting",
        }
        or not str(manifest.get("gate_job_id") or "").isdigit()
    ):
        raise ValueError("campaign is not in a reconcilable state")
    gate = str(manifest["gate_job_id"])
    probe_specs = dict(manifest.get("infrastructure_probes") or {})
    unknown_probe_keys = set(probe_specs) - set(PROBE_PARTITIONS)
    if unknown_probe_keys:
        raise ValueError("campaign contains unknown infrastructure probes")
    for partition, spec in probe_specs.items():
        attempt = spec.get("attempt")
        if not isinstance(attempt, int) or attempt < 1:
            raise ValueError(
                "probe manifest predates attempt-safe recovery; "
                "reconcile it only with its original reviewed commit"
            )
        expected_spec = _probe_spec(
            expected_plan_sha, partition, root, attempt
        )
        if any(
            spec.get(field) != expected_spec[field]
            for field in (
                "output", "probe_id", "partition", "attempt", "comment",
                "job_name",
            )
        ):
            raise ValueError("recorded probe specification identity mismatch")
    gate_observation = _resolve_gate_state(
        plan, gate, expected_plan_sha
    )
    gate_state = gate_observation["state"]
    needs_probe_submission = (
        set(probe_specs) != set(PROBE_PARTITIONS)
        or any(
            not str(spec.get("job_id") or "").isdigit()
            for spec in probe_specs.values()
        )
    )
    if needs_probe_submission:
        if gate_state != "PENDING":
            raise ValueError(
                "missing/unrecorded probes cannot be recovered after gate release"
            )
        _require_gate_held(plan, gate, expected_plan_sha)
    for partition in PROBE_PARTITIONS:
        spec = probe_specs.get(partition)
        if spec is None:
            spec = _probe_spec(
                expected_plan_sha, partition, root, attempt=1
            )
            probe_specs[partition] = spec
            manifest["infrastructure_probes"] = dict(probe_specs)
            manifest["probe_state"] = "submitting"
            _replace_json(manifest_path, manifest)
        if not str(spec.get("job_id") or "").isdigit():
            recovered = _discover_probe_job(
                plan, expected_plan_sha, partition, spec
            )
            if recovered is None:
                recovered = _submit_probe(
                    plan,
                    root / "approved-plan.json",
                    expected_plan_sha,
                    spec,
                    root / "logs",
                )
            spec["job_id"] = recovered
            manifest["infrastructure_probes"] = dict(probe_specs)
            manifest["probe_state"] = "running"
            _replace_json(manifest_path, manifest)
    probe_results = _wait_for_probes(
        plan, expected_plan_sha, probe_specs, timeout_s=120
    )
    manifest["probe_results"] = probe_results
    manifest["probe_state"] = "evaluated"
    _replace_json(manifest_path, manifest)
    if not _probes_compatible(probe_results) and retry_failed_probes:
        if gate_state != "PENDING":
            raise ValueError("failed probes cannot retry after gate release")
        _require_gate_held(plan, gate, expected_plan_sha)
        retry_partitions = []
        for partition, result in probe_results.items():
            if result.get("compatible") is True:
                continue
            if (
                _hard_probe_mismatch(result)
                or result.get("state_resolution") in {
                    "environment_mismatch", "artifact_identity_mismatch",
                    "identity_mismatch",
                }
            ):
                manifest["probe_state"] = "failed_gate_retained"
                manifest["gate_state"] = "held_probe_failure"
                _replace_json(manifest_path, manifest)
                raise ValueError(
                    f"non-retryable probe mismatch on {partition}; retry refused"
                )
            if result.get("state") not in PROBE_RETRYABLE_STATES:
                manifest["probe_state"] = "waiting_gate_retained"
                manifest["gate_state"] = "held_probe_waiting"
                _replace_json(manifest_path, manifest)
                raise ValueError(
                    f"probe {partition} is not terminal; retry refused"
                )
            retry_partitions.append(partition)
        history = manifest.setdefault("probe_attempt_history", {})
        for partition in retry_partitions:
            previous = dict(probe_specs[partition])
            history.setdefault(partition, []).append({
                "spec": previous,
                "result": probe_results[partition],
            })
            spec = _probe_spec(
                expected_plan_sha, partition, root,
                attempt=int(previous["attempt"]) + 1,
            )
            probe_specs[partition] = spec
            manifest["infrastructure_probes"] = dict(probe_specs)
            manifest["probe_state"] = "retry_submitting"
            _replace_json(manifest_path, manifest)
            spec["job_id"] = _submit_probe(
                plan,
                root / "approved-plan.json",
                expected_plan_sha,
                spec,
                root / "logs",
            )
            manifest["infrastructure_probes"] = dict(probe_specs)
            _replace_json(manifest_path, manifest)
        manifest["probe_state"] = "retry_running"
        _replace_json(manifest_path, manifest)
        probe_results = _wait_for_probes(
            plan, expected_plan_sha, probe_specs, timeout_s=120
        )
        manifest["probe_results"] = probe_results
    if not _probes_compatible(probe_results):
        waiting = _probes_waiting(probe_results)
        manifest["probe_state"] = (
            "waiting_gate_retained" if waiting
            else "failed_gate_retained"
        )
        manifest["gate_state"] = (
            "held_probe_waiting" if waiting
            else "held_probe_failure"
        )
        _replace_json(manifest_path, manifest)
        raise ValueError(
            "infrastructure probes are not both compatible; gate retained"
        )
    manifest["probe_state"] = "passed"
    if manifest.get("gate_state") in {
        "held_probe_failure", "held_probe_waiting",
    }:
        manifest["gate_state"] = "held"
    _replace_json(manifest_path, manifest)
    if gate_state == "PENDING" and not release_held_gate:
        manifest["gate_state"] = "held_probe_passed"
        _replace_json(manifest_path, manifest)
        return manifest
    if gate_state == "PENDING" and release_held_gate:
        scontrol_path = _require_gate_held(
            plan, gate, expected_plan_sha
        )
        manifest["gate_state"] = "release_retry_attempting"
        _replace_json(manifest_path, manifest)
        released = _bounded_query(
            subprocess.run,
            [str(scontrol_path), "release", gate],
        )
        if released.returncode != 0:
            manifest["gate_state"] = "held_release_failed"
            manifest["release_error"] = (
                released.stderr or released.stdout
            ).strip()
            _replace_json(manifest_path, manifest)
            raise RuntimeError("gate release retry failed")
        manifest["gate_state"] = "release_retry_requested"
        _replace_json(manifest_path, manifest)
        return manifest
    if gate_state != "COMPLETED":
        raise ValueError("gate is not proven completed")
    manifest["gate_state"] = "released_reconciled"
    manifest["submitted"] = True
    manifest["gate_reconciliation"] = {
        "source": gate_observation["source"],
        "gate_job_id": gate, "state": "COMPLETED",
    }
    _replace_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--approved-plan-sha256", required=True)
    parser.add_argument("--release-held-gate", action="store_true")
    parser.add_argument("--resume-missing-arrays", action="store_true")
    parser.add_argument("--retry-failed-probes", action="store_true")
    args = parser.parse_args(argv)
    payload = reconcile(
        args.campaign_root,
        args.approved_plan_sha256,
        release_held_gate=args.release_held_gate,
        resume_missing_arrays=args.resume_missing_arrays,
        retry_failed_probes=args.retry_failed_probes,
    )
    print(f"LADDER GATE STATE: {payload['gate_state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

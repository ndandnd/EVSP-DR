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
    _probe_spec,
    _replace_json,
    _sbatch,
    _submit_probe,
    _submit_array,
    _wait_for_probes,
)


def _require_gate_held(plan, gate):
    scontrol = plan.get("scontrol") or {}
    path = Path(str(scontrol.get("path") or ""))
    if (
        scontrol.get("available") is not True
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest()
        != scontrol.get("sha256")
    ):
        raise ValueError("approved scontrol unavailable/changed")
    shown = subprocess.run(
        [str(path), "show", "job", str(gate), "-o"],
        text=True, capture_output=True, check=False,
    )
    if (
        shown.returncode != 0
        or "JobState=PENDING" not in shown.stdout
        or "Reason=JobHeldUser" not in shown.stdout
    ):
        raise ValueError("gate is not proven held by the user")
    return path


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
    listed = subprocess.run(
        [str(squeue_path), "-h", "-o", "%i|%k"],
        text=True, capture_output=True, check=False,
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
    sacct = plan.get("sacct") or {}
    sacct_path = Path(str(sacct.get("path") or ""))
    if (
        sacct.get("available") is not True
        or not sacct_path.is_file()
        or hashlib.sha256(sacct_path.read_bytes()).hexdigest()
        != sacct.get("sha256")
    ):
        raise ValueError("approved sacct executable unavailable/changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("gate_state") in {
        "creating", "held", "held_after_partial_submission",
        "held_probe_failure",
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
        listed = subprocess.run(
            [str(squeue_path), "-h", "-o", "%F|%k"],
            text=True, capture_output=True, check=False,
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
            scontrol = plan.get("scontrol") or {}
            scontrol_path = Path(str(scontrol.get("path") or ""))
            if (
                scontrol.get("available") is not True
                or not scontrol_path.is_file()
                or hashlib.sha256(scontrol_path.read_bytes()).hexdigest()
                != scontrol.get("sha256")
            ):
                raise ValueError("approved scontrol unavailable/changed")
            shown = subprocess.run(
                [
                    str(scontrol_path), "show", "job",
                    str(gate_for_resume), "-o",
                ],
                text=True, capture_output=True, check=False,
            )
            if (
                shown.returncode != 0
                or "JobState=PENDING" not in shown.stdout
                or "Reason=JobHeldUser" not in shown.stdout
            ):
                raise ValueError(
                    "cannot resume arrays unless the gate is proven held"
                )
            if not str(gate_for_resume or "").isdigit():
                raise ValueError("cannot resume arrays without a proven gate")
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
            "held_probe_passed",
        }
        or not str(manifest.get("gate_job_id") or "").isdigit()
    ):
        raise ValueError("campaign is not in a reconcilable state")
    gate = str(manifest["gate_job_id"])
    completed = subprocess.run(
        [
            str(sacct_path), "-X", "-n", "-P", "-j", gate,
            "--format=JobIDRaw,State",
        ],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot query sacct")
    states = {
        fields[0]: fields[1].split()[0].split("+", 1)[0]
        for line in completed.stdout.splitlines()
        if len(fields := line.split("|")) >= 2
    }
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
            )
        ):
            raise ValueError("recorded probe specification identity mismatch")
    needs_probe_submission = (
        set(probe_specs) != set(PROBE_PARTITIONS)
        or any(
            not str(spec.get("job_id") or "").isdigit()
            for spec in probe_specs.values()
        )
    )
    if needs_probe_submission:
        if states.get(gate) != "PENDING":
            raise ValueError(
                "missing/unrecorded probes cannot be recovered after gate release"
            )
        _require_gate_held(plan, gate)
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
        if states.get(gate) != "PENDING":
            raise ValueError("failed probes cannot retry after gate release")
        _require_gate_held(plan, gate)
        retry_partitions = []
        for partition, result in probe_results.items():
            if result.get("compatible") is True:
                continue
            if _hard_probe_mismatch(result):
                manifest["probe_state"] = "failed_gate_retained"
                manifest["gate_state"] = "held_probe_failure"
                _replace_json(manifest_path, manifest)
                raise ValueError(
                    f"portable environment mismatch on {partition}; retry refused"
                )
            if result.get("state") not in PROBE_RETRYABLE_STATES:
                manifest["probe_state"] = "waiting_gate_retained"
                manifest["gate_state"] = "held_probe_failure"
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
        manifest["probe_state"] = "failed_gate_retained"
        manifest["gate_state"] = "held_probe_failure"
        _replace_json(manifest_path, manifest)
        raise ValueError(
            "infrastructure probes are not both compatible; gate retained"
        )
    manifest["probe_state"] = "passed"
    if manifest.get("gate_state") == "held_probe_failure":
        manifest["gate_state"] = "held"
    _replace_json(manifest_path, manifest)
    if states.get(gate) == "PENDING" and not release_held_gate:
        manifest["gate_state"] = "held_probe_passed"
        _replace_json(manifest_path, manifest)
        return manifest
    if states.get(gate) == "PENDING" and release_held_gate:
        scontrol_path = _require_gate_held(plan, gate)
        manifest["gate_state"] = "release_retry_attempting"
        _replace_json(manifest_path, manifest)
        released = subprocess.run(
            [str(scontrol_path), "release", gate],
            text=True, capture_output=True, check=False,
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
    if states.get(gate) != "COMPLETED":
        raise ValueError("gate is not proven completed")
    manifest["gate_state"] = "released_reconciled"
    manifest["submitted"] = True
    manifest["gate_reconciliation"] = {
        "source": "sacct", "gate_job_id": gate, "state": "COMPLETED",
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

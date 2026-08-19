#!/usr/bin/env python3
"""Reconcile manually recovered scale-ladder probes without using sacct.

The preview phase validates immutable probe artifacts, the controller audit,
and independent evidence that the probe-gated science campaign became
eligible.  The apply phase requires the exact preview hash, revalidates all
evidence, preserves the prior manifest, and atomically publishes the repair.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path


SCHEMA = "evsp-dr-scale-ladder-probe-reconciliation-v1"
AUDIT_SCHEMA = "evsp-dr-controller-probe-recovery-v1"
PROBE_SCHEMA = "evsp-dr-scale-ladder-environment-probe-v1"
COMPLETION_SCHEMA = "evsp-dr-scale-ladder-worker-completion-v1"
PARTITIONS = ("default_partition", "scaglione")
PROBE_IDS = {
    "default_partition": "default",
    "scaglione": "scaglione",
}
REQUIRED_GROUPS = {
    "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
    "MIP_RAW", "MIP_KNOWN",
}
AUDIT_KEYS = {
    "schema", "plan_sha256", "gate_job_id",
    "default_probe_job_id", "scaglione_probe_job_id",
    "default_output", "scaglione_output", "worker_sha256",
}
BLOCKED_REASONS = {
    "Dependency", "DependencyNeverSatisfied", "JobHeldUser",
}
CONTROLLER_QUERY_TIMEOUT_S = 10.0


def canonical(payload):
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lexical_absolute(path):
    """Return an absolute path without following any filesystem symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def require_regular(path, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"{label} missing: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path


def read_json_regular(path, label):
    path = require_regular(path, label)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def validate_sidecar(path, *, basename_only):
    path = require_regular(path, "artifact")
    sidecar = require_regular(Path(str(path) + ".sha256"), "checksum sidecar")
    lines = sidecar.read_text().splitlines()
    if len(lines) != 1:
        raise ValueError(f"checksum sidecar must contain exactly one line: {sidecar}")
    fields = lines[0].split()
    if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        raise ValueError(f"malformed checksum sidecar: {sidecar}")
    recorded = Path(fields[1])
    if basename_only:
        if recorded.as_posix() != path.name:
            raise ValueError(
                f"checksum sidecar must use the artifact basename: {sidecar}"
            )
    elif recorded.is_absolute():
        if recorded.resolve() != path.resolve():
            raise ValueError(f"checksum sidecar names a different artifact: {sidecar}")
    elif recorded.as_posix() != path.name:
        raise ValueError(f"checksum sidecar names a different artifact: {sidecar}")
    observed = digest_file(path)
    if fields[0] != observed:
        raise ValueError(f"checksum mismatch: {path}")
    return observed, digest_file(sidecar)


def parse_scontrol_line(line):
    fields = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in fields:
            raise ValueError(f"duplicate scontrol field: {key}")
        fields[key] = value
    return fields


def dependency_job_ids(raw):
    if not raw or raw in {"(null)", "None"} or "?" in raw:
        raise ValueError("gate dependency is absent or uses OR semantics")
    job_ids = []
    for clause in raw.split(","):
        clause = re.sub(r"\([^)]*\)$", "", clause)
        if not clause.startswith("afterok:"):
            raise ValueError("gate dependency must contain only afterok clauses")
        values = clause[len("afterok:"):].split(":")
        if not values or any(not value.isdigit() for value in values):
            raise ValueError("gate dependency contains an invalid job ID")
        job_ids.extend(values)
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("gate dependency repeats a job ID")
    return set(job_ids)


def expected_gate_identity(plan_sha, gate):
    return {
        "job_id": str(gate),
        "job_name": f"LDG{plan_sha[:5]}",
        "partition": "default_partition",
        "comment": f"SLADG:{plan_sha[:20]}",
    }


def run_controller_query(runner, command):
    try:
        return runner(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=CONTROLLER_QUERY_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"controller query timed out after {exc.timeout} seconds"
        ) from exc


def parse_controller_audit(path, plan, plan_sha, campaign_root):
    path = require_regular(path, "controller audit").resolve()
    audit_sha, sidecar_sha = validate_sidecar(path, basename_only=False)
    key_values = {}
    controller_lines = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("JobId="):
            controller_lines.append(line)
            continue
        if "=" not in line:
            raise ValueError("controller audit contains an unrecognized line")
        key, value = line.split("=", 1)
        if key not in AUDIT_KEYS or key in key_values:
            raise ValueError(f"controller audit key is unknown or duplicated: {key}")
        key_values[key] = value
    if set(key_values) != AUDIT_KEYS or len(controller_lines) != 1:
        raise ValueError("controller audit is incomplete")
    if key_values["schema"] != AUDIT_SCHEMA:
        raise ValueError("controller audit schema mismatch")
    if key_values["plan_sha256"] != plan_sha:
        raise ValueError("controller audit plan hash mismatch")
    if key_values["worker_sha256"] != plan.get("probe_worker_sha256"):
        raise ValueError("controller audit worker hash mismatch")
    gate = key_values["gate_job_id"]
    probe_jobs = {
        "default_partition": key_values["default_probe_job_id"],
        "scaglione": key_values["scaglione_probe_job_id"],
    }
    if (
        not gate.isdigit()
        or any(not value.isdigit() for value in probe_jobs.values())
        or len(set(probe_jobs.values())) != 2
    ):
        raise ValueError("controller audit job IDs are invalid")
    controller = parse_scontrol_line(controller_lines[0])
    expected_gate = expected_gate_identity(plan_sha, gate)
    if (
        controller.get("JobId") != expected_gate["job_id"]
        or controller.get("JobName") != expected_gate["job_name"]
        or controller.get("Partition") != expected_gate["partition"]
        or controller.get("JobState") != "PENDING"
        or controller.get("Reason") != "JobHeldUser"
        or controller.get("Comment") != expected_gate["comment"]
        or dependency_job_ids(controller.get("Dependency"))
        != set(probe_jobs.values())
    ):
        raise ValueError("controller audit does not prove the exact held afterok gate")
    expected_outputs = {
        partition: lexical_absolute(
            campaign_root / "probes" / f"{partition}.attempt2.json"
        )
        for partition in PARTITIONS
    }
    recorded_outputs = {
        "default_partition": lexical_absolute(key_values["default_output"]),
        "scaglione": lexical_absolute(key_values["scaglione_output"]),
    }
    if recorded_outputs != expected_outputs:
        raise ValueError("controller audit probe output path mismatch")
    return {
        "path": str(path),
        "sha256": audit_sha,
        "sidecar_sha256": sidecar_sha,
        "gate_job_id": gate,
        "probe_job_ids": probe_jobs,
        "probe_outputs": {
            key: str(value) for key, value in expected_outputs.items()
        },
        "gate_dependency": sorted(probe_jobs.values()),
    }


def validate_probe(partition, path, job_id, plan, plan_sha):
    path = require_regular(path, f"{partition} probe").resolve()
    artifact_sha, sidecar_sha = validate_sidecar(path, basename_only=True)
    payload = read_json_regular(path, f"{partition} probe")
    probe_id = PROBE_IDS[partition]
    portable = (plan.get("python_identity") or {}).get(
        "portable_identity_sha256"
    )
    if (
        payload.get("schema") != PROBE_SCHEMA
        or payload.get("probe_id") != probe_id
        or payload.get("probe_attempt") != 2
        or str(payload.get("slurm_job_id")) != job_id
        or payload.get("slurm_partition") != partition
        or payload.get("plan_sha256") != plan_sha
        or payload.get("compatible") is not True
        or payload.get("differences") != []
        or payload.get("planned_portable_identity_sha256") != portable
        or payload.get("observed_portable_identity_sha256") != portable
    ):
        raise ValueError(f"{partition} probe identity or compatibility mismatch")
    return {
        "path": str(path),
        "sha256": artifact_sha,
        "sidecar_sha256": sidecar_sha,
        "payload": payload,
    }


def verified_binary(plan, key):
    identity = plan.get(key) or {}
    path = require_regular(identity.get("path") or "", f"approved {key}")
    if (
        identity.get("available") is not True
        or digest_file(path) != identity.get("sha256")
    ):
        raise ValueError(f"approved {key} executable unavailable or changed")
    return path


def validate_worker_completion(plan, plan_sha):
    for job in sorted(plan.get("jobs") or [], key=lambda item: item.get("job_key", "")):
        if job.get("phase") not in {"PREFLIGHT", "SEED"}:
            continue
        output = Path(str(job.get("output") or "")).resolve()
        completion_path = Path(str(output) + ".worker-completion.json")
        if not completion_path.is_file() or completion_path.is_symlink():
            continue
        completion = read_json_regular(completion_path, "worker completion")
        if (
            completion.get("schema") != COMPLETION_SCHEMA
            or completion.get("phase") != job.get("phase")
            or completion.get("plan_sha256") != plan_sha
            or completion.get("instance_file_sha256")
            != (job.get("instance") or {}).get("instance_file_sha256")
            or completion.get("job_key") != job.get("job_key")
            or completion.get("arm") is not None
        ):
            raise ValueError("downstream worker completion identity mismatch")
        hashes = completion.get("artifact_sha256")
        if not isinstance(hashes, dict) or str(output) not in hashes:
            raise ValueError("downstream worker completion omits its primary output")
        for raw, expected in sorted(hashes.items()):
            artifact = require_regular(raw, "completed downstream artifact")
            if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
                raise ValueError(
                    "downstream worker completion contains an invalid hash"
                )
            if digest_file(artifact) != expected:
                raise ValueError("downstream worker completion artifact changed")
        return {
            "type": "validated_worker_completion",
            "job_key": job["job_key"],
            "phase": job["phase"],
            "completion_path": str(completion_path),
            "completion_sha256": digest_file(completion_path),
        }
    return None


def reconciler_identity():
    path = Path(__file__).resolve()
    root = path.parents[1]
    relative = path.relative_to(root).as_posix()
    commands = (
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        ["git", "-C", str(root), "ls-files", "--error-unmatch", relative],
        [
            "git", "-C", str(root), "status", "--porcelain",
            "--untracked-files=no", "--", relative,
        ],
        ["git", "-C", str(root), "show", f"HEAD:{relative}"],
    )
    completed = [
        subprocess.run(
            command, capture_output=True,
            text=index != 3, check=False,
        )
        for index, command in enumerate(commands)
    ]
    if any(completed[index].returncode != 0 for index in (0, 1, 2, 3)):
        raise ValueError("reconciler must be tracked in a Git commit")
    if completed[2].stdout.strip():
        raise ValueError("reconciler has tracked working-tree changes")
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    if symbolic.returncode == 0:
        raise ValueError("reconciler checkout must be detached")
    current_sha = digest_file(path)
    committed_sha = digest_bytes(completed[3].stdout)
    if committed_sha != current_sha:
        raise ValueError("reconciler bytes do not match the checked-out commit")
    return {
        "git_commit": completed[0].stdout.strip(),
        "relative_path": relative,
        "sha256": current_sha,
        "detached": True,
        "tracked_clean": True,
    }


def validate_controller_proof(proof, plan, plan_sha, manifest, gate):
    if not isinstance(proof, dict):
        raise ValueError("controller proof must be an object")
    proof_type = proof.get("type")
    arrays = manifest.get("submitted_arrays") or {}
    if proof_type == "live_gate_state":
        expected_gate = expected_gate_identity(plan_sha, gate)
        if (
            str(proof.get("gate_job_id")) != expected_gate["job_id"]
            or proof.get("job_name") != expected_gate["job_name"]
            or proof.get("partition") != expected_gate["partition"]
            or proof.get("job_state") != "COMPLETED"
            or proof.get("exit_code") != "0:0"
            or proof.get("comment") != expected_gate["comment"]
            or proof.get("reason") in BLOCKED_REASONS
        ):
            raise ValueError("live gate controller proof is invalid")
        return copy.deepcopy(proof)
    if proof_type == "live_downstream_array":
        record = proof.get("record") or {}
        expected = {
            str(arrays[group]): f"SLAD:{plan_sha[:20]}:{group}"
            for group in ("PREFLIGHT", "SEED")
        }
        if (
            str(record.get("array_job_id")) not in expected
            or not str(record.get("array_task_id") or "").isdigit()
            or record.get("comment")
            != expected.get(str(record.get("array_job_id")))
            or record.get("state")
            not in {"PENDING", "RUNNING", "COMPLETING"}
            or record.get("reason") in BLOCKED_REASONS
        ):
            raise ValueError("live downstream controller proof is invalid")
        return copy.deepcopy(proof)
    if proof_type == "validated_worker_completion":
        current = validate_worker_completion(plan, plan_sha)
        if current != proof:
            raise ValueError("worker-completion controller proof changed")
        return copy.deepcopy(proof)
    raise ValueError("controller proof type is unsupported")


def collect_controller_proof(plan, plan_sha, manifest, gate, *, runner=subprocess.run):
    completion = validate_worker_completion(plan, plan_sha)
    if completion is not None:
        return completion
    scontrol = verified_binary(plan, "scontrol")
    shown = run_controller_query(
        runner,
        [str(scontrol), "show", "job", gate, "-o"],
    )
    if shown.returncode == 0:
        fields = parse_scontrol_line(shown.stdout.strip())
        expected_gate = expected_gate_identity(plan_sha, gate)
        if (
            fields.get("JobId") == expected_gate["job_id"]
            and fields.get("JobName") == expected_gate["job_name"]
            and fields.get("Partition") == expected_gate["partition"]
            and fields.get("Comment") == expected_gate["comment"]
            and fields.get("JobState") == "COMPLETED"
            and fields.get("ExitCode") == "0:0"
            and fields.get("Reason") not in BLOCKED_REASONS
        ):
            return {
                "type": "live_gate_state",
                "gate_job_id": gate,
                "job_name": fields["JobName"],
                "partition": fields["Partition"],
                "job_state": fields["JobState"],
                "exit_code": fields["ExitCode"],
                "comment": fields["Comment"],
                "reason": fields.get("Reason"),
            }
    squeue = verified_binary(plan, "squeue")
    arrays = manifest.get("submitted_arrays") or {}
    parent_ids = [str(arrays[group]) for group in ("PREFLIGHT", "SEED")]
    queued = run_controller_query(
        runner,
        [
            str(squeue), "-h", "-r", "-j", ",".join(parent_ids),
            "-o", "%F|%K|%k|%T|%r",
        ],
    )
    if queued.returncode == 0:
        candidates = []
        expected_comments = {
            str(arrays[group]): f"SLAD:{plan_sha[:20]}:{group}"
            for group in ("PREFLIGHT", "SEED")
        }
        for line in queued.stdout.splitlines():
            fields = line.split("|", 4)
            if len(fields) != 5:
                continue
            parent, task, comment, state_name, reason = fields
            if (
                parent in expected_comments
                and task.isdigit()
                and comment == expected_comments[parent]
                and state_name in {"PENDING", "RUNNING", "COMPLETING"}
                and reason not in BLOCKED_REASONS
            ):
                candidates.append({
                    "array_job_id": parent,
                    "array_task_id": task,
                    "comment": comment,
                    "state": state_name,
                    "reason": reason,
                })
        if candidates:
            return {
                "type": "live_downstream_array",
                "record": sorted(
                    candidates,
                    key=lambda item: (
                        int(item["array_job_id"]), int(item["array_task_id"])
                    ),
                )[0],
            }
    # A worker may have completed while the two live controller queries ran.
    completion = validate_worker_completion(plan, plan_sha)
    if completion is not None:
        return completion
    raise ValueError("no controller-independent proof that the afterok gate passed")


def fresh_specs_and_results(audit, probes, plan_sha, controller):
    specs = {}
    results = {}
    for partition in PARTITIONS:
        probe_id = PROBE_IDS[partition]
        job_id = audit["probe_job_ids"][partition]
        output = audit["probe_outputs"][partition]
        specs[partition] = {
            "job_id": job_id,
            "output": output,
            "probe_id": probe_id,
            "partition": partition,
            "attempt": 2,
            "comment": f"SLADP:{plan_sha[:20]}:{probe_id}:2",
        }
        payload = probes[partition]["payload"]
        results[partition] = {
            **specs[partition],
            "state": "COMPLETED",
            "state_source": "slurm_afterok_gate_controller_proof",
            "compatible": True,
            "artifact_sha256": probes[partition]["sha256"],
            "differences": [],
            "path_bound": True,
            "observed_node_metadata": payload.get("observed_node_metadata"),
            "controller_proof_type": controller["type"],
        }
    return specs, results


def build_preview(
    campaign_root, plan_sha, controller_audit, *, runner=subprocess.run,
    controller_proof=None, reconciler=None,
):
    campaign_root = Path(campaign_root).resolve()
    plan_path = require_regular(campaign_root / "approved-plan.json", "approved plan")
    manifest_path = require_regular(
        campaign_root / "campaign.json", "campaign manifest"
    )
    plan_raw = plan_path.read_bytes()
    if digest_bytes(plan_raw) != plan_sha:
        raise ValueError("approved plan hash mismatch")
    plan = json.loads(plan_raw)
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if not isinstance(plan, dict) or not isinstance(manifest, dict):
        raise ValueError("plan and manifest must be JSON objects")
    if Path(str(plan.get("campaign_root") or "")).resolve() != campaign_root:
        raise ValueError("plan campaign root mismatch")
    if manifest.get("approval_sha256") != plan_sha:
        raise ValueError("campaign approval hash mismatch")
    if manifest.get("submitted") is not False or manifest.get("gate_state") not in {
        "held", "held_probe_failure",
    }:
        raise ValueError("campaign is not in the expected pre-reconciliation state")
    arrays = manifest.get("submitted_arrays") or {}
    if set(arrays) != REQUIRED_GROUPS or any(
        not str(value).isdigit() for value in arrays.values()
    ):
        raise ValueError("campaign array identity is incomplete")
    audit = parse_controller_audit(
        controller_audit, plan, plan_sha, campaign_root
    )
    if str(manifest.get("gate_job_id")) != audit["gate_job_id"]:
        raise ValueError("manifest/controller gate ID mismatch")
    probes = {
        partition: validate_probe(
            partition,
            audit["probe_outputs"][partition],
            audit["probe_job_ids"][partition],
            plan,
            plan_sha,
        )
        for partition in PARTITIONS
    }
    if controller_proof is None:
        controller = collect_controller_proof(
            plan, plan_sha, manifest, audit["gate_job_id"], runner=runner
        )
    else:
        controller = validate_controller_proof(
            controller_proof, plan, plan_sha, manifest,
            audit["gate_job_id"],
        )
    reconciler = copy.deepcopy(
        reconciler if reconciler is not None else reconciler_identity()
    )
    if (
        not isinstance(reconciler, dict)
        or not re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
            str(reconciler.get("git_commit")),
        )
        or not re.fullmatch(r"[0-9a-f]{64}", str(reconciler.get("sha256")))
        or reconciler.get("relative_path")
        != "src/reconcile_scale_ladder_probe_artifacts.py"
        or reconciler.get("detached") is not True
        or reconciler.get("tracked_clean") is not True
    ):
        raise ValueError("reconciler identity is invalid")
    evidence = {
        "approved_plan_sha256": plan_sha,
        "manifest_before_sha256": digest_bytes(manifest_raw),
        "controller_audit": audit,
        "probes": {
            partition: {
                key: value for key, value in probes[partition].items()
                if key != "payload"
            }
            for partition in PARTITIONS
        },
        "controller_proof": controller,
        "reconciler": reconciler,
        "sacct_used": False,
    }
    evidence_sha = digest_bytes(canonical(evidence))
    proposed = copy.deepcopy(manifest)
    old_specs = copy.deepcopy(proposed.get("infrastructure_probes") or {})
    old_results = copy.deepcopy(proposed.get("probe_results") or {})
    history = copy.deepcopy(proposed.get("probe_attempt_history") or {})
    for partition in PARTITIONS:
        entry = {
            "spec": old_specs.get(partition),
            "result": old_results.get(partition),
            "disposition": "superseded_without_reuse",
        }
        history.setdefault(partition, []).append(entry)
    fresh_specs, fresh_results = fresh_specs_and_results(
        audit, probes, plan_sha, controller
    )
    proposed["probe_attempt_history"] = history
    proposed["infrastructure_probes"] = fresh_specs
    proposed["probe_results"] = fresh_results
    proposed["probe_state"] = "passed"
    proposed["submitted"] = True
    proposed["gate_state"] = "released_reconciled"
    proposed["gate_reconciliation"] = {
        "source": "probe_artifacts_afterok_controller_proof",
        "gate_job_id": audit["gate_job_id"],
        "controller_proof": controller,
        "sacct_used": False,
    }
    proposed["manual_probe_reconciliation"] = {
        "schema": SCHEMA,
        "evidence_sha256": evidence_sha,
        "manifest_before_sha256": evidence["manifest_before_sha256"],
        "controller_audit_path": audit["path"],
        "controller_audit_sha256": audit["sha256"],
        "attempt": 2,
        "sacct_used": False,
        "reconciler": reconciler,
    }
    proposed_raw = canonical(proposed)
    return {
        "schema": SCHEMA,
        "campaign_root": str(campaign_root),
        "evidence": evidence,
        "evidence_sha256": evidence_sha,
        "manifest_before_sha256": evidence["manifest_before_sha256"],
        "manifest_after_sha256": digest_bytes(proposed_raw),
        "proposed_manifest": proposed,
    }


def write_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = Path(str(path) + ".sha256")
    write_sidecar = f"{digest_bytes(raw)}  {path.name}\n".encode()
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.tmp.{os.getpid()}")
    with temporary_sidecar.open("xb") as handle:
        handle.write(write_sidecar)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary_sidecar, sidecar)
    finally:
        temporary_sidecar.unlink(missing_ok=True)


def publish_preview(preview, output):
    raw = canonical(preview)
    write_new(output, raw)
    return digest_bytes(raw)


def apply_report(
    report_path, approved_sha, controller_audit, *, runner=subprocess.run,
    reconciler=None,
):
    report_path = require_regular(report_path, "reconciliation report")
    report_raw = report_path.read_bytes()
    if digest_bytes(report_raw) != approved_sha:
        raise ValueError("approved reconciliation report hash mismatch")
    report = json.loads(report_raw)
    if report.get("schema") != SCHEMA:
        raise ValueError("reconciliation report schema mismatch")
    campaign_root = Path(report["campaign_root"]).resolve()
    manifest_path = campaign_root / "campaign.json"
    lock_path = campaign_root / ".probe-reconciliation.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current_raw = require_regular(manifest_path, "campaign manifest").read_bytes()
        current_sha = digest_bytes(current_raw)
        if current_sha == report.get("manifest_after_sha256"):
            return {"status": "already_applied", "manifest_sha256": current_sha}
        if current_sha != report.get("manifest_before_sha256"):
            raise ValueError("campaign manifest changed after preview")
        rebuilt = build_preview(
            campaign_root,
            report["evidence"]["approved_plan_sha256"],
            controller_audit,
            runner=runner,
            controller_proof=report["evidence"]["controller_proof"],
            reconciler=reconciler,
        )
        if canonical(rebuilt) != canonical(report):
            raise ValueError("reconciliation evidence changed after preview")
        proposed_raw = canonical(report["proposed_manifest"])
        if digest_bytes(proposed_raw) != report["manifest_after_sha256"]:
            raise ValueError("proposed manifest hash mismatch")
        backup = campaign_root / (
            f"campaign.json.before-probe-reconciliation.{current_sha}.json"
        )
        if backup.exists():
            if backup.is_symlink() or digest_file(backup) != current_sha:
                raise ValueError("manifest backup collision")
        else:
            os.link(manifest_path, backup)
        temporary = manifest_path.with_name(
            f".{manifest_path.name}.probe-reconcile.{os.getpid()}"
        )
        with temporary.open("xb") as handle:
            handle.write(proposed_raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
        directory_fd = os.open(campaign_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {
            "status": "applied",
            "manifest_sha256": digest_file(manifest_path),
            "backup": str(backup),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--controller-audit", type=Path, required=True)
    parser.add_argument("--preview-out", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--approved-report-sha256")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.apply:
        if (
            args.report is None
            or not args.approved_report_sha256
            or args.preview_out is not None
        ):
            parser.error("apply requires --report and --approved-report-sha256")
        result = apply_report(
            args.report, args.approved_report_sha256, args.controller_audit
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if (
        args.campaign_root is None
        or not args.approved_plan_sha256
        or args.preview_out is None
        or args.report is not None
    ):
        parser.error(
            "preview requires --campaign-root, --approved-plan-sha256, "
            "and --preview-out"
        )
    preview = build_preview(
        args.campaign_root,
        args.approved_plan_sha256,
        args.controller_audit,
    )
    digest = publish_preview(preview, args.preview_out)
    print(json.dumps({
        "preview": str(args.preview_out.resolve()),
        "approved_report_sha256": digest,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only post-hoc audit for pre-receipt scale-ladder campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from build_tariff_response_manifest import REPO_ROOT, sha256_file


SCHEMA = "evsp-dr-scale-ladder-legacy-posthoc-audit-v1"
CAPTURE_SCHEMA = "evsp-dr-legacy-scale-ladder-raw-scheduler-capture-v2"
STATUSES = {
    "legacy_posthoc_audited",
    "legacy_scheduler_unverified",
}
STRICT_FIELDS = {
    "gate_release_verification",
    "gate_reconciliation",
    "array_submission_verifications",
}
SCIENCE_GROUPS = (
    "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
    "MIP_RAW", "MIP_KNOWN",
)


def _canonical(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _artifact(path, role, *, job_key=None):
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"audit artifact is not a regular file: {path}")
    return {
        "path": str(path),
        "role": role,
        "job_key": job_key,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_new(path, payload):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(path) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _verify_git_code_hashes(plan, commit):
    for relative, expected in (plan.get("code_hashes") or {}).items():
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        if (
            completed.returncode != 0
            or hashlib.sha256(completed.stdout).hexdigest() != expected
        ):
            raise ValueError(
                f"approved code hash is not bound to {commit}: {relative}"
            )


def _validate_task_mapping(plan):
    jobs = plan.get("jobs") or []
    job_keys = [str(job.get("job_key") or "") for job in jobs]
    if (
        not jobs
        or any(not key for key in job_keys)
        or len(set(job_keys)) != len(job_keys)
    ):
        raise ValueError("legacy plan jobs are empty or duplicated")
    groups = plan.get("task_groups") or {}
    if set(groups) != set(SCIENCE_GROUPS):
        raise ValueError("legacy task groups differ")
    mapped = [
        key for group in SCIENCE_GROUPS for key in groups[group]
    ]
    if (
        len(mapped) != len(set(mapped))
        or set(mapped) != set(job_keys)
        or int(plan.get("task_count", len(mapped))) != len(mapped)
    ):
        raise ValueError("legacy task mapping is incomplete or duplicated")
    return hashlib.sha256(_canonical(groups)).hexdigest()


def _validate_plan_inputs(plan, artifacts):
    for path_key, hash_key, role in (
        ("input_manifest", "input_manifest_sha256", "input_manifest"),
        (
            "membership_preflight",
            "membership_preflight_sha256",
            "membership_preflight",
        ),
        ("instance_manifest", "instance_manifest_sha256", "instance_manifest"),
    ):
        path = Path(str(plan.get(path_key) or ""))
        expected = str(plan.get(hash_key) or "")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"legacy plan input mismatch: {path_key}")
        artifacts.append(_artifact(path, role))
    tariff = plan.get("tariff") or {}
    for path_key, hash_key, role in (
        (
            "primary_tariff_relative_path",
            "primary_tariff_sha256",
            "primary_tariff",
        ),
        (
            "extended_comparator_relative_path",
            "extended_comparator_sha256",
            "tariff_comparator",
        ),
    ):
        path = REPO_ROOT / str(tariff.get(path_key) or "")
        if not path.is_file() or sha256_file(path) != tariff.get(hash_key):
            raise ValueError(f"legacy tariff mismatch: {path_key}")
        artifacts.append(_artifact(path, role))
    seen_instances = set()
    for job in plan["jobs"]:
        instance = job.get("instance") or {}
        path = Path(str(instance.get("path") or ""))
        expected = instance.get("instance_file_sha256")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(
                f"legacy instance mismatch: {job['job_key']}"
            )
        if str(path.resolve()) not in seen_instances:
            artifacts.append(_artifact(path, "instance_input"))
            seen_instances.add(str(path.resolve()))


def _validate_selected_routes(plan, jobs_by_key):
    from expanded_path_realization import (
        BLOCK_SCHEDULE_SCHEMA,
        charging_block_schedule_sha256,
    )
    validated = []
    for job in plan["jobs"]:
        if job.get("phase") != "MIP":
            continue
        result = json.loads(Path(job["output"]).read_text())
        if result.get("incumbent_found") is not True:
            validated.append({
                "job_key": job["job_key"],
                "incumbent_found": False,
                "selected_partition_validated": None,
            })
            continue
        dependency = jobs_by_key[job["dependency_cg"]]
        cg_status_path = Path(dependency["output"])
        cg_status = json.loads(cg_status_path.read_text())
        cg_journal = Path(cg_status["columns_journal"])
        if (
            result.get("source_result_sha256")
            != sha256_file(cg_status_path)
            or result.get("source_journal_sha256")
            != sha256_file(cg_journal)
        ):
            raise ValueError(
                f"legacy selected routes use a mismatched pool: "
                f"{job['job_key']}"
            )
        expected_trips = {
            int(trip) for trip in cg_status.get("trip_ids") or []
        }
        selected = result.get("selected_routes") or []
        counts = Counter(
            int(trip)
            for route in selected
            for trip in route.get("trips") or []
        )
        if (
            not expected_trips
            or len(selected) != result.get("buses")
            or set(counts) != expected_trips
            or any(value != 1 for value in counts.values())
        ):
            raise ValueError(
                f"legacy selected routes are not an exact partition: "
                f"{job['job_key']}"
            )
        for route in selected:
            blocks = route.get("continuous_realized_charging_blocks")
            physical = route.get("physical_realization") or {}
            if (
                not isinstance(blocks, list)
                or charging_block_schedule_sha256(blocks)
                != physical.get(
                    "continuous_realized_charging_blocks_sha256"
                )
                or physical.get(
                    "continuous_realized_charging_blocks_schema"
                ) != BLOCK_SCHEDULE_SCHEMA
            ):
                raise ValueError(
                    f"legacy selected route physical evidence differs: "
                    f"{job['job_key']}"
                )
        validated.append({
            "job_key": job["job_key"],
            "incumbent_found": True,
            "selected_partition_validated": True,
            "selected_route_count": len(selected),
        })
    return validated


def _expected_dependencies(group, arrays, gate):
    dependencies = {"afterok": [str(gate)]}
    if group in {"CG", "CG_SENSITIVITY"}:
        dependencies["afterok"].append(f"{arrays['PREFLIGHT']}_*")
    elif group == "MIP_RAW":
        dependencies["aftercorr"] = [f"{arrays['CG']}_*"]
    elif group == "MIP_KNOWN":
        dependencies["aftercorr"] = sorted([
            f"{arrays['CG']}_*", f"{arrays['SEED']}_*",
        ])
    return {
        kind: sorted(values) for kind, values in dependencies.items()
    }


def _validate_scheduler_capture(path, plan, manifest, plan_sha, commit):
    from reconcile_scale_ladder_gate import (
        _dependency_semantics,
        _scontrol_record_values,
        _simple_array_task_ids,
    )

    capture_path = Path(path).expanduser().resolve()
    capture = json.loads(capture_path.read_text())
    arrays = manifest["submitted_arrays"]
    gate_id = str(manifest.get("gate_job_id") or "")
    user = str((plan.get("runtime_environment") or {}).get("USER") or "")
    if (
        capture.get("schema") != CAPTURE_SCHEMA
        or capture.get("plan_sha256") != plan_sha
        or capture.get("source_commit") != commit
        or capture.get("user") != user
        or capture.get("source")
        != "operator_read_only_scontrol_sacct_export"
        or capture.get("scontrol_sha256")
        != (plan.get("scontrol") or {}).get("sha256")
        or capture.get("sacct_sha256")
        != (plan.get("sacct") or {}).get("sha256")
    ):
        raise ValueError("legacy scheduler capture identity mismatch")
    gate = capture.get("gate") or {}
    gate_control = _scontrol_record_values(
        gate.get("scontrol_raw") or ""
    )
    gate_accounting_fields = [
        field.strip()
        for field in str(gate.get("sacct_raw") or "").split("|")
    ]
    if (
        gate_control.get("JobId") != gate_id
        or str(gate_control.get("UserId") or "").split("(", 1)[0] != user
        or gate_control.get("JobName") != f"LDG{plan_sha[:5]}"
        or gate_control.get("Partition") != "default_partition"
        or gate_control.get("Comment") != f"SLADG:{plan_sha[:20]}"
        or gate_control.get("JobState") != "COMPLETED"
        or gate_control.get("ExitCode") != "0:0"
        or len(gate_accounting_fields) != 7
        or gate_accounting_fields != [
            gate_id,
            user,
            f"LDG{plan_sha[:5]}",
            "COMPLETED",
            "default_partition",
            f"SLADG:{plan_sha[:20]}",
            "0:0",
        ]
    ):
        raise ValueError("legacy scheduler gate capture mismatch")
    captured_arrays = capture.get("arrays") or {}
    if set(captured_arrays) != set(SCIENCE_GROUPS):
        raise ValueError("legacy scheduler array capture is incomplete")
    prefixes = {
        "PREFLIGHT": "LDPF",
        "SEED": "LDSD",
        "CG": "LDCG",
        "CG_SENSITIVITY": "LDCS",
        "MIP_RAW": "LDMR",
        "MIP_KNOWN": "LDMK",
    }
    for group in SCIENCE_GROUPS:
        row = captured_arrays[group]
        parent = str(arrays[group])
        name = prefixes[group] + plan_sha[:4]
        partition = (
            "scaglione" if group.startswith("MIP")
            else "default_partition"
        )
        comment = f"SLAD:{plan_sha[:20]}:{group}"
        expected_tasks = set(range(len(plan["task_groups"][group])))
        expected_dependencies = {
            kind: set(values) for kind, values in
            _expected_dependencies(group, arrays, gate_id).items()
        }
        control_tasks = set()
        for raw in row.get("scontrol_raw") or []:
            values = _scontrol_record_values(raw)
            tasks = _simple_array_task_ids(values.get("ArrayTaskId"))
            if (
                values.get("ArrayJobId") != parent
                or str(values.get("UserId") or "").split("(", 1)[0] != user
                or values.get("JobName") != name
                or values.get("Partition") != partition
                or values.get("Comment") != comment
                or values.get("JobState") != "COMPLETED"
                or values.get("ExitCode") != "0:0"
                or _dependency_semantics(values.get("Dependency"))
                != expected_dependencies
                or control_tasks & tasks
            ):
                raise ValueError(
                    f"legacy scheduler scontrol capture mismatch: {group}"
                )
            control_tasks.update(tasks)
        accounting_tasks = set()
        for raw in row.get("sacct_raw") or []:
            fields = [field.strip() for field in str(raw).split("|")]
            if len(fields) != 7:
                raise ValueError(
                    f"legacy scheduler sacct capture malformed: {group}"
                )
            (
                job_id, row_user, row_name, state, row_partition,
                row_comment, exit_code,
            ) = fields
            prefix = f"{parent}_"
            if not job_id.startswith(prefix):
                raise ValueError(
                    f"legacy scheduler sacct task mismatch: {group}"
                )
            task_text = job_id[len(prefix):]
            if not task_text.isdigit():
                raise ValueError(
                    f"legacy scheduler sacct task invalid: {group}"
                )
            task = int(task_text)
            if (
                row_user != user
                or row_name != name
                or state != "COMPLETED"
                or row_partition != partition
                or row_comment != comment
                or exit_code != "0:0"
                or task in accounting_tasks
            ):
                raise ValueError(
                    f"legacy scheduler sacct capture mismatch: {group}"
                )
            accounting_tasks.add(task)
        if (
            control_tasks != expected_tasks
            or accounting_tasks != expected_tasks
        ):
            raise ValueError(
                f"legacy scheduler array capture mismatch: {group}"
            )
    return capture_path


def _derive_legacy_payload(
    campaign_root,
    *,
    expected_commit,
    scheduler_capture=None,
):
    root = Path(campaign_root).expanduser().resolve()
    auditor_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if auditor_head.returncode != 0:
        raise ValueError("cannot bind legacy auditor Git commit")
    auditor_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if auditor_tree.returncode != 0:
        raise ValueError("cannot bind legacy auditor Git tree")
    plan_path = root / "approved-plan.json"
    manifest_path = root / "campaign.json"
    plan_raw = plan_path.read_bytes()
    manifest_raw = manifest_path.read_bytes()
    plan_sha = hashlib.sha256(plan_raw).hexdigest()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    plan = json.loads(plan_raw)
    manifest = json.loads(manifest_raw)
    if (
        plan.get("checkout_identity", {}).get("commit") != expected_commit
        or manifest.get("approval_sha256") != plan_sha
        or manifest.get("submitted") is not True
        or manifest.get("gate_state")
        not in {"released", "released_reconciled"}
    ):
        raise ValueError("legacy campaign commit/approval/state mismatch")
    present_strict = STRICT_FIELDS & set(manifest)
    if present_strict:
        if present_strict != STRICT_FIELDS:
            raise ValueError("mixed old/new scheduler evidence is forbidden")
        raise ValueError(
            "campaign already carries prospective scheduler evidence"
        )
    arrays = manifest.get("submitted_arrays") or {}
    if (
        set(arrays) != set(SCIENCE_GROUPS)
        or any(not str(value).isdigit() for value in arrays.values())
        or len(set(map(str, arrays.values()))) != len(arrays)
        or not str(manifest.get("gate_job_id") or "").isdigit()
    ):
        raise ValueError("legacy scheduler IDs are incomplete or duplicated")
    mapping_sha = _validate_task_mapping(plan)
    artifacts = [
        _artifact(plan_path, "approved_plan"),
        _artifact(manifest_path, "original_campaign_manifest"),
    ]
    _validate_plan_inputs(plan, artifacts)
    _verify_git_code_hashes(plan, expected_commit)
    jobs_by_key = {job["job_key"]: job for job in plan["jobs"]}
    completion_hashes = {}
    from summarize_scale_ladder import _validate_completion
    for job in plan["jobs"]:
        _validate_completion(job, plan_sha)
        completion_path = Path(
            str(job["output"]) + ".worker-completion.json"
        )
        completion = json.loads(completion_path.read_text())
        artifacts.append(_artifact(
            completion_path, "worker_completion",
            job_key=job["job_key"],
        ))
        for path, digest in (completion.get("artifact_sha256") or {}).items():
            artifact = _artifact(
                path, "worker_artifact", job_key=job["job_key"]
            )
            if artifact["sha256"] != digest:
                raise ValueError(
                    f"legacy worker artifact hash mismatch: {path}"
                )
            artifacts.append(artifact)
        completion_hashes[job["job_key"]] = sha256_file(completion_path)
    selected_route_validation = _validate_selected_routes(
        plan, jobs_by_key
    )
    capture_path = None
    status = "legacy_scheduler_unverified"
    if scheduler_capture is not None:
        capture_path = _validate_scheduler_capture(
            scheduler_capture, plan, manifest, plan_sha, expected_commit
        )
        artifacts.append(_artifact(
            capture_path, "scheduler_capture"
        ))
        status = "legacy_posthoc_audited"
    payload = {
        "schema": SCHEMA,
        "auditor_git_commit": auditor_head.stdout.strip(),
        "auditor_git_tree": auditor_tree.stdout.strip(),
        "auditor_code_sha256": sha256_file(Path(__file__).resolve()),
        "legacy_evidence_status": status,
        "normalization_authorized": True,
        "normalization_scope": (
            "artifact_and_scheduler_posthoc"
            if capture_path is not None
            else "artifact_provenance_only_scheduler_unverified"
        ),
        "campaign_root": str(root),
        "source_commit": expected_commit,
        "approved_plan_sha256": plan_sha,
        "original_manifest_sha256": manifest_sha,
        "task_mapping_sha256": mapping_sha,
        "task_count": len(plan["jobs"]),
        "gate_job_id": str(manifest["gate_job_id"]),
        "submitted_arrays": {
            group: str(arrays[group]) for group in SCIENCE_GROUPS
        },
        "physics_sha256": hashlib.sha256(
            _canonical(plan.get("physics") or {})
        ).hexdigest(),
        "tariff_sha256": (
            plan.get("tariff") or {}
        ).get("primary_tariff_sha256"),
        "worker_completion_sha256": completion_hashes,
        "selected_route_validation": selected_route_validation,
        "artifact_inventory": sorted(
            artifacts,
            key=lambda item: (
                item["role"], item.get("job_key") or "", item["path"]
            ),
        ),
        "scheduler_capture_sha256": (
            sha256_file(capture_path) if capture_path is not None else None
        ),
    }
    return payload


def audit_legacy_campaign(
    campaign_root,
    sidecar_out,
    *,
    expected_commit,
    scheduler_capture=None,
):
    root = Path(campaign_root).expanduser().resolve()
    sidecar = Path(sidecar_out).expanduser().resolve()
    checksum = Path(str(sidecar) + ".sha256")
    if (
        sidecar == root
        or root in sidecar.parents
        or checksum == root
        or root in checksum.parents
    ):
        raise ValueError(
            "legacy audit sidecar/checksum must be outside campaign root"
        )
    if sidecar.exists() or checksum.exists():
        raise FileExistsError("legacy audit sidecar/checksum already exists")
    payload = _derive_legacy_payload(
        campaign_root,
        expected_commit=expected_commit,
        scheduler_capture=scheduler_capture,
    )
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode() + b"\n"
    _write_new(sidecar, encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    _write_new(
        checksum,
        f"{digest}  {sidecar.name}\n".encode(),
    )
    return payload


def validate_legacy_sidecar(campaign_root, sidecar_path):
    root = Path(campaign_root).expanduser().resolve()
    sidecar = Path(sidecar_path).expanduser().resolve()
    checksum = Path(str(sidecar) + ".sha256")
    if (
        sidecar.is_symlink()
        or checksum.is_symlink()
        or not sidecar.is_file()
        or not checksum.is_file()
    ):
        raise ValueError("legacy audit sidecar/checksum is missing")
    fields = checksum.read_text().split()
    if (
        len(fields) != 2
        or fields[1] != sidecar.name
        or fields[0] != sha256_file(sidecar)
    ):
        raise ValueError("legacy audit sidecar checksum mismatch")
    payload = json.loads(sidecar.read_text())
    plan_path = root / "approved-plan.json"
    manifest_path = root / "campaign.json"
    if (
        payload.get("schema") != SCHEMA
        or payload.get("legacy_evidence_status") not in STATUSES
        or payload.get("normalization_authorized") is not True
        or payload.get("campaign_root") != str(root)
        or payload.get("auditor_code_sha256")
        != sha256_file(Path(__file__).resolve())
        or payload.get("approved_plan_sha256") != sha256_file(plan_path)
        or payload.get("original_manifest_sha256")
        != sha256_file(manifest_path)
    ):
        raise ValueError("legacy audit sidecar campaign binding mismatch")
    inventory = payload.get("artifact_inventory")
    selected_validation = payload.get("selected_route_validation")
    completion_hashes = payload.get("worker_completion_sha256")
    if (
        not isinstance(inventory, list)
        or not inventory
        or not isinstance(selected_validation, list)
        or not isinstance(completion_hashes, dict)
        or len(completion_hashes) != payload.get("task_count")
    ):
        raise ValueError("legacy audit sidecar evidence set is incomplete")
    for artifact in inventory:
        path = Path(str(artifact.get("path") or ""))
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != artifact.get("sha256")
            or path.stat().st_size != artifact.get("size_bytes")
        ):
            raise ValueError(
                f"legacy sidecar artifact changed: {path}"
            )
    scheduler_artifacts = [
        Path(item["path"]) for item in inventory
        if item.get("role") == "scheduler_capture"
    ]
    status = payload["legacy_evidence_status"]
    if (
        (status == "legacy_posthoc_audited"
         and len(scheduler_artifacts) != 1)
        or (status == "legacy_scheduler_unverified"
            and scheduler_artifacts)
    ):
        raise ValueError("legacy scheduler evidence label is inconsistent")
    rederived = _derive_legacy_payload(
        root,
        expected_commit=payload["source_commit"],
        scheduler_capture=(
            scheduler_artifacts[0] if scheduler_artifacts else None
        ),
    )
    if _canonical(rederived) != _canonical(payload):
        raise ValueError("legacy audit sidecar evidence derivation mismatch")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--sidecar-out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--scheduler-capture", type=Path)
    args = parser.parse_args(argv)
    payload = audit_legacy_campaign(
        args.campaign_root,
        args.sidecar_out,
        expected_commit=args.expected_commit,
        scheduler_capture=args.scheduler_capture,
    )
    print(json.dumps({
        "legacy_evidence_status": payload["legacy_evidence_status"],
        "approved_plan_sha256": payload["approved_plan_sha256"],
        "task_count": payload["task_count"],
        "sidecar": str(args.sidecar_out.resolve()),
        "checksum": str(
            Path(str(args.sidecar_out.resolve()) + ".sha256")
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

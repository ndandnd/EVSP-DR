#!/usr/bin/env python3
"""Reconcile an ambiguous ladder gate release from Slurm accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

from launch_scale_ladder import (
    BOUND_RELEASE_COMMAND_ATTEMPTS,
    BOUND_RELEASE_VERIFY_ATTEMPTS,
    BOUND_RELEASE_VERIFY_DELAY_S,
    PROBE_PARTITIONS,
    PROBE_RETRYABLE_STATES,
    PROBE_TERMINAL_STATES,
    AMBIGUOUS_DISCOVERY_ATTEMPTS,
    AMBIGUOUS_DISCOVERY_DELAY_S,
    _campaign_lock,
    _probes_compatible,
    _probes_waiting,
    _probe_spec,
    _replace_json,
    _array_name,
    _sbatch,
    _submit_probe,
    _submit_array,
    _wait_for_probes,
)

SLURM_QUERY_TIMEOUT_S = 10.0
SCIENCE_GROUPS = (
    "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY", "MIP_RAW",
    "MIP_KNOWN",
)


class GateTerminalObservationError(ValueError):
    def __init__(self, message, observation):
        super().__init__(message)
        self.observation = observation


def _persist_gate_terminal_failure(
    manifest, manifest_path, gate, observation,
):
    manifest["gate_state"] = "terminal_failed"
    manifest["submitted"] = False
    manifest["gate_terminal_failure"] = {
        "verified": True,
        "gate_job_id": str(gate),
        "observation": observation,
        "source": observation.get("source"),
        "state": observation.get("state"),
        "exit_code": observation.get("exit_code"),
    }
    _replace_json(manifest_path, manifest)


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
        if live_rows[0]["state"] not in {
            "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
            "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL",
            "DEADLINE", "REVOKED", "SPECIAL_EXIT",
        }:
            return {**live_rows[0], "source": "squeue", "live": True}
        # squeue carries no ExitCode.  A terminal-looking live row is not a
        # completion certificate; resolve it through scontrol/accounting.

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
        terminal = controller_row["state"] in {
            "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
            "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL",
            "DEADLINE", "REVOKED", "SPECIAL_EXIT",
        }
        if terminal:
            exit_code = str(controller_row.get("exit_code") or "")
            if re.fullmatch(r"[0-9]+:[0-9]+", exit_code) is None:
                observation = {
                    **controller_row, "source": "scontrol", "live": False,
                }
                raise GateTerminalObservationError(
                    "terminal gate lacks an exact exit code", observation
                )
            if (
                controller_row["state"] == "COMPLETED"
                and exit_code != "0:0"
            ):
                observation = {
                    **controller_row, "source": "scontrol", "live": False,
                }
                raise GateTerminalObservationError(
                    "completed gate has nonzero exit code", observation
                )
            return {
                **controller_row, "source": "scontrol", "live": False,
            }
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
        raise GateTerminalObservationError(
            "completed gate accounting has nonzero exit code",
            {**rows[0], "source": "sacct", "live": False},
        )
    if (
        rows[0]["state"] in {
            "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
            "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL",
            "DEADLINE", "REVOKED", "SPECIAL_EXIT",
        }
        and re.fullmatch(r"[0-9]+:[0-9]+", rows[0]["exit_code"]) is None
    ):
        raise GateTerminalObservationError(
            "terminal gate accounting lacks an exact exit code",
            {**rows[0], "source": "sacct", "live": False},
        )
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


def _gate_release_observation_result(observation):
    """Classify an exact scientific-gate release observation.

    ``None`` means that the exact gate remains authoritatively user-held and
    an idempotent release request is eligible.  A successful result means the
    hold-clearing postcondition was observed, not merely that a command exited
    zero.
    """
    state = observation.get("state")
    if state in PROBE_TERMINAL_STATES:
        if state == "COMPLETED" and observation.get("exit_code") == "0:0":
            return subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout=json.dumps(observation, sort_keys=True), stderr="",
            )
        return subprocess.CompletedProcess(
            args=[], returncode=3,
            stdout=json.dumps(observation, sort_keys=True),
            stderr=f"scientific gate became terminal in state {state}",
        )
    if observation.get("live") is not True:
        return subprocess.CompletedProcess(
            args=[], returncode=3, stdout="",
            stderr="scientific gate has no authoritative live state",
        )
    if state in {"CONFIGURING", "RUNNING", "COMPLETING"}:
        return subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps(observation, sort_keys=True), stderr="",
        )
    if state == "PENDING":
        reason = str(observation.get("reason") or "")
        if not reason:
            return subprocess.CompletedProcess(
                args=[], returncode=3, stdout="",
                stderr="scientific gate pending reason is missing",
            )
        if reason == "JobHeldUser":
            return None
        if (
            reason.startswith("JobHeld")
            or reason in {"Dependency", "DependencyNeverSatisfied"}
        ):
            return subprocess.CompletedProcess(
                args=[], returncode=3, stdout="",
                stderr=(
                    "scientific gate has an invalid post-release reason: "
                    f"{reason}"
                ),
            )
        return subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps(observation, sort_keys=True), stderr="",
        )
    return subprocess.CompletedProcess(
        args=[], returncode=3, stdout="",
        stderr=f"scientific gate is not proven held or released: {state}",
    )


def _gate_release_verification(
    gate, observation, command_attempts, command_diagnostics,
):
    return {
        "verified": True,
        "job_id": str(gate),
        "command_attempts": command_attempts,
        "observation": observation,
        "command_diagnostics": list(command_diagnostics),
    }


def _release_gate_with_postcondition(
    plan, gate, expected_plan_sha, *, runner=None, resolver=None,
    sleeper=None,
):
    """Request gate release and verify the exact scheduler postcondition."""
    runner = subprocess.run if runner is None else runner
    resolver = _resolve_gate_state if resolver is None else resolver
    sleeper = time.sleep if sleeper is None else sleeper
    try:
        observation = resolver(
            plan, gate, expected_plan_sha, runner=runner
        )
    except GateTerminalObservationError as exc:
        return subprocess.CompletedProcess(
            args=[], returncode=3,
            stdout=json.dumps(exc.observation, sort_keys=True),
            stderr=str(exc),
        )
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            args=[], returncode=3, stdout="", stderr=str(exc)
        )
    classified = _gate_release_observation_result(observation)
    if classified is not None:
        if classified.returncode != 0:
            return classified
        return subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps(_gate_release_verification(
                gate, observation, 0, [],
            ), sort_keys=True),
            stderr="",
        )

    try:
        scontrol_path = _approved_tool_path(plan, "scontrol")
    except ValueError as exc:
        return subprocess.CompletedProcess(
            args=[], returncode=3, stdout="", stderr=str(exc)
        )
    command_diagnostics = []
    for command_attempt in range(1, BOUND_RELEASE_COMMAND_ATTEMPTS + 1):
        try:
            requested = _bounded_query(
                runner, [str(scontrol_path), "release", str(gate)]
            )
            command_diagnostics.append({
                "attempt": command_attempt,
                "returncode": requested.returncode,
                "stdout": (requested.stdout or "").strip(),
                "stderr": (requested.stderr or "").strip(),
            })
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            command_diagnostics.append({
                "attempt": command_attempt,
                "query_error": str(exc),
            })

        verified_observation = False
        held_after_command = False
        for verify_attempt in range(1, BOUND_RELEASE_VERIFY_ATTEMPTS + 1):
            sleeper(BOUND_RELEASE_VERIFY_DELAY_S)
            try:
                observation = resolver(
                    plan, gate, expected_plan_sha, runner=runner
                )
            except GateTerminalObservationError as exc:
                return subprocess.CompletedProcess(
                    args=[], returncode=3,
                    stdout=json.dumps(exc.observation, sort_keys=True),
                    stderr=str(exc),
                )
            except ValueError as exc:
                return subprocess.CompletedProcess(
                    args=[], returncode=3, stdout="", stderr=str(exc)
                )
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                command_diagnostics.append({
                    "attempt": command_attempt,
                    "verification": verify_attempt,
                    "query_error": str(exc),
                })
                continue
            verified_observation = True
            classified = _gate_release_observation_result(observation)
            if classified is None:
                held_after_command = True
                continue
            if classified.returncode != 0:
                return classified
            return subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout=json.dumps(_gate_release_verification(
                    gate, observation, command_attempt,
                    command_diagnostics,
                ), sort_keys=True),
                stderr="",
            )
        if not verified_observation:
            break
        if not held_after_command:
            break

    return subprocess.CompletedProcess(
        args=[], returncode=3, stdout="",
        stderr=(
            "scientific gate release postcondition was not observed: "
            + json.dumps(command_diagnostics, sort_keys=True)
        ),
    )


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


def _dependency_semantics(value):
    raw = str(value or "")
    clauses = raw.split(",")
    if (
        not raw
        or "?" in raw
        or any(not clause for clause in clauses)
    ):
        raise ValueError("held array dependency syntax is invalid")
    semantics = {}
    seen = set()
    for clause in clauses:
        fields = clause.split(":")
        if (
            len(fields) < 2
            or fields[0] not in {
                "after", "afterany", "afterburstbuffer", "aftercorr",
                "afternotok", "afterok",
            }
            or any(not dependency_token for dependency_token in fields[1:])
        ):
            raise ValueError("held array dependency syntax is invalid")
        kind = fields[0]
        dependency_ids = []
        for dependency_token in fields[1:]:
            matched = re.fullmatch(
                r"(?P<job>[0-9]+(?:_\*)?)"
                r"(?:\((?:unfulfilled|failed)\))?",
                dependency_token,
            )
            if matched is None:
                raise ValueError("held array dependency syntax is invalid")
            dependency_id = matched.group("job")
            dependency = (kind, dependency_id)
            if dependency in seen:
                raise ValueError("held array dependency is duplicated")
            seen.add(dependency)
            dependency_ids.append(dependency_id)
        semantics.setdefault(kind, set()).update(dependency_ids)
    return semantics


def _scontrol_record_values(record):
    """Parse one ``scontrol -o`` record without accepting duplicate keys."""
    values = {}
    for token in str(record).split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in values:
            raise ValueError(f"duplicate scontrol field: {key}")
        values[key] = value
    return values


def _simple_array_task_ids(value):
    """Expand the simple scalar/range forms emitted for held array records."""
    matched = re.fullmatch(
        r"(?P<start>[0-9]+)(?:-(?P<end>[0-9]+))?", str(value or "")
    )
    if matched is None:
        raise ValueError("held array task expression is invalid")
    start = int(matched.group("start"))
    end = int(matched.group("end") or start)
    if end < start:
        raise ValueError("held array task range is descending")
    return set(range(start, end + 1))


def _validate_held_array_controller(
    plan, expected_plan_sha, group, job_id, gate_id, array_ids,
):
    scontrol_path = _approved_tool_path(plan, "scontrol")
    shown = _bounded_query(
        subprocess.run,
        [str(scontrol_path), "show", "job", str(job_id), "-o"],
    )
    if shown.returncode != 0 or not shown.stdout.strip():
        raise ValueError("held array is absent from scontrol")
    controller_rows = [
        line.strip() for line in shown.stdout.splitlines() if line.strip()
    ]
    if not controller_rows:
        raise ValueError("held array has no controller/task records")
    expected_partition = (
        "scaglione" if group.startswith("MIP") else "default_partition"
    )
    expected_common = {
        "JobName": _array_name(group, expected_plan_sha),
        "JobState": "PENDING",
        "Partition": expected_partition,
        "Reason": "Dependency",
        "Comment": f"SLAD:{expected_plan_sha[:20]}:{group}",
        "RunTime": "00:00:00",
        "ArrayJobId": str(job_id),
    }
    expected_dependencies = {"afterok": {str(gate_id)}}
    if group in {"CG", "CG_SENSITIVITY"}:
        if "PREFLIGHT" not in array_ids:
            raise ValueError("CG array exists without its PREFLIGHT dependency")
        expected_dependencies["afterok"].add(
            f"{array_ids['PREFLIGHT']}_*"
        )
    elif group == "MIP_RAW":
        if "CG" not in array_ids:
            raise ValueError("RAW MIP array exists without its CG dependency")
        expected_dependencies["aftercorr"] = {
            f"{array_ids['CG']}_*"
        }
    elif group == "MIP_KNOWN":
        if "CG" not in array_ids or "SEED" not in array_ids:
            raise ValueError(
                "KNOWN MIP array exists without CG/SEED dependencies"
            )
        expected_dependencies["aftercorr"] = {
            f"{array_ids['CG']}_*", f"{array_ids['SEED']}_*",
        }

    expected_user = str(
        (plan.get("runtime_environment") or {}).get("USER") or ""
    )
    if not expected_user:
        raise ValueError("approved runtime user is missing")
    expected_tasks = set(range(len(plan["task_groups"][group])))
    if not expected_tasks:
        raise ValueError("held array task group is empty")
    observed_tasks = set()
    observed_job_records = set()

    for row_index, controller_row in enumerate(controller_rows):
        values = _scontrol_record_values(controller_row)
        errors = [
            {
                "field": field,
                "expected": expected_value,
                "observed": (
                    _normalized_state(values.get(field))
                    if field == "JobState" else values.get(field, "")
                ),
            }
            for field, expected_value in expected_common.items()
            if (
                _normalized_state(values.get(field))
                if field == "JobState" else values.get(field, "")
            ) != expected_value
        ]
        if errors:
            raise ValueError(
                f"held array record {row_index} mismatch: {errors}"
            )

        raw_job_id = str(values.get("JobId") or "")
        if not raw_job_id.isdigit():
            raise ValueError("held array record JobId is invalid")
        if raw_job_id in observed_job_records:
            raise ValueError("held array contains a duplicate JobId record")
        observed_job_records.add(raw_job_id)

        user_id = str(values.get("UserId") or "")
        if re.fullmatch(
            rf"{re.escape(expected_user)}\([0-9]+\)", user_id
        ) is None:
            raise ValueError("held array record UserId mismatch")

        row_tasks = _simple_array_task_ids(values.get("ArrayTaskId"))
        if not row_tasks <= expected_tasks:
            raise ValueError("held array record contains an out-of-range task")
        overlap = observed_tasks & row_tasks
        if overlap:
            raise ValueError(
                f"held array task coverage overlaps: {sorted(overlap)}"
            )
        observed_tasks.update(row_tasks)

        observed_dependencies = _dependency_semantics(
            values.get("Dependency")
        )
        if observed_dependencies != expected_dependencies:
            raise ValueError(
                "held array dependency fingerprint mismatch: "
                f"expected={expected_dependencies} "
                f"observed={observed_dependencies}"
            )

    if observed_tasks != expected_tasks:
        raise ValueError(
            "held array task coverage mismatch: "
            f"expected={sorted(expected_tasks)} "
            f"observed={sorted(observed_tasks)}"
        )


def _discover_held_science_jobs(plan, expected_plan_sha, manifest):
    """Recover exact held gate/array parents without trusting comments alone."""
    user = str((plan.get("runtime_environment") or {}).get("USER") or "")
    if not user:
        raise ValueError("approved runtime user is missing")
    squeue_path = _approved_tool_path(plan, "squeue")
    listed = _bounded_query(subprocess.run, [
        str(squeue_path), "-h", "-u", user,
        "-o", "%F|%j|%T|%P|%R|%k",
    ])
    if listed.returncode != 0:
        raise RuntimeError("cannot query squeue")

    prefix = f"SLAD:{expected_plan_sha[:20]}:"
    gate_comment = f"SLADG:{expected_plan_sha[:20]}"
    discovered = {}
    discovered_gate = None
    seen_rows = set()
    for line in listed.stdout.splitlines():
        fields = tuple(value.strip() for value in line.split("|", 5))
        if len(fields) != 6:
            continue
        job_id, job_name, state, partition, reason, comment = fields
        if comment != gate_comment and not comment.startswith(prefix):
            continue
        if fields in seen_rows:
            # ``%F`` is the parent array ID, so squeue may emit one identical
            # row per task.  Collapse exact duplicates, but reject any mixed
            # fingerprint or parent below.
            continue
        seen_rows.add(fields)
        if not job_id.isdigit():
            raise ValueError("held-science parent job ID is invalid")
        state = _normalized_state(state)
        if state != "PENDING":
            raise ValueError("held-science recovery found a released job")
        if comment == gate_comment:
            if (
                job_name != f"LDG{expected_plan_sha[:5]}"
                or partition != "default_partition"
                or reason != "JobHeldUser"
            ):
                raise ValueError("held gate fingerprint mismatch")
            if discovered_gate is not None and discovered_gate != job_id:
                raise ValueError("multiple gates share one plan comment")
            discovered_gate = job_id
            continue

        group = comment[len(prefix):]
        if group not in SCIENCE_GROUPS:
            raise ValueError(f"unknown scale-ladder array group: {group}")
        expected_partition = (
            "scaglione" if group.startswith("MIP")
            else "default_partition"
        )
        if (
            job_name != _array_name(group, expected_plan_sha)
            or partition != expected_partition
            or reason != "Dependency"
            or comment != f"{prefix}{group}"
        ):
            raise ValueError(
                f"held array fingerprint mismatch for group {group}"
            )
        if group in discovered and discovered[group] != job_id:
            raise ValueError("multiple arrays share one plan/group comment")
        if group in discovered:
            continue
        if job_id in discovered.values():
            raise ValueError("one parent job ID is bound to multiple groups")
        discovered[group] = job_id

    recorded = manifest.get("submitted_arrays") or {}
    if set(recorded) - set(SCIENCE_GROUPS):
        raise ValueError("campaign records an unknown array group")
    if any(not str(value).isdigit() for value in recorded.values()):
        raise ValueError("recorded array parent ID is invalid")
    for group, recorded_id in recorded.items():
        if group not in discovered:
            raise ValueError("recorded held array is absent from live squeue")
        if str(discovered[group]) != str(recorded_id):
            raise ValueError("recorded/discovered array ID conflict")
    recorded_gate = manifest.get("gate_job_id")
    if recorded_gate is not None:
        if not str(recorded_gate).isdigit():
            raise ValueError("recorded gate ID is invalid")
        if discovered_gate is None:
            raise ValueError("recorded held gate is absent from live squeue")
        if str(recorded_gate) != str(discovered_gate):
            raise ValueError("recorded/discovered gate ID conflict")
    gate_id = discovered_gate or (
        str(recorded_gate) if recorded_gate is not None else None
    )
    if discovered and gate_id is None:
        raise ValueError("held arrays exist without an exact held gate")
    all_array_ids = {**discovered, **recorded}
    for group, job_id in discovered.items():
        _validate_held_array_controller(
            plan, expected_plan_sha, group, job_id, gate_id, all_array_ids
        )
    return discovered, discovered_gate


def _discover_intended_science_jobs(
    plan, expected_plan_sha, manifest, *, sleeper=time.sleep,
):
    """Boundedly recover every accepted-before-ID science submission.

    A durable intent means a prior process may have died immediately after
    Slurm accepted the held gate or array.  Absence is therefore ambiguous and
    can never authorize a replacement submission.
    """
    gate_intent = manifest.get("gate_submission_intent")
    array_intents = dict(manifest.get("array_submission_intents") or {})
    for attempt_index in range(AMBIGUOUS_DISCOVERY_ATTEMPTS):
        discovered, discovered_gate = _discover_held_science_jobs(
            plan, expected_plan_sha, manifest
        )
        missing_gate = gate_intent is not None and discovered_gate is None
        missing_arrays = set(array_intents) - set(discovered)
        if not missing_gate and not missing_arrays:
            return discovered, discovered_gate
        if attempt_index + 1 < AMBIGUOUS_DISCOVERY_ATTEMPTS:
            sleeper(AMBIGUOUS_DISCOVERY_DELAY_S)
    raise RuntimeError(
        "prior held gate/array submission remains ambiguous; one or more "
        "exact jobs are not yet visible, so replacement submission is refused"
    )


def _reconcile_locked(
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
    recorded_gate = str(manifest.get("gate_job_id") or "")
    if (
        recorded_gate.isdigit()
        and str(
            (plan.get("runtime_environment") or {}).get("USER") or ""
        )
    ):
        try:
            early_gate_observation = _resolve_gate_state(
                plan, recorded_gate, expected_plan_sha
            )
        except GateTerminalObservationError as exc:
            _persist_gate_terminal_failure(
                manifest, manifest_path, recorded_gate, exc.observation
            )
            raise
        if (
            early_gate_observation.get("state") in PROBE_TERMINAL_STATES
            and early_gate_observation.get("state") != "COMPLETED"
        ):
            _persist_gate_terminal_failure(
                manifest,
                manifest_path,
                recorded_gate,
                early_gate_observation,
            )
            raise ValueError(
                "scientific gate is terminal but not completed: "
                f"{early_gate_observation.get('state')}"
            )
    if manifest.get("gate_state") in {
        "creating", "held", "held_after_partial_submission",
        "held_probe_failure", "held_probe_waiting",
    }:
        has_submission_intent = bool(
            manifest.get("gate_submission_intent")
            or manifest.get("array_submission_intents")
        )
        if has_submission_intent:
            discovered, discovered_gate = _discover_intended_science_jobs(
                plan, expected_plan_sha, manifest
            )
        else:
            discovered, discovered_gate = _discover_held_science_jobs(
                plan, expected_plan_sha, manifest
            )
        recorded = manifest.get("submitted_arrays") or {}
        combined = {**discovered, **recorded}
        manifest["submitted_arrays"] = combined
        if discovered_gate:
            manifest["gate_job_id"] = discovered_gate
            manifest.pop("gate_submission_intent", None)
        array_intents = dict(
            manifest.get("array_submission_intents") or {}
        )
        for group in discovered:
            array_intents.pop(group, None)
        if array_intents:
            manifest["array_submission_intents"] = array_intents
        else:
            manifest.pop("array_submission_intents", None)
        gate_for_resume = manifest.get("gate_job_id")
        missing_groups = [
            group for group in SCIENCE_GROUPS if group not in combined
        ]
        if missing_groups and resume_missing_arrays:
            if not str(gate_for_resume or "").isdigit():
                manifest["gate_submission_intent"] = {
                    "job_name": f"LDG{expected_plan_sha[:5]}",
                    "partition": "default_partition",
                    "comment": f"SLADG:{expected_plan_sha[:20]}",
                }
                _replace_json(manifest_path, manifest)
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
                manifest.pop("gate_submission_intent", None)
                manifest["gate_state"] = "held"
                _replace_json(manifest_path, manifest)
            if not str(gate_for_resume or "").isdigit():
                raise ValueError("cannot resume arrays without a proven gate")
            _require_gate_held(
                plan, str(gate_for_resume), expected_plan_sha
            )
            logs = root / "logs"
            for group in SCIENCE_GROUPS:
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
                array_intents = dict(
                    manifest.get("array_submission_intents") or {}
                )
                array_intents[group] = {
                    "job_name": _array_name(group, expected_plan_sha),
                    "partition": (
                        "scaglione" if group.startswith("MIP")
                        else "default_partition"
                    ),
                    "comment": (
                        f"SLAD:{expected_plan_sha[:20]}:{group}"
                    ),
                }
                manifest["array_submission_intents"] = array_intents
                _replace_json(manifest_path, manifest)
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
                array_intents.pop(group, None)
                if array_intents:
                    manifest["array_submission_intents"] = array_intents
                else:
                    manifest.pop("array_submission_intents", None)
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
    try:
        gate_observation = _resolve_gate_state(
            plan, gate, expected_plan_sha
        )
    except GateTerminalObservationError as exc:
        _persist_gate_terminal_failure(
            manifest, manifest_path, gate, exc.observation
        )
        raise
    gate_state = gate_observation["state"]
    if gate_state in PROBE_TERMINAL_STATES and gate_state != "COMPLETED":
        _persist_gate_terminal_failure(
            manifest, manifest_path, gate, gate_observation
        )
        raise ValueError(
            f"scientific gate is terminal but not completed: {gate_state}"
        )
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
    if gate_state == "COMPLETED":
        manifest["gate_state"] = "released_reconciled"
        manifest["submitted"] = True
        manifest["gate_reconciliation"] = {
            "source": gate_observation["source"],
            "gate_job_id": gate, "state": "COMPLETED",
        }
        _replace_json(manifest_path, manifest)
        return manifest
    classified = _gate_release_observation_result(gate_observation)
    if classified is None and not release_held_gate:
        manifest["gate_state"] = "held_probe_passed"
        manifest.pop("gate_release_verification", None)
        _replace_json(manifest_path, manifest)
        return manifest
    if classified is not None and classified.returncode != 0:
        manifest["gate_state"] = "held_release_failed"
        manifest["release_error"] = classified.stderr
        _replace_json(manifest_path, manifest)
        raise ValueError(classified.stderr)

    if classified is None:
        manifest["gate_state"] = "release_retry_attempting"
        manifest.pop("gate_release_verification", None)
        _replace_json(manifest_path, manifest)
        released = _release_gate_with_postcondition(
            plan, gate, expected_plan_sha
        )
        if released.returncode != 0:
            try:
                terminal_observation = (
                    json.loads(released.stdout)
                    if released.stdout else None
                )
            except json.JSONDecodeError:
                terminal_observation = None
            if (
                isinstance(terminal_observation, dict)
                and terminal_observation.get("state")
                in PROBE_TERMINAL_STATES
            ):
                _persist_gate_terminal_failure(
                    manifest, manifest_path, gate, terminal_observation
                )
                raise RuntimeError(
                    "scientific gate reached a terminal non-success state"
                )
            manifest["gate_state"] = "held_release_failed"
            manifest["release_error"] = (
                released.stderr or released.stdout
            ).strip()
            _replace_json(manifest_path, manifest)
            raise RuntimeError(
                "scientific gate release postcondition was not verified"
            )
        verification = json.loads(released.stdout)
    else:
        # A restart may observe a prior release request already in progress.
        # Preserve that state without issuing another mutation.
        verification = _gate_release_verification(
            gate, gate_observation, 0, []
        )
    manifest["gate_release_verification"] = verification
    manifest["gate_state"] = "release_retry_requested"
    manifest.pop("release_error", None)
    _replace_json(manifest_path, manifest)
    return manifest


def reconcile(
    root, expected_plan_sha, *,
    release_held_gate=False,
    resume_missing_arrays=False,
    retry_failed_probes=False,
):
    """Serialize every public mutation of one campaign.

    The activation controller already owns this same lock and therefore calls
    ``_reconcile_locked`` directly.  Operators and the CLI always enter here,
    preventing two reconcilers (or a reconciler and activation) from accepting
    duplicate held Slurm jobs before their IDs are durable.
    """
    root = Path(root).resolve()
    with _campaign_lock(root):
        return _reconcile_locked(
            root,
            expected_plan_sha,
            release_held_gate=release_held_gate,
            resume_missing_arrays=resume_missing_arrays,
            retry_failed_probes=retry_failed_probes,
        )


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

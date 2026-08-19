#!/usr/bin/env python3
"""Identity-bound, observed-postcondition helpers for Slurm mutations."""

from __future__ import annotations

import json
import re
import subprocess
import time


TERMINAL_STATES = frozenset({
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
})
ACTIVE_STATES = frozenset({"CONFIGURING", "RUNNING", "COMPLETING"})
DEFAULT_COMMAND_ATTEMPTS = 3
DEFAULT_VERIFY_ATTEMPTS = 5
DEFAULT_VERIFY_DELAY_S = 1.0
DEFAULT_QUERY_TIMEOUT_S = 10.0


class SlurmContractError(RuntimeError):
    """A scheduler identity, precondition, or postcondition was not proved."""

    def __init__(
        self, message, *, observation=None, diagnostics=None,
    ):
        super().__init__(message)
        self.observation = observation
        self.diagnostics = list(diagnostics or [])


def normalized_state(value):
    words = str(value or "").strip().split()
    return words[0].split("+", 1)[0].upper() if words else ""


def _bounded_query(runner, command, timeout_s):
    try:
        return runner(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise SlurmContractError(
            f"Slurm query timed out after {exc.timeout} seconds"
        ) from exc


def _user_from_user_id(value):
    matched = re.fullmatch(r"([^()\s]+)(?:\([0-9]+\))?", str(value or ""))
    return matched.group(1) if matched else str(value or "")


def _identity_errors(spec, row):
    return [
        {
            "field": field,
            "expected": str(spec.get(field) or ""),
            "observed": str(row.get(field) or ""),
        }
        for field in (
            "job_id", "user", "job_name", "partition", "comment",
        )
        if str(row.get(field) or "") != str(spec.get(field) or "")
    ]


def _require_spec(spec, *, require_job_id=True):
    required = {
        "user", "job_name", "partition", "comment", "role",
    }
    missing = sorted(
        field for field in required if not str(spec.get(field) or "")
    )
    if require_job_id and not str(spec.get("job_id") or "").isdigit():
        missing.append("job_id")
    if missing:
        raise SlurmContractError(
            f"Slurm job specification is incomplete: {sorted(set(missing))}"
        )


def _terminal_row(row, source):
    state = normalized_state(row.get("state"))
    exit_code = str(row.get("exit_code") or "")
    if state not in TERMINAL_STATES:
        raise SlurmContractError("internal terminal-row classification error")
    if re.fullmatch(r"[0-9]+:[0-9]+", exit_code) is None:
        raise SlurmContractError(
            "terminal Slurm observation lacks an exact exit code",
            observation={**row, "state": state, "source": source},
        )
    return {
        **row,
        "state": state,
        "exit_code": exit_code,
        "source": source,
        "live": False,
    }


def _parse_scontrol_record(payload):
    values = {}
    for token in str(payload).split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in values:
            raise SlurmContractError(
                f"duplicate scontrol field: {key}"
            )
        values[key] = value
    return values


def resolve_exact_job(
    spec,
    *,
    runner=None,
    squeue="squeue",
    scontrol="scontrol",
    sacct="sacct",
    timeout_s=DEFAULT_QUERY_TIMEOUT_S,
):
    """Resolve one exact job, preferring the live controller over accounting."""
    _require_spec(spec)
    runner = subprocess.run if runner is None else runner
    job_id = str(spec["job_id"])
    diagnostics = []

    try:
        listed = _bounded_query(
            runner,
            [
                str(squeue), "-h", "-u", str(spec["user"]),
                "-o", "%i|%u|%j|%T|%P|%R|%k",
            ],
            timeout_s,
        )
    except SlurmContractError as exc:
        diagnostics.append({"source": "squeue", "error": str(exc)})
    else:
        if listed.returncode != 0:
            diagnostics.append({
                "source": "squeue",
                "returncode": listed.returncode,
                "stdout": (listed.stdout or "").strip(),
                "stderr": (listed.stderr or "").strip(),
            })
        else:
            rows = []
            for line in listed.stdout.splitlines():
                fields = [field.strip() for field in line.split("|", 6)]
                if len(fields) == 7 and fields[0] == job_id:
                    rows.append({
                        "job_id": fields[0],
                        "user": fields[1],
                        "job_name": fields[2],
                        "state": normalized_state(fields[3]),
                        "partition": fields[4],
                        "reason": fields[5],
                        "comment": fields[6],
                        "exit_code": None,
                    })
            if len(rows) > 1:
                raise SlurmContractError(
                    "multiple live rows match the exact Slurm job ID"
                )
            if rows:
                errors = _identity_errors(spec, rows[0])
                if errors:
                    raise SlurmContractError(
                        "live Slurm job identity mismatch: "
                        + json.dumps(errors, sort_keys=True),
                        observation=rows[0],
                    )
                if rows[0]["state"] not in TERMINAL_STATES:
                    return {
                        **rows[0], "source": "squeue", "live": True,
                    }
                diagnostics.append({
                    "source": "squeue",
                    "note": "terminal-looking row requires exit-code source",
                })

    try:
        shown = _bounded_query(
            runner,
            [str(scontrol), "show", "job", job_id, "-o"],
            timeout_s,
        )
    except SlurmContractError as exc:
        diagnostics.append({"source": "scontrol", "error": str(exc)})
    else:
        if shown.returncode == 0 and shown.stdout.strip():
            values = _parse_scontrol_record(shown.stdout)
            row = {
                "job_id": values.get("JobId", ""),
                "user": _user_from_user_id(values.get("UserId")),
                "job_name": values.get("JobName", ""),
                "state": normalized_state(values.get("JobState")),
                "partition": values.get("Partition", ""),
                "reason": values.get("Reason", ""),
                "comment": values.get("Comment", ""),
                "exit_code": values.get("ExitCode"),
            }
            errors = _identity_errors(spec, row)
            if errors:
                raise SlurmContractError(
                    "controller Slurm job identity mismatch: "
                    + json.dumps(errors, sort_keys=True),
                    observation=row,
                )
            if row["state"] in TERMINAL_STATES:
                return _terminal_row(row, "scontrol")
            return {**row, "source": "scontrol", "live": True}
        diagnostics.append({
            "source": "scontrol",
            "returncode": shown.returncode,
            "stdout": (shown.stdout or "").strip(),
            "stderr": (shown.stderr or "").strip(),
        })

    try:
        accounted = _bounded_query(
            runner,
            [
                str(sacct), "-X", "-n", "-P", "-j", job_id,
                "--format=JobIDRaw,User,JobName%64,State,Partition%64,"
                "Comment%256,ExitCode",
            ],
            timeout_s,
        )
    except SlurmContractError as exc:
        diagnostics.append({"source": "sacct", "error": str(exc)})
    else:
        if accounted.returncode != 0:
            diagnostics.append({
                "source": "sacct",
                "returncode": accounted.returncode,
                "stdout": (accounted.stdout or "").strip(),
                "stderr": (accounted.stderr or "").strip(),
            })
        else:
            rows = []
            for line in accounted.stdout.splitlines():
                fields = [field.strip() for field in line.split("|", 6)]
                if len(fields) == 7 and fields[0] == job_id:
                    rows.append({
                        "job_id": fields[0],
                        "user": fields[1],
                        "job_name": fields[2],
                        "state": normalized_state(fields[3]),
                        "partition": fields[4],
                        "comment": fields[5],
                        "exit_code": fields[6],
                    })
            if len(rows) != 1:
                diagnostics.append({
                    "source": "sacct",
                    "error": "no unique exact accounting row",
                    "matching_rows": len(rows),
                })
            else:
                errors = _identity_errors(spec, rows[0])
                if errors:
                    raise SlurmContractError(
                        "accounting Slurm job identity mismatch: "
                        + json.dumps(errors, sort_keys=True),
                        observation=rows[0],
                    )
                if rows[0]["state"] not in TERMINAL_STATES:
                    raise SlurmContractError(
                        "non-live accounting row is not terminal",
                        observation={
                            **rows[0], "source": "sacct", "live": False,
                        },
                        diagnostics=diagnostics,
                    )
                return _terminal_row(rows[0], "sacct")

    raise SlurmContractError(
        "exact Slurm job state could not be resolved",
        diagnostics=diagnostics,
    )


def verify_held_receipt(
    spec,
    *,
    runner=None,
    resolver=None,
    sleeper=None,
    attempts=DEFAULT_VERIFY_ATTEMPTS,
    delay_s=DEFAULT_VERIFY_DELAY_S,
):
    """Prove that a parsed submission receipt names the intended held job."""
    runner = subprocess.run if runner is None else runner
    resolver = resolve_exact_job if resolver is None else resolver
    sleeper = time.sleep if sleeper is None else sleeper
    diagnostics = []
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            sleeper(delay_s)
        try:
            observation = resolver(spec, runner=runner)
        except SlurmContractError as exc:
            if exc.observation is not None:
                raise
            diagnostics.append({
                "attempt": attempt,
                "query_error": str(exc),
                "query_diagnostics": exc.diagnostics,
            })
            continue
        if (
            observation.get("live") is True
            and observation.get("state") == "PENDING"
            and observation.get("reason") == "JobHeldUser"
        ):
            return {
                "verified": True,
                "role": spec["role"],
                "job_id": str(spec["job_id"]),
                "attempts": attempt,
                "observation": observation,
                "diagnostics": diagnostics,
            }
        raise SlurmContractError(
            "submission receipt did not resolve to the exact held job",
            observation=observation,
            diagnostics=diagnostics,
        )
    raise SlurmContractError(
        "held submission receipt could not be verified",
        diagnostics=diagnostics,
    )


def _release_classification(
    observation, *, dependency_is_valid, terminal_success_required=True,
):
    state = observation.get("state")
    if state in TERMINAL_STATES:
        if (
            not terminal_success_required
            or (
                state == "COMPLETED"
                and observation.get("exit_code") == "0:0"
            )
        ):
            return "released"
        return "terminal_failed"
    if observation.get("live") is not True:
        return "invalid"
    if state in ACTIVE_STATES:
        return "released"
    if state != "PENDING":
        return "invalid"
    reason = str(observation.get("reason") or "")
    if reason == "JobHeldUser":
        return "held"
    if not reason or reason.startswith("JobHeld"):
        return "invalid"
    if reason == "DependencyNeverSatisfied":
        return "invalid"
    if reason == "Dependency" and not dependency_is_valid:
        return "invalid"
    return "released"


def release_with_postcondition(
    spec,
    *,
    dependency_is_valid=False,
    terminal_success_required=True,
    runner=None,
    resolver=None,
    sleeper=None,
    command_attempts=DEFAULT_COMMAND_ATTEMPTS,
    verify_attempts=DEFAULT_VERIFY_ATTEMPTS,
    delay_s=DEFAULT_VERIFY_DELAY_S,
    scontrol="scontrol",
):
    """Release one exact held job and prove the observed state transition."""
    runner = subprocess.run if runner is None else runner
    resolver = resolve_exact_job if resolver is None else resolver
    sleeper = time.sleep if sleeper is None else sleeper
    diagnostics = []
    try:
        observation = resolver(spec, runner=runner)
    except SlurmContractError:
        raise
    classification = _release_classification(
        observation,
        dependency_is_valid=dependency_is_valid,
        terminal_success_required=terminal_success_required,
    )
    if classification == "released":
        return {
            "verified": True,
            "role": spec["role"],
            "job_id": str(spec["job_id"]),
            "command_attempts": 0,
            "observation": observation,
            "command_diagnostics": diagnostics,
        }
    if classification == "terminal_failed":
        raise SlurmContractError(
            "exact Slurm job is terminal without successful completion",
            observation=observation,
        )
    if classification != "held":
        raise SlurmContractError(
            "exact Slurm job is not in a valid release precondition",
            observation=observation,
        )

    for command_attempt in range(1, command_attempts + 1):
        try:
            requested = _bounded_query(
                runner,
                [str(scontrol), "release", str(spec["job_id"])],
                DEFAULT_QUERY_TIMEOUT_S,
            )
            diagnostics.append({
                "attempt": command_attempt,
                "returncode": requested.returncode,
                "stdout": (requested.stdout or "").strip(),
                "stderr": (requested.stderr or "").strip(),
            })
        except SlurmContractError as exc:
            diagnostics.append({
                "attempt": command_attempt,
                "command_error": str(exc),
            })

        saw_exact_held = False
        saw_exact_observation = False
        for verify_attempt in range(1, verify_attempts + 1):
            sleeper(delay_s)
            try:
                observation = resolver(spec, runner=runner)
            except SlurmContractError as exc:
                if exc.observation is not None:
                    raise
                diagnostics.append({
                    "attempt": command_attempt,
                    "verification": verify_attempt,
                    "query_error": str(exc),
                    "query_diagnostics": exc.diagnostics,
                })
                continue
            saw_exact_observation = True
            classification = _release_classification(
                observation,
                dependency_is_valid=dependency_is_valid,
                terminal_success_required=terminal_success_required,
            )
            if classification == "released":
                return {
                    "verified": True,
                    "role": spec["role"],
                    "job_id": str(spec["job_id"]),
                    "command_attempts": command_attempt,
                    "observation": observation,
                    "command_diagnostics": diagnostics,
                }
            if classification == "terminal_failed":
                raise SlurmContractError(
                    "exact Slurm job became terminal without success",
                    observation=observation,
                    diagnostics=diagnostics,
                )
            if classification != "held":
                raise SlurmContractError(
                    "exact Slurm job has an invalid post-release state",
                    observation=observation,
                    diagnostics=diagnostics,
                )
            saw_exact_held = True
        if not saw_exact_observation:
            break
        if not saw_exact_held:
            break

    raise SlurmContractError(
        "release postcondition was not observed",
        observation=observation,
        diagnostics=diagnostics,
    )


def _array_task_ids(value):
    raw = str(value or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    raw = raw.split("%", 1)[0]
    if not raw:
        raise SlurmContractError("array task expression is missing")
    tasks = set()
    for item in raw.split(","):
        matched = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", item)
        if matched is None:
            raise SlurmContractError(
                f"unsupported array task expression: {value!r}"
            )
        start = int(matched.group(1))
        end = int(matched.group(2) or start)
        if end < start:
            raise SlurmContractError("descending array task expression")
        tasks.update(range(start, end + 1))
    return tasks


def verify_array_receipt(
    parent_id,
    task_specs,
    *,
    runner=None,
    sleeper=None,
    scontrol="scontrol",
    attempts=DEFAULT_VERIFY_ATTEMPTS,
    delay_s=DEFAULT_VERIFY_DELAY_S,
):
    """Verify exact coverage and identity of a direct Slurm array receipt."""
    parent_id = str(parent_id)
    if not parent_id.isdigit() or not task_specs:
        raise SlurmContractError("array receipt specification is invalid")
    expected_tasks = set(task_specs)
    if expected_tasks != set(range(len(task_specs))):
        raise SlurmContractError("array task specification is not contiguous")
    for task, spec in task_specs.items():
        _require_spec(spec)
        if str(spec["job_id"]) != f"{parent_id}_{task}":
            raise SlurmContractError("array task job ID specification differs")
    runner = subprocess.run if runner is None else runner
    sleeper = time.sleep if sleeper is None else sleeper
    diagnostics = []
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            sleeper(delay_s)
        try:
            shown = _bounded_query(
                runner,
                [str(scontrol), "show", "job", parent_id, "-o"],
                DEFAULT_QUERY_TIMEOUT_S,
            )
        except SlurmContractError as exc:
            diagnostics.append({
                "attempt": attempt,
                "error": str(exc),
            })
            continue
        if shown.returncode != 0 or not shown.stdout.strip():
            diagnostics.append({
                "attempt": attempt,
                "returncode": shown.returncode,
                "stdout": (shown.stdout or "").strip(),
                "stderr": (shown.stderr or "").strip(),
            })
            continue
        observations = {}
        for line_index, line in enumerate(shown.stdout.splitlines()):
            if not line.strip():
                continue
            values = _parse_scontrol_record(line)
            if str(values.get("ArrayJobId") or "") != parent_id:
                raise SlurmContractError(
                    "array controller parent ID mismatch"
                )
            tasks = _array_task_ids(values.get("ArrayTaskId"))
            if not tasks <= expected_tasks:
                raise SlurmContractError(
                    "array controller contains an unexpected task"
                )
            sample = task_specs[min(tasks)]
            common = {
                "user": _user_from_user_id(values.get("UserId")),
                "job_name": values.get("JobName", ""),
                "partition": values.get("Partition", ""),
                "comment": values.get("Comment", ""),
            }
            errors = [
                {
                    "field": field,
                    "expected": str(sample[field]),
                    "observed": str(common[field]),
                }
                for field in ("user", "job_name", "partition", "comment")
                if str(common[field]) != str(sample[field])
            ]
            if errors:
                raise SlurmContractError(
                    "array receipt identity mismatch: "
                    + json.dumps(errors, sort_keys=True)
                )
            state = normalized_state(values.get("JobState"))
            if not state:
                raise SlurmContractError("array receipt state is missing")
            exit_code = values.get("ExitCode")
            if (
                state in TERMINAL_STATES
                and re.fullmatch(
                    r"[0-9]+:[0-9]+", str(exit_code or "")
                ) is None
            ):
                raise SlurmContractError(
                    "terminal array receipt lacks an exact exit code"
                )
            for task in tasks:
                if task in observations:
                    raise SlurmContractError(
                        "array controller task coverage overlaps"
                    )
                observations[task] = {
                    "job_id": f"{parent_id}_{task}",
                    **common,
                    "state": state,
                    "reason": values.get("Reason", ""),
                    "exit_code": exit_code,
                    "source": "scontrol_array",
                    "live": state not in TERMINAL_STATES,
                    "record_index": line_index,
                }
        if set(observations) == expected_tasks:
            return {
                "verified": True,
                "parent_job_id": parent_id,
                "attempts": attempt,
                "task_count": len(expected_tasks),
                "task_observations": {
                    str(task): observations[task]
                    for task in sorted(observations)
                },
                "diagnostics": diagnostics,
            }
        diagnostics.append({
            "attempt": attempt,
            "error": "array task coverage incomplete",
            "observed_tasks": sorted(observations),
            "expected_tasks": sorted(expected_tasks),
        })
    raise SlurmContractError(
        "direct array submission receipt could not be verified",
        diagnostics=diagnostics,
    )


def cancel_with_postcondition(
    spec,
    *,
    runner=None,
    resolver=None,
    sleeper=None,
    command_attempts=DEFAULT_COMMAND_ATTEMPTS,
    verify_attempts=DEFAULT_VERIFY_ATTEMPTS,
    delay_s=DEFAULT_VERIFY_DELAY_S,
    scancel="scancel",
):
    """Cancel one exact held job and prove its exact terminal cancellation."""
    runner = subprocess.run if runner is None else runner
    resolver = resolve_exact_job if resolver is None else resolver
    sleeper = time.sleep if sleeper is None else sleeper
    observation = resolver(spec, runner=runner)
    if observation.get("state") == "CANCELLED":
        return {
            "verified": True,
            "role": spec["role"],
            "job_id": str(spec["job_id"]),
            "command_attempts": 0,
            "observation": observation,
            "command_diagnostics": [],
        }
    if observation.get("state") in TERMINAL_STATES:
        raise SlurmContractError(
            "exact job terminated without verified cancellation",
            observation=observation,
        )
    if (
        observation.get("live") is not True
        or observation.get("state") != "PENDING"
        or observation.get("reason") != "JobHeldUser"
    ):
        raise SlurmContractError(
            "cancellation precondition is not exact PENDING/JobHeldUser",
            observation=observation,
        )

    diagnostics = []
    for command_attempt in range(1, command_attempts + 1):
        try:
            requested = _bounded_query(
                runner,
                [str(scancel), str(spec["job_id"])],
                DEFAULT_QUERY_TIMEOUT_S,
            )
            diagnostics.append({
                "attempt": command_attempt,
                "returncode": requested.returncode,
                "stdout": (requested.stdout or "").strip(),
                "stderr": (requested.stderr or "").strip(),
            })
        except SlurmContractError as exc:
            diagnostics.append({
                "attempt": command_attempt,
                "command_error": str(exc),
            })
        saw_live = False
        saw_observation = False
        for verify_attempt in range(1, verify_attempts + 1):
            sleeper(delay_s)
            try:
                observation = resolver(spec, runner=runner)
            except SlurmContractError as exc:
                if exc.observation is not None:
                    raise
                diagnostics.append({
                    "attempt": command_attempt,
                    "verification": verify_attempt,
                    "query_error": str(exc),
                    "query_diagnostics": exc.diagnostics,
                })
                continue
            saw_observation = True
            if observation.get("state") == "CANCELLED":
                return {
                    "verified": True,
                    "role": spec["role"],
                    "job_id": str(spec["job_id"]),
                    "command_attempts": command_attempt,
                    "observation": observation,
                    "command_diagnostics": diagnostics,
                }
            if observation.get("state") in TERMINAL_STATES:
                raise SlurmContractError(
                    "exact job terminated without verified cancellation",
                    observation=observation,
                    diagnostics=diagnostics,
                )
            if observation.get("live") is not True:
                raise SlurmContractError(
                    "cancellation has no authoritative live postcondition",
                    observation=observation,
                    diagnostics=diagnostics,
                )
            saw_live = True
        if not saw_observation:
            break
        if not saw_live:
            break
    raise SlurmContractError(
        "cancellation postcondition was not observed",
        observation=observation,
        diagnostics=diagnostics,
    )


def discover_live_job_by_identity(
    spec,
    *,
    runner=None,
    squeue="squeue",
    timeout_s=DEFAULT_QUERY_TIMEOUT_S,
):
    """Discover one live job by immutable identity when its ID was not saved."""
    _require_spec(spec, require_job_id=False)
    runner = subprocess.run if runner is None else runner
    listed = _bounded_query(
        runner,
        [
            str(squeue), "-h", "-u", str(spec["user"]),
            "-o", "%i|%u|%j|%T|%P|%R|%k",
        ],
        timeout_s,
    )
    if listed.returncode != 0:
        raise SlurmContractError("cannot discover exact live Slurm job")
    matches = []
    for line in listed.stdout.splitlines():
        fields = [field.strip() for field in line.split("|", 6)]
        if len(fields) != 7 or not fields[0].isdigit():
            continue
        row = {
            "job_id": fields[0],
            "user": fields[1],
            "job_name": fields[2],
            "state": normalized_state(fields[3]),
            "partition": fields[4],
            "reason": fields[5],
            "comment": fields[6],
        }
        if all(
            str(row[field]) == str(spec[field])
            for field in ("user", "job_name", "partition", "comment")
        ):
            matches.append(row)
    if len(matches) > 1:
        raise SlurmContractError(
            "multiple live Slurm jobs share the immutable execution identity"
        )
    return matches[0] if matches else None


def verified_gate_evidence(manifest, expected_spec):
    """Validate persisted release and successful-terminal gate evidence."""
    _require_spec(expected_spec)
    recorded_spec = manifest.get("gate_spec")
    if not isinstance(recorded_spec, dict):
        raise ValueError("legacy tariff gate evidence is unverified")
    for field in (
        "job_id", "user", "job_name", "partition", "comment", "role",
    ):
        if str(recorded_spec.get(field) or "") != str(
            expected_spec.get(field) or ""
        ):
            raise ValueError(f"tariff gate specification mismatch: {field}")
    release = manifest.get("gate_release_verification")
    terminal = manifest.get("gate_terminal_verification")
    if (
        not isinstance(release, dict)
        or release.get("verified") is not True
        or release.get("role") != expected_spec["role"]
        or str(release.get("job_id") or "") != expected_spec["job_id"]
    ):
        raise ValueError("tariff gate release evidence is unverified")
    release_observation = release.get("observation") or {}
    errors = _identity_errors(expected_spec, release_observation)
    if errors or _release_classification(
        release_observation,
        dependency_is_valid=False,
        terminal_success_required=True,
    ) != "released":
        raise ValueError("tariff gate release observation is invalid")
    if (
        not isinstance(terminal, dict)
        or terminal.get("verified") is not True
        or terminal.get("role") != expected_spec["role"]
        or str(terminal.get("job_id") or "") != expected_spec["job_id"]
    ):
        raise ValueError("tariff gate successful completion is unverified")
    terminal_observation = terminal.get("observation") or {}
    errors = _identity_errors(expected_spec, terminal_observation)
    if (
        errors
        or terminal_observation.get("state") != "COMPLETED"
        or terminal_observation.get("exit_code") != "0:0"
        or terminal_observation.get("source") not in {"scontrol", "sacct"}
    ):
        raise ValueError("tariff gate terminal observation is invalid")
    return {
        "release": release,
        "terminal": terminal,
    }

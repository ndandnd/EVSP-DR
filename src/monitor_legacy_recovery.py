#!/usr/bin/env python3
"""Read-only health monitor for legacy big-tariff exact-CG recovery.

The original campaign is a Slurm job array, while each repaired continuation
is a separately submitted job.  This monitor deliberately uses ``sacct``'s
``JobID`` field for array identities (for example ``867334_22``).  ``JobIDRaw``
is retained only as a diagnostic because on Unicorn it is an unrelated
internal per-task id (for example ``867359``).

The monitor never edits a campaign artifact.  It validates the migration
attestation, reads the append-only iteration log, and combines those facts
with Slurm state into one HEALTHY/WARN/FAIL verdict.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import getpass
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


MIGRATION_SCHEMA = "evsp-dr-legacy-exact-pool-migration-v1"
ITERATION_HEADER = (
    "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,pool_columns"
)
TERMINAL_STOPS = {
    "certified",
    "wall_limit",
    "max_iters",
    "no_path",
    "degenerate_stall",
    "stalled_marginal_returns",
    "master_failed",
}
FAIL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
ACTIVE_STATES = {
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SUSPENDED",
}
ERROR_PATTERN = re.compile(
    r"\b(?:FATAL|Traceback|JSONDecodeError|DurableFileError|MigrationError)\b"
)


@dataclass(frozen=True)
class SlurmRecord:
    job_id: str
    job_id_raw: str
    job_name: str
    state: str
    elapsed: str = ""
    exit_code: str = ""
    start: str = ""
    end: str = ""
    reason: str = ""
    node: str = ""
    source: str = "sacct"


@dataclass(frozen=True)
class IterationRow:
    elapsed_s: float
    iteration: int
    lp_obj: float
    route_weight: float
    artificials: float
    min_rc: float
    pool_columns: int


def normalize_state(value: str) -> str:
    """Return a stable Slurm state without decorations or reason text."""

    return value.strip().split()[0].rstrip("+") if value.strip() else "UNKNOWN"


def parse_sacct(text: str) -> list[SlurmRecord]:
    """Parse headerless ``sacct -P`` output.

    Field order is JobID, JobIDRaw, JobName, State, Elapsed, ExitCode, Start,
    End.  Array logic must use ``job_id`` and never ``job_id_raw``.
    """

    records = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        fields = raw_line.rstrip("\n").split("|")
        if fields and fields[-1] == "":
            fields.pop()
        fields += [""] * (8 - len(fields))
        if len(fields) < 4:
            continue
        records.append(
            SlurmRecord(
                job_id=fields[0].strip(),
                job_id_raw=fields[1].strip(),
                job_name=fields[2].strip(),
                state=normalize_state(fields[3]),
                elapsed=fields[4].strip(),
                exit_code=fields[5].strip(),
                start=fields[6].strip(),
                end=fields[7].strip(),
            )
        )
    return records


def parse_squeue(text: str) -> list[SlurmRecord]:
    """Parse headerless ``squeue`` output with an explicit pipe format."""

    records = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        fields = raw_line.rstrip("\n").split("|")
        fields += [""] * (6 - len(fields))
        records.append(
            SlurmRecord(
                job_id=fields[0].strip(),
                job_id_raw="",
                job_name=fields[1].strip(),
                state=normalize_state(fields[2]),
                elapsed=fields[3].strip(),
                reason=fields[4].strip(),
                node=fields[5].strip(),
                source="squeue",
            )
        )
    return records


def array_task_number(job_id: str, array_job: int) -> Optional[int]:
    """Extract a task from sacct's JobID, excluding steps and ranges."""

    match = re.fullmatch(rf"{re.escape(str(array_job))}_(\d+)", job_id)
    return int(match.group(1)) if match else None


def array_task_records(
    records: Iterable[SlurmRecord], array_job: int
) -> dict[int, SlurmRecord]:
    selected = {}
    for record in records:
        task = array_task_number(record.job_id, array_job)
        if task is not None:
            selected[task] = record
    return selected


def unsuccessful_array_records(
    records: Iterable[SlurmRecord], array_job: int
) -> list[dict[str, Any]]:
    """Return every failure-like array task with its true array identity.

    In particular, ``task`` is parsed from ``JobID`` and cannot accidentally
    become Slurm's internal numeric ``JobIDRaw``.
    """

    unsuccessful = []
    for task, record in sorted(array_task_records(records, array_job).items()):
        if record.state not in FAIL_STATES:
            continue
        unsuccessful.append({"task": task, **asdict(record)})
    return unsuccessful


def run_command(arguments: Sequence[str]) -> tuple[str, Optional[str]]:
    try:
        completed = subprocess.run(
            list(arguments),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return "", f"{arguments[0]} unavailable: {exc}"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return "", f"{' '.join(arguments)} failed ({completed.returncode}): {detail}"
    return completed.stdout, None


def query_array(array_job: int) -> tuple[list[SlurmRecord], Optional[str]]:
    output, error = run_command(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "--array",
            "-j",
            str(array_job),
            "--format=JobID%64,JobIDRaw,JobName%64,State,Elapsed,ExitCode,Start,End",
        ]
    )
    return parse_sacct(output), error


def query_queue(user: str) -> tuple[list[SlurmRecord], Optional[str]]:
    output, error = run_command(
        [
            "squeue",
            "-h",
            "-u",
            user,
            "-o",
            "%i|%j|%T|%M|%R|%N",
        ]
    )
    return parse_squeue(output), error


def query_job(job_id: str) -> tuple[list[SlurmRecord], Optional[str]]:
    output, error = run_command(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job_id,
            "--format=JobID%64,JobIDRaw,JobName%64,State,Elapsed,ExitCode,Start,End",
        ]
    )
    return parse_sacct(output), error


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def sha256_prefix(path: Path, length: Optional[int] = None) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as handle:
        while remaining is None or remaining > 0:
            amount = 1024 * 1024
            if remaining is not None:
                amount = min(amount, remaining)
            block = handle.read(amount)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining is not None and remaining != 0:
        raise ValueError(
            f"{path} ended {remaining} bytes before the attested prefix"
        )
    return digest.hexdigest()


def same_path(left: Any, right: Path) -> bool:
    if not left:
        return False
    return os.path.realpath(os.path.expanduser(str(left))) == os.path.realpath(str(right))


def verify_attestation(
    result_path: Path,
    status: dict[str, Any],
    *,
    array_job: int,
    task: int,
    mode: str,
) -> dict[str, Any]:
    """Validate migration identity and, by default, immutable prefixes."""

    attestation_path = Path(str(result_path) + ".migration_attestation.json")
    journal_path = Path(str(result_path) + ".columns.jsonl")
    iters_path = Path(str(result_path) + ".iters.csv")
    raw_dir = result_path.parent / f"{result_path.name}.legacy_raw"
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []
    attestation: dict[str, Any] = {}

    try:
        attestation = load_json_object(attestation_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "errors": [f"migration attestation unreadable: {exc}"],
            "warnings": warnings,
            "checks": checks,
            "path": str(attestation_path),
            "repairs": {},
            "tool_commit": None,
            "continuation_commit": None,
            "mode": mode,
        }

    if attestation.get("schema") != MIGRATION_SCHEMA:
        errors.append(f"unexpected attestation schema {attestation.get('schema')!r}")
    else:
        checks.append("schema")
    migration_id = attestation.get("migration_id")
    parent = status.get("resume_parent") or {}
    if not isinstance(parent, dict):
        parent = {}
    if not migration_id or parent.get("migration_id") != migration_id:
        errors.append("status resume_parent does not match migration_id")
    else:
        checks.append("status-parent")
    if parent.get("schema") != MIGRATION_SCHEMA:
        errors.append("status resume_parent has no migration schema")

    source = attestation.get("source") or {}
    if str(source.get("slurm_array_job")) != str(array_job):
        errors.append(
            f"attestation array job {source.get('slurm_array_job')!r} != {array_job}"
        )
    if str(source.get("slurm_task")) != str(task):
        errors.append(f"attestation task {source.get('slurm_task')!r} != {task}")
    if not any("attestation array" in item or "attestation task" in item for item in errors):
        checks.append("slurm-source")

    tool = attestation.get("tool") or {}
    tool_commit = tool.get("commit")
    provenance = status.get("provenance") or {}
    if not isinstance(provenance, dict):
        provenance = {}
    continuation_claim = provenance.get("git_commit")
    continuation_commit = (
        continuation_claim
        if isinstance(continuation_claim, str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", continuation_claim)
        else None
    )
    if continuation_commit is None:
        errors.append(
            "status provenance has no valid 40-character continuation commit"
        )
    else:
        checks.append("continuation-commit")
    if not tool_commit or parent.get("tool_commit") != tool_commit:
        errors.append("status resume_parent tool commit does not match migration tool")
    else:
        checks.append("migration-tool-commit")

    validation = attestation.get("validation") or {}
    false_validation = sorted(key for key, value in validation.items() if value is not True)
    if false_validation:
        errors.append("attestation validation is not true: " + ", ".join(false_validation))
    elif validation:
        checks.append("migration-validation")
    else:
        errors.append("attestation has no validation block")

    destination = attestation.get("destination") or {}
    expected_paths = {
        "result": result_path,
        "journal": journal_path,
        "iters": iters_path,
    }
    for key, expected in expected_paths.items():
        if not same_path(destination.get(key), expected):
            errors.append(f"attestation destination {key} does not name {expected}")
        if not expected.is_file():
            errors.append(f"destination {key} is missing: {expected}")
    if not any(item.startswith("attestation destination") for item in errors):
        checks.append("destination-paths")

    manifest_path = raw_dir / "raw_manifest.json"
    try:
        manifest = load_json_object(manifest_path)
        if manifest != attestation:
            errors.append("raw_manifest.json differs from migration attestation")
        else:
            checks.append("raw-manifest")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"raw migration manifest unreadable: {exc}")

    if mode in {"prefix", "full"}:
        for label, path in (("journal", journal_path), ("iters", iters_path)):
            try:
                initial_bytes = int(destination[f"{label}_initial_bytes"])
                expected_hash = str(destination[f"{label}_initial_sha256"])
                actual_hash = sha256_prefix(path, initial_bytes)
                if actual_hash != expected_hash:
                    errors.append(f"{label} migrated prefix hash mismatch")
                else:
                    checks.append(f"{label}-prefix-sha256")
            except (KeyError, OSError, TypeError, ValueError) as exc:
                errors.append(f"could not verify {label} migrated prefix: {exc}")

    if mode == "full":
        raw_sources = {
            "result": raw_dir / "source_result.json",
            "journal": raw_dir / "source_result.json.columns.jsonl",
            "iters": raw_dir / "source_result.json.iters.csv",
        }
        for label, path in raw_sources.items():
            expected_hash = source.get(f"{label}_sha256")
            try:
                actual_hash = sha256_prefix(path)
            except OSError as exc:
                errors.append(f"raw {label} unreadable: {exc}")
                continue
            if not expected_hash or actual_hash != expected_hash:
                errors.append(f"raw {label} hash mismatch")
            else:
                checks.append(f"raw-{label}-sha256")
        for log in source.get("logs") or []:
            if not isinstance(log, dict):
                errors.append("attested source log entry is not an object")
                continue
            path = raw_dir / str(log.get("archive_name", ""))
            try:
                actual_hash = sha256_prefix(path)
            except OSError as exc:
                errors.append(f"raw source log unreadable: {exc}")
                continue
            if actual_hash != log.get("sha256"):
                errors.append(f"raw source log hash mismatch: {path.name}")
            else:
                checks.append(f"raw-log-{path.name}-sha256")

    repairs = attestation.get("repairs") or {}
    if not isinstance(repairs, dict):
        warnings.append("attestation repairs field is not an object")
        repairs = {}
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "path": str(attestation_path),
        "repairs": repairs,
        "tool_commit": tool_commit,
        "continuation_commit": continuation_commit,
        "mode": mode,
    }


def tail_lines(path: Path, count: int, block_size: int = 65536) -> list[str]:
    """Read at most ``count`` complete-ish lines without scanning the file."""

    if count <= 0:
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        chunks = []
        newlines = 0
        while position > 0 and newlines <= count:
            amount = min(block_size, position)
            position -= amount
            handle.seek(position)
            block = handle.read(amount)
            chunks.append(block)
            newlines += block.count(b"\n")
        data = b"".join(reversed(chunks))
    return data.decode("utf-8", errors="replace").splitlines()[-count:]


def parse_iteration_rows(lines: Iterable[str]) -> list[IterationRow]:
    rows = []
    for line in lines:
        if not line.strip() or line.strip() == ITERATION_HEADER:
            continue
        try:
            fields = next(csv.reader([line]))
            if len(fields) != 7:
                continue
            values = [float(item) for item in fields]
            if not all(math.isfinite(value) for index, value in enumerate(values) if index != 5):
                continue
            if not (math.isfinite(values[5]) or values[5] == math.inf):
                continue
            rows.append(
                IterationRow(
                    elapsed_s=values[0],
                    iteration=int(values[1]),
                    lp_obj=values[2],
                    route_weight=values[3],
                    artificials=values[4],
                    min_rc=values[5],
                    pool_columns=int(values[6]),
                )
            )
        except (ValueError, csv.Error, StopIteration):
            continue
    return rows


def iteration_trend(rows: Sequence[IterationRow], minutes: float) -> Optional[dict[str, Any]]:
    if len(rows) < 2:
        return None
    latest = rows[-1]
    cutoff = latest.elapsed_s - minutes * 60.0
    window = [row for row in rows if row.elapsed_s >= cutoff]
    if len(window) < 2:
        window = list(rows[-2:])
    first, last = window[0], window[-1]
    duration_s = last.elapsed_s - first.elapsed_s
    if duration_s <= 0:
        return None
    scale = 3600.0 / duration_s
    return {
        "minutes": duration_s / 60.0,
        "rows": len(window),
        "iterations_per_hour": (last.iteration - first.iteration) * scale,
        "objective_drop_per_hour": (first.lp_obj - last.lp_obj) * scale,
        "weight_drop_per_hour": (first.route_weight - last.route_weight) * scale,
        "columns_added_per_hour": (last.pool_columns - first.pool_columns) * scale,
        "first_iteration": first.iteration,
        "last_iteration": last.iteration,
    }


def file_info(path: Path, now: float) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": stat.st_size,
        "mtime": dt.datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        "age_s": max(0.0, now - stat.st_mtime),
    }


def discover_result(root: Path, array_job: int, task: int) -> Optional[Path]:
    campaign = root / "src" / "results" / "legacy_recovery" / f"job{array_job}"
    candidates = []
    for path in campaign.glob(f"c*/task{task}/*.json"):
        name = path.name
        if (
            name.endswith(".migration_attestation.json")
            or name.endswith(".snapshot.json")
            or ".legacy_raw" in str(path)
        ):
            continue
        siblings = [path, Path(str(path) + ".iters.csv"), Path(str(path) + ".columns.jsonl")]
        newest = max((item.stat().st_mtime for item in siblings if item.exists()), default=0.0)
        candidates.append((newest, str(path), path))
    return max(candidates)[2] if candidates else None


def latest_snapshot(result_path: Path) -> Optional[dict[str, Any]]:
    stem = str(result_path)
    if stem.endswith(".json"):
        stem = stem[:-5]
    snapshots = []
    for path in result_path.parent.glob(Path(stem).name + ".m*.snapshot.json"):
        try:
            payload = load_json_object(path)
            mark = float(payload.get("snapshot_mark_minutes"))
            journal = Path(str(path) + ".columns.jsonl")
            snapshots.append((mark, path.stat().st_mtime, path, journal, payload))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not snapshots:
        return None
    mark, _, path, journal, payload = max(snapshots)
    final = payload.get("final") or {}
    return {
        "path": str(path),
        "mark_minutes": mark,
        "stop_reason": payload.get("stop_reason"),
        "iterations": payload.get("iterations"),
        "columns": payload.get("columns"),
        "wall_s": payload.get("wall_s"),
        "journal_exists": journal.is_file(),
        "journal_bytes": journal.stat().st_size if journal.is_file() else None,
        "lp_obj": final.get("lp_obj"),
        "route_weight": final.get("route_weight"),
        "artificials": final.get("artificials"),
        "min_rc": final.get("min_rc"),
    }


def scan_log_errors(log_dir: Path, recovery_job: Optional[str]) -> list[str]:
    if not recovery_job or not log_dir.is_dir():
        return []
    findings = []
    for path in sorted(log_dir.glob(f"*_{recovery_job}.out")) + sorted(
        log_dir.glob(f"*_{recovery_job}.err")
    ):
        try:
            for line in tail_lines(path, 300):
                if ERROR_PATTERN.search(line):
                    findings.append(f"{path.name}: {line.strip()}")
        except OSError as exc:
            findings.append(f"could not inspect {path}: {exc}")
    return findings[-10:]


def human_bytes(value: Optional[int]) -> str:
    if value is None:
        return "NA"
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024.0 or unit == "TiB":
            return f"{number:.1f}{unit}" if unit != "B" else f"{int(number)}B"
        number /= 1024.0
    return f"{number:.1f}TiB"


def human_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "NA"
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def fmt_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def choose_recovery_records(
    queue_records: Sequence[SlurmRecord],
    *,
    task: int,
    continuation_commit: Optional[str],
    explicit_job: Optional[str],
) -> list[SlurmRecord]:
    if explicit_job:
        return [record for record in queue_records if record.job_id == explicit_job]
    prefix = f"R{task}-"
    candidates = [record for record in queue_records if record.job_name.startswith(prefix)]
    if not continuation_commit:
        return []
    suffix = f"-c{continuation_commit[:6]}"
    return [record for record in candidates if record.job_name.endswith(suffix)]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    now = dt.datetime.now().timestamp()
    root = args.root.resolve()
    result_path = args.result.resolve() if args.result else discover_result(
        root, args.array_job, args.task
    )
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    status: dict[str, Any] = {}
    status_error = None
    if result_path is None:
        status_error = "no recovery result was discovered"
        warnings.append(status_error)
    else:
        try:
            status = load_json_object(result_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            status_error = f"recovery status unreadable: {exc}"
            errors.append(status_error)

    attestation = None
    if result_path is not None and status:
        attestation = verify_attestation(
            result_path,
            status,
            array_job=args.array_job,
            task=args.task,
            mode=args.attestation_mode,
        )
        errors.extend(attestation["errors"])
        warnings.extend(attestation["warnings"])

    array_records: list[SlurmRecord] = []
    array_query_error = None
    queue_records: list[SlurmRecord] = []
    queue_error = None
    recovery_records: list[SlurmRecord] = []
    recovery_job = args.recovery_job
    if not args.no_slurm:
        array_records, array_query_error = query_array(args.array_job)
        queue_records, queue_error = query_queue(args.user)
        recovery_records = choose_recovery_records(
            queue_records,
            task=args.task,
            continuation_commit=(attestation or {}).get("continuation_commit"),
            explicit_job=args.recovery_job,
        )
        if args.recovery_job and not recovery_records:
            historical, historical_error = query_job(args.recovery_job)
            if historical_error:
                warnings.append(historical_error)
            recovery_records = [
                record
                for record in historical
                if record.job_id == args.recovery_job
            ]
        if not recovery_job and recovery_records:
            recovery_job = recovery_records[0].job_id
        if array_query_error:
            warnings.append(array_query_error)
        if queue_error:
            warnings.append(queue_error)
    else:
        notes.append("Slurm queries disabled")

    if len(recovery_records) > 1:
        errors.append(
            "multiple matching recovery jobs are active: "
            + ", ".join(record.job_id for record in recovery_records)
        )
    recovery_record = recovery_records[0] if len(recovery_records) == 1 else None
    if not args.no_slurm and recovery_record is None:
        warnings.append("no matching recovery job was found in squeue/sacct")
    if recovery_record is not None:
        if recovery_record.state in FAIL_STATES:
            errors.append(
                f"recovery job {recovery_record.job_id} is {recovery_record.state}"
            )
        elif recovery_record.state == "COMPLETED":
            stop = str(status.get("stop_reason") or "")
            if stop not in TERMINAL_STOPS:
                warnings.append(
                    f"recovery job completed but status is nonterminal ({stop!r})"
                )
            elif stop != "certified":
                warnings.append(
                    f"recovery completed without pricing certificate ({stop})"
                )
        elif recovery_record.state not in ACTIVE_STATES:
            warnings.append(
                f"recovery job has unrecognized state {recovery_record.state}"
            )

    task_records = array_task_records(array_records, args.array_job)
    state_counts = Counter(record.state for record in task_records.values())
    unsuccessful_tasks = unsuccessful_array_records(
        array_records, args.array_job
    )
    original_task = task_records.get(args.task)
    if array_records and original_task is None:
        warnings.append(
            f"sacct returned no JobID {args.array_job}_{args.task}; "
            "JobIDRaw was deliberately not used as an array identity"
        )

    artifacts: dict[str, Any] = {}
    rows: list[IterationRow] = []
    trend = None
    snapshot = None
    if result_path is not None:
        paths = {
            "status": result_path,
            "journal": Path(str(result_path) + ".columns.jsonl"),
            "iters": Path(str(result_path) + ".iters.csv"),
            "attestation": Path(str(result_path) + ".migration_attestation.json"),
        }
        artifacts = {name: file_info(path, now) for name, path in paths.items()}
        iters_path = paths["iters"]
        if iters_path.is_file():
            try:
                rows = parse_iteration_rows(tail_lines(iters_path, args.trend_rows))
                trend = iteration_trend(rows, args.trend_minutes)
                if not rows:
                    warnings.append("iteration CSV has no parseable data rows in its tail")
            except OSError as exc:
                errors.append(f"iteration CSV unreadable: {exc}")
        else:
            warnings.append("iteration CSV is missing")
        snapshot = latest_snapshot(result_path)

    latest = asdict(rows[-1]) if rows else None
    if recovery_record is not None and recovery_record.state == "RUNNING":
        iters_age = (artifacts.get("iters") or {}).get("age_s")
        if iters_age is None:
            warnings.append("running recovery has no iteration heartbeat")
        elif iters_age > args.stale_minutes * 60:
            warnings.append(
                f"iteration heartbeat is stale ({human_age(iters_age)} old)"
            )
        if trend is not None and trend["iterations_per_hour"] <= 0:
            warnings.append("recent iteration rate is not positive")

    stop_reason = str(status.get("stop_reason") or "") if status else ""
    if stop_reason == "running" and recovery_record is None and not args.no_slurm:
        warnings.append("status says running but no recovery allocation is active")
    if stop_reason in {"master_failed", "no_path"}:
        warnings.append(f"pricing stopped with {stop_reason}")

    log_dir = root / "src" / "cluster_logs" / "legacy_recovery" / f"job{args.array_job}"
    log_errors = scan_log_errors(log_dir, recovery_job)
    if log_errors:
        errors.extend(f"recovery log: {item}" for item in log_errors)

    verdict = "FAIL" if errors else ("WARN" if warnings else "HEALTHY")
    return {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "verdict": verdict,
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "root": str(root),
        "result": str(result_path) if result_path else None,
        "array_job": args.array_job,
        "task": args.task,
        "array_state_counts": dict(sorted(state_counts.items())),
        "original_unsuccessful_tasks": unsuccessful_tasks,
        "original_task": asdict(original_task) if original_task else None,
        "recovery_job": asdict(recovery_record) if recovery_record else None,
        "recovery_candidates": [asdict(record) for record in recovery_records],
        "status_error": status_error,
        "status": {
            "stop_reason": status.get("stop_reason"),
            "certified_rc_optimal": status.get("certified_rc_optimal"),
            "iterations": status.get("iterations"),
            "attempt_iterations": status.get("attempt_iterations"),
            "columns": status.get("columns"),
            "wall_s": status.get("wall_s"),
            "attempt_wall_s": status.get("attempt_wall_s"),
            "final": status.get("final"),
            "final_lp_source": status.get("final_lp_source"),
        }
        if status
        else None,
        "attestation": attestation,
        "artifacts": artifacts,
        "snapshot": snapshot,
        "latest_iteration": latest,
        "trend": trend,
        "log_errors": log_errors,
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        f"EVSP-DR legacy recovery monitor  {report['generated_at']}",
        f"result: {report['result'] or 'NOT FOUND'}",
        "",
        f"=== original cg-bigtar array {report['array_job']} ===",
    ]
    counts = report["array_state_counts"]
    lines.append(
        "aggregate: "
        + (" ".join(f"{key}={value}" for key, value in counts.items()) if counts else "unavailable")
    )
    original = report.get("original_task")
    if original:
        lines.append(
            f"task {report['task']}: JobID={original['job_id']} "
            f"JobIDRaw={original['job_id_raw']} state={original['state']} "
            f"elapsed={original['elapsed']} exit={original['exit_code']}"
        )
    else:
        lines.append(f"task {report['task']}: no sacct array-task record")
    unsuccessful = report.get("original_unsuccessful_tasks") or []
    if unsuccessful:
        lines.append(f"unsuccessful original tasks ({len(unsuccessful)}):")
        for item in unsuccessful:
            lines.append(
                f"  task={item['task']} JobID={item['job_id']} "
                f"JobIDRaw={item['job_id_raw']} state={item['state']} "
                f"elapsed={item['elapsed']} exit={item['exit_code']}"
            )
    else:
        lines.append("unsuccessful original tasks: none reported")

    lines.extend(["", "=== recovery allocation ==="])
    recovery = report.get("recovery_job")
    if recovery:
        place = recovery.get("node") or recovery.get("reason") or "NA"
        lines.append(
            f"job={recovery['job_id']} name={recovery['job_name']} "
            f"state={recovery['state']} elapsed={recovery['elapsed']} place={place}"
        )
    else:
        lines.append("no single matching recovery allocation")

    lines.extend(["", "=== migration attestation ==="])
    attestation = report.get("attestation")
    if attestation:
        label = "PASS" if attestation["ok"] else "FAIL"
        lines.append(
            f"{label} mode={attestation['mode']} "
            f"migration_tool_commit={attestation.get('tool_commit') or 'NA'}"
        )
        lines.append(
            "current_continuation_commit="
            f"{attestation.get('continuation_commit') or 'NA'}"
        )
        lines.append("checks: " + ", ".join(attestation.get("checks") or ["none"]))
        repairs = attestation.get("repairs") or {}
        for kind in ("journal", "iters"):
            repair = repairs.get(kind) or {}
            normalizations = repair.get("legacy_line_normalizations") or []
            lines.append(
                f"{kind} repair: applied={repair.get('applied')} "
                f"changed_from={repair.get('changed_from_byte')} "
                f"normalizations={len(normalizations)}"
            )
            for item in normalizations[:3]:
                lines.append(
                    f"  - {item.get('kind')} offset={item.get('original_offset')} "
                    f"recovered_objects={item.get('recovered_objects')}"
                )
    else:
        lines.append("not available")

    lines.extend(["", "=== durable artifacts ==="])
    for name in ("status", "journal", "iters", "attestation"):
        info = report.get("artifacts", {}).get(name)
        if not info:
            continue
        if not info.get("exists"):
            lines.append(f"{name:<12} MISSING  {info['path']}")
        else:
            lines.append(
                f"{name:<12} {human_bytes(info.get('bytes')):>10} "
                f"age={human_age(info.get('age_s')):>7} mtime={info.get('mtime')}"
            )

    status = report.get("status")
    if status:
        final = status.get("final") or {}
        lines.extend(
            [
                "",
                "=== latest persisted status ===",
                f"stop={status.get('stop_reason')} certified={status.get('certified_rc_optimal')} "
                f"iterations={status.get('iterations')} columns={status.get('columns')} "
                f"wall={fmt_number(status.get('wall_s'), 1)}s",
                f"status final: obj={fmt_number(final.get('lp_obj'), 2)} "
                f"weight={fmt_number(final.get('route_weight'), 4)} "
                f"art={fmt_number(final.get('artificials'), 2)} "
                f"min_rc={fmt_number(final.get('min_rc'), 3)}",
            ]
        )

    snapshot = report.get("snapshot")
    lines.extend(["", "=== latest immutable snapshot ==="])
    if snapshot:
        lines.append(
            f"m{snapshot['mark_minutes']:g} stop={snapshot['stop_reason']} "
            f"iterations={snapshot['iterations']} columns={snapshot['columns']} "
            f"wall={fmt_number(snapshot['wall_s'], 1)}s "
            f"journal={human_bytes(snapshot['journal_bytes'])}"
        )
        lines.append(
            f"snapshot final: obj={fmt_number(snapshot['lp_obj'], 2)} "
            f"weight={fmt_number(snapshot['route_weight'], 4)} "
            f"art={fmt_number(snapshot['artificials'], 2)} "
            f"min_rc={fmt_number(snapshot['min_rc'], 3)}"
        )
    else:
        lines.append("none discovered")

    latest = report.get("latest_iteration")
    lines.extend(["", "=== live iteration trajectory ==="])
    if latest:
        lines.append(
            f"latest: elapsed={latest['elapsed_s'] / 3600:.2f}h "
            f"it={latest['iteration']} columns={latest['pool_columns']} "
            f"obj={fmt_number(latest['lp_obj'], 2)} "
            f"weight={fmt_number(latest['route_weight'], 4)} "
            f"art={fmt_number(latest['artificials'], 2)} "
            f"min_rc={fmt_number(latest['min_rc'], 3)}"
        )
    else:
        lines.append("no parseable iteration row")
    trend = report.get("trend")
    if trend:
        lines.append(
            f"trend ({trend['minutes']:.1f}m, {trend['rows']} rows): "
            f"{trend['iterations_per_hour']:.2f} it/h; "
            f"objective drop={trend['objective_drop_per_hour']:,.2f}/h; "
            f"weight drop={trend['weight_drop_per_hour']:.5f}/h; "
            f"columns added={trend['columns_added_per_hour']:.1f}/h"
        )

    lines.extend(["", "=== verdict ===", report["verdict"]])
    for item in report.get("errors") or []:
        lines.append(f"FAIL: {item}")
    for item in report.get("warnings") or []:
        lines.append(f"WARN: {item}")
    for item in report.get("notes") or []:
        lines.append(f"NOTE: {item}")
    if report["verdict"] == "HEALTHY":
        lines.append("migration identity is intact and the recovery heartbeat is current")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=script_root)
    parser.add_argument("--array-job", type=int, default=867334)
    parser.add_argument("--task", type=int, default=22)
    parser.add_argument("--result", type=Path, help="Pin a canonical recovery result JSON")
    parser.add_argument("--recovery-job", help="Pin the continuation Slurm job id")
    parser.add_argument("--user", default=os.environ.get("USER") or getpass.getuser())
    parser.add_argument(
        "--attestation-mode",
        choices=("structural", "prefix", "full"),
        default="prefix",
        help="prefix verifies immutable migrated prefixes; full also hashes the raw archive",
    )
    parser.add_argument("--trend-minutes", type=float, default=60.0)
    parser.add_argument("--trend-rows", type=int, default=500)
    parser.add_argument("--stale-minutes", type=float, default=60.0)
    parser.add_argument("--no-slurm", action="store_true", help="Skip squeue/sacct queries")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON")
    args = parser.parse_args(argv)
    if args.task <= 0 or args.array_job <= 0:
        parser.error("--task and --array-job must be positive")
    if args.trend_minutes <= 0 or args.trend_rows < 2 or args.stale_minutes <= 0:
        parser.error("trend/staleness values must be positive and trend rows >= 2")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(render_report(report))
    return 2 if report["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

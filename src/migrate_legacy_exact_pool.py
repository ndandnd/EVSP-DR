"""Prepare an attested, copy-only resume from a legacy exact-CG checkpoint.

Checkpoints written before the durable-resume work did not put input hashes in
periodic status JSONs.  They must never be resumed in place or granted a broad
"trust legacy" exception.  This tool reconstructs input identity from
independent hashed statuses, archives the source bytes, repairs only working
copies, and publishes a new resume status last as the migration commit marker.

The reconstructed hashes prove that the *current* source and destination input
bytes agree with authenticated witnesses.  They do not retroactively prove
that the legacy checkpoint itself recorded those hashes; that limitation is
stored in every attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from durable_io import (
    atomic_copy,
    atomic_write_bytes,
    atomic_write_json,
    exclusive_output_lock,
    read_jsonl_records,
)
from exact_pricer_expanded import load_column_pool, load_iteration_log


SRC = Path(__file__).resolve().parent
REPO_ROOT = SRC.parent
SCHEMA = "evsp-dr-legacy-exact-pool-migration-v1"
MODEL_FIELDS = (
    "soc_step", "block_min", "g_kwh", "charge_kw", "min_soc_frac",
    "master_sense",
)
TERMINAL_WITNESS_STOPS = {
    "certified", "wall_limit", "max_iters", "no_path",
    "degenerate_stall", "stalled_marginal_returns", "master_failed",
}


class MigrationError(ValueError):
    """Raised when legacy evidence is insufficient or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json_bytes(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise MigrationError(f"JSON payload is not an object: {path}")
    return payload, raw


def _safe_data_path(root: Path, relative: str, label: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise MigrationError(f"{label} escapes data directory: {relative}") from exc
    if not candidate.is_file():
        raise MigrationError(f"{label} is missing: {candidate}")
    return candidate


def _model_matches(candidate: dict, legacy: dict) -> bool:
    for field in MODEL_FIELDS:
        left = candidate.get(field)
        right = legacy.get(field)
        if isinstance(right, float):
            try:
                if not math.isclose(
                        float(left), right, rel_tol=0.0, abs_tol=1e-9):
                    return False
            except (TypeError, ValueError):
                return False
        elif left != right:
            return False
    return True


def _witness_is_stable_status(status: dict) -> bool:
    stop = str(status.get("stop_reason") or "")
    if stop.startswith("snapshot_m"):
        return bool(status.get("snapshot_mark_minutes") is not None)
    return stop in TERMINAL_WITNESS_STOPS


def discover_witness(
    roots: list[Path],
    legacy: dict,
    *,
    path_field: str,
    hash_field: str,
    expected_hash: str,
    require_trip_ids: bool,
) -> dict:
    """Find agreeing authenticated statuses for one independently hashed input."""

    relative = legacy.get(path_field)
    candidates = []
    observed_hashes = set()
    seen_paths = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                status, raw = _load_json_bytes(path)
            except (OSError, MigrationError):
                continue
            provenance = status.get("provenance") or {}
            if not isinstance(provenance, dict):
                continue
            saved_hash = provenance.get(hash_field)
            if (status.get(path_field) != relative
                    or not isinstance(saved_hash, str)
                    or len(saved_hash) != 64
                    or not _witness_is_stable_status(status)
                    or not _model_matches(status, legacy)):
                continue
            if require_trip_ids and status.get("trip_ids") != legacy.get("trip_ids"):
                continue
            observed_hashes.add(saved_hash)
            candidates.append({
                "path": str(path.resolve()),
                "status_sha256": _sha256_bytes(raw),
                "input_sha256": saved_hash,
                "stop_reason": status.get("stop_reason"),
                "git_commit": provenance.get("git_commit"),
            })
    if len(observed_hashes) > 1:
        raise MigrationError(
            f"conflicting {hash_field} witnesses for {relative}: "
            f"{sorted(observed_hashes)}"
        )
    matching = [
        candidate for candidate in candidates
        if candidate["input_sha256"] == expected_hash
    ]
    if not matching:
        raise MigrationError(
            f"no authenticated {hash_field} witness for {relative} with "
            f"current hash {expected_hash}"
        )
    matching.sort(key=lambda value: value["path"])
    return {
        "selected": matching[0],
        "agreeing_witnesses": matching,
        "sha256": expected_hash,
    }


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _tool_identity() -> dict:
    commit = _git_value("rev-parse", "HEAD")
    tracked_status = _git_value(
        "status", "--porcelain", "--untracked-files=no"
    )
    if (commit is None or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise MigrationError(
            "migration tool has no verifiable 40-character Git HEAD"
        )
    if tracked_status is None:
        raise MigrationError("could not verify migration tool worktree state")
    if tracked_status:
        raise MigrationError(
            "migration tool has tracked changes; commit and use a clean "
            "checkout before attesting a continuation"
        )
    return {
        "commit": commit,
        "branch": _git_value("branch", "--show-current"),
        "dirty": False,
    }


def _common_prefix_size(left: bytes, right: bytes) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _repair_report(original: bytes, repaired: bytes) -> dict:
    common = _common_prefix_size(original, repaired)
    changed_tail = original[common:]
    return {
        "applied": original != repaired,
        "original_size": len(original),
        "repaired_size": len(repaired),
        "changed_from_byte": common if original != repaired else None,
        "original_tail_bytes": len(changed_tail),
        "original_tail_sha256": (
            _sha256_bytes(changed_tail) if changed_tail else None
        ),
        "original_sha256": _sha256_bytes(original),
        "repaired_sha256": _sha256_bytes(repaired),
    }


def _repair_and_validate_working_copies(
    journal: Path, iters: Path, trip_ids: list[int],
) -> dict:
    """Repair archived working copies and validate their usable prefix."""

    journal_before = journal.read_bytes()
    records = read_jsonl_records(
        journal,
        repair_trailing=True,
        allow_unparseable_trailing=True,
    )
    pool = load_column_pool(records, trip_ids)
    journal_after = journal.read_bytes()

    iters_before = iters.read_bytes()
    iteration_rows = load_iteration_log(iters, repair_trailing=True)
    elapsed_values = [float(row[0]) for row in iteration_rows]
    if any(right + 1e-9 < left for left, right in zip(
            elapsed_values, elapsed_values[1:])):
        raise MigrationError("legacy iteration elapsed_s is not monotone")
    iters_after = iters.read_bytes()
    return {
        "records": records,
        "pool": pool,
        "iteration_rows": iteration_rows,
        "elapsed_values": elapsed_values,
        "journal_before": journal_before,
        "journal_after": journal_after,
        "iters_before": iters_before,
        "iters_after": iters_after,
        "preview": {
            "journal": {
                **_repair_report(journal_before, journal_after),
                "complete_records": len(records),
                "unique_incidences": len(pool),
            },
            "iters": {
                **_repair_report(iters_before, iters_after),
                "valid_rows": len(iteration_rows),
            },
        },
    }


def _preview_legacy_repair(
    source_journal: Path, source_iters: Path, trip_ids: list[int],
) -> dict:
    """Exercise the full repair on temporary copies without changing inputs."""

    with tempfile.TemporaryDirectory(prefix="evsp_legacy_preview_") as tmp:
        root = Path(tmp)
        journal = root / "working.columns.jsonl"
        iters = root / "working.iters.csv"
        atomic_copy(source_journal, journal)
        atomic_copy(source_iters, iters)
        return _repair_and_validate_working_copies(
            journal, iters, trip_ids
        )["preview"]


def _migration_id(plan: dict) -> str:
    identity = {
        "source_hashes": plan["source_hashes"],
        "source_log_hashes": [
            log["sha256"] for log in plan.get("source_logs") or []
        ],
        "instance_hash": plan["instance_hash"],
        "prices_hash": plan["prices_hash"],
        "legacy_commit_claim": plan.get("legacy_commit_claim"),
        "tool_commit": plan["tool_commit"],
    }
    digest = _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )[:20]
    return (
        f"legacy-{plan.get('slurm_array_job') or 'unknown'}-"
        f"{plan.get('slurm_task') or 'unknown'}-{digest}"
    )


def _prefix_sha256(path: Path, size: int, *, label: str) -> str:
    digest = hashlib.sha256()
    remaining = size
    with open(path, "rb") as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise MigrationError(
                    f"destination {label} is shorter than its migration "
                    f"prefix: {path}"
                )
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _existing_destination_attestation(
    destination: Path, plan: dict,
) -> dict | None:
    """Return a fully validated existing attestation, or fail closed."""

    journal = Path(str(destination) + ".columns.jsonl")
    iters = Path(str(destination) + ".iters.csv")
    attestation_path = Path(str(destination) + ".migration_attestation.json")
    raw_dir = destination.parent / f"{destination.name}.legacy_raw"
    siblings = [journal, iters, attestation_path, raw_dir]
    if not destination.is_file():
        orphans = [str(path) for path in siblings if path.exists()]
        if orphans:
            raise MigrationError(
                "destination status is missing but migration artifacts exist; "
                "preserve them and use a new output path: " + ", ".join(orphans)
            )
        return None
    status, _ = _load_json_bytes(destination)
    parent = status.get("resume_parent") or {}
    hashes = plan["source_hashes"]
    if (parent.get("schema") != SCHEMA
            or parent.get("migration_id") != plan["migration_id"]
            or parent.get("source_status_sha256") != hashes["status"]
            or parent.get("source_journal_sha256") != hashes["journal"]
            or parent.get("source_iters_sha256") != hashes["iters"]
            or parent.get("tool_commit") != plan["tool_commit"]):
        raise MigrationError(
            f"existing destination belongs to different or unidentified work: "
            f"{destination}"
        )
    provenance = status.get("provenance") or {}
    if (not isinstance(provenance, dict)
            or provenance.get("git_commit") != plan["tool_commit"]
            or provenance.get("instance_sha256") != plan["instance_hash"]
            or provenance.get("prices_sha256") != plan["prices_hash"]):
        raise MigrationError(
            f"existing destination has incompatible provenance: {destination}"
        )
    if not journal.is_file() or not iters.is_file():
        raise MigrationError(
            f"existing migrated status has lost its journal or iterations: "
            f"{journal}, {iters}"
        )
    try:
        journal_prefix_size = int(parent["migrated_prefix_bytes"])
        journal_prefix_hash = str(parent["migrated_prefix_sha256"])
        iters_prefix_size = int(parent["migrated_iters_prefix_bytes"])
        iters_prefix_hash = str(parent["migrated_iters_prefix_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MigrationError(
            "existing migration has no verifiable journal/iteration prefix"
        ) from exc
    if (_prefix_sha256(
            journal, journal_prefix_size, label="journal")
            != journal_prefix_hash):
        raise MigrationError(
            f"existing destination journal no longer has its migrated source prefix: "
            f"{journal}"
        )
    if (_prefix_sha256(
            iters, iters_prefix_size, label="iteration log")
            != iters_prefix_hash):
        raise MigrationError(
            f"existing destination iteration log no longer has its migrated "
            f"source prefix: {iters}"
        )
    if not attestation_path.is_file():
        raise MigrationError(
            f"existing migrated status has lost its attestation: "
            f"{attestation_path}"
        )
    attestation, _ = _load_json_bytes(attestation_path)
    source_attestation = attestation.get("source") or {}
    if (attestation.get("schema") != SCHEMA
            or attestation.get("migration_id") != plan["migration_id"]
            or source_attestation.get("result_sha256") != hashes["status"]
            or source_attestation.get("journal_sha256") != hashes["journal"]
            or source_attestation.get("iters_sha256") != hashes["iters"]
            or source_attestation.get("logs")
            != (plan.get("source_logs") or [])):
        raise MigrationError(
            f"existing migration attestation does not match its source: "
            f"{attestation_path}"
        )
    raw_expected = {
        "source_result.json": hashes["status"],
        "source_result.json.columns.jsonl": hashes["journal"],
        "source_result.json.iters.csv": hashes["iters"],
    }
    raw_expected.update({
        log["archive_name"]: log["sha256"]
        for log in plan.get("source_logs") or []
    })
    for name, expected_hash in raw_expected.items():
        path = raw_dir / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise MigrationError(
                f"raw migration archive is missing or corrupt: {path}"
            )
    raw_manifest = raw_dir / "raw_manifest.json"
    if not raw_manifest.is_file():
        raise MigrationError(f"raw migration manifest is missing: {raw_manifest}")
    manifest, _ = _load_json_bytes(raw_manifest)
    if manifest != attestation:
        raise MigrationError(
            f"raw migration manifest differs from attestation: {raw_manifest}"
        )
    repairs = attestation.get("repairs") or {}
    for label, filename in (
            ("journal", "journal_changed_tail.bin"),
            ("iters", "iters_changed_tail.bin")):
        repair = repairs.get(label) or {}
        if not repair.get("applied"):
            continue
        tail = raw_dir / filename
        if (not tail.is_file()
                or tail.stat().st_size != repair.get("original_tail_bytes")
                or sha256_file(tail) != repair.get("original_tail_sha256")):
            raise MigrationError(
                f"raw migration repair quarantine is missing or corrupt: {tail}"
            )
    return attestation


def _copy_source_verified(source: Path, destination: Path, expected_hash: str) -> None:
    atomic_copy(source, destination)
    if sha256_file(destination) != expected_hash:
        raise MigrationError(f"copied bytes do not match source hash: {source}")
    try:
        if os.path.samefile(source, destination):
            raise MigrationError(
                f"migration copy unexpectedly aliases its source inode: {source}"
            )
    except FileNotFoundError as exc:
        raise MigrationError("source or copied artifact disappeared") from exc


def build_migration_plan(
    *,
    source_result: Path,
    destination: Path,
    source_data_dir: Path,
    destination_data_dir: Path,
    witness_roots: list[Path],
    source_logs: list[Path],
    legacy_commit: str | None,
    slurm_array_job: str | None,
    slurm_task: str | None,
) -> dict:
    source_result = source_result.resolve()
    destination = destination.resolve()
    if source_result == destination:
        raise MigrationError("destination must not overwrite the legacy status")
    if not source_result.is_file():
        raise MigrationError(f"legacy status is missing: {source_result}")
    source_journal = Path(str(source_result) + ".columns.jsonl")
    source_iters = Path(str(source_result) + ".iters.csv")
    for label, path in (("journal", source_journal), ("iteration log", source_iters)):
        if not path.is_file():
            raise MigrationError(f"legacy {label} is missing: {path}")

    legacy, status_raw = _load_json_bytes(source_result)
    required = ("csv", "prices_csv", "trip_ids", *MODEL_FIELDS)
    missing = [key for key in required if legacy.get(key) is None]
    if missing:
        raise MigrationError(f"legacy status lacks required fields: {missing}")
    if not isinstance(legacy.get("trip_ids"), list):
        raise MigrationError("legacy trip_ids must be a list")

    source_instance = _safe_data_path(
        source_data_dir, str(legacy["csv"]), "source instance"
    )
    destination_instance = _safe_data_path(
        destination_data_dir, str(legacy["csv"]), "destination instance"
    )
    source_prices = _safe_data_path(
        source_data_dir, str(legacy["prices_csv"]), "source prices"
    )
    destination_prices = _safe_data_path(
        destination_data_dir, str(legacy["prices_csv"]), "destination prices"
    )
    instance_hashes = {sha256_file(source_instance), sha256_file(destination_instance)}
    price_hashes = {sha256_file(source_prices), sha256_file(destination_prices)}
    if len(instance_hashes) != 1:
        raise MigrationError("source and destination instance bytes differ")
    if len(price_hashes) != 1:
        raise MigrationError("source and destination price bytes differ")
    instance_hash = next(iter(instance_hashes))
    prices_hash = next(iter(price_hashes))

    instance_witness = discover_witness(
        witness_roots, legacy,
        path_field="csv", hash_field="instance_sha256",
        expected_hash=instance_hash, require_trip_ids=True,
    )
    prices_witness = discover_witness(
        witness_roots, legacy,
        path_field="prices_csv", hash_field="prices_sha256",
        expected_hash=prices_hash, require_trip_ids=False,
    )
    source_hashes = {
        "status": _sha256_bytes(status_raw),
        "journal": sha256_file(source_journal),
        "iters": sha256_file(source_iters),
    }
    log_records = []
    for index, path in enumerate(source_logs, start=1):
        path = path.resolve()
        if not path.is_file():
            raise MigrationError(f"requested source log is missing: {path}")
        suffix = path.suffix or ".log"
        log_records.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "archive_name": f"source_log_{index}{suffix}",
        })
    repair_preview = _preview_legacy_repair(
        source_journal, source_iters, list(legacy["trip_ids"])
    )
    if ({
            "status": sha256_file(source_result),
            "journal": sha256_file(source_journal),
            "iters": sha256_file(source_iters),
    } != source_hashes):
        raise MigrationError("legacy source changed during read-only preflight")
    for log in log_records:
        if sha256_file(Path(log["path"])) != log["sha256"]:
            raise MigrationError(
                f"legacy source log changed during read-only preflight: "
                f"{log['path']}"
            )
    tool = _tool_identity()
    plan = {
        "schema": SCHEMA,
        "source_result": str(source_result),
        "source_journal": str(source_journal),
        "source_iters": str(source_iters),
        "destination": str(destination),
        "source_hashes": source_hashes,
        "source_logs": log_records,
        "repair_preview": repair_preview,
        "legacy": legacy,
        "instance_hash": instance_hash,
        "prices_hash": prices_hash,
        "instance_witness": instance_witness,
        "prices_witness": prices_witness,
        "legacy_commit_claim": legacy_commit,
        "slurm_array_job": slurm_array_job,
        "slurm_task": slurm_task,
        "tool_commit": tool["commit"],
        "tool_branch": tool["branch"],
        "tool_dirty": tool["dirty"],
    }
    plan["migration_id"] = _migration_id(plan)
    return plan


def apply_migration(plan: dict) -> dict:
    destination = Path(plan["destination"])
    metadata = {
        "operation": "legacy_exact_pool_migration",
        "pid": os.getpid(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "migration_id": plan.get("migration_id"),
        "tool_commit": plan.get("tool_commit"),
    }
    with exclusive_output_lock(destination, metadata):
        return _apply_migration_locked(plan)


def _apply_migration_locked(plan: dict) -> dict:
    source_result = Path(plan["source_result"])
    source_journal = Path(plan["source_journal"])
    source_iters = Path(plan["source_iters"])
    destination = Path(plan["destination"])
    destination_journal = Path(str(destination) + ".columns.jsonl")
    destination_iters = Path(str(destination) + ".iters.csv")
    attestation_path = Path(str(destination) + ".migration_attestation.json")
    raw_dir = destination.parent / f"{destination.name}.legacy_raw"
    hashes = plan["source_hashes"]

    existing = _existing_destination_attestation(destination, plan)
    if existing is not None:
        print(f"[MIGRATE] already prepared: {destination}")
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.migration.partial.",
        dir=destination.parent,
    ))
    try:
        stage_raw = stage / "raw"
        stage_raw.mkdir()
        raw_status = stage_raw / "source_result.json"
        raw_journal = stage_raw / "source_result.json.columns.jsonl"
        raw_iters = stage_raw / "source_result.json.iters.csv"
        _copy_source_verified(source_result, raw_status, hashes["status"])
        _copy_source_verified(source_journal, raw_journal, hashes["journal"])
        _copy_source_verified(source_iters, raw_iters, hashes["iters"])
        for log in plan.get("source_logs") or []:
            source_log = Path(log["path"])
            _copy_source_verified(
                source_log, stage_raw / log["archive_name"],
                log["sha256"],
            )
        if ({
                "status": sha256_file(source_result),
                "journal": sha256_file(source_journal),
                "iters": sha256_file(source_iters),
        } != hashes):
            raise MigrationError("legacy source changed while it was being copied")
        for log in plan.get("source_logs") or []:
            if sha256_file(Path(log["path"])) != log["sha256"]:
                raise MigrationError(
                    f"legacy source log changed while it was being copied: "
                    f"{log['path']}"
                )

        working_journal = stage / "working.columns.jsonl"
        working_iters = stage / "working.iters.csv"
        atomic_copy(raw_journal, working_journal)
        atomic_copy(raw_iters, working_iters)
        validation = _repair_and_validate_working_copies(
            working_journal,
            working_iters,
            list(plan["legacy"]["trip_ids"]),
        )
        if validation["preview"] != plan["repair_preview"]:
            raise MigrationError(
                "legacy repair outcome changed between preflight and apply"
            )
        records = validation["records"]
        pool = validation["pool"]
        iteration_rows = validation["iteration_rows"]
        elapsed_values = validation["elapsed_values"]
        journal_before = validation["journal_before"]
        journal_after = validation["journal_after"]
        iters_before = validation["iters_before"]
        iters_after = validation["iters_after"]
        journal_repair = validation["preview"]["journal"]
        iters_repair = validation["preview"]["iters"]
        if journal_repair["applied"]:
            common = int(journal_repair["changed_from_byte"])
            atomic_write_bytes(
                stage_raw / "journal_changed_tail.bin",
                journal_before[common:],
            )
        if iters_repair["applied"]:
            common = int(iters_repair["changed_from_byte"])
            atomic_write_bytes(
                stage_raw / "iters_changed_tail.bin",
                iters_before[common:],
            )

        legacy = plan["legacy"]
        saved_wall = float(legacy.get("wall_s") or 0.0)
        last_elapsed = elapsed_values[-1] if elapsed_values else 0.0
        saved_iterations = int(legacy.get("iterations") or 0)
        logged_iterations = [int(float(row[1])) for row in iteration_rows]
        cumulative_iterations = max(
            [saved_iterations, len(iteration_rows), *logged_iterations]
        )
        wall_s = max(saved_wall, last_elapsed)
        migration_id = plan["migration_id"]
        resume_parent = {
            "schema": SCHEMA,
            "migration_id": migration_id,
            "source_status": str(source_result),
            "source_status_sha256": hashes["status"],
            "source_journal": str(source_journal),
            "source_journal_sha256": hashes["journal"],
            "source_iters": str(source_iters),
            "source_iters_sha256": hashes["iters"],
            "migrated_prefix_bytes": len(journal_after),
            "migrated_prefix_sha256": _sha256_bytes(journal_after),
            "migrated_iters_prefix_bytes": len(iters_after),
            "migrated_iters_prefix_sha256": _sha256_bytes(iters_after),
            "legacy_generation_commit_claim": plan.get("legacy_commit_claim"),
            "tool_commit": plan.get("tool_commit"),
            "instance_witness": plan["instance_witness"]["selected"],
            "prices_witness": plan["prices_witness"]["selected"],
            "input_hash_origin": "reconstructed_split_status_witness",
            "limitation": (
                "The legacy checkpoint omitted input hashes; this migration "
                "proves current input bytes against independent witnesses, "
                "not a contemporaneous hash stored by the legacy checkpoint."
            ),
        }
        legacy_provenance = legacy.get("provenance") or {}
        if not isinstance(legacy_provenance, dict):
            legacy_provenance = {}
        provenance_args = {
            "csv": legacy["csv"],
            "prices_csv": legacy["prices_csv"],
            **{field: legacy[field] for field in MODEL_FIELDS},
            "rc_eps": (legacy_provenance.get("rc_eps") or 1e-4),
            "out": str(destination),
            "resume": True,
        }
        provenance = {
            "git_commit": plan.get("tool_commit"),
            "git_branch": plan.get("tool_branch"),
            "git_dirty": plan.get("tool_dirty"),
            "python": platform.python_version(),
            "instance_sha256": plan["instance_hash"],
            "prices_sha256": plan["prices_hash"],
            "rc_eps": provenance_args["rc_eps"],
            "args": provenance_args,
            "input_hash_origin": "reconstructed_split_status_witness",
        }
        migrated = dict(legacy)
        migrated.update({
            "iterations": cumulative_iterations,
            "attempt_iterations": 0,
            "certified_rc_optimal": False,
            "columns": len(pool),
            "columns_journal": str(destination_journal),
            "wall_s": wall_s,
            "attempt_wall_s": 0.0,
            "stop_reason": "prepared_legacy_resume",
            "final_lp": None,
            "final_lp_source": None,
            "provenance": provenance,
            "resume_parent": resume_parent,
        })
        attestation = {
            "schema": SCHEMA,
            "migration_id": migration_id,
            "tool": {
                "commit": plan.get("tool_commit"),
                "branch": plan.get("tool_branch"),
                "dirty": plan.get("tool_dirty"),
                "python": platform.python_version(),
            },
            "source": {
                "result": str(source_result),
                "result_sha256": hashes["status"],
                "journal": str(source_journal),
                "journal_sha256": hashes["journal"],
                "iters": str(source_iters),
                "iters_sha256": hashes["iters"],
                "legacy_generation_commit_claim": plan.get("legacy_commit_claim"),
                "slurm_array_job": plan.get("slurm_array_job"),
                "slurm_task": plan.get("slurm_task"),
                "logs": plan.get("source_logs") or [],
            },
            "repairs": {
                "journal": journal_repair,
                "iters": iters_repair,
            },
            "input_attestation": {
                "method": "split_status_witness",
                "instance": plan["instance_witness"],
                "prices": plan["prices_witness"],
                "limitation": resume_parent["limitation"],
            },
            "validation": {
                "source_unchanged_during_copy": True,
                "copy_has_distinct_inode": True,
                "pricing_record_fields_valid": True,
                "trip_ids_witnessed": True,
                "elapsed_time_monotone": True,
            },
            "destination": {
                "result": str(destination),
                "journal": str(destination_journal),
                "journal_initial_sha256": _sha256_bytes(journal_after),
                "journal_initial_bytes": len(journal_after),
                "iters": str(destination_iters),
                "iters_initial_sha256": _sha256_bytes(iters_after),
                "iters_initial_bytes": len(iters_after),
            },
            "timing_scope": (
                "Mixed-code continuation. Do not use the combined wall-time "
                "trajectory as a clean single-version timing benchmark."
            ),
        }
        atomic_write_json(stage_raw / "raw_manifest.json", attestation)

        os.replace(stage_raw, raw_dir)
        atomic_copy(working_journal, destination_journal)
        atomic_copy(working_iters, destination_iters)
        atomic_write_json(attestation_path, attestation)
        # Status is the commit marker.  Exact pricing cannot start until this
        # final atomic publication establishes identity for the copied pool.
        atomic_write_json(destination, migrated)
        print(
            f"[MIGRATE] prepared {destination}: {len(records)} records, "
            f"{len(pool)} unique incidences, journal repair="
            f"{journal_repair['applied']}, iters repair={iters_repair['applied']}"
        )
        return attestation
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--destination-data-dir", type=Path, required=True)
    parser.add_argument(
        "--witness-root", type=Path, action="append", required=True,
        help="Result tree to scan; repeat for multiple trees.",
    )
    parser.add_argument(
        "--source-log", type=Path, action="append", default=[],
        help="Optional legacy stdout/stderr to archive; repeat as needed.",
    )
    parser.add_argument("--legacy-commit", default=None)
    parser.add_argument("--slurm-array-job", default=None)
    parser.add_argument("--slurm-task", default=None)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the isolated migration; otherwise print a read-only plan.",
    )
    args = parser.parse_args(argv)
    try:
        plan = build_migration_plan(
            source_result=args.source_result,
            destination=args.destination,
            source_data_dir=args.source_data_dir,
            destination_data_dir=args.destination_data_dir,
            witness_roots=args.witness_root,
            source_logs=args.source_log,
            legacy_commit=args.legacy_commit,
            slurm_array_job=args.slurm_array_job,
            slurm_task=args.slurm_task,
        )
        if args.apply:
            result = apply_migration(plan)
        else:
            result = {
                key: value for key, value in plan.items() if key != "legacy"
            }
            result["mode"] = "read_only_plan"
        print(json.dumps(result, indent=2))
        return 0
    except (MigrationError, OSError, ValueError) as exc:
        print(f"[MIGRATE] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Portable no-clobber bundles committed by ``completion.json``.

The destination directory is reserved atomically with ``mkdir``. Members are
written through same-directory temporary files and fsynced. ``completion.json``
is published last and is the only commit marker consumers may trust.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import shutil
import tempfile
import stat
from pathlib import Path
from typing import Callable


COMPLETION_SCHEMA = "evsp-dr-portable-bundle-completion-v1"


class BundlePublicationError(RuntimeError):
    pass


class BundleExistsError(FileExistsError):
    pass


class IncompleteBundleError(BundlePublicationError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_member(name: str) -> Path:
    path = Path(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 1
        or path.name == "completion.json"
    ):
        raise ValueError(f"unsafe/reserved bundle member: {name!r}")
    return path


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _reject_symlink_components(path: Path, *, allow_missing_leaf: bool) -> None:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for index, part in enumerate(absolute.parts[1:]):
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if allow_missing_leaf or index < len(absolute.parts[1:]) - 1:
                continue
            raise
        if stat.S_ISLNK(mode):
            raise BundlePublicationError(
                f"symlinked path component is forbidden: {current}"
            )


def _atomic_member(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(6)}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if (
                not path.is_file()
                or path.is_symlink()
                or sha256_file(path)
                != hashlib.sha256(payload).hexdigest()
                or path.stat().st_size != len(payload)
            ):
                raise FileExistsError(
                    f"bundle member already differs/exists: {path}"
                )
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_new_file(path: Path, payload: bytes) -> None:
    destination = _absolute_lexical(path)
    _reject_symlink_components(destination.parent, allow_missing_leaf=False)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{secrets.token_hex(6)}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _completion_bytes(members: dict[str, bytes], metadata: dict) -> bytes:
    payload = {
        "schema": COMPLETION_SCHEMA,
        "protocol": {
            "destination_reserved_by": "mkdir",
            "member_publication": "same-directory-temp-plus-rename",
            "commit_marker": "completion.json-published-last",
            "renameat2_required": False,
        },
        "members": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
            for name, content in sorted(members.items())
        },
        "metadata": metadata,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def inspect_bundle(
    destination: Path,
    *,
    required_members: tuple[str, ...] = (),
    recoverable_validator: Callable[[dict], None] | None = None,
    result_member: str = "result.json",
) -> dict:
    path = _absolute_lexical(Path(destination))
    try:
        _reject_symlink_components(path, allow_missing_leaf=True)
    except (BundlePublicationError, OSError) as exc:
        return {
            "state": "invalid",
            "path": str(path),
            "recoverable": False,
            "errors": [str(exc)],
        }
    if not path.exists():
        return {
            "state": "missing",
            "path": str(path),
            "recoverable": False,
            "errors": [],
        }
    try:
        destination_mode = os.lstat(path).st_mode
    except OSError as exc:
        return {
            "state": "invalid",
            "path": str(path),
            "recoverable": False,
            "errors": [str(exc)],
        }
    if not stat.S_ISDIR(destination_mode) or stat.S_ISLNK(destination_mode):
        return {
            "state": "invalid",
            "path": str(path),
            "recoverable": False,
            "errors": ["destination is not a regular directory"],
        }
    completion_path = path / "completion.json"
    if completion_path.is_symlink():
        return {
            "state": "invalid",
            "path": str(path),
            "recoverable": False,
            "errors": ["completion.json is symlinked"],
        }
    if not completion_path.is_file():
        corruption_errors = []
        missing_errors = []
        result_payload = None
        candidate = path / result_member
        if candidate.exists() and candidate.is_symlink():
            corruption_errors.append("result member is symlinked")
        elif candidate.is_file():
            try:
                result_payload = json.loads(candidate.read_text())
                if not isinstance(result_payload, dict):
                    raise ValueError("result member is not an object")
                if recoverable_validator is not None:
                    recoverable_validator(result_payload)
            except (OSError, ValueError, TypeError) as exc:
                corruption_errors.append(str(exc))
        missing_required = [
            name for name in required_members if not (path / name).is_file()
        ]
        if missing_required:
            missing_errors.append(
                "missing required members: " + ", ".join(missing_required)
            )
        recoverable = (
            result_payload is not None
            and not corruption_errors
            and not missing_errors
            and recoverable_validator is not None
        )
        candidate_invalid = candidate.exists() and bool(corruption_errors)
        return {
            "state": (
                "recoverable_validated"
                if recoverable
                else ("invalid" if candidate_invalid else "incomplete_publication")
            ),
            "path": str(path),
            "recoverable": recoverable,
            "errors": [*corruption_errors, *missing_errors],
            "present_members": sorted(
                str(member.relative_to(path))
                for member in path.rglob("*") if member.is_file()
            ),
        }
    try:
        completion_raw = completion_path.read_bytes()
        completion = json.loads(completion_raw)
    except (OSError, ValueError) as exc:
        return {
            "state": "invalid",
            "path": str(path),
            "recoverable": False,
            "errors": [f"completion.json is malformed: {exc}"],
        }
    errors = []
    if (
        not isinstance(completion, dict)
        or completion.get("schema") != COMPLETION_SCHEMA
        or not isinstance(completion.get("members"), dict)
        or not completion.get("members")
    ):
        errors.append("completion schema/members are invalid")
        members = {}
    else:
        members = completion["members"]
    for name, expected in members.items():
        try:
            relative = _safe_member(name)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        member = path / relative
        try:
            member_mode = os.lstat(member).st_mode
        except OSError as exc:
            errors.append(f"committed member unreadable: {name}: {exc}")
            continue
        if not stat.S_ISREG(member_mode) or stat.S_ISLNK(member_mode):
            errors.append(f"committed member missing/non-regular: {name}")
            continue
        if not isinstance(expected, dict):
            errors.append(f"invalid committed metadata: {name}")
            continue
        try:
            mismatch = (
                sha256_file(member) != expected.get("sha256")
                or member.stat().st_size != expected.get("size")
            )
        except OSError as exc:
            errors.append(f"committed member unreadable: {name}: {exc}")
            continue
        if mismatch:
            errors.append(f"committed member hash/size mismatch: {name}")
    for name in required_members:
        if name not in members:
            errors.append(f"required member not committed: {name}")
    return {
        "state": "complete_valid" if not errors else "invalid",
        "path": str(path),
        "recoverable": False,
        "errors": errors,
        "completion_sha256": hashlib.sha256(completion_raw).hexdigest(),
        "completion": completion,
    }


def publish_bundle(
    destination: Path,
    *,
    members: dict[str, bytes],
    metadata: dict,
    allow_existing_incomplete: bool = False,
    fault_at: str | None = None,
) -> dict:
    path = _absolute_lexical(Path(destination))
    _reject_symlink_components(path.parent, allow_missing_leaf=False)
    if not members:
        raise ValueError("portable bundle requires at least one member")
    normalized = {}
    for name, content in members.items():
        relative = _safe_member(name)
        if not isinstance(content, bytes):
            raise TypeError(f"bundle member {name} is not bytes")
        normalized[str(relative)] = content
    path.parent.mkdir(parents=True, exist_ok=True)
    reserved = False
    try:
        path.mkdir(mode=0o700)
        reserved = True
        _fsync_directory(path.parent)
    except FileExistsError:
        inspection = inspect_bundle(path)
        if inspection["state"] == "complete_valid":
            raise BundleExistsError(f"complete bundle already exists: {path}")
        if inspection["state"] == "invalid":
            raise IncompleteBundleError(
                f"invalid existing bundle cannot be recovered: {path}"
            )
        if not allow_existing_incomplete:
            raise IncompleteBundleError(
                f"incomplete/invalid bundle already exists: {path}"
            )
        if not path.is_dir() or path.is_symlink():
            raise IncompleteBundleError(
                f"existing destination is not recoverable: {path}"
            )
    if fault_at == "after_reservation":
        raise RuntimeError("injected interruption after reservation")
    for name, content in sorted(normalized.items()):
        member_path = path / name
        if member_path.exists():
            if (
                not member_path.is_file()
                or member_path.is_symlink()
                or sha256_file(member_path)
                != hashlib.sha256(content).hexdigest()
            ):
                raise IncompleteBundleError(
                    f"existing member differs from recovery payload: {name}"
                )
            continue
        _atomic_member(member_path, content)
        if fault_at == f"after_member:{name}":
            raise RuntimeError(f"injected interruption after {name}")
    _fsync_directory(path)
    if fault_at == "before_completion":
        raise RuntimeError("injected interruption before completion")
    completion = _completion_bytes(normalized, metadata)
    completion_path = path / "completion.json"
    if completion_path.exists():
        inspection = inspect_bundle(path)
        if inspection["state"] == "complete_valid":
            raise BundleExistsError(f"complete bundle already exists: {path}")
        raise IncompleteBundleError(
            f"invalid completion marker already exists: {completion_path}"
        )
    _atomic_member(completion_path, completion)
    _fsync_directory(path)
    inspection = inspect_bundle(
        path, required_members=tuple(sorted(normalized))
    )
    if inspection["state"] != "complete_valid":
        raise BundlePublicationError(
            "portable bundle failed post-publication verification: "
            + " | ".join(inspection["errors"])
        )
    inspection["reserved_new_destination"] = reserved
    return inspection


def legacy_renameat2_diagnostic(parent: Path) -> dict:
    """Report renameat2 behavior without making it part of publication."""

    source = Path(tempfile.mkdtemp(dir=parent, prefix=".renameat2-src."))
    destination = parent / f".renameat2-dst.{secrets.token_hex(6)}"
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            return {"supported": False, "errno": "ENOSYS"}
        renameat2.argtypes = [
            ctypes.c_int, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100, os.fsencode(source),
            -100, os.fsencode(destination),
            1,
        )
        if result == 0:
            return {"supported": True, "errno": None}
        error = ctypes.get_errno()
        return {
            "supported": False,
            "errno": errno.errorcode.get(error, str(error)),
        }
    finally:
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(destination, ignore_errors=True)


def capability_probe(
    parent: Path,
    *,
    renameat2_probe: Callable[[Path], dict] = legacy_renameat2_diagnostic,
) -> dict:
    root = Path(parent).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    probe_root = Path(tempfile.mkdtemp(dir=root, prefix=".portable-probe."))
    bundle = probe_root / "bundle"
    try:
        legacy = renameat2_probe(probe_root)
        publication = publish_bundle(
            bundle,
            members={"result.json": b'{"probe":true}\n'},
            metadata={"kind": "filesystem-capability-probe"},
        )
        return {
            "schema": "evsp-dr-portable-publication-probe-v1",
            "parent": str(root),
            "portable_protocol": publication["state"],
            "legacy_renameat2": legacy,
            "ready_for_recovery_probe_only": (
                publication["state"] == "complete_valid"
            ),
        }
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)

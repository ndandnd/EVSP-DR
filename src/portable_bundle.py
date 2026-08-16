"""Portable no-clobber bundles committed by ``completion.json``.

The destination directory is reserved atomically with ``mkdir``. Members are
written through same-directory temporary files and fsynced. ``completion.json``
is published last and is the only commit marker consumers may trust.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
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


def _open_directory_chain(path: Path) -> int:
    absolute = _absolute_lexical(path)
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _hash_at(directory_fd: int, name: str) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise BundlePublicationError(f"member is not regular: {name}")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), os.fstat(descriptor).st_size
    finally:
        os.close(descriptor)


def _atomic_member_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    accept_identical_existing: bool = True,
) -> None:
    temporary = f".{name}.tmp.{os.getpid()}.{secrets.token_hex(6)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            if not accept_identical_existing:
                raise
            digest, size = _hash_at(directory_fd, name)
            if (
                digest != hashlib.sha256(payload).hexdigest()
                or size != len(payload)
            ):
                raise
        os.fsync(directory_fd)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _atomic_member(path: Path, payload: bytes) -> None:
    path = _absolute_lexical(path)
    parent_fd = _open_directory_chain(path.parent)
    try:
        _atomic_member_at(parent_fd, path.name, payload)
    finally:
        os.close(parent_fd)


def atomic_write_new_file(path: Path, payload: bytes) -> None:
    destination = _absolute_lexical(path)
    parent_fd = _open_directory_chain(destination.parent)
    try:
        _atomic_member_at(
            parent_fd,
            destination.name,
            payload,
            accept_identical_existing=False,
        )
    finally:
        os.close(parent_fd)


def completion_bytes(members: dict[str, bytes], metadata: dict) -> bytes:
    payload = {
        "schema": COMPLETION_SCHEMA,
        "protocol": {
            "destination_reserved_by": "mkdir",
            "member_publication": "same-directory-temp-plus-hardlink-noreplace",
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
        safe_result_member = str(_safe_member(result_member))
        safe_required = tuple(str(_safe_member(name)) for name in required_members)
    except ValueError as exc:
        return {
            "state": "invalid",
            "path": str(path),
            "recoverable": False,
            "errors": [str(exc)],
        }
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
    try:
        completion_mode = os.lstat(completion_path).st_mode
    except FileNotFoundError:
        completion_mode = None
    except OSError as exc:
        return {
            "state": "invalid", "path": str(path),
            "recoverable": False, "errors": [str(exc)],
        }
    if completion_mode is not None and stat.S_ISLNK(completion_mode):
        return {
            "state": "invalid",
            "path": str(path),
            "recoverable": False,
            "errors": ["completion.json is symlinked"],
        }
    if completion_mode is not None and not stat.S_ISREG(completion_mode):
        return {
            "state": "invalid",
            "path": str(path),
            "recoverable": False,
            "errors": ["completion.json is present but non-regular"],
        }
    if completion_mode is None:
        corruption_errors = []
        missing_errors = []
        result_payload = None
        candidate = path / safe_result_member
        try:
            candidate_mode = os.lstat(candidate).st_mode
        except FileNotFoundError:
            candidate_mode = None
        except OSError as exc:
            corruption_errors.append(str(exc))
            candidate_mode = None
        if candidate_mode is not None and stat.S_ISLNK(candidate_mode):
            corruption_errors.append("result member is symlinked")
        elif candidate_mode is not None and not stat.S_ISREG(candidate_mode):
            corruption_errors.append("result member is present but non-regular")
        elif candidate_mode is not None:
            try:
                result_payload = json.loads(candidate.read_text())
                if not isinstance(result_payload, dict):
                    raise ValueError("result member is not an object")
                if recoverable_validator is not None:
                    recoverable_validator(result_payload)
            except (OSError, ValueError, TypeError) as exc:
                corruption_errors.append(str(exc))
        missing_required = []
        for name in safe_required:
            required_path = path / name
            try:
                required_mode = os.lstat(required_path).st_mode
            except FileNotFoundError:
                missing_required.append(name)
                continue
            except OSError as exc:
                corruption_errors.append(str(exc))
                continue
            if not stat.S_ISREG(required_mode) or stat.S_ISLNK(required_mode):
                corruption_errors.append(
                    f"required member is non-regular/symlinked: {name}"
                )
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
        candidate_invalid = candidate_mode is not None and bool(corruption_errors)
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
        protocol = completion.get("protocol")
        if protocol != {
            "destination_reserved_by": "mkdir",
            "member_publication":
                "same-directory-temp-plus-hardlink-noreplace",
            "commit_marker": "completion.json-published-last",
            "renameat2_required": False,
        }:
            errors.append("completion protocol attestation is invalid")
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
    for name in safe_required:
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
    if not path.parent.is_dir():
        raise BundlePublicationError(
            f"bundle parent must already exist: {path.parent}"
        )
    parent_fd = _open_directory_chain(path.parent)
    reserved = False
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent_fd)
        reserved = True
        os.fsync(parent_fd)
    except FileExistsError:
        inspection = inspect_bundle(path)
        if inspection["state"] == "complete_valid":
            os.close(parent_fd)
            raise BundleExistsError(f"complete bundle already exists: {path}")
        if inspection["state"] == "invalid":
            os.close(parent_fd)
            raise IncompleteBundleError(
                f"invalid existing bundle cannot be recovered: {path}"
            )
        if not allow_existing_incomplete:
            os.close(parent_fd)
            raise IncompleteBundleError(
                f"incomplete/invalid bundle already exists: {path}"
            )
        if not path.is_dir() or path.is_symlink():
            os.close(parent_fd)
            raise IncompleteBundleError(
                f"existing destination is not recoverable: {path}"
            )
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    destination_fd = os.open(path.name, flags, dir_fd=parent_fd)
    lock_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock_fd = os.open(
        ".publication.lock", lock_flags, 0o600, dir_fd=destination_fd
    )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if not reserved:
            inspection = inspect_bundle(path)
            if inspection["state"] == "complete_valid":
                raise BundleExistsError(
                    f"complete bundle already exists: {path}"
                )
            if inspection["state"] == "invalid":
                raise IncompleteBundleError(
                    f"invalid existing bundle cannot be recovered: {path}"
                )
        if fault_at == "after_reservation":
            raise RuntimeError("injected interruption after reservation")
        for name, content in sorted(normalized.items()):
            try:
                existing_digest, existing_size = _hash_at(
                    destination_fd, name
                )
            except FileNotFoundError:
                existing_digest = None
                existing_size = None
            if existing_digest is not None:
                if (
                    existing_digest != hashlib.sha256(content).hexdigest()
                    or existing_size != len(content)
                ):
                    raise IncompleteBundleError(
                        f"existing member differs from recovery payload: {name}"
                    )
            else:
                _atomic_member_at(destination_fd, name, content)
            if fault_at == f"after_member:{name}":
                raise RuntimeError(f"injected interruption after {name}")
        os.fsync(destination_fd)
        if fault_at == "before_completion":
            raise RuntimeError("injected interruption before completion")
        completion = completion_bytes(normalized, metadata)
        try:
            _hash_at(destination_fd, "completion.json")
        except FileNotFoundError:
            _atomic_member_at(
                destination_fd,
                "completion.json",
                completion,
                accept_identical_existing=False,
            )
        else:
            raise BundleExistsError(
                f"completion marker already exists: {path}"
            )
        os.fsync(destination_fd)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        os.close(destination_fd)
        os.close(parent_fd)
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
            "implementation_sha256": sha256_file(
                Path(__file__).resolve()
            ),
            "portable_protocol": publication["state"],
            "legacy_renameat2": legacy,
            "ready_for_recovery_probe_only": (
                publication["state"] == "complete_valid"
            ),
        }
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)

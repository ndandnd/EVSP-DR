"""Small durability helpers for long, preemptible cluster runs.

The exact-pricer journal is append-only by design.  A hard preemption can
interrupt its final write, leaving one truncated JSON record.  Recovery is
therefore deliberately narrow: only the final malformed non-empty record may
be removed.  Corruption before the last record is never hidden.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class DurableFileError(ValueError):
    """Raised when a persisted artifact is corrupt beyond safe tail repair."""


def flush_and_fsync(handle) -> None:
    """Make writes on an open file visible to a restarted allocation.

    ``flush()`` alone only drains Python's userspace buffer.  Long Unicorn
    jobs can be preempted at any instruction, so checkpoint data is not
    considered durable until the operating system has also acknowledged it.
    """

    handle.flush()
    os.fsync(handle.fileno())


@contextmanager
def exclusive_output_lock(output_path: Path, metadata: dict | None = None):
    """Hold a non-blocking process lock for one canonical output stem.

    The lock file intentionally remains after release as a diagnostic record;
    the kernel lock, not file existence, determines ownership.  This prevents
    duplicate Slurm submissions from interleaving JSONL appends while allowing
    a requeued allocation to acquire the same output after its predecessor has
    exited or been killed.
    """

    output_path = Path(output_path)
    lock_path = Path(str(output_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise DurableFileError(
                f"another process holds the exact-CG output lock "
                f"{lock_path}: {owner}"
            ) from exc
        payload = dict(metadata or {})
        payload["output"] = str(output_path)
        payload["lock_path"] = str(lock_path)
        handle.seek(0)
        handle.truncate()
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        flush_and_fsync(handle)
        yield lock_path
    finally:
        try:
            if acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def atomic_write_json(path: Path, payload: Any, *, indent: int = 1) -> None:
    """Write JSON through a flushed temporary file and atomic rename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with open(temporary, "w") as fh:
            json.dump(payload, fh, indent=indent)
            fh.write("\n")
            flush_and_fsync(fh)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text through a flushed temporary file and atomic rename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with open(temporary, "w") as fh:
            fh.write(text)
            flush_and_fsync(fh)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes through a flushed temporary file and atomic rename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with open(temporary, "wb") as fh:
            fh.write(payload)
            flush_and_fsync(fh)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy a file completely before making it visible at *destination*."""

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with open(source, "rb") as src, open(temporary, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            flush_and_fsync(dst)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _recover_complete_dict_prefix(
    raw_line: bytes, *, allow_unparseable_trailing: bool,
) -> tuple[list[dict], bytes]:
    """Recover complete concatenated JSON objects before an incomplete tail.

    Legacy appenders could reopen a file whose last valid object lacked a
    newline and concatenate the next object onto it.  Complete dictionary
    values are unambiguous and can be normalized to JSONL.  Anything other
    than dictionary objects followed by an optional incomplete ``{...``
    suffix is treated as corruption rather than guessed away.  A copy-only
    legacy migration may set ``allow_unparseable_trailing`` after archiving
    the original bytes; that mode keeps any complete leading dictionary
    records and quarantines the otherwise uninterpretable final suffix.
    """

    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        if allow_unparseable_trailing:
            return [], b""
        raise DurableFileError(
            "malformed final JSONL data is not valid UTF-8"
        ) from exc
    decoder = json.JSONDecoder()
    recovered = []
    serialized = []
    position = 0
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        start = position
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            suffix = text[start:].lstrip()
            if suffix.startswith("{"):
                break
            if allow_unparseable_trailing:
                break
            raise DurableFileError(
                "malformed final JSONL data is not a dictionary record or "
                "an incomplete dictionary suffix"
            ) from exc
        if not isinstance(value, dict):
            if allow_unparseable_trailing:
                break
            raise DurableFileError(
                "malformed final JSONL data contains a non-object value"
            )
        recovered.append(value)
        serialized.append(text[start:position].strip())
    normalized = "".join(f"{item}\n" for item in serialized).encode("utf-8")
    return recovered, normalized


def read_jsonl_records(
    path: Path,
    *,
    repair_trailing: bool = False,
    collect: bool = True,
    allow_unparseable_trailing: bool = False,
) -> list[dict]:
    """Read a JSONL file, optionally repairing one interrupted final record.

    A valid final record without a newline receives one before an append-mode
    writer is opened.  A malformed final non-empty line is truncated only when
    ``repair_trailing`` is true.  Malformed data followed by any later
    non-whitespace bytes is treated as interior corruption and always raises.
    Set ``collect=False`` for a streaming validation pass over a large pool.
    ``allow_unparseable_trailing`` is only appropriate for an archived,
    copy-only legacy migration; current-code resume paths leave it false.
    """

    path = Path(path)
    records: list[dict] = []
    saw_record = False
    last_valid_had_newline = True
    repair_offset: int | None = None
    repair_bytes = b""
    with open(path, "rb") as fh:
        while True:
            offset = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.strip():
                last_valid_had_newline = line.endswith(b"\n")
                continue
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                remainder = fh.read()
                if remainder.strip():
                    raise DurableFileError(
                        f"{path} has malformed JSONL data before EOF at byte "
                        f"{offset}; refusing automatic repair"
                    ) from exc
                if not repair_trailing:
                    raise DurableFileError(
                        f"{path} has a malformed final JSONL record at byte "
                        f"{offset}"
                    ) from exc
                recovered, repair_bytes = _recover_complete_dict_prefix(
                    line,
                    allow_unparseable_trailing=allow_unparseable_trailing,
                )
                if collect:
                    records.extend(recovered)
                if recovered:
                    saw_record = True
                    last_valid_had_newline = True
                repair_offset = offset
                break
            if not isinstance(record, dict):
                raise DurableFileError(
                    f"{path} JSONL record at byte {offset} is not an object"
                )
            saw_record = True
            if collect:
                records.append(record)
            last_valid_had_newline = line.endswith(b"\n")

    if repair_trailing:
        if repair_offset is not None:
            with open(path, "r+b") as fh:
                fh.seek(repair_offset)
                if repair_bytes:
                    fh.write(repair_bytes)
                fh.truncate(repair_offset + len(repair_bytes))
                flush_and_fsync(fh)
        elif saw_record and not last_valid_had_newline:
            with open(path, "ab") as fh:
                fh.write(b"\n")
                flush_and_fsync(fh)
    return records


def valid_json_object(path: Path, required_keys=()) -> bool:
    """Return whether *path* is a JSON object containing every required key."""

    try:
        with open(path) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and all(key in payload for key in required_keys)

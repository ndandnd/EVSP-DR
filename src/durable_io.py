"""Small durability helpers for long, preemptible cluster runs.

The exact-pricer journal is append-only by design.  A hard preemption can
interrupt its final write, leaving one truncated JSON record.  Recovery is
therefore deliberately narrow: only the final malformed non-empty record may
be removed.  Corruption before the last record is never hidden.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


class DurableFileError(ValueError):
    """Raised when a persisted artifact is corrupt beyond safe tail repair."""


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
            fh.flush()
            os.fsync(fh.fileno())
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
            fh.flush()
            os.fsync(fh.fileno())
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
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_jsonl_records(path: Path, *, repair_trailing: bool = False,
                       collect: bool = True) -> list[dict]:
    """Read a JSONL file, optionally repairing one interrupted final record.

    A valid final record without a newline receives one before an append-mode
    writer is opened.  A malformed final non-empty line is truncated only when
    ``repair_trailing`` is true.  Malformed data followed by any later
    non-whitespace bytes is treated as interior corruption and always raises.
    Set ``collect=False`` for a streaming validation pass over a large pool.
    """

    path = Path(path)
    records: list[dict] = []
    saw_record = False
    last_valid_had_newline = True
    repair_offset: int | None = None
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
                fh.truncate(repair_offset)
                fh.flush()
                os.fsync(fh.fileno())
        elif saw_record and not last_valid_had_newline:
            with open(path, "ab") as fh:
                fh.write(b"\n")
                fh.flush()
                os.fsync(fh.fileno())
    return records


def valid_json_object(path: Path, required_keys=()) -> bool:
    """Return whether *path* is a JSON object containing every required key."""

    try:
        with open(path) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and all(key in payload for key in required_keys)

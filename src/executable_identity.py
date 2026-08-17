"""Content-bound executable resolution for PATH-free Slurm workers."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_executable(
    path: Path | str,
    *,
    expected_sha256: str | None,
    label: str,
) -> tuple[Path, str]:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{label} executable path must be absolute")
    resolved = candidate.resolve(strict=True)
    mode = os.stat(resolved, follow_symlinks=False).st_mode
    if not stat.S_ISREG(mode) or not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} executable is not regular/executable")
    observed = sha256_file(resolved)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"{label} executable SHA-256 mismatch")
    return resolved, observed


def resolve_executable(
    explicit: Path | str | None,
    *,
    command: str,
    label: str,
) -> tuple[Path, str]:
    raw = str(explicit) if explicit is not None else shutil.which(command)
    if not raw:
        raise ValueError(
            f"{label} executable unavailable; pass its absolute path"
        )
    return validate_executable(
        raw, expected_sha256=None, label=label
    )

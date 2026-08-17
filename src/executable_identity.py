"""Content-bound executable resolution for PATH-free Slurm workers."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
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
    if candidate.is_symlink():
        raise ValueError(f"{label} executable path must not be a symlink")
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
        Path(raw).expanduser().resolve(),
        expected_sha256=None,
        label=label,
    )


def run_bound_executable(
    path: Path | str,
    *,
    expected_sha256: str,
    label: str,
    arguments: list[str],
    **kwargs,
) -> subprocess.CompletedProcess:
    candidate, _observed = validate_executable(
        path, expected_sha256=expected_sha256, label=label
    )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(candidate, flags)
    memfd = None
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode) or not (mode & 0o111):
            raise ValueError(f"{label} executable descriptor is invalid")
        if not hasattr(os, "memfd_create"):
            raise RuntimeError("memfd_create is required for bound execution")
        memfd = os.memfd_create(f"evsp-{label}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            os.write(memfd, chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"{label} executable SHA-256 mismatch")
        os.fchmod(memfd, mode & 0o777)
        os.lseek(memfd, 0, os.SEEK_SET)
        executable = f"/proc/self/fd/{memfd}"
        return subprocess.run(
            [executable, *arguments],
            pass_fds=(memfd,),
            **kwargs,
        )
    finally:
        if memfd is not None:
            os.close(memfd)
        os.close(descriptor)

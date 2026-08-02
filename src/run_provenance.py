"""Small, testable provenance helpers for long-running experiment resumes."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path
from typing import Iterable


class GitProvenanceError(RuntimeError):
    """Raised when a deterministic Git worktree fingerprint cannot be built."""


def _git_bytes(repo_root: Path, *git_args: str) -> bytes:
    result = subprocess.run(
        ["git", *git_args],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitProvenanceError(
            f"git {' '.join(git_args)} failed with exit {result.returncode}: {detail}"
        )
    return result.stdout


def _length_prefixed(parts: Iterable[bytes]) -> bytes:
    payload = bytearray()
    for part in parts:
        payload.extend(len(part).to_bytes(8, "big"))
        payload.extend(part)
    return bytes(payload)


def worktree_content_fingerprint(repo_root: Path) -> str:
    """Hash tracked changes plus every nonignored untracked file.

    The Git commit already identifies all unchanged tracked bytes.  This digest
    therefore covers the remaining state needed to distinguish two worktrees
    at the same commit: staged/unstaged tracked changes, file-mode changes,
    deletions, and the path/type/content of nonignored untracked files.
    Ignored result pools are intentionally excluded.
    """

    root = Path(repo_root).resolve()
    tracked_diff = _git_bytes(
        root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
    )
    raw_untracked = _git_bytes(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    relative_paths = sorted(path for path in raw_untracked.split(b"\0") if path)

    digest = hashlib.sha256()
    digest.update(b"EVSP-DR-worktree-v1\0")
    digest.update(_length_prefixed((b"tracked-diff", tracked_diff)))

    for raw_relative in relative_paths:
        relative = os.fsdecode(raw_relative)
        path = root / relative
        metadata = path.lstat()
        mode = stat.S_IFMT(metadata.st_mode)
        if stat.S_ISLNK(metadata.st_mode):
            payload = os.fsencode(os.readlink(path))
            kind = b"symlink"
        elif stat.S_ISREG(metadata.st_mode):
            payload = path.read_bytes()
            kind = b"file"
        else:
            # Git tracks files and symlinks, but fail closed if an unusual
            # nonignored filesystem entry appears rather than silently omitting it.
            raise GitProvenanceError(
                f"unsupported nonignored untracked entry: {relative!r} (mode={oct(mode)})"
            )
        digest.update(
            _length_prefixed(
                (b"untracked", raw_relative, kind, str(metadata.st_mode).encode(), payload)
            )
        )

    return digest.hexdigest()


def mismatches(
    checkpoint: dict,
    expected: dict,
    *,
    compare_missing: bool = False,
) -> dict[str, tuple[object, object]]:
    """Return deterministic checkpoint/current mismatches for selected fields."""

    return {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if (compare_missing or key in checkpoint) and checkpoint.get(key) != value
    }

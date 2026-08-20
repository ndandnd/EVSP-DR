#!/usr/bin/env python3
"""Atomically install one hash-bound profiler input below checkout data/."""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import shutil
import stat
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(source: Path, data_root: Path, relative: Path, expected: str) -> Path:
    source = source.expanduser().resolve()
    data_root = data_root.expanduser().absolute()
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe relative data path: {relative}")
    if sha256_file(source) != expected:
        raise ValueError("staged source hash mismatch")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    opened = []
    temporary_name = None
    try:
        try:
            parent_fd = os.open(data_root, directory_flags)
        except OSError as exc:
            raise ValueError(
                f"data root is missing, non-directory, or symlinked: {data_root}"
            ) from exc
        opened.append(parent_fd)
        parent_path = data_root
        for part in relative.parent.parts:
            try:
                next_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o755, dir_fd=parent_fd)
                except FileExistsError:
                    pass
                try:
                    next_fd = os.open(part, directory_flags, dir_fd=parent_fd)
                except OSError as exc:
                    raise ValueError(
                        f"unsafe concurrent data parent: {parent_path / part}"
                    ) from exc
            except OSError as exc:
                raise ValueError(
                    f"unsafe data parent: {parent_path / part}"
                ) from exc
            parent_fd = next_fd
            opened.append(parent_fd)
            parent_path = parent_path / part

        destination_name = relative.name
        parent_identity = os.fstat(parent_fd)

        def assert_parent_unchanged() -> None:
            try:
                current_parent = os.stat(parent_path, follow_symlinks=False)
            except OSError as exc:
                raise ValueError(
                    "data parent changed during installation"
                ) from exc
            if (not stat.S_ISDIR(current_parent.st_mode)
                    or current_parent.st_dev != parent_identity.st_dev
                    or current_parent.st_ino != parent_identity.st_ino):
                raise ValueError("data parent changed during installation")

        def hash_at(name: str) -> str:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(name, flags, dir_fd=parent_fd)
            except OSError as exc:
                raise ValueError(
                    f"unsafe data destination: {data_root / relative}"
                ) from exc
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError(
                        f"data destination is not regular: {data_root / relative}"
                    )
                digest = hashlib.sha256()
                with os.fdopen(os.dup(descriptor), "rb") as handle:
                    for chunk in iter(
                            lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()
            finally:
                os.close(descriptor)

        try:
            existing_hash = hash_at(destination_name)
        except ValueError:
            existing_hash = None
        if existing_hash is not None:
            if existing_hash != expected:
                raise ValueError(
                    f"existing data hash mismatch: {data_root / relative}"
                )
            assert_parent_unchanged()
            return data_root / relative

        temporary_name = (
            f".{destination_name}.profile-input.{os.getpid()}."
            f"{secrets.token_hex(6)}"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(
            temporary_name, flags, 0o600, dir_fd=parent_fd
        )
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, output)
            output.flush()
            os.fsync(output.fileno())
        if hash_at(temporary_name) != expected:
            raise ValueError("temporary data copy hash mismatch")
        try:
            os.link(
                temporary_name,
                destination_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            if hash_at(destination_name) != expected:
                raise ValueError(
                    f"concurrent data destination mismatch: "
                    f"{data_root / relative}"
                )
        assert_parent_unchanged()
        return data_root / relative
    finally:
        if temporary_name is not None and opened:
            try:
                os.unlink(temporary_name, dir_fd=opened[-1])
            except FileNotFoundError:
                pass
        for descriptor in reversed(opened):
            os.close(descriptor)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--relative", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    if (len(args.sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in args.sha256.lower())):
        parser.error("--sha256 must be 64 hexadecimal characters")
    print(install(
        args.source, args.data_root, args.relative, args.sha256.lower()
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

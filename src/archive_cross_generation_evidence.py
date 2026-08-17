#!/usr/bin/env python3
"""Create a deterministic, checksummed archive of an evidence build."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path


ARCHIVE_SCHEMA = "evsp-dr-cross-generation-archive-v1"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"archive input is not regular: {path}")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _validated_build(build_dir: Path) -> dict[str, bytes]:
    root = build_dir.expanduser().resolve()
    completion_path = root / "completion.json"
    completion_raw = _read_regular(completion_path)
    completion = json.loads(completion_raw)
    if (
        completion.get("schema")
        != "evsp-dr-cross-generation-output-completion-v1"
        or not isinstance(completion.get("members"), dict)
    ):
        raise ValueError("evidence build completion marker is invalid")
    files = {}
    for name, expected_sha in sorted(completion["members"].items()):
        relative = Path(name)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name in {"", ".", ".."}
        ):
            raise ValueError(f"unsafe evidence member: {name}")
        payload = _read_regular(root / relative)
        if _sha(payload) != expected_sha:
            raise ValueError(f"evidence member hash mismatch: {name}")
        files[f"build/{name}"] = payload
    files["build/completion.json"] = completion_raw
    return files


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o444
            tar.addfile(info, io.BytesIO(payload))
    return archive.getvalue()


def archive_evidence(
    build_dir: Path,
    input_manifest: Path,
    output_dir: Path,
) -> dict:
    output = output_dir.expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"archive output exists: {output}")
    files = _validated_build(build_dir)
    manifest_raw = _read_regular(input_manifest.expanduser().resolve())
    files["input_manifest.json"] = manifest_raw
    archive_manifest = {
        "schema": ARCHIVE_SCHEMA,
        "input_manifest_sha256": _sha(manifest_raw),
        "members": {
            name: _sha(payload) for name, payload in sorted(files.items())
        },
    }
    archive_manifest_raw = (
        json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    files["ARCHIVE_MANIFEST.json"] = archive_manifest_raw
    tar_payload = _tar_bytes(files)
    tar_sha = _sha(tar_payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        dir=output.parent, prefix=f".{output.name}.tmp."
    ))
    try:
        members = {
            "evidence.tar": tar_payload,
            "evidence.tar.sha256": f"{tar_sha}  evidence.tar\n".encode(),
            "ARCHIVE_MANIFEST.json": archive_manifest_raw,
        }
        for name, payload in sorted(members.items()):
            path = staging / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        completion = {
            "schema": "evsp-dr-cross-generation-archive-completion-v1",
            "members": {
                name: _sha(payload) for name, payload in sorted(members.items())
            },
        }
        with (staging / "completion.json").open("x") as handle:
            json.dump(completion, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        output.mkdir(mode=0o755)
        for name in sorted(members):
            os.link(staging / name, output / name)
        output_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(output_fd)
        os.link(
            staging / "completion.json",
            output / "completion.json",
        )
        os.fsync(output_fd)
        os.close(output_fd)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(parent_fd)
        os.close(parent_fd)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "schema": ARCHIVE_SCHEMA,
        "output_dir": str(output),
        "archive_sha256": tar_sha,
        "members": len(files),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(archive_evidence(
        args.build_dir, args.input_manifest, args.out_dir
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

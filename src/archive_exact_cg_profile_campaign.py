#!/usr/bin/env python3
"""Create a checksummed archive of one completed profile campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.tmp.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite archive artifact: {path}"
            ) from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def archive(campaign_root: Path, output: Path) -> dict:
    root = campaign_root.expanduser().resolve()
    output = output.expanduser().resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("archive output must be outside campaign root")
    manifest_path = root / "campaign.json"
    campaign = json.loads(manifest_path.read_text())
    sidecar = Path(str(output) + ".manifest.json")
    checksum_path = Path(str(output) + ".sha256")
    for path in (output, sidecar, checksum_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite archive: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in root.rglob("*") if path.is_file())
    before = {
        str(path.relative_to(root)): sha256_file(path)
        for path in files
    }
    temporary_archive = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.tmp.",
            delete=False,
        ) as handle:
            temporary_archive = Path(handle.name)
        with tarfile.open(temporary_archive, "w:gz") as archive_file:
            for path in files:
                archive_file.add(
                    path,
                    arcname=str(Path(root.name) / path.relative_to(root)),
                    recursive=False,
                )
        after = {
            str(path.relative_to(root)): sha256_file(path)
            for path in files
        }
        if after != before:
            raise RuntimeError("campaign changed while being archived")
        archive_sha = sha256_file(temporary_archive)
        try:
            os.link(temporary_archive, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite archive: {output}"
            ) from exc
    finally:
        if temporary_archive is not None and temporary_archive.exists():
            temporary_archive.unlink()

    record = {
        "schema": "evsp-dr-exact-cg-profile-archive-v1",
        "campaign": campaign.get("campaign"),
        "campaign_root": str(root),
        "expected_commit": (
            campaign.get("checkout_identity") or {}
        ).get("expected_commit"),
        "profile_core_commit": campaign.get("profile_core_commit"),
        "campaign_manifest_sha256": before.get("campaign.json"),
        "files": before,
        "archive": str(output),
        "archive_sha256": archive_sha,
    }
    _publish_new(
        sidecar, (json.dumps(record, indent=2) + "\n").encode()
    )
    _publish_new(
        checksum_path,
        f"{archive_sha}  {output.name}\n".encode(),
    )
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(archive(args.campaign_root, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

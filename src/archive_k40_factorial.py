#!/usr/bin/env python3
"""Compute-node-only no-clobber archive for validated k40 factorial evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path

from k40_factorial_artifacts import validate_campaign, validate_historical


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parent,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _campaign_log_files(campaign_dir: Path) -> list[Path]:
    root = campaign_dir.resolve()
    try:
        repo_root = root.parents[3]
    except IndexError as exc:
        raise ValueError(f"cannot infer repository root from {root}") from exc
    candidates = (
        repo_root / "src/cluster_logs/k40_factorial" / root.name,
        repo_root / "src/logs/k40_factorial" / root.name,
    )
    files = []
    for candidate in candidates:
        if candidate.is_dir():
            files.extend(
                path for path in candidate.rglob("*") if path.is_file()
            )
    if not files:
        raise ValueError(f"campaign logs are missing for {root}")
    return files


def _publish_archive(
    sources: dict[str, Path],
    output: Path,
    archive_metadata: dict,
    watched_roots: list[Path],
    virtual_members: dict[str, bytes],
) -> dict:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"archive output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    watched_before = {
        str(path.resolve())
        for root in watched_roots
        for path in root.rglob("*")
        if path.is_file()
    }
    temporary_archive = None
    with tempfile.TemporaryDirectory(
            dir=output.parent,
            prefix=f".{output.name}.snapshot.") as snapshot_text:
        snapshot = Path(snapshot_text)
        staged = {}
        hashes = {}
        for member, payload in sorted(virtual_members.items()):
            destination = snapshot / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            staged[member] = destination
            hashes[member] = sha256_file(destination)
        for member, source_path in sorted(sources.items()):
            if member in hashes:
                raise ValueError(f"duplicate archive member: {member}")
            if source_path.is_symlink():
                raise ValueError(f"refusing symlinked source: {source_path}")
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(source_path, flags)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError(f"source is not regular: {source_path}")
                destination = snapshot / member
                destination.parent.mkdir(parents=True, exist_ok=True)
                with (
                    os.fdopen(os.dup(descriptor), "rb") as source,
                    destination.open("xb") as target,
                ):
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
            finally:
                os.close(descriptor)
            digest = sha256_file(destination)
            if sha256_file(source_path) != digest:
                raise RuntimeError(f"source changed while staging: {source_path}")
            staged[member] = destination
            hashes[member] = digest

        internal_manifest = {
            "schema": "evsp-dr-k40-factorial-archive-v1",
            **archive_metadata,
            "members": hashes,
        }
        manifest_bytes = (
            json.dumps(internal_manifest, indent=2) + "\n"
        ).encode()
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.tmp.",
            delete=False,
        ) as handle:
            temporary_archive = Path(handle.name)
        try:
            with tarfile.open(temporary_archive, "w:gz") as tar:
                for member, staged_path in sorted(staged.items()):
                    tar.add(staged_path, arcname=member, recursive=False)
                info = tarfile.TarInfo("ARCHIVE_MANIFEST.json")
                info.size = len(manifest_bytes)
                tar.addfile(info, io.BytesIO(manifest_bytes))
            for member, source_path in sources.items():
                if sha256_file(source_path) != hashes[member]:
                    raise RuntimeError(
                        f"source changed while archiving: {source_path}"
                    )
            watched_after = {
                str(path.resolve())
                for root in watched_roots
                for path in root.rglob("*")
                if path.is_file()
            }
            if watched_after != watched_before:
                raise RuntimeError(
                    "campaign/log file set changed while archiving"
                )
            with tarfile.open(temporary_archive, "r:gz") as tar:
                names = set(tar.getnames())
                if names != {*hashes, "ARCHIVE_MANIFEST.json"}:
                    raise RuntimeError("archive member set mismatch")
                archived_manifest = json.load(
                    tar.extractfile("ARCHIVE_MANIFEST.json")
                )
                if archived_manifest != internal_manifest:
                    raise RuntimeError("archive internal manifest mismatch")
                for member, expected in hashes.items():
                    digest = hashlib.sha256(
                        tar.extractfile(member).read()
                    ).hexdigest()
                    if digest != expected:
                        raise RuntimeError(
                            f"archive member checksum mismatch: {member}"
                        )
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
    return {
        **internal_manifest,
        "archive": str(output),
        "archive_sha256": archive_sha,
    }


def archive(
    campaign_dirs: list[Path],
    historical: Path,
    accounting: Path,
    output: Path,
    *,
    require_compute: bool = True,
) -> dict:
    if len(campaign_dirs) != 2:
        raise ValueError("exactly two factorial campaign directories required")
    resolved_campaigns = [
        path.expanduser().resolve() for path in campaign_dirs
    ]
    if len(set(resolved_campaigns)) != 2:
        raise ValueError("factorial campaign directories must be distinct")
    if require_compute:
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("archive helper must run in a Slurm allocation")
        if "login" in platform.node().lower():
            raise RuntimeError("archive helper refuses Unicorn login nodes")
    accounting = accounting.expanduser().resolve()
    if not accounting.is_file():
        raise ValueError(f"Slurm accounting input is missing: {accounting}")
    validated = [
        validate_campaign(path, replicate=f"R{index}")
        for index, path in enumerate(resolved_campaigns, start=1)
    ]
    historical_record = validate_historical(historical)
    sources: dict[str, Path] = {}
    watched_roots = []
    for campaign in validated:
        campaign_root = Path(campaign["campaign_dir"])
        watched_roots.append(campaign_root)
        for source in campaign_root.rglob("*"):
            if not source.is_file():
                continue
            member = str(
                Path("campaigns") / campaign["replicate"]
                / source.relative_to(campaign_root)
            )
            sources[member] = source
        repo_root = campaign_root.parents[3]
        campaign_logs = _campaign_log_files(campaign_root)
        watched_roots.extend(
            candidate for candidate in (
                repo_root / "src/cluster_logs/k40_factorial"
                / campaign_root.name,
                repo_root / "src/logs/k40_factorial"
                / campaign_root.name,
            )
            if candidate.is_dir()
        )
        for log in campaign_logs:
            member = str(
                Path("logs") / campaign["replicate"]
                / log.relative_to(repo_root)
            )
            sources[member] = log
    historical_path = Path(historical_record["status_path"])
    for source_text in historical_record["files"]:
        source = Path(source_text)
        sources[str(Path("historical") / source.name)] = source
    historical_iters = Path(str(historical_path) + ".iters.csv")
    if not historical_iters.is_file():
        raise ValueError("historical iteration trajectory is missing")
    sources[
        str(Path("historical") / historical_iters.name)
    ] = historical_iters
    sources["slurm/accounting.txt"] = accounting
    git_commit = _git("rev-parse", "HEAD")
    git_status = _git("status", "--porcelain", "--untracked-files=no")
    if git_commit is None or git_status is None:
        raise RuntimeError("could not collect Git archive identity")
    metadata = {
        "created_by_commit": git_commit,
        "created_by_git_status": git_status,
        "campaigns": [
            {
                "replicate": record["replicate"],
                "campaign": record["campaign"],
                "prep": record["prep"],
            }
            for record in validated
        ],
        "historical": historical_record,
        "slurm_accounting_sha256": sha256_file(accounting),
    }
    resolved_output = output.expanduser().resolve()
    if any(
            resolved_output == root.resolve()
            or root.resolve() in resolved_output.parents
            for root in watched_roots):
        raise ValueError("archive output must be outside campaign/log roots")
    return _publish_archive(
        sources,
        output,
        metadata,
        watched_roots,
        {
            "git/commit.txt": (git_commit + "\n").encode(),
            "git/tracked-status.txt": (git_status + "\n").encode(),
        },
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--slurm-accounting", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = archive(
        args.campaign_dir,
        args.historical,
        args.slurm_accounting,
        args.out,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

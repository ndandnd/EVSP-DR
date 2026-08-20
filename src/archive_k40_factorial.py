#!/usr/bin/env python3
"""Compute-node-only no-clobber archive for validated k40 factorial evidence."""

from __future__ import annotations

import argparse
import csv
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

from k40_factorial_artifacts import (
    _validate_trajectory,
    validate_campaign,
    validate_historical,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"refusing symlink in archive source: {path}")
        if path.is_file():
            files.append(path)
    return sorted(files)


def _hash_stream(handle) -> str:
    digest = hashlib.sha256()
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


def _validate_compute_allocation() -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    hostname = platform.node().split(".", 1)[0]
    if not job_id:
        raise RuntimeError("archive helper must run in a Slurm allocation")
    if "login" in hostname.lower():
        raise RuntimeError("archive helper refuses Unicorn login nodes")
    job = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        text=True,
        capture_output=True,
        check=False,
    )
    if job.returncode != 0 or "JobState=RUNNING" not in job.stdout:
        raise RuntimeError("could not verify a live running Slurm allocation")
    fields = dict(
        token.split("=", 1)
        for token in job.stdout.split()
        if "=" in token
    )
    node_list = fields.get("NodeList")
    if not node_list or node_list == "(null)":
        raise RuntimeError("Slurm allocation has no compute nodes")
    hosts = subprocess.run(
        ["scontrol", "show", "hostnames", node_list],
        text=True,
        capture_output=True,
        check=False,
    )
    allocated = {
        line.strip().split(".", 1)[0]
        for line in hosts.stdout.splitlines() if line.strip()
    }
    if hosts.returncode != 0 or hostname not in allocated:
        raise RuntimeError("current host is not assigned to the Slurm job")
    if _git("symbolic-ref", "-q", "HEAD") is not None:
        raise RuntimeError("archive checkout must be detached")
    if _git("status", "--porcelain", "--untracked-files=no") != "":
        raise RuntimeError("archive checkout has tracked changes")


def _campaign_log_files(
    campaign_dir: Path,
    launch_rows: list[dict],
) -> tuple[list[Path], dict[str, str]]:
    root = campaign_dir.resolve()
    try:
        repo_root = root.parents[3]
    except IndexError as exc:
        raise ValueError(f"cannot infer repository root from {root}") from exc
    log_root = repo_root / "src/cluster_logs/k40_factorial" / root.name
    if not log_root.is_dir():
        raise ValueError(f"campaign logs are missing for {root}")
    files = _regular_files(log_root)
    for row in launch_rows:
        stem = f"{row['job_name']}_{row['job_id']}"
        stdout = log_root / f"{stem}.out"
        stderr = log_root / f"{stem}.err"
        if not stdout.is_file() or not stderr.is_file():
            raise ValueError(f"matching stdout/stderr are missing for {stem}")
        marker = (
            "[K40-PREP] READY"
            if row["role"] == "prep"
            else "[K40-FACTORIAL] DONE"
        )
        if marker not in stdout.read_bytes().decode(errors="replace"):
            raise ValueError(f"completion marker is missing from {stdout}")
    return files, {
        str(path.resolve()): sha256_file(path) for path in files
    }


def _validate_accounting(path: Path, campaigns: list[dict]) -> str:
    required = {
        str(row["job_id"]): row["job_name"]
        for campaign in campaigns
        for row in campaign["launch"]
    }
    records = {}
    raw = path.read_bytes()
    with io.StringIO(raw.decode(), newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        required_fields = {"JobIDRaw", "JobName", "State", "ExitCode"}
        if not required_fields.issubset(reader.fieldnames or ()):
            raise ValueError(
                "Slurm accounting must be pipe-delimited sacct output with "
                "JobIDRaw,JobName,State,ExitCode"
            )
        for row in reader:
            job_id = str(row.get("JobIDRaw") or "")
            if "." not in job_id and job_id:
                if job_id in records:
                    raise ValueError(
                        f"Slurm accounting duplicates root job {job_id}"
                    )
                records[job_id] = row
    missing = set(required) - set(records)
    if missing:
        raise ValueError(f"Slurm accounting is missing jobs: {sorted(missing)}")
    for job_id, job_name in required.items():
        row = records[job_id]
        if (
            not str(row.get("State") or "").startswith("COMPLETED")
            or row.get("ExitCode") != "0:0"
            or row.get("JobName") != job_name
        ):
            raise ValueError(f"Slurm job {job_id} did not complete cleanly")
    return hashlib.sha256(raw).hexdigest()


def _publish_archive(
    sources: dict[str, Path],
    output: Path,
    archive_metadata: dict,
    watched_roots: list[Path],
    virtual_members: dict[str, bytes],
    expected_source_hashes: dict[str, str],
) -> dict:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"archive output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_roots = [root.resolve() for root in watched_roots]
    watched_before = {
        str(path.resolve())
        for path in sources.values()
        if any(
            path.resolve() == root or root in path.resolve().parents
            for root in resolved_roots
        )
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
            expected_source = expected_source_hashes.get(
                str(source_path.resolve())
            )
            if (
                expected_source is not None
                and sha256_file(source_path) != expected_source
            ):
                raise RuntimeError(
                    f"validated source changed before staging: {source_path}"
                )
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
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"archive member is missing: {member}")
                    digest = _hash_stream(extracted)
                    if digest != expected:
                        raise RuntimeError(
                            f"archive member checksum mismatch: {member}"
                        )
            with temporary_archive.open("rb") as handle:
                os.fsync(handle.fileno())
            archive_sha = sha256_file(temporary_archive)
            for member, source_path in sources.items():
                if sha256_file(source_path) != hashes[member]:
                    raise RuntimeError(
                        f"source changed during archive verification: "
                        f"{source_path}"
                    )
            watched_after = {
                str(path.resolve())
                for root in resolved_roots
                for path in _regular_files(root)
            }
            if watched_after != watched_before:
                raise RuntimeError(
                    "campaign/log file set changed while archiving"
                )
            if (
                _git("rev-parse", "HEAD")
                != archive_metadata["created_by_commit"]
                or _git("status", "--porcelain", "--untracked-files=no")
                != archive_metadata["created_by_git_status"]
            ):
                raise RuntimeError("Git identity changed while archiving")
            try:
                os.link(temporary_archive, output)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite archive: {output}"
                ) from exc
            parent = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
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
        _validate_compute_allocation()
    accounting = accounting.expanduser().resolve()
    if not accounting.is_file():
        raise ValueError(f"Slurm accounting input is missing: {accounting}")
    validated = [
        validate_campaign(path, replicate=f"R{index}")
        for index, path in enumerate(resolved_campaigns, start=1)
    ]
    if set(validated[0]["job_ids"]) & set(validated[1]["job_ids"]):
        raise ValueError("factorial campaigns reuse Slurm job IDs")
    if validated[0]["trip_set_sha256"] != validated[1]["trip_set_sha256"]:
        raise ValueError("factorial campaigns have mixed trip sets")
    historical_record = validate_historical(historical)
    if (
        historical_record["trip_set_sha256"]
        != validated[0]["trip_set_sha256"]
    ):
        raise ValueError("historical and factorial trip sets differ")
    accounting_sha = _validate_accounting(accounting, validated)
    sources: dict[str, Path] = {}
    watched_roots = []
    expected_source_hashes = {str(accounting): accounting_sha}
    for campaign in validated:
        expected_source_hashes.update(
            campaign["validated_file_hashes"]
        )
        campaign_root = Path(campaign["campaign_dir"])
        watched_roots.append(campaign_root)
        for source in _regular_files(campaign_root):
            member = str(
                Path("campaigns") / campaign["replicate"]
                / source.relative_to(campaign_root)
            )
            sources[member] = source
        repo_root = campaign_root.parents[3]
        campaign_logs, log_hashes = _campaign_log_files(
            campaign_root, campaign["launch"]
        )
        expected_source_hashes.update(log_hashes)
        watched_roots.append(
            repo_root / "src/cluster_logs/k40_factorial"
            / campaign_root.name
        )
        for log in campaign_logs:
            member = str(
                Path("logs") / campaign["replicate"]
                / log.relative_to(repo_root)
            )
            sources[member] = log
        sources[
            str(Path("inputs") / campaign["replicate"]
                / Path(campaign["instance_path"]).name)
        ] = Path(campaign["instance_path"])
        sources[
            str(Path("inputs") / campaign["replicate"]
                / Path(campaign["prices_path"]).name)
        ] = Path(campaign["prices_path"])
    historical_path = Path(historical_record["status_path"])
    expected_source_hashes.update(
        historical_record["validated_file_hashes"]
    )
    for source_text in historical_record["files"]:
        source = Path(source_text)
        sources[str(Path("historical") / source.name)] = source
    historical_iters = Path(str(historical_path) + ".iters.csv")
    if not historical_iters.is_file():
        raise ValueError("historical iteration trajectory is missing")
    if (
        sha256_file(historical_path)
        != historical_record["status_sha256"]
    ):
        raise RuntimeError("historical status changed before trajectory validation")
    historical_trajectory_sha = _validate_trajectory(
        historical_iters,
        json.loads(historical_path.read_text()),
    )
    expected_source_hashes[
        str(historical_iters.resolve())
    ] = historical_trajectory_sha
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
        "slurm_accounting_sha256": accounting_sha,
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
        expected_source_hashes,
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

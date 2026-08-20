#!/usr/bin/env python3
"""Create a checksummed archive of one completed profile campaign."""

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

from exact_cg_profile_results import (
    validate_campaign_manifest,
    validate_profile_payload,
)
from launch_exact_cg_profile_campaign import (
    _approval_payload,
    _approval_sha256,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    manifest_errors = validate_campaign_manifest(campaign)
    if manifest_errors:
        raise ValueError(
            "invalid campaign manifest: " + " | ".join(manifest_errors)
        )
    if campaign.get("submitted") is not True:
        raise ValueError("campaign is not fully submitted")
    for job in campaign["jobs"]:
        if job.get("submission_state") not in {
                "submitted", "submitted_reconciled"}:
            raise ValueError(f"{job['label']}: job is not submitted")
        if not job.get("job_id"):
            raise ValueError(f"{job['label']}: job ID is missing")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    log_root = Path(campaign["log_root"]).expanduser().resolve()
    if not log_root.is_dir():
        raise ValueError(f"campaign log directory is missing: {log_root}")
    try:
        output.relative_to(log_root)
    except ValueError:
        pass
    else:
        raise ValueError("archive output must be outside campaign log root")

    def collect():
        entries = []
        for prefix, source_root in (("campaign", root), ("logs", log_root)):
            entries.extend(
                (
                    f"{prefix}/{path.relative_to(source_root)}",
                    path,
                )
                for path in source_root.rglob("*") if path.is_file()
            )
        return sorted(entries)

    entries_before = collect()
    temporary_archive = None
    with tempfile.TemporaryDirectory(
            dir=output.parent,
            prefix=f".{output.name}.snapshot.") as snapshot_dir_text:
        snapshot_dir = Path(snapshot_dir_text)
        staged_entries = []
        before = {}
        for archive_name, source_path in entries_before:
            if source_path.is_symlink():
                raise ValueError(
                    f"refusing symlinked campaign artifact: {source_path}"
                )
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(source_path, flags)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError(
                        f"campaign artifact is not regular: {source_path}"
                    )
                staged_path = snapshot_dir / archive_name
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                with (
                    os.fdopen(os.dup(descriptor), "rb") as source,
                    staged_path.open("xb") as destination,
                ):
                    shutil.copyfileobj(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
            finally:
                os.close(descriptor)
            digest = sha256_file(staged_path)
            if sha256_file(source_path) != digest:
                raise RuntimeError(
                    f"campaign artifact changed while staging: {source_path}"
                )
            before[archive_name] = digest
            staged_entries.append((archive_name, staged_path))

        staged_manifest = json.loads(
            (snapshot_dir / "campaign/campaign.json").read_text()
        )
        staged_manifest_errors = validate_campaign_manifest(staged_manifest)
        if staged_manifest_errors:
            raise ValueError(
                "invalid staged campaign manifest: "
                + " | ".join(staged_manifest_errors)
            )
        if staged_manifest.get("submitted") is not True:
            raise ValueError("staged campaign is not fully submitted")
        if Path(staged_manifest["log_root"]).resolve() != log_root:
            raise RuntimeError("campaign log root changed during archival")
        for job in staged_manifest["jobs"]:
            if job.get("submission_state") not in {
                    "submitted", "submitted_reconciled"}:
                raise ValueError(
                    f"{job['label']}: staged job is not submitted"
                )
            if not job.get("job_id"):
                raise ValueError(
                    f"{job['label']}: staged job ID is missing"
                )
        campaign = staged_manifest
        if campaign.get("approval_sha256") != _approval_sha256(
                _approval_payload(campaign)):
            raise ValueError("campaign approval digest is invalid")

        def require_campaign_hash(path_text, expected, label):
            path = Path(path_text).resolve()
            try:
                relative = path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"{label} is outside campaign root: {path}"
                ) from exc
            archive_name = f"campaign/{relative}"
            if before.get(archive_name) != expected:
                raise ValueError(f"{label} hash is missing or mismatched")

        require_campaign_hash(
            campaign["worker"],
            campaign["worker_sha256"],
            "staged worker",
        )
        for job in staged_manifest["jobs"]:
            spec_path = Path(job["job_spec_path"])
            require_campaign_hash(
                spec_path, job["job_spec_sha256"], f"{job['label']} job spec"
            )
            staged_spec = json.loads(
                (snapshot_dir / (
                    "campaign/" + str(spec_path.resolve().relative_to(root))
                )).read_text()
            )
            if staged_spec != job["job_spec"]:
                raise ValueError(f"{job['label']}: staged job spec differs")
            for path_key, hash_key, description in (
                (
                    "staged_result", "staged_result_sha256",
                    "staged result",
                ),
                (
                    "staged_journal", "staged_journal_sha256",
                    "staged journal",
                ),
                (
                    "staged_instance", "staged_instance_sha256",
                    "staged instance",
                ),
                (
                    "staged_prices", "staged_prices_sha256",
                    "staged prices",
                ),
            ):
                require_campaign_hash(
                    staged_spec[path_key],
                    staged_spec[hash_key],
                    f"{job['label']} {description}",
                )
            result_path = Path(job["output"])
            result_name = f"campaign/{result_path.relative_to(root)}"
            staged_result = snapshot_dir / result_name
            if not staged_result.is_file():
                raise ValueError(f"{job['label']}: profile output is missing")
            payload = json.loads(staged_result.read_text())
            errors = validate_profile_payload(
                payload, job, staged_manifest
            )
            if errors:
                raise ValueError(
                    f"{job['label']}: invalid profile: "
                    + " | ".join(errors)
                )
            log_stem = f"{job['job_name']}_{job['job_id']}"
            stdout = snapshot_dir / f"logs/{log_stem}.out"
            stderr = snapshot_dir / f"logs/{log_stem}.err"
            if not stdout.is_file() or not stderr.is_file():
                raise ValueError(
                    f"{job['label']}: stdout/stderr logs are incomplete"
                )
            done_marker = f"[PROFILE] DONE label={job['label']} "
            if done_marker not in stdout.read_text(errors="replace"):
                raise ValueError(
                    f"{job['label']}: completion marker is missing"
                )

        record = {
            "schema": "evsp-dr-exact-cg-profile-archive-v1",
            "campaign": campaign.get("campaign"),
            "campaign_root": str(root),
            "log_root": str(log_root),
            "expected_commit": (
                campaign.get("checkout_identity") or {}
            ).get("expected_commit"),
            "profile_core_commit": campaign.get("profile_core_commit"),
            "campaign_manifest_sha256": before.get(
                "campaign/campaign.json"
            ),
            "files": before,
        }
        record_bytes = (json.dumps(record, indent=2) + "\n").encode()
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.tmp.",
            delete=False,
        ) as handle:
            temporary_archive = Path(handle.name)
        try:
            with tarfile.open(temporary_archive, "w:gz") as archive_file:
                for archive_name, staged_path in staged_entries:
                    archive_file.add(
                        staged_path,
                        arcname=archive_name,
                        recursive=False,
                    )
                info = tarfile.TarInfo("ARCHIVE_MANIFEST.json")
                info.size = len(record_bytes)
                archive_file.addfile(info, io.BytesIO(record_bytes))
            entries_after = collect()
            after = {
                archive_name: sha256_file(path)
                for archive_name, path in entries_after
            }
            if after != before or [
                    name for name, _path in entries_after
            ] != [
                    name for name, _path in entries_before
            ]:
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

    record["archive"] = str(output)
    record["archive_sha256"] = archive_sha
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

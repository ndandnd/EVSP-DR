#!/usr/bin/env python3
"""Validate and submit EVSP-DR cluster campaigns without shell placeholders.

The command is a dry run unless ``--submit`` is present.  It deliberately
owns the Slurm partition, log paths, result path, and worker arguments so an
unset shell variable cannot create a doomed allocation.

Example from the repository root::

    python src/cluster_campaign.py mip \
      --result src/results/exact_big/INSTANCE.partition_ready.snapshot.json \
      --minutes 60 --cover --submit
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from run_exact_pool_mip import resolve_pool_journal


REPO_ROOT = Path(__file__).resolve().parents[1]
MIP_WORKER = REPO_ROOT / "src" / "submit_exact_pool_mip.sub"
MIP_RUNNER = REPO_ROOT / "src" / "run_exact_pool_mip.py"


def _nonempty_json(path_text: str) -> Path:
    if not path_text.strip() or "/absolute/path/to/" in path_text:
        raise argparse.ArgumentTypeError("replace the placeholder with a real JSON path")
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise argparse.ArgumentTypeError(f"file is missing or empty: {path}")
    if not path.name.endswith(".snapshot.json"):
        raise argparse.ArgumentTypeError(
            f"expected an immutable *.snapshot.json pool: {path}"
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wall_time(minutes: int) -> str:
    total_seconds = minutes * 60 + 10 * 60
    hours, remainder = divmod(total_seconds, 3600)
    mins, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{mins:02d}:{seconds:02d}"


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _write_manifest(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def submit_mip(args: argparse.Namespace) -> int:
    result: Path = args.result
    mode = "cover" if args.cover else "partition"
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    campaign = args.campaign or f"mip_{mode}_{timestamp}"
    if campaign in {".", ".."} or not all(
        character.isalnum() or character in "_.-" for character in campaign
    ):
        raise SystemExit(
            "campaign must be a non-dot name containing only letters, numbers, "
            "dot, dash, and underscore"
        )

    campaign_parent = REPO_ROOT / "src" / "results" / "cluster_campaigns"
    campaign_root = campaign_parent / campaign
    log_dir = REPO_ROOT / "src" / "logs" / "cluster_campaigns" / campaign
    input_dir = campaign_root / "input"
    staged_result = input_dir / result.name
    source_result_bytes = result.read_bytes()
    status = json.loads(source_result_bytes)
    source_journal = resolve_pool_journal(result, status).resolve()
    staged_journal = input_dir / source_journal.name
    output = campaign_root / f"{result.stem}_{mode}_{args.minutes}m.json"
    manifest = campaign_root / "submission.json"

    if output.resolve() == result.resolve():
        raise SystemExit("refusing to overwrite the input result")
    if staged_result == staged_journal:
        raise SystemExit("snapshot and journal names collide")
    if campaign_root.exists() or log_dir.exists():
        raise SystemExit(
            f"campaign or log directory already exists; choose a new name: {campaign}"
        )

    validation = [
        sys.executable,
        str(MIP_RUNNER),
        "--result",
        str(result),
        "--validate-only",
    ]
    if not args.cover:
        validation.append("--require-singleton-partition")
    print("[preflight]", " ".join(validation), flush=True)
    _run_checked(validation)

    wall = _wall_time(args.minutes)
    sbatch = [
        "sbatch",
        "--parsable",
        "--partition=scaglione",
        "--no-requeue",
        f"--time={wall}",
        f"--job-name=EXACTMIP-{mode}",
        f"--output={log_dir}/%x_%j.out",
        f"--error={log_dir}/%x_%j.err",
        "--export=ALL,EVSP_DR_ROOT=" + str(REPO_ROOT) +
        ",EXACT_MIP_COVER=" + ("1" if args.cover else "0"),
        str(MIP_WORKER),
        str(staged_result),
        str(args.minutes * 60),
        str(output),
    ]

    source_result_sha256 = hashlib.sha256(source_result_bytes).hexdigest()
    source_journal_sha256 = _sha256(source_journal)
    record = {
        "campaign": campaign,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "mode": mode,
        "minutes": args.minutes,
        "partition": "scaglione",
        "requeue": False,
        "source_result": str(result),
        "source_result_sha256": source_result_sha256,
        "source_journal": str(source_journal),
        "source_journal_sha256": source_journal_sha256,
        "input_result": str(staged_result),
        "input_result_sha256": None,
        "input_journal": str(staged_journal),
        "input_journal_sha256": None,
        "output": str(output),
        "logs": str(log_dir),
        "command": sbatch,
        "submitted": False,
        "job_id": None,
    }
    print("[slurm]", " ".join(sbatch), flush=True)
    if not args.submit:
        print(
            "[dry-run] immutable snapshot validated; add --submit to stage and enqueue",
            flush=True,
        )
        return 0

    campaign_parent.mkdir(parents=True, exist_ok=True)
    campaign_root.mkdir(exist_ok=False)
    input_dir.mkdir()
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_journal, staged_journal)
    staged_status = dict(status)
    staged_status["columns_journal"] = str(staged_journal)
    staged_result.write_text(json.dumps(staged_status, indent=2) + "\n")

    # A snapshot is expected to be immutable. Refuse submission if either
    # source changed while its campaign copy was being staged.
    if (_sha256(result) != source_result_sha256 or
            _sha256(source_journal) != source_journal_sha256 or
            _sha256(staged_journal) != source_journal_sha256):
        record["staging_error"] = "source snapshot or journal changed while staging"
        _write_manifest(manifest, record)
        raise SystemExit(record["staging_error"])

    record["input_result_sha256"] = _sha256(staged_result)
    record["input_journal_sha256"] = _sha256(staged_journal)
    staged_validation = list(validation)
    staged_validation[staged_validation.index(str(result))] = str(staged_result)
    print("[staged-preflight]", " ".join(staged_validation), flush=True)
    try:
        _run_checked(staged_validation)
    except subprocess.CalledProcessError as exc:
        record["staging_error"] = f"staged preflight failed with exit {exc.returncode}"
        _write_manifest(manifest, record)
        raise

    _write_manifest(manifest, record)
    try:
        completed = subprocess.run(
            sbatch,
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        record["submission_error"] = (exc.stderr or str(exc)).strip()
        _write_manifest(manifest, record)
        raise SystemExit(f"sbatch failed: {record['submission_error']}") from exc
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id:
        record["submission_error"] = "sbatch returned no job id"
        _write_manifest(manifest, record)
        raise SystemExit(record["submission_error"])
    record["submitted"] = True
    record["job_id"] = job_id
    _write_manifest(manifest, record)
    print(f"[submitted] job={job_id} manifest={manifest}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    mip = subparsers.add_parser("mip", help="validate and submit one exact-pool MIP")
    mip.add_argument("--result", type=_nonempty_json, required=True)
    mip.add_argument("--minutes", type=int, required=True)
    mip.add_argument("--cover", action="store_true")
    mip.add_argument("--campaign")
    mip.add_argument("--submit", action="store_true")
    mip.set_defaults(handler=submit_mip)
    args = parser.parse_args(argv)
    if getattr(args, "minutes", 1) <= 0:
        parser.error("--minutes must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

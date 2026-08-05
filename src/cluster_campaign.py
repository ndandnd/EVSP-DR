#!/usr/bin/env python3
"""Validate and submit EVSP-DR cluster campaigns without shell placeholders.

The command is a dry run unless ``--submit`` is present.  It deliberately
owns the Slurm partition, log paths, result path, and worker arguments so an
unset shell variable cannot create a doomed allocation.

Example from the repository root::

    python src/cluster_campaign.py mip \
      --result src/results/exact_big/INSTANCE.json \
      --minutes 60 --cover --submit
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path


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
    if path.suffix != ".json":
        raise argparse.ArgumentTypeError(f"expected a .json result: {path}")
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


def submit_mip(args: argparse.Namespace) -> int:
    result: Path = args.result
    mode = "cover" if args.cover else "partition"
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    campaign = args.campaign or f"mip_{mode}_{timestamp}"
    if not all(character.isalnum() or character in "_.-" for character in campaign):
        raise SystemExit("campaign may contain only letters, numbers, dot, dash, underscore")

    campaign_root = REPO_ROOT / "src" / "results" / "cluster_campaigns" / campaign
    log_dir = REPO_ROOT / "src" / "logs" / "cluster_campaigns" / campaign
    output = campaign_root / f"{result.stem}_{mode}_{args.minutes}m.json"
    manifest = campaign_root / "submission.json"

    if output.resolve() == result.resolve():
        raise SystemExit("refusing to overwrite the input result")
    if output.exists() and not args.allow_existing_output:
        raise SystemExit(f"output already exists: {output}")

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
        str(result),
        str(args.minutes * 60),
        str(output),
    ]

    record = {
        "campaign": campaign,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "mode": mode,
        "minutes": args.minutes,
        "partition": "scaglione",
        "requeue": False,
        "input_result": str(result),
        "input_sha256": _sha256(result),
        "output": str(output),
        "logs": str(log_dir),
        "command": sbatch,
        "submitted": False,
        "job_id": None,
    }
    print("[slurm]", " ".join(sbatch), flush=True)
    if not args.submit:
        print("[dry-run] validated successfully; add --submit to enqueue", flush=True)
        return 0

    campaign_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        sbatch,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    job_id = completed.stdout.strip().split(";", 1)[0]
    record["submitted"] = True
    record["job_id"] = job_id
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(manifest)
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
    mip.add_argument("--allow-existing-output", action="store_true")
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

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
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from run_exact_pool_mip import resolve_pool_journal


REPO_ROOT = Path(__file__).resolve().parents[1]
MIP_WORKER = REPO_ROOT / "src" / "submit_exact_pool_mip.sub"
MIP_RUNNER = REPO_ROOT / "src" / "run_exact_pool_mip.py"
DEFAULT_MIP_GAP = 1e-4


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


def _nonempty_routes_json(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise argparse.ArgumentTypeError(f"file is missing or empty: {path}")
    if path.suffix.lower() != ".json":
        raise argparse.ArgumentTypeError(
            f"expected a routes JSON file: {path}"
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
    environment = os.environ.copy()
    for name in (
        "EVSP_EXPECTED_COMMIT",
        "EVSP_REQUIRE_DETACHED",
        "SLURM_JOB_ID",
        "EVSP_MIP_EXPECTED_RESULT_SHA256",
        "EVSP_MIP_EXPECTED_JOURNAL_SHA256",
        "EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256",
    ):
        environment.pop(name, None)
    subprocess.run(
        command, cwd=REPO_ROOT, check=True, env=environment
    )


def _reviewed_checkout_identity(repo_root: Path) -> dict:
    """Require an immutable, detached, tracked-clean reviewed checkout."""

    def git(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=False,
            text=True,
            capture_output=True,
        )

    head = git("rev-parse", "--verify", "HEAD")
    commit = head.stdout.strip()
    if (head.returncode != 0 or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise SystemExit("MIP launcher has no verifiable 40-character Git HEAD")
    symbolic = git("symbolic-ref", "-q", "HEAD")
    if symbolic.returncode == 0:
        raise SystemExit(
            "MIP submission requires an immutable detached checkout; "
            f"found {symbolic.stdout.strip()}"
        )
    if symbolic.returncode != 1:
        raise SystemExit("could not verify that MIP launcher HEAD is detached")
    tracked = git("status", "--porcelain", "--untracked-files=no")
    if tracked.returncode != 0:
        raise SystemExit("could not verify MIP launcher worktree state")
    if tracked.stdout.strip():
        raise SystemExit(
            "MIP launcher checkout has tracked modifications; commit them "
            "and use a clean detached checkout"
        )
    return {
        "expected_commit": commit,
        "observed_commit": commit,
        "detached": True,
        "tracked_clean": True,
    }


def _reviewed_git_blob(
    repo_root: Path, commit: str, relative_path: str,
) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        raise SystemExit(
            f"could not read reviewed Git blob {commit}:{relative_path}"
        )
    return result.stdout


def _write_manifest(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def _mip_arm(two_stage: bool, validated_start: bool) -> str:
    return {
        (False, False): "A",
        (True, False): "B",
        (False, True): "C",
        (True, True): "D",
    }[(two_stage, validated_start)]


def _mip_job_name(
    status: dict,
    mode: str,
    minutes: int,
    *,
    two_stage: bool = False,
    validated_start: bool = False,
) -> str:
    """Build a compact name that exposes the MIP's scientific configuration."""

    source = str(status.get("csv", ""))
    match = re.search(r"k0*(\d+)_r0*(\d+)", source, flags=re.IGNORECASE)
    if match:
        case = f"{int(match.group(1)):02d}r{int(match.group(2))}"
    else:
        bus_match = re.search(r"(\d+)bus", source, flags=re.IGNORECASE)
        if bus_match:
            case = f"b{int(bus_match.group(1))}"
        else:
            case = "x" + hashlib.sha256(source.encode()).hexdigest()[:3]
    if len(case) > 5:
        case = "x" + hashlib.sha256(source.encode()).hexdigest()[:3]

    try:
        battery = str(int(round(float(status["g_kwh"]) / 10)))
        reserve = str(int(round(float(status["min_soc_frac"]) * 10)))
    except (KeyError, TypeError, ValueError):
        battery, reserve = "x", "x"

    mode_tag = "C" if mode == "cover" else "P"
    arm_tag = _mip_arm(two_stage, validated_start)
    duration = f"T{minutes}"
    name = f"M{mode_tag}{arm_tag}{case}G{battery}R{reserve}{duration}"
    if len(name) > 15:
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        value = minutes
        encoded = ""
        while value:
            value, remainder = divmod(value, 36)
            encoded = digits[remainder] + encoded
        if minutes % 60 == 0:
            hours = minutes // 60
            encoded = ""
            while hours:
                hours, remainder = divmod(hours, 36)
                encoded = digits[remainder] + encoded
            duration = "H" + (encoded or "0")
        else:
            duration = "Z" + (encoded or "0")
        name = f"M{mode_tag}{arm_tag}{case}G{battery}R{reserve}{duration}"
    if len(name) > 15:
        duration = "Q" + hashlib.sha256(str(minutes).encode()).hexdigest()[:2]
        name = f"M{mode_tag}{arm_tag}{case}G{battery}R{reserve}{duration}"
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]{0,14}", name):
        raise SystemExit(f"could not build a <=15-character MIP job name: {name}")
    return name


def submit_mip(args: argparse.Namespace) -> int:
    result: Path = args.result
    checkout_identity = _reviewed_checkout_identity(REPO_ROOT)
    expected_commit = checkout_identity["expected_commit"]
    reviewed_worker_bytes = _reviewed_git_blob(
        REPO_ROOT, expected_commit, "src/submit_exact_pool_mip.sub"
    )
    reviewed_worker_sha256 = hashlib.sha256(
        reviewed_worker_bytes
    ).hexdigest()
    reviewed_runner_sha256 = hashlib.sha256(_reviewed_git_blob(
        REPO_ROOT, expected_commit, "src/run_exact_pool_mip.py"
    )).hexdigest()
    if _sha256(MIP_WORKER) != reviewed_worker_sha256:
        raise SystemExit("working-tree MIP worker differs from reviewed Git blob")
    if _sha256(MIP_RUNNER) != reviewed_runner_sha256:
        raise SystemExit("working-tree MIP runner differs from reviewed Git blob")
    two_stage = bool(getattr(args, "two_stage", False))
    initial_partition = getattr(args, "initial_partition_routes", None)
    experiment_arm = _mip_arm(
        two_stage, initial_partition is not None
    )
    mip_gap = float(getattr(args, "mip_gap", DEFAULT_MIP_GAP))
    if not math.isfinite(mip_gap) or not 0.0 <= mip_gap < 1.0:
        raise SystemExit("--mip-gap must be finite and in [0, 1)")
    mip_gap_text = format(mip_gap, ".17g")
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
    staged_worker = input_dir / "submit_exact_pool_mip.reviewed.sub"
    staged_initial_partition = (
        input_dir / f"initial_partition_{initial_partition.name}"
        if initial_partition is not None else None
    )
    output = campaign_root / f"{result.stem}_{mode}_{args.minutes}m.json"
    manifest = campaign_root / "submission.json"

    if output.resolve() == result.resolve():
        raise SystemExit("refusing to overwrite the input result")
    if staged_result == staged_journal:
        raise SystemExit("snapshot and journal names collide")
    if (staged_initial_partition is not None
            and staged_initial_partition in {staged_result, staged_journal}):
        raise SystemExit("staged initial partition name collides with pool inputs")
    if campaign_root.exists() or log_dir.exists():
        raise SystemExit(
            f"campaign or log directory already exists; choose a new name: {campaign}"
        )

    source_result_sha256 = hashlib.sha256(source_result_bytes).hexdigest()
    source_journal_sha256 = _sha256(source_journal)
    source_initial_partition_sha256 = (
        _sha256(initial_partition)
        if initial_partition is not None else None
    )
    staged_status = dict(status)
    staged_status["columns_journal"] = str(staged_journal)
    staged_result_bytes = (
        json.dumps(staged_status, indent=2) + "\n"
    ).encode()
    expected_input_result_sha256 = hashlib.sha256(
        staged_result_bytes
    ).hexdigest()

    validation = [
        sys.executable,
        str(MIP_RUNNER),
        "--result",
        str(result),
        "--validate-only",
    ]
    if not args.cover:
        validation.append("--require-singleton-partition")
    if initial_partition is not None:
        # The supplied exact partition is a stronger feasibility witness than
        # singleton fallback, and the runner validates every route physically.
        if "--require-singleton-partition" in validation:
            validation.remove("--require-singleton-partition")
        validation.extend([
            "--initial-partition-routes", str(initial_partition),
        ])
    print("[preflight]", " ".join(validation), flush=True)
    _run_checked(validation)

    wall = _wall_time(args.minutes)
    job_name = _mip_job_name(
        status,
        mode,
        args.minutes,
        two_stage=two_stage,
        validated_start=initial_partition is not None,
    )
    # Do not include ALL: Slurm gives inherited values precedence over explicit
    # assignments when ALL is present, which could silently change the arm.
    export_value = (
        "HOME,PATH,USER,EVSP_DR_ROOT=" + str(REPO_ROOT)
        + ",EXACT_MIP_COVER=" + ("1" if args.cover else "0")
        + ",EXACT_MIP_TWO_STAGE=" + ("1" if two_stage else "0")
        + ",EVSP_EXPECTED_COMMIT=" + expected_commit
        + ",EVSP_REQUIRE_DETACHED=1"
        + ",EVSP_MIP_EXPECTED_WORKER_SHA256=" + reviewed_worker_sha256
        + ",EVSP_MIP_EXPECTED_RUNNER_SHA256=" + reviewed_runner_sha256
        + ",EVSP_MIP_EXPECTED_RESULT_SHA256="
        + expected_input_result_sha256
        + ",EVSP_MIP_EXPECTED_JOURNAL_SHA256="
        + source_journal_sha256
        + ",EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256="
        + (source_initial_partition_sha256 or "")
    )
    sbatch = [
        "sbatch",
        "--parsable",
        "--partition=scaglione",
        "--no-requeue",
        f"--time={wall}",
        f"--job-name={job_name}",
        f"--output={log_dir}/%x_%j.out",
        f"--error={log_dir}/%x_%j.err",
        "--export=" + export_value,
        str(staged_worker),
        str(staged_result),
        str(args.minutes * 60),
        str(output),
        mip_gap_text,
    ]
    if staged_initial_partition is not None:
        sbatch.append(str(staged_initial_partition))

    record = {
        "campaign": campaign,
        "job_name": job_name,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "mode": mode,
        "experiment_arm": experiment_arm,
        "two_stage": two_stage,
        "requested_mip_gap": mip_gap,
        "checkout_identity": checkout_identity,
        "reviewed_worker_git_path": "src/submit_exact_pool_mip.sub",
        "reviewed_worker_sha256": reviewed_worker_sha256,
        "reviewed_runner_git_path": "src/run_exact_pool_mip.py",
        "reviewed_runner_sha256": reviewed_runner_sha256,
        "input_worker": str(staged_worker),
        "input_worker_sha256": reviewed_worker_sha256,
        "expected_git_commit": expected_commit,
        "launcher_observed_git_commit": checkout_identity["observed_commit"],
        "minutes": args.minutes,
        "partition": "scaglione",
        "requeue": False,
        "source_result": str(result),
        "source_result_sha256": source_result_sha256,
        "source_journal": str(source_journal),
        "source_journal_sha256": source_journal_sha256,
        "input_result": str(staged_result),
        "input_result_sha256": expected_input_result_sha256,
        "input_journal": str(staged_journal),
        "input_journal_sha256": source_journal_sha256,
        "initial_partition_source": (
            str(initial_partition) if initial_partition is not None else None
        ),
        "initial_partition_source_sha256": (
            source_initial_partition_sha256
        ),
        "input_initial_partition": (
            str(staged_initial_partition)
            if staged_initial_partition is not None else None
        ),
        "input_initial_partition_sha256": (
            source_initial_partition_sha256
        ),
        "output": str(output),
        "logs": str(log_dir),
        "command": sbatch,
        "submission_state": "planned",
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
    staged_worker.write_bytes(reviewed_worker_bytes)
    staged_worker.chmod(0o500)
    shutil.copyfile(source_journal, staged_journal)
    if staged_initial_partition is not None:
        shutil.copyfile(initial_partition, staged_initial_partition)
    staged_result.write_bytes(staged_result_bytes)

    # A snapshot is expected to be immutable. Refuse submission if either
    # source changed while its campaign copy was being staged.
    initial_partition_changed = (
        initial_partition is not None
        and (
            _sha256(initial_partition) != source_initial_partition_sha256
            or _sha256(staged_initial_partition)
            != source_initial_partition_sha256
        )
    )
    if (_sha256(result) != source_result_sha256 or
            _sha256(source_journal) != source_journal_sha256 or
            _sha256(staged_journal) != source_journal_sha256 or
            _sha256(staged_worker) != reviewed_worker_sha256 or
            initial_partition_changed):
        record["staging_error"] = (
            "source snapshot, journal, reviewed worker, or initial partition "
            "changed while staging"
        )
        _write_manifest(manifest, record)
        raise SystemExit(record["staging_error"])

    record["input_result_sha256"] = _sha256(staged_result)
    record["input_journal_sha256"] = _sha256(staged_journal)
    if staged_initial_partition is not None:
        record["input_initial_partition_sha256"] = _sha256(
            staged_initial_partition
        )
    staged_validation = list(validation)
    staged_validation[staged_validation.index(str(result))] = str(staged_result)
    if initial_partition is not None:
        staged_validation[
            staged_validation.index(str(initial_partition))
        ] = str(staged_initial_partition)
    print("[staged-preflight]", " ".join(staged_validation), flush=True)
    try:
        _run_checked(staged_validation)
    except subprocess.CalledProcessError as exc:
        record["staging_error"] = f"staged preflight failed with exit {exc.returncode}"
        _write_manifest(manifest, record)
        raise

    pre_submission_identity = _reviewed_checkout_identity(REPO_ROOT)
    if pre_submission_identity["observed_commit"] != expected_commit:
        record["staging_error"] = (
            "launcher commit changed between preflight and submission"
        )
        _write_manifest(manifest, record)
        raise SystemExit(record["staging_error"])
    record["pre_submission_observed_git_commit"] = (
        pre_submission_identity["observed_commit"]
    )
    record["submission_state"] = "attempting"
    record["submission_attempted_at"] = (
        dt.datetime.now().astimezone().isoformat()
    )
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
        record["submission_state"] = "failed"
        _write_manifest(manifest, record)
        raise SystemExit(f"sbatch failed: {record['submission_error']}") from exc
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id:
        record["submission_error"] = "sbatch returned no job id"
        record["submission_state"] = "failed"
        _write_manifest(manifest, record)
        raise SystemExit(record["submission_error"])
    record["submitted"] = True
    record["job_id"] = job_id
    record["submission_state"] = "submitted"
    _write_manifest(manifest, record)
    print(f"[submitted] job={job_id} manifest={manifest}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    mip = subparsers.add_parser("mip", help="validate and submit one exact-pool MIP")
    mip.add_argument("--result", type=_nonempty_json, required=True)
    mip.add_argument("--minutes", type=int, required=True)
    mip.add_argument(
        "--mip-gap",
        type=float,
        default=DEFAULT_MIP_GAP,
        help=f"Explicit relative MIP gap (default: {DEFAULT_MIP_GAP:g}).",
    )
    mip.add_argument("--cover", action="store_true")
    mip.add_argument(
        "--two-stage",
        action="store_true",
        help="Minimize fleet first, then cost only after fleet proof.",
    )
    mip.add_argument(
        "--initial-partition-routes",
        type=_nonempty_routes_json,
        help="Runner-format routes JSON that must validate as an exact "
             "partition and becomes the explicit MIP start.",
    )
    mip.add_argument("--campaign")
    mip.add_argument("--submit", action="store_true")
    mip.set_defaults(handler=submit_mip)
    args = parser.parse_args(argv)
    if getattr(args, "minutes", 1) <= 0:
        parser.error("--minutes must be positive")
    mip_gap = getattr(args, "mip_gap", DEFAULT_MIP_GAP)
    if not math.isfinite(mip_gap) or not 0.0 <= mip_gap < 1.0:
        parser.error("--mip-gap must be finite and in [0, 1)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

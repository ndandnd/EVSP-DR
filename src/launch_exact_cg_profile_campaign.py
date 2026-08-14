#!/usr/bin/env python3
"""Stage and optionally submit the five-pool exact-CG profiling campaign.

Dry-run is the default.  Submission requires an explicit ``--submit`` and one
new campaign name/output namespace; retries must use a fresh campaign.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from durable_io import flush_and_fsync
from run_exact_pool_mip import resolve_pool_journal


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_CORE_COMMIT = "702491e2b9fa548b75a8b140ba5a4213c06df24f"
WORKER_GIT_PATH = "src/submit_exact_cg_profile.sub"
LABEL_ARGUMENTS = (
    ("historical", "historical"),
    ("ca", "ca"),
    ("cs", "cs"),
    ("pa", "pa"),
    ("ps", "ps"),
)
JOB_TAGS = {
    "historical": "hist",
    "ca": "ca",
    "cs": "cs",
    "pa": "pa",
    "ps": "ps",
}
PREFIXES = [1000, 5000, 10000, 25000, 50000]
METHODS = ["highs", "highs-ds", "highs-ipm"]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, binary: bool = False):
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=not binary,
    )


def reviewed_checkout_identity() -> dict:
    head = _git("rev-parse", "--verify", "HEAD")
    commit = head.stdout.strip()
    if (head.returncode != 0 or len(commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in commit)):
        raise SystemExit("profile launcher has no verifiable Git HEAD")
    symbolic = _git("symbolic-ref", "-q", "HEAD")
    if symbolic.returncode == 0:
        raise SystemExit(
            "profile launcher requires a detached checkout"
        )
    if symbolic.returncode != 1:
        raise SystemExit("cannot verify detached profile checkout")
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    if tracked.returncode != 0 or tracked.stdout.strip():
        raise SystemExit(
            "profile launcher checkout is not tracked-clean"
        )
    ancestor = _git("merge-base", "--is-ancestor", PROFILE_CORE_COMMIT, "HEAD")
    if ancestor.returncode != 0:
        raise SystemExit("reviewed profiler commit is not an ancestor of HEAD")
    core_paths = (
        "src/profile_exact_pool_prefixes.py",
        "src/exact_pricer_expanded.py",
        "src/exact_cg_telemetry.py",
        "src/master_lp_scipy.py",
    )
    changed = _git("diff", "--quiet", PROFILE_CORE_COMMIT, "--", *core_paths)
    if changed.returncode != 0:
        raise SystemExit(
            f"profiler core differs from reviewed commit {PROFILE_CORE_COMMIT}"
        )
    return {
        "expected_commit": commit,
        "observed_commit": commit,
        "detached": True,
        "tracked_clean": True,
        "profile_core_commit": PROFILE_CORE_COMMIT,
    }


def reviewed_worker_bytes(commit: str) -> bytes:
    result = _git("show", f"{commit}:{WORKER_GIT_PATH}", binary=True)
    if result.returncode != 0 or not result.stdout:
        raise SystemExit("cannot read reviewed profile worker Git blob")
    return result.stdout


def validated_python(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"profile Python is missing/not executable: {path}")
    check = subprocess.run(
        [
            str(path),
            "-c",
            "import sys; assert sys.version_info[:2] == (3, 12); "
            "import numpy, pandas, scipy; print(sys.version.split()[0])",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if check.returncode != 0:
        raise SystemExit(
            f"profile environment validation failed: "
            f"{(check.stderr or check.stdout).strip()}"
        )
    return {"path": str(path), "version": check.stdout.strip()}


def _safe_relative(value: str, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SystemExit(f"unsafe {label} path in snapshot: {value}")
    return path


def _find_input(snapshot: Path, relative: Path) -> Path:
    candidates = [REPO_ROOT / "data" / relative]
    candidates.extend(
        parent / "data" / relative for parent in snapshot.parents
    )
    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"cannot locate input data/{relative} for {snapshot}"
    )


def _wall_time(hours: int) -> str:
    days, remainder = divmod(hours, 24)
    return f"{days}-{remainder:02d}:00:00" if days else f"{hours:02d}:00:00"


def _write_new(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            flush_and_fsync(handle)
    except FileExistsError as exc:
        raise SystemExit(f"refusing to overwrite staged file: {path}") from exc
    if executable:
        path.chmod(0o500)


def _snapshot_plan(
    label: str,
    source_result: Path,
    *,
    campaign_root: Path,
    expected_commit: str,
    python_info: dict,
    solve_limit_s: float,
    repeat: int,
) -> dict:
    source_result = source_result.expanduser().resolve()
    if not source_result.name.endswith(".snapshot.json"):
        raise SystemExit(f"{label}: expected immutable *.snapshot.json")
    source_result_raw = source_result.read_bytes()
    status = json.loads(source_result_raw)
    if not isinstance(status, dict):
        raise SystemExit(f"{label}: snapshot status is not an object")
    source_journal = resolve_pool_journal(source_result, status).resolve()
    csv_relative = _safe_relative(status["csv"], "csv")
    prices_relative = _safe_relative(status["prices_csv"], "prices")
    source_instance = _find_input(source_result, csv_relative)
    source_prices = _find_input(source_result, prices_relative)
    source_hashes = {
        "result": sha256_bytes(source_result_raw),
        "journal": sha256_file(source_journal),
        "instance": sha256_file(source_instance),
        "prices": sha256_file(source_prices),
    }
    provenance = status.get("provenance") or {}
    if (provenance.get("instance_sha256") != source_hashes["instance"]
            or provenance.get("prices_sha256") != source_hashes["prices"]):
        raise SystemExit(f"{label}: data bytes do not match provenance")

    cell_root = campaign_root / "input" / label
    staged_result = cell_root / source_result.name
    staged_journal = cell_root / source_journal.name
    staged_instance = cell_root / "data" / csv_relative
    staged_prices = cell_root / "data" / prices_relative
    output = campaign_root / "outputs" / f"{label}.prefix-profile.json"
    job_spec_path = cell_root / "job.json"
    staged_status = dict(status)
    staged_status["columns_journal"] = str(staged_journal)
    staged_result_raw = (
        json.dumps(staged_status, indent=2) + "\n"
    ).encode()
    job_spec = {
        "label": label,
        "source_result": str(source_result),
        "source_hashes": source_hashes,
        "staged_result": str(staged_result),
        "staged_result_sha256": sha256_bytes(staged_result_raw),
        "staged_journal": str(staged_journal),
        "staged_journal_sha256": source_hashes["journal"],
        "staged_instance": str(staged_instance),
        "staged_instance_sha256": source_hashes["instance"],
        "csv": str(csv_relative),
        "staged_prices": str(staged_prices),
        "staged_prices_sha256": source_hashes["prices"],
        "prices_csv": str(prices_relative),
        "output": str(output),
        "prefixes": PREFIXES,
        "methods": METHODS,
        "repeat": repeat,
        "time_limit_s": solve_limit_s,
        "expected_commit": expected_commit,
        "profile_core_commit": PROFILE_CORE_COMMIT,
        "python": python_info,
    }
    job_spec_raw = (json.dumps(job_spec, indent=2) + "\n").encode()
    return {
        "label": label,
        "source_result": str(source_result),
        "source_journal": str(source_journal),
        "source_instance": str(source_instance),
        "source_prices": str(source_prices),
        "source_hashes": source_hashes,
        "staged_result": str(staged_result),
        "staged_journal": str(staged_journal),
        "staged_instance": str(staged_instance),
        "staged_prices": str(staged_prices),
        "staged_result_bytes": staged_result_raw,
        "job_spec": job_spec,
        "job_spec_path": str(job_spec_path),
        "job_spec_bytes": job_spec_raw,
        "job_spec_sha256": sha256_bytes(job_spec_raw),
        "output": str(output),
        "job_id": None,
        "submission_state": "planned",
    }


def launch(args) -> dict:
    identity = reviewed_checkout_identity()
    python_info = validated_python(args.python)
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    campaign = args.campaign or f"exact_cg_profile_{timestamp}"
    if campaign in {".", ".."} or not all(
            character.isalnum() or character in "_.-"
            for character in campaign):
        raise SystemExit("invalid campaign name")
    campaign_root = (
        REPO_ROOT / "src" / "results" / "exact_cg_profiles" / campaign
    )
    log_root = (
        REPO_ROOT / "src" / "logs" / "exact_cg_profiles" / campaign
    )
    if campaign_root.exists() or log_root.exists():
        raise SystemExit(
            f"campaign already exists; retries need a fresh name: {campaign}"
        )
    snapshots = {
        label: getattr(args, argument)
        for label, argument in LABEL_ARGUMENTS
    }
    plans = [
        _snapshot_plan(
            label,
            snapshots[label],
            campaign_root=campaign_root,
            expected_commit=identity["expected_commit"],
            python_info=python_info,
            solve_limit_s=args.solve_limit_s,
            repeat=args.repeat,
        )
        for label, _argument in LABEL_ARGUMENTS
    ]
    outputs = [plan["output"] for plan in plans]
    if len(outputs) != len(set(outputs)):
        raise SystemExit("profile outputs are not unique")

    worker_bytes = reviewed_worker_bytes(identity["expected_commit"])
    worker_sha = sha256_bytes(worker_bytes)
    staged_worker = campaign_root / "input" / "submit_exact_cg_profile.sub"
    for label, value in (
        ("repository path", str(REPO_ROOT)),
        ("Python path", python_info["path"]),
    ):
        if "," in value:
            raise SystemExit(
                f"{label} contains a comma and is unsafe for Slurm export"
            )
    for plan in plans:
        short = identity["expected_commit"][:6]
        job_name = f"PF{JOB_TAGS[plan['label']]}-{short}"
        if len(job_name) > 15:
            raise SystemExit(f"profile job name exceeds 15 characters: {job_name}")
        export = (
            "HOME,PATH,USER,EVSP_DR_ROOT=" + str(REPO_ROOT)
            + ",EVSP_PROFILE_PYTHON=" + python_info["path"]
            + ",EVSP_EXPECTED_COMMIT=" + identity["expected_commit"]
            + ",EVSP_PROFILE_EXPECTED_WORKER_SHA256=" + worker_sha
        )
        command = [
            "sbatch",
            "--parsable",
            "--partition=default_partition",
            "--no-requeue",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            f"--mem={args.mem_gb}G",
            f"--time={_wall_time(args.job_hours)}",
            f"--job-name={job_name}",
            f"--output={log_root}/%x_%j.out",
            f"--error={log_root}/%x_%j.err",
            "--export=" + export,
            str(staged_worker),
            plan["job_spec_path"],
            plan["job_spec_sha256"],
        ]
        plan["job_name"] = job_name
        plan["command"] = command

    manifest = {
        "schema": "evsp-dr-exact-cg-profile-campaign-v1",
        "campaign": campaign,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "submitted": False,
        "checkout_identity": identity,
        "profile_core_commit": PROFILE_CORE_COMMIT,
        "worker": str(staged_worker),
        "worker_sha256": worker_sha,
        "python": python_info,
        "resources": {
            "partition": "default_partition",
            "cpus": 1,
            "mem_gb": args.mem_gb,
            "job_hours": args.job_hours,
            "blas_openmp_threads": 1,
            "requeue": False,
        },
        "profiler": {
            "prefixes": PREFIXES,
            "methods": METHODS,
            "repeat": args.repeat,
            "per_solve_time_limit_s": args.solve_limit_s,
            "phase_telemetry": False,
        },
        "jobs": plans,
    }
    for plan in plans:
        print("[profile-plan]", " ".join(plan["command"]))
    if not args.submit:
        print("[dry-run] validated five pools; add --submit only after review")
        return manifest

    campaign_root.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=False)
    _write_new(staged_worker, worker_bytes, executable=True)
    for plan in plans:
        staged_result = Path(plan["staged_result"])
        _write_new(staged_result, plan["staged_result_bytes"])
        _write_new(
            Path(plan["staged_journal"]),
            Path(plan["source_journal"]).read_bytes(),
        )
        _write_new(
            Path(plan["staged_instance"]),
            Path(plan["source_instance"]).read_bytes(),
        )
        _write_new(
            Path(plan["staged_prices"]),
            Path(plan["source_prices"]).read_bytes(),
        )
        _write_new(Path(plan["job_spec_path"]), plan["job_spec_bytes"])
        for source_key, staged_key, hash_key in (
            ("source_journal", "staged_journal", "journal"),
            ("source_instance", "staged_instance", "instance"),
            ("source_prices", "staged_prices", "prices"),
        ):
            if sha256_file(Path(plan[staged_key])) != plan["source_hashes"][hash_key]:
                raise SystemExit(f"{plan['label']}: staged {staged_key} hash mismatch")
            if sha256_file(Path(plan[source_key])) != plan["source_hashes"][hash_key]:
                raise SystemExit(f"{plan['label']}: source changed while staging")
        if sha256_file(Path(plan["source_result"])) != plan["source_hashes"]["result"]:
            raise SystemExit(f"{plan['label']}: source result changed while staging")
        if (sha256_file(Path(plan["staged_result"]))
                != plan["job_spec"]["staged_result_sha256"]):
            raise SystemExit(f"{plan['label']}: staged result hash mismatch")
        if sha256_file(Path(plan["job_spec_path"])) != plan["job_spec_sha256"]:
            raise SystemExit(f"{plan['label']}: job spec hash mismatch")
        plan.pop("staged_result_bytes")
        plan.pop("job_spec_bytes")

    manifest_path = campaign_root / "campaign.json"
    _write_new(
        manifest_path,
        (json.dumps(manifest, indent=2) + "\n").encode(),
    )
    for plan in plans:
        plan["submission_state"] = "attempting"
        _write_new(
            campaign_root / f".{plan['label']}.attempt.json",
            (json.dumps({
                "label": plan["label"],
                "attempted_at": dt.datetime.now().astimezone().isoformat(),
                "command": plan["command"],
            }, indent=2) + "\n").encode(),
        )
        completed = subprocess.run(
            plan["command"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            plan["submission_state"] = "failed"
            plan["submission_error"] = (
                completed.stderr or completed.stdout
            ).strip()
            _replace_manifest(manifest_path, manifest)
            raise SystemExit(
                f"{plan['label']}: sbatch failed: {plan['submission_error']}"
            )
        job_id = completed.stdout.strip().split(";", 1)[0]
        if not job_id:
            raise SystemExit(f"{plan['label']}: sbatch returned no job id")
        plan["job_id"] = job_id
        plan["submission_state"] = "submitted"
        _replace_manifest(manifest_path, manifest)
    manifest["submitted"] = True
    _replace_manifest(manifest_path, manifest)
    print(f"[submitted] campaign={campaign} manifest={manifest_path}")
    return manifest


def _replace_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        flush_and_fsync(handle)
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for _label, argument in LABEL_ARGUMENTS:
        parser.add_argument(
            f"--{argument}",
            type=Path,
            required=True,
            help=f"Immutable {argument.upper()} snapshot JSON.",
        )
    parser.add_argument("--campaign")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Absolute Python 3.12 environment executable.",
    )
    parser.add_argument("--solve-limit-s", type=float, default=1800.0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--mem-gb", type=int, default=64)
    parser.add_argument("--job-hours", type=int, default=24)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.solve_limit_s)
            or args.solve_limit_s <= 0.0):
        parser.error("--solve-limit-s must be positive and finite")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if not 40 <= args.mem_gb <= 64:
        parser.error("--mem-gb must be in [40, 64]")
    if args.job_hours <= 0:
        parser.error("--job-hours must be positive")
    return args


def main(argv=None) -> int:
    launch(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

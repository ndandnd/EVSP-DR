#!/usr/bin/env python3
"""Resolve and submit EVSP-DR Slurm tasks with semantic job names.

Array workers keep their task-index interface. The launcher submits one held
array, assigns every element its semantic name, and releases it only after all
renames succeed. Pending, running, and completed records are therefore useful
without risking a partially launched campaign.

The command is a dry run unless ``--submit`` is present.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9-]{0,14}$")


@dataclass(frozen=True)
class TaskSpec:
    campaign: str
    task: int
    case: str
    csv: str
    prices_csv: str
    price_tag: str


def _dive_specs() -> list[TaskSpec]:
    cases = [*(f"30r{r}" for r in range(1, 7)), *(f"40r{r}" for r in range(1, 5))]
    return [
        TaskSpec(
            campaign="dive",
            task=index,
            case=case,
            csv=(
                "duty_unions_big/Practice_Custom_DutyUnion_"
                f"k{case.split('r')[0]}_r{case.split('r')[1]}.csv"
            ),
            prices_csv="hourly_prices_flat.csv",
            price_tag="flat",
        )
        for index, case in enumerate(cases, start=1)
    ]


def _peak_specs() -> list[TaskSpec]:
    instances = (
        ("08r3", "duty_unions/Practice_Custom_DutyUnion_k08_r3.csv"),
        ("13r2", "duty_unions/Practice_Custom_DutyUnion_k13_r2.csv"),
        ("15r6", "duty_unions_big/Practice_Custom_DutyUnion_k15_r6.csv"),
    )
    prices = (
        ("flat", "hourly_prices_flat.csv"),
        ("peak08", "hourly_prices_single_peak_08.csv"),
        ("peak12", "hourly_prices_single_peak_12.csv"),
        ("peak18", "hourly_prices_single_peak_18.csv"),
    )
    specs = []
    task = 1
    for price_tag, prices_csv in prices:
        for case, csv in instances:
            specs.append(TaskSpec("peaks", task, case, csv, prices_csv, price_tag))
            task += 1
    return specs


CAMPAIGNS = {"dive": _dive_specs(), "peaks": _peak_specs()}
WORKERS = {
    "dive": "submit_exact_dive.sub",
    "peaks": "submit_exact_peaks.sub",
}


def _theta_tag(environment: Mapping[str, str]) -> str:
    raw = environment.get("DIVE_THETA", "0.7")
    try:
        scaled = (Decimal(raw) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise ValueError(f"DIVE_THETA must be numeric; got {raw!r}") from exc
    return str(int(scaled))


def job_name(spec: TaskSpec, environment=os.environ) -> str:
    if spec.campaign == "dive":
        name = f"XD-{spec.case}-t{_theta_tag(environment)}"
    elif spec.campaign == "peaks":
        short_price = {
            "flat": "f",
            "peak08": "p08",
            "peak12": "p12",
            "peak18": "p18",
        }[spec.price_tag]
        name = f"XP-{spec.case}-{short_price}"
    else:  # pragma: no cover - CAMPAIGNS is closed above
        raise ValueError(f"unknown campaign: {spec.campaign}")
    if not JOB_NAME_RE.fullmatch(name):
        raise ValueError(f"unsafe or overlong Slurm job name: {name!r}")
    return name


def task_spec(campaign: str, task: int) -> TaskSpec:
    for spec in CAMPAIGNS[campaign]:
        if spec.task == task:
            return spec
    raise ValueError(f"{campaign} has no task {task}")


def _parse_tasks(text: str | None, campaign: str) -> list[TaskSpec]:
    if text is None:
        return list(CAMPAIGNS[campaign])
    requested = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start_text, end_text = piece.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid task range: {piece}")
            requested.extend(range(start, end + 1))
        else:
            requested.append(int(piece))
    if not requested:
        raise ValueError("--tasks selected no tasks")
    if len(requested) != len(set(requested)):
        raise ValueError("--tasks contains duplicates")
    return [task_spec(campaign, task) for task in requested]


def missing_inputs(root: Path, specs: list[TaskSpec]) -> list[Path]:
    paths = {
        root / "data" / relative
        for spec in specs
        for relative in (spec.csv, spec.prices_csv)
    }
    return sorted(path for path in paths if not path.is_file())


def _write_manifest(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def submit_campaign(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    try:
        specs = _parse_tasks(args.tasks, args.campaign)
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    missing = missing_inputs(root, specs)
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise SystemExit(
            "campaign preflight found missing inputs; nothing was submitted:\n  "
            + formatted
        )

    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_tag = args.run_tag or f"{args.campaign}_{timestamp}"
    if run_tag in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_tag):
        raise SystemExit("--run-tag must be a non-dot filesystem-safe name")
    worker = root / "src" / WORKERS[args.campaign]
    if not worker.is_file():
        raise SystemExit(f"missing Slurm worker: {worker}")

    log_dir = root / "src" / "logs" / "slurm_campaigns" / run_tag
    result_dir = root / "src" / "results" / "slurm_campaigns" / run_tag
    manifest = result_dir / "submission.json"
    if log_dir.exists() or result_dir.exists():
        raise SystemExit(f"run tag already exists; choose another: {run_tag}")

    task_records = []
    for spec in specs:
        name = job_name(spec)
        task_records.append(
            {
                **asdict(spec),
                "job_name": name,
                "element_job_id": None,
                "renamed": False,
                "csv_sha256": _sha256(root / "data" / spec.csv),
                "prices_sha256": _sha256(root / "data" / spec.prices_csv),
            }
        )
        print(
            f"[name] task={spec.task} job={name} csv={spec.csv} "
            f"price={spec.price_tag}",
            flush=True,
        )

    array_tasks = ",".join(str(spec.task) for spec in specs)
    held_name = "XD-HELD" if args.campaign == "dive" else "XP-HELD"
    sbatch = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--array={array_tasks}",
        f"--job-name={held_name}",
        f"--output={log_dir}/%A_%a.out",
        f"--error={log_dir}/%A_%a.err",
        "--export=ALL,EVSP_DR_ROOT=" + str(root) +
        ",EVSP_SLURM_RUN_TAG=" + run_tag,
        str(worker),
    ]
    print("[slurm]", " ".join(sbatch), flush=True)

    if not args.submit:
        print(
            f"[dry-run] {len(task_records)} inputs, hashes, and names validated; "
            "add --submit to enqueue the held-then-renamed array",
            flush=True,
        )
        return 0

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        tracked_changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("could not verify the campaign Git checkout") from exc
    if tracked_changes:
        raise SystemExit(
            "refusing to submit from a checkout with tracked modifications:\n"
            + tracked_changes
        )

    log_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)

    record = {
        "campaign": args.campaign,
        "run_tag": run_tag,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": commit,
        "repo_root": str(root),
        "state": "prepared",
        "submitted": False,
        "array_job_id": None,
        "sbatch_command": sbatch,
        "worker_results": str(
            root / "src" / "results" /
            ("exact_dive" if args.campaign == "dive" else "exact_peaks") /
            run_tag
        ),
        "worker_environment": {
            "DIVE_THETA": os.environ.get("DIVE_THETA", "0.7"),
            "DIVE_RC_EPS": os.environ.get("DIVE_RC_EPS", "5"),
            "DIVE_ROUND_WALL": os.environ.get("DIVE_ROUND_WALL", "5400"),
        } if args.campaign == "dive" else {},
        "tasks": task_records,
    }
    _write_manifest(manifest, record)
    try:
        completed = subprocess.run(
            sbatch,
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        record["state"] = "sbatch_failed"
        record["submission_error"] = (
            getattr(exc, "stderr", None) or str(exc)
        ).strip()
        _write_manifest(manifest, record)
        raise SystemExit(f"sbatch failed: {record['submission_error']}") from exc

    array_job_id = completed.stdout.strip().split(";", 1)[0]
    if not array_job_id:
        record["state"] = "sbatch_failed"
        record["submission_error"] = "sbatch returned no array job id"
        _write_manifest(manifest, record)
        raise SystemExit(record["submission_error"])
    record["array_job_id"] = array_job_id
    record["state"] = "held_renaming"
    _write_manifest(manifest, record)

    for task_record in task_records:
        element_job_id = f"{array_job_id}_{task_record['task']}"
        rename = [
            "scontrol",
            "update",
            f"JobId={element_job_id}",
            f"JobName={task_record['job_name']}",
        ]
        try:
            subprocess.run(rename, cwd=root, check=True, text=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            record["state"] = "held_rename_failed"
            record["submission_error"] = (
                getattr(exc, "stderr", None) or str(exc)
            ).strip()
            _write_manifest(manifest, record)
            raise SystemExit(
                f"array {array_job_id} remains HELD because naming failed for "
                f"{element_job_id}: {record['submission_error']}"
            ) from exc
        task_record["element_job_id"] = element_job_id
        task_record["renamed"] = True
        _write_manifest(manifest, record)

    try:
        subprocess.run(
            ["scontrol", "release", array_job_id],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        record["state"] = "held_release_failed"
        record["submission_error"] = (
            getattr(exc, "stderr", None) or str(exc)
        ).strip()
        _write_manifest(manifest, record)
        raise SystemExit(
            f"named array {array_job_id} remains HELD because release failed: "
            f"{record['submission_error']}"
        ) from exc

    record["state"] = "released"
    record["submitted"] = True
    _write_manifest(manifest, record)
    print(f"[submitted] array={array_job_id} manifest={manifest}", flush=True)
    return 0


def emit_task(args: argparse.Namespace) -> int:
    try:
        spec = task_spec(args.campaign, args.task)
        name = job_name(spec)
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print("\t".join((spec.csv, spec.prices_csv, spec.price_tag, name)))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    task = subparsers.add_parser("task", help="emit one worker task as TSV")
    task.add_argument("campaign", choices=sorted(CAMPAIGNS))
    task.add_argument("--task", type=int, required=True)
    task.set_defaults(handler=emit_task)

    submit = subparsers.add_parser("submit", help="validate and submit named tasks")
    submit.add_argument("campaign", choices=sorted(CAMPAIGNS))
    submit.add_argument("--tasks", help="comma-separated task IDs/ranges")
    submit.add_argument("--run-tag")
    submit.add_argument("--root", type=Path, default=REPO_ROOT)
    submit.add_argument("--submit", action="store_true")
    submit.set_defaults(handler=submit_campaign)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a conservative local Goal-1 benchmark batch (dry-run by default)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "src" / "run_ex_unicorn.py"

# Keep these two runner-facing spellings together.  If the pending pricing
# integration renames either option, this is the only launcher section to edit.
RUNNER_QUEUE_FLAG = "--queue_order"
RUNNER_OUTPUT_SELECTION_FLAG = "--pricing_output_selection"
RUNNER_DOMINANCE_FLAG = "--dominance_mode"
RUNNER_GAP_FLAG = "--max_charge2trip"
RUNNER_TRIP_GAP_FLAG = "--max_trip2trip"

THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class Profile:
    active_hours: float
    milestones: str
    max_labels: int
    pricing_tiers: str
    pricing_wall_s: int


PROFILES = {
    "5m": Profile(5.0 / 60.0, "0.0833333333", 25_000, "10000:60,25000:120", 240),
    "30m": Profile(0.5, "0.0833333333,0.5", 50_000, "25000:180,50000:600", 900),
    "3h": Profile(3.0, "0.5,3", 75_000, "25000:500,75000:1800", 2400),
}


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    csv_path: Path
    csv_sha256: str
    bus_count: int
    replicate: int
    seed: int | str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int_set(raw: str, *, name: str) -> set[int] | None:
    if raw.strip().lower() == "all":
        return None
    try:
        values = {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError as exc:
        raise ValueError(f"{name} must be 'all' or comma-separated integers") from exc
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return values


def _resolve_manifest_source(manifest: dict[str, object]) -> Path:
    source_info = manifest.get("source") or {}
    source_text = source_info.get("path")
    if not source_text:
        raise ValueError("Manifest source.path is missing")
    source_path = Path(source_text)
    return source_path if source_path.is_absolute() else REPO_ROOT / source_path


def load_cases(
    manifest_path: Path,
    *,
    sizes: set[int],
    replicates: set[int] | None,
    include_20: bool,
    include_hard_reference: bool,
) -> tuple[dict[str, object], list[BenchmarkCase]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("synthetic") is not True or manifest.get("single_day_verified") is not False:
        raise ValueError("Manifest must explicitly identify synthetic, unverified-day instances")
    if 20 in sizes and not include_20:
        raise ValueError("20-bus runs are deferred; pass --include-20 explicitly")

    source_info = manifest.get("source") or {}
    source_path = _resolve_manifest_source(manifest)
    if not source_path.exists():
        raise FileNotFoundError(f"Manifest source is unavailable: {source_path}")
    actual_source_hash = sha256_file(source_path)
    if actual_source_hash != source_info.get("sha256"):
        raise ValueError("Tracked source SHA256 no longer matches the manifest")

    cases: list[BenchmarkCase] = []
    for item in manifest.get("instances", []):
        size = int(item["bus_count"])
        replicate = int(item["replicate"])
        if size not in sizes or (replicates is not None and replicate not in replicates):
            continue
        if item.get("synthetic") is not True or item.get("single_day_verified") is not False:
            raise ValueError(f"Instance n={size} r={replicate} lacks synthetic provenance")
        csv_path = manifest_path.parent / item["output_csv"]
        if not csv_path.exists():
            raise FileNotFoundError(f"Generated instance is missing: {csv_path}")
        actual_hash = sha256_file(csv_path)
        if actual_hash != item["output_sha256"]:
            raise ValueError(f"Generated instance hash mismatch: {csv_path}")
        cases.append(
            BenchmarkCase(
                name=f"random_{size}b_r{replicate:02d}",
                csv_path=csv_path,
                csv_sha256=actual_hash,
                bus_count=size,
                replicate=replicate,
                seed=item["seed"],
            )
        )

    if include_hard_reference:
        hard_path = REPO_ROOT / "data" / "Practice_10bus.csv"
        cases.append(
            BenchmarkCase(
                name="hard_first10",
                csv_path=hard_path,
                csv_sha256=sha256_file(hard_path),
                bus_count=10,
                replicate=0,
                seed="historical-first10",
            )
        )

    # Interleave sizes at each replicate so the first two processes exercise
    # 10 and 15 buses, rather than launching two same-size cases first.
    cases.sort(
        key=lambda case: (
            case.name == "hard_first10",
            case.replicate,
            case.bus_count,
            case.name,
        )
    )
    if not cases:
        raise ValueError("No manifest instances matched the requested sizes/replicates")
    return manifest, cases


def build_command(
    case: BenchmarkCase,
    *,
    python: Path,
    profile_name: str,
    results_root: Path,
    batch_tag: str,
    initializer: str,
    queue_order: str,
    pricing_output_selection: str,
    dominance_mode: str,
    max_charge2trip: int,
    max_trip2trip: int = 57,
) -> list[str]:
    profile = PROFILES[profile_name]
    return [
        str(python),
        "-u",
        str(RUNNER),
        "--csv",
        str(case.csv_path),
        "--G",
        "300",
        f"--{initializer}",
        "--master_backend",
        "scipy",
        "--skip_final_mip",
        "--prices_csv",
        "hourly_prices_flat.csv",
        "--price_tag",
        "flat",
        RUNNER_QUEUE_FLAG,
        queue_order,
        RUNNER_OUTPUT_SELECTION_FLAG,
        pricing_output_selection,
        RUNNER_DOMINANCE_FLAG,
        dominance_mode,
        RUNNER_TRIP_GAP_FLAG,
        str(max_trip2trip),
        RUNNER_GAP_FLAG,
        str(max_charge2trip),
        "--active_time_limit_hours",
        f"{profile.active_hours:.12g}",
        "--milestones_hours",
        profile.milestones,
        "--max_labels",
        str(profile.max_labels),
        "--pricing_tiers",
        profile.pricing_tiers,
        "--pricing_wall_per_iter",
        str(profile.pricing_wall_s),
        "--master_time_limit",
        "120",
        "--kbest",
        "150",
        "--min_trips_per_route",
        "1",
        "--stagnation_window",
        "999999",
        "--improvement_bound",
        "-2",
        "--run_tag",
        f"{batch_tag}_{case.name}",
        "--results_root",
        str(results_root),
        "--no_resume",
    ]


def _runner_preflight(
    python: Path,
    queue_order: str,
    pricing_output_selection: str,
    dominance_mode: str,
) -> None:
    result = subprocess.run(
        [str(python), str(RUNNER), "--help"],
        cwd=REPO_ROOT,
        env={**os.environ, **THREAD_ENVIRONMENT},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    help_text = result.stdout
    required = (
        "--master_backend",
        "--matching",
        RUNNER_QUEUE_FLAG,
        RUNNER_OUTPUT_SELECTION_FLAG,
        RUNNER_DOMINANCE_FLAG,
        RUNNER_TRIP_GAP_FLAG,
        RUNNER_GAP_FLAG,
    )
    missing = [flag for flag in required if flag not in help_text]
    if result.returncode or missing:
        raise RuntimeError(
            "Local runner preflight failed. The pricing integration must expose "
            f"{required}; missing={missing}. Runner output:\n{help_text[-2000:]}"
        )
    if queue_order not in help_text:
        raise RuntimeError(
            f"Runner help does not advertise queue order {queue_order!r}; "
            "finish the pricing repair or select the implemented successor with --queue-order."
        )
    if pricing_output_selection not in help_text:
        raise RuntimeError(
            "Runner help does not advertise pricing output selection "
            f"{pricing_output_selection!r}."
        )
    if dominance_mode not in help_text:
        raise RuntimeError(
            f"Runner help does not advertise dominance mode {dominance_mode!r}."
        )


def _git_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _pump_output(name: str, process: subprocess.Popen[str], log_path: Path) -> None:
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{name}] {line}", end="", flush=True)


def execute_commands(
    commands: Sequence[tuple[BenchmarkCase, list[str], Path]],
    *,
    max_workers: int,
) -> int:
    environment = {**os.environ, **THREAD_ENVIRONMENT}
    pending = list(commands)
    running: list[tuple[BenchmarkCase, subprocess.Popen[str], threading.Thread, Path]] = []
    failures: list[tuple[str, int]] = []

    try:
        while pending or running:
            while pending and len(running) < max_workers:
                case, command, log_path = pending.pop(0)
                print(f"[launch] {case.name}: {shlex.join(command)}", flush=True)
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                )
                thread = threading.Thread(
                    target=_pump_output,
                    args=(case.name, process, log_path),
                    daemon=True,
                )
                thread.start()
                running.append((case, process, thread, log_path))

            still_running = []
            for case, process, thread, log_path in running:
                return_code = process.poll()
                if return_code is None:
                    still_running.append((case, process, thread, log_path))
                    continue
                thread.join()
                print(f"[done] {case.name}: exit={return_code}, log={log_path}")
                if return_code:
                    failures.append((case.name, return_code))
            running = still_running
            if running:
                time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nInterrupted; terminating active local pricing processes.", file=sys.stderr)
        for _, process, _, _ in running:
            process.terminate()
        for _, process, thread, _ in running:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            thread.join(timeout=2)
        return 130

    if failures:
        print(f"Failed cases: {failures}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=tuple(PROFILES))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sizes", default="10,15", help="Default: 10,15. 20 requires --include-20.")
    parser.add_argument("--replicates", default="all", help="'all' or comma-separated replicate IDs")
    parser.add_argument("--include-20", action="store_true")
    parser.add_argument("--include-hard-reference", action="store_true")
    parser.add_argument("--max-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--initializer",
        choices=("matching", "greedy"),
        default="matching",
        help="Model-derived matching cover by default; use greedy as a control.",
    )
    parser.add_argument(
        "--queue-order",
        choices=("time", "reduced_cost", "reduced_cost_bound", "start_fair_bound"),
        default="reduced_cost_bound",
    )
    parser.add_argument(
        "--pricing-output-selection",
        choices=("reduced_cost", "diversified"),
        default="reduced_cost",
        help="How the DP chooses at most K negative columns from its eligible pool.",
    )
    parser.add_argument(
        "--dominance-mode",
        choices=("resource", "incidence_diverse"),
        default="resource",
        help="Experimental label-dominance policy; resource preserves current behavior.",
    )
    parser.add_argument("--max-charge2trip", type=int, default=1560)
    parser.add_argument("--max-trip2trip", type=int, default=57)
    parser.add_argument("--batch-tag", default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch processes. Without this flag, print a dry run only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sizes = _parse_int_set(args.sizes, name="sizes")
        assert sizes is not None
        replicates = _parse_int_set(args.replicates, name="replicates")
        _, cases = load_cases(
            args.manifest,
            sizes=sizes,
            replicates=replicates,
            include_20=args.include_20,
            include_hard_reference=args.include_hard_reference,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_tag = args.batch_tag or f"local_{args.profile}_{timestamp}"
    if not all(character.isalnum() or character in "._-" for character in batch_tag):
        raise SystemExit("ERROR: --batch-tag may contain only letters, digits, dot, underscore, hyphen")
    results_root = (args.results_root or REPO_ROOT / "src" / "results" / "local_goal1" / batch_tag).resolve()
    log_root = results_root / "launcher_logs"

    commands: list[tuple[BenchmarkCase, list[str], Path]] = []
    for case in cases:
        command = build_command(
            case,
            python=args.python,
            profile_name=args.profile,
            results_root=results_root,
            batch_tag=batch_tag,
            initializer=args.initializer,
            queue_order=args.queue_order,
            pricing_output_selection=args.pricing_output_selection,
            dominance_mode=args.dominance_mode,
            max_charge2trip=args.max_charge2trip,
            max_trip2trip=args.max_trip2trip,
        )
        commands.append((case, command, log_root / f"{case.name}.log"))

    print("Synthetic/random cases are not verified single-day GIRO instances.")
    print(
        f"Profile={args.profile}; initializer={args.initializer}; "
        f"output_selection={args.pricing_output_selection}; "
        f"dominance={args.dominance_mode}; "
        f"max_trip2trip={args.max_trip2trip}; "
        f"cases={len(cases)}; max_workers={args.max_workers}"
    )
    print(f"Thread limits: {THREAD_ENVIRONMENT}")
    for case, command, log_path in commands:
        print(
            f"  {case.name}: n={case.bus_count}, trips file={case.csv_path.name}, "
            f"sha256={case.csv_sha256[:12]}..., log={log_path}"
        )
        print(f"    {shlex.join(command)}")

    if not args.execute:
        print("Dry run only. Re-run with --execute after reviewing the commands.")
        return 0
    if _git_is_dirty() and not args.allow_dirty:
        raise SystemExit(
            "ERROR: checkout is dirty; commit the pricing repair first or pass --allow-dirty "
            "for an explicitly provisional diagnostic."
        )
    if args.max_trip2trip <= 0 or args.max_charge2trip <= 0:
        raise SystemExit(
            "ERROR: --max-trip2trip and --max-charge2trip must be positive"
        )
    try:
        _runner_preflight(
            args.python,
            args.queue_order,
            args.pricing_output_selection,
            args.dominance_mode,
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    log_root.mkdir(parents=True, exist_ok=True)
    collision = [log_path for _, _, log_path in commands if log_path.exists()]
    if collision:
        raise SystemExit(
            "ERROR: refusing to overwrite existing launcher logs: "
            + ", ".join(str(path) for path in collision)
        )
    return execute_commands(commands, max_workers=args.max_workers)


if __name__ == "__main__":
    raise SystemExit(main())

"""Continue exact CG from an immutable timed snapshot in an isolated pool.

The source snapshot and journal are copied once.  A synthetic first iters.csv
row anchors cumulative elapsed time at the snapshot mark, so `--wall-limit-s`
and later timed snapshots refer to total CG age rather than the new Slurm
allocation.  Requeues resume the isolated journal and never touch the live
campaign that produced the source snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from durable_io import (
    atomic_copy,
    atomic_write_json,
    atomic_write_text,
    read_jsonl_records,
    valid_json_object,
)
from exact_pricer_expanded import load_iteration_log
from run_exact_pool_mip import resolve_pool_journal


SRC = Path(__file__).resolve().parent
DATA_DIR = SRC.parent / "data"
TERMINAL_STOP_REASONS = {
    "certified", "no_path", "max_iters", "degenerate_stall",
    "stalled_marginal_returns",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _starts_with_file(path: Path, prefix_path: Path) -> bool:
    """Whether *path* contains the immutable source bytes as a prefix."""

    with open(path, "rb") as candidate, open(prefix_path, "rb") as prefix:
        while True:
            expected = prefix.read(1024 * 1024)
            if not expected:
                return True
            if candidate.read(len(expected)) != expected:
                return False


def verify_snapshot_problem_inputs(status: dict) -> None:
    """Bind a continuation to the exact instance and tariff bytes priced.

    A copied status/journal is not sufficient provenance if the mutable files
    in ``data/`` have since changed.  New scientific controls therefore refuse
    snapshots that lack hashes or whose current inputs differ.
    """

    provenance = status.get("provenance") or {}
    for status_key, hash_key in (
        ("csv", "instance_sha256"),
        ("prices_csv", "prices_sha256"),
    ):
        relative = status.get(status_key)
        expected = provenance.get(hash_key)
        if not relative or not expected:
            raise ValueError(
                f"snapshot lacks required {status_key}/{hash_key} provenance"
            )
        path = (DATA_DIR / str(relative)).resolve()
        try:
            path.relative_to(DATA_DIR.resolve())
        except ValueError as exc:
            raise ValueError(
                f"snapshot {status_key} is outside the repository data dir: "
                f"{relative}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"snapshot input is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"snapshot {status_key} hash mismatch: expected {expected}, "
                f"found {actual} for {path}"
            )


def control_result_complete(path: Path, total_wall_s: float,
                            expected_commit: str | None = None) -> bool:
    if not valid_json_object(path, ("stop_reason", "wall_s")):
        return False
    with open(path) as fh:
        payload = json.load(fh)
    if (expected_commit is not None
            and (payload.get("provenance") or {}).get("git_commit")
            != expected_commit):
        return False
    reason = payload.get("stop_reason")
    if reason in TERMINAL_STOP_REASONS:
        return True
    if reason == "wall_limit":
        try:
            requested_wall_s = float(
                ((payload.get("provenance") or {}).get("args") or {})
                .get("wall_limit_s")
            )
            return (
                math.isclose(requested_wall_s, float(total_wall_s),
                             rel_tol=0.0, abs_tol=1e-6)
                and float(payload.get("wall_s", 0.0))
                >= float(total_wall_s) - 120.0
            )
        except (TypeError, ValueError):
            return False
    return False


def prepare_snapshot_resume(snapshot: Path, out: Path,
                            snapshot_minutes: float,
                            continuation_commit: str | None = None) -> dict:
    if snapshot.resolve() == out.resolve():
        raise ValueError("control output must not overwrite its source snapshot")
    with open(snapshot) as fh:
        status = json.load(fh)
    source_journal = resolve_pool_journal(snapshot, status)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_journal = Path(str(out) + ".columns.jsonl")
    iters_path = Path(str(out) + ".iters.csv")
    snapshot_sha = _sha256(snapshot)
    source_journal_sha = _sha256(source_journal)
    status_path_existed = out.exists()
    iteration_rows = []
    if iters_path.exists() and iters_path.stat().st_size:
        iteration_rows = load_iteration_log(
            iters_path, repair_trailing=True
        )
    # Preparation writes exactly one synthetic anchor.  A second valid data
    # row proves that exact pricing has begun, even if a hard preemption occurs
    # before the first periodic status checkpoint updates stop_reason.
    progress_started = len(iteration_rows) > 1
    existing = None
    if valid_json_object(out, ("stop_reason", "columns_journal")):
        with open(out) as fh:
            existing = json.load(fh)
        parent = existing.get("resume_parent") or {}
        if (parent.get("snapshot_sha256") != snapshot_sha
                or parent.get("journal_sha256") != source_journal_sha):
            raise ValueError(
                f"existing control {out} belongs to a different source "
                "snapshot or journal"
            )
        if not out_journal.is_file():
            raise ValueError(
                f"existing control {out} has lost its isolated journal "
                f"{out_journal}; use a new output path"
            )
        if not _starts_with_file(out_journal, source_journal):
            raise ValueError(
                f"existing control journal {out_journal} no longer contains "
                "its immutable source journal as a prefix; preserve it for "
                "diagnosis and use a new output path"
            )
        # A hard preemption may truncate only the last append. Preserve every
        # complete record and refuse interior corruption.
        read_jsonl_records(
            out_journal, repair_trailing=True, collect=False
        )
        if not iters_path.is_file():
            # Status is published after the initial journal copy but before
            # the synthetic anchor. Recover only that one provable preparation
            # interruption; a changed journal means pricing may have started.
            if (existing.get("stop_reason") != "prepared_snapshot_resume"
                    or _sha256(out_journal) != source_journal_sha):
                raise ValueError(
                    f"existing control {out} has lost its iteration "
                    "trajectory after pricing may have started; preserve it "
                    "for diagnosis and use a new output path"
                )
        prior_commit = parent.get("continuation_commit")
        if (continuation_commit is not None
                and prior_commit != continuation_commit):
            if (existing.get("stop_reason") == "prepared_snapshot_resume"
                    and not progress_started
                    and _sha256(out_journal) == source_journal_sha):
                existing = None
            else:
                raise ValueError(
                    f"existing control {out} contains pricing work from "
                    f"commit {prior_commit!r}; use a new output path rather "
                    "than mixing algorithms"
                )
    elif (status_path_existed or iters_path.exists()
          or (out_journal.exists()
              and _sha256(out_journal) != source_journal_sha)):
        raise ValueError(
            f"control status {out} is missing or corrupt while persisted "
            "control artifacts exist; preserve them for diagnosis and use a "
            "new output path"
        )

    initializing = existing is None
    if initializing:
        # With no published status, any destination journal can only be an
        # interrupted initial copy. Recreate it from the immutable source.
        if (not out_journal.exists()
                or _sha256(out_journal) != source_journal_sha):
            atomic_copy(source_journal, out_journal)
        if (_sha256(snapshot) != snapshot_sha
                or _sha256(source_journal) != source_journal_sha
                or _sha256(out_journal) != source_journal_sha):
            raise ValueError(
                "source snapshot or journal changed while preparing control"
            )
        read_jsonl_records(out_journal, repair_trailing=False, collect=False)

    if existing is None:
        try:
            recorded_wall_s = float(status.get("wall_s", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("snapshot has a non-numeric wall_s") from exc
        if not math.isfinite(recorded_wall_s) or recorded_wall_s < 0.0:
            raise ValueError("snapshot has an invalid wall_s")
        snapshot_actual_wall_s = max(
            float(snapshot_minutes) * 60.0, recorded_wall_s
        )
        prepared = dict(status)
        prepared["columns_journal"] = str(out_journal)
        prepared["stop_reason"] = "prepared_snapshot_resume"
        prepared["resume_parent"] = {
            "snapshot": str(snapshot),
            "snapshot_sha256": snapshot_sha,
            "journal": str(source_journal),
            "journal_sha256": source_journal_sha,
            "snapshot_minutes": snapshot_minutes,
            "snapshot_actual_wall_s": snapshot_actual_wall_s,
            "continuation_commit": continuation_commit,
            "source_provenance": status.get("provenance"),
            "designed_override": {
                "stall_window_min": None,
                "purpose": "no-stall continuation control",
            },
        }
        atomic_write_json(out, prepared)

    write_anchor = initializing or not iters_path.is_file()
    if write_anchor:
        parent = ((prepared if existing is None else existing)
                  .get("resume_parent") or {})
        snapshot_actual_wall_s = float(parent.get(
            "snapshot_actual_wall_s", float(snapshot_minutes) * 60.0
        ))
        final = status.get("final") or {}
        final_lp = status.get("final_lp") or {}
        lp_obj = final.get("lp_obj", final_lp.get("objective", 0.0))
        route_weight = final.get(
            "route_weight", final_lp.get("route_weight", 0.0)
        )
        artificials = final.get(
            "artificials", final_lp.get("artificial_total", 0.0)
        )
        min_rc = final.get("min_rc", 0.0)
        iteration = status.get("iterations", 0)
        pool_columns = status.get("columns", 0)
        atomic_write_text(
            iters_path,
            "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,"
            "pool_columns\n"
            f"{snapshot_actual_wall_s:.2f},{iteration},{lp_obj},"
            f"{route_weight},{artificials},{min_rc},{pool_columns}\n"
        )
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-minutes", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total-wall-s", type=int, default=259200,
                        help="Total CG age, including the source snapshot.")
    parser.add_argument("--snapshot-at-minutes", default="720,1440,2880,4320")
    parser.add_argument("--expected-commit", default=None)
    args = parser.parse_args(argv)

    with open(args.snapshot) as fh:
        source_status = json.load(fh)
    verify_snapshot_problem_inputs(source_status)

    if control_result_complete(
            args.out, args.total_wall_s, args.expected_commit):
        print(f"[CG-CONTROL] {args.out} already reached a terminal state for "
              f"the requested {args.total_wall_s:.0f}s total age")
        return 0

    status = prepare_snapshot_resume(
        args.snapshot, args.out, args.snapshot_minutes,
        continuation_commit=args.expected_commit,
    )
    required = ("csv", "prices_csv", "soc_step", "block_min")
    missing = [field for field in required if status.get(field) is None]
    if missing:
        raise SystemExit(f"snapshot lacks required fields: {missing}")

    source_args = ((status.get("provenance") or {}).get("args") or {})
    rc_eps = source_args.get(
        "rc_eps", (status.get("provenance") or {}).get("rc_eps", 1e-4)
    )
    columns_per_iter = source_args.get("columns_per_iter", 30)
    checkpoint_every = source_args.get("checkpoint_every", 25)
    command = [
        sys.executable, "-u", str(SRC / "exact_pricer_expanded.py"),
        "--csv", str(status["csv"]),
        "--prices_csv", str(status["prices_csv"]),
        "--soc-step", str(status["soc_step"]),
        "--block-min", str(status["block_min"]),
        "--g-kwh", str(status.get("g_kwh", 300.0)),
        "--charge-kw", str(status.get("charge_kw", 300.0)),
        "--min-soc-frac", str(status.get("min_soc_frac", 0.0)),
        "--master-sense", str(status.get("master_sense", "partition")),
        "--rc-eps", str(rc_eps),
        "--columns_per_iter", str(columns_per_iter),
        "--max-iters", "200000",
        "--wall-limit-s", str(args.total_wall_s),
        "--checkpoint-every", str(checkpoint_every),
        "--snapshot-at-minutes", args.snapshot_at_minutes,
        "--resume",
        "--out", str(args.out),
    ]
    print("[CG-CONTROL] " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=SRC).returncode


if __name__ == "__main__":
    raise SystemExit(main())

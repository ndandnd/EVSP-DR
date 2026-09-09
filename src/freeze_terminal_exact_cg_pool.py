#!/usr/bin/env python3
"""Stream one terminal exact-CG RAW pool into an immutable MIP snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from durable_io import atomic_write_json, flush_and_fsync
from run_exact_pool_mip import resolve_pool_journal


SCHEMA = "evsp-dr-terminal-exact-cg-pool-snapshot-v1"
TERMINAL_STOPS = {
    "certified", "wall_limit", "master_failed", "max_iters", "no_path",
    "stalled_marginal_returns", "degenerate_stall",
}


def terminal_artificials(status: dict) -> tuple[float, str]:
    final = status.get("final") or {}
    if "artificials" in final:
        return float(final["artificials"]), "final"
    final_lp = status.get("final_lp") or {}
    if "artificial_total" in final_lp:
        return float(final_lp["artificial_total"]), "final_lp"
    return math.nan, "missing"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty {label}: {path}")
    return path.resolve()


def valid_status(
    path: Path, cell: str, instance_sha: str, solver_commit: str,
    master_sense: str = "partition",
) -> tuple[dict, bytes]:
    raw = required(path, "CG status").read_bytes()
    status = json.loads(raw)
    final = status.get("final") or {}
    provenance = status.get("provenance") or {}
    observed = {
        "time_model": status.get("time_model"),
        "arc_mode": (status.get("network_metrics") or {}).get("arc_mode"),
        "soc_step": float(status.get("soc_step", -1)),
        "block_min": int(status.get("block_min", -1)),
        "g_kwh": float(status.get("g_kwh", -1)),
        "charge_kw": float(status.get("charge_kw", -1)),
        "min_soc_frac": float(status.get("min_soc_frac", -1)),
        "prices_csv": status.get("prices_csv"),
        "master_sense": status.get("master_sense"),
        "initial_pool": status.get("initial_pool"),
        "columns_per_iter": int(status.get("columns_per_iter", -1)),
        "column_pool_treatment": status.get("column_pool_treatment"),
        "column_selection": status.get("column_selection"),
        "column_diversity_weight": float(
            status.get("column_diversity_weight", -1)
        ),
        "column_candidate_multiplier": int(
            status.get("column_candidate_multiplier", -1)
        ),
    }
    expected = {
        "time_model": "event", "arc_mode": "lazy", "soc_step": 2.5,
        "block_min": 5, "g_kwh": 240.0, "charge_kw": 240.0,
        "min_soc_frac": 0.0, "prices_csv": "hourly_prices_flat.csv",
        "master_sense": master_sense, "initial_pool": "singletons",
        "columns_per_iter": 30, "column_pool_treatment": "RAW",
        "column_selection": "reduced_cost", "column_diversity_weight": 0.0,
        "column_candidate_multiplier": 4,
    }
    if observed != expected:
        raise ValueError(f"configuration mismatch for {cell}: {observed}")
    if status.get("stop_reason") not in TERMINAL_STOPS:
        raise ValueError(f"source is not terminal for {cell}: {status.get('stop_reason')}")
    if status.get("stop_reason") == "certified" and status.get("certified_rc_optimal") is not True:
        raise ValueError(f"certified stop lacks certificate flag for {cell}")
    artificials, _artificial_source = terminal_artificials(status)
    if not math.isfinite(artificials) or artificials < 0 or artificials > 1e-7:
        raise ValueError(f"source retains artificials for {cell}")
    if provenance.get("instance_sha256") != instance_sha:
        raise ValueError(f"instance provenance mismatch for {cell}")
    if provenance.get("git_commit") != solver_commit:
        raise ValueError(f"source solver commit mismatch for {cell}")
    if float(provenance.get("rc_eps", math.nan)) != 1e-4:
        raise ValueError(f"source reduced-cost tolerance mismatch for {cell}")
    if int(status.get("columns", -1)) <= 0:
        raise ValueError(f"source has no columns for {cell}")
    return status, raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-status", type=Path, required=True)
    parser.add_argument("--resume-status", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--instance-relative-to-data", required=True)
    parser.add_argument("--instance-sha256", required=True)
    parser.add_argument("--source-solver-commit", required=True)
    parser.add_argument(
        "--master-sense", choices=("partition", "cover"),
        default="partition",
        help="Expected master sense of both source CG statuses.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.expanduser().resolve()
    record_path = args.record.expanduser().resolve()
    output_journal = Path(str(output) + ".columns.jsonl")
    output_iters = Path(str(output) + ".iters.csv")
    temporary_journal = Path(str(output_journal) + f".tmp.{os.getpid()}")
    temporary_iters = Path(str(output_iters) + f".tmp.{os.getpid()}")
    for path in (output, output_journal, output_iters, record_path):
        if os.path.lexists(path):
            raise FileExistsError(path)

    candidates = []
    if args.resume_status.is_file():
        candidates.append(("continuation", args.resume_status))
    candidates.append(("baseline", args.base_status))
    failures = []
    selected = None
    for stage, path in candidates:
        try:
            status, status_bytes = valid_status(
                path, args.cell, args.instance_sha256,
                args.source_solver_commit, args.master_sense,
            )
            if stage == "baseline" and status.get("certified_rc_optimal") is not True:
                raise ValueError("baseline fallback is not certified")
            selected = (stage, path.resolve(), status, status_bytes)
            break
        except Exception as exc:
            failures.append(f"{stage}: {exc}")
    if selected is None:
        raise SystemExit(f"no usable terminal source for {args.cell}: {'; '.join(failures)}")
    stage, source, status, status_bytes = selected
    source_journal = required(resolve_pool_journal(source, status), "source journal")
    source_iters = required(Path(str(source) + ".iters.csv"), "source iteration log")
    source_status_sha = hashlib.sha256(status_bytes).hexdigest()
    source_journal_before = sha256(source_journal)
    source_iters_before = sha256(source_iters)

    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    unique = set()
    records = 0
    previous_iter = -10**18
    try:
        with source_journal.open("rb") as incoming, temporary_journal.open("xb") as outgoing:
            for ordinal, raw in enumerate(incoming, start=1):
                if not raw.endswith(b"\n"):
                    raise ValueError(f"incomplete journal line {ordinal}")
                item = json.loads(raw)
                found = item.get("found_iter", 0)
                trips = item.get("trips")
                if (
                    not isinstance(found, int) or isinstance(found, bool)
                    or found < previous_iter
                    or not isinstance(trips, list) or not trips
                    or len(trips) != len(set(trips))
                    or any(not isinstance(trip, int) or isinstance(trip, bool) for trip in trips)
                    or not math.isfinite(float(item.get("cost", math.nan)))
                ):
                    raise ValueError(f"invalid journal line {ordinal}")
                previous_iter = found
                unique.add(frozenset(trips))
                outgoing.write(raw)
                digest.update(raw)
                records += 1
            flush_and_fsync(outgoing)
        if len(unique) != int(status["columns"]):
            raise ValueError(
                f"pool count mismatch for {args.cell}: unique={len(unique)} "
                f"status={status['columns']}"
            )
        with source_iters.open("rb") as incoming, temporary_iters.open("xb") as outgoing:
            for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                outgoing.write(chunk)
            flush_and_fsync(outgoing)
        if (
            hashlib.sha256(source.read_bytes()).hexdigest() != source_status_sha
            or sha256(source_journal) != source_journal_before
            or sha256(source_iters) != source_iters_before
        ):
            raise RuntimeError("source CG artifacts changed while freezing")
        os.replace(temporary_journal, output_journal)
        os.replace(temporary_iters, output_iters)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(parent_fd)
        os.close(parent_fd)
    finally:
        temporary_journal.unlink(missing_ok=True)
        temporary_iters.unlink(missing_ok=True)

    snapshot = dict(status)
    snapshot["csv"] = args.instance_relative_to_data
    snapshot["columns_journal"] = str(output_journal)
    snapshot["terminal_pool_snapshot"] = {
        "schema": SCHEMA,
        "cell": args.cell,
        "selected_source_stage": stage,
        "source_status": str(source),
        "source_status_sha256": source_status_sha,
        "source_journal": str(source_journal),
        "source_journal_sha256": source_journal_before,
        "source_iterations": str(source_iters),
        "source_iterations_sha256": source_iters_before,
        "journal_records": records,
        "unique_pool_columns": len(unique),
        "journal_sha256": digest.hexdigest(),
        "source_certified": status.get("certified_rc_optimal") is True,
        "source_stop_reason": status.get("stop_reason"),
        "path_rebound_to_execution_data": True,
    }
    atomic_write_json(output, snapshot)
    record = {
        "schema": "evsp-dr-terminal-exact-cg-pool-freeze-record-v1",
        "cell": args.cell,
        "snapshot": str(output),
        "snapshot_sha256": sha256(output),
        "journal": str(output_journal),
        "journal_sha256": sha256(output_journal),
        "iterations": str(output_iters),
        "iterations_sha256": sha256(output_iters),
        "source_stage": stage,
        "source_stop_reason": status.get("stop_reason"),
        "source_certified": status.get("certified_rc_optimal") is True,
        "columns": len(unique),
        "artificials": terminal_artificials(status)[0],
        "artificials_source": terminal_artificials(status)[1],
        "instance_sha256": args.instance_sha256,
        "master_sense": args.master_sense,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(record_path, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

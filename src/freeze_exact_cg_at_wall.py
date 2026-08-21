#!/usr/bin/env python3
"""Freeze the last complete exact-CG state at or before a wall-time budget."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from durable_io import atomic_write_json, flush_and_fsync
from run_exact_pool_mip import resolve_pool_journal


SCHEMA = "evsp-dr-exact-cg-matched-wall-snapshot-v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_iteration_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError("iteration log has no header")
        return list(reader.fieldnames), rows


def _selected_row(rows: list[dict], budget_s: float) -> dict:
    eligible = [
        row for row in rows
        if float(row["elapsed_s"]) <= float(budget_s) + 1e-9
    ]
    if not eligible:
        raise ValueError("no complete CG iteration exists within wall budget")
    return eligible[-1]


def _freeze_journal(
    journal_bytes: bytes,
    *,
    before_iteration: int,
) -> tuple[bytes, int, int]:
    retained_lines = []
    best_by_incidence = {}
    for ordinal, line in enumerate(journal_bytes.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"journal line {ordinal} is malformed") from exc
        found = record.get("found_iter", 0)
        if not isinstance(found, int) or isinstance(found, bool):
            raise ValueError(
                f"journal line {ordinal} has invalid found_iter"
            )
        # The iteration CSV row is written before that iteration's candidate
        # columns are inserted, so its complete state contains found_iter < t.
        if found >= before_iteration:
            continue
        trips = record.get("trips")
        if (
            not isinstance(trips, list)
            or not trips
            or any(
                not isinstance(trip, int) or isinstance(trip, bool)
                for trip in trips
            )
        ):
            raise ValueError(f"journal line {ordinal} has invalid trips")
        retained_lines.append(line)
        key = frozenset(trips)
        current = best_by_incidence.get(key)
        if (
            current is None
            or float(record["cost"]) < float(current["cost"]) - 1e-9
        ):
            best_by_incidence[key] = record
    payload = (
        b"\n".join(retained_lines) + (b"\n" if retained_lines else b"")
    )
    return payload, len(retained_lines), len(best_by_incidence)


def _snapshot_peak_rss_mb(telemetry: Path | None, budget_s: float):
    if telemetry is None:
        return None, None
    telemetry = telemetry.expanduser().resolve(strict=True)
    payload = telemetry.read_bytes()
    peaks = []
    for raw_line in payload.splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        elapsed = record.get("elapsed_session_s")
        peak = record.get("peak_rss_bytes")
        if (
            elapsed is not None
            and peak is not None
            and float(elapsed) <= float(budget_s) + 1e-9
        ):
            peaks.append(int(peak))
    return (
        max(peaks) / (1024.0 * 1024.0) if peaks else None,
        {
            "path": str(telemetry),
            "sha256": _sha256(payload),
        },
    )


def freeze(args) -> dict:
    source = Path(args.result).expanduser().resolve(strict=True)
    output = Path(args.out).expanduser().resolve()
    output_journal = Path(str(output) + ".columns.jsonl")
    output_iterations = Path(str(output) + ".iters.csv")
    for path in (output, output_journal, output_iterations):
        if os.path.lexists(path):
            raise FileExistsError(path)
    source_status_bytes = source.read_bytes()
    source_status = json.loads(source_status_bytes)
    source_journal = resolve_pool_journal(
        source, source_status
    ).resolve(strict=True)
    source_journal_bytes = source_journal.read_bytes()
    source_iterations = Path(
        str(source) + ".iters.csv"
    ).resolve(strict=True)
    iteration_fields, iteration_rows = _load_iteration_rows(
        source_iterations
    )
    selected = _selected_row(iteration_rows, args.budget_s)
    selected_iteration = int(selected["iteration"])
    frozen_journal, journal_records, unique_columns = _freeze_journal(
        source_journal_bytes,
        before_iteration=selected_iteration,
    )
    expected_columns = int(selected["pool_columns"])
    if unique_columns != expected_columns:
        raise ValueError(
            "frozen journal does not reproduce iteration pool size: "
            f"{unique_columns} != {expected_columns}"
        )
    frozen_rows = [
        row for row in iteration_rows
        if int(row["iteration"]) <= selected_iteration
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output_journal.open("xb") as handle:
        handle.write(frozen_journal)
        flush_and_fsync(handle)
    with output_iterations.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=iteration_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(frozen_rows)
        flush_and_fsync(handle)
    peak_rss_mb, telemetry_identity = _snapshot_peak_rss_mb(
        args.telemetry, args.budget_s
    )
    final = {
        "iter": selected_iteration,
        "attempt_iter": selected_iteration,
        "lp_obj": float(selected["lp_obj"]),
        "route_weight": float(selected["route_weight"]),
        "artificials": float(selected["artificials"]),
        "min_rc": float(selected["min_rc"]),
        "max_row_violation": None,
        "max_bound_violation": None,
    }
    snapshot = dict(source_status)
    snapshot.update({
        "iterations": selected_iteration,
        "attempt_iterations": selected_iteration,
        "certified_rc_optimal": False,
        "final": final,
        "columns": unique_columns,
        "columns_journal": str(output_journal),
        "wall_s": float(selected["elapsed_s"]),
        "attempt_wall_s": float(selected["elapsed_s"]),
        "peak_rss_mb": peak_rss_mb,
        "stop_reason": "matched_wall_snapshot",
        "termination_signal": None,
        "history_tail": [final],
        "final_lp": None,
        "final_lp_source": None,
        "resume_parent": {
            "kind": "posthoc_last_complete_iteration_snapshot",
            "source_status": str(source),
            "source_status_sha256": _sha256(source_status_bytes),
            "source_journal": str(source_journal),
            "source_journal_sha256": _sha256(source_journal_bytes),
            "source_iterations": str(source_iterations),
            "source_iterations_sha256":
                _sha256(source_iterations.read_bytes()),
        },
        "matched_wall_snapshot": {
            "schema": SCHEMA,
            "requested_budget_s": float(args.budget_s),
            "included_iteration": selected_iteration,
            "included_elapsed_s": float(selected["elapsed_s"]),
            "journal_record_count": journal_records,
            "unique_pool_columns": unique_columns,
            "telemetry": telemetry_identity,
            "conservative_boundary":
                "columns_found_at_included_iteration_are_excluded",
        },
    })
    atomic_write_json(output, snapshot)
    if (
        source.read_bytes() != source_status_bytes
        or source_journal.read_bytes() != source_journal_bytes
    ):
        raise RuntimeError("source CG artifacts changed during snapshot")
    return snapshot


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--budget-s", type=float, required=True)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.budget_s <= 0:
        parser.error("--budget-s must be positive")
    snapshot = freeze(args)
    print(json.dumps({
        "iteration": snapshot["iterations"],
        "elapsed_s": snapshot["wall_s"],
        "columns": snapshot["columns"],
        "journal_sha256": _sha256(
            Path(snapshot["columns_journal"]).read_bytes()
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

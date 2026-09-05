#!/usr/bin/env python3
"""Freeze a historical exact-CG journal prefix with bounded memory.

Unlike ``freeze_exact_cg_at_wall.py``, this tool is intended for a durable CG
run that is still appending work after the requested boundary. It authenticates
and copies only records with ``found_iter`` strictly before the selected
iteration. Later append-only records cannot change that historical prefix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from durable_io import atomic_write_json, flush_and_fsync
from run_exact_pool_mip import resolve_pool_journal


SCHEMA = "evsp-dr-exact-cg-prefix-snapshot-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_iteration(path: Path, budget_s: float) -> tuple[list[str], list[dict], dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            try:
                elapsed = float(row["elapsed_s"])
                iteration = int(row["iteration"])
                int(row["pool_columns"])
            except (KeyError, TypeError, ValueError):
                # A concurrently appended final row may be incomplete. It is
                # irrelevant when the requested historical boundary precedes it.
                continue
            row["elapsed_s"] = str(elapsed)
            row["iteration"] = str(iteration)
            rows.append(row)
    if not fieldnames:
        raise ValueError("iteration log has no header")
    if any(
        int(current["iteration"]) <= int(previous["iteration"])
        or float(current["elapsed_s"]) < float(previous["elapsed_s"])
        for previous, current in zip(rows, rows[1:])
    ):
        raise ValueError("iteration log is not strictly ordered")
    eligible = [
        row for row in rows if float(row["elapsed_s"]) <= budget_s + 1e-9
    ]
    if not eligible:
        raise ValueError("no complete CG iteration exists within wall budget")
    return fieldnames, rows, eligible[-1]


def freeze_prefix(args) -> dict:
    source = Path(args.result).expanduser().resolve(strict=True)
    output = Path(args.out).expanduser().resolve()
    output_journal = Path(str(output) + ".columns.jsonl")
    output_iterations = Path(str(output) + ".iters.csv")
    for path in (output, output_journal, output_iterations):
        if os.path.lexists(path):
            raise FileExistsError(path)

    source_status_bytes = source.read_bytes()
    source_status = json.loads(source_status_bytes)
    source_journal = resolve_pool_journal(source, source_status).resolve(strict=True)
    source_iterations = Path(str(source) + ".iters.csv").resolve(strict=True)
    fields, iteration_rows, selected = selected_iteration(
        source_iterations, args.budget_s
    )
    selected_iter = int(selected["iteration"])
    expected_columns = int(selected["pool_columns"])

    output.parent.mkdir(parents=True, exist_ok=True)
    prefix_digest = hashlib.sha256()
    retained_records = 0
    retained_bytes = 0
    incidences = set()
    previous_found_iter = -1
    observed_post_boundary = False
    with source_journal.open("rb") as source_stream, output_journal.open("xb") as out:
        for ordinal, raw in enumerate(source_stream, start=1):
            if not raw.endswith(b"\n"):
                raise ValueError(
                    f"journal line {ordinal} is an incomplete live append"
                )
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"journal line {ordinal} is malformed") from exc
            found_iter = record.get("found_iter", 0)
            if not isinstance(found_iter, int) or isinstance(found_iter, bool):
                raise ValueError(f"journal line {ordinal} has invalid found_iter")
            if found_iter < previous_found_iter:
                raise ValueError("column journal found_iter order regressed")
            previous_found_iter = found_iter
            if found_iter >= selected_iter:
                observed_post_boundary = True
                break
            trips = record.get("trips")
            if (
                not isinstance(trips, list) or not trips
                or any(not isinstance(trip, int) or isinstance(trip, bool) for trip in trips)
                or len(trips) != len(set(trips))
            ):
                raise ValueError(f"journal line {ordinal} has invalid trips")
            try:
                cost = float(record["cost"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"journal line {ordinal} has invalid cost"
                ) from exc
            if not math.isfinite(cost):
                raise ValueError(f"journal line {ordinal} has invalid cost")
            incidence = frozenset(trips)
            incidences.add(incidence)
            out.write(raw)
            prefix_digest.update(raw)
            retained_records += 1
            retained_bytes += len(raw)
        flush_and_fsync(out)
    if not observed_post_boundary:
        raise ValueError(
            "live journal has not advanced past the selected historical iteration"
        )
    if len(incidences) != expected_columns:
        raise ValueError(
            "frozen journal prefix does not reproduce iteration pool size: "
            f"records={retained_records}, unique={len(incidences)}, "
            f"expected={expected_columns}"
        )

    frozen_rows = [
        row for row in iteration_rows if int(row["iteration"]) <= selected_iter
    ]
    with output_iterations.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(frozen_rows)
        flush_and_fsync(handle)

    final = {
        "iter": selected_iter,
        "attempt_iter": selected_iter,
        "lp_obj": float(selected["lp_obj"]),
        "route_weight": float(selected["route_weight"]),
        "artificials": float(selected["artificials"]),
        "min_rc": float(selected["min_rc"]),
        "max_row_violation": None,
        "max_bound_violation": None,
    }
    snapshot = dict(source_status)
    snapshot.update({
        "iterations": selected_iter,
        "attempt_iterations": selected_iter,
        "certified_rc_optimal": False,
        "final": final,
        "columns": expected_columns,
        "columns_journal": str(output_journal),
        "wall_s": float(selected["elapsed_s"]),
        "attempt_wall_s": float(selected["elapsed_s"]),
        "peak_rss_mb": None,
        "stop_reason": "historical_prefix_snapshot",
        "termination_signal": None,
        "history_tail": [final],
        "final_lp": None,
        "final_lp_source": None,
        "resume_parent": {
            "kind": "posthoc_append_only_journal_prefix_snapshot",
            "source_status": str(source),
            "source_status_sha256": hashlib.sha256(source_status_bytes).hexdigest(),
            "source_journal": str(source_journal),
            "source_journal_prefix_sha256": prefix_digest.hexdigest(),
            "source_journal_prefix_bytes": retained_bytes,
            "source_iterations": str(source_iterations),
            "source_iterations_observed_sha256": file_sha256(source_iterations),
        },
        "matched_wall_snapshot": {
            "schema": SCHEMA,
            "requested_budget_s": float(args.budget_s),
            "included_iteration": selected_iter,
            "included_elapsed_s": float(selected["elapsed_s"]),
            "journal_record_count": retained_records,
            "unique_pool_columns": len(incidences),
            "journal_prefix_sha256": prefix_digest.hexdigest(),
            "observed_first_post_boundary_record": observed_post_boundary,
            "conservative_boundary":
                "columns_found_at_included_iteration_are_excluded",
        },
    })
    atomic_write_json(output, snapshot)
    return snapshot


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--budget-s", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.budget_s <= 0:
        parser.error("--budget-s must be positive")
    snapshot = freeze_prefix(args)
    print(json.dumps({
        "iteration": snapshot["iterations"],
        "elapsed_s": snapshot["wall_s"],
        "columns": snapshot["columns"],
        "journal_sha256": file_sha256(Path(snapshot["columns_journal"])),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

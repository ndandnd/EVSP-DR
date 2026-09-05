#!/usr/bin/env python3
"""Freeze the four hard k=5 RAW pools at a common 36-hour CG boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


EXPECTED_CELLS = {"k05_p5", "k05_xenergy", "k05_xgap", "k05_xtrip"}
EXPECTED_REPRESENTATION = "event_2p5_event5"
EXPECTED_BUDGET_S = 36 * 60 * 60
MAX_BOUNDARY_LAG_S = 10 * 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty {label}: {path}")
    return path.resolve()


def load_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise SystemExit(f"empty resume matrix: {path}")
    selected = [row for row in rows if row.get("target_fleet") == "5"]
    cells = {row.get("cell") for row in selected}
    if len(selected) != 4 or cells != EXPECTED_CELLS:
        raise SystemExit(
            "expected exactly the four predeclared hard k=5 rows; "
            f"found {len(selected)} rows and cells={sorted(str(x) for x in cells)}"
        )
    indices = [row.get("source_panel_index") for row in selected]
    if len(indices) != len(set(indices)):
        raise SystemExit("k=5 resume rows repeat a source panel index")
    return sorted(selected, key=lambda row: row["cell"])


def validate_source(
    row: dict[str, str], budget_s: float, planned_columns_per_iter: int = 30,
) -> Path:
    source = required_file(Path(row["resume_status"]), "resume status")
    status = json.loads(source.read_text())
    expected = {
        "time_model": "event",
        "event_arc_mode": "lazy",
        "soc_step": 2.5,
        "block_min": 5,
        "g_kwh": 240.0,
        "charge_kw": 240.0,
        "min_soc_frac": 0.0,
        "prices_csv": "hourly_prices_flat.csv",
        "master_sense": "partition",
        "initial_pool": "singletons",
        "columns_per_iter": planned_columns_per_iter,
        "column_pool_treatment": "RAW",
    }
    observed = {
        "time_model": status.get("time_model"),
        "event_arc_mode": (status.get("network_metrics") or {}).get("arc_mode"),
        "soc_step": float(status.get("soc_step", -1)),
        "block_min": int(status.get("block_min", -1)),
        "g_kwh": float(status.get("g_kwh", -1)),
        "charge_kw": float(status.get("charge_kw", -1)),
        "min_soc_frac": float(status.get("min_soc_frac", -1)),
        "prices_csv": status.get("prices_csv"),
        "master_sense": status.get("master_sense"),
        "initial_pool": status.get("initial_pool"),
        # The reviewed resume solver predates this field in some periodic
        # status payloads. In that case its immutable execution plan is the
        # authority; a present status value must still agree with the plan.
        "columns_per_iter": int(
            status.get("columns_per_iter", planned_columns_per_iter)
        ),
        "column_pool_treatment": status.get("column_pool_treatment"),
    }
    if observed != expected:
        raise SystemExit(
            f"source configuration mismatch for {row['cell']}: {observed}"
        )
    if row.get("representation_id") != EXPECTED_REPRESENTATION:
        raise SystemExit(f"unexpected representation for {row['cell']}")
    if str(status.get("csv")) != row.get("instance_csv"):
        raise SystemExit(f"instance path mismatch for {row['cell']}")
    provenance = status.get("provenance")
    required_hashes = (
        "instance_sha256", "prices_sha256", "reference_sha256",
        "deadhead_sha256",
    )
    if not isinstance(provenance, dict) or any(
        not isinstance(provenance.get(key), str) or len(provenance[key]) != 64
        for key in required_hashes
    ):
        raise SystemExit(f"incomplete hashed provenance for {row['cell']}")
    iterations = required_file(
        Path(str(source) + ".iters.csv"), "resume iteration log"
    )
    with iterations.open(newline="") as handle:
        iteration_rows = list(csv.DictReader(handle))
    eligible = []
    for item in iteration_rows:
        try:
            elapsed = float(item["elapsed_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if elapsed <= budget_s + 1e-9:
            eligible.append(elapsed)
    if not eligible:
        raise SystemExit(f"no complete iteration before 36h for {row['cell']}")
    if eligible[-1] < budget_s - MAX_BOUNDARY_LAG_S:
        raise SystemExit(
            f"last iteration before 36h is too early for {row['cell']}: "
            f"{eligible[-1]:.3f}s"
        )
    return source


def run_freezer(
    freezer: Path,
    python: Path,
    source: Path,
    output: Path,
    budget_s: float,
    attempts: int,
) -> None:
    """Retry only the benign race where a live durable source advances."""

    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        temporary = Path(tempfile.mkdtemp(
            prefix=f".freeze-{output.stem}-", dir=output.parent
        ))
        candidate = temporary / output.name
        command = [
            str(python), str(freezer), "--result", str(source),
            "--budget-s", str(budget_s), "--out", str(candidate),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode == 0:
            candidate_journal = Path(str(candidate) + ".columns.jsonl")
            candidate_iters = Path(str(candidate) + ".iters.csv")
            final_journal = Path(str(output) + ".columns.jsonl")
            final_iters = Path(str(output) + ".iters.csv")
            for path in (candidate, candidate_journal, candidate_iters):
                required_file(path, "temporary frozen artifact")
            for path in (output, final_journal, final_iters):
                if os.path.lexists(path):
                    raise SystemExit(f"refusing to overwrite frozen artifact: {path}")
            os.replace(candidate_journal, final_journal)
            os.replace(candidate_iters, final_iters)
            payload = json.loads(candidate.read_text())
            payload["columns_journal"] = str(final_journal.resolve())
            rewritten = candidate.with_name(f".{candidate.name}.canonical")
            with rewritten.open("x") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(rewritten, output)
            shutil.rmtree(temporary)
            print(completed.stdout.strip())
            return
        diagnostic = (completed.stderr + "\n" + completed.stdout).strip()
        shutil.rmtree(temporary)
        if "source CG artifacts changed during snapshot" not in diagnostic:
            raise SystemExit(
                f"freezer failed for {source} (attempt {attempt}, "
                f"return code {completed.returncode}):\n{diagnostic}"
            )
        if attempt == attempts:
            raise SystemExit(
                f"live source changed during all {attempts} freeze attempts: {source}"
            )
        print(
            f"source advanced during freeze attempt {attempt}/{attempts}; retrying",
            file=sys.stderr,
        )
        time.sleep(5)


def validate_snapshot(path: Path, cell: str, budget_s: float) -> dict:
    payload = json.loads(required_file(path, "snapshot").read_text())
    matched = payload.get("matched_wall_snapshot") or {}
    final = payload.get("final") or {}
    journal = required_file(
        Path(str(payload.get("columns_journal", ""))), "snapshot journal"
    )
    if matched.get("schema") not in {
        "evsp-dr-exact-cg-matched-wall-snapshot-v1",
        "evsp-dr-exact-cg-prefix-snapshot-v1",
    }:
        raise SystemExit(f"snapshot schema mismatch for {cell}")
    if abs(float(matched.get("requested_budget_s", -1)) - budget_s) > 1e-9:
        raise SystemExit(f"snapshot budget mismatch for {cell}")
    elapsed = float(matched.get("included_elapsed_s", -1))
    if not budget_s - MAX_BOUNDARY_LAG_S <= elapsed <= budget_s + 1e-9:
        raise SystemExit(f"snapshot boundary mismatch for {cell}: {elapsed}")
    if int(matched.get("unique_pool_columns", -1)) != int(payload.get("columns", -2)):
        raise SystemExit(f"snapshot pool count mismatch for {cell}")
    if abs(float(final.get("route_weight", -1)) - 5.0) > 1e-7:
        raise SystemExit(f"36h RMP fleet endpoint is not 5 for {cell}")
    if abs(float(final.get("artificials", -1))) > 1e-7:
        raise SystemExit(f"36h RMP retains artificials for {cell}")
    if payload.get("column_pool_treatment") != "RAW":
        raise SystemExit(f"snapshot is not RAW for {cell}")
    return {
        "snapshot": str(path.resolve()),
        "snapshot_sha256": sha256(path),
        "journal": str(journal),
        "journal_sha256": sha256(journal),
        "iteration": int(matched["included_iteration"]),
        "elapsed_s": elapsed,
        "pool_columns": int(payload["columns"]),
        "route_weight_endpoint": float(final["route_weight"]),
        "artificials": float(final["artificials"]),
        "min_rc": float(final["min_rc"]),
        "lp_obj": float(final["lp_obj"]),
        "instance": payload["csv"],
        "instance_sha256": payload["provenance"]["instance_sha256"],
    }


def write_manifests(output_root: Path, records: list[dict]) -> None:
    fieldnames = list(records[0])
    csv_path = output_root / "snapshot_manifest.csv"
    tsv_path = output_root / "snapshot_manifest.tsv"
    for path, delimiter in ((csv_path, ","), (tsv_path, "\t")):
        with path.open("x", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, delimiter=delimiter,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(records)
    (output_root / "snapshot_manifest.sha256").write_text(
        f"{sha256(csv_path)}  {csv_path.name}\n"
        f"{sha256(tsv_path)}  {tsv_path.name}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--freezer", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--budget-s", type=float, default=EXPECTED_BUDGET_S)
    parser.add_argument("--race-attempts", type=int, default=12)
    args = parser.parse_args()
    if abs(args.budget_s - EXPECTED_BUDGET_S) > 1e-9:
        parser.error("this experiment requires the predeclared 36-hour boundary")
    if args.race_attempts <= 0:
        parser.error("--race-attempts must be positive")

    resume_root = args.resume_root.resolve()
    output_root = args.output_root.resolve()
    freezer = required_file(args.freezer.resolve(), "reviewed freezer")
    python = required_file(args.python.resolve(), "Python interpreter")
    plan_path = required_file(
        resume_root / "execution_plan.json", "resume execution plan"
    )
    plan = json.loads(plan_path.read_text())
    expected_plan = {
        "schema": "evsp-dr-wall-capped-event-resume-v1",
        "cells": 23,
        "parent_cumulative_wall_limit_s": 43200.0,
        "cumulative_scientific_wall_limit_s": 172800.0,
        "columns_per_iter": 30,
    }
    observed_plan = {
        "schema": plan.get("schema"),
        "cells": int(plan.get("cells", -1)),
        "parent_cumulative_wall_limit_s": float(
            plan.get("parent_cumulative_wall_limit_s", -1)
        ),
        "cumulative_scientific_wall_limit_s": float(
            plan.get("cumulative_scientific_wall_limit_s", -1)
        ),
        "columns_per_iter": int(plan.get("columns_per_iter", -1)),
    }
    if observed_plan != expected_plan:
        raise SystemExit(f"resume execution-plan mismatch: {observed_plan}")
    rows = load_matrix(required_file(resume_root / "matrix.tsv", "resume matrix"))
    output_root.mkdir(parents=True, exist_ok=True)
    if any((output_root / name).exists() for name in (
        "snapshots", "mip", "snapshot_manifest.csv", "snapshot_manifest.tsv",
    )):
        raise SystemExit(f"output root already contains prepared artifacts: {output_root}")
    (output_root / "snapshots").mkdir()
    (output_root / "mip").mkdir()
    (output_root / "logs").mkdir(exist_ok=True)

    records = []
    for local_index, row in enumerate(rows):
        source = validate_source(
            row, args.budget_s, planned_columns_per_iter=plan["columns_per_iter"]
        )
        snapshot = (
            output_root / "snapshots"
            / f"M__{row['cell']}__{EXPECTED_REPRESENTATION}.snapshot.json"
        )
        run_freezer(
            freezer, python, source, snapshot, args.budget_s,
            args.race_attempts,
        )
        records.append({
            "local_index": local_index,
            "source_panel_index": row["source_panel_index"],
            "cell": row["cell"],
            "target_fleet": 5,
            "representation_id": EXPECTED_REPRESENTATION,
            "source_status": str(source),
            **validate_snapshot(snapshot, row["cell"], args.budget_s),
        })
    write_manifests(output_root, records)
    print(f"Frozen four RAW k=5 pools at cumulative 36h: {output_root}")
    print(f"CSV: {output_root / 'snapshot_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

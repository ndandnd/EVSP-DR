#!/usr/bin/env python3
"""Stage immutable 12h-to-24h continuations of wall-capped event-CG cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--solver-commit", required=True)
    parser.add_argument("--parent-wall-limit-s", type=float, required=True)
    parser.add_argument("--wall-limit-s", type=float, required=True)
    parser.add_argument("--expected-cells", type=int)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    out_root = args.out_root.resolve()
    if out_root.exists():
        raise SystemExit(f"resume root already exists: {out_root}")
    if args.wall_limit_s <= args.parent_wall_limit_s:
        raise SystemExit("continuation cap must exceed parent cap")

    source_plan_path = required_file(
        source_root / "execution_plan.json", "source execution plan"
    )
    source_plan = json.loads(source_plan_path.read_text())
    if source_plan.get("solver_commit") != args.solver_commit:
        raise SystemExit("source solver commit does not match launcher")
    if abs(
        float(source_plan.get("wall_limit_s", 0.0))
        - args.parent_wall_limit_s
    ) > 1e-9:
        raise SystemExit("source execution-plan wall cap mismatch")

    matrix_path = required_file(source_root / "matrix.tsv", "source matrix")
    audit_path = required_file(
        source_root / "medium_event_summary.csv", "source campaign audit"
    )
    with audit_path.open(newline="") as handle:
        audit_rows = {int(row["index"]): row for row in csv.DictReader(handle)}
    with matrix_path.open(newline="") as handle:
        matrix = list(csv.reader(handle, delimiter="\t"))
    if set(audit_rows) != set(range(len(matrix))):
        raise SystemExit("source audit and matrix index sets differ")

    selected = []
    for fields in matrix:
        if len(fields) != 11:
            raise SystemExit(f"unexpected source matrix width: {len(fields)}")
        (
            index_text,
            cell,
            scale,
            _replicate,
            _trips,
            instance_csv,
            instance_sha256,
            representation,
            soc_step,
            block_min,
            row_wall_limit,
        ) = fields
        source_index = int(index_text)
        audit = audit_rows[source_index]
        source_status = source_root / "cg" / f"M__{cell}__{representation}.json"
        if audit.get("stop_reason") != "wall_limit":
            continue
        source_status = required_file(source_status, "wall-capped status")
        if (
            audit.get("result_present") != "True"
            or audit.get("result_sha256") != sha256(source_status)
            or audit.get("slurm_state") != "COMPLETED"
        ):
            raise SystemExit(f"unqualified wall-capped audit row {source_index}")
        status = json.loads(source_status.read_text())
        wall_s = float(status.get("wall_s", 0.0))
        if (
            status.get("certified_rc_optimal") is True
            or status.get("stop_reason") != "wall_limit"
            or wall_s < args.parent_wall_limit_s - 120.0
            or wall_s >= args.wall_limit_s - 120.0
        ):
            raise SystemExit(f"invalid wall-capped status at {source_index}")
        if abs(float(row_wall_limit) - args.parent_wall_limit_s) > 1e-9:
            raise SystemExit(f"source row wall cap mismatch at {source_index}")
        instance_path = required_file(Path(instance_csv), "instance CSV")
        if sha256(instance_path) != instance_sha256:
            raise SystemExit(f"instance hash changed at {source_index}")
        status_csv = required_file(
            Path(str(status.get("csv", ""))), "status CSV"
        )
        if status_csv != instance_path:
            raise SystemExit(f"status instance mismatch at {source_index}")
        expected_config = {
            "time_model": "event",
            "event_arc_mode": "lazy",
            "soc_step": float(soc_step),
            "block_min": int(block_min),
            "g_kwh": 240.0,
            "charge_kw": 240.0,
            "min_soc_frac": 0.0,
        }
        observed_config = {
            "time_model": status.get("time_model"),
            "event_arc_mode": (status.get("network_metrics") or {}).get(
                "arc_mode"
            ),
            "soc_step": float(status.get("soc_step", -1)),
            "block_min": int(status.get("block_min", -1)),
            "g_kwh": float(status.get("g_kwh", -1)),
            "charge_kw": float(status.get("charge_kw", -1)),
            "min_soc_frac": float(status.get("min_soc_frac", -1)),
        }
        if observed_config != expected_config:
            raise SystemExit(
                f"status configuration mismatch at {source_index}: "
                f"{observed_config}"
            )
        source_journal_text = status.get("columns_journal")
        if not source_journal_text:
            raise SystemExit(f"missing columns journal at {source_index}")
        source_journal = required_file(
            Path(source_journal_text), "wall-capped columns journal"
        )
        source_iters = required_file(
            Path(str(source_status) + ".iters.csv"), "wall-capped iteration log"
        )
        source_telemetry = required_file(
            Path(str(source_status) + ".phase-telemetry.jsonl"),
            "wall-capped phase telemetry",
        )
        selected.append({
            "source_index": source_index,
            "cell": cell,
            "scale": scale,
            "instance_csv": str(instance_path),
            "representation": representation,
            "soc_step": soc_step,
            "block_min": block_min,
            "source_status": source_status,
            "source_journal": source_journal,
            "source_iters": source_iters,
            "source_telemetry": source_telemetry,
            "source_wall_s": wall_s,
        })

    if args.expected_cells is not None and len(selected) != args.expected_cells:
        raise SystemExit(
            f"expected {args.expected_cells} wall-capped cells, "
            f"found {len(selected)}"
        )
    (out_root / "cg").mkdir(parents=True)
    (out_root / "logs").mkdir()
    manifest_rows = []
    fieldnames = [
        "local_index", "source_panel_index", "cell", "target_fleet",
        "instance_csv", "representation_id", "soc_step", "block_min",
        "source_status", "source_status_sha256", "source_journal",
        "source_journal_sha256", "resume_status", "resume_journal",
        "staged_status_sha256", "staged_journal_sha256",
    ]
    for local_index, item in enumerate(selected):
        source_status = item["source_status"]
        source_journal = item["source_journal"]
        destination = out_root / "cg" / source_status.name
        destination_journal = Path(str(destination) + ".columns.jsonl")
        for source, target in (
            (source_status, destination),
            (source_journal, destination_journal),
            (item["source_iters"], Path(str(destination) + ".iters.csv")),
            (
                item["source_telemetry"],
                Path(str(destination) + ".source-phase-telemetry.jsonl"),
            ),
        ):
            shutil.copyfile(source, target)
        manifest_rows.append({
            "local_index": local_index,
            "source_panel_index": item["source_index"],
            "cell": item["cell"],
            "target_fleet": item["scale"],
            "instance_csv": item["instance_csv"],
            "representation_id": item["representation"],
            "soc_step": item["soc_step"],
            "block_min": item["block_min"],
            "source_status": str(source_status),
            "source_status_sha256": sha256(source_status),
            "source_journal": str(source_journal),
            "source_journal_sha256": sha256(source_journal),
            "resume_status": str(destination),
            "resume_journal": str(destination_journal),
            "staged_status_sha256": sha256(destination),
            "staged_journal_sha256": sha256(destination_journal),
        })
    with (out_root / "matrix.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    (out_root / "execution_plan.json").write_text(
        json.dumps({
            "schema": "evsp-dr-wall-capped-event-resume-v1",
            "source_root": str(source_root),
            "source_execution_plan_sha256": sha256(source_plan_path),
            "source_matrix_sha256": sha256(matrix_path),
            "source_audit_sha256": sha256(audit_path),
            "solver_commit": args.solver_commit,
            "cells": len(manifest_rows),
            "selection_stop_reason": "wall_limit_at_parent_cap",
            "parent_cumulative_wall_limit_s": args.parent_wall_limit_s,
            "cumulative_scientific_wall_limit_s": args.wall_limit_s,
            "max_iters": 20000,
            "columns_per_iter": 30,
            "telemetry_policy": (
                "archive source telemetry and start fresh identity-bound streams"
            ),
            "preserves_original_artifacts": True,
        }, indent=2, sort_keys=True) + "\n"
    )
    counts = {}
    for item in selected:
        counts[item["scale"]] = counts.get(item["scale"], 0) + 1
    print(f"staged {len(selected)} wall-capped event continuations: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

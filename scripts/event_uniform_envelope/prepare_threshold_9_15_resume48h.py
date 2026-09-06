#!/usr/bin/env python3
"""Stage immutable cumulative-48h resumes for the k9--k15 baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified(source: Path, target: Path) -> str:
    source_digest = sha256(source)
    if target.is_file() and sha256(target) == source_digest:
        return source_digest
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != source_digest:
            raise SystemExit(f"copied artifact hash mismatch: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return source_digest


def required(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--solver-commit", required=True)
    parser.add_argument("--parent-wall-limit-s", type=float, required=True)
    parser.add_argument("--wall-limit-s", type=float, required=True)
    parser.add_argument("--expected-cells", type=int, required=True)
    parser.add_argument("--resume-incomplete", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    out_root = args.out_root.resolve()
    completion_marker = out_root / "STAGING_COMPLETE"
    if out_root.exists() and not args.resume_incomplete:
        raise SystemExit(f"resume root already exists: {out_root}")
    if completion_marker.exists():
        raise SystemExit(f"resume root is already complete: {out_root}")
    if args.wall_limit_s <= args.parent_wall_limit_s:
        raise SystemExit("continuation cap must exceed parent cap")

    plan_path = required(source_root / "execution_plan.json", "source plan")
    matrix_path = required(source_root / "matrix.tsv", "source matrix")
    selection_path = required(
        source_root / "input_selection_manifest.csv", "selection manifest"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "evsp-dr-threshold-9-15-event-cg-v1":
        raise SystemExit("unexpected source campaign schema")
    if plan.get("solver_commit") != args.solver_commit:
        raise SystemExit("source solver commit mismatch")
    if int(plan.get("cells", -1)) != 70:
        raise SystemExit("source campaign does not contain 70 cells")
    if abs(
        float(plan.get("wall_limit_s_per_cg_arm", 0.0))
        - args.parent_wall_limit_s
    ) > 1e-9:
        raise SystemExit("source parent wall limit mismatch")
    if plan.get("representation") != "event_2p5_event5":
        raise SystemExit("source representation mismatch")
    if sha256(selection_path) != plan.get("input_selection_manifest_sha256"):
        raise SystemExit("selection manifest hash mismatch")
    arms = plan.get("arms") or []
    if arms != [{
        "arm": "b030_reduced", "columns_per_iter": 30,
        "selection": "reduced_cost", "diversity_weight": 0.0,
    }]:
        raise SystemExit(f"unexpected source arm: {arms}")

    with matrix_path.open(newline="", encoding="utf-8") as handle:
        matrix = list(csv.reader(handle, delimiter="\t"))
    if len(matrix) != 70:
        raise SystemExit(f"expected 70 source rows, found {len(matrix)}")

    selected: list[dict] = []
    terminal = Counter()
    for fields in matrix:
        if len(fields) != 11:
            raise SystemExit(f"unexpected source matrix width: {len(fields)}")
        (
            index, cell, scale, _replicate, _trips, instance_text,
            instance_hash, representation, soc_step, block_min, row_cap,
        ) = fields
        if abs(float(row_cap) - args.parent_wall_limit_s) > 1e-9:
            raise SystemExit(f"source row wall cap mismatch at {index}")
        instance = required(Path(instance_text), "instance CSV")
        if sha256(instance) != instance_hash:
            raise SystemExit(f"instance hash mismatch at {index}")
        status_path = required(
            source_root / "cg" / "b030_reduced"
            / f"M__{cell}__{representation}.json",
            "source status",
        )
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("certified_rc_optimal") is True:
            terminal["certified"] += 1
            continue
        if status.get("stop_reason") != "wall_limit":
            raise SystemExit(
                f"nonterminal source result at {index}: "
                f"{status.get('stop_reason')}"
            )
        wall_s = float(status.get("wall_s") or 0.0)
        if not args.parent_wall_limit_s - 120.0 <= wall_s < args.wall_limit_s - 120.0:
            raise SystemExit(f"source wall time outside resume boundary at {index}")
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
            "prices_csv": status.get("prices_csv"),
            "time_model": status.get("time_model"),
            "event_arc_mode": (status.get("network_metrics") or {}).get(
                "arc_mode"
            ),
            "soc_step": float(status.get("soc_step", -1)),
            "block_min": int(status.get("block_min", -1)),
            "g_kwh": float(status.get("g_kwh", -1)),
            "charge_kw": float(status.get("charge_kw", -1)),
            "min_soc_frac": float(status.get("min_soc_frac", -1)),
            "master_sense": status.get("master_sense"),
            "initial_pool": status.get("initial_pool"),
            "columns_per_iter": int(status.get("columns_per_iter", -1)),
            "column_selection": status.get("column_selection"),
            "column_diversity_weight": float(
                status.get("column_diversity_weight", -1)
            ),
            "column_candidate_multiplier": int(
                status.get("column_candidate_multiplier", -1)
            ),
            "column_pool_treatment": status.get("column_pool_treatment"),
        }
        expected_config.update({
            "prices_csv": "hourly_prices_flat.csv",
            "master_sense": "partition",
            "initial_pool": "singletons",
            "columns_per_iter": 30,
            "column_selection": "reduced_cost",
            "column_diversity_weight": 0.0,
            "column_candidate_multiplier": 4,
            "column_pool_treatment": "RAW",
        })
        if observed_config != expected_config:
            raise SystemExit(
                f"source configuration mismatch at {index}: {observed_config}"
            )
        status_csv = required(Path(str(status.get("csv", ""))), "status CSV")
        if status_csv != instance:
            raise SystemExit(f"status instance mismatch at {index}")
        source_journal = required(
            Path(str(status.get("columns_journal", ""))), "source journal"
        )
        source_iters = required(
            Path(str(status_path) + ".iters.csv"), "source iteration log"
        )
        source_telemetry = required(
            Path(str(status_path) + ".phase-telemetry.jsonl"),
            "source telemetry",
        )
        cache = required(
            source_root / "network_cache"
            / f"M__{cell}__{representation}.pkl",
            "event-network cache",
        )
        cache_manifest = required(
            Path(str(cache) + ".manifest.json"), "cache manifest"
        )
        manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
        identity = manifest.get("identity") or {}
        if (
            manifest.get("schema") != "evsp-dr-event-network-cache-v1"
            or not isinstance(manifest.get("pickle_sha256"), str)
            or len(manifest["pickle_sha256"]) != 64
            or identity.get("git_commit") != args.solver_commit
            or identity.get("instance_sha256") != instance_hash
            or float(identity.get("soc_step", -1)) != float(soc_step)
            or int(identity.get("block_min", -1)) != int(block_min)
            or float(identity.get("g_kwh", -1)) != 240.0
            or float(identity.get("charge_kw", -1)) != 240.0
            or float(identity.get("reserve_kwh", -1)) != 0.0
            or identity.get("event_arc_mode") != "lazy"
        ):
            raise SystemExit(f"event-network cache identity mismatch at {index}")
        provenance = status.get("provenance") or {}
        if (
            provenance.get("git_commit") != args.solver_commit
            or provenance.get("instance_sha256") != instance_hash
            or float(provenance.get("rc_eps", -1)) != 0.0001
        ):
            raise SystemExit(f"source provenance mismatch at {index}")
        selected.append({
            "source_index": index, "cell": cell, "scale": scale,
            "instance": instance, "representation": representation,
            "soc_step": soc_step, "block_min": block_min,
            "source_status": status_path, "source_journal": source_journal,
            "source_iters": source_iters,
            "source_telemetry": source_telemetry, "cache": cache,
            "cache_manifest": cache_manifest,
        })
        terminal["wall_limit"] += 1

    if len(selected) != args.expected_cells:
        raise SystemExit(
            f"expected {args.expected_cells} wall-capped cells, "
            f"found {len(selected)}; outcomes={dict(terminal)}"
        )

    (out_root / "cg").mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(exist_ok=True)
    rows = []
    for local_index, item in enumerate(selected):
        destination = out_root / "cg" / item["source_status"].name
        destination_journal = Path(str(destination) + ".columns.jsonl")
        copied_digests = {}
        for source, target in (
            (item["source_status"], destination),
            (item["source_journal"], destination_journal),
            (item["source_iters"], Path(str(destination) + ".iters.csv")),
            (
                item["source_telemetry"],
                Path(str(destination) + ".source-phase-telemetry.jsonl"),
            ),
        ):
            copied_digests[target] = copy_verified(source, target)
        rows.append({
            "local_index": local_index,
            "source_panel_index": item["source_index"],
            "cell": item["cell"],
            "target_fleet": item["scale"],
            "instance_csv": str(item["instance"]),
            "representation_id": item["representation"],
            "soc_step": item["soc_step"],
            "block_min": item["block_min"],
            "source_status": str(item["source_status"]),
            "source_status_sha256": copied_digests[destination],
            "source_journal": str(item["source_journal"]),
            "source_journal_sha256": copied_digests[destination_journal],
            "resume_status": str(destination),
            "resume_journal": str(destination_journal),
            "staged_status_sha256": copied_digests[destination],
            "staged_journal_sha256": copied_digests[destination_journal],
            "event_network_cache": str(item["cache"]),
            "event_network_cache_manifest": str(item["cache_manifest"]),
            "event_network_cache_manifest_sha256": sha256(
                item["cache_manifest"]
            ),
        })

    with (out_root / "matrix.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    plan_output = out_root / "execution_plan.json"
    plan_output.write_text(json.dumps({
        "schema": "evsp-dr-threshold-9-15-resume48h-v1",
        "source_root": str(source_root),
        "source_execution_plan_sha256": sha256(plan_path),
        "source_matrix_sha256": sha256(matrix_path),
        "source_selection_manifest_sha256": sha256(selection_path),
        "solver_commit": args.solver_commit,
        "cells": len(rows),
        "source_outcomes": dict(sorted(terminal.items())),
        "selection_stop_reason": "wall_limit_at_12h",
        "parent_cumulative_wall_limit_s": args.parent_wall_limit_s,
        "cumulative_scientific_wall_limit_s": args.wall_limit_s,
        "max_iters": 50000,
        "columns_per_iter": 30,
        "column_selection": "reduced_cost",
        "column_candidate_multiplier": 4,
        "event_network_cache_policy": "reuse_source_require_valid",
        "preserves_original_artifacts": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completion_marker.write_text(
        f"execution_plan.json {sha256(plan_output)}\n"
        f"matrix.tsv {sha256(out_root / 'matrix.tsv')}\n",
        encoding="utf-8",
    )
    counts = Counter(row["target_fleet"] for row in rows)
    print(
        f"staged {len(rows)} cumulative-48h continuations: "
        f"{dict(sorted(counts.items(), key=lambda item: int(item[0])))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

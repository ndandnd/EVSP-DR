#!/usr/bin/env python3
"""Prepare immutable control files for the k9--k15 RAW-pool MIP campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


CELLS = 70
REPRESENTATION = "event_2p5_event5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty {label}: {path}")
    return path.resolve()


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--execution-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--source-dependency-job", default="")
    parser.add_argument("--active-source-index", type=int, required=True)
    parser.add_argument("--max-concurrent-total", type=int, required=True)
    parser.add_argument("--mip-memory", required=True)
    args = parser.parse_args()

    base = args.base_root.expanduser().resolve(strict=True)
    resume = args.resume_root.expanduser().resolve(strict=True)
    repo = args.execution_repo.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output root already exists: {output}")
    if git(repo, "rev-parse", "HEAD") != args.execution_commit:
        raise SystemExit("execution checkout commit mismatch")
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("execution checkout has tracked modifications")
    symbolic = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "-q", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    if symbolic.returncode == 0:
        raise SystemExit("execution checkout must be detached")
    if symbolic.returncode != 1:
        raise SystemExit("could not verify detached execution checkout")

    base_plan_path = required(base / "execution_plan.json", "base plan")
    base_matrix_path = required(base / "matrix.tsv", "base matrix")
    selection_path = required(
        base / "input_selection_manifest.csv", "selection manifest"
    )
    resume_plan_path = required(resume / "execution_plan.json", "resume plan")
    resume_matrix_path = required(resume / "matrix.tsv", "resume matrix")
    base_plan = json.loads(base_plan_path.read_text(encoding="utf-8"))
    resume_plan = json.loads(resume_plan_path.read_text(encoding="utf-8"))
    if (
        base_plan.get("schema") != "evsp-dr-threshold-9-15-event-cg-v1"
        or int(base_plan.get("cells", -1)) != CELLS
        or base_plan.get("representation") != REPRESENTATION
        or sha256(selection_path)
        != base_plan.get("input_selection_manifest_sha256")
    ):
        raise SystemExit("base campaign identity mismatch")
    if (
        resume_plan.get("schema") != "evsp-dr-threshold-9-15-resume48h-v1"
        or Path(resume_plan.get("source_root", "")).resolve() != base
        or resume_plan.get("source_execution_plan_sha256") != sha256(base_plan_path)
        or resume_plan.get("source_matrix_sha256") != sha256(base_matrix_path)
        or resume_plan.get("solver_commit") != base_plan.get("solver_commit")
        or float(resume_plan.get("cumulative_scientific_wall_limit_s", -1))
        != 172800.0
    ):
        raise SystemExit("continuation campaign identity mismatch")

    with selection_path.open(newline="", encoding="utf-8") as handle:
        selections = {row["cell_id"]: row for row in csv.DictReader(handle)}
    if len(selections) != CELLS:
        raise SystemExit(f"expected {CELLS} selection rows")
    with base_matrix_path.open(newline="", encoding="utf-8") as handle:
        base_rows = list(csv.reader(handle, delimiter="\t"))
    if len(base_rows) != CELLS or any(len(row) != 11 for row in base_rows):
        raise SystemExit("base matrix shape mismatch")

    output.mkdir(parents=True)
    for relative in ("snapshots", "records", "mip", "progress", "logs/freeze", "logs/mip"):
        (output / relative).mkdir(parents=True)
    rows = []
    for expected_index, fields in enumerate(base_rows):
        index, cell, scale, replicate, trips, _old_path, instance_sha, rep, soc, block, _wall = fields
        if int(index) != expected_index or rep != REPRESENTATION or soc != "2.5" or block != "5":
            raise SystemExit(f"base matrix identity mismatch at row {expected_index}")
        selected = selections.get(cell)
        if (
            selected is None
            or selected["instance_file_sha256"] != instance_sha
            or selected["scale"] != scale
        ):
            raise SystemExit(f"selection identity mismatch for {cell}")
        relative = selected["relative_path"]
        if not relative.startswith("data/"):
            raise SystemExit(f"instance is outside data/: {relative}")
        current_instance = required(repo / relative, f"current instance {cell}")
        if sha256(current_instance) != instance_sha:
            raise SystemExit(f"current instance hash mismatch for {cell}")
        stem = f"M__{cell}__{rep}.json"
        rows.append({
            "index": index,
            "cell": cell,
            "scale": scale,
            "replicate": replicate,
            "trips": trips,
            "representation": rep,
            "instance_relative_to_data": relative.removeprefix("data/"),
            "instance_sha256": instance_sha,
            "base_status": str(base / "cg" / "b030_reduced" / stem),
            "resume_status": str(resume / "cg" / stem),
            "snapshot": str(output / "snapshots" / stem),
            "record": str(output / "records" / f"{stem}.freeze.json"),
            "mip_output": str(output / "mip" / f"{stem}.raw_pool_mip8h.json"),
        })
    matrix = output / "matrix.tsv"
    with matrix.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    code_paths = (
        "scripts/event_uniform_envelope/threshold_pool_freeze.sub",
        "scripts/event_uniform_envelope/threshold_pool_mip8h.sub",
        "src/freeze_terminal_exact_cg_pool.py",
        "src/run_exact_pool_mip.py",
    )
    plan = {
        "schema": "evsp-dr-threshold-raw-pool-mip8h-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_commit": args.execution_commit,
        "source_solver_commit": base_plan["solver_commit"],
        "base_root": str(base),
        "resume_root": str(resume),
        "base_plan_sha256": sha256(base_plan_path),
        "base_matrix_sha256": sha256(base_matrix_path),
        "selection_sha256": sha256(selection_path),
        "resume_plan_sha256": sha256(resume_plan_path),
        "resume_matrix_sha256": sha256(resume_matrix_path),
        "matrix_sha256": sha256(matrix),
        "cells": CELLS,
        "source_dependency_job": args.source_dependency_job or None,
        "active_source_index": args.active_source_index,
        "max_concurrent_total": args.max_concurrent_total,
        "source_policy": (
            "newest terminal continuation when present; otherwise certified baseline; "
            "master_failed and wall_limit continuations are accepted only as uncertified RAW pools"
        ),
        "pool_treatment": "RAW; no extra routes and no external initial partition",
        "mip": {
            "backend": "gurobi",
            "partition": "scaglione",
            "threads": 8,
            "memory": args.mip_memory,
            "scientific_time_limit_s_total": 28800,
            "slurm_time_limit": "10:30:00",
            "mip_gap": 0.0001,
            "two_stage": True,
            "time_contract": (
                "fleet stage may use all 28800 seconds; cost stage runs only with the remainder"
            ),
            "requeue": False,
        },
        "code_sha256": {path: sha256(repo / path) for path in code_paths},
    }
    plan_path = output / "execution_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = output / "PREPARATION_COMPLETE"
    marker.write_text(
        f"execution_plan.json {sha256(plan_path)}\nmatrix.tsv {sha256(matrix)}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_root": str(output), "cells": len(rows),
        "plan_sha256": sha256(plan_path), "matrix_sha256": sha256(matrix),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

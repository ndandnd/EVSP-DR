#!/usr/bin/env python3
"""Run selected ladder SEED/exact-CG diagnostics locally with bounded parallelism."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from build_tariff_response_manifest import REPO_ROOT, sha256_file
from launch_scale_ladder import CG_GRIDS
from prepare_scale_ladder_known_partition import prepare
from audit_scale_ladder_known_membership import audit, write_outputs
from scale_ladder_trip_identity import identity
from tariff_response_environment import (
    CG_PORTABLE_FIELDS,
    identity as portable_environment,
)


INSTANCE_MANIFEST = (
    REPO_ROOT
    / "data/scale_ladder/instances/"
    "scale_ladder_instance_manifest_6sel_seed20260803.csv"
)
LOCAL_CODE_PATHS = (
    "src/run_scale_ladder_local_diagnostics.py",
    "src/launch_scale_ladder.py",
    "src/build_tariff_response_manifest.py",
    "src/scale_ladder_trip_identity.py",
    "src/prepare_scale_ladder_known_partition.py",
    "src/audit_scale_ladder_known_membership.py",
    "src/fixed_duty_expanded_optimizer.py",
    "src/tariff_response_core.py",
    "src/rerealize_routes.py",
    "src/run_exact_pool_mip.py",
    "src/exact_pricer_expanded.py",
    "src/exact_initial_pools.py",
    "src/greedy_init.py",
    "src/expanded_path_realization.py",
    "src/audit_giro_known_columns.py",
    "src/master_lp_scipy.py",
    "src/exact_cg_telemetry.py",
    "src/durable_io.py",
    "src/utils_v2.py",
    "src/config.py",
    "src/tariff_response_environment.py",
    "src/matching_init.py",
    "src/pricing_dp_og.py",
    "src/make_giro_seed_routes.py",
    "src/prepare_k40_giro40_partition.py",
)


def _current_environment():
    environment = portable_environment("cg")
    return {
        "schema": environment["schema"],
        "portable_profile": "cg",
        "portable": environment["portable"],
        "portable_identity_sha256":
            environment["portable_identity_sha256"],
        "code_hashes": {
            path: sha256_file(REPO_ROOT / path)
            for path in LOCAL_CODE_PATHS
        },
    }


def _project_cg_environment(environment):
    portable = environment.get("portable") or {}
    projected = {
        field: portable.get(field) for field in CG_PORTABLE_FIELDS
    }
    encoded = json.dumps(
        projected, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "schema": "evsp-dr-portable-environment-v1",
        "portable_profile": "cg",
        "portable": projected,
        "portable_identity_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _write_completion(job, plan_sha):
    output = Path(job["output"])
    paths = [output]
    if job["phase"] == "SEED" and job.get("membership_output"):
        membership = Path(job["membership_output"])
        paths.extend([membership, membership.with_suffix(".csv")])
    elif job["phase"] in {"CG", "CG_SENSITIVITY"}:
        paths.extend([
            Path(str(output) + ".columns.jsonl"),
            Path(str(output) + ".iters.csv"),
        ])
        if job.get("telemetry"):
            paths.append(Path(job["telemetry"]))
        for snapshot in output.parent.glob(
            f"{output.stem}.m*.snapshot.json"
        ):
            paths.extend([
                snapshot, Path(str(snapshot) + ".columns.jsonl"),
            ])
    payload = {
        "schema": "evsp-dr-scale-ladder-worker-completion-v1",
        "phase": job["phase"],
        "plan_sha256": plan_sha,
        "instance_file_sha256":
            job["instance"]["instance_file_sha256"],
        "job_key": job["job_key"],
        "arm": None,
        "snapshot_availability": (
            json.loads(output.read_text()).get("snapshot_availability", {})
            if job["phase"] in {"CG", "CG_SENSITIVITY"} else {}
        ),
        "artifact_sha256": {
            str(path.resolve()): sha256_file(path) for path in paths
        },
    }
    target = Path(str(output) + ".worker-completion.json")
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_seed_and_preflight(cell, root, membership):
    instance = Path(cell["instance"]["path"])
    seed = root / "outputs" / f"seed_{cell['cell_id']}.json"
    prepare(instance, cell["instance"]["instance_file_sha256"], seed)
    preflight = root / "outputs" / f"preflight_{cell['cell_id']}.json"
    write_outputs(
        membership, preflight, preflight.with_suffix(".csv")
    )


def _run_cg(job, budget_s):
    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u",
        str(REPO_ROOT / "src/exact_pricer_expanded.py"),
        "--csv", job["instance"]["relative_path"],
        "--prices_csv", "hourly_prices_flat.csv",
        "--g-kwh", str(job["g_kwh"]),
        "--charge-kw", str(job["charge_kw"]),
        "--min-soc-frac", str(job["min_soc_frac"]),
        "--soc-step", str(job["soc_step"]),
        "--block-min", str(job["block_min"]),
        "--master-sense", job["master_sense"],
        "--initial-pool", job["initial_pool"],
        "--objective", job["objective"],
        "--columns_per_iter", str(job["columns_per_iter"]),
        "--max-iters", str(job["max_iters"]),
        "--diversify-rounds", str(job["diversify_rounds"]),
        "--wall-limit-s", str(budget_s),
        "--checkpoint-every", str(job["checkpoint_every"]), "--resume",
        "--snapshot-at-minutes", ",".join(
            str(value) for value in job["snapshot_minutes"]
            if value * 60 <= budget_s
        ),
        "--out", str(output),
    ]
    if job.get("telemetry"):
        command.extend(["--phase-telemetry", job["telemetry"]])
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"local exact CG failed: {job['job_key']}")


def run(args):
    with INSTANCE_MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected_scales = set(args.scale or [])
    selected_cells = set(args.cell or [])
    instances = []
    for row in rows:
        cell_id = (
            f"k{int(row['scale']):02d}_s{int(row['selection_replicate'])}"
        )
        if selected_cells and cell_id not in selected_cells:
            continue
        if selected_scales and int(row["scale"]) not in selected_scales:
            continue
        if not selected_cells and not selected_scales:
            continue
        path = REPO_ROOT / row["relative_path"]
        instances.append({
            "cell_id": cell_id,
            "scale": int(row["scale"]),
            "selection_replicate": int(row["selection_replicate"]),
            "cg_replicate": 1,
            "campaign_replicate": int(row["selection_replicate"]),
            "target_fleet": int(row["target_fleet"]),
            "instance": {
                **row,
                **identity(path),
                "path": str(path),
                "relative_path": row["relative_path"].removeprefix("data/"),
            },
        })
    if not instances:
        raise ValueError("no local diagnostic cells selected")
    root = Path(args.out_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    (root / "outputs").mkdir(parents=True)
    current_environment = _current_environment()
    reference_environment = None
    if args.reference_plan:
        reference_plan = json.loads(
            Path(args.reference_plan).read_text()
        )
        reference_environment = reference_plan.get(
            "local_diagnostic_environment"
        )
        if reference_environment is None:
            approved = reference_plan.get("python_identity") or {}
            reference_environment = {
                **_project_cg_environment(approved),
                "code_hashes": {
                    path: (reference_plan.get("code_hashes") or {}).get(path)
                    for path in LOCAL_CODE_PATHS
                },
            }
    diagnostic_only = current_environment != reference_environment
    jobs = []
    memberships = {}
    for cell in instances:
        jobs.append(
            {
                **cell, "job_key": f"seed_{cell['cell_id']}",
                "phase": "SEED", "arm": None,
                "soc_step": 15.0, "block_min": 10,
                "output": str(
                    root / "outputs" / f"seed_{cell['cell_id']}.json"
                ),
                "telemetry": None, "progress_dir": None,
                "snapshot_minutes": [],
                "membership_output": str(
                    root / "outputs" / f"preflight_{cell['cell_id']}.json"
                ),
            }
        )
        membership = audit(
            Path(cell["instance"]["path"]),
            cell["instance"]["instance_file_sha256"],
            cell["scale"],
            cell["selection_replicate"],
        )
        memberships[cell["cell_id"]] = membership
        for grid_index, grid in enumerate(CG_GRIDS):
            soc_step, block_min = grid["soc_step"], grid["block_min"]
            primary = grid["grid_role"] == "primary"
            key = (
                f"cg_{cell['cell_id']}" if primary
                else f"cg_{grid['grid_id']}_{cell['cell_id']}"
            )
            jobs.append({
                **cell,
                "job_key": key,
                "phase": "CG" if primary else "CG_SENSITIVITY",
                "arm": None,
                **grid,
                "grid_index": grid_index,
                "diagnostic_only": False,
                "soc_step": soc_step,
                "block_min": block_min,
                "budget_s": args.budget_s,
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "min_soc_frac": 0.0,
                "columns_per_iter": 30,
                "max_iters": 100000,
                "diversify_rounds": 0,
                "initial_pool": "singletons",
                "objective": "combined-cost",
                "master_sense": "partition",
                "checkpoint_every": 25,
                "output": str(root / "outputs" / f"{key}.json"),
                "telemetry": str(root / "telemetry" / f"{key}.jsonl"),
                "progress_dir": None,
                "snapshot_minutes": [
                    value for value in (5, 15, 30, 60, 120)
                    if value * 60 <= args.budget_s
                ],
            })
    plan = {
        "schema": "evsp-dr-scale-ladder-local-diagnostic-v1",
        "execution_mode": "local_diagnostic",
        "diagnostic_only": diagnostic_only,
        "local_diagnostic_environment": current_environment,
        "reference_environment": reference_environment,
        "cg_grids": [dict(grid) for grid in CG_GRIDS],
        "jobs": jobs,
        "k40_reuse_slots": [],
        "trip_identity_schema": "evsp-dr-trip-identity-v1",
        "tariff": {
            "primary_tariff_sha256":
                "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200"
        },
        "physics": {
            "g_kwh": 300.0, "charge_kw": 300.0,
            "reserve_kwh": 0.0, "soc_step_kwh": 15.0, "block_min": 10,
        },
        "task_groups": {
            "SEED": [
                job["job_key"] for job in jobs if job["phase"] == "SEED"
            ],
            "CG": [
                job["job_key"] for job in jobs
                if job["phase"] in {"CG", "CG_SENSITIVITY"}
            ],
        },
        "code_hashes": {},
        "python_identity": current_environment,
        "checkout_identity": {"commit": None},
    }
    plan_raw = json.dumps(
        plan, sort_keys=True, separators=(",", ":")
    ).encode()
    plan_sha = hashlib.sha256(plan_raw).hexdigest()
    (root / "approved-plan.json").write_bytes(plan_raw)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_parallel
    ) as executor:
        list(executor.map(
            lambda cell: _run_seed_and_preflight(
                cell, root, memberships[cell["cell_id"]]
            ),
            instances,
        ))
        cg_jobs = [
            job for job in jobs
            if job["phase"] in {"CG", "CG_SENSITIVITY"}
        ]
        list(executor.map(
            lambda job: _run_cg(job, args.budget_s),
            cg_jobs,
        ))
    for job in jobs:
        _write_completion(job, plan_sha)
    manifest = {
        "approval_sha256": plan_sha,
        "execution_mode": "local_diagnostic",
        "diagnostic_only": diagnostic_only,
        "submitted": False,
        "gate_state": "not_applicable_local",
        "submitted_arrays": {},
    }
    (root / "campaign.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append")
    parser.add_argument("--scale", type=int, action="append")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--budget-s", type=int, default=7200)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--reference-plan", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.max_parallel <= 3:
        parser.error("--max-parallel must be in [1, 3]")
    root = run(args)
    print(json.dumps({
        "output": str(root),
        "diagnostic_only": json.loads(
            (root / "campaign.json").read_text()
        )["diagnostic_only"],
        "max_parallel": args.max_parallel,
        "slurm_used": False,
        "gurobi_used": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

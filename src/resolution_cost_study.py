#!/usr/bin/env python3
"""Freeze and locally execute the exact-CG resolution-cost study."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from audit_giro_known_columns import HORIZON_MIN, STATIONS
from build_tariff_response_manifest import REPO_ROOT, sha256_file
from launch_scale_ladder import CG_BUDGET_S, MIP_BUDGET_S


SCHEMA = "evsp-dr-resolution-cost-study-v2"
SCALES = (2, 3, 5, 8, 13, 20)
GRIDS = (
    (15.0, 10.0, "historical_anchor"),
    (10.0, 10.0, "commensurate"),
    (2.5, 10.0, "commensurate"),
    (10.0, 5.0, "commensurate"),
    (2.5, 5.0, "commensurate"),
    (10.0, 2.5, "commensurate"),
    (2.5, 2.5, "commensurate"),
)
PROFILES = (
    ("p240", 240.0, 240.0, SCALES),
    ("p300_bridge", 300.0, 300.0, (2, 3)),
)
INSTANCE_MANIFEST = (
    REPO_ROOT / "data/scale_ladder/instances/scale_ladder_instance_manifest.csv"
)


def _slug(value):
    return f"{float(value):g}".replace(".", "p")


def _sha256_json(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _git(*arguments):
    result = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, text=True,
        capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _estimated_dag_nodes(trips, g_kwh, soc_step, block_min):
    levels = int(g_kwh / soc_step) + 1
    blocks = int(math.floor(HORIZON_MIN / block_min + 1e-9))
    return 2 + levels * (trips + len(STATIONS) * blocks)


def _resources(scale, soc_step, block_min, estimated_nodes):
    if scale >= 13 and block_min <= 2.5:
        memory_gb, concurrency = 96, 1
    elif scale >= 8 and (block_min <= 2.5 or soc_step <= 2.5):
        memory_gb, concurrency = 64, 1
    elif scale >= 13:
        memory_gb, concurrency = 48, 2
    elif estimated_nodes >= 400_000:
        memory_gb, concurrency = 32, 2
    else:
        memory_gb, concurrency = 16, 4
    return {
        "memory_gb": memory_gb,
        "memory_limit_mb": int(memory_gb * 1024 * 0.9),
        "max_concurrency": concurrency,
        "resource_basis": "resolution_cost_conservative_v1",
    }


def _arc_resources(cg_resources):
    memory_gb = max(32, min(192, cg_resources["memory_gb"] * 2))
    return {
        "memory_gb": memory_gb,
        "memory_limit_mb": int(memory_gb * 1024 * 0.9),
        "max_concurrency": 1,
        "resource_basis": "direct_arcflow_conservative_v1",
    }


def _instances():
    with INSTANCE_MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        scale = int(row["scale"])
        selection = int(row["selection_replicate"])
        if scale not in SCALES or selection not in {1, 2, 3}:
            continue
        path = REPO_ROOT / row["relative_path"]
        if sha256_file(path) != row["instance_file_sha256"]:
            raise ValueError(f"instance hash mismatch: {path}")
        selected.append({
            "cell_id": f"k{scale:02d}_s{selection}",
            "scale": scale,
            "selection_replicate": selection,
            "target_fleet": int(row["target_fleet"]),
            "trip_count": int(row["trip_count"]),
            "relative_path": row["relative_path"].removeprefix("data/"),
            "instance_sha256": row["instance_file_sha256"],
        })
    if len(selected) != 18:
        raise ValueError(f"expected 18 instances, found {len(selected)}")
    return sorted(selected, key=lambda row: (
        row["scale"], row["selection_replicate"],
    ))


def build_plan(artifact_root):
    artifact_root = str(Path(artifact_root))
    jobs = []
    for profile, g_kwh, charge_kw, scales in PROFILES:
        for instance in _instances():
            if instance["scale"] not in scales:
                continue
            for grid_index, (soc_step, block_min, role) in enumerate(GRIDS):
                charge_per_block = charge_kw * block_min / 60.0
                credited = math.floor(
                    charge_per_block / soc_step + 1e-9
                ) * soc_step
                commensurate = math.isclose(
                    charge_per_block, credited, abs_tol=1e-9,
                )
                grid_id = f"soc{_slug(soc_step)}_b{_slug(block_min)}"
                pair_key = f"{profile}_{instance['cell_id']}_{grid_id}"
                key = f"rc_{pair_key}"
                estimated = _estimated_dag_nodes(
                    instance["trip_count"], g_kwh, soc_step, block_min,
                )
                base = {
                    **instance,
                    "paired_cell_key": pair_key,
                    "physics_profile": profile,
                    "g_kwh": g_kwh,
                    "charge_kw": charge_kw,
                    "min_soc_frac": 0.0,
                    "soc_step": soc_step,
                    "block_min": block_min,
                    "grid_id": grid_id,
                    "grid_index": grid_index,
                    "grid_role": role,
                    "charge_kwh_per_block": charge_per_block,
                    "credited_charge_kwh_per_block": credited,
                    "charge_grid_loss_kwh": charge_per_block - credited,
                    "commensurate": commensurate,
                    "max_iters": 1_000_000_000,
                    "wall_limit_s": CG_BUDGET_S[instance["scale"]],
                    "columns_per_iter": 30,
                    "master_sense": "partition",
                    "initial_pool": "singletons",
                    "objective": "combined-cost",
                    "checkpoint_every": 25,
                    "estimated_dag_nodes_upper": estimated,
                }
                cg_resources = _resources(
                    instance["scale"], soc_step, block_min, estimated,
                )
                jobs.append({
                    **base, **cg_resources,
                    "method_arm": "exact_cg",
                    "job_key": key,
                    "integer_wall_limit_s": MIP_BUDGET_S[instance["scale"]],
                    "output": str(
                        Path(artifact_root) / "outputs" / f"{key}.json"
                    ),
                    "integer_output": str(
                        Path(artifact_root) / "outputs"
                        / f"{key}.pool_mip.json"
                    ),
                })
                arc_key = f"af_{pair_key}"
                jobs.append({
                    **base, **_arc_resources(cg_resources),
                    "method_arm": "arc_flow",
                    "job_key": arc_key,
                    "output": str(
                        Path(artifact_root) / "outputs" / f"{arc_key}.json"
                    ),
                })
    if len(jobs) != 336:
        raise ValueError(f"expected 336 arm jobs, found {len(jobs)}")
    manifest_sha = sha256_file(INSTANCE_MANIFEST)
    plan = {
        "schema": SCHEMA,
        "execution_mode": "operator_cluster_or_local_validation",
        "cluster_submission_by_agent": False,
        "code_identity": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_dirty": bool(_git("status", "--porcelain")),
        },
        "instance_manifest": str(INSTANCE_MANIFEST.relative_to(REPO_ROOT)),
        "instance_manifest_sha256": manifest_sha,
        "tariff": {
            "path": "data/hourly_prices_flat.csv",
            "sha256": sha256_file(REPO_ROOT / "data/hourly_prices_flat.csv"),
        },
        "reference_sha256": sha256_file(REPO_ROOT / "data/Ref_dict.csv"),
        "deadhead_sha256": sha256_file(REPO_ROOT / "data/par_ref_dhd.csv"),
        "code_hashes": {
            path: sha256_file(REPO_ROOT / path) for path in (
                "src/exact_pricer_expanded.py",
                "src/master_lp_scipy.py",
                "src/expanded_path_realization.py",
                "src/resolution_cost_study.py",
                "src/resolution_cost_arcflow.py",
                "src/resolution_cost_pool_mip.py",
                "src/arcflow_oracle.py",
                "src/summarize_resolution_cost.py",
            )
        },
        "grids": [
            {"soc_step": soc, "block_min": block, "grid_role": role}
            for soc, block, role in GRIDS
        ],
        "jobs": jobs,
        "scientific_cells": 168,
        "method_arms": ["exact_cg", "arc_flow"],
        "prediction_target": {
            "scale": 40, "trip_count": 947,
            "g_kwh": 240.0, "charge_kw": 240.0,
            "soc_step": 1.0, "block_min": 5.0,
            "affordable_wall_hours": 24.0,
        },
    }
    plan["plan_sha256"] = _sha256_json(plan)
    return plan


def write_plan(path, artifact_root):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    plan = build_plan(artifact_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def _failure_status(job, output, returncode):
    payload = {}
    if output.is_file():
        try:
            payload = json.loads(output.read_text())
        except (OSError, ValueError):
            pass
    memory = returncode in {-9, 137}
    payload.update({
        "job_key": job["job_key"],
        "method_arm": job.get("method_arm", "exact_cg"),
        "stop_reason": "memory" if memory else "process_error",
        "process_returncode": returncode,
        "certified": False,
        "certified_rc_optimal": False,
        "dag_nodes": payload.get("dag_nodes"),
        "dag_arcs": payload.get("dag_arcs"),
        "dag_build_wall_s": payload.get("dag_build_wall_s"),
        "estimated_dag_nodes_upper": job["estimated_dag_nodes_upper"],
        "peak_rss_mb": payload.get("peak_rss_mb"),
        "pool_columns_final": payload.get("columns", 0),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_job(job, budget_override_s=None):
    output = Path(job["output"])
    for key, expected in (
        ("instance_sha256", job["instance_sha256"]),
    ):
        if key == "instance_sha256" and sha256_file(
            REPO_ROOT / "data" / job["relative_path"]
        ) != expected:
            raise ValueError(f"{job['job_key']} input changed")
    budget = float(budget_override_s or job["wall_limit_s"])
    if job["method_arm"] == "exact_cg":
        command = [
            sys.executable, "-u",
            str(REPO_ROOT / "src/exact_pricer_expanded.py"),
            "--csv", job["relative_path"],
            "--prices_csv", "hourly_prices_flat.csv",
            "--g-kwh", str(job["g_kwh"]),
            "--charge-kw", str(job["charge_kw"]),
            "--min-soc-frac", str(job["min_soc_frac"]),
            "--soc-step", str(job["soc_step"]),
            "--block-min", str(job["block_min"]),
            "--master-sense", job["master_sense"],
            "--initial-pool", job["initial_pool"],
            "--columns_per_iter", str(job["columns_per_iter"]),
            "--max-iters", str(job["max_iters"]),
            "--wall-limit-s", str(int(math.ceil(budget))),
            "--memory-limit-mb", str(job["memory_limit_mb"]),
            "--checkpoint-every", str(job["checkpoint_every"]),
            "--out", str(output),
        ]
        if output.exists():
            command.append("--resume")
    else:
        command = [
            sys.executable, "-u",
            str(REPO_ROOT / "src/resolution_cost_arcflow.py"),
            "--csv", job["relative_path"],
            "--prices-csv", "hourly_prices_flat.csv",
            "--g-kwh", str(job["g_kwh"]),
            "--charge-kw", str(job["charge_kw"]),
            "--reserve-kwh", str(job["min_soc_frac"] * job["g_kwh"]),
            "--soc-step", str(job["soc_step"]),
            "--block-min", str(job["block_min"]),
            "--time-limit-s", str(int(math.ceil(budget))),
            "--memory-limit-mb", str(job["memory_limit_mb"]),
            "--out", str(output),
        ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode:
        _failure_status(job, output, completed.returncode)
        return completed.returncode
    if job["method_arm"] == "exact_cg":
        integer_output = Path(job["integer_output"])
        integer_budget = float(
            budget_override_s or job["integer_wall_limit_s"]
        )
        integer_command = [
            sys.executable, "-u",
            str(REPO_ROOT / "src/resolution_cost_pool_mip.py"),
            "--cg-status", str(output),
            "--time-limit-s", str(int(math.ceil(integer_budget))),
            "--out", str(integer_output),
        ]
        integer = subprocess.run(
            integer_command, cwd=REPO_ROOT, check=False,
        )
        if integer.returncode:
            _failure_status(job, integer_output, integer.returncode)
        return integer.returncode
    return 0


def run_selected(
    plan, scales, profiles, selections, arms, max_parallel, budget,
):
    jobs = [
        job for job in plan["jobs"]
        if (not scales or job["scale"] in scales)
        and (not profiles or job["physics_profile"] in profiles)
        and (not selections or job["selection_replicate"] in selections)
        and (not arms or job["method_arm"] in arms)
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_parallel,
    ) as executor:
        return list(executor.map(
            lambda job: run_job(job, budget), jobs,
        ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("plan")
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--artifact-root", type=Path, required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--scale", type=int, action="append")
    execute.add_argument("--profile", action="append")
    execute.add_argument("--selection", type=int, action="append")
    execute.add_argument("--arm", choices=("exact_cg", "arc_flow"),
                         action="append")
    execute.add_argument("--max-parallel", type=int, default=1)
    execute.add_argument("--budget-override-s", type=float)
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = write_plan(args.out, args.artifact_root)
        print(json.dumps({
            "plan": str(args.out), "jobs": len(plan["jobs"]),
            "scientific_cells": plan["scientific_cells"],
            "plan_sha256": plan["plan_sha256"],
        }, sort_keys=True))
        return 0
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != SCHEMA:
        raise ValueError("unexpected study plan schema")
    identity = {key: value for key, value in plan.items()
                if key != "plan_sha256"}
    if _sha256_json(identity) != plan.get("plan_sha256"):
        raise ValueError("study plan hash mismatch")
    if sha256_file(INSTANCE_MANIFEST) != plan["instance_manifest_sha256"]:
        raise ValueError("instance manifest changed after planning")
    changed = [
        path for path, expected in plan["code_hashes"].items()
        if sha256_file(REPO_ROOT / path) != expected
    ]
    if changed:
        raise ValueError(f"study code changed after planning: {changed}")
    if not 1 <= args.max_parallel <= 4:
        parser.error("--max-parallel must be in [1,4]")
    codes = run_selected(
        plan, set(args.scale or []), set(args.profile or []),
        set(args.selection or []), set(args.arm or []), args.max_parallel,
        args.budget_override_s,
    )
    print(json.dumps({
        "jobs": len(codes), "nonzero_returncodes": sum(bool(code) for code in codes),
        "cluster_submitted": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

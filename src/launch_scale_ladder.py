#!/usr/bin/env python3
"""Hash-approved launcher for the flat-tariff exact-CG scale ladder."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from build_tariff_response_manifest import REPO_ROOT, sha256_file
from scale_ladder_trip_identity import SCHEMA as TRIP_SCHEMA, identity


SCHEMA = "evsp-dr-scale-ladder-plan-v1"
REVIEWED_BASE = "77baf667a06946c692f959d66fed4e2bca36cd32"
INPUT_MANIFEST = (
    REPO_ROOT / "data/scale_ladder/instances/campaign_input_manifest.json"
)
MEMBERSHIP_PREFLIGHT = (
    REPO_ROOT / "data/scale_ladder/known_membership_preflight.json"
)
MEMBERSHIP_PREFLIGHT_SHA256 = (
    "5124534373e8d3aff981c55891b8f7ed321fdf1efe96c8bbfd093d957c1b94c8"
)
INSTANCE_MANIFEST = (
    REPO_ROOT
    / "data/scale_ladder/instances/scale_ladder_instance_manifest.csv"
)
HISTORICAL_FLAT_SHA256 = (
    "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200"
)
WORKER = REPO_ROOT / "src/submit_scale_ladder.sub"
PROBE_WORKER = REPO_ROOT / "src/submit_scale_ladder_probe.sub"
ACTIVATION_WORKER = REPO_ROOT / "src/submit_scale_ladder_activation.sub"
CODE_PATHS = (
    "src/build_tariff_response_manifest.py",
    "src/launch_scale_ladder.py",
    "src/submit_scale_ladder.sub",
    "src/build_scale_ladder_inputs.py",
    "src/scale_ladder_trip_identity.py",
    "src/prepare_scale_ladder_known_partition.py",
    "src/exact_pricer_expanded.py",
    "src/exact_initial_pools.py",
    "src/greedy_init.py",
    "src/run_exact_pool_mip.py",
    "src/expanded_path_realization.py",
    "src/audit_giro_known_columns.py",
    "src/tariff_response_core.py",
    "src/fixed_duty_expanded_optimizer.py",
    "src/rerealize_routes.py",
    "src/mip_convergence.py",
    "src/exact_cg_telemetry.py",
    "src/master_lp_scipy.py",
    "src/durable_io.py",
    "src/utils_v2.py",
    "src/config.py",
    "src/tariff_response_environment.py",
    "src/reconcile_scale_ladder_gate.py",
    "src/recover_scale_ladder_mip_progress.py",
    "src/audit_scale_ladder_known_membership.py",
    "src/run_scale_ladder_local_diagnostics.py",
    "src/run_scale_ladder_environment_probe.py",
    "src/submit_scale_ladder_probe.sub",
    "src/submit_scale_ladder_activation.sub",
    "src/run_reviewed_python.py",
    "src/matching_init.py",
    "src/pricing_dp_og.py",
    "src/make_giro_seed_routes.py",
    "src/prepare_k40_giro40_partition.py",
)
CG_BUDGET_S = {
    2: 7200, 3: 7200, 5: 7200,
    8: 21600, 13: 21600, 20: 43200,
    30: 86400, 40: 86400,
}
CG_GRIDS = (
    {"grid_id": "soc15_b10", "soc_step": 15.0, "block_min": 10,
     "grid_role": "primary"},
    {"grid_id": "soc5_b10", "soc_step": 5.0, "block_min": 10,
     "grid_role": "resolution"},
    {"grid_id": "soc2p5_b10", "soc_step": 2.5, "block_min": 10,
     "grid_role": "resolution"},
    {"grid_id": "soc1_b10", "soc_step": 1.0, "block_min": 10,
     "grid_role": "resolution"},
    {"grid_id": "soc1_b5", "soc_step": 1.0, "block_min": 5,
     "grid_role": "resolution"},
)
MIP_BUDGET_S = {
    2: 1800, 3: 1800, 5: 1800, 8: 1800,
    13: 3600, 20: 7200, 30: 14400,
}
SNAPSHOT_MINUTES = (5, 15, 30, 60, 120, 240, 480, 720, 1440)
PROBE_PARTITIONS = ("default_partition", "scaglione")
AMBIGUOUS_DISCOVERY_ATTEMPTS = 6
AMBIGUOUS_DISCOVERY_DELAY_S = 1.0
PROBE_TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
    "SPECIAL_EXIT",
}
PROBE_RETRYABLE_STATES = PROBE_TERMINAL_STATES - {"COMPLETED"}
PROBE_WAITING_RESOLUTIONS = {
    "live", "awaiting_accounting", "controller_query_error",
    "accounting_query_error", "awaiting_artifact", "observer_deadline",
}
PROBE_SLURM_QUERY_TIMEOUT_S = 10.0
SBATCH_TIMEOUT_S = 30.0
BOUND_RELEASE_COMMAND_ATTEMPTS = 3
BOUND_RELEASE_VERIFY_ATTEMPTS = 5
BOUND_RELEASE_VERIFY_DELAY_S = 1.0


def canonical(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def checkout_identity(require_detached=False):
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True,
            capture_output=True, check=False,
        )
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    ancestor = git("merge-base", "--is-ancestor", REVIEWED_BASE, "HEAD")
    symbolic = git("symbolic-ref", "-q", "HEAD")
    if (
        head.returncode != 0
        or status.returncode != 0
        or status.stdout.strip()
        or ancestor.returncode != 0
        or (require_detached and symbolic.returncode == 0)
    ):
        raise ValueError("checkout must be reviewed, exact, clean and detached")
    return {
        "commit": head.stdout.strip(),
        "reviewed_base": REVIEWED_BASE,
        "detached": symbolic.returncode == 1,
        "tracked_clean": True,
    }


def _environment(python):
    python = Path(python).expanduser().absolute()
    if python.is_symlink() or not python.is_file():
        raise ValueError("Python must be a real absolute executable, not a symlink")
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    completed = subprocess.run(
        [
            str(python), "-I", "-B",
            str(REPO_ROOT / "src/tariff_response_environment.py"),
        ],
        text=True, capture_output=True, check=False, env=environment,
    )
    if completed.returncode != 0:
        raise ValueError("Python/Gurobi environment unavailable")
    payload = json.loads(completed.stdout)
    if not payload["portable"]["python"].startswith("3.12."):
        raise ValueError("scale ladder requires Python 3.12")
    return payload


def _name(job, nonce):
    scale = int(job["scale"])
    replicate = int(job["campaign_replicate"])
    code = (
        "P" if job["phase"] == "PREFLIGHT"
        else "S" if job["phase"] == "SEED"
        else "X" if job["phase"] == "CG"
        else "D" if job["phase"] == "CG_SENSITIVITY"
        else "R" if job["arm"] == "RAW"
        else "K"
    )
    grid = ""
    if job["phase"] == "CG_SENSITIVITY":
        grid = (
            "1B5" if float(job.get("soc_step")) == 1.0
            and int(job.get("block_min")) == 5
            else {
                5.0: "5", 2.5: "25", 1.0: "1",
            }[float(job.get("soc_step", 15.0))]
        )
    name = f"L{scale:02d}R{replicate}{code}{grid}{nonce}"
    if len(name) > 15:
        raise ValueError("Slurm job name too long")
    return name


def _resources(phase, scale, soc_step, block_min):
    if phase == "MIP":
        return 48, 16, "scaglione_mip"
    if phase not in {"CG", "CG_SENSITIVITY"}:
        return 16, 16, "small_non_cg"
    if scale >= 30 and soc_step <= 1.0 and block_min <= 5:
        return 128, 1, "unbenchmarked_fine_large_conservative"
    if (scale >= 30 and soc_step <= 2.5) or (
        scale >= 20 and soc_step <= 1.0
    ):
        return 64, 2, "unbenchmarked_fine_conservative"
    if scale >= 20:
        return 24, 8, "measured_large_scale_policy"
    return 16, 16, "measured_small_scale_policy"


def build_plan(campaign, python, reservation_root):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,79}", campaign):
        raise ValueError("unsafe campaign name")
    input_raw = INPUT_MANIFEST.read_bytes()
    input_manifest = json.loads(input_raw)
    if sha256_file(MEMBERSHIP_PREFLIGHT) != MEMBERSHIP_PREFLIGHT_SHA256:
        raise ValueError("membership preflight package hash mismatch")
    preflight_package = json.loads(MEMBERSHIP_PREFLIGHT.read_text())
    if (
        preflight_package.get("schema")
        != "evsp-dr-scale-ladder-membership-preflight-v1"
        or preflight_package.get("instance_manifest_sha256")
        != sha256_file(INSTANCE_MANIFEST)
    ):
        raise ValueError("membership preflight package identity mismatch")
    prelaunch_membership = {
        (int(cell["scale"]), int(cell["selection_replicate"])): cell
        for cell in preflight_package["cells"]
    }
    if input_manifest.get("schema") != (
        "evsp-dr-scale-ladder-input-manifest-v1"
    ):
        raise ValueError("input manifest schema mismatch")
    if input_manifest.get("trip_identity_schema") != TRIP_SCHEMA:
        raise ValueError("trip identity schema mismatch")
    tariff = input_manifest["tariff"]
    if (
        tariff["primary_tariff_sha256"] != HISTORICAL_FLAT_SHA256
        or tariff["equivalence_verified"] is not True
        or Path(tariff["primary_tariff_relative_path"]).name
        != "hourly_prices_flat.csv"
    ):
        raise ValueError("historical flat tariff identity changed")
    with INSTANCE_MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 22:
        raise ValueError("scale ladder instance row count differs")
    instances = {}
    for row in rows:
        path = REPO_ROOT / row["relative_path"]
        observed = identity(path)
        for field in (
            "instance_file_sha256",
            "ordered_trip_id_set_sha256",
            "solver_local_trip_index_sha256",
            "ordered_trip_sequence_sha256",
            "trip_identity_schema",
        ):
            if str(observed[field]) != str(row[field]):
                raise ValueError(f"instance identity mismatch: {field}")
        key = (int(row["scale"]), int(row["selection_replicate"]))
        if key in instances:
            raise ValueError("duplicate instance selection")
        instances[key] = {
            **row,
            "path": str(path.resolve()),
            "trip_count": int(row["trip_count"]),
            "scale": int(row["scale"]),
            "selection_replicate": int(row["selection_replicate"]),
            "cg_replicates": int(row["cg_replicates"]),
            "target_fleet": int(row["target_fleet"]),
            "duties": json.loads(row["duties_json"]),
        }
    root = (
        REPO_ROOT / "src/results/scale_ladder" / campaign
    ).resolve()
    cg_cells = []
    for key in sorted(instances):
        instance = instances[key]
        for cg_replicate in range(1, instance["cg_replicates"] + 1):
            cg_cells.append({
                "cell_id": (
                    f"k{instance['scale']:02d}_s"
                    f"{instance['selection_replicate']}_c{cg_replicate}"
                ),
                "scale": instance["scale"],
                "selection_replicate": instance["selection_replicate"],
                "cg_replicate": cg_replicate,
                "campaign_replicate": (
                    cg_replicate
                    if instance["scale"] == 40
                    else instance["selection_replicate"]
                ),
                "target_fleet": instance["target_fleet"],
                "instance": instance,
            })
    non_k40 = [cell for cell in cg_cells if cell["scale"] < 40]
    k40 = [cell for cell in cg_cells if cell["scale"] == 40]
    if len(non_k40) != 21 or len(k40) != 2:
        raise ValueError("CG ladder cell count differs")
    cg_cells = non_k40 + k40
    jobs = []
    nonce = hashlib.sha256(campaign.encode()).hexdigest()[:2]
    preflight_key_by_cell = {}
    seed_key_by_cell = {}
    cg_key_by_cell = {}
    for instance_key in sorted(instances):
        instance = instances[instance_key]
        cell = next(
            value for value in cg_cells
            if value["scale"] == instance["scale"]
            and value["selection_replicate"]
            == instance["selection_replicate"]
        )
        key = (
            f"preflight_k{instance['scale']:02d}_"
            f"s{instance['selection_replicate']}"
        )
        preflight_key_by_cell[(
            instance["scale"], instance["selection_replicate"]
        )] = key
        job = _job(root, cell, key, "PREFLIGHT", None, nonce)
        job["prelaunch_membership_sha256"] = hashlib.sha256(canonical(
            prelaunch_membership[instance_key]
        )).hexdigest()
        jobs.append(job)
    for cell in non_k40:
        key = f"seed_{cell['cell_id']}"
        seed_key_by_cell[cell["cell_id"]] = key
        jobs.append(_job(root, cell, key, "SEED", None, nonce))
    for cell in cg_cells:
        for grid_index, grid in enumerate(CG_GRIDS):
            primary = grid["grid_role"] == "primary"
            key = (
                f"cg_{cell['cell_id']}" if primary
                else f"cg_{grid['grid_id']}_{cell['cell_id']}"
            )
            job = _job(
                root, cell, key,
                "CG" if primary else "CG_SENSITIVITY",
                None, nonce,
                diagnostic_soc_step=grid["soc_step"],
                diagnostic_block_min=grid["block_min"],
            )
            job.update({
                **grid,
                "grid_index": grid_index,
                "diagnostic_only": False,
            })
            job["dependency_preflight"] = preflight_key_by_cell[(
                cell["scale"], cell["selection_replicate"]
            )]
            jobs.append(job)
            if primary:
                cg_key_by_cell[cell["cell_id"]] = key
    for arm in ("RAW", "KNOWN-PARTITION"):
        for cell in non_k40:
            key = f"mip_{arm.lower().replace('-', '_')}_{cell['cell_id']}"
            job = _job(root, cell, key, "MIP", arm, nonce)
            job["dependency_cg"] = cg_key_by_cell[cell["cell_id"]]
            job["dependency_seed"] = (
                seed_key_by_cell[cell["cell_id"]]
                if arm == "KNOWN-PARTITION" else None
            )
            jobs.append(job)
    code_hashes = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in CODE_PATHS
    }
    for job in jobs:
        staged_instance = (
            root / "input/instances"
            / f"{job['cell_id']}_{Path(job['instance']['path']).name}"
        )
        job["instance"] = {
            **job["instance"],
            "source_path": job["instance"]["path"],
            "path": str(staged_instance),
            "relative_path": (
                f"scale_ladder_inputs/{job['cell_id']}_"
                f"{Path(job['instance']['path']).name}"
            ),
        }
        job["execution_digest"] = hashlib.sha256(canonical({
            "cell_id": job["cell_id"],
            "phase": job["phase"],
            "arm": job["arm"],
            "scale": job["scale"],
            "selection_replicate": job["selection_replicate"],
            "cg_replicate": job["cg_replicate"],
            "budget_s": job["budget_s"],
            "snapshot_minutes": job["snapshot_minutes"],
            "soc_step": job["soc_step"],
            "block_min": job["block_min"],
            "instance_identity": {
                field: job["instance"][field] for field in (
                    "instance_file_sha256",
                    "ordered_trip_id_set_sha256",
                    "solver_local_trip_index_sha256",
                    "ordered_trip_sequence_sha256",
                    "trip_identity_schema",
                    "duty_set_sha256",
                )
            },
            "code_hashes": code_hashes,
            "tariff_sha256": HISTORICAL_FLAT_SHA256,
        })).hexdigest()
    groups = {
        "PREFLIGHT": [
            job["job_key"] for job in jobs
            if job["phase"] == "PREFLIGHT"
        ],
        "SEED": [job["job_key"] for job in jobs if job["phase"] == "SEED"],
        "CG": [job["job_key"] for job in jobs if job["phase"] == "CG"],
        "CG_SENSITIVITY": [
            job["job_key"] for job in jobs
            if job["phase"] == "CG_SENSITIVITY"
        ],
        "MIP_RAW": [
            job["job_key"] for job in jobs
            if job["phase"] == "MIP" and job["arm"] == "RAW"
        ],
        "MIP_KNOWN": [
            job["job_key"] for job in jobs
            if job["phase"] == "MIP" and job["arm"] == "KNOWN-PARTITION"
        ],
    }
    if {key: len(value) for key, value in groups.items()} != {
        "PREFLIGHT": 22, "SEED": 21, "CG": 23,
        "CG_SENSITIVITY": 92, "MIP_RAW": 21, "MIP_KNOWN": 21,
    }:
        raise ValueError("task group counts differ")
    reuse_slots = [
        {
            "cell_id": cell["cell_id"],
            "scale": 40,
            "cg_replicate": cell["cg_replicate"],
            "arm": arm,
            "submission_policy": "reuse_only_never_submit",
            "required_instance_file_sha256":
                cell["instance"]["instance_file_sha256"],
            "required_tariff_sha256": HISTORICAL_FLAT_SHA256,
            "required_physics": input_manifest["physics"],
            "required_cg_job_key": cg_key_by_cell[cell["cell_id"]],
            "accepted_producer_commits": [
                "636dc0912f47e6ce85284fad3b36af30b4135887",
                "77baf667a06946c692f959d66fed4e2bca36cd32",
            ],
            "required_time_limit_s": (
                28800 if arm == "RAW" else 7200
            ),
            "required_mip_gap": 1e-4,
            "required_threads": 8,
            "required_gurobi_seed": 0,
            "required_known_partition_sha256": (
                "8f9944f93f26cf0121e9ecab2fa412d573e90a0189b7a38008d3b2535f54d428"
                if arm == "KNOWN-PARTITION" else None
            ),
            "required_reference_sha256":
                "7bda0e1f439dc8bf5081499566eb2c6a0314190ef27294707f1403fd2c13e3a0",
            "required_deadhead_sha256":
                "5993e922c671f053611635578b32a1be13bab87b3b5fd8c02b699b81fe0eb66c",
        }
        for cell in k40
        for arm in ("RAW", "KNOWN-PARTITION")
    ]
    python_identity = _environment(python)
    def binary_identity(name):
        value = shutil.which(name)
        if not value:
            return {"path": None, "sha256": None, "available": False}
        path = Path(value).resolve()
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "available": True,
        }
    scontrol_identity = binary_identity("scontrol")
    sbatch_identity = binary_identity("sbatch")
    sacct_identity = binary_identity("sacct")
    squeue_identity = binary_identity("squeue")
    return {
        "schema": SCHEMA,
        "campaign": campaign,
        "campaign_root": str(root),
        "checkout_identity": checkout_identity(False),
        "input_manifest": str(INPUT_MANIFEST),
        "input_manifest_sha256": hashlib.sha256(input_raw).hexdigest(),
        "membership_preflight": str(MEMBERSHIP_PREFLIGHT),
        "membership_preflight_sha256": MEMBERSHIP_PREFLIGHT_SHA256,
        "prelaunch_membership": {
            f"k{scale:02d}_s{replicate}": value
            for (scale, replicate), value in prelaunch_membership.items()
        },
        "instance_manifest": str(INSTANCE_MANIFEST),
        "instance_manifest_sha256": sha256_file(INSTANCE_MANIFEST),
        "trip_identity_schema": TRIP_SCHEMA,
        "tariff": tariff,
        "physics": input_manifest["physics"],
        "code_hashes": code_hashes,
        "worker": str(WORKER),
        "worker_sha256": sha256_file(WORKER),
        "probe_worker": str(PROBE_WORKER),
        "probe_worker_sha256": sha256_file(PROBE_WORKER),
        "activation_worker": str(ACTIVATION_WORKER),
        "activation_worker_sha256": sha256_file(ACTIVATION_WORKER),
        "submission_protocol": "probe_first_activation_v1",
        "python_identity": python_identity,
        "python": {
            "path": python_identity["portable"]["executable"],
            "sha256": python_identity["portable"]["executable_sha256"],
        },
        "scontrol": scontrol_identity,
        "sbatch": sbatch_identity,
        "sacct": sacct_identity,
        "squeue": squeue_identity,
        "runtime_environment": {
            "HOME": str(Path.home()),
            "USER": os.environ.get("USER", ""),
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        },
        "reservation_root": str(Path(reservation_root).resolve()),
        "jobs": jobs,
        "task_groups": groups,
        "cg_grids": [dict(grid) for grid in CG_GRIDS],
        "resource_policy": {
            "schema": "evsp-dr-scale-ladder-resource-policy-v1",
            "fine_grid_large_scale_status":
                "unbenchmarked_conservative_allocation",
            "memory_gb_field": "memory_gb",
            "concurrency_field": "max_concurrency",
        },
        "task_count": sum(map(len, groups.values())),
        "preflight_task_count": 22,
        "cg_task_count": 115,
        "primary_cg_task_count": 23,
        "sensitivity_cg_task_count": 92,
        "mip_task_count": 42,
        "seed_task_count": 21,
        "k40_mip_submission_count": 0,
        "infrastructure_probe_task_count": 2,
        "infrastructure_activation_task_count": 1,
        "infrastructure_task_count": 3,
        "k40_reuse_slots": reuse_slots,
    }


def _job(
    root, cell, key, phase, arm, nonce, *,
    diagnostic_soc_step=None, diagnostic_block_min=None,
):
    scale = cell["scale"]
    budget = (
        4 * 3600 if phase in {"SEED", "PREFLIGHT"}
        else CG_BUDGET_S[scale]
        if phase in {"CG", "CG_SENSITIVITY"}
        else MIP_BUDGET_S[scale]
    )
    marks = [
        value for value in SNAPSHOT_MINUTES
        if value * 60 <= budget
    ] if phase in {"CG", "CG_SENSITIVITY"} else []
    soc_step = (
        float(diagnostic_soc_step)
        if diagnostic_soc_step is not None else 15.0
    )
    block_min = (
        int(diagnostic_block_min)
        if diagnostic_block_min is not None else 10
    )
    memory_gb, max_concurrency, resource_basis = _resources(
        phase, scale, soc_step, block_min
    )
    job = {
        **cell,
        "job_key": key,
        "phase": phase,
        "arm": arm,
        "scientific_role": (
            "feasibility_integral_assembly_diagnostic_not_algorithmic_recovery"
            if arm == "KNOWN-PARTITION" else None
        ),
        "budget_s": budget,
        "soc_step": soc_step,
        "block_min": block_min,
        "memory_gb": memory_gb,
        "max_concurrency": max_concurrency,
        "resource_basis": resource_basis,
        "snapshot_minutes": marks,
        "partition": (
            "scaglione" if phase == "MIP" else "default_partition"
        ),
        "threads": 8 if phase == "MIP" else 2,
        "job_name": None,
        "output": str(root / "outputs" / f"{key}.json"),
        "progress_dir": (
            str(root / "progress" / key) if phase == "MIP" else None
        ),
        "telemetry": (
            str(root / "telemetry" / f"{key}.jsonl")
            if phase in {"CG", "CG_SENSITIVITY"} and scale <= 20
            else None
        ),
    }
    job["job_name"] = _name(job, nonce)
    return job


def write_plan(plan, plan_path, matrix_path):
    plan_raw = canonical(plan)
    plan_sha = hashlib.sha256(plan_raw).hexdigest()
    _write_new(plan_path, plan_raw)
    with matrix_path.open("x", newline="") as handle:
        fields = (
            "job_key", "phase", "scale", "selection_replicate",
            "cg_replicate", "arm", "grid_id", "grid_role", "grid_index",
            "soc_step", "block_min", "memory_gb", "max_concurrency",
            "resource_basis", "budget_s", "partition", "threads", "job_name",
            "output",
        )
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(plan["jobs"])
    return plan_sha


def _write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _campaign_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".submission.lock"
    with lock_path.open("a+b") as handle:
        # The top-level submitter and its activation controller can overlap
        # briefly.  Serialize them instead of making the valid controller fail
        # merely because the submitter is still publishing its final state.
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _discover_bound_job(plan, spec, *, attempts=1, delay_s=0.0):
    if not isinstance(attempts, int) or attempts < 1 or delay_s < 0:
        raise ValueError("bound-job discovery policy is invalid")
    user = str((plan.get("runtime_environment") or {}).get("USER") or "")
    if not user:
        raise ValueError("approved runtime user is missing")
    for attempt_index in range(attempts):
        listed = _run_probe_slurm_query(
            subprocess.run,
            [
                plan["squeue"]["path"], "-h", "-u", user,
                "-o", "%i|%j|%T|%P|%R|%k",
            ],
        )
        if listed.returncode != 0:
            raise RuntimeError("cannot query squeue for accepted job")
        same_comment = [
            {
                "job_id": fields[0], "job_name": fields[1],
                "state": _normalized_slurm_state(fields[2]),
                "partition": fields[3], "reason": fields[4],
                "comment": fields[5],
            }
            for line in listed.stdout.splitlines()
            if len(fields := [value.strip() for value in line.split("|", 5)])
            == 6
            and fields[5] == spec["comment"]
        ]
        if any(
            not row["job_id"].isdigit()
            or row["partition"] != spec["partition"]
            or row["job_name"] != spec["job_name"]
            or row["state"] != "PENDING"
            or row["reason"] != "JobHeldUser"
            for row in same_comment
        ):
            raise ValueError(
                "bound Slurm comment has a mismatched fingerprint"
            )
        matches = {row["job_id"] for row in same_comment}
        if len(matches) > 1:
            raise ValueError("multiple Slurm jobs share one bound comment")
        if matches:
            return next(iter(matches))
        if attempt_index + 1 < attempts:
            time.sleep(delay_s)
    return None


def _recover_probe_job_id(plan, plan_sha, partition, spec):
    output = Path(spec["output"])
    if output.is_file():
        payload = json.loads(output.read_text())
        recovered = str(payload.get("slurm_job_id") or "")
        if (
            recovered.isdigit()
            and payload.get("plan_sha256") == plan_sha
            and payload.get("probe_id") == spec["probe_id"]
            and payload.get("probe_attempt") == spec["attempt"]
            and payload.get("slurm_partition") == partition
        ):
            return recovered
        raise ValueError("unrecorded probe artifact identity mismatch")
    ambiguous = (
        spec.get("submission_intent") == "accepted_or_ambiguous"
        or spec.get("submission_outcome_ambiguous") is True
    )
    recovered = _discover_bound_job(
        plan, spec,
        attempts=(AMBIGUOUS_DISCOVERY_ATTEMPTS if ambiguous else 1),
        delay_s=(AMBIGUOUS_DISCOVERY_DELAY_S if ambiguous else 0.0),
    )
    if recovered is None and ambiguous:
        raise RuntimeError(
            "prior probe submission remains ambiguous; exact job not yet "
            "visible, so replacement submission is refused"
        )
    return recovered


def _activation_lineage_id(plan, plan_sha):
    return hashlib.sha256(canonical({
        "schema": "evsp-dr-scale-ladder-activation-lineage-v1",
        "plan_sha256": plan_sha,
        "campaign_root": str(Path(plan["campaign_root"]).resolve()),
        "submission_protocol": "probe_first_activation_v1",
    })).hexdigest()


def _ensure_reservations(plan, plan_sha, activation_lineage_id):
    if not re.fullmatch(r"[0-9a-f]{64}", str(activation_lineage_id or "")):
        raise ValueError("stable activation lineage is required")
    root = Path(plan["reservation_root"])
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for job in plan["jobs"]:
        path = root / f"{job['execution_digest']}.json"
        payload = canonical({
            "schema": "evsp-dr-scale-ladder-reservation-v3",
            "submission_protocol": "probe_first_activation_v1",
            "plan_sha256": plan_sha,
            "activation_lineage_id": activation_lineage_id,
            "job_key": job["job_key"],
            "execution_digest": job["execution_digest"],
        })
        if path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError(
                    f"reservation belongs to another execution: {path}"
                )
        else:
            _write_new(path, payload)
        paths.append(path)
    return paths


def _validate_submission_contract(plan):
    if (
        plan["scontrol"]["available"] is not True
        or plan["sbatch"]["available"] is not True
        or plan["sacct"]["available"] is not True
        or plan["squeue"]["available"] is not True
        or sha256_file(Path(plan["scontrol"]["path"]))
        != plan["scontrol"]["sha256"]
        or sha256_file(Path(plan["sbatch"]["path"]))
        != plan["sbatch"]["sha256"]
        or sha256_file(Path(plan["sacct"]["path"]))
        != plan["sacct"]["sha256"]
        or sha256_file(Path(plan["squeue"]["path"]))
        != plan["squeue"]["sha256"]
    ):
        raise ValueError("approved Slurm executables unavailable/changed")
    observed = checkout_identity(True)
    if observed != plan["checkout_identity"]:
        raise ValueError("submission checkout differs")


def _validate_recorded_probe_spec(plan_sha, root, partition, spec):
    attempt = spec.get("attempt")
    if not isinstance(attempt, int) or attempt < 1:
        raise ValueError("recorded probe attempt is invalid")
    expected = _probe_spec(plan_sha, partition, root, attempt)
    if any(
        spec.get(field) != expected[field]
        for field in (
            "output", "probe_id", "partition", "attempt", "comment",
            "job_name",
        )
    ):
        raise ValueError("recorded probe specification mismatch")
    return expected


def _probe_result_identity_errors(partition, spec, result):
    expected = {
        "job_id": str(spec.get("job_id") or ""),
        "output": str(spec.get("output") or ""),
        "probe_id": spec.get("probe_id"),
        "partition": partition,
        "attempt": spec.get("attempt"),
        "comment": spec.get("comment"),
        "job_name": spec.get("job_name"),
    }
    errors = []
    for field, expected_value in expected.items():
        observed = result.get(field)
        if field == "job_id":
            observed = str(observed or "")
        if observed != expected_value:
            errors.append({
                "field": field,
                "expected": expected_value,
                "observed": observed,
            })
    return errors


def _hard_probe_retry_reason(result):
    resolution = str(result.get("state_resolution") or "")
    if resolution in {
        "environment_mismatch", "artifact_identity_mismatch",
        "identity_mismatch",
    }:
        return resolution
    if result.get("identity_errors") or result.get("artifact_identity_errors"):
        return "identity_mismatch"
    if result.get("differences"):
        return "environment_mismatch"
    return None


def _validate_terminal_activation(plan, activation):
    if not isinstance(activation, dict):
        raise RuntimeError(
            "infrastructure retry refused: no prior controller is recorded"
        )
    if not str(activation.get("job_id") or "").isdigit():
        raise RuntimeError(
            "infrastructure retry refused: prior controller has no exact job ID"
        )
    observation = _resolve_bound_job(plan, activation)
    if (
        observation.get("live") is True
        or observation.get("state") not in PROBE_TERMINAL_STATES
    ):
        raise RuntimeError(
            "infrastructure retry refused: recorded controller is live or "
            "not proven terminal"
        )
    return observation


def _authorize_infrastructure_retry(
    plan, plan_sha, root, manifest, manifest_path, *, kind,
):
    """Atomically authorize one new pre-science infrastructure attempt.

    The authorization is durable before any replacement ``sbatch`` call.  A
    subsequent exact submission can therefore finish an interrupted retry
    without incrementing attempts again.
    """
    pending = manifest.get("infrastructure_retry")
    if pending is not None:
        if (
            isinstance(pending, dict)
            and pending.get("state") in {
                "dispatched", "terminal_retry_required",
            }
        ):
            if pending.get("state") == "dispatched":
                # One explicit retry invocation is enough after a dispatched
                # replacement controller fails.  Exact live/unknown state
                # raises here and never increments the attempt.
                terminal_observation = _validate_terminal_activation(
                    plan, manifest.get("activation")
                )
                pending = {
                    **pending,
                    "state": "terminal_retry_required",
                    "terminal_observation": terminal_observation,
                }
            manifest.setdefault("infrastructure_retry_history", []).append(
                dict(pending)
            )
            manifest.pop("infrastructure_retry", None)
            pending = None
            _replace_json(manifest_path, manifest)
        elif (
            isinstance(pending, dict)
            and pending.get("kind") == kind
            and pending.get("state") in {
                "authorized", "probe_ids_recorded", "activation_held",
                "dispatched",
            }
        ):
            return manifest
        else:
            raise RuntimeError(
                "another infrastructure retry is already recorded"
            )
    activation = manifest.get("activation")
    activation_observation = _validate_terminal_activation(plan, activation)
    probe_specs = dict(manifest.get("infrastructure_probes") or {})
    if set(probe_specs) != set(PROBE_PARTITIONS):
        raise ValueError("both recorded probe specifications are required")
    for partition, spec in probe_specs.items():
        _validate_recorded_probe_spec(plan_sha, root, partition, spec)
        if not str(spec.get("job_id") or "").isdigit():
            raise ValueError("recorded probe job ID is invalid")

    recorded_results = manifest.get("probe_results")
    recorded_results_valid = bool(
        isinstance(recorded_results, dict)
        and set(recorded_results) == set(PROBE_PARTITIONS)
        and not any(
            _probe_result_identity_errors(
                partition, probe_specs[partition],
                recorded_results[partition],
            )
            for partition in PROBE_PARTITIONS
        )
    )
    # Do not trust an old controller summary alone.  Retry authorization makes
    # one bounded, live-first observation of the exact jobs and artifacts.
    results = _wait_for_probes(
        plan, plan_sha, probe_specs, timeout_s=2
    )
    if set(results) != set(PROBE_PARTITIONS):
        raise RuntimeError("probe re-observation is incomplete")
    if recorded_results is not None and not recorded_results_valid:
        manifest.setdefault("probe_result_history", []).append({
            "reason": "superseded_by_exact_retry_observation",
            "value": recorded_results,
        })
    manifest["probe_results"] = results
    _replace_json(manifest_path, manifest)
    replacements = []
    scheduler_observations = {}
    for partition in PROBE_PARTITIONS:
        spec = probe_specs[partition]
        result = results[partition]
        identity_errors = _probe_result_identity_errors(
            partition, spec, result
        )
        if identity_errors:
            raise ValueError(
                f"recorded probe result identity mismatch on {partition}: "
                f"{identity_errors}"
            )
        hard_reason = _hard_probe_retry_reason(result)
        if hard_reason:
            raise ValueError(
                f"non-retryable probe mismatch on {partition}: {hard_reason}"
            )
        observation = _resolve_bound_job(plan, spec)
        scheduler_observations[partition] = observation
        if observation.get("live") is True:
            raise RuntimeError(
                f"probe retry refused: {partition} is still live"
            )
        if result.get("compatible") is True:
            if (
                observation.get("state") != "COMPLETED"
                or result.get("state") != "COMPLETED"
            ):
                raise ValueError(
                    f"compatible probe {partition} lacks a completed outcome"
                )
            continue
        publication_incomplete = bool(
            result.get("state") == "COMPLETED"
            and observation.get("state") == "COMPLETED"
            and result.get("state_resolution") == "awaiting_artifact"
            and result.get("artifact_status")
            in {"missing", "awaiting_sidecar"}
            and result.get("observer_deadline_reached") is True
        )
        if publication_incomplete:
            replacements.append(partition)
            continue
        if _probe_result_waiting(result):
            raise RuntimeError(
                f"probe retry refused: {partition} is live or unresolved"
            )
        if (
            result.get("state_resolution") != "scheduler_failure"
            or result.get("state") not in PROBE_RETRYABLE_STATES
            or observation.get("state") not in PROBE_RETRYABLE_STATES
        ):
            raise ValueError(
                f"probe retry refused: {partition} is not a proven "
                "scheduler-terminal failure"
            )
        replacements.append(partition)

    if kind == "failed_probes" and not replacements:
        raise RuntimeError("probe retry refused: no retryable probe failed")
    if kind == "failed_activation" and replacements:
        raise RuntimeError(
            "activation-only retry refused: failed probes require "
            "--retry-failed-probes"
        )
    if kind == "failed_activation" and not _probes_compatible(results):
        raise RuntimeError(
            "activation-only retry refused: probes are not both compatible"
        )

    probe_history = manifest.setdefault("probe_attempt_history", {})
    for partition in replacements:
        previous = dict(probe_specs[partition])
        probe_history.setdefault(partition, []).append({
            "spec": previous,
            "result": (
                recorded_results[partition]
                if recorded_results_valid else results[partition]
            ),
            "retry_observation": results[partition],
            "scheduler_observation": scheduler_observations[partition],
        })
        probe_specs[partition] = _probe_spec(
            plan_sha, partition, root,
            attempt=int(previous["attempt"]) + 1,
        )

    activation_history = manifest.setdefault(
        "activation_attempt_history", []
    )
    activation_history.append({
        "spec": dict(activation),
        "observation": activation_observation,
        "probe_results": (
            recorded_results if recorded_results_valid else results
        ),
        "retry_probe_observation": results,
    })
    target_attempt = int(activation["attempt"]) + 1
    manifest["infrastructure_probes"] = probe_specs
    # The replacement controller cannot be fully bound until every replacement
    # probe ID is durable.  The retry record is the atomic authorization to
    # construct exactly this attempt after those held submissions complete.
    manifest["activation"] = None
    manifest["infrastructure_retry"] = {
        "schema": "evsp-dr-scale-ladder-infrastructure-retry-v1",
        "kind": kind,
        "state": "authorized",
        "target_activation_attempt": target_attempt,
        "source_activation_job_id": str(activation["job_id"]),
        "replaced_probe_partitions": replacements,
    }
    manifest["probe_state"] = "retry_authorized"
    manifest["submission_state"] = "infrastructure_retry_authorized"
    _replace_json(manifest_path, manifest)
    return manifest


def _stage_scientific_inputs(plan):
    root = Path(plan["campaign_root"])
    for job in plan["jobs"]:
        source = Path(job["instance"]["source_path"])
        target = Path(job["instance"]["path"])
        _copy_new(source, target, job["instance"]["instance_file_sha256"])
    _copy_new(
        REPO_ROOT / plan["tariff"]["primary_tariff_relative_path"],
        root / "input/tariff/hourly_prices_flat.csv",
        HISTORICAL_FLAT_SHA256,
    )
    _copy_new(
        Path(plan["input_manifest"]),
        root / "input/manifests/campaign_input_manifest.json",
        plan["input_manifest_sha256"],
    )
    _copy_new(
        Path(plan["instance_manifest"]),
        root / "input/manifests/scale_ladder_instance_manifest.csv",
        plan["instance_manifest_sha256"],
    )
    _copy_new(
        Path(plan["membership_preflight"]),
        root / "input/manifests/known_membership_preflight.json",
        plan["membership_preflight_sha256"],
    )
    return root / "input"


def submit(
    plan, plan_sha, *, retry_failed_activation=False,
    retry_failed_probes=False,
):
    """Submit only infrastructure probes and their held controller.

    Scientific reservations, the gate, and all six scientific arrays are
    created later by the activation controller, and only after it validates
    both probe artifacts independently.
    """
    if retry_failed_activation and retry_failed_probes:
        raise ValueError("choose exactly one infrastructure retry mode")
    _validate_submission_contract(plan)
    if plan.get("submission_protocol") != "probe_first_activation_v1":
        raise ValueError("unsupported scale-ladder submission protocol")
    root = Path(plan["campaign_root"])
    with _campaign_lock(root):
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        plan_path = root / "approved-plan.json"
        manifest_path = root / "campaign.json"
        plan_raw = canonical(plan)
        if plan_path.exists():
            if (
                plan_path.read_bytes() != plan_raw
                or hashlib.sha256(plan_raw).hexdigest() != plan_sha
                or not manifest_path.is_file()
            ):
                raise ValueError("existing campaign differs from approved plan")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("approval_sha256") != plan_sha:
                raise ValueError("existing campaign approval mismatch")
        else:
            if manifest_path.exists():
                raise ValueError("campaign manifest exists without approved plan")
            _write_new(plan_path, plan_raw)
            manifest = {
                **plan,
                "approval_sha256": plan_sha,
                "submitted": False,
                "submission_state": "probe_submitting",
                "probe_state": "submitting",
                "reservation_state": "not_created",
                "gate_state": "not_created",
                "gate_job_id": None,
                "submitted_arrays": {},
                "reservations": [],
            }
            _replace_json(manifest_path, manifest)
        if manifest.get("submitted") is True:
            return manifest
        if (
            manifest.get("gate_state") != "not_created"
            or manifest.get("reservations")
            or manifest.get("submitted_arrays")
            or manifest.get("gate_job_id") is not None
        ):
            raise ValueError(
                "scientific submission already began; use gate reconciliation"
            )

        requested_retry_kind = (
            "failed_probes" if retry_failed_probes
            else "failed_activation" if retry_failed_activation
            else None
        )
        pending_retry = manifest.get("infrastructure_retry")
        if requested_retry_kind is not None:
            manifest = _authorize_infrastructure_retry(
                plan, plan_sha, root, manifest, manifest_path,
                kind=requested_retry_kind,
            )
            pending_retry = manifest.get("infrastructure_retry")
        elif pending_retry is not None:
            raise RuntimeError(
                "an explicitly authorized infrastructure retry is pending; "
                "resume it with the same reviewed retry flag"
            )

        probe_specs = dict(manifest.get("infrastructure_probes") or {})
        try:
            for partition in PROBE_PARTITIONS:
                spec = probe_specs.get(partition)
                if spec is None:
                    if pending_retry is not None:
                        raise ValueError(
                            "retry authorization lacks a probe specification"
                        )
                    spec = _probe_spec(
                        plan_sha, partition, root, attempt=1
                    )
                    probe_specs[partition] = spec
                _validate_recorded_probe_spec(
                    plan_sha, root, partition, spec
                )
                manifest["infrastructure_probes"] = dict(probe_specs)
                manifest["probe_state"] = (
                    "retry_submitting" if pending_retry is not None
                    else "submitting"
                )
                _replace_json(manifest_path, manifest)
                if not str(spec.get("job_id") or "").isdigit():
                    recovered = _recover_probe_job_id(
                        plan, plan_sha, partition, spec
                    )
                    if recovered is None:
                        # Publish intent before sbatch.  If this process dies
                        # after scheduler acceptance but before the returned ID
                        # is durable, a later invocation may only boundedly
                        # rediscover this exact fingerprint; it may not submit
                        # a duplicate.
                        spec["submission_intent"] = "accepted_or_ambiguous"
                        manifest["infrastructure_probes"] = dict(probe_specs)
                        _replace_json(manifest_path, manifest)
                        try:
                            recovered = _submit_probe(
                                plan, plan_path, plan_sha, spec, logs,
                                held=True,
                            )
                        except RuntimeError:
                            spec["submission_outcome_ambiguous"] = True
                            manifest["infrastructure_probes"] = dict(
                                probe_specs
                            )
                            _replace_json(manifest_path, manifest)
                            raise
                    spec["job_id"] = recovered
                    spec.pop("submission_intent", None)
                    spec.pop("submission_outcome_ambiguous", None)
                    manifest["infrastructure_probes"] = dict(probe_specs)
                    _replace_json(manifest_path, manifest)
        except Exception as exc:
            manifest["probe_state"] = "submission_failed"
            manifest["probe_error"] = repr(exc)
            manifest["submission_state"] = "probe_submission_failed"
            _replace_json(manifest_path, manifest)
            raise
        manifest["probe_state"] = (
            "retry_held" if pending_retry is not None else "held"
        )
        manifest["submission_state"] = "activation_submitting"
        if pending_retry is not None:
            pending_retry["state"] = "probe_ids_recorded"
            manifest["infrastructure_retry"] = pending_retry
        _replace_json(manifest_path, manifest)
        activation = manifest.get("activation")
        if activation is None:
            attempt = (
                pending_retry["target_activation_attempt"]
                if pending_retry is not None else 1
            )
            activation = _activation_spec(
                plan_sha, root, probe_specs, attempt=attempt
            )
        elif not isinstance(activation.get("attempt"), int):
            raise ValueError("recorded activation attempt is invalid")
        expected_activation = _activation_spec(
            plan_sha, root, probe_specs, attempt=activation["attempt"],
            dependency_job_ids=activation.get("dependency_job_ids"),
        )
        if activation is not None and any(
            activation.get(field) != expected_activation[field]
            for field in (
                "attempt", "partition", "comment", "job_name",
                "probe_job_ids", "dependency_job_ids", "campaign_root",
            )
        ):
            raise ValueError("recorded activation specification mismatch")
        manifest["activation"] = activation
        if pending_retry is not None:
            pending_retry["state"] = "probe_ids_recorded"
            manifest["infrastructure_retry"] = pending_retry
        _replace_json(manifest_path, manifest)
        if str(activation.get("job_id") or "").isdigit():
            observation = _resolve_bound_job(plan, activation)
            terminal = observation.get("state") in PROBE_TERMINAL_STATES
            if terminal:
                activation["state"] = "terminal_retry_required"
                activation["terminal_observation"] = observation
                manifest["activation"] = activation
                if pending_retry is not None:
                    pending_retry["state"] = "terminal_retry_required"
                    pending_retry["terminal_observation"] = observation
                    manifest["infrastructure_retry"] = pending_retry
                manifest["submission_state"] = (
                    "activation_terminal_retry_required"
                )
                _replace_json(manifest_path, manifest)
                raise RuntimeError(
                    "activation controller is terminal before campaign "
                    "completion; inspect its logs, then rerun the exact "
                    "approved command with --retry-failed-activation"
                )
        if not str(activation.get("job_id") or "").isdigit():
            activation_ambiguous = (
                activation.get("submission_intent")
                == "accepted_or_ambiguous"
                or activation.get("submission_outcome_ambiguous") is True
            )
            recovered = _discover_bound_job(
                plan, activation,
                attempts=(
                    AMBIGUOUS_DISCOVERY_ATTEMPTS
                    if activation_ambiguous else 1
                ),
                delay_s=(
                    AMBIGUOUS_DISCOVERY_DELAY_S
                    if activation_ambiguous else 0.0
                ),
            )
            if recovered is None and activation_ambiguous:
                raise RuntimeError(
                    "prior activation submission remains ambiguous; exact "
                    "job not yet visible, so replacement submission is refused"
                )
            if recovered is None:
                activation["submission_intent"] = (
                    "accepted_or_ambiguous"
                )
                manifest["activation"] = activation
                _replace_json(manifest_path, manifest)
                try:
                    recovered = _submit_activation(
                        plan, plan_path, plan_sha, activation, logs
                    )
                except RuntimeError:
                    activation["submission_outcome_ambiguous"] = True
                    manifest["activation"] = activation
                    _replace_json(manifest_path, manifest)
                    raise
            activation["job_id"] = recovered
            activation.pop("submission_intent", None)
            activation.pop("submission_outcome_ambiguous", None)
            manifest["activation"] = activation
            manifest["submission_state"] = "activation_held"
            if pending_retry is not None:
                pending_retry["state"] = "activation_held"
                manifest["infrastructure_retry"] = pending_retry
            _replace_json(manifest_path, manifest)
        # Only release infrastructure work after both probe IDs and the
        # controller ID are durable.  This gives every accepted-before-record
        # crash window a unique held job that the exact same submission can
        # discover and resume.
        # Clear every cached claim in one durable write before revalidating
        # any of them.  If the process stops or one verification fails, the
        # manifest cannot retain a stale ``released=true`` for work that was
        # not observed during this invocation.
        for spec in probe_specs.values():
            spec["released"] = False
            spec.pop("release_verification", None)
        activation["released"] = False
        activation.pop("release_verification", None)
        manifest["infrastructure_probes"] = dict(probe_specs)
        manifest["activation"] = activation
        _replace_json(manifest_path, manifest)
        for partition in PROBE_PARTITIONS:
            spec = probe_specs[partition]
            # Revalidate scheduler state on every restart.  A Boolean written
            # by a prior process cannot prove that Slurm cleared the hold.
            released = _release_held_probe(plan, spec)
            if released.returncode != 0:
                manifest["submission_state"] = "probe_release_failed"
                manifest["probe_release_error"] = (
                    released.stderr or released.stdout or ""
                ).strip()
                manifest["infrastructure_probes"] = dict(probe_specs)
                _replace_json(manifest_path, manifest)
                raise RuntimeError(
                    f"probe release failed on {partition}"
                )
            spec["released"] = True
            manifest["infrastructure_probes"] = dict(probe_specs)
            _replace_json(manifest_path, manifest)
        manifest["probe_state"] = "submitted"
        manifest["submission_state"] = "activation_held"
        _replace_json(manifest_path, manifest)
        released = _release_held_activation(plan, activation)
        if released.returncode != 0:
            manifest["submission_state"] = "activation_release_failed"
            manifest["activation_release_error"] = (
                released.stderr or released.stdout or ""
            ).strip()
            _replace_json(manifest_path, manifest)
            raise RuntimeError("activation-controller release failed")
        activation["released"] = True
        manifest["activation"] = activation
        manifest["submission_state"] = "activation_released"
        _replace_json(manifest_path, manifest)
        if pending_retry is not None:
            pending_retry["state"] = "dispatched"
            pending_retry["activation_job_id"] = str(activation["job_id"])
            pending_retry["probe_job_ids"] = dict(
                activation["probe_job_ids"]
            )
            manifest["infrastructure_retry"] = pending_retry
            _replace_json(manifest_path, manifest)
        return manifest


def _probe_id(partition):
    if partition == "default_partition":
        return "default"
    if partition == "scaglione":
        return "scaglione"
    raise ValueError(f"unsupported probe partition: {partition}")


def _probe_spec(plan_sha, partition, root, attempt):
    if not isinstance(attempt, int) or attempt < 1:
        raise ValueError("probe attempt must be a positive integer")
    probe_id = _probe_id(partition)
    return {
        "job_id": None,
        "output": str(
            Path(root).resolve() / "probes"
            / f"{partition}.attempt{attempt}.json"
        ),
        "probe_id": probe_id,
        "partition": partition,
        "attempt": attempt,
        "comment": (
            f"SLADP:{plan_sha[:20]}:{probe_id}:{attempt}"
        ),
        "job_name": f"LDP{probe_id[:2].upper()}{attempt}{plan_sha[:3]}",
    }


def _activation_spec(
    plan_sha, root, probe_specs, attempt, *, dependency_job_ids=None,
):
    if not isinstance(attempt, int) or attempt < 1:
        raise ValueError("activation attempt must be a positive integer")
    probe_job_ids = {
        partition: str(probe_specs[partition].get("job_id") or "")
        for partition in PROBE_PARTITIONS
    }
    if any(not value.isdigit() for value in probe_job_ids.values()):
        raise ValueError("both recorded probe job IDs are required")
    if len(set(probe_job_ids.values())) != 2:
        raise ValueError("probe jobs must be distinct")
    # Always retain both exact probe jobs in the activation barrier.  Cached
    # ``released`` flags are not scheduler evidence, and ``afterany`` remains
    # valid when a probe is already terminal.
    derived_dependencies = [
        probe_job_ids[partition] for partition in PROBE_PARTITIONS
    ]
    if dependency_job_ids is None:
        dependency_job_ids = derived_dependencies
    if not isinstance(dependency_job_ids, list):
        raise ValueError("activation dependency job IDs are invalid")
    dependency_job_ids = [str(value) for value in dependency_job_ids]
    if dependency_job_ids != derived_dependencies:
        raise ValueError(
            "activation dependencies must equal both ordered probe job IDs"
        )
    return {
        "job_id": None,
        "attempt": attempt,
        "partition": "default_partition",
        "comment": f"SLADA:{plan_sha[:20]}:{attempt}",
        "job_name": f"LDA{attempt}{plan_sha[:5]}",
        "probe_job_ids": probe_job_ids,
        "dependency_job_ids": dependency_job_ids,
        "campaign_root": str(Path(root).resolve()),
    }


def _submit_activation(plan, plan_path, plan_sha, spec, logs):
    expected = _activation_spec(
        plan_sha,
        Path(plan["campaign_root"]),
        {
            partition: {
                "job_id": job_id,
                "released": (
                    job_id not in spec.get("dependency_job_ids", [])
                ),
            }
            for partition, job_id in spec["probe_job_ids"].items()
        },
        spec.get("attempt"),
        dependency_job_ids=spec.get("dependency_job_ids"),
    )
    for field in (
        "attempt", "partition", "comment", "job_name", "probe_job_ids",
        "dependency_job_ids", "campaign_root",
    ):
        if spec.get(field) != expected[field]:
            raise ValueError("activation specification identity mismatch")
    arguments = [
        "--hold", "--partition=default_partition", "--requeue",
        "--time=00:45:00",
        "--cpus-per-task=1", "--mem=4G",
        f"--job-name={spec['job_name']}",
        f"--comment={spec['comment']}",
        f"--output={logs}/activation_a{spec['attempt']}_%j.out",
        f"--error={logs}/activation_a{spec['attempt']}_%j.err",
        "--export=NONE",
        str(ACTIVATION_WORKER), str(plan_path), plan_sha,
        plan["python"]["path"], plan["python"]["sha256"],
        str(REPO_ROOT), plan["runtime_environment"]["HOME"],
        plan["activation_worker_sha256"],
    ]
    if spec["dependency_job_ids"]:
        dependency = ":".join(spec["dependency_job_ids"])
        arguments.insert(5, f"--dependency=afterany:{dependency}")
    return _sbatch(plan, arguments)


def _bound_job_fingerprint_errors(spec, row):
    return [
        {
            "field": field,
            "expected": str(spec.get(field) or ""),
            "observed": str(row.get(field) or ""),
        }
        for field in ("job_id", "job_name", "partition", "comment")
        if str(row.get(field) or "") != str(spec.get(field) or "")
    ]


def _resolve_bound_job(plan, spec, *, runner=None):
    """Resolve one exact infrastructure job, preferring live state."""
    runner = subprocess.run if runner is None else runner
    job_id = str(spec.get("job_id") or "")
    if not job_id.isdigit():
        raise ValueError("bound job ID is invalid")
    user = str((plan.get("runtime_environment") or {}).get("USER") or "")
    if not user:
        raise ValueError("approved runtime user is missing")

    listed = _run_probe_slurm_query(
        runner,
        [
            plan["squeue"]["path"], "-h", "-u", user,
            "-o", "%i|%j|%T|%P|%R|%k",
        ],
    )
    if listed.returncode != 0:
        raise RuntimeError("cannot query live infrastructure job state")
    live_rows = []
    for line in listed.stdout.splitlines():
        fields = [value.strip() for value in line.split("|", 5)]
        if len(fields) == 6 and fields[0] == job_id:
            live_rows.append({
                "job_id": fields[0], "job_name": fields[1],
                "state": _normalized_slurm_state(fields[2]),
                "partition": fields[3], "reason": fields[4],
                "comment": fields[5],
            })
    if len(live_rows) > 1:
        raise ValueError("multiple live rows match bound job ID")
    if live_rows:
        errors = _bound_job_fingerprint_errors(spec, live_rows[0])
        if errors:
            raise ValueError(f"live bound-job fingerprint mismatch: {errors}")
        if live_rows[0]["state"] not in PROBE_TERMINAL_STATES:
            return {**live_rows[0], "source": "squeue", "live": True}
        # A terminal-looking squeue row has no exit code.  Keep resolving via
        # scontrol/sacct rather than treating it as a successful certificate.

    shown = _run_probe_slurm_query(
        runner,
        [plan["scontrol"]["path"], "show", "job", job_id, "-o"],
    )
    if shown.returncode == 0 and shown.stdout.strip():
        values = {}
        for token in shown.stdout.split():
            if "=" in token:
                key, value = token.split("=", 1)
                values[key] = value
        controller_row = {
            "job_id": values.get("JobId", ""),
            "job_name": values.get("JobName", ""),
            "state": _normalized_slurm_state(values.get("JobState")),
            "partition": values.get("Partition", ""),
            "reason": values.get("Reason", ""),
            "comment": values.get("Comment", ""),
            "exit_code": values.get("ExitCode"),
        }
        errors = _bound_job_fingerprint_errors(spec, controller_row)
        if errors:
            raise ValueError(
                f"controller bound-job fingerprint mismatch: {errors}"
            )
        if controller_row["state"] in PROBE_TERMINAL_STATES:
            if not re.fullmatch(
                r"[0-9]+:[0-9]+", str(controller_row.get("exit_code") or "")
            ):
                raise ValueError(
                    "terminal bound job lacks an exact exit code"
                )
            if (
                controller_row["state"] == "COMPLETED"
                and controller_row["exit_code"] != "0:0"
            ):
                raise ValueError("completed bound job has nonzero exit code")
            return {
                **controller_row, "source": "scontrol", "live": False,
            }
        return {**controller_row, "source": "scontrol", "live": True}
    absent_message = (shown.stderr or shown.stdout).lower()
    if shown.returncode != 0 and "invalid job id" not in absent_message:
        raise RuntimeError("cannot prove infrastructure job absent")

    accounted = _run_probe_slurm_query(
        runner,
        [
            plan["sacct"]["path"], "-X", "-n", "-P", "-j", job_id,
            "--format=JobIDRaw,JobName%64,State,Partition%64,"
            "Comment%256,ExitCode",
        ],
    )
    if accounted.returncode != 0:
        raise RuntimeError("cannot query infrastructure job accounting")
    rows = []
    for line in accounted.stdout.splitlines():
        fields = [value.strip() for value in line.split("|", 5)]
        if len(fields) == 6 and fields[0] == job_id:
            rows.append({
                "job_id": fields[0], "job_name": fields[1],
                "state": _normalized_slurm_state(fields[2]),
                "partition": fields[3], "comment": fields[4],
                "exit_code": fields[5],
            })
    if len(rows) != 1:
        raise ValueError("bound-job accounting has no unique exact row")
    errors = _bound_job_fingerprint_errors(spec, rows[0])
    if errors:
        raise ValueError(f"accounting bound-job mismatch: {errors}")
    if rows[0]["state"] == "COMPLETED" and rows[0]["exit_code"] != "0:0":
        raise ValueError("completed bound job has nonzero exit code")
    if (
        rows[0]["state"] in PROBE_TERMINAL_STATES
        and not re.fullmatch(r"[0-9]+:[0-9]+", rows[0]["exit_code"])
    ):
        raise ValueError("terminal bound job lacks an exact exit code")
    return {**rows[0], "source": "sacct", "live": False}


def _bound_release_observation_result(observation, label):
    """Classify an exact bound-job observation after/before release.

    ``None`` means the same job is authoritatively still user-held and a
    bounded release retry is permitted.  Every other outcome is terminal for
    the release operation and is returned as a ``CompletedProcess``.
    """
    state = observation["state"]
    if state in PROBE_TERMINAL_STATES:
        if label == "activation" and state != "COMPLETED":
            return subprocess.CompletedProcess(
                args=[], returncode=3, stdout="",
                stderr=f"activation controller ended in {state}",
            )
        return subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps(observation, sort_keys=True), stderr="",
        )
    if observation.get("live") is not True:
        return subprocess.CompletedProcess(
            args=[], returncode=3, stdout="",
            stderr=f"{label} has no authoritative live or terminal state",
        )
    if state in {"RUNNING", "CONFIGURING", "COMPLETING"}:
        return subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps(observation, sort_keys=True), stderr="",
        )
    if state == "PENDING":
        reason = str(observation.get("reason") or "")
        if not reason:
            return subprocess.CompletedProcess(
                args=[], returncode=3, stdout="",
                stderr=f"{label} pending reason is missing",
            )
        if reason == "JobHeldUser":
            return None
        if reason.startswith("JobHeld"):
            return subprocess.CompletedProcess(
                args=[], returncode=3, stdout="",
                stderr=f"{label} remains scheduler-held: {reason}",
            )
        if reason == "DependencyNeverSatisfied" or (
            label == "probe" and reason == "Dependency"
        ):
            return subprocess.CompletedProcess(
                args=[], returncode=3, stdout="",
                stderr=(
                    f"{label} has an invalid post-release reason: {reason}"
                ),
            )
        return subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps(observation, sort_keys=True), stderr="",
        )
    return subprocess.CompletedProcess(
        args=[], returncode=3, stdout="",
        stderr=f"{label} is not proven held or released",
    )


def _record_release_verification(
    spec, observation, label, command_attempts, result,
):
    if result.returncode == 0:
        spec["release_verification"] = {
            "verified": True,
            "role": label,
            "job_id": str(spec["job_id"]),
            "command_attempts": command_attempts,
            "observation": observation,
        }
    return result


def _release_bound_job(plan, spec, label, *, sleeper=time.sleep):
    spec.pop("release_verification", None)
    try:
        observation = _resolve_bound_job(plan, spec)
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            args=[], returncode=3, stdout="",
            stderr=str(exc),
        )
    classified = _bound_release_observation_result(observation, label)
    if classified is not None:
        return _record_release_verification(
            spec, observation, label, 0, classified
        )

    release_diagnostics = []
    for release_attempt in range(1, BOUND_RELEASE_COMMAND_ATTEMPTS + 1):
        try:
            released = _run_probe_slurm_query(
                subprocess.run,
                [
                    plan["scontrol"]["path"], "release",
                    str(spec["job_id"]),
                ],
            )
            release_diagnostics.append({
                "attempt": release_attempt,
                "returncode": released.returncode,
                "stdout": (released.stdout or "").strip(),
                "stderr": (released.stderr or "").strip(),
            })
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            release_diagnostics.append({
                "attempt": release_attempt,
                "query_error": str(exc),
            })

        held_after_command = False
        verified_observation = False
        for verify_attempt in range(1, BOUND_RELEASE_VERIFY_ATTEMPTS + 1):
            sleeper(BOUND_RELEASE_VERIFY_DELAY_S)
            try:
                observation = _resolve_bound_job(plan, spec)
            except ValueError as exc:
                return subprocess.CompletedProcess(
                    args=[], returncode=3, stdout="", stderr=str(exc)
                )
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                release_diagnostics.append({
                    "attempt": release_attempt,
                    "verification": verify_attempt,
                    "query_error": str(exc),
                })
                continue

            verified_observation = True
            classified = _bound_release_observation_result(
                observation, label
            )
            if classified is not None:
                return _record_release_verification(
                    spec, observation, label, release_attempt, classified
                )
            held_after_command = True
            continue

        if not verified_observation:
            break
        if not held_after_command:
            break

    return subprocess.CompletedProcess(
        args=[], returncode=3, stdout="",
        stderr=(
            f"{label} release postcondition was not observed: "
            + json.dumps(release_diagnostics, sort_keys=True)
        ),
    )


def _release_held_activation(plan, spec):
    return _release_bound_job(plan, spec, "activation")


def _release_held_probe(plan, spec):
    return _release_bound_job(plan, spec, "probe")


def _submit_probe(
    plan, plan_path, plan_sha, spec, logs, *, held=False,
):
    partition = spec["partition"]
    probe_id = _probe_id(partition)
    expected = _probe_spec(
        plan_sha, partition, Path(plan["campaign_root"]),
        spec.get("attempt"),
    )
    if (
        any(spec.get(key) != expected[key] for key in (
            "output", "probe_id", "partition", "attempt", "comment",
            "job_name",
        ))
    ):
        raise ValueError("probe specification identity mismatch")
    arguments = [
        f"--partition={partition}", "--no-requeue",
        "--time=00:10:00", "--cpus-per-task=1", "--mem=4G",
        f"--job-name={spec['job_name']}",
        f"--comment={spec['comment']}",
        f"--output={logs}/probe_{probe_id}_a{spec['attempt']}_%j.out",
        f"--error={logs}/probe_{probe_id}_a{spec['attempt']}_%j.err",
        "--export=NONE",
        str(PROBE_WORKER), str(plan_path), plan_sha, probe_id,
        str(spec["attempt"]),
        plan["python"]["path"], plan["python"]["sha256"],
        str(REPO_ROOT), plan["runtime_environment"]["HOME"], spec["output"],
        plan["probe_worker_sha256"],
    ]
    if held:
        arguments.insert(0, "--hold")
    return _sbatch(plan, arguments)


def _normalized_slurm_state(value):
    words = str(value or "").strip().split()
    return words[0].split("+", 1)[0].upper() if words else ""


def _probe_fingerprint_errors(spec, row):
    errors = []
    for field in ("job_name", "comment", "partition"):
        expected = str(spec.get(field) or "")
        observed = str(row.get(field) or "")
        if not expected or observed != expected:
            errors.append({
                "field": field,
                "expected": expected,
                "observed": observed,
            })
    return errors


def _parse_live_probe_rows(payload, wanted_ids):
    rows = {}
    for line in payload.splitlines():
        fields = [field.strip() for field in line.split("|", 4)]
        if len(fields) != 5 or fields[0] not in wanted_ids:
            continue
        rows.setdefault(fields[0], []).append({
            "job_id": fields[0],
            "job_name": fields[1],
            "state": _normalized_slurm_state(fields[2]),
            "partition": fields[3],
            "comment": fields[4],
        })
    return rows


def _parse_accounting_probe_rows(payload, wanted_ids):
    rows = {}
    for line in payload.splitlines():
        fields = [field.strip() for field in line.split("|", 6)]
        if len(fields) < 6 or fields[0] not in wanted_ids:
            continue
        rows.setdefault(fields[0], []).append({
            "job_id": fields[0],
            "job_name": fields[1],
            "state": _normalized_slurm_state(fields[2]),
            "partition": fields[3],
            "comment": fields[4],
            "exit_code": fields[5],
        })
    return rows


def _run_probe_slurm_query(
    runner, command, *, observer_deadline=None,
):
    timeout = PROBE_SLURM_QUERY_TIMEOUT_S
    if observer_deadline is not None:
        remaining = observer_deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, 0)
        timeout = min(timeout, remaining)
    return runner(
        command,
        text=True, capture_output=True, check=False, timeout=timeout,
    )


def _probe_job_states(
    plan, probe_specs, *, runner=None, observer_deadline=None,
):
    """Resolve probe state with the live controller taking precedence.

    ``sacct`` can lag, and on the Unicorn deployment it has returned a stale
    terminal row while ``squeue`` still held the same job ID as PENDING.  A
    live controller record is therefore authoritative.  Accounting is used
    only after the exact job ID is absent from the live queue.
    """
    runner = subprocess.run if runner is None else runner
    observations = {}
    recorded = {
        partition: str(spec.get("job_id") or "")
        for partition, spec in probe_specs.items()
    }
    wanted_ids = {value for value in recorded.values() if value.isdigit()}
    for partition, job_id in recorded.items():
        if not job_id.isdigit():
            observations[partition] = {
                "state": "UNRECORDED", "source": "manifest",
                "resolution": "identity_mismatch", "live": False,
                "terminal": True,
                "identity_errors": [{
                    "field": "job_id", "expected": "positive integer",
                    "observed": job_id,
                }],
            }

    squeue_path = str((plan.get("squeue") or {}).get("path") or "")
    if not squeue_path:
        for partition in probe_specs:
            observations.setdefault(partition, {
                "state": "CONTROLLER_QUERY_ERROR",
                "source": "squeue_missing", "resolution":
                    "controller_query_error",
                "live": False, "terminal": False,
            })
        return observations
    user = str(
        (plan.get("runtime_environment") or {}).get("USER") or ""
    )
    if not user:
        for partition in probe_specs:
            observations.setdefault(partition, {
                "state": "CONTROLLER_QUERY_ERROR",
                "source": "squeue_user_missing", "resolution":
                    "controller_query_error",
                "live": False, "terminal": False,
            })
        return observations
    squeue_command = [
            squeue_path, "-h", "-u", user,
            "-o", "%i|%j|%T|%P|%k",
    ]
    try:
        listed = _run_probe_slurm_query(
            runner, squeue_command, observer_deadline=observer_deadline,
        )
    except subprocess.TimeoutExpired as exc:
        for partition in probe_specs:
            observations.setdefault(partition, {
                "state": "CONTROLLER_QUERY_TIMEOUT",
                "source": "squeue_timeout", "resolution":
                    "controller_query_error",
                "live": False, "terminal": False,
                "query_error": (
                    f"squeue timed out after {exc.timeout} seconds"
                ),
            })
        return observations
    if listed.returncode != 0:
        for partition in probe_specs:
            observations.setdefault(partition, {
                "state": "CONTROLLER_QUERY_ERROR",
                "source": "squeue_error", "resolution":
                    "controller_query_error",
                "live": False, "terminal": False,
                "query_error": (listed.stderr or listed.stdout).strip(),
            })
        return observations
    live_rows = _parse_live_probe_rows(listed.stdout, wanted_ids)

    sacct_path = str((plan.get("sacct") or {}).get("path") or "")
    accounting_rows = {}
    accounting_error = None
    if sacct_path and wanted_ids:
        sacct_command = [
                sacct_path, "-X", "-n", "-P", "-j",
                ",".join(sorted(wanted_ids)),
                "--format=JobIDRaw,JobName%64,State,Partition%64,"
                "Comment%256,ExitCode",
        ]
        try:
            completed = _run_probe_slurm_query(
                runner, sacct_command,
                observer_deadline=observer_deadline,
            )
        except subprocess.TimeoutExpired as exc:
            accounting_error = (
                f"sacct timed out after {exc.timeout} seconds"
            )
        else:
            if completed.returncode == 0:
                accounting_rows = _parse_accounting_probe_rows(
                    completed.stdout, wanted_ids
                )
            else:
                accounting_error = (
                    completed.stderr or completed.stdout
                ).strip()

    for partition, spec in probe_specs.items():
        if partition in observations:
            continue
        job_id = recorded[partition]
        live = live_rows.get(job_id, [])
        accounted = accounting_rows.get(job_id, [])
        matching_accounted = [
            row for row in accounted
            if not _probe_fingerprint_errors(spec, row)
        ]
        stale_accounted = [
            row for row in accounted
            if _probe_fingerprint_errors(spec, row)
        ]
        accounting_evidence = {
            "accounting_rows": accounted,
            "stale_accounting_rows": stale_accounted,
        }
        if live:
            if len(live) != 1:
                observations[partition] = {
                    "state": "CONTROLLER_IDENTITY_MISMATCH",
                    "source": "squeue", "resolution":
                        "identity_mismatch",
                    "live": True, "terminal": True,
                    "identity_errors": [{
                        "field": "job_id", "expected": "one live row",
                        "observed": len(live),
                    }],
                    **accounting_evidence,
                }
                continue
            errors = _probe_fingerprint_errors(spec, live[0])
            if errors:
                observations[partition] = {
                    "state": "CONTROLLER_IDENTITY_MISMATCH",
                    "source": "squeue", "resolution":
                        "identity_mismatch",
                    "live": True, "terminal": True,
                    "identity_errors": errors,
                    "live_row": live[0],
                    **accounting_evidence,
                }
                continue
            accounting_state = (
                matching_accounted[0]["state"]
                if len(matching_accounted) == 1 else None
            )
            live_state = live[0]["state"]
            observations[partition] = {
                "state": live_state,
                "source": "squeue", "resolution": "live",
                "live": True, "terminal": False,
                "live_row": live[0],
                "accounting_state": accounting_state,
                "state_disagreement": bool(
                    accounting_state and accounting_state != live_state
                ),
                "stale_accounting_conflict": bool(
                    stale_accounted
                    or accounting_state in PROBE_TERMINAL_STATES
                ),
                **accounting_evidence,
            }
            continue
        if accounting_error is not None or not sacct_path:
            observations[partition] = {
                "state": "ACCOUNTING_QUERY_ERROR",
                "source": "sacct_error" if sacct_path else "sacct_missing",
                "resolution": "accounting_query_error",
                "live": False, "terminal": False,
                "query_error": accounting_error,
            }
            continue
        if len(matching_accounted) > 1:
            observations[partition] = {
                "state": "ACCOUNTING_IDENTITY_MISMATCH",
                "source": "sacct", "resolution": "identity_mismatch",
                "live": False, "terminal": True,
                "identity_errors": [{
                    "field": "job_id", "expected":
                        "one matching accounting row",
                    "observed": len(matching_accounted),
                }],
                **accounting_evidence,
            }
            continue
        if not matching_accounted:
            if accounted:
                observations[partition] = {
                    "state": "ACCOUNTING_IDENTITY_MISMATCH",
                    "source": "sacct", "resolution":
                        "identity_mismatch",
                    "live": False, "terminal": True,
                    "identity_errors": [
                        error for row in accounted
                        for error in _probe_fingerprint_errors(spec, row)
                    ],
                    **accounting_evidence,
                }
            else:
                observations[partition] = {
                    "state": "ACCOUNTING_PENDING",
                    "source": "sacct", "resolution":
                        "awaiting_accounting",
                    "live": False, "terminal": False,
                }
            continue
        row = matching_accounted[0]
        state = row["state"]
        if state not in PROBE_TERMINAL_STATES:
            observations[partition] = {
                "state": state or "ACCOUNTING_PENDING",
                "source": "sacct", "resolution":
                    "awaiting_accounting",
                "live": False, "terminal": False,
                "accounting_row": row,
            }
            continue
        if not re.fullmatch(r"[0-9]+:[0-9]+", row["exit_code"]):
            observations[partition] = {
                "state": "ACCOUNTING_OUTCOME_MISMATCH",
                "source": "sacct", "resolution": "identity_mismatch",
                "live": False, "terminal": True,
                "identity_errors": [{
                    "field": "exit_code", "expected": "N:N",
                    "observed": row["exit_code"],
                }],
                "accounting_row": row,
            }
            continue
        if state == "COMPLETED" and row["exit_code"] != "0:0":
            observations[partition] = {
                "state": "ACCOUNTING_OUTCOME_MISMATCH",
                "source": "sacct", "resolution": "identity_mismatch",
                "live": False, "terminal": True,
                "identity_errors": [{
                    "field": "exit_code", "expected": "0:0",
                    "observed": row["exit_code"],
                }],
                "accounting_row": row,
            }
            continue
        observations[partition] = {
            "state": state,
            "source": "sacct", "resolution": "accounting_terminal",
            "live": False, "terminal": True,
            "exit_code": row["exit_code"],
            "accounting_row": row,
        }
    return observations


def _probe_artifact_observation(plan, plan_sha, partition, spec):
    path = Path(str(spec.get("output") or ""))
    sidecar = Path(str(path) + ".sha256")
    expected_path = (
        Path(plan["campaign_root"]) / "probes"
        / f"{partition}.attempt{spec.get('attempt')}.json"
    ).resolve()
    base = {
        "output": str(path),
        "artifact_sha256": None,
        "differences": [],
        "observed_node_metadata": None,
        "path_bound": path.resolve() == expected_path,
    }
    if not path.is_file():
        return {
            **base,
            "status": "invalid" if sidecar.exists() else "missing",
            "artifact_error": (
                "sidecar exists without artifact" if sidecar.exists()
                else None
            ),
        }
    if not sidecar.is_file():
        return {**base, "status": "awaiting_sidecar"}
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("probe artifact must be a JSON object")
        digest = sha256_file(path)
        sidecar_parts = sidecar.read_text().split()
        if len(sidecar_parts) != 2:
            raise ValueError("probe sidecar must contain digest and basename")
        sidecar_digest, sidecar_name = sidecar_parts
        identity_errors = []
        planned_identity = str(
            (plan.get("python_identity") or {}).get(
                "portable_identity_sha256"
            ) or ""
        )
        if re.fullmatch(r"[0-9a-f]{64}", planned_identity) is None:
            identity_errors.append({
                "field": "plan.python_identity.portable_identity_sha256",
                "expected": "64 lowercase hexadecimal characters",
                "observed": planned_identity,
            })
        expected = {
            "schema": "evsp-dr-scale-ladder-environment-probe-v1",
            "plan_sha256": plan_sha,
            "probe_id": _probe_id(partition),
            "probe_attempt": spec.get("attempt"),
            "slurm_job_id": str(spec.get("job_id")),
            "slurm_partition": partition,
            "planned_portable_identity_sha256": planned_identity,
            "observed_portable_identity_sha256": planned_identity,
        }
        for field, expected_value in expected.items():
            observed = payload.get(field)
            if field == "slurm_job_id":
                observed = str(observed)
            if observed != expected_value:
                identity_errors.append({
                    "field": field,
                    "expected": expected_value,
                    "observed": observed,
                })
        if path.resolve() != expected_path:
            identity_errors.append({
                "field": "output", "expected": str(expected_path),
                "observed": str(path.resolve()),
            })
        if digest != sidecar_digest:
            identity_errors.append({
                "field": "artifact_sha256", "expected": digest,
                "observed": sidecar_digest,
            })
        if sidecar_name != path.name:
            identity_errors.append({
                "field": "sidecar_basename", "expected": path.name,
                "observed": sidecar_name,
            })
        differences = payload.get("differences")
        if not isinstance(differences, list):
            identity_errors.append({
                "field": "differences", "expected": "list",
                "observed": type(differences).__name__,
            })
            differences = []
        if not isinstance(payload.get("compatible"), bool):
            identity_errors.append({
                "field": "compatible", "expected": "boolean",
                "observed": payload.get("compatible"),
            })
        elif payload.get("compatible") is True and differences:
            identity_errors.append({
                "field": "compatible", "expected":
                    "false when differences are present",
                "observed": True,
            })
        elif payload.get("compatible") is False and not differences:
            identity_errors.append({
                "field": "differences", "expected":
                    "nonempty when compatible is false",
                "observed": differences,
            })
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            **base, "status": "invalid",
            "artifact_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        **base,
        "status": "valid" if not identity_errors else "invalid",
        "artifact_sha256": digest,
        "artifact_compatible": (
            not identity_errors and payload.get("compatible") is True
        ),
        "artifact_identity_errors": identity_errors,
        "differences": differences,
        "observed_node_metadata": payload.get("observed_node_metadata"),
    }


def _wait_for_probes(plan, plan_sha, probe_specs, timeout_s=900):
    deadline = time.monotonic() + timeout_s
    observations = {}
    artifacts = {}
    observer_deadline_reached = False
    first_poll = True
    while first_poll or time.monotonic() < deadline:
        first_poll = False
        if time.monotonic() >= deadline:
            observer_deadline_reached = True
            break
        observations = _probe_job_states(
            plan, probe_specs, observer_deadline=deadline,
        )
        artifacts = {
            partition: _probe_artifact_observation(
                plan, plan_sha, partition, spec
            )
            for partition, spec in probe_specs.items()
        }
        if any(
            artifact.get("status") == "invalid"
            for artifact in artifacts.values()
        ):
            break
        ready = True
        for partition, observed in observations.items():
            if observed.get("terminal") is not True:
                ready = False
                break
            if (
                observed.get("state") == "COMPLETED"
                and artifacts[partition]["status"]
                in {"missing", "awaiting_sidecar"}
            ):
                ready = False
                break
        if ready:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            observer_deadline_reached = True
            break
        time.sleep(min(2, remaining))
    else:
        observer_deadline_reached = True
    results = {}
    for partition, spec in probe_specs.items():
        observation = observations.get(partition, {
            "state": "OBSERVER_DEADLINE", "source": "wait_deadline",
            "resolution": "observer_deadline", "live": False,
            "terminal": False,
        })
        artifact = artifacts.get(partition) or _probe_artifact_observation(
            plan, plan_sha, partition, spec
        )
        awaiting_artifact = bool(
            observation.get("state") == "COMPLETED"
            and artifact["status"] in {"missing", "awaiting_sidecar"}
        )
        waiting = bool(
            observation.get("terminal") is not True or awaiting_artifact
        )
        if artifact.get("status") == "invalid":
            resolution = "artifact_identity_mismatch"
            waiting = False
        elif (
            observation.get("state") in PROBE_RETRYABLE_STATES
            and artifact.get("status") == "valid"
            and artifact.get("artifact_compatible") is False
            and artifact.get("differences")
        ):
            resolution = "environment_mismatch"
            waiting = False
        elif awaiting_artifact:
            resolution = "awaiting_artifact"
        elif observation.get("state") in PROBE_RETRYABLE_STATES:
            resolution = "scheduler_failure"
        else:
            resolution = observation.get("resolution")
        result = {
            "job_id": spec.get("job_id"),
            "state": observation["state"],
            "state_source": observation.get("source"),
            "state_resolution": resolution,
            "live_at_observation": observation.get("live", False),
            "observer_deadline_reached": bool(
                observer_deadline_reached and waiting
            ),
            "wait_timed_out": bool(
                observer_deadline_reached and waiting
            ),
            "compatible": bool(
                observation.get("resolution") == "accounting_terminal"
                and observation.get("state") == "COMPLETED"
                and observation.get("exit_code") == "0:0"
                and artifact.get("status") == "valid"
                and artifact.get("artifact_compatible") is True
            ),
            "artifact_status": artifact.get("status"),
            "probe_id": spec.get("probe_id"),
            "partition": spec.get("partition"),
            "attempt": spec.get("attempt"),
            "comment": spec.get("comment"),
            "job_name": spec.get("job_name"),
            **{
                key: value for key, value in artifact.items()
                if key != "artifact_compatible"
                and value is not None
            },
        }
        for field in (
            "identity_errors", "query_error", "live_row",
            "accounting_row", "accounting_rows",
            "stale_accounting_rows", "accounting_state",
            "state_disagreement", "stale_accounting_conflict",
            "exit_code",
        ):
            if field in observation:
                result[field] = observation[field]
        results[partition] = result
    return results


def _probes_compatible(results):
    return (
        set(results) == set(PROBE_PARTITIONS)
        and len({
            str(result.get("job_id")) for result in results.values()
        }) == 2
        and len({
            str(Path(result.get("output") or "").resolve())
            for result in results.values()
        }) == 2
        and len({
            str(Path(result.get("output") or "").resolve().parent)
            for result in results.values()
        }) == 1
        and all(
            Path(result.get("output") or "").name
            == f"{key}.attempt{result.get('attempt')}.json"
            and Path(result.get("output") or "").parent.name == "probes"
            and result.get("path_bound") is True
            and result.get("partition") == key
            and result.get("probe_id") == _probe_id(key)
            and isinstance(result.get("attempt"), int)
            and result.get("attempt") >= 1
            for key, result in results.items()
        )
        and all(
            result.get("compatible") is True
            for result in results.values()
        )
    )


def _probe_result_waiting(result):
    return bool(
        result.get("observer_deadline_reached") is True
        or result.get("state_resolution") in PROBE_WAITING_RESOLUTIONS
    )


def _probes_waiting(results):
    return bool(results) and any(
        _probe_result_waiting(result) for result in results.values()
    ) and all(
        result.get("compatible") is True
        or _probe_result_waiting(result)
        for result in results.values()
    )


def _load_bound_campaign(root, expected_plan_sha):
    root = Path(root).resolve()
    plan_path = root / "approved-plan.json"
    plan_raw = plan_path.read_bytes()
    if hashlib.sha256(plan_raw).hexdigest() != expected_plan_sha:
        raise ValueError("approved plan hash mismatch")
    plan = json.loads(plan_raw)
    if (
        Path(plan.get("campaign_root") or "").resolve() != root
        or plan.get("submission_protocol")
        != "probe_first_activation_v1"
    ):
        raise ValueError("campaign root/protocol identity mismatch")
    manifest_path = root / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("approval_sha256") != expected_plan_sha:
        raise ValueError("campaign manifest approval mismatch")
    return root, plan_path, plan, manifest_path, manifest


def _require_activation_runtime(plan, manifest):
    _validate_submission_contract(plan)
    for relative, expected in plan["code_hashes"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise ValueError(f"reviewed code hash mismatch: {relative}")
    activation = manifest.get("activation") or {}
    probe_specs = manifest.get("infrastructure_probes") or {}
    expected = _activation_spec(
        manifest["approval_sha256"],
        Path(plan["campaign_root"]),
        probe_specs,
        activation.get("attempt"),
        dependency_job_ids=activation.get("dependency_job_ids"),
    )
    if any(
        activation.get(field) != expected[field]
        for field in (
            "attempt", "partition", "comment", "job_name",
            "probe_job_ids", "dependency_job_ids", "campaign_root",
        )
    ):
        raise ValueError("activation manifest fingerprint mismatch")
    job_id = str(os.environ.get("SLURM_JOB_ID") or "")
    if (
        not job_id.isdigit()
        or str(activation.get("job_id") or "") != job_id
    ):
        raise ValueError("activation runtime identity mismatch")
    observation = _resolve_bound_job(plan, activation)
    if (
        observation.get("live") is not True
        or observation.get("state") != "RUNNING"
    ):
        raise ValueError(
            "activation job is not exact-bound and RUNNING in Slurm"
        )
    return job_id


def activate_existing(root, expected_plan_sha, *, wait_s=600):
    """Validate probes, then idempotently create/release scientific work."""
    root = Path(root).resolve()
    with _campaign_lock(root):
        return _activate_existing_locked(root, expected_plan_sha, wait_s)


def _activate_existing_locked(root, expected_plan_sha, wait_s):
    (
        root, _plan_path, plan, manifest_path, manifest,
    ) = _load_bound_campaign(root, expected_plan_sha)
    activation_job_id = _require_activation_runtime(plan, manifest)
    if manifest.get("submitted") is True:
        return manifest

    probe_specs = dict(manifest.get("infrastructure_probes") or {})
    if set(probe_specs) != set(PROBE_PARTITIONS):
        raise ValueError("campaign does not record both infrastructure probes")
    for partition, spec in probe_specs.items():
        expected = _probe_spec(
            expected_plan_sha, partition, root, spec.get("attempt")
        )
        if any(
            spec.get(field) != expected[field]
            for field in (
                "output", "probe_id", "partition", "attempt", "comment",
                "job_name",
            )
        ) or not str(spec.get("job_id") or "").isdigit():
            raise ValueError("recorded probe specification mismatch")

    manifest["submission_state"] = "probe_evaluating"
    manifest["activation"]["state"] = "probe_evaluating"
    _replace_json(manifest_path, manifest)
    probe_results = _wait_for_probes(
        plan, expected_plan_sha, probe_specs, timeout_s=min(wait_s, 600)
    )
    manifest["probe_results"] = probe_results
    if not _probes_compatible(probe_results):
        if _probes_waiting(probe_results):
            manifest["probe_state"] = "observation_incomplete"
            manifest["submission_state"] = "probe_observation_incomplete"
            manifest["activation"]["state"] = "observation_incomplete"
            _replace_json(manifest_path, manifest)
            raise RuntimeError(
                "probe accounting is incomplete; scientific work remains absent"
            )
        manifest["probe_state"] = "failed"
        manifest["submission_state"] = "probe_failed"
        manifest["activation"]["state"] = "blocked_probe_failure"
        _replace_json(manifest_path, manifest)
        raise RuntimeError(
            "infrastructure probes failed; no scientific work was created"
        )
    manifest["probe_state"] = "passed"
    manifest["activation"]["state"] = "probe_passed"

    if manifest.get("gate_state") == "not_created":
        if (
            manifest.get("reservations")
            or manifest.get("submitted_arrays")
            or manifest.get("gate_job_id") is not None
        ):
            raise ValueError("pre-science campaign invariant violated")
        manifest["submission_state"] = "input_staging"
        _replace_json(manifest_path, manifest)
        _stage_scientific_inputs(plan)
        manifest["submission_state"] = "reserving"
        manifest["reservation_state"] = "creating"
        expected_lineage = _activation_lineage_id(plan, expected_plan_sha)
        recorded_lineage = manifest.get("activation_lineage_id")
        if recorded_lineage not in (None, expected_lineage):
            raise ValueError("activation lineage mismatch")
        manifest["activation_lineage_id"] = expected_lineage
        _replace_json(manifest_path, manifest)
        reservations = _ensure_reservations(
            plan, expected_plan_sha, expected_lineage
        )
        manifest["reservations"] = [str(path) for path in reservations]
        manifest["reservation_state"] = "complete"
        manifest["submission_state"] = "gate_creating"
        manifest["gate_state"] = "creating"
        _replace_json(manifest_path, manifest)

    from reconcile_scale_ladder_gate import (
        GateTerminalObservationError,
        _persist_gate_terminal_failure,
        _reconcile_locked,
        _resolve_gate_state,
    )

    if manifest.get("gate_state") in {
        "creating", "held", "held_after_partial_submission",
        "held_probe_failure", "held_probe_waiting",
        "held_probe_passed", "release_attempting",
        "held_release_failed", "release_retry_attempting",
    }:
        manifest = _reconcile_locked(
            root,
            expected_plan_sha,
            release_held_gate=True,
            resume_missing_arrays=True,
        )
    if manifest.get("gate_state") == "release_retry_requested":
        deadline = time.monotonic() + wait_s
        gate = manifest.get("gate_job_id")
        observation = None
        while time.monotonic() < deadline:
            try:
                observation = _resolve_gate_state(
                    plan, gate, expected_plan_sha
                )
            except GateTerminalObservationError as exc:
                _persist_gate_terminal_failure(
                    manifest, manifest_path, gate, exc.observation
                )
                raise RuntimeError(str(exc)) from exc
            if observation["state"] in PROBE_TERMINAL_STATES:
                break
            time.sleep(2)
        state = observation["state"] if observation else None
        if state != "COMPLETED":
            if state in PROBE_TERMINAL_STATES:
                _persist_gate_terminal_failure(
                    manifest, manifest_path, gate, observation
                )
            raise RuntimeError(
                f"scientific gate did not complete after release: {state}"
            )
        manifest = _reconcile_locked(root, expected_plan_sha)
    manifest["submission_state"] = "submitted"
    manifest["activation"]["state"] = "complete"
    _replace_json(manifest_path, manifest)
    return manifest
def _submit_array(
    plan, plan_path, plan_sha, group, gate, logs, dependency=None
):
    tasks = plan["task_groups"][group]
    partition = (
        "scaglione" if group.startswith("MIP") else "default_partition"
    )
    max_budget = max(
        job["budget_s"] for job in plan["jobs"]
        if job["job_key"] in tasks
    )
    task_jobs = [
        job for job in plan["jobs"] if job["job_key"] in tasks
    ]
    cg_group = group in {"CG", "CG_SENSITIVITY"}
    memory_gb = (
        max(32, max(job.get("memory_gb", 32) for job in task_jobs))
        if cg_group else 64 if group.startswith("MIP") else 32
    )
    concurrency = (
        min(job.get("max_concurrency", 16) for job in task_jobs)
        if cg_group else None
    )
    arguments = [
        f"--array=0-{len(tasks)-1}"
        + (f"%{concurrency}" if concurrency is not None else ""),
        f"--partition={partition}",
        "--requeue"
        if group in {"CG", "CG_SENSITIVITY"}
        else "--no-requeue",
        "--signal=B:USR1@180",
        "--cpus-per-task=8" if group.startswith("MIP")
        else "--cpus-per-task=2",
        f"--mem={memory_gb}G",
        f"--time={math.ceil(max_budget/60)+(30 if group == 'CG' else 10)}",
        f"--job-name={_array_name(group, plan_sha)}",
        f"--comment=SLAD:{plan_sha[:20]}:{group}",
        f"--output={logs}/%x_%A_%a.out",
        f"--error={logs}/%x_%A_%a.err",
        f"--dependency=afterok:{gate}"
        + (f",{dependency}" if dependency else ""),
        "--export=NONE",
        str(WORKER), str(plan_path), plan_sha, group,
        plan["python"]["path"], plan["python"]["sha256"],
        plan["scontrol"]["path"], plan["scontrol"]["sha256"],
        plan["worker_sha256"],
        str(REPO_ROOT),
        plan["runtime_environment"]["HOME"],
        plan["runtime_environment"]["USER"],
    ]
    return _sbatch(plan, arguments)


def _sbatch(plan, arguments):
    sbatch_path = Path(plan["sbatch"]["path"])
    if (
        plan["sbatch"]["available"] is not True
        or not sbatch_path.is_file()
        or sha256_file(sbatch_path) != plan["sbatch"]["sha256"]
    ):
        raise ValueError("approved sbatch unavailable/changed")
    try:
        completed = subprocess.run(
            [str(sbatch_path), "--parsable", *arguments],
            cwd=REPO_ROOT, text=True, capture_output=True, check=False,
            timeout=SBATCH_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "sbatch outcome ambiguous after bounded timeout"
        ) from exc
    job_id = completed.stdout.strip().split(";", 1)[0]
    if completed.returncode != 0 or not job_id.isdigit():
        raise RuntimeError("sbatch outcome ambiguous")
    return job_id


def _array_name(group, plan_sha):
    code = {
        "PREFLIGHT": "LDPF",
        "SEED": "LDSD",
        "CG": "LDCG",
        "CG_SENSITIVITY": "LDCS",
        "MIP_RAW": "LDMR",
        "MIP_KNOWN": "LDMK",
    }[group]
    return f"{code}{plan_sha[:4]}"


def _copy_new(source, target, expected_sha):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != expected_sha:
            raise ValueError("staged input hash mismatch")
        return
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    with source.open("rb") as reader, temporary.open("xb") as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    try:
        if sha256_file(temporary) != expected_sha:
            raise ValueError("staged input copy mismatch")
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_json(path, payload):
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--reservation-root", type=Path)
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--matrix-out", type=Path)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--retry-failed-activation", action="store_true")
    parser.add_argument("--retry-failed-probes", action="store_true")
    parser.add_argument("--activate-existing", type=Path)
    args = parser.parse_args(argv)
    if args.activate_existing is not None:
        if (
            args.submit
            or args.retry_failed_activation
            or args.retry_failed_probes
            or args.campaign is not None
            or args.reservation_root is not None
            or not args.approved_plan_sha256
        ):
            parser.error(
                "--activate-existing requires only "
                "--approved-plan-sha256"
            )
        activate_existing(
            args.activate_existing, args.approved_plan_sha256
        )
        return 0
    if args.campaign is None or args.reservation_root is None:
        parser.error("--campaign and --reservation-root are required")
    if args.submit and not args.approved_plan_sha256:
        parser.error("--submit requires --approved-plan-sha256")
    if args.retry_failed_activation and not args.submit:
        parser.error("--retry-failed-activation requires --submit")
    if args.retry_failed_probes and not args.submit:
        parser.error("--retry-failed-probes requires --submit")
    if args.retry_failed_activation and args.retry_failed_probes:
        parser.error("choose only one infrastructure retry flag")
    plan = build_plan(args.campaign, args.python, args.reservation_root)
    plan_raw = canonical(plan)
    plan_sha = hashlib.sha256(plan_raw).hexdigest()
    print(json.dumps(plan, indent=2))
    print(f"[approval-sha256] {plan_sha}")
    if args.plan_out:
        if args.matrix_out is None:
            parser.error("--plan-out requires --matrix-out")
        observed = write_plan(plan, args.plan_out, args.matrix_out)
        if observed != plan_sha:
            raise ValueError("plan publication hash mismatch")
    if not args.submit:
        print(
            f"[dry-run] tasks={plan['task_count']} "
            "PREFLIGHT=22 SEED=21 PRIMARY_CG=23 "
            "RESOLUTION_CG=92 MIP=42 k40_MIP=0"
        )
        return 0
    if args.approved_plan_sha256 != plan_sha:
        raise ValueError("current plan differs from approved SHA")
    submit(
        plan, plan_sha,
        retry_failed_activation=args.retry_failed_activation,
        retry_failed_probes=args.retry_failed_probes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

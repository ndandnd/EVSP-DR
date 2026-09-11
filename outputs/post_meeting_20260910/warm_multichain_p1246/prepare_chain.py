#!/usr/bin/env python3
"""Create one immutable inherited-column warm-chain campaign from audited network caches.

The source cache campaign is read-only.  Every cache pickle, cache manifest,
and selected CSV is hash checked before a hard link is created.  The target
manifest changes only its execution-code identity from the cache builder
commit to the reviewed event-pricer commit; all physics and input hashes are
retained and recorded in ``execution_plan.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


SCALES = tuple(range(2, 11))
ALLOWED_REPLICATES = (1, 2, 4, 6)
EXPECTED_IDENTITY = {
    "block_min": 5,
    "charge_kw": 240.0,
    "event_arc_mode": "lazy",
    "g_kwh": 240.0,
    "prices_sha256": "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200",
    "reference_sha256": "7bda0e1f439dc8bf5081499566eb2c6a0314190ef27294707f1403fd2c13e3a0",
    "reserve_kwh": 0.0,
    "schema": "evsp-dr-event-network-cache-v1",
    "soc_step": 2.5,
    "strict_tariff_coverage": False,
    "deadhead_sha256": "5993e922c671f053611635578b32a1be13bab87b3b5fd8c02b699b81fe0eb66c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_path(execution: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.parts and relative.parts[0] == "data":
        relative = Path(*relative.parts[1:])
    return execution / "data" / relative


def fail(message: str) -> None:
    raise SystemExit(f"[prepare inherited chain] {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("execution", type=Path)
    parser.add_argument("source_cache", type=Path)
    parser.add_argument("commit")
    parser.add_argument("--replicate", type=int, choices=ALLOWED_REPLICATES, required=True)
    parser.add_argument(
        "--worker-source",
        type=Path,
        required=True,
        help="Reviewed CG worker staged outside the new campaign.",
    )
    args = parser.parse_args()

    campaign = args.campaign.expanduser().resolve()
    execution = args.execution.expanduser().resolve()
    source_cache = args.source_cache.expanduser().resolve()
    commit = args.commit
    worker_source = args.worker_source.expanduser().resolve()
    replicate = args.replicate
    if campaign.exists():
        fail(f"refusing existing campaign path: {campaign}")
    if not execution.is_dir() or not source_cache.is_dir():
        fail("execution and source_cache must be existing directories")
    try:
        observed_commit = subprocess.check_output(
            ["git", "-C", str(execution), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        tracked_status = subprocess.check_output(
            ["git", "-C", str(execution), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        fail(f"execution repo cannot be authenticated: {exc}")
    if observed_commit != commit:
        fail(f"execution HEAD {observed_commit} does not match {commit}")
    if tracked_status:
        fail("execution repo has tracked modifications")
    if not worker_source.is_file():
        fail(f"immutable CG worker source is missing: {worker_source}")

    selection_source = source_cache / "input_selection_manifest.csv"
    if not selection_source.is_file():
        fail(f"missing selection manifest: {selection_source}")
    selection_sha = sha256(selection_source)
    with selection_source.open(newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    selected = [
        row for row in all_rows
        if row.get("sample_family") == "probability"
        and row.get("selection_replicate") == str(replicate)
        and 2 <= int(row["scale"]) <= 10
    ]
    selected.sort(key=lambda row: int(row["scale"]))
    if [int(row["scale"]) for row in selected] != list(SCALES):
        fail("selection manifest does not contain exactly selected replicate scales k2..k10")

    campaign.mkdir(parents=True, exist_ok=False)
    for name in ("network_cache", "cg", "logs", "snapshots", "records", "mip", "progress", "locks"):
        (campaign / name).mkdir()
    worker = campaign / "warm_chain_cg.sub"
    shutil.copy2(worker_source, worker)
    worker.chmod(0o755)
    selection_target = campaign / "source_input_selection_manifest.csv"
    selection_target.write_bytes(selection_source.read_bytes())

    cache_rows = []
    selection_rows = []
    for row in selected:
        scale = int(row["scale"])
        cell = f"k{scale:02d}_p{replicate}"
        name = f"M__{cell}__event_2p5_event5.pkl"
        original = source_cache / "network_cache" / name
        original_manifest = Path(str(original) + ".manifest.json")
        target = campaign / "network_cache" / name
        target_manifest = Path(str(target) + ".manifest.json")
        if not original.is_file() or not original_manifest.is_file():
            fail(f"missing source cache or manifest for k{scale}: {original}")
        if target.exists() or target_manifest.exists():
            fail(f"refusing to replace target cache: {target}")
        payload = json.loads(original_manifest.read_text())
        if sha256(original) != payload.get("pickle_sha256"):
            fail(f"source cache pickle hash mismatch for k{scale}")
        if payload.get("pickle_bytes") != original.stat().st_size:
            fail(f"source cache byte count mismatch for k{scale}")
        identity = payload.get("identity", {})
        for key, expected in EXPECTED_IDENTITY.items():
            if identity.get(key) != expected:
                fail(f"cache identity {key} mismatch for k{scale}: {identity.get(key)!r}")
        relative_path = row["relative_path"]
        instance = csv_path(execution, relative_path)
        if not instance.is_file():
            fail(f"selected instance is missing: {instance}")
        instance_sha = sha256(instance)
        if instance_sha != row["instance_file_sha256"]:
            fail(f"selected instance hash mismatch for k{scale}")
        if instance_sha != identity.get("instance_sha256"):
            fail(f"cache/CSV instance identity mismatch for k{scale}")
        if row.get("cell_id") != cell or row.get("selection_replicate") != str(replicate):
            fail(f"unexpected selected-cell identity for k{scale}")

        source_manifest_sha = sha256(original_manifest)
        old_commit = identity.get("git_commit")
        payload["identity"]["git_commit"] = commit
        os.link(original, target)
        target_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        cache_rows.append({
            "scale": scale,
            "cell": cell,
            "source": str(original),
            "target": str(target),
            "source_manifest": str(original_manifest),
            "target_manifest": str(target_manifest),
            "source_manifest_sha256": source_manifest_sha,
            "target_manifest_sha256": sha256(target_manifest),
            "pickle_sha256": payload["pickle_sha256"],
            "pickle_bytes": payload["pickle_bytes"],
            "source_cache_git_commit": old_commit,
            "target_execution_git_commit": commit,
            "instance_sha256": instance_sha,
            "relative_instance": relative_path,
        })
        selection_rows.append({
            "cell_id": row["cell_id"],
            "scale": scale,
            "selection_replicate": int(row["selection_replicate"]),
            "nested_chain_id": row.get("nested_chain_id"),
            "nested_parent_cell_id": row.get("nested_parent_cell_id"),
            "addition_rank_duty": row.get("addition_rank_duty"),
            "relative_path": relative_path,
            "instance_file_sha256": row["instance_file_sha256"],
            "target_fleet": row.get("target_fleet"),
            "trip_count": row.get("trip_count"),
            "peak_concurrency_lb": row.get("peak_concurrency_lb"),
            "service_kwh_total": row.get("service_kwh_total"),
        })

    plan = {
        "schema": "evsp-dr-dependent-warm-chain-plan-v2",
        "campaign_kind": "additional_inherited_column_warm_chain",
        "campaign_root": str(campaign),
        "source_cache_root": str(source_cache),
        "source_selection_manifest": str(selection_source),
        "source_selection_manifest_sha256": selection_sha,
        "replicate": replicate,
        "nested_chain_id": f"nested_probability_{replicate}",
        "scales": list(SCALES),
        "source_commit": commit,
        "worker_sha256": sha256(worker),
        "source_exact_pricer_sha256": sha256(execution / "src" / "exact_pricer_expanded.py"),
        "source_event_pricer_sha256": sha256(execution / "src" / "event_pricer_network.py"),
        "scientific_question": (
            "Across an additional random nested chain, does inheriting the complete "
            "child-replayable predecessor pool improve the covering CG pool or "
            "the downstream integer solution?"
        ),
        "physics": {
            "time_model": "event",
            "event_arc_mode": "lazy",
            "master_sense": "cover",
            "battery_kwh": 240.0,
            "charge_kw": 240.0,
            "soc_step_kwh": 2.5,
            "block_min": 5,
            "min_soc_frac": 0.0,
            "columns_per_iteration": 30,
            "column_selection": "reduced_cost",
            "reduced_cost_tolerance": 0.0001,
            "column_candidate_multiplier": 4,
            "column_diversity_weight": 0.0,
            "initial_pool": "real singletons",
        },
        "cg": {
            "dependency": "afterok predecessor scale",
            "inheritance": "full predecessor event-column pool plus real child singletons",
            "inheritance_workers": 8,
            "child_replay": "fixed-sequence charging reoptimization and physical replay in child event graph",
            "mapping_key": "Ordered_Trip_ID",
            "inherited_state": "columns only",
            "not_inherited": ["duals", "LP basis", "LP certificate"],
            "max_iters": 50000,
            "wall_limit_s": 28800,
            "checkpoint_every": 25,
            "record_import_runtime_separately": True,
            "no_alternate_replay_algorithm_claim": True,
        },
        "mip": {
            "execution_commit": "871d057e1067411f09581e37d78f7c1ca43f68bb",
            "runner_sha256": "bcb5a6b76040ff6ddfa932433d296a1f0f72207b28cbba738b1b4dd39f1eaac7",
            "freeze_sha256": "dda2af16aa37d702076626cf48fd94751d634ae9785530bc9c63ca62bbac67c6",
            "license_preflight_sha256": "6c1ca6c89caa3ed236e5a087f179f6536ebea4dcec73305f134af3ff1fad8a8f",
            "cover": True,
            "two_stage": True,
            "total_timelimit_s": 3600,
            "stage1_max_timelimit_s": 1800,
            "stage2_budget": "remaining wall time",
            "stage2_fleet_constraint": "at_most best validated stage-1 incumbent",
            "stage2_runs_without_global_stage1_proof": True,
            "stage2_objective": "charging-related route cost including charge start fee",
            "saved_start_policy": "normal greedy pool start only; no external route augmentation",
            "physical_validation_required": True,
        },
        "slurm": {
            "cg": {
                "partition": "default_partition",
                "exclude": "scaglione-compute-01",
                "cpus_per_task": 8,
                "memory": "96G",
                "time": "08:15:00",
                "requeue": True,
                "inherit_workers": 8,
            },
            "freeze": {
                "partition": "default_partition",
                "exclude": "scaglione-compute-01",
                "cpus_per_task": 1,
                "memory": "16G",
                "time": "02:00:00",
                "requeue": False,
            },
            "mip": {
                "partition": "scaglione",
                "exclude": "scaglione-compute-01",
                "cpus_per_task": 8,
                "memory": "48G",
                "time": "02:00:00",
                "solver_time_limit": "01:00:00",
                "requeue": False,
                "sbatch_binary": "/usr/local/slurm/slurm-25.05.5/bin/sbatch",
                "max_concurrent_cases": 2,
                "actual_chain_concurrency": 1,
            },
        },
        "selection_rows": selection_rows,
        "cache_rows": cache_rows,
        "provenance_policy": {
            "fresh_baseline_reference": f"p{replicate} fresh statuses may be compared by scale, but never merged into this warm pool",
            "mip_start_provenance": "greedy pool start generated from this frozen pool; no fresh-solver routes injected",
            "claims": "report CG import time and optimization time separately; do not claim warm speedup until matched comparison exists",
        },
    }
    plan_path = campaign / "execution_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "campaign": str(campaign),
        "cache_count": len(cache_rows),
        "selection_manifest_sha256": selection_sha,
        "worker_sha256": plan["worker_sha256"],
        "execution_plan_sha256": sha256(plan_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

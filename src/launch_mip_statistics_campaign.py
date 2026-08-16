#!/usr/bin/env python3
"""Inventory and approval-gated launcher for MIP convergence statistics."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from durable_io import flush_and_fsync
from mip_statistics_inventory import (
    PILOT_BUDGET_HOURS,
    SECONDARY_AGES,
    SECONDARY_SCALES,
    inventory,
    representative_candidates,
    select_age_candidate,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEWED_BASE = "ae736fbc9c5fef71f39d7d758b7062355c485313"
WORKER_PATH = REPO_ROOT / "src/submit_mip_statistics.sub"
RUNNER_PATH = REPO_ROOT / "src/run_exact_pool_mip.py"
DEFAULT_ROOTS = {
    "repool_small": REPO_ROOT / "results/repool_small",
    "exact_big": REPO_ROOT / "results/exact_big",
    "k40_factorial": REPO_ROOT / "results/k40_factorial",
    "bigtar_snapshots": REPO_ROOT / "results/bigtar_snapshots",
}


def _git(*args, binary=False):
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=not binary,
    )


def checkout_identity(*, require_detached: bool) -> dict:
    head = _git("rev-parse", "--verify", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    symbolic = _git("symbolic-ref", "-q", "HEAD")
    ancestor = _git(
        "merge-base", "--is-ancestor", REVIEWED_BASE, "HEAD"
    )
    commit = head.stdout.strip()
    if (
        head.returncode != 0
        or len(commit) != 40
        or status.returncode != 0
        or status.stdout.strip()
        or ancestor.returncode != 0
    ):
        raise SystemExit(
            "checkout must be tracked-clean and descend from reviewed base"
        )
    if symbolic.returncode == 0:
        detached = False
        branch = symbolic.stdout.strip()
    elif symbolic.returncode == 1:
        detached = True
        branch = ""
    else:
        raise SystemExit("cannot verify checkout branch state")
    if require_detached and not detached:
        raise SystemExit("submission requires a detached reviewed checkout")
    return {
        "expected_commit": commit,
        "reviewed_base_commit": REVIEWED_BASE,
        "detached": detached,
        "branch": branch,
        "tracked_clean": True,
    }


def _write_new_atomic(path: Path, payload: bytes, *, executable=False) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.tmp.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if executable:
            temporary.chmod(0o500)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite {path}") from exc
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _replace_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        flush_and_fsync(handle)
    os.replace(temporary, path)


def _canonical(payload: dict) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()


def _parse_assignments(values, *, value_type=Path) -> dict:
    parsed = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"expected NAME=VALUE, found {value!r}")
        name, raw = value.split("=", 1)
        if not name or name in parsed:
            raise SystemExit(f"duplicate/empty assignment {name!r}")
        parsed[name] = value_type(raw)
    return parsed


def _validated_start(path: Path, candidate: dict) -> dict:
    source = path.expanduser().resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    routes = payload.get("routes") if isinstance(payload, dict) else None
    if (
        not isinstance(routes, list)
        or not routes
        or payload.get("infeasible") not in (None, [])
    ):
        raise ValueError(f"GIRO start is missing/partial: {source}")
    status = json.loads(Path(candidate["status_path"]).read_text())
    expected = set(status["trip_ids"])
    counts = {trip: 0 for trip in expected}
    for route in routes:
        nodes = route.get("route", route.get("route_nodes", []))
        trips = [
            node for node in nodes
            if isinstance(node, int) and not isinstance(node, bool)
        ]
        if not trips or len(trips) != len(set(trips)):
            raise ValueError(f"GIRO start has an invalid route: {source}")
        for trip in trips:
            if trip not in counts:
                raise ValueError(f"GIRO start has an unknown trip: {trip}")
            counts[trip] += 1
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"GIRO start is not an exact partition: {source}")
    return {
        "path": str(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "route_count": len(routes),
        "trip_set_sha256": candidate["trip_set_sha256"],
    }


def _rep_code(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    return (cleaned or "X")[-2:].rjust(2, "0")


def _job_name(job, campaign: str) -> str:
    age = (
        f"A{int(round(job['age_hours'])):02d}"
        if job.get("matrix") == "secondary" else ""
    )
    arm = "R" if job["arm"] == "RAW" else "G"
    nonce = hashlib.sha256(
        f"{campaign}|{job['cell_id']}".encode()
    ).hexdigest()[:2].upper()
    name = (
        f"S{job['scale']:02d}{_rep_code(job['replicate'])}"
        f"{arm}{job['budget_hours']:02d}H{age}{nonce}"
    )
    if len(name) > 15:
        raise ValueError(f"Slurm name exceeds 15 characters: {name}")
    return name


def _fresh_preparation(scale: int) -> dict:
    filename = f"Practice_{scale}bus.csv"
    return {
        "scale": scale,
        "classification": "first-N stress instance",
        "verified_service_day_sample": False,
        "instance": filename,
        "prices_csv": "hourly_prices_flat.csv",
        "physics": {
            "g_kwh": 300,
            "charge_kw": 300,
            "min_soc_frac": 0.0,
            "soc_step": 15,
            "block_min": 10,
        },
        "column_discovery": {
            "master_sense": "cover",
            "initial_pool": "singletons",
        },
        "final_mip_partitioning": "strict_exact_once",
        "snapshot_minutes": [60, 180, 360, 600, 720, 900, 1440],
        "command": (
            "python -u src/exact_pricer_expanded.py "
            f"--csv {filename} --prices_csv hourly_prices_flat.csv "
            "--g-kwh 300 --charge-kw 300 --min-soc-frac 0 "
            "--soc-step 15 --block-min 10 --master-sense cover "
            "--initial-pool singletons --snapshot-at-minutes "
            "60,180,360,600,720,900,1440 --resume "
            f"--out results/mip_statistics_prep/k{scale}.json"
        ),
        "planned_only": True,
    }


def _job_from_candidate(
    candidate,
    *,
    arm,
    budget_hours,
    start_map,
    matrix,
    age_label=None,
):
    start = None
    blocked = []
    if arm == "GIRO":
        start_path = start_map.get(str(candidate["scale"]))
        if start_path is None:
            blocked.append("validated_giro_start_missing")
        else:
            try:
                start = _validated_start(start_path, candidate)
            except (OSError, ValueError) as exc:
                blocked.append(f"validated_giro_start_invalid: {exc}")
    cell = (
        f"k{candidate['scale']}_{candidate['replicate']}_{arm.lower()}"
        + (f"_a{age_label}" if age_label is not None else "")
    )
    return {
        "cell_id": cell,
        "matrix": matrix,
        "scale": candidate["scale"],
        "replicate": candidate["replicate"],
        "arm": arm,
        "augmentation_changes_column_set": arm == "GIRO",
        "partitioning": "strict_exact_once",
        "two_stage": True,
        "budget_hours": budget_hours,
        "time_limit_s": int(budget_hours * 3600),
        "threads": 8,
        "mip_gap": 1e-4,
        "age_hours": candidate.get("age_hours"),
        "source": candidate,
        "validated_start": start,
        "blocked_reasons": blocked,
    }


def build_plan(
    inventory_payload,
    *,
    mode: str,
    campaign: str,
    start_map: dict,
    identity: dict,
) -> dict:
    selected = representative_candidates(inventory_payload)
    preparations = [_fresh_preparation(scale) for scale in (10, 20)]
    jobs = []
    if mode == "pilot":
        for scale, budget in PILOT_BUDGET_HOURS.items():
            candidate = selected.get(scale)
            if candidate is None:
                continue
            for arm in ("RAW", "GIRO"):
                jobs.append(_job_from_candidate(
                    candidate,
                    arm=arm,
                    budget_hours=budget,
                    start_map=start_map,
                    matrix="pilot",
                ))
    elif mode == "secondary":
        candidates = inventory_payload.get("candidates") or []
        for scale in SECONDARY_SCALES:
            for target in SECONDARY_AGES:
                candidate = select_age_candidate(
                    candidates, scale=scale, target=target
                )
                if candidate is None:
                    continue
                label = (
                    "-or-".join(str(value) for value in target)
                    if isinstance(target, tuple) else str(target)
                )
                jobs.append(_job_from_candidate(
                    candidate,
                    arm="GIRO",
                    budget_hours=2,
                    start_map=start_map,
                    matrix="secondary",
                    age_label=label,
                ))
    else:
        raise ValueError(mode)
    campaign_root = (
        REPO_ROOT / "src/results/mip_statistics" / campaign
    ).resolve()
    log_root = (
        REPO_ROOT / "src/logs/mip_statistics" / campaign
    ).resolve()
    for job in jobs:
        job["job_name"] = _job_name(job, campaign)
        cell_root = campaign_root / "input" / job["cell_id"]
        source_status_name = Path(job["source"]["status_path"]).name
        status_name = (
            source_status_name
            if source_status_name.endswith(".snapshot.json")
            else f"{Path(source_status_name).stem}.frozen.snapshot.json"
        )
        job.update({
            "staged_status": str(cell_root / status_name),
            "staged_journal": str(
                cell_root / f"{status_name}.columns.jsonl"
            ),
            "staged_instance": str(
                cell_root / "data" / job["source"]["csv"]
            ),
            "staged_tariff": str(
                cell_root / "data" / job["source"]["prices_csv"]
            ),
            "staged_start": (
                str(cell_root / "validated_start.json")
                if job["validated_start"] else None
            ),
            "job_spec": str(cell_root / "job.json"),
            "output": str(
                campaign_root / "outputs" / f"{job['cell_id']}.json"
            ),
            "progress_dir": str(
                campaign_root / "progress" / job["cell_id"]
            ),
            "job_id": None,
            "submission_state": "planned",
        })
    inventory_sha = hashlib.sha256(
        _canonical(inventory_payload)
    ).hexdigest()
    worker_sha = (
        sha256_file(WORKER_PATH) if WORKER_PATH.is_file() else None
    )
    runner_sha = sha256_file(RUNNER_PATH)
    missing_scales = [
        scale for scale in PILOT_BUDGET_HOURS if scale not in selected
    ]
    expected_secondary_cells = len(SECONDARY_SCALES) * len(SECONDARY_AGES)
    missing_matrix_cells = (
        expected_secondary_cells - len(jobs)
        if mode == "secondary" else 0
    )
    plan = {
        "schema": "evsp-dr-mip-statistics-approved-plan-v1",
        "campaign": campaign,
        "mode": mode,
        "campaign_root": str(campaign_root),
        "log_root": str(log_root),
        "checkout_identity": identity,
        "inventory_sha256": inventory_sha,
        "selection_rule": inventory_payload["selection_rule"],
        "selected_candidates": {
            str(scale): candidate["candidate_id"]
            for scale, candidate in selected.items()
        },
        "fresh_exact_cg_preparations": preparations,
        "resources": {
            "partition": "scaglione",
            "threads": 8,
            "requeue": False,
            "signal": "B:USR1@180",
        },
        "worker": str(WORKER_PATH),
        "worker_sha256": worker_sha,
        "runner": str(RUNNER_PATH),
        "runner_sha256": runner_sha,
        "jobs": jobs,
        "missing_scales": missing_scales,
        "missing_matrix_cells": missing_matrix_cells,
        "blocked": (
            any(job["blocked_reasons"] for job in jobs)
            or (mode == "pilot" and bool(missing_scales))
            or (mode == "secondary" and missing_matrix_cells > 0)
        ),
        "global_route_space_optimality_claimed": False,
    }
    return plan


def _stage_and_submit(plan: dict, plan_sha: str) -> dict:
    if not plan["jobs"]:
        raise SystemExit("approved plan contains no runnable jobs")
    if plan["blocked"]:
        raise SystemExit("approved plan contains blocked cells")
    root = Path(plan["campaign_root"])
    logs = Path(plan["log_root"])
    if root.exists() or logs.exists():
        raise SystemExit("campaign already exists; reruns need a new name")
    if plan["worker_sha256"] is None:
        raise SystemExit("worker is not committed/available")
    root.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    worker = root / "input/submit_mip_statistics.sub"
    runner = root / "input/run_exact_pool_mip.py"
    _write_new_atomic(worker, WORKER_PATH.read_bytes(), executable=True)
    _write_new_atomic(runner, RUNNER_PATH.read_bytes())
    plan_path = root / "approved-plan.json"
    plan_raw = _canonical(plan)
    _write_new_atomic(plan_path, plan_raw)
    if hashlib.sha256(plan_raw).hexdigest() != plan_sha:
        raise SystemExit("staged approval plan hash mismatch")
    for job in plan["jobs"]:
        source = job["source"]
        copies = (
            ("status_path", "staged_status", "status_sha256"),
            ("journal_path", "staged_journal", "journal_sha256"),
            ("instance_path", "staged_instance", "instance_sha256"),
            ("tariff_path", "staged_tariff", "tariff_sha256"),
        )
        for source_key, staged_key, hash_key in copies:
            source_path = Path(source[source_key])
            if sha256_file(source_path) != source[hash_key]:
                raise SystemExit(f"{job['cell_id']}: source changed")
            _write_new_atomic(
                Path(job[staged_key]), source_path.read_bytes()
            )
            if sha256_file(Path(job[staged_key])) != source[hash_key]:
                raise SystemExit(f"{job['cell_id']}: staged hash mismatch")
        if job["validated_start"]:
            start = job["validated_start"]
            start_path = Path(start["path"])
            if sha256_file(start_path) != start["sha256"]:
                raise SystemExit(f"{job['cell_id']}: start changed")
            _write_new_atomic(
                Path(job["staged_start"]), start_path.read_bytes()
            )
        spec = {
            "schema": "evsp-dr-mip-statistics-job-v1",
            "cell_id": job["cell_id"],
            "arm": job["arm"],
            "scale": job["scale"],
            "replicate": job["replicate"],
            "time_limit_s": job["time_limit_s"],
            "threads": 8,
            "mip_gap": job["mip_gap"],
            "status": job["staged_status"],
            "status_sha256": source["status_sha256"],
            "journal": job["staged_journal"],
            "journal_sha256": source["journal_sha256"],
            "instance": job["staged_instance"],
            "instance_sha256": source["instance_sha256"],
            "instance_relative": source["csv"],
            "tariff": job["staged_tariff"],
            "tariff_sha256": source["tariff_sha256"],
            "tariff_relative": source["prices_csv"],
            "validated_start": job["staged_start"],
            "validated_start_sha256": (
                job["validated_start"]["sha256"]
                if job["validated_start"] else None
            ),
            "runner_sha256": plan["runner_sha256"],
            "worker_sha256": plan["worker_sha256"],
            "approved_plan_sha256": plan_sha,
            "approved_plan": str(plan_path),
            "output": job["output"],
            "progress_dir": job["progress_dir"],
            "expected_commit": plan["checkout_identity"][
                "expected_commit"
            ],
        }
        spec_raw = (json.dumps(spec, indent=2) + "\n").encode()
        _write_new_atomic(Path(job["job_spec"]), spec_raw)
        spec_sha = hashlib.sha256(spec_raw).hexdigest()
        export = (
            "HOME=" + os.environ.get("HOME", str(Path.home()))
            + ",USER=" + os.environ.get("USER", "")
            + ",PATH=/usr/local/bin:/usr/bin:/bin"
            + ",EVSP_DR_ROOT=" + str(REPO_ROOT)
            + ",EVSP_EXPECTED_COMMIT="
            + plan["checkout_identity"]["expected_commit"]
            + ",EVSP_MIP_EXPECTED_WORKER_SHA256="
            + plan["worker_sha256"]
        )
        wall_hours = job["budget_hours"]
        command = [
            "sbatch", "--parsable", "--partition=scaglione",
            "--no-requeue", "--signal=B:USR1@180",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
            f"--time={wall_hours:02d}:10:00",
            f"--job-name={job['job_name']}",
            f"--output={logs}/%x_%j.out",
            f"--error={logs}/%x_%j.err",
            "--export=" + export,
            str(worker), job["job_spec"], spec_sha,
        ]
        job["submission_state"] = "attempting"
        manifest = {
            **plan,
            "approval_sha256": plan_sha,
            "submitted": False,
        }
        _replace_json(root / "campaign.json", manifest)
        completed = subprocess.run(
            command, cwd=REPO_ROOT, text=True,
            capture_output=True, check=False,
        )
        if completed.returncode != 0:
            job["submission_state"] = "failed"
            job["submission_error"] = (
                completed.stderr or completed.stdout
            ).strip()
            _replace_json(root / "campaign.json", manifest)
            raise SystemExit(f"{job['cell_id']}: sbatch failed")
        job_id = completed.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise SystemExit("sbatch returned an invalid job ID")
        job["job_id"] = job_id
        job["submission_state"] = "submitted"
        _replace_json(root / "campaign.json", manifest)
    manifest["submitted"] = True
    _replace_json(root / "campaign.json", manifest)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("inventory", "pilot", "secondary"),
        required=True,
    )
    parser.add_argument("--campaign")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--data-root", type=Path, action="append")
    parser.add_argument("--giro-start", action="append", default=[])
    parser.add_argument("--inventory-out", type=Path)
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.mode != "inventory" and not args.campaign:
        parser.error("pilot/secondary modes require --campaign")
    if args.submit and not args.approved_plan_sha256:
        parser.error("--submit requires --approved-plan-sha256")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    roots = dict(DEFAULT_ROOTS)
    roots.update(_parse_assignments(args.root))
    data_roots = args.data_root or [REPO_ROOT / "data"]
    payload = inventory(roots, data_roots=data_roots)
    if args.inventory_out:
        _write_new_atomic(
            args.inventory_out,
            (json.dumps(payload, indent=2) + "\n").encode(),
        )
    if args.mode == "inventory":
        print(json.dumps(payload, indent=2))
        return 0
    identity = checkout_identity(require_detached=args.submit)
    start_map = _parse_assignments(args.giro_start)
    plan = build_plan(
        payload,
        mode=args.mode,
        campaign=args.campaign,
        start_map=start_map,
        identity=identity,
    )
    plan_raw = _canonical(plan)
    plan_sha = hashlib.sha256(plan_raw).hexdigest()
    print(json.dumps(plan, indent=2))
    print(f"[approval-sha256] {plan_sha}")
    if args.plan_out:
        _write_new_atomic(args.plan_out, plan_raw)
    if not args.submit:
        print("[dry-run] no Slurm jobs submitted")
        return 0
    if args.approved_plan_sha256 != plan_sha:
        raise SystemExit("current plan differs from approved SHA-256")
    _stage_and_submit(plan, plan_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

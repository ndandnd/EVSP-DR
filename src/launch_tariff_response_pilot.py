#!/usr/bin/env python3
"""Build and optionally submit the hash-approved tariff-response pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from build_tariff_response_manifest import REPO_ROOT, sha256_file
from tariff_response_core import (
    PHYSICS,
    giro_routes_for_instance,
    load_tariff_manifest,
)


SCHEMA = "evsp-dr-tariff-response-pilot-plan-v1"
REVIEWED_BASE = "636dc0912f47e6ce85284fad3b36af30b4135887"
FROZEN_K40_INSTANCE_SHA256 = (
    "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd"
)
WORKER = REPO_ROOT / "src/submit_tariff_response_pilot.sub"
CODE_PATHS = (
    "src/launch_tariff_response_pilot.py",
    "src/submit_tariff_response_pilot.sub",
    "src/build_tariff_response_manifest.py",
    "src/tariff_response_core.py",
    "src/fixed_duty_expanded_optimizer.py",
    "src/prepare_tariff_fixed_duty_seed.py",
    "src/run_fixed_giro_tariff_response.py",
    "src/exact_pricer_expanded.py",
    "src/run_exact_pool_mip.py",
    "src/expanded_path_realization.py",
    "src/audit_giro_known_columns.py",
    "src/mip_convergence.py",
    "src/durable_io.py",
    "src/master_lp_scipy.py",
    "src/utils_v2.py",
    "src/config.py",
    "src/tariff_response_environment.py",
    "src/assemble_tariff_response_campaign.py",
    "src/validate_tariff_response_archive.py",
    "src/reconcile_tariff_response_gate.py",
)
MATRIX_FIELDS = (
    "job_key", "phase", "scale", "tariff_id", "treatment",
    "partition", "threads", "wall_limit_s", "solver_limit_s",
    "separate_k40_gate", "dependency_key", "output",
)
TARIFF_CODES = {
    "flat": "fl",
    "peak08": "p8",
    "peak12": "p12",
    "peak18": "p18",
    "sek": "sek",
    "solar_parx_midday_free": "sol",
    "peak12_alpha_0p0": "a0",
    "peak12_alpha_0p25": "a25",
    "peak12_alpha_0p5": "a5",
    "peak12_alpha_1p0": "a1",
    "peak12_alpha_2p0": "a2",
}


def canonical(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def assignments(values):
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"assignment lacks '=': {value}")
        key, raw = value.split("=", 1)
        if key in parsed or not key:
            raise ValueError(f"duplicate/empty assignment: {key}")
        parsed[key] = raw
    return parsed


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
        raise ValueError(
            "checkout must be exact, tracked-clean, reviewed, and detached "
            "for submission"
        )
    return {
        "commit": head.stdout.strip(),
        "detached": symbolic.returncode == 1,
        "tracked_clean": True,
        "reviewed_base": REVIEWED_BASE,
    }


def _job_name(phase, scale, treatment, tariff_id):
    phase_code = {
        "FIXED_FULL": "F",
        "SEED": "S",
        "CG": "C",
        "MIP": "M",
    }[phase]
    treatment_code = {
        "FIXED": "F",
        "RAW": "R",
        "GIRO-AUGMENTED": "G",
        "GIRO40-AUGMENTED": "G",
    }[treatment]
    tariff_code = TARIFF_CODES[tariff_id]
    name = f"T{phase_code}{scale}{treatment_code}{tariff_code}"
    if len(name) > 15:
        raise ValueError(f"job name too long: {name}")
    return name


def _job(
    *,
    campaign_root,
    phase,
    scale,
    tariff,
    treatment,
    separate_k40_gate=False,
    dependency_key=None,
):
    key = (
        f"{phase.lower()}_k{scale}_{treatment.lower().replace('-', '_')}_"
        f"{tariff['tariff_id']}"
    )
    if phase == "FIXED_FULL":
        key = "fixed_full_giro40_all_tariffs"
    partition = "scaglione" if phase == "MIP" else "default_partition"
    threads = 8 if phase == "MIP" else 2
    wall_limit_s = (
        8 * 3600 if scale == 40 and phase == "CG"
        else 4 * 3600 if scale == 8 and phase in {"CG", "MIP"}
        else 2 * 3600 if phase in {"CG", "MIP"}
        else 4 * 3600
    )
    solver_limit_s = wall_limit_s if phase == "MIP" else None
    output = (
        campaign_root / "outputs" / key
        if phase == "FIXED_FULL"
        else campaign_root / "outputs" / f"{key}.json"
    )
    return {
        "job_key": key,
        "job_name": _job_name(
            phase, scale, treatment,
            "flat" if phase == "FIXED_FULL" else tariff["tariff_id"],
        ),
        "phase": phase,
        "scale": scale,
        "tariff_id": (
            "ALL" if phase == "FIXED_FULL" else tariff["tariff_id"]
        ),
        "tariff_sha256": (
            None if phase == "FIXED_FULL" else tariff["sha256"]
        ),
        "tariff_relative_path": (
            None if phase == "FIXED_FULL"
            else tariff["relative_path"].removeprefix("data/")
        ),
        "treatment": treatment,
        "partition": partition,
        "threads": threads,
        "wall_limit_s": wall_limit_s,
        "solver_limit_s": solver_limit_s,
        "separate_k40_gate": separate_k40_gate,
        "dependency_key": dependency_key,
        "output": str(output),
        "progress_dir": (
            str(campaign_root / "progress" / key)
            if phase == "MIP" else None
        ),
        "phase_telemetry": (
            str(campaign_root / "telemetry" / f"{key}.jsonl")
            if phase == "CG" else None
        ),
    }


def build_plan(
    *,
    campaign,
    instance_paths,
    instance_hashes,
    tariff_manifest,
    identity,
    reservation_root,
    python_path,
    results_root=None,
):
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,79}", campaign):
        raise ValueError("unsafe campaign name")
    if set(instance_paths) != {"k5", "k8", "k40"}:
        raise ValueError("exact k5/k8/k40 instance paths are required")
    if set(instance_hashes) != set(instance_paths):
        raise ValueError("every instance requires a SHA-256")
    instances = {}
    for key in ("k5", "k8", "k40"):
        path = Path(instance_paths[key]).expanduser().resolve()
        expected = instance_hashes[key]
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected)
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise ValueError(f"{key} instance hash mismatch")
        expected_duties = int(key[1:])
        routes = giro_routes_for_instance(
            REPO_ROOT / "data/Par_VehicleDetails_Updated.csv",
            path,
        )
        if len(routes) != expected_duties:
            raise ValueError(
                f"{key} instance has {len(routes)} GIRO duties"
            )
        if key == "k40" and expected != FROZEN_K40_INSTANCE_SHA256:
            raise ValueError("k40 is not the frozen reviewed instance")
        instances[key] = {
            "path": str(path),
            "sha256": expected,
            "relative_path":
                f"tariff_response_inputs/{key}_{expected[:12]}.csv",
            "duty_count": len(routes),
            "trip_count": sum(len(route["trips"]) for route in routes),
        }
    tariff_manifest = tariff_manifest.expanduser().resolve()
    tariffs = load_tariff_manifest(tariff_manifest)
    if set(TARIFF_CODES) != {row["tariff_id"] for row in tariffs}:
        raise ValueError("pilot tariff set differs")
    python_path = python_path.expanduser().resolve()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    version = subprocess.run(
        [
            str(python_path), "-I", "-B",
            str(REPO_ROOT / "src/tariff_response_environment.py"),
        ],
        text=True, capture_output=True, check=False, env=environment,
    )
    if (
        version.returncode != 0
        or not python_path.is_file()
    ):
        raise ValueError("pilot Python/Gurobi environment is unavailable")
    try:
        environment_identity = json.loads(version.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("pilot environment identity is invalid") from exc
    if (
        not str(environment_identity.get("python", "")).startswith("3.12.")
        or not python_path.is_file()
    ):
        raise ValueError("pilot requires an explicit Python 3.12")
    namespace = Path(
        results_root
        if results_root is not None
        else REPO_ROOT / "src/results/tariff_response"
    ).resolve()
    root = (namespace / campaign).resolve()
    k40_root = (
        namespace / f"{campaign}-k40prep"
    ).resolve()
    jobs = []
    jobs.append(_job(
        campaign_root=root,
        phase="FIXED_FULL",
        scale=40,
        tariff=next(row for row in tariffs if row["tariff_id"] == "flat"),
        treatment="FIXED",
    ))
    for scale in (5, 8):
        for tariff in tariffs:
            seed = _job(
                campaign_root=root,
                phase="SEED",
                scale=scale,
                tariff=tariff,
                treatment="GIRO-AUGMENTED",
            )
            jobs.append(seed)
            for treatment in ("RAW", "GIRO-AUGMENTED"):
                cg = _job(
                    campaign_root=root,
                    phase="CG",
                    scale=scale,
                    tariff=tariff,
                    treatment=treatment,
                    dependency_key=(
                        seed["job_key"]
                        if treatment == "GIRO-AUGMENTED" else None
                    ),
                )
                jobs.append(cg)
                jobs.append(_job(
                    campaign_root=root,
                    phase="MIP",
                    scale=scale,
                    tariff=tariff,
                    treatment=treatment,
                    dependency_key=cg["job_key"],
                ))
    for tariff in tariffs:
        seed = _job(
            campaign_root=k40_root,
            phase="SEED",
            scale=40,
            tariff=tariff,
            treatment="GIRO40-AUGMENTED",
            separate_k40_gate=True,
        )
        jobs.append(seed)
        for treatment in ("RAW", "GIRO40-AUGMENTED"):
            jobs.append(_job(
                campaign_root=k40_root,
                phase="CG",
                scale=40,
                tariff=tariff,
                treatment=treatment,
                separate_k40_gate=True,
                dependency_key=(
                    seed["job_key"]
                    if treatment == "GIRO40-AUGMENTED" else None
                ),
            ))
    campaign_nonce = hashlib.sha256(campaign.encode()).hexdigest()[:2]
    for job in jobs:
        job["job_name"] = f"{job['job_name']}{campaign_nonce}"
        if len(job["job_name"]) > 15:
            raise ValueError("campaign-specific job name exceeds 15 chars")
    if len({job["job_key"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate pilot jobs")
    code_hashes = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in CODE_PATHS
    }
    job_by_key = {job["job_key"]: job for job in jobs}
    for job in jobs:
        dependency = job["dependency_key"]
        if dependency and dependency not in job_by_key:
            raise ValueError("job dependency is missing")
        instance = dict(instances[f"k{job['scale']}"])
        instance["source_path"] = instance["path"]
        job_root = k40_root if job["separate_k40_gate"] else root
        instance["path"] = str(
            job_root / "input/instances"
            / Path(instance["relative_path"]).name
        )
        job["instance"] = instance
        job["seed_output"] = (
            job_by_key[dependency]["output"]
            if job["treatment"] in {
                "GIRO-AUGMENTED", "GIRO40-AUGMENTED"
            } and dependency and job["phase"] == "CG"
            else next((
                candidate["output"] for candidate in jobs
                if candidate["phase"] == "SEED"
                and candidate["scale"] == job["scale"]
                and candidate["tariff_id"] == job["tariff_id"]
            ), None)
            if job["phase"] == "MIP"
            and job["treatment"] != "RAW"
            else None
        )
        job["source_cg_output"] = (
            job_by_key[dependency]["output"]
            if job["phase"] == "MIP" else None
        )
        execution_identity = {
            key: job[key] for key in (
                "phase", "scale", "tariff_id", "tariff_sha256",
                "treatment", "wall_limit_s", "solver_limit_s",
                "separate_k40_gate", "instance", "seed_output",
                "source_cg_output",
            )
        }
        execution_identity["code_sha256"] = code_hashes
        job["execution_digest"] = hashlib.sha256(
            canonical(execution_identity)
        ).hexdigest()
    return {
        "schema": SCHEMA,
        "campaign": campaign,
        "campaign_root": str(root),
        "k40_campaign_root": str(k40_root),
        "checkout_identity": identity,
        "physics": PHYSICS,
        "tariff_manifest": str(tariff_manifest),
        "tariff_manifest_sha256": sha256_file(tariff_manifest),
        "instances": instances,
        "reservation_root": str(reservation_root.expanduser().resolve()),
        "worker": str(WORKER),
        "worker_sha256": sha256_file(WORKER),
        "code_sha256": code_hashes,
        "python": {
            "path": str(python_path),
            "sha256": sha256_file(python_path),
            "version": environment_identity["python"],
        },
        "environment_identity": environment_identity,
        "jobs": jobs,
        "main_submission_job_count": sum(
            not job["separate_k40_gate"] for job in jobs
        ),
        "k40_preparation_job_count": sum(
            job["separate_k40_gate"] for job in jobs
        ),
        "k40_mip_submission_allowed": False,
        "continuous_cost_pricing_certified": False,
    }


def write_matrix(plan, path):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MATRIX_FIELDS, extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sorted(
            plan["jobs"], key=lambda job: job["job_key"]
        ))


def _reserve(plan, plan_sha, selected):
    root = Path(plan["reservation_root"])
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for job in selected:
            path = root / f"{job['execution_digest']}.json"
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
            )
            with os.fdopen(descriptor, "w") as handle:
                json.dump({
                    "schema": "evsp-dr-tariff-response-reservation-v1",
                    "plan_sha256": plan_sha,
                    "job_key": job["job_key"],
                }, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            paths.append(path)
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    return paths


def submit(plan, plan_sha, *, k40_preparation):
    root = Path(
        plan["k40_campaign_root"]
        if k40_preparation else plan["campaign_root"]
    )
    if root.exists():
        raise ValueError("campaign output already exists")
    observed = checkout_identity(require_detached=True)
    if observed != plan["checkout_identity"]:
        raise ValueError("submission checkout differs from plan")
    selected = [
        job for job in plan["jobs"]
        if bool(job["separate_k40_gate"]) == bool(k40_preparation)
    ]
    if not selected:
        raise ValueError("submission selection is empty")
    reservations = _reserve(plan, plan_sha, selected)
    root.mkdir(parents=True)
    logs = root / "logs"
    logs.mkdir()
    for job in selected:
        source = Path(job["instance"]["source_path"])
        staged = Path(job["instance"]["path"])
        _copy_new(source, staged, job["instance"]["sha256"])
    tariff_input = root / "input/tariffs"
    _copy_new(
        Path(plan["tariff_manifest"]),
        tariff_input / "tariff_manifest.csv",
        plan["tariff_manifest_sha256"],
    )
    for tariff in load_tariff_manifest(Path(plan["tariff_manifest"])):
        source = REPO_ROOT / tariff["relative_path"]
        _copy_new(
            source,
            tariff_input / Path(tariff["relative_path"]).name,
            tariff["sha256"],
        )
    plan_path = root / "approved-plan.json"
    _write_new_file(plan_path, canonical(plan))
    manifest = {
        **plan,
        "approval_sha256": plan_sha,
        "submission_scope": (
            "k40_preparation_only" if k40_preparation
            else "main_k5_k8_pilot"
        ),
        "submitted_jobs": [],
        "reservations": [str(path) for path in reservations],
    }
    for reservation in reservations:
        _copy_new(
            reservation,
            root / "input/reservations" / reservation.name,
            sha256_file(reservation),
        )
    manifest_path = root / "campaign.json"
    _write_manifest(
        manifest_path, manifest
    )
    job_ids = {}
    gate = subprocess.run(
        [
            "sbatch", "--parsable", "--hold",
            "--partition=default_partition", "--time=00:05:00",
            f"--job-name=TRG{plan_sha[:6]}",
            f"--output={logs}/gate_%j.out",
            f"--error={logs}/gate_%j.err",
            "--wrap=/bin/true",
        ],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    gate_id = gate.stdout.strip().split(";", 1)[0]
    if gate.returncode != 0 or not gate_id.isdigit():
        manifest["gate_state"] = "ambiguous_held_gate"
        manifest["gate_error"] = (gate.stderr or gate.stdout).strip()
        _write_manifest(manifest_path, manifest)
        raise RuntimeError(
            "held submission gate outcome is ambiguous; reconcile before retry"
        )
    manifest["gate_job_id"] = gate_id
    manifest["gate_state"] = "held"
    _write_manifest(manifest_path, manifest)
    try:
        for job in selected:
            dependency = job["dependency_key"]
            command = [
                "sbatch", "--parsable",
                f"--partition={job['partition']}",
                "--requeue" if job["phase"] == "CG" else "--no-requeue",
                f"--cpus-per-task={job['threads']}",
                "--mem=48G",
                f"--time={max(1, math_ceil(job['wall_limit_s']/60)+10)}",
                f"--job-name={job['job_name']}",
                f"--comment=TRSP:{job['execution_digest'][:28]}",
                f"--output={logs}/%x_%j.out",
                f"--error={logs}/%x_%j.err",
            ]
            if job["phase"] in {"CG", "MIP"}:
                command.append("--signal=B:USR1@180")
            dependency_ids = [gate_id]
            if dependency:
                if dependency not in job_ids:
                    raise ValueError("dependency was not submitted first")
                dependency_ids.append(job_ids[dependency])
            command.append(
                "--dependency=afterok:" + ":".join(dependency_ids)
            )
            command.extend([
                str(WORKER), str(plan_path), plan_sha, job["job_key"],
                plan["python"]["path"], plan["python"]["sha256"],
            ])
            completed = subprocess.run(
                command, cwd=REPO_ROOT, text=True,
                capture_output=True, check=False,
            )
            job_id = completed.stdout.strip().split(";", 1)[0]
            if completed.returncode != 0 or not job_id.isdigit():
                raise RuntimeError(
                    "sbatch outcome ambiguous; reservations remain"
                )
            job_ids[job["job_key"]] = job_id
            manifest["submitted_jobs"].append({
                "job_key": job["job_key"], "job_id": job_id,
            })
            _write_manifest(manifest_path, manifest)
    except Exception as exc:
        manifest["gate_state"] = "held_after_partial_submission"
        manifest["submission_error"] = repr(exc)
        _write_manifest(manifest_path, manifest)
        raise
    manifest["gate_state"] = "release_attempting"
    _write_manifest(manifest_path, manifest)
    release = subprocess.run(
        ["scontrol", "release", gate_id],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    if release.returncode != 0:
        manifest["gate_state"] = "held_release_failed"
        manifest["gate_error"] = (
            release.stderr or release.stdout
        ).strip()
        _write_manifest(manifest_path, manifest)
        raise RuntimeError(
            "submission gate release failed; experiment jobs remain blocked"
        )
    manifest["gate_state"] = "released"
    _write_manifest(manifest_path, manifest)
    return manifest


def _write_manifest(path, payload):
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _copy_new(source, target, expected_sha256):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != expected_sha256:
            raise ValueError(f"staged input hash mismatch: {target}")
        return
    descriptor = os.open(
        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
    )
    with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    if sha256_file(target) != expected_sha256:
        raise ValueError(f"staged input copy mismatch: {target}")


def _write_new_file(path, payload):
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def math_ceil(value):
    import math
    return int(math.ceil(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--instance", action="append", default=[])
    parser.add_argument("--instance-sha256", action="append", default=[])
    parser.add_argument(
        "--tariff-manifest", type=Path,
        default=REPO_ROOT / "data/tariff_response/tariff_manifest.csv",
    )
    parser.add_argument("--reservation-root", type=Path, required=True)
    parser.add_argument(
        "--python", type=Path, default=Path(sys.executable)
    )
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--matrix-out", type=Path)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--submit-k40-preparation", action="store_true")
    args = parser.parse_args(argv)
    if args.submit and not args.approved_plan_sha256:
        parser.error("--submit requires --approved-plan-sha256")
    if args.submit_k40_preparation and not args.submit:
        parser.error("--submit-k40-preparation requires --submit")
    identity = checkout_identity(require_detached=args.submit)
    plan = build_plan(
        campaign=args.campaign,
        instance_paths=assignments(args.instance),
        instance_hashes=assignments(args.instance_sha256),
        tariff_manifest=args.tariff_manifest,
        identity=identity,
        reservation_root=args.reservation_root,
        python_path=args.python,
        results_root=None,
    )
    plan_raw = canonical(plan)
    plan_sha = hashlib.sha256(plan_raw).hexdigest()
    if args.plan_out:
        if args.plan_out.exists():
            raise FileExistsError(args.plan_out)
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_bytes(plan_raw)
    if args.matrix_out:
        write_matrix(plan, args.matrix_out)
    print(json.dumps(plan, indent=2))
    print(f"[approval-sha256] {plan_sha}")
    if not args.submit:
        print("[dry-run] no Slurm jobs submitted")
        return 0
    if args.approved_plan_sha256 != plan_sha:
        raise ValueError("current plan differs from approved SHA-256")
    submit(
        plan, plan_sha,
        k40_preparation=args.submit_k40_preparation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

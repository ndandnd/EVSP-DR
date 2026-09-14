#!/usr/bin/env python3
"""Validate, run, and collect the frozen k1 pricing-boundary campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "manifest.json"
DRIVER_COMMIT = "309d98d266ebaf6b7e99543a67f8f2be5736874a"
GUROBI_LICENSE = "/share/apps/software/gurobi/gurobi.lic"
WATCHDOG_GRACE_S = 120


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text())


def git_value(code_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(code_root), *arguments],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def validate_manifest(manifest: dict, code_root: Path) -> dict:
    errors = []
    cases = manifest.get("cases", [])
    if not 8 <= len(cases) <= 16:
        errors.append("campaign must contain 8--16 cases")
    indices = [case.get("index") for case in cases]
    if indices != list(range(len(cases))):
        errors.append("case indices must be contiguous and ordered")
    identifiers = [case.get("case_id") for case in cases]
    if len(set(identifiers)) != len(identifiers):
        errors.append("case identifiers must be unique")
    if manifest.get("code", {}).get("commit") != DRIVER_COMMIT:
        errors.append("manifest is not pinned to the dedicated capacity driver")
    if not manifest.get("code", {}).get("generic_trip_set_mip_forbidden"):
        errors.append("generic trip-set MIP prohibition is missing")
    policy = manifest.get("resource_policy", {})
    if policy.get("partition") != "default_partition":
        errors.append("capacity CG must use default_partition")
    if "scaglione-compute-01" not in policy.get("exclude", []):
        errors.append("reserved node exclusion is missing")
    if policy.get("submission_authorized") is not True:
        errors.append("frozen launch authorization is missing")
    if manifest.get("common_model", {}).get("terminal_65_percent_floor") is not False:
        errors.append("campaign must not infer a 65% terminal floor")
    for case in cases:
        if case["cg_wall_s"] + case["mip_wall_s"] > case["slurm_time_s"] - 300:
            errors.append(f"{case['case_id']} lacks at least 300 s wrapper margin")
        if case["capacity_selector"] not in {"reference", "prefix-memo"}:
            errors.append(f"{case['case_id']} has invalid capacity selector")
        if case["arm"] not in {"baseline", "capacity", "parx60", "combined"}:
            errors.append(f"{case['case_id']} has invalid arm")
        if not 3600 <= case["slurm_time_s"] <= 14400:
            errors.append(f"{case['case_id']} is outside the 1--4 h allocation range")
    input_checks = {}
    for name, spec in manifest.get("source_inputs", {}).items():
        path = code_root / spec["path"]
        observed = sha256_file(path) if path.is_file() else None
        input_checks[name] = {
            "path": str(path), "expected_sha256": spec["sha256"],
            "observed_sha256": observed, "valid": observed == spec["sha256"],
        }
        if observed != spec["sha256"]:
            errors.append(f"source hash mismatch: {name}")
    return {
        "valid": not errors,
        "errors": errors,
        "case_count": len(cases),
        "input_checks": input_checks,
    }


def validate_checkout(code_root: Path, result: dict) -> dict:
    try:
        observed_commit = git_value(code_root, "rev-parse", "HEAD")
        tracked_status = git_value(
            code_root, "status", "--porcelain", "--untracked-files=no",
        )
    except subprocess.CalledProcessError as exc:
        result["errors"].append(f"could not inspect code checkout: {exc}")
        result["valid"] = False
        return result
    result["code_checkout"] = {
        "root": str(code_root),
        "expected_commit": DRIVER_COMMIT,
        "observed_commit": observed_commit,
        "tracked_clean": not bool(tracked_status),
    }
    if observed_commit != DRIVER_COMMIT:
        result["errors"].append(
            f"code commit mismatch: {observed_commit} != {DRIVER_COMMIT}"
        )
    if tracked_status:
        result["errors"].append("capacity driver checkout is tracked-dirty")
    result["valid"] = not result["errors"]
    return result


def driver_parse_command(code_root: Path, command: list[str]) -> list[str]:
    """Return a no-solve invocation of the driver's actual argument parser."""
    parser_source = (
        "import sys; sys.path.insert(0, 'src'); "
        "import run_capacity_speed_event_cg as d; "
        "d.parser().parse_args(sys.argv[1:])"
    )
    return [command[0], "-c", parser_source, *command[3:]]


def validate_driver_commands(manifest: dict, code_root: Path) -> dict:
    failures = []
    parsed = 0
    environment = execution_environment()
    for case in manifest["cases"]:
        commands = build_commands(
            manifest, code_root, Path("/tmp/capacity-command-parse") / case["case_id"],
            case,
        )
        for stage, command in commands.items():
            completed = subprocess.run(
                driver_parse_command(code_root, command), cwd=code_root,
                env=environment, capture_output=True, text=True, check=False,
            )
            if completed.returncode:
                failures.append({
                    "case_id": case["case_id"], "stage": stage,
                    "returncode": completed.returncode,
                    "command": shlex.join(command),
                    "stderr": completed.stderr[-2000:],
                })
            else:
                parsed += 1
    return {"valid": not failures, "parsed_command_count": parsed,
            "failures": failures}


def case_paths(manifest: dict, code_root: Path, case: dict) -> tuple[Path, Path]:
    inputs = manifest["source_inputs"]
    return (
        code_root / inputs[case["instance"]]["path"],
        code_root / inputs[case["prices"]]["path"],
    )


def build_commands(
    manifest: dict, code_root: Path, output_dir: Path, case: dict,
    *, python: str = sys.executable,
) -> dict:
    instance, prices = case_paths(manifest, code_root, case)
    common_model = manifest["common_model"]
    common = [
        python, "-u", "src/run_capacity_speed_event_cg.py",
        "--arm", case["arm"],
        "--instance", str(instance),
        "--prices", str(prices),
        "--reference-data-dir", str(code_root / "data"),
        "--expected-commit", DRIVER_COMMIT,
        "--require-clean",
        "--battery-kwh", str(case["battery_kwh"]),
        "--non-parx-kw", str(common_model["non_parx_kw"]),
        "--reserve-kwh", str(case["reserve_kwh"]),
        "--soc-step", str(common_model["soc_step_kwh"]),
        "--block-min", str(common_model["event_block_min"]),
        "--rc-eps", str(common_model["rc_eps"]),
        "--threads", str(common_model["threads"]),
    ]
    cg_status = output_dir / "cg.json"
    pool = output_dir / "pool.jsonl"
    cg = common + [
        "--mode", "cg",
        "--capacity-selector", case["capacity_selector"],
        "--max-iters", str(common_model["max_iters"]),
        "--cg-wall-s", str(case["cg_wall_s"]),
        "--out", str(cg_status),
        "--pool-out", str(pool),
    ]
    mip = common + [
        "--mode", "mip",
        "--mip-gap", str(common_model["mip_gap"]),
        "--mip-wall-s", str(case["mip_wall_s"]),
        "--out", str(output_dir / "mip.json"),
        "--pool", str(pool),
        "--cg-status", str(cg_status),
    ]
    return {"cg": cg, "mip": mip}


def execution_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "GRB_LICENSE_FILE": GUROBI_LICENSE,
    })
    for inherited in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LM_LICENSE_FILE"):
        environment.pop(inherited, None)
    return environment


def artifact_record(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def read_json_if_valid(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def collect_attempt(case: dict, attempt_dir: Path) -> dict:
    worker = read_json_if_valid(attempt_dir / "worker_status.json")
    allocation = read_json_if_valid(attempt_dir / "allocation.json")
    cg = read_json_if_valid(attempt_dir / "cg.json")
    mip = read_json_if_valid(attempt_dir / "mip.json")
    mip_result = (mip or {}).get("result", {})
    stage1 = mip_result.get("stage1", {})
    stage2 = mip_result.get("stage2", {})
    stages = worker.get("stages", {}) if isinstance(worker, dict) else {}
    cg_stage_ok = stages.get("cg", {}).get("returncode") == 0
    mip_stage_ok = stages.get("mip", {}).get("returncode") == 0
    return {
        "case_id": case["case_id"],
        "case_index": case["index"],
        "attempt_id": attempt_dir.name,
        "expected": case,
        "worker_returncode": (
            worker.get("returncode") if isinstance(worker, dict) else None
        ),
        "stage_status": stages,
        "manifest_sha256": (
            worker.get("manifest_sha256") if isinstance(worker, dict) else None
        ),
        "code_commit": (
            allocation.get("code_commit") if isinstance(allocation, dict) else None
        ),
        "input_validation": (
            allocation.get("input_validation")
            if isinstance(allocation, dict) else None
        ),
        "native_record_eligibility": {
            "cg": bool(cg_stage_ok and cg is not None),
            "mip": bool(mip_stage_ok and mip is not None),
        },
        "cg": {
            "artifact_complete": cg is not None,
            "status": (cg or {}).get("status"),
            "stop_reason": (cg or {}).get("stop_reason"),
            "certified_rc_optimal": (cg or {}).get("certified_rc_optimal"),
            "terminal_exact_min_reduced_cost":
                (cg or {}).get("terminal_exact_min_reduced_cost"),
            "runtime_s": (cg or {}).get("runtime_s"),
            "completed_iterations": len((cg or {}).get("iterations", [])),
            "final": (cg or {}).get("final"),
            "pricing_certificate_scope":
                (cg or {}).get("pricing_certificate_scope"),
            "pool_sha256": (cg or {}).get("pool_sha256"),
            "checkpoint": (cg or {}).get("checkpoint"),
        },
        "finite_pool_mip": {
            "artifact_complete": mip is not None,
            "pool_acceptance": (mip or {}).get("pool_acceptance"),
            "status": mip_result.get("status"),
            "has_solution": mip_result.get("has_solution"),
            "fleet": mip_result.get("fleet"),
            "charging_related_cost": mip_result.get("charging_related_cost"),
            "stage1_fleet_proven": stage1.get("fleet_proven"),
            "stage1_fleet_integer_lower_bound":
                stage1.get("fleet_integer_lower_bound"),
            "stage2_status": stage2.get("status"),
            "physical_station_capacity_audit":
                (mip or {}).get("physical_station_capacity_audit"),
            "duplicate_service_audit":
                (mip or {}).get("duplicate_service_audit"),
        },
        "artifacts": [
            artifact_record(attempt_dir / name)
            for name in (
                "commands.json", "allocation.json", "cg.json", "pool.jsonl",
                "mip.json", "worker_status.json",
            )
        ],
    }


def compact_cg(row: dict) -> dict:
    path = next(item["path"] for item in row["artifacts"]
                if item["path"].endswith("/cg.json"))
    native = read_json_if_valid(Path(path)) or {}
    result = {
        key: native.get(key) for key in (
            "schema", "mode", "arm", "capacity_selector", "status",
            "stop_reason", "certified_rc_optimal",
            "terminal_exact_min_reduced_cost", "runtime_s", "final",
            "checkpoint", "physics", "pricing_certificate_scope",
            "pool_sha256", "provenance",
        )
    }
    certified = native.get("certified_rc_optimal") is True
    return {
        "phase": "cg", "case_id": row["case_id"],
        "attempt_id": row["attempt_id"],
        "path": path,
        "sha256": next(item["sha256"] for item in row["artifacts"]
                       if item["path"].endswith("/cg.json")),
        "provisional": not certified,
        "result": result,
    }


def compact_mip(row: dict) -> dict:
    path = next(item["path"] for item in row["artifacts"]
                if item["path"].endswith("/mip.json"))
    native = read_json_if_valid(Path(path)) or {}
    solve_result = dict(native.get("result", {}))
    solve_result.pop("selected_indices", None)
    for stage_name in ("stage1", "stage2"):
        stage = dict(solve_result.get(stage_name, {}))
        stage.pop("selected_indices", None)
        solve_result[stage_name] = stage
    result = {
        key: native.get(key) for key in (
            "schema", "mode", "arm", "cg_status_sha256", "pool_sha256",
            "pool_acceptance", "physical_station_capacity_audit",
            "duplicate_service_audit", "capacity_enforced_in_mip",
            "provenance",
        )
    }
    result["result"] = solve_result
    stage1_proven = solve_result.get("stage1", {}).get("fleet_proven") is True
    stage2_optimal = solve_result.get("stage2", {}).get("status") == "OPTIMAL"
    return {
        "phase": "mip", "case_id": row["case_id"],
        "attempt_id": row["attempt_id"],
        "path": path,
        "sha256": next(item["sha256"] for item in row["artifacts"]
                       if item["path"].endswith("/mip.json")),
        "provisional": not (stage1_proven and stage2_optimal),
        "result": result,
    }


def collect_campaign(
    manifest: dict, campaign_root: Path, *, manifest_path: Path = MANIFEST_PATH,
) -> dict:
    rows = []
    for case in manifest["cases"]:
        case_root = campaign_root / "results" / case["case_id"]
        if not case_root.is_dir():
            continue
        rows.extend(
            collect_attempt(case, attempt)
            for attempt in sorted(case_root.iterdir())
            if attempt.is_dir()
        )
    cg_records = [compact_cg(row) for row in rows
                  if row["native_record_eligibility"]["cg"]]
    mip_records = [compact_mip(row) for row in rows
                   if row["native_record_eligibility"]["mip"]]
    return {
        "schema": "evsp-dr-strict-capacity-parallel-collection-v1",
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(campaign_root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "code_commit": manifest["code"]["commit"],
        "expected_case_count": len(manifest["cases"]),
        "observed_attempt_count": len(rows),
        "rows": rows,
        "root": str(campaign_root),
        "cg": cg_records,
        "mip": mip_records,
        "records": [*cg_records, *mip_records],
        "workflow": {
            "manifest.json": str(manifest_path),
            "jobs.json": str(campaign_root / "jobs.json"),
            "attempt_progress": rows,
        },
        "interpretation": manifest["reporting_contract"],
        "artifact_hash_note": (
            "Hashes are observations at collection time; only completed "
            "attempt artifacts are immutable result evidence."
        ),
    }


def run_worker(args) -> int:
    manifest = load_manifest(args.manifest)
    code_root = args.code_root.resolve(strict=True)
    validation = validate_checkout(
        code_root, validate_manifest(manifest, code_root),
    )
    if not validation["valid"]:
        raise RuntimeError(json.dumps(validation, sort_keys=True))
    observed_commit = validation["code_checkout"]["observed_commit"]
    case = manifest["cases"][args.index]
    if case["index"] != args.index:
        raise RuntimeError("manifest case/index mismatch")
    output_dir = args.campaign_root.resolve() / "results" / case["case_id"] / args.attempt_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite attempt: {output_dir}")
    output_dir.mkdir(parents=True)
    commands = build_commands(
        manifest, code_root, output_dir, case, python=args.python,
    )
    atomic_json(output_dir / "commands.json", commands)
    atomic_json(output_dir / "allocation.json", {
        "case": case,
        "attempt_id": args.attempt_id,
        "slurm": {
            key: value for key, value in os.environ.items()
            if key.startswith("SLURM_")
        },
        "code_commit": observed_commit,
        "input_validation": validation["input_checks"],
    })
    environment = execution_environment()
    stages = {}
    returncode = 0
    manifest_sha256 = sha256_file(args.manifest)
    commands_sha256 = sha256_file(output_dir / "commands.json")

    def write_status(status: str) -> None:
        artifacts = [
            artifact_record(output_dir / name)
            for name in (
                "commands.json", "allocation.json", "cg.json", "pool.jsonl",
                "mip.json", "cg.stdout.log", "cg.stderr.log",
                "mip.stdout.log", "mip.stderr.log",
            )
        ]
        atomic_json(output_dir / "worker_status.json", {
            "status": status,
            "case_id": case["case_id"],
            "attempt_id": args.attempt_id,
            "returncode": returncode,
            "stages": stages,
            "manifest_sha256": manifest_sha256,
            "commands_sha256": commands_sha256,
            "artifacts": artifacts,
            "interpretation": manifest["reporting_contract"],
        })

    write_status("running")
    for stage in ("cg", "mip"):
        stdout = output_dir / f"{stage}.stdout.log"
        stderr = output_dir / f"{stage}.stderr.log"
        with stdout.open("x") as out_handle, stderr.open("x") as err_handle:
            budget_s = case[f"{stage}_wall_s"]
            try:
                completed = subprocess.run(
                    commands[stage], cwd=code_root, env=environment,
                    stdout=out_handle, stderr=err_handle, check=False,
                    timeout=budget_s + WATCHDOG_GRACE_S,
                )
                stage_returncode = completed.returncode
                watchdog_expired = False
            except subprocess.TimeoutExpired:
                stage_returncode = 124
                watchdog_expired = True
        stages[stage] = {
            "returncode": stage_returncode,
            "requested_solver_wall_s": budget_s,
            "outer_watchdog_s": budget_s + WATCHDOG_GRACE_S,
            "outer_watchdog_expired": watchdog_expired,
            "stdout": artifact_record(stdout),
            "stderr": artifact_record(stderr),
        }
        if stage_returncode != 0:
            returncode = stage_returncode
            write_status("failed")
            break
        write_status("running" if stage == "cg" else "complete")
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    validate.add_argument("--code-root", type=Path, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    worker.add_argument("--campaign-root", type=Path, required=True)
    worker.add_argument("--code-root", type=Path, required=True)
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--attempt-id", required=True)
    worker.add_argument("--python", default=sys.executable)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    collect.add_argument("--campaign-root", type=Path, required=True)
    collect.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        code_root = args.code_root.resolve(strict=True)
        result = validate_checkout(
            code_root,
            validate_manifest(load_manifest(args.manifest), code_root),
        )
        if result["valid"]:
            command_validation = validate_driver_commands(
                load_manifest(args.manifest), code_root,
            )
            result["driver_command_validation"] = command_validation
            if not command_validation["valid"]:
                result["errors"].append("one or more driver commands did not parse")
                result["valid"] = False
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.command == "collect":
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite collection: {args.out}")
        payload = collect_campaign(
            load_manifest(args.manifest), args.campaign_root.resolve(),
            manifest_path=args.manifest.resolve(strict=True),
        )
        atomic_json(args.out.resolve(), payload)
        print(json.dumps({
            "out": str(args.out.resolve()),
            "observed_attempt_count": payload["observed_attempt_count"],
        }, sort_keys=True))
        return 0
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())

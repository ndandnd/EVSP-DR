#!/usr/bin/env python3
"""Run one matched one-hour MIP against a completed parent CG pool."""

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path


SOURCE_ROOT = Path("/home/nc437/ladder-lite/strict_capacity_parallel_20260914")
FOLLOW_ROOT = Path("/home/nc437/ladder-lite/strict_capacity_mip1h_20260914")
CODE_ROOT = SOURCE_ROOT / "code"
SOURCE_MANIFEST_SHA256 = "bce15219e18ec2b3e2cb9fac6ff47223b1cded4d9eb8934510cc91cd14811b62"


def load_campaign():
    spec = importlib.util.spec_from_file_location("strict_capacity_campaign", SOURCE_ROOT / "campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_arg(command, name, value):
    command[command.index(name) + 1] = str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    campaign = load_campaign()
    manifest = json.loads((FOLLOW_ROOT / "manifest.json").read_text())
    source_manifest = json.loads((SOURCE_ROOT / "manifest.json").read_text())
    if campaign.sha256_file(SOURCE_ROOT / "manifest.json") != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("source campaign manifest changed")
    case = manifest["cases"][args.index]
    if case["index"] != args.index:
        raise RuntimeError("case/index mismatch")
    parent = SOURCE_ROOT / "results" / case["case_id"] / f"{case['parent_job_id']}_r0"
    worker = json.loads((parent / "worker_status.json").read_text())
    if worker.get("manifest_sha256") != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("parent worker manifest hash mismatch")
    if worker.get("stages", {}).get("cg", {}).get("returncode") != 0:
        raise RuntimeError("parent CG stage was not successful")
    cg_path = parent / "cg.json"
    pool_path = parent / "pool.jsonl"
    cg = json.loads(cg_path.read_text())
    if cg.get("pool_sha256") != campaign.sha256_file(pool_path):
        raise RuntimeError("parent pool hash mismatch")
    attempt_id = f"{__import__('os').environ['SLURM_JOB_ID']}_r{__import__('os').environ.get('SLURM_RESTART_COUNT', '0')}"
    output = FOLLOW_ROOT / "results" / case["case_id"] / attempt_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    command_case = dict(source_manifest["cases"][args.index])
    command_case["mip_wall_s"] = 3600
    command = campaign.build_commands(
        source_manifest, CODE_ROOT, output, command_case,
        python="/home/nc437/evsp_env/bin/python",
    )["mip"]
    replace_arg(command, "--pool", pool_path)
    replace_arg(command, "--cg-status", cg_path)
    campaign.atomic_json(output / "allocation.json", {
        "schema": "evsp-dr-strict-capacity-mip1h-allocation-v1",
        "case": case,
        "source_parent": {
            "job_id": case["parent_job_id"],
            "attempt": str(parent),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "worker_status_sha256": campaign.sha256_file(parent / "worker_status.json"),
            "cg_status_sha256": campaign.sha256_file(cg_path),
            "pool_sha256": campaign.sha256_file(pool_path),
            "cg_certified_rc_optimal": cg.get("certified_rc_optimal"),
            "cg_stop_reason": cg.get("stop_reason"),
        },
        "command": command,
        "driver_commit": campaign.DRIVER_COMMIT,
    })
    with (output / "mip.stdout.log").open("x") as stdout, (output / "mip.stderr.log").open("x") as stderr:
        try:
            completed = subprocess.run(
                command, cwd=CODE_ROOT, env=campaign.execution_environment(),
                stdout=stdout, stderr=stderr, check=False, timeout=3720,
            )
            returncode = completed.returncode
            watchdog = False
        except subprocess.TimeoutExpired:
            returncode = 124
            watchdog = True
    campaign.atomic_json(output / "worker_status.json", {
        "schema": "evsp-dr-strict-capacity-mip1h-worker-v1",
        "case_id": case["case_id"], "attempt_id": attempt_id,
        "returncode": returncode,
        "manifest_sha256": campaign.sha256_file(FOLLOW_ROOT / "manifest.json"),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "stages": {"mip": {"returncode": returncode,
            "requested_solver_wall_s": 3600, "outer_watchdog_s": 3720,
            "outer_watchdog_expired": watchdog}},
        "artifacts": [campaign.artifact_record(output / name) for name in
            ("allocation.json", "mip.json", "mip.stdout.log", "mip.stderr.log")],
    })
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())

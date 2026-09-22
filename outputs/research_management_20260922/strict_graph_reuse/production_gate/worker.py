#!/usr/bin/env python3
"""Restart-safe worker for the authentic strict-pool lineage gate; no solvers."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    root = Path(sys.argv[1]).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    attempt = root / "attempts" / (os.environ["SLURM_JOB_ID"] + "_r" + os.environ.get("SLURM_RESTART_COUNT", "0"))
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    execution = {"audit_code_commit": manifest["audit_code_commit"],
        "model_execution_commit": manifest["model_execution_commit"],
        "manifest_sha256": sha(manifest_path), "job_id": os.environ["SLURM_JOB_ID"],
        "restart": os.environ.get("SLURM_RESTART_COUNT", "0"), "host": os.uname().nodename,
        "solver_started": False, "graph_built": False, "started_epoch": time.time()}
    save(attempt / "execution.json", execution)
    try:
        code = root / "code"
        if sha(root / "gate_source.tar") != manifest["gate_archive_sha256"]:
            raise ValueError("gate source archive checksum mismatch")
        if sha(root / "worker.py") != manifest["gate_source_sha256"]["scripts/strict_pool_lineage_worker.py"]:
            raise ValueError("worker checksum mismatch")
        for relative, expected in manifest["gate_source_sha256"].items():
            if sha(code / relative) != expected:
                raise ValueError("gate source checksum mismatch: " + relative)
        input_root = attempt / "input_artifacts"
        input_root.mkdir()
        if shutil.disk_usage(input_root).free < 2 * 1024**3:
            raise RuntimeError("less than 2 GiB headroom for authentic-input snapshots and output")
        copied = {}
        for key, item in manifest["input_files"].items():
            destination = input_root / item["local_name"]
            shutil.copyfile(item["path"], destination)
            actual = sha(destination)
            if actual != item["sha256"] or destination.stat().st_size != item["bytes"]:
                raise ValueError("original input changed or copy checksum mismatch: " + key)
            copied[key] = {"path": str(destination), "sha256": actual, "bytes": destination.stat().st_size}
        save(attempt / "staged_inputs.json", copied)
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, str(code / "scripts/audit_strict_pool_lineage.py"),
            "--manifest", str(manifest_path), "--input-root", str(input_root),
            "--model-root", manifest["model_source_root"], "--out", str(attempt / "gate_result.json")]
        save(attempt / "command.json", command)
        with (attempt / "gate.log").open("x") as log:
            process = subprocess.run(command, env=environment, cwd=code, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=1200)
        if process.returncode != 0:
            raise RuntimeError(f"lineage gate exited {process.returncode}; inspect gate.log")
        result = json.loads((attempt / "gate_result.json").read_text())
        expected = manifest["expected"]
        if (result["status"] != "passed" or result["inherited_routes_replayed"] != expected["inherited_routes"]
                or result["all_routes_physically_replayed"] != expected["child_routes"]
                or result["singleton_routes"] != expected["singleton_routes"]
                or result["solver_started"] or result["graph_built"]):
            raise ValueError("gate result scope/count mismatch")
        summary = {**execution, "status": "passed", "runtime_s": time.monotonic() - started,
            "child_maxrss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "gate_result_sha256": sha(attempt / "gate_result.json"), "log_sha256": sha(attempt / "gate.log"),
            "inputs_replayed": result["all_routes_physically_replayed"],
            "fresh_singleton_optimum_equality_proved": False,
            "scope": "authentic complete inherited-column equality and saved singleton replay only"}
        save(attempt / "result.json", summary)
        save(attempt / "COMPLETE.json", {"result_sha256": sha(attempt / "result.json"),
            "gate_result_sha256": summary["gate_result_sha256"], "manifest_sha256": sha(manifest_path),
            "audit_code_commit": manifest["audit_code_commit"], "solver_started": False, "graph_built": False})
    except BaseException as error:
        save(attempt / "FAILED.json", {**execution, "status": "failed", "error": str(error),
                                      "runtime_s": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()

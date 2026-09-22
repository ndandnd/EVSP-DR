#!/usr/bin/env python3
"""Solver-free persistence and lineage tests on Unicorn's actual home filesystem."""
import hashlib
import json
import os
from pathlib import Path
import re
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


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    root = Path(sys.argv[1]).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    attempt = root / "attempts" / (
        os.environ["SLURM_JOB_ID"] + "_r" + os.environ.get("SLURM_RESTART_COUNT", "0")
    )
    attempt.mkdir(parents=True, exist_ok=False)
    scratch = attempt / "nfs_tmp"
    scratch.mkdir()
    code = root / "code"
    started = time.monotonic()
    execution = {
        "execution_commit": manifest["execution_commit"],
        "manifest_sha256": sha(manifest_path),
        "host": os.uname().nodename,
        "job": os.environ["SLURM_JOB_ID"],
        "restart": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "tmpdir": str(scratch),
        "source_files_verified": False,
        "solver_started": False,
        "started_epoch": time.time(),
    }
    save(attempt / "execution.json", execution)
    for relative, expected in manifest["source_sha256"].items():
        if sha(code / relative) != expected:
            raise ValueError("source hash mismatch: " + relative)
    if sha(root / "worker.py") != manifest["worker_sha256"]:
        raise ValueError("worker hash mismatch")
    execution["source_files_verified"] = True
    execution["disk_free_bytes_before"] = shutil.disk_usage(scratch).free
    if execution["disk_free_bytes_before"] < 1024 ** 3:
        raise RuntimeError("less than 1 GiB filesystem headroom for tiny preflight")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(code / "src") + os.pathsep + str(code / "tests")
    environment["TMPDIR"] = str(scratch)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p",
               "test_strict_event_graph_cache.py", "-v"]
    execution["command"] = command
    save(attempt / "execution.json", execution)
    with (attempt / "tests.log").open("w") as log:
        result = subprocess.run(command, cwd=code, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, timeout=600)
    log_text = (attempt / "tests.log").read_text()
    match = re.search(r"Ran (\d+) tests? in ([0-9.]+)s", log_text)
    passed = result.returncode == 0 and match is not None and "\nOK\n" in log_text
    summary = {
        **execution, "status": "passed" if passed else "failed",
        "test_count": int(match.group(1)) if match else None,
        "unittest_elapsed_s": float(match.group(2)) if match else None,
        "returncode": result.returncode,
        "runtime_s": time.monotonic() - started,
        "child_maxrss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "log_sha256": sha(attempt / "tests.log"),
        "disk_free_bytes_after": shutil.disk_usage(scratch).free,
        "scope": "tiny synthetic same-physics cache/replay/lineage tests; no full k19 graph or CG",
    }
    save(attempt / "result.json", summary)
    if not passed:
        raise RuntimeError("native strict-graph preflight failed; inspect tests.log")
    save(attempt / "COMPLETE.json", {
        "result_sha256": sha(attempt / "result.json"),
        "execution_commit": manifest["execution_commit"],
        "manifest_sha256": sha(manifest_path),
        "test_count": summary["test_count"],
        "solver_started": False,
    })


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate portable environment identity and publish one probe artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tariff_response_environment import compare_portable, identity


def _write_new(path, payload):
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(encoded).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    temporary_sidecar = sidecar.with_name(
        f".{sidecar.name}.tmp.{os.getpid()}"
    )
    with temporary_sidecar.open("x") as handle:
        handle.write(f"{digest}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary_sidecar, sidecar)
    finally:
        temporary_sidecar.unlink(missing_ok=True)
    return digest


def probe(plan_path, plan_sha, probe_id, output):
    raw = plan_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan_sha:
        raise ValueError("approved plan hash mismatch")
    plan = json.loads(raw)
    observed = identity()
    differences = compare_portable(plan["python_identity"], observed)
    payload = {
        "schema": "evsp-dr-scale-ladder-environment-probe-v1",
        "probe_id": probe_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "plan_sha256": plan_sha,
        "compatible": not differences,
        "differences": differences,
        "planned_portable_identity_sha256":
            plan["python_identity"]["portable_identity_sha256"],
        "observed_portable_identity_sha256":
            observed["portable_identity_sha256"],
        "observed_node_metadata": observed["node_metadata"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = _write_new(output, payload)
    payload["artifact_sha256"] = digest
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = probe(
        args.plan, args.plan_sha256, args.probe_id, args.out
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["compatible"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate and portably publish one k40 factorial raw MIP result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from k40_factorial_mip_result import (
    enrich_result,
    publish_result_bundle,
    sha256_file,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out-bundle", type=Path, required=True)
    parser.add_argument("--source-status", type=Path, required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--job-spec-sha256", required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)
    if sha256_file(args.job_spec) != args.job_spec_sha256:
        raise SystemExit("job spec hash mismatch")
    spec = json.loads(args.job_spec.read_text())
    raw = json.loads(args.raw.read_text())
    source = json.loads(args.source_status.read_text())
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    recovery = {
        "job_spec_sha256": args.job_spec_sha256,
        "worker_sha256": args.worker_sha256,
        "original_job_id": args.job_id,
        "raw_sha256": sha256_file(args.raw),
        "recovery_commit": commit,
        "recovery_method": "worker_raw_result",
    }
    result = enrich_result(raw, spec=spec, recovery=recovery)
    publication = publish_result_bundle(
        args.out_bundle,
        result=result,
        spec=spec,
        source_status=source,
        recovery=recovery,
        allow_existing_incomplete=False,
    )
    print(json.dumps(publication, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

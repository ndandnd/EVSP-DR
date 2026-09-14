#!/usr/bin/env python3
"""Verify and compact the two native strict-capacity campaign roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SCHEMA = "evsp-dr-strict-capacity-parallel-collection-v1"
PYTHON = "/home/nc437/evsp_env/bin/python"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def invoke_native_collector(root: Path, kind: str, python: str) -> dict:
    command = [python, str(root / "collect.py"), "--campaign-root", str(root)]
    if kind == "pilot":
        command[2:2] = ["--manifest", str(root / "manifest.json")]
    process = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr[-4000:])
    data = json.loads(process.stdout)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"unexpected strict-capacity schema at {root}")
    return data


def artifact_map(marker: dict) -> dict:
    return {
        str(Path(item["path"]).resolve()): item
        for item in marker.get("artifacts", [])
        if isinstance(item, dict) and item.get("path")
    }


def verify_file(evidence: dict, path: Path, expected_sha: str | None = None) -> str:
    resolved = str(path.resolve())
    item = evidence.get(resolved)
    observed = sha256_file(path)
    if not item or item.get("exists") is not True or item.get("sha256") != observed:
        raise ValueError(f"worker artifact binding failed: {path}")
    if expected_sha is not None and observed != expected_sha:
        raise ValueError(f"collector artifact hash differs: {path}")
    return observed


def verified_pilot(root: Path, data: dict, manifest: dict,
                   manifest_sha: str) -> dict:
    cases = {case["case_id"]: case for case in manifest["cases"]}
    for record in data.get("records", []):
        path = Path(record["path"])
        attempt = path.parent
        marker = read_json(attempt / "worker_status.json")
        allocation = read_json(attempt / "allocation.json")
        evidence = artifact_map(marker)
        phase = record.get("phase")
        if not (marker.get("manifest_sha256") == manifest_sha == data.get("manifest_sha256")
                and marker.get("status") == "complete"
                and marker.get("returncode") == 0
                and marker.get("case_id") == record.get("case_id")
                and marker.get("attempt_id") == record.get("attempt_id")
                and marker.get("stages", {}).get(phase, {}).get("returncode") == 0
                and allocation.get("code_commit") == manifest["code"]["commit"]):
            raise ValueError(f"unverified strict-capacity pilot stage: {path}")
        verify_file(evidence, path, record.get("sha256"))
        verify_file(evidence, attempt / "allocation.json")
        commands_sha = verify_file(evidence, attempt / "commands.json")
        if commands_sha != marker.get("commands_sha256"):
            raise ValueError(f"commands hash differs from worker marker: {attempt}")
        for name, check in (allocation.get("input_validation") or {}).items():
            expected = manifest["source_inputs"][name]["sha256"]
            if not (check.get("valid") is True
                    and check.get("expected_sha256") == expected
                    and check.get("observed_sha256") == expected):
                raise ValueError(f"pilot input binding failed: {attempt}:{name}")
        case = cases[record["case_id"]]
        record.update(
            case_metadata=case,
            job_id=(allocation.get("slurm") or {}).get("SLURM_JOB_ID"),
            provisional=False,
            artifact_complete=True,
            stage_completion_verified=True,
        )
        if phase == "cg":
            row = next((row for row in data.get("rows", [])
                        if row.get("case_id") == record["case_id"]
                        and row.get("attempt_id") == record["attempt_id"]), {})
            record["result"]["iteration_count"] = (row.get("cg") or {}).get(
                "completed_iterations")
    return data


def verified_followup(root: Path, data: dict, manifest: dict,
                      manifest_sha: str) -> dict:
    source_root = Path(manifest["source_campaign"]["root"])
    source_manifest_sha = manifest["source_campaign"]["manifest_sha256"]
    if sha256_file(source_root / "manifest.json") != source_manifest_sha:
        raise ValueError("matched-MIP source manifest changed")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    for record in data.get("records", []):
        if record.get("phase") != "mip":
            raise ValueError("matched-MIP campaign emitted a new CG record")
        path = Path(record["path"])
        attempt = path.parent
        marker = read_json(attempt / "worker_status.json")
        allocation = read_json(attempt / "allocation.json")
        evidence = artifact_map(marker)
        case = cases[record["case_id"]]
        source = allocation.get("source_parent") or {}
        parent = (source_root / "results" / record["case_id"]
                  / case["source_attempt_id"])
        source_worker = read_json(parent / "worker_status.json")
        cg_path = parent / "cg.json"
        pool_path = parent / "pool.jsonl"
        cg = read_json(cg_path)
        result = record.get("result") or {}
        if not (marker.get("manifest_sha256") == manifest_sha == data.get("manifest_sha256")
                and marker.get("returncode") == 0
                and marker.get("case_id") == record.get("case_id")
                and marker.get("attempt_id") == record.get("attempt_id")
                and marker.get("stages", {}).get("mip", {}).get("returncode") == 0
                and source.get("job_id") == case["parent_job_id"]
                and Path(source.get("attempt", "")).resolve() == parent.resolve()
                and source.get("manifest_sha256") == source_manifest_sha
                and source_worker.get("manifest_sha256") == source_manifest_sha
                and source_worker.get("stages", {}).get("cg", {}).get("returncode") == 0
                and sha256_file(parent / "worker_status.json") == source.get("worker_status_sha256")
                and sha256_file(cg_path) == source.get("cg_status_sha256")
                and sha256_file(pool_path) == source.get("pool_sha256") == cg.get("pool_sha256")
                and result.get("cg_status_sha256") == source.get("cg_status_sha256")
                and result.get("pool_sha256") == source.get("pool_sha256")
                and allocation.get("driver_commit") == manifest["code"]["commit"]
                and (result.get("provenance") or {}).get("git_commit") == manifest["code"]["commit"]):
            raise ValueError(f"unverified matched one-hour MIP: {path}")
        verify_file(evidence, path, record.get("sha256"))
        verify_file(evidence, attempt / "allocation.json")
        record.update(
            case_metadata=case,
            job_id=record["attempt_id"].split("_", 1)[0],
            source_parent_job_id=case["parent_job_id"],
            source_cg_binding_verified=True,
            no_new_cg=True,
            provisional=False,
            artifact_complete=True,
            stage_completion_verified=True,
        )
    if data.get("cg"):
        raise ValueError("matched-MIP native collection contains CG endpoints")
    return data


def collect_strict_capacity(root: Path, kind: str, python: str = PYTHON) -> dict:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    data = invoke_native_collector(root, kind, python)
    if kind == "pilot":
        data = verified_pilot(root, data, manifest, manifest_sha)
    elif kind == "matched_mip1h":
        data = verified_followup(root, data, manifest, manifest_sha)
    else:
        raise ValueError(f"unknown strict-capacity kind: {kind}")
    data["cg"] = []
    data["mip"] = []
    data.setdefault("workflow", {})["manifest.json"] = manifest
    jobs = root / "jobs.json"
    if jobs.exists():
        data["workflow"]["jobs.json"] = read_json(jobs)
    data["adapter_kind"] = kind
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("pilot", "matched_mip1h"), required=True)
    parser.add_argument("--python", default=PYTHON)
    args = parser.parse_args()
    print(json.dumps(collect_strict_capacity(args.root, args.kind, args.python),
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

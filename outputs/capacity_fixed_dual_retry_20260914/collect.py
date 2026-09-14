#!/usr/bin/env python3
"""Standalone hash-verifying collector for fixed-dual diagnostic endpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import campaign


def collect(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = campaign.sha256_file(manifest_path)
    records, errors, progress = [], [], []
    for case_id, case in manifest["cases"].items():
        endpoint = root / "results" / case_id / "completion.json"
        attempts = []
        case_root = root / "results" / case_id
        for state_path in sorted(case_root.glob("*_r*/worker_status.json")):
            try:
                state = json.loads(state_path.read_text())
                attempts.append({
                    "attempt_id": state_path.parent.name,
                    "status": state.get("status"), "error": state.get("error"),
                    "manifest_sha256": state.get("manifest_sha256"),
                    "state_path": str(state_path),
                    "state_sha256": campaign.sha256_file(state_path),
                })
            except Exception as error:
                attempts.append({"attempt_id": state_path.parent.name,
                                 "status": "invalid", "error": repr(error)})
        item = {"case_id": case_id, "completion_exists": endpoint.is_file(),
                "attempts": attempts}
        if not endpoint.is_file():
            progress.append(item)
            continue
        try:
            completion = json.loads(endpoint.read_text())
            if (completion.get("case_id") != case_id
                    or completion["manifest_sha256"] != manifest_sha
                    or completion["status"] != "finished"):
                raise ValueError("completion identity/status mismatch")
            artifacts = completion["artifacts"]
            for name, record in artifacts.items():
                path = Path(record["path"])
                if campaign.sha256_file(path) != record["sha256"]:
                    raise ValueError(f"artifact hash mismatch: {name}")
            diagnostic = json.loads(Path(artifacts["diagnostic.json"]["path"]).read_text())
            start = json.loads(Path(artifacts["starting_state.json"]["path"]).read_text())
            call = json.loads(Path(artifacts["pricing_call.json"]["path"]).read_text())
            if (diagnostic.get("case_id") != case_id
                    or diagnostic.get("manifest_sha256") != manifest_sha
                    or start.get("case_id") != case_id
                    or start.get("manifest_sha256") != manifest_sha):
                raise ValueError("artifact case/manifest binding mismatch")
            if (start["dual_vector_sha256"] != case["starting_dual_vector_sha256"]
                    or start["source_pool_sha256"] != case["source_pool_sha256"]):
                raise ValueError("starting dual hash mismatch")
            if (call.get("case_id") != case_id
                    or call.get("pair_id") != case["pair_id"]
                    or call.get("capacity_selector") != case["capacity_selector"]
                    or call.get("attempted_calls") != 1
                    or call.get("completed_calls") not in (0, 1)):
                raise ValueError("pricing telemetry binding/call-count mismatch")
            if (call.get("raw_dual_vector_sha256")
                    != case["starting_raw_dual_vector_sha256"]
                    or call.get("normalized_9dp_dual_vector_sha256")
                    != case["starting_normalized_9dp_dual_vector_sha256"]):
                raise ValueError("actual pricing-call dual hash mismatch")
            expected_inputs = {
                "instance": case["instance_sha256"], "prices": case["prices_sha256"],
                "reference": case["reference_sha256"], "deadhead": case["deadhead_sha256"],
            }
            if start.get("input_sha256") != expected_inputs:
                raise ValueError("starting input binding mismatch")
            records.append({
                "case_id": case_id, "pair_id": case["pair_id"],
                "selector": case["capacity_selector"],
                "job_id": completion["attempt_id"].split("_", 1)[0],
                "attempt_id": completion["attempt_id"],
                "source_pool_sha256": case["source_pool_sha256"],
                "starting_dual_vector_sha256": start["dual_vector_sha256"],
                "starting_raw_dual_vector_sha256": start["raw_dual_vector_sha256"],
                "actual_pricing_raw_dual_vector_sha256": call["raw_dual_vector_sha256"],
                "rmp": start["rmp"], "dual_stats": {
                    key: start[key] for key in (
                        "trip_dual_count", "nonzero_capacity_dual_count",
                        "capacity_dual_min", "capacity_dual_max", "capacity_dual_l1",
                    )
                },
                "pricing_call": call,
                "driver_endpoint": diagnostic["driver_endpoint"],
                "proof_scope": diagnostic["proof_scope"],
                "artifact_hashes": artifacts,
            })
            item.update(status="finished", pricing_status=call["status"])
        except Exception as error:
            errors.append({"case_id": case_id, "error": repr(error)})
            item.update(status="invalid")
        progress.append(item)
    pair_checks = {}
    for pair_id in sorted({case["pair_id"] for case in manifest["cases"].values()}):
        subset = [r for r in records if r["pair_id"] == pair_id]
        enough = len(subset) == 2
        pair_checks[pair_id] = {
            "completed_records": len(subset),
            "identical_starting_dual_hash": (
                len({r["starting_dual_vector_sha256"] for r in subset}) == 1
                if enough else None
            ),
            "identical_source_pool_hash": (
                len({r["source_pool_sha256"] for r in subset}) == 1
                if enough else None
            ),
            "identical_actual_raw_pricing_dual_hash": (
                len({r["actual_pricing_raw_dual_vector_sha256"] for r in subset}) == 1
                if enough else None
            ),
            "candidate_hashes": {r["selector"]: r["pricing_call"].get("candidate_route_key_sha256") for r in subset},
        }
    return {
        "schema": "evsp-dr-fixed-capacity-collector-v1", "root": str(root),
        "manifest_sha256": manifest_sha, "records": records, "errors": errors,
        "workflow": {"manifest": str(manifest_path), "jobs": str(root / "jobs.json"),
                     "attempt_progress": progress},
        "pair_checks": pair_checks,
        "integration": {
            "record_kind": "fixed_master_single_exact_pricing_call",
            "optimization_run": True, "cg_run": False, "mip_run": False,
            "certificate_rule": "Only a driver exact_nonnegative_reduced_cost endpoint can certify; max_iters and pricing_deadline endpoints do not.",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root.resolve()), sort_keys=True, allow_nan=False))

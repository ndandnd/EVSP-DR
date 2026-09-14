#!/usr/bin/env python3
"""Native allocated preflight: freeze start vectors and exercise one-call instrumentation."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import campaign


def main(root: Path) -> None:
    draft = json.loads((root / "draft_manifest.json").read_text())
    code = Path(draft["code_root"])
    campaign.verify_checkout(code, draft["execution_commit"])
    driver = campaign.load_driver(code)
    out = root / "preflight" / os.environ.get("SLURM_JOB_ID", "local")
    out.mkdir(parents=True, exist_ok=False)
    starts = {}
    for pair_id, spec in draft["source_pairs"].items():
        campaign.require_inputs(spec)
        pool = out / f"{pair_id}.pool.jsonl"
        campaign.atomic_copy(Path(spec["source_pool_path"]), pool, spec["source_pool_sha256"])
        state = campaign.audit_starting_state(
            driver, code, spec, pool, out / f"{pair_id}.rmp.gurobi.log",
        )
        starts[pair_id] = state
        campaign.atomic_json(out / f"{pair_id}.starting_state.json", state)

    smoke = draft["smoke"]
    smoke_states = []
    for selector in ("reference", "prefix-memo"):
        case = {**smoke, "capacity_selector": selector}
        pool = out / f"smoke_{selector}.pool.jsonl"
        campaign.atomic_copy(Path(case["source_pool_path"]), pool, case["source_pool_sha256"])
        start = campaign.audit_starting_state(
            driver, code, case, pool, out / f"smoke_{selector}.rmp.gurobi.log",
        )
        result = campaign.run_one_call(
            driver, code, case, pool, out / f"smoke_{selector}.cg.json",
            out / f"smoke_{selector}.pricing_call.json",
        )
        call = json.loads((out / f"smoke_{selector}.pricing_call.json").read_text())
        case["starting_dual_vector_sha256"] = start["dual_vector_sha256"]
        case["starting_raw_dual_vector_sha256"] = start["raw_dual_vector_sha256"]
        case["starting_normalized_9dp_dual_vector_sha256"] = start["normalized_9dp_dual_vector_sha256"]
        summary = campaign.validate_endpoint(case, start, call, result)
        if call["status"] != "complete" or len(result["iterations"]) != 1:
            raise ValueError("smoke did not complete exactly one pricing call")
        smoke_states.append((start, call, summary))
    if smoke_states[0][0]["dual_vector_sha256"] != smoke_states[1][0]["dual_vector_sha256"]:
        raise ValueError("smoke selector starting dual hashes differ")
    if (smoke_states[0][1]["raw_dual_vector_sha256"]
            != smoke_states[1][1]["raw_dual_vector_sha256"]
            or smoke_states[0][1]["raw_dual_vector_sha256"]
            != smoke_states[0][0]["raw_dual_vector_sha256"]):
        raise ValueError("smoke actual pricing dual arguments differ")
    if smoke_states[0][1]["candidate_route_key_sha256"] != smoke_states[1][1]["candidate_route_key_sha256"]:
        raise ValueError("smoke selector candidate hashes differ")
    result = {
        "schema": "evsp-dr-fixed-capacity-native-preflight-v1",
        "status": "passed", "job_id": os.environ.get("SLURM_JOB_ID"),
        "code_commit": draft["execution_commit"], "starts": starts,
        "smoke": {
            "starting_dual_vector_sha256": smoke_states[0][0]["dual_vector_sha256"],
            "actual_pricing_raw_dual_vector_sha256": smoke_states[0][1]["raw_dual_vector_sha256"],
            "candidate_route_key_sha256": smoke_states[0][1]["candidate_route_key_sha256"],
            "reference_call_s": smoke_states[0][1]["elapsed_s"],
            "prefix_memo_call_s": smoke_states[1][1]["elapsed_s"],
            "min_reduced_cost": smoke_states[0][1]["min_reduced_cost"],
        },
    }
    campaign.atomic_json(out / "preflight_result.json", result)
    campaign.atomic_json(root / "preflight_result.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

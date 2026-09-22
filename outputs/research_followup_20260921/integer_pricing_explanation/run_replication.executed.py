#!/usr/bin/env python3
"""Matched dive+solver budget; measure all external overhead explicitly."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def remaining_solver_budget(total, charged):
    """Never extend the shared budget with a solver floor."""
    return max(0, math.floor(total - charged))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("case")
    parser.add_argument("arm", choices=["control", "treatment"])
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    started = time.monotonic()
    manifest = json.loads((args.work / "manifest.json").read_text())
    case = manifest["cases"][args.case]
    code = Path(manifest["code"])
    source = Path(case["fresh_cg_result"])
    journal = Path(case["fresh_journal"])
    expected = [case["fresh_cg_result_sha256"], case["fresh_journal_sha256"]]
    assert [sha(source), sha(journal)] == expected, "frozen source identity changed"
    out = args.work / "results" / args.case / f"{args.arm}_s{args.seed}" / f"{os.environ.get('SLURM_JOB_ID', 'local')}_r{os.environ.get('SLURM_RESTART_COUNT', '0')}"
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(EVSP_EXPECTED_COMMIT=manifest["execution_commit"], EVSP_REQUIRE_DETACHED="1")
    receipt = {"case": args.case, "arm": args.arm, "seed": args.seed,
               "execution_commit": manifest["execution_commit"], "source_hashes": expected,
               "status": "running", "global_certificate": None,
               "external_witness_columns_used": False,
               "budget_scope": "3600s dive subprocess wall plus MIP solver time; MIP setup/replay overhead external",
               "original_graph_build_s": case.get("target_external_graph_build_s"),
               "graph_policy": "verified reusable graph prerequisite; original build not repeated or hidden"}
    (out / "execution.json").write_text(json.dumps(receipt, indent=1))

    def run(argv, label):
        (out / f"{label}_argv.json").write_text(json.dumps(argv, indent=1))
        t = time.monotonic()
        with open(out / f"{label}_stdout.log", "w") as stdout, open(out / f"{label}_stderr.log", "w") as stderr:
            p = subprocess.run(argv, cwd=code, env=env, stdout=stdout, stderr=stderr)
        receipt[f"{label}_wall_s"] = time.monotonic() - t
        receipt[f"{label}_returncode"] = p.returncode
        (out / "execution.json").write_text(json.dumps(receipt, indent=1))
        if p.returncode:
            raise RuntimeError(f"{label} failed with {p.returncode}; see preserved logs")

    charged = 0.0
    start_args = []
    if args.arm == "treatment":
        p = manifest["preregistered"]
        argv = [sys.executable, str(code / "src/diving_pricing_pilot.py"), "--result", str(source),
                "--out-dir", str(out / "dive"), "--data-dir", str(code / "data"),
                "--fleet-cap", str(case["target_k"]), "--wall-limit-s", "2400", "--reserve-s", "60",
                "--event-network-cache", case["event_network_cache"], "--cache-commit-bridge",
                "--seed", str(args.seed), "--gurobi-log", str(out / "dive_gurobi.log")]
        for flag, key in [("rc-eps", "rc_eps"), ("columns-per-iter", "columns_per_iter"),
                          ("max-pricing-iters", "max_pricing_iters"), ("node-time-s", "node_time_s"),
                          ("max-nodes", "max_nodes"), ("max-alternatives", "max_alternatives"), ("max-restarts", "max_restarts")]:
            argv.extend(["--" + flag, str(p[key])])
        run(argv, "dive")
        dive = json.loads((out / "dive/manifest.json").read_text())
        source = Path(dive["augmented_pool"]["augmented_result"])
        journal = Path(dive["augmented_pool"]["augmented_journal"])
        exported = dive.get("incumbent_export")
        if exported:
            assert sha(exported["path"]) == exported["sha256"]
            env["EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256"] = exported["sha256"]
            start_args = ["--initial-partition-routes", exported["path"], "--verified-expanded-initial-partition"]
        receipt["incumbent_export"] = exported
        receipt["dive_integer_buses"] = (dive["dive"].get("integer_solution") or {}).get("buses")
        receipt["columns_generated"] = dive["dive"]["columns_generated"]
        charged = receipt["dive_wall_s"]
    limit = remaining_solver_budget(3600, charged)
    receipt["mip_solver_budget_s"] = limit
    env.update(EVSP_MIP_EXPECTED_RESULT_SHA256=sha(source), EVSP_MIP_EXPECTED_JOURNAL_SHA256=sha(journal))
    if limit:
        run([sys.executable, str(code / "src/run_exact_pool_mip.py"), "--result", str(source),
             "--data-dir", str(code / "data"), "--reference-data-dir", str(code / "data"),
             "--cover", "--two-stage", "--timelimit", str(limit), "--stage1-timelimit", str(limit / 2),
             "--threads", "8", "--seed", str(args.seed), "--mipgap", "0.0001",
             "--gurobi-log", str(out / "mip_gurobi.log"), "--out", str(out / "result.json"), *start_args], "mip")
        result = json.loads((out / "result.json").read_text())
        receipt["actual_solver_runtime_s"] = result.get("runtime_s")
        receipt["physical_pool_preparation_wall_s"] = result.get("physical_pool_preparation_wall_s")
        receipt["mip_external_overhead_s"] = receipt["mip_wall_s"] - result["runtime_s"]
        receipt["actual_charged_dive_plus_solver_s"] = charged + result["runtime_s"]
        if start_args:
            start = result["mip_start"]
            assert start["validated"] and start["validated_bus_count"] == exported["buses"]
            assert start["pool_columns_added"] == 0 and start["pool_columns_replaced"] == 0, "handoff changed pool"
            receipt["handoff_validation"] = start
    else:
        receipt["mip_skipped"] = "shared budget exhausted; no floor extension"
    assert [sha(case["fresh_cg_result"]), sha(case["fresh_journal"])] == expected
    receipt.update(status="finished", source_immutable=True, actual_end_to_end_wall_s=time.monotonic() - started,
                   shared_budget_s=3600, charged_dive_wall_s=charged,
                   strict_end_to_end_budget_claim=False)
    receipt["output_sha256"] = {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "execution.json"}
    (out / "execution.json").write_text(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate and validate the diving-pricing pilot manifest.

Runs on the cluster (or anywhere the frozen sources are visible), submits
nothing, and writes nothing outside ``--work``.  It

  * re-hashes every frozen source (fresh ``cg.json``, its column journal and
    the event-network cache manifest) against the identities pinned in
    ``cases_k08.json``;
  * refuses any source carrying warm/witness/inherited provenance, and any
    path that looks like a warm or witness artifact;
  * audits the event-network cache commit bridge (the k=8 caches were built
    at ``e091a4db``/``ecb60c15``; the pilot runs at a later commit) by
    comparing every graph-critical ``EventExpandedNetwork`` method;
  * computes both preregistered budget arms, including the explicit
    graph-build accounting;
  * emits ``manifest.json`` plus a human-readable ``validation.txt``.

Usage::

    python make_manifest.py --work /home/nc437/ladder-lite/diving_pricing_20260919 \\
                            --code /path/to/pinned/checkout

``--code`` is the pinned checkout whose ``src`` supplies the pilot and the
trusted MIP runner; its HEAD is recorded as the execution commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "evsp-dr-diving-pricing-manifest-v1"

# Preregistered, fixed before any case is run.
FLEET_CAP = 8
SEED = 20260919
BASE_BUDGET_S = 3600
DIVE_WALL_LIMIT_S = 2400
MIP_FLOOR_S = 1200
STAGE1_FRACTION = 0.5
RC_EPS = 1e-4
COLUMNS_PER_ITER = 30
MAX_PRICING_ITERS = 400
NODE_TIME_S = 150
MAX_NODES = 40
MAX_ALTERNATIVES = 3
MAX_RESTARTS = 2
MIP_THREADS = 8


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git(code: Path, *args) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=code, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True,
                        help="campaign directory; the only thing written to")
    parser.add_argument("--code", type=Path, required=True,
                        help="pinned checkout supplying src/")
    parser.add_argument("--cases", type=Path, default=HERE / "cases_k08.json")
    parser.add_argument("--python", default="/home/nc437/evsp_env/bin/python")
    parser.add_argument("--allow-missing-cache", action="store_true",
                        help="record, rather than refuse, an absent cache")
    args = parser.parse_args(argv)

    code = args.code.expanduser().resolve()
    sys.path.insert(0, str(code / "src"))
    from diving_cache_identity import compare_graph_methods  # noqa: E402
    from diving_pricing_pilot import (  # noqa: E402
        FORBIDDEN_PATH_PATTERN, FORBIDDEN_STATUS_KEYS,
    )

    table = json.loads(args.cases.read_text())
    work = args.work.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    execution_commit = git(code, "rev-parse", "HEAD")
    if not execution_commit:
        raise SystemExit("[MANIFEST] --code has no verifiable git HEAD")
    if git(code, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("[MANIFEST] --code has tracked modifications")

    problems = []
    cases = {}
    for case_id, spec in sorted(table["cases"].items()):
        entry = {"case_id": case_id, **spec}
        result_path = Path(spec["fresh_cg_result"])
        for path in (result_path,):
            for part in path.parts:
                if FORBIDDEN_PATH_PATTERN.search(part):
                    problems.append(
                        f"{case_id}: source path looks warm/witness: {path}"
                    )
        if not result_path.is_file():
            # Keep going: budget arms and the cache audit are still worth
            # reporting from a machine that cannot see the frozen sources.
            problems.append(f"{case_id}: missing {result_path}")
        else:
            observed = sha256(result_path)
            entry["fresh_cg_result_sha256_observed"] = observed
            if observed != spec["fresh_cg_result_sha256"]:
                problems.append(f"{case_id}: fresh cg.json hash moved")
            status = json.loads(result_path.read_text())
            for key in FORBIDDEN_STATUS_KEYS:
                if status.get(key):
                    problems.append(f"{case_id}: source status carries {key}")
            if status.get("column_pool_treatment") not in (None, "RAW"):
                problems.append(f"{case_id}: source pool is not RAW")
            if status.get("certified_rc_optimal") is not True:
                problems.append(f"{case_id}: source CG is not rc-certified")
            journal_path = Path(status["columns_journal"])
            entry["fresh_journal"] = str(journal_path)
            if journal_path.is_file():
                entry["fresh_journal_sha256"] = sha256(journal_path)
                entry["fresh_journal_bytes"] = journal_path.stat().st_size
            else:
                problems.append(f"{case_id}: missing journal {journal_path}")

        # ---- event-network cache -------------------------------------
        cache_manifest_path = Path(spec["event_network_cache_manifest"])
        cache_audit = {"manifest_path": str(cache_manifest_path)}
        if cache_manifest_path.is_file():
            observed_manifest = sha256(cache_manifest_path)
            cache_audit["manifest_sha256_observed"] = observed_manifest
            if observed_manifest != spec["event_network_cache_manifest_sha256"]:
                problems.append(f"{case_id}: cache manifest hash moved")
            cache_manifest = json.loads(cache_manifest_path.read_text())
            recorded = cache_manifest.get("identity", {})
            cache_audit["recorded_identity"] = recorded
            cache_audit["pickle_bytes"] = cache_manifest.get("pickle_bytes")
            cache_audit["original_build_s"] = cache_manifest.get(
                "original_build_s"
            )
            producer = recorded.get("git_commit")
            cache_audit["producer_commit"] = producer
            cache_audit["consumer_commit"] = execution_commit
            if producer == execution_commit:
                cache_audit["bridge_required"] = False
            else:
                cache_audit["bridge_required"] = True
                try:
                    method_audit = compare_graph_methods(code, str(producer))
                except Exception as exc:
                    method_audit = {"identical": False, "error": repr(exc)}
                cache_audit["method_audit"] = method_audit
                if not method_audit.get("identical"):
                    problems.append(
                        f"{case_id}: cache commit bridge refused "
                        f"({producer} -> {execution_commit})"
                    )
            for field, expected in spec["cache_identity"].items():
                if field == "git_commit":
                    continue
                if recorded.get(field) != expected:
                    problems.append(
                        f"{case_id}: cache identity field {field} moved"
                    )
        elif args.allow_missing_cache:
            cache_audit["status"] = "absent"
        else:
            problems.append(
                f"{case_id}: missing cache manifest {cache_manifest_path}"
            )
        entry["event_network_cache_audit"] = cache_audit

        # ---- budgets --------------------------------------------------
        graph_build_s = float(spec["target_external_graph_build_s"])
        entry["budget"] = {
            "arm_a_shared_prerequisite": {
                "control_mip_timelimit_s": BASE_BUDGET_S,
                "control_stage1_timelimit_s":
                    int(BASE_BUDGET_S * STAGE1_FRACTION),
                "treatment_total_s": BASE_BUDGET_S,
                "treatment_dive_wall_limit_s": DIVE_WALL_LIMIT_S,
                "treatment_mip_floor_s": MIP_FLOOR_S,
                "graph_build_s_charged": 0.0,
                "graph_build_s_declared": graph_build_s,
                "rationale": (
                    "The event graph is a shared prerequisite: the frozen "
                    "fresh pool the control MIP consumes could not exist "
                    "without it. Cache load time is measured and charged to "
                    "the treatment."
                ),
            },
            "arm_b_graph_charged": {
                "control_mip_timelimit_s": int(BASE_BUDGET_S + graph_build_s),
                "control_stage1_timelimit_s":
                    int((BASE_BUDGET_S + graph_build_s) * STAGE1_FRACTION),
                "treatment_total_s": BASE_BUDGET_S,
                "treatment_graph_build_s_debited": graph_build_s,
                "treatment_charged_total_s":
                    int(BASE_BUDGET_S + graph_build_s),
                "rationale": (
                    "Matched adjusted budget. A 3,600 s treatment that "
                    "includes a fresh graph build is impossible for this "
                    "benchmark (build alone is "
                    f"{graph_build_s:.0f} s). Instead both arms are charged "
                    "3,600 s + the recorded build; the control spends that "
                    "as real MIP time while the treatment is debited it, so "
                    "the build is accounted, never excluded."
                ),
            },
            "graph_build_s": graph_build_s,
            "graph_build_fraction_of_base_budget":
                graph_build_s / BASE_BUDGET_S,
            # "Feasible" means the build leaves room for a minimally useful
            # dive (300 s) *and* the floor MIP, not merely that it fits.
            "sixty_minute_treatment_including_build_feasible":
                graph_build_s + MIP_FLOOR_S + 300 <= BASE_BUDGET_S,
            "residual_after_build_s": BASE_BUDGET_S - graph_build_s,
        }
        cases[case_id] = entry

    manifest = {
        "schema": SCHEMA,
        "track": "diving_pricing_20260919",
        "work": str(work),
        "code": str(code),
        "python": args.python,
        "execution_commit": execution_commit,
        # The pool MIP runs from the pinned execution checkout: the
        # historical campaign commit (871d057e) predates --seed, so a
        # seeded control/treatment pair is impossible with it.
        "mip_execution_commit": execution_commit,
        "mip_execution_commit_historical":
            table["mip_execution_commit_historical"],
        "mip_execution_commit_note": table["mip_execution_commit_note"],
        "cases_table": str(args.cases.resolve()),
        "cases_table_sha256": sha256(args.cases),
        "physics": table["physics"],
        "preregistered": {
            "fleet_cap": FLEET_CAP,
            "seed": SEED,
            "seeds_per_case": 1,
            "rc_eps": RC_EPS,
            "columns_per_iter": COLUMNS_PER_ITER,
            "max_pricing_iters": MAX_PRICING_ITERS,
            "node_time_s": NODE_TIME_S,
            "max_nodes": MAX_NODES,
            "max_alternatives": MAX_ALTERNATIVES,
            "max_restarts": MAX_RESTARTS,
            "mip_threads": MIP_THREADS,
            "mipgap": RC_EPS,
            "control": (
                "run_exact_pool_mip.py on the UNMODIFIED frozen fresh pool"
            ),
            "treatment": (
                "cache load + diving-with-pricing + run_exact_pool_mip.py on "
                "the augmented pool (original columns preserved)"
            ),
            "primary_outcome": (
                "buses and fleet_proven from the final pool MIP of each arm"
            ),
            "secondary_outcome": (
                "whether the dive itself reached an 8-bus integer cover"
            ),
            "no_target_import": (
                "No known 8-bus solution, warm pool or witness column is an "
                "input to either arm."
            ),
        },
        "resources": {
            "partition": "default_partition",
            "exclude": "scaglione-compute-01",
            "dive_cpus": 8, "dive_mem": "96G",
            "mip_cpus": MIP_THREADS, "mip_mem": "24G",
        },
        "cases": cases,
        "validation": {
            "problems": problems,
            "status": "passed" if not problems else "failed",
        },
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    lines = [
        f"diving_pricing_20260919 manifest",
        f"execution commit : {execution_commit}",
        f"cases            : {', '.join(sorted(cases))}",
        "",
    ]
    for case_id, entry in sorted(cases.items()):
        budget = entry.get("budget", {})
        audit = entry.get("event_network_cache_audit", {})
        lines += [
            f"[{case_id}] fresh pool buses={entry.get('fresh_buses')} "
            f"bound={entry.get('fresh_pool_fleet_bound')}",
            f"  graph build        : {budget.get('graph_build_s', 0):.0f} s"
            f"  (60-min treatment incl. build feasible: "
            f"{budget.get('sixty_minute_treatment_including_build_feasible')})",
            f"  cache producer     : {audit.get('producer_commit')} "
            f"bridge_required={audit.get('bridge_required')} "
            f"methods_identical="
            f"{(audit.get('method_audit') or {}).get('identical')}",
            f"  arm A control MIP  : "
            f"{budget.get('arm_a_shared_prerequisite', {}).get('control_mip_timelimit_s')} s",
            f"  arm B control MIP  : "
            f"{budget.get('arm_b_graph_charged', {}).get('control_mip_timelimit_s')} s",
        ]
    lines += ["", f"validation: {manifest['validation']['status']}"]
    lines += [f"  - {problem}" for problem in problems]
    (work / "validation.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())

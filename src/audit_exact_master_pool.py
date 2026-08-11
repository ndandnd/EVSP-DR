"""Re-solve one persisted exact-CG restricted master without pricing.

This is the cheap gate before resuming multi-day pricing.  It loads the exact
column journal referenced by a status/snapshot JSON, solves the partition LP
with each requested HiGHS method, and records raw row residuals.  It never
modifies the source pool and does not require Gurobi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy

from config import BIG_M_PENALTY
from durable_io import atomic_copy, atomic_write_json
from master_lp_scipy import build_route_incidence, solve_restricted_master_lp
from run_exact_pool_mip import load_pool, resolve_pool_journal


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parent,
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def freeze_audit_source(source: Path, destination_dir: Path,
                        stem: str) -> tuple[Path, dict]:
    """Freeze one stable status+journal pair before the potentially long LP."""

    source_result_sha256 = _sha256(source)
    with open(source) as fh:
        status = json.load(fh)
    journal = resolve_pool_journal(source, status)
    source_journal_sha256 = _sha256(journal)

    destination_dir.mkdir(parents=True, exist_ok=True)
    frozen_status_path = destination_dir / f"{stem}.source.json"
    frozen_journal_path = Path(str(frozen_status_path) + ".columns.jsonl")
    atomic_copy(journal, frozen_journal_path)
    if (_sha256(source) != source_result_sha256
            or _sha256(journal) != source_journal_sha256
            or _sha256(frozen_journal_path) != source_journal_sha256):
        raise RuntimeError(
            "source status or journal changed while freezing the audit input"
        )
    frozen_status = dict(status)
    frozen_status["columns_journal"] = str(frozen_journal_path)
    frozen_status["audit_source"] = {
        "result": str(source),
        "result_sha256": source_result_sha256,
        "journal": str(journal),
        "journal_sha256": source_journal_sha256,
    }
    atomic_write_json(frozen_status_path, frozen_status)
    return frozen_status_path, frozen_status["audit_source"]


def audit_pool(result_path: Path, methods, feasibility_tolerance: float,
               method_time_limit_s: float | None = None) -> dict:
    with open(result_path) as fh:
        source_status = json.load(fh)
    journal_path = resolve_pool_journal(result_path, source_status)
    source_result_sha256 = _sha256(result_path)
    source_journal_sha256 = _sha256(journal_path)
    status, routes, trips = load_pool(result_path)
    if (_sha256(result_path) != source_result_sha256
            or _sha256(journal_path) != source_journal_sha256):
        raise RuntimeError(
            "source status or journal changed while loading; audit an "
            "immutable snapshot instead"
        )
    incidence = build_route_incidence(
        trip_ids=trips,
        route_trip_ids=[route["trips"] for route in routes],
    )
    costs = [route["cost"] for route in routes]
    method_results = []
    for method in methods:
        started = time.time()
        try:
            lp = solve_restricted_master_lp(
                trip_ids=trips,
                route_incidence=incidence,
                route_costs=costs,
                artificial_penalty=BIG_M_PENALTY,
                method=method,
                coverage_sense="partition",
                feasibility_tolerance=feasibility_tolerance,
                time_limit_s=method_time_limit_s,
            )
            raw_coverage = incidence @ np.asarray(lp.route_values) + np.asarray(
                [lp.artificial_values[trip] for trip in trips]
            )
            method_results.append({
                "method": method,
                "success": True,
                "objective": lp.objective,
                "route_weight": lp.route_weight,
                "artificial_total": lp.artificial_total,
                "positive_route_values": sum(
                    value > 0.0 for value in lp.route_values
                ),
                "max_row_violation": lp.max_row_violation,
                "max_bound_violation": lp.max_bound_violation,
                "recomputed_max_row_violation": float(
                    np.max(np.abs(raw_coverage - 1.0))
                ),
                "runtime_s": lp.runtime_s,
            })
        except Exception as exc:
            method_results.append({
                "method": method,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_s": time.time() - started,
            })

    return {
        "source_result": str(result_path),
        "source_journal": str(journal_path),
        "source_result_sha256": source_result_sha256,
        "source_journal_sha256": source_journal_sha256,
        "source_stop_reason": status.get("stop_reason"),
        "source_provenance": status.get("provenance"),
        "instance": status.get("csv"),
        "prices_csv": status.get("prices_csv"),
        "physics": {
            "soc_step": status.get("soc_step"),
            "block_min": status.get("block_min"),
            "g_kwh": status.get("g_kwh"),
            "charge_kw": status.get("charge_kw"),
            "min_soc_frac": status.get("min_soc_frac"),
        },
        "trip_count": len(trips),
        "pool_columns": len(routes),
        "feasibility_tolerance": feasibility_tolerance,
        "method_time_limit_s": method_time_limit_s,
        "artificial_penalty": BIG_M_PENALTY,
        "any_method_succeeded": any(row["success"] for row in method_results),
        "methods": method_results,
        "audit_provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": platform.python_version(),
            "scipy": scipy.__version__,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", default="highs-ds,highs-ipm,highs")
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-6)
    parser.add_argument("--method-time-limit-s", type=float, default=None)
    args = parser.parse_args(argv)

    methods = [method.strip() for method in args.methods.split(",")
               if method.strip()]
    frozen_result, original_source = freeze_audit_source(
        args.result, args.out.parent / "inputs", args.out.stem
    )
    report = audit_pool(
        frozen_result, methods, args.feasibility_tolerance,
        args.method_time_limit_s,
    )
    report["original_source"] = original_source
    report["original_source_result_sha256"] = original_source["result_sha256"]
    report["original_source_journal_sha256"] = original_source["journal_sha256"]
    atomic_write_json(args.out, report)

    successes = [row for row in report["methods"] if row["success"]]
    if successes:
        best = min(successes, key=lambda row: row["runtime_s"])
        print(f"[LP-AUDIT] OK {args.result.name}: {report['pool_columns']} cols, "
              f"weight={best['route_weight']:.6f}, "
              f"art={best['artificial_total']:.3g}, "
              f"residual={best['max_row_violation']:.3g} ({best['method']})")
        return 0
    print(f"[LP-AUDIT] FAILED {args.result.name}: "
          + " | ".join(row["error"] for row in report["methods"]))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

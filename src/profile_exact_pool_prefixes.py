"""Read-only cold-master profiling on prefixes of one frozen exact-CG pool.

The source status, journal, instance, and tariff are hashed before and after.
No source file is repaired or opened for writing.  Prefixes follow production
pool semantics: first appearance establishes a route incidence and a later
strictly cheaper duplicate replaces it without increasing pool size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

from config import BIG_M_PENALTY
from durable_io import atomic_write_json, read_jsonl_records
from exact_cg_telemetry import peak_rss_bytes
from exact_pricer_expanded import DATA_DIR, load_column_pool
from master_lp_scipy import build_route_incidence, solve_restricted_master_lp
from run_exact_pool_mip import resolve_pool_journal


SCHEMA = "evsp-dr-frozen-pool-prefix-profile-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def ordered_unique_prefixes(
    records: list[dict],
    trip_ids: list[int],
    targets: list[int],
) -> dict[int, list[dict]]:
    """Reconstruct first-reached unique-pool prefixes after strict validation."""

    load_column_pool(records, trip_ids)
    remaining = sorted(set(targets))
    snapshots: dict[int, list[dict]] = {}
    pool: dict[frozenset, dict] = {}
    for record in records:
        key = frozenset(record["trips"])
        if (key not in pool
                or float(record["cost"]) < float(pool[key]["cost"]) - 1e-9):
            pool[key] = record
        while remaining and len(pool) >= remaining[0]:
            target = remaining.pop(0)
            snapshots[target] = list(pool.values())
        if not remaining:
            break
    return snapshots


def profile(args) -> dict:
    result_path = args.result.expanduser().resolve()
    if args.out is not None and args.out.expanduser().resolve() == result_path:
        raise ValueError("--out must not overwrite the source status")
    status_raw = result_path.read_bytes()
    status = json.loads(status_raw)
    if not isinstance(status, dict):
        raise ValueError("source result is not a JSON object")
    journal_path = resolve_pool_journal(result_path, status).resolve()
    instance_path = (DATA_DIR / str(status["csv"])).resolve()
    prices_path = (DATA_DIR / str(status["prices_csv"])).resolve()
    data_root = DATA_DIR.resolve()
    for label, path in (("instance", instance_path), ("prices", prices_path)):
        try:
            path.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(
                f"{label} escapes repository data directory: {path}"
            ) from exc
    for label, path in (
        ("journal", journal_path),
        ("instance", instance_path),
        ("prices", prices_path),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")

    before = {
        "result": hashlib.sha256(status_raw).hexdigest(),
        "journal": sha256_file(journal_path),
        "instance": sha256_file(instance_path),
        "prices": sha256_file(prices_path),
    }
    provenance = status.get("provenance") or {}
    if (provenance.get("instance_sha256") != before["instance"]
            or provenance.get("prices_sha256") != before["prices"]):
        raise ValueError("current instance/tariff bytes do not match provenance")

    records = read_jsonl_records(
        journal_path, repair_trailing=False
    )
    trip_ids = [int(trip) for trip in status["trip_ids"]]
    prefixes = ordered_unique_prefixes(records, trip_ids, args.prefixes)
    rows = []
    for target in args.prefixes:
        routes = prefixes.get(target)
        if routes is None:
            rows.append({
                "prefix_columns": target,
                "available": False,
                "reason": "journal has fewer unique incidences",
            })
            continue
        incidence_started = time.perf_counter()
        incidence = build_route_incidence(
            trip_ids=trip_ids,
            route_trip_ids=[route["trips"] for route in routes],
        )
        incidence_s = time.perf_counter() - incidence_started
        incidence_bytes = int(
            incidence.data.nbytes
            + incidence.indices.nbytes
            + incidence.indptr.nbytes
        )
        methods = []
        for method in args.methods:
            started = time.perf_counter()
            try:
                solved = solve_restricted_master_lp(
                    trip_ids=trip_ids,
                    route_incidence=incidence,
                    route_costs=[route["cost"] for route in routes],
                    artificial_penalty=BIG_M_PENALTY,
                    method=method,
                    coverage_sense=status.get(
                        "master_sense", "partition"
                    ),
                    time_limit_s=args.time_limit_s,
                )
            except Exception as exc:
                methods.append({
                    "method": method,
                    "outcome": "error",
                    "total_s": time.perf_counter() - started,
                    "error": repr(exc),
                    "peak_rss_bytes": peak_rss_bytes(),
                })
                continue
            methods.append({
                "method": method,
                "outcome": "ok",
                "total_s": time.perf_counter() - started,
                "backend_s": solved.runtime_s,
                "objective": solved.objective,
                "route_weight": solved.route_weight,
                "artificial_total": solved.artificial_total,
                "max_row_violation": solved.max_row_violation,
                "max_bound_violation": solved.max_bound_violation,
                "peak_rss_bytes": peak_rss_bytes(),
            })
        rows.append({
            "prefix_columns": target,
            "available": True,
            "incidence_s": incidence_s,
            "incidence_rows": int(incidence.shape[0]),
            "incidence_columns": int(incidence.shape[1]),
            "incidence_nnz": int(incidence.nnz),
            "incidence_csr_bytes": incidence_bytes,
            "methods": methods,
        })

    after = {
        "result": sha256_file(result_path),
        "journal": sha256_file(journal_path),
        "instance": sha256_file(instance_path),
        "prices": sha256_file(prices_path),
    }
    if after != before:
        raise RuntimeError(
            f"source inputs changed during profiling: before={before}, "
            f"after={after}"
        )
    return {
        "schema": SCHEMA,
        "mode": "read_only_frozen_pool_prefix_profile",
        "source_result": str(result_path),
        "source_journal": str(journal_path),
        "source_hashes_before": before,
        "source_hashes_after": after,
        "source_unchanged": True,
        "trip_count": len(trip_ids),
        "physical_records": len(records),
        "requested_prefixes": args.prefixes,
        "methods": args.methods,
        "time_limit_s": args.time_limit_s,
        "profiles": rows,
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": platform.python_version(),
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--prefixes",
        default="1000,5000,10000,25000,50000",
        help="Comma-separated unique-column prefix sizes.",
    )
    parser.add_argument(
        "--methods",
        default="highs,highs-ds,highs-ipm",
        help="Comma-separated cold SciPy/HiGHS methods.",
    )
    parser.add_argument("--time-limit-s", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        args.prefixes = sorted({
            int(value) for value in args.prefixes.split(",") if value.strip()
        })
    except ValueError as exc:
        parser.error(f"invalid --prefixes: {exc}")
    if not args.prefixes or any(value <= 0 for value in args.prefixes):
        parser.error("--prefixes must contain positive integers")
    args.methods = [
        value.strip() for value in args.methods.split(",") if value.strip()
    ]
    allowed = {"highs", "highs-ds", "highs-ipm"}
    if not args.methods or any(method not in allowed for method in args.methods):
        parser.error(f"--methods must be drawn from {sorted(allowed)}")
    if (args.time_limit_s is not None
            and (not math.isfinite(args.time_limit_s)
                 or args.time_limit_s <= 0.0)):
        parser.error("--time-limit-s must be positive and finite")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    payload = profile(args)
    if args.out is not None:
        atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

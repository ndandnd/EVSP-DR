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
import os
import platform
import statistics
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from config import BIG_M_PENALTY
from durable_io import (
    DurableFileError,
    flush_and_fsync,
    read_jsonl_records,
)
from exact_cg_telemetry import peak_rss_bytes
from exact_pricer_expanded import DATA_DIR, load_column_pool
from master_lp_scipy import build_route_incidence, solve_restricted_master_lp
from run_exact_pool_mip import resolve_pool_journal


SCHEMA = "evsp-dr-frozen-pool-prefix-profile-v2"
AGREEMENT_REL_TOL = 1e-9
AGREEMENT_ABS_TOL = 1e-6


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


def _solution_signature(solved) -> dict:
    return {
        "objective": float(solved.objective),
        "route_weight": float(solved.route_weight),
        "artificial_total": float(solved.artificial_total),
        "max_row_violation": float(solved.max_row_violation),
        "max_bound_violation": float(solved.max_bound_violation),
    }


def _assert_solutions_agree(
    reference: dict,
    candidate: dict,
    *,
    context: str,
) -> None:
    for field in reference:
        if not math.isclose(
                reference[field],
                candidate[field],
                rel_tol=AGREEMENT_REL_TOL,
                abs_tol=AGREEMENT_ABS_TOL):
            raise ValueError(
                f"successful master solutions disagree for {context}: "
                f"{field}={reference[field]} versus {candidate[field]}"
            )


@contextmanager
def _reserve_output_lock(output_path: Path, metadata: dict):
    """Atomically reserve a new output without opening any existing inode."""

    lock_path = Path(str(output_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise DurableFileError(
            f"profiler output lock already exists: {lock_path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            payload = {
                **metadata,
                "output": str(output_path),
                "lock_path": str(lock_path),
                "pid": os.getpid(),
                "created_epoch_s": time.time(),
            }
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            flush_and_fsync(handle)
            yield lock_path
    finally:
        # The reservation remains as diagnostic evidence. A failed or
        # interrupted profile must use a new output path rather than guessing
        # whether an existing reservation is stale.
        pass


def _resolve_sources(args):
    """Resolve source paths without modifying any source or lock file."""

    result_path = args.result.expanduser().resolve()
    if not result_path.name.endswith(".snapshot.json"):
        raise ValueError(
            "profiler requires an immutable *.snapshot.json source"
        )
    status_raw = result_path.read_bytes()
    status = json.loads(status_raw)
    if not isinstance(status, dict):
        raise ValueError("source result is not a JSON object")
    journal_path = resolve_pool_journal(result_path, status).resolve()
    instance_path = (DATA_DIR / str(status["csv"])).resolve()
    prices_path = (DATA_DIR / str(status["prices_csv"])).resolve()
    return (
        result_path,
        status_raw,
        status,
        journal_path,
        instance_path,
        prices_path,
    )


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
    (
        result_path,
        status_raw,
        status,
        journal_path,
        instance_path,
        prices_path,
    ) = _resolve_sources(args)
    if args.out is not None:
        output_path = args.out.expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing profiler output: {output_path}"
            )
        protected = {
            result_path, journal_path, instance_path, prices_path,
        }
        if output_path in protected:
            raise ValueError(
                "--out must not overwrite status, journal, instance, or tariff"
            )
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
    if before["result"] != args.expected_result_sha256:
        raise ValueError(
            "source status hash does not match --expected-result-sha256"
        )
    if before["journal"] != args.expected_journal_sha256:
        raise ValueError(
            "resolved journal hash does not match --expected-journal-sha256"
        )
    provenance = status.get("provenance") or {}
    if (provenance.get("instance_sha256") != before["instance"]
            or provenance.get("prices_sha256") != before["prices"]):
        raise ValueError("current instance/tariff bytes do not match provenance")

    records = read_jsonl_records(
        journal_path, repair_trailing=False
    )
    raw_trip_ids = status.get("trip_ids")
    if (not isinstance(raw_trip_ids, list) or not raw_trip_ids
            or any(not isinstance(trip, int) or isinstance(trip, bool)
                   for trip in raw_trip_ids)
            or len(raw_trip_ids) != len(set(raw_trip_ids))):
        raise ValueError(
            "snapshot trip_ids must be a nonempty list of unique integers"
        )
    trip_ids = list(raw_trip_ids)
    allowed_trip_ids = set(trip_ids)
    for ordinal, record in enumerate(records, start=1):
        record_trips = record.get("trips")
        if (not isinstance(record_trips, list) or not record_trips
                or any(not isinstance(trip, int) or isinstance(trip, bool)
                       for trip in record_trips)
                or len(record_trips) != len(set(record_trips))):
            raise ValueError(
                f"journal record {ordinal} must contain unique integer trips"
            )
        unknown = [
            trip for trip in record_trips if trip not in allowed_trip_ids
        ]
        if unknown:
            raise ValueError(
                f"journal record {ordinal} has unknown trips: {unknown[:10]}"
            )
    full_pool = load_column_pool(records, trip_ids)
    recorded_columns = status.get("columns")
    if (not isinstance(recorded_columns, int)
            or isinstance(recorded_columns, bool)
            or recorded_columns < 0):
        raise ValueError("snapshot columns must be a nonnegative integer")
    if len(full_pool) != recorded_columns:
        raise ValueError(
            f"snapshot records {recorded_columns} columns but its journal "
            f"contains {len(full_pool)} unique incidences"
        )
    final_lp = status.get("final_lp")
    if final_lp is not None and not isinstance(final_lp, dict):
        raise ValueError("snapshot final_lp must be an object or null")
    positive_routes = (final_lp or {}).get("positive_routes", [])
    if not isinstance(positive_routes, list):
        raise ValueError("snapshot final_lp positive_routes must be a list")
    pool_keys = set(full_pool)
    allowed_trips = allowed_trip_ids
    for ordinal, route in enumerate(positive_routes, start=1):
        if not isinstance(route, dict) or not isinstance(
                route.get("trips"), list):
            raise ValueError(
                f"snapshot positive route {ordinal} has invalid trips"
            )
        try:
            trips = route["trips"]
            key = frozenset(trips)
        except TypeError as exc:
            raise ValueError(
                f"snapshot positive route {ordinal} has unhashable trips"
            ) from exc
        if not trips or len(trips) != len(key):
            raise ValueError(
                f"snapshot positive route {ordinal} has empty/repeated trips"
            )
        if any(not isinstance(trip, int) or isinstance(trip, bool)
               for trip in trips):
            raise ValueError(
                f"snapshot positive route {ordinal} has non-integer trips"
            )
        unknown = [trip for trip in trips if trip not in allowed_trips]
        if unknown:
            raise ValueError(
                f"snapshot positive route {ordinal} has unknown trips: "
                f"{unknown[:10]}"
            )
        if key not in pool_keys:
            raise ValueError(
                f"snapshot positive route {ordinal} is missing from journal"
            )
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
        successful_signatures = []
        for method in args.methods:
            repetitions = []
            method_signature = None
            for repetition in range(1, args.repeat + 1):
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
                    repetitions.append({
                        "repetition": repetition,
                        "outcome": "error",
                        "total_s": time.perf_counter() - started,
                        "error": repr(exc),
                        "peak_rss_bytes": peak_rss_bytes(),
                    })
                    continue
                signature = _solution_signature(solved)
                if method_signature is None:
                    method_signature = signature
                else:
                    _assert_solutions_agree(
                        method_signature,
                        signature,
                        context=f"prefix {target}, method {method} repeats",
                    )
                repetitions.append({
                    "repetition": repetition,
                    "outcome": "ok",
                    "total_s": time.perf_counter() - started,
                    "backend_s": solved.runtime_s,
                    **signature,
                    "peak_rss_bytes": peak_rss_bytes(),
                })
            successful = [
                repetition for repetition in repetitions
                if repetition["outcome"] == "ok"
            ]
            if method_signature is not None:
                successful_signatures.append((method, method_signature))
            totals = [repetition["total_s"] for repetition in successful]
            backends = [repetition["backend_s"] for repetition in successful]
            methods.append({
                "method": method,
                "outcome": (
                    "ok" if len(successful) == args.repeat else "error"
                ),
                "successful_repetitions": len(successful),
                "requested_repetitions": args.repeat,
                "repetitions": repetitions,
                "solution": method_signature,
                "timing": ({
                    "total_min_s": min(totals),
                    "total_median_s": statistics.median(totals),
                    "total_max_s": max(totals),
                    "backend_min_s": min(backends),
                    "backend_median_s": statistics.median(backends),
                    "backend_max_s": max(backends),
                } if totals else None),
            })
        if successful_signatures:
            reference_method, reference = successful_signatures[0]
            for method, signature in successful_signatures[1:]:
                _assert_solutions_agree(
                    reference,
                    signature,
                    context=(
                        f"prefix {target}, methods "
                        f"{reference_method} versus {method}"
                    ),
                )
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
        "repeat": args.repeat,
        "agreement_tolerance": {
            "relative": AGREEMENT_REL_TOL,
            "absolute": AGREEMENT_ABS_TOL,
            "fields": [
                "objective", "route_weight", "artificial_total",
                "max_row_violation", "max_bound_violation",
            ],
        },
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
    def sha256_value(value):
        normalized = value.strip().lower()
        if (len(normalized) != 64
                or any(character not in "0123456789abcdef"
                       for character in normalized)):
            raise argparse.ArgumentTypeError(
                "expected a 64-character SHA-256"
            )
        return normalized

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--expected-result-sha256",
        type=sha256_value,
        required=True,
    )
    parser.add_argument(
        "--expected-journal-sha256",
        type=sha256_value,
        required=True,
    )
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
    parser.add_argument("--repeat", type=int, default=3)
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
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    return args


def run_profile(args) -> dict:
    if args.out is None:
        return profile(args)
    output_path = args.out.expanduser().resolve()
    (
        result_path,
        _status_raw,
        _status,
        journal_path,
        instance_path,
        prices_path,
    ) = _resolve_sources(args)
    protected = {
        result_path, journal_path, instance_path, prices_path,
    }
    lock_path = Path(str(output_path) + ".lock").resolve()
    if output_path in protected or lock_path in protected:
        raise ValueError(
            "profiler output or lock path aliases a protected source"
        )
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing profiler output: {output_path}"
        )
    metadata = {
        "operation": "frozen_pool_prefix_profile",
        "source_result": str(args.result.expanduser().resolve()),
        "expected_result_sha256": args.expected_result_sha256,
        "expected_journal_sha256": args.expected_journal_sha256,
    }
    with _reserve_output_lock(output_path, metadata):
        if output_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing profiler output: {output_path}"
            )
        payload = profile(args)
        if output_path.exists():
            raise FileExistsError(
                f"profiler output appeared during run: {output_path}"
            )
        _write_json_no_clobber(output_path, payload)
        return payload


def _write_json_no_clobber(path: Path, payload: dict) -> None:
    """Publish a complete JSON file atomically without replacing any inode."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.tmp.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=1)
            handle.write("\n")
            flush_and_fsync(handle)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite profiler output created concurrently: "
                f"{path}"
            ) from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None) -> int:
    args = parse_args(argv)
    payload = run_profile(args)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

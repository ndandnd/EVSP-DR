"""Isolated benchmark of omitted incidence construction; no production edits.

Uses synthetic route-incidence workloads, not physically validated bus duties.
The paired Gurobi replay freezes every pool, cost, row and solver method.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("EVSP_BENCH_SOURCE", "/private/tmp/evsp-algorithm-review-source-20260912"))
sys.path.insert(0, str(SOURCE / "src"))
from master_lp_scipy import build_route_incidence
from master_lp_gurobi import GurobiRestrictedMaster, RestrictedMasterInputError, gurobi_preflight


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_routes(n_trips, n_routes, seed):
    rng = random.Random(seed)
    routes = [{"trips": [i], "cost": 100010.0 + i / 100} for i in range(n_trips)]
    for _ in range(n_routes - n_trips):
        sequence = sorted(rng.sample(range(n_trips), rng.randint(3, min(20, n_trips))))
        routes.append({"trips": sequence, "cost": 100000.0 + rng.uniform(10, 200)})
    return routes


def prepare(trips, routes, treatment):
    if treatment == "baseline":
        incidence = build_route_incidence(trip_ids=trips, route_trip_ids=[r["trips"] for r in routes])
        return incidence.shape, int(incidence.nnz)
    # Gurobi's sync_routes still validates every route. Only the unused matrix
    # is omitted. The prototype deliberately retains O(pool) telemetry work.
    return (len(trips), len(routes)), sum(len(r["trips"]) for r in routes)


def audit_result(master, result):
    dual = result.trip_duals
    min_route_rc = min((cost - sum(dual[t] for t in trips) for trips, cost, _ in master._routes), default=0.0)
    min_artificial_rc = min(master.artificial_penalty - dual[t] for t in master.trip_ids)
    gap = result.objective - sum(dual.values())
    sign_violation = max(0.0, -min(dual.values())) if master.coverage_sense == "cover" else 0.0
    if min_route_rc < -1e-6 or min_artificial_rc < -1e-6 or abs(gap) > 1e-5 or sign_violation > 1e-6:
        raise AssertionError("Restricted-master dual audit failed")
    return {
        "objective": result.objective,
        "route_weight": result.route_weight,
        "artificials": result.artificial_total,
        "route_values": list(result.route_values),
        "trip_duals": list(dual.values()),
        "max_row_violation": result.max_row_violation,
        "max_bound_violation": result.max_bound_violation,
        "min_restricted_route_rc": min_route_rc,
        "min_artificial_rc": min_artificial_rc,
        "primal_dual_gap": gap,
    }


def solve_replay(trips, routes, treatment, sense):
    start = time.perf_counter()
    master = GurobiRestrictedMaster(trip_ids=trips, artificial_penalty=1e7, coverage_sense=sense, threads=1)
    phases = {"prepare_s": 0.0, "sync_s": 0.0, "solve_s": 0.0}
    results = []
    pools = []
    try:
        for stop in range(len(trips), len(routes) + 1, 80):
            pools.append([dict(r) for r in routes[:stop]])
        if len(pools[-1]) != len(routes):
            pools.append([dict(r) for r in routes])
        rewrite = [dict(r) for r in routes]
        rewrite[-1] = {**rewrite[-1], "cost": rewrite[-1]["cost"] - 5.0}
        pools.append(rewrite)
        for pool in pools:
            t = time.perf_counter()
            shape, nnz = prepare(trips, pool, treatment)
            phases["prepare_s"] += time.perf_counter() - t
            t = time.perf_counter()
            master.sync_routes(pool)
            phases["sync_s"] += time.perf_counter() - t
            t = time.perf_counter()
            result = master.solve()
            phases["solve_s"] += time.perf_counter() - t
            result_record = audit_result(master, result)
            result_record.update(shape=shape, nnz=nnz)
            results.append(result_record)
        return {"total_s": time.perf_counter() - start, **phases}, results
    finally:
        master.close()


def assert_equal(first, second):
    if len(first) != len(second):
        raise AssertionError("Different iteration count")
    max_error = 0.0
    for a, b in zip(first, second):
        for key in a:
            av, bv = a[key], b[key]
            if isinstance(av, (list, tuple)):
                if len(av) != len(bv):
                    raise AssertionError(key)
                differences = [abs(x - y) for x, y in zip(av, bv)]
            else:
                differences = [abs(av - bv)]
            max_error = max(max_error, *differences)
    if max_error > 1e-7:
        raise AssertionError(f"Paired replay mismatch {max_error}")
    return max_error


def main():
    license_path = Path.home() / "gurobi.lic"
    if "GRB_LICENSE_FILE" not in os.environ and license_path.is_file():
        os.environ["GRB_LICENSE_FILE"] = str(license_path)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        preflight = gurobi_preflight()
    result = {
        "source_commit": "a29992196acb74d02b8c7891be4061718889999f",
        "source_hashes": {name: hashlib.sha256((SOURCE / "src" / name).read_bytes()).hexdigest()
                          for name in ("master_lp_scipy.py", "master_lp_gurobi.py", "exact_pricer_expanded.py")},
        "python": sys.version,
        "platform": platform.platform(),
        "gurobi_version": preflight["version"],
        "threads": 1,
        "workload": "Synthetic frozen incidence pools; no physical route-space claim",
        "microbenchmarks": [], "master_replays": [],
    }
    with open("/private/tmp/evsp-algorithm-benchmarks-20260912.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for count in (1000, 10000, 50000):
            trips = tuple(range(250))
            routes = make_routes(len(trips), count, 913)
            assert prepare(trips, routes, "baseline") == prepare(trips, routes, "skip_matrix")
            samples = {"baseline": [], "skip_matrix": []}
            for repetition in range(7):
                order = ("baseline", "skip_matrix") if repetition % 2 == 0 else ("skip_matrix", "baseline")
                for treatment in order:
                    t = time.perf_counter()
                    prepare(trips, routes, treatment)
                    samples[treatment].append(time.perf_counter() - t)
            medians = {name: statistics.median(values) for name, values in samples.items()}
            result["microbenchmarks"].append({
                "n_trips": len(trips), "n_routes": count,
                "nnz": sum(len(r["trips"]) for r in routes), "pool_sha256": digest(routes),
                "samples_s": samples, "medians_s": medians,
                "speedup": medians["baseline"] / medians["skip_matrix"],
                "absolute_saved_s": medians["baseline"] - medians["skip_matrix"],
            })
        trips = tuple(range(80))
        routes = make_routes(80, 1040, 914)
        for sense in ("cover", "partition"):
            samples = {"baseline": [], "skip_matrix": []}
            reference = None
            max_error = 0.0
            for repetition in range(7):
                order = ("baseline", "skip_matrix") if repetition % 2 == 0 else ("skip_matrix", "baseline")
                for treatment in order:
                    timing, solutions = solve_replay(trips, routes, treatment, sense)
                    if reference is None:
                        reference = solutions
                    else:
                        max_error = max(max_error, assert_equal(reference, solutions))
                    samples[treatment].append(timing)
            medians = {treatment: statistics.median(v["total_s"] for v in values)
                       for treatment, values in samples.items()}
            result["master_replays"].append({
                "sense": sense, "n_trips": 80, "final_routes": 1040, "iterations": len(reference),
                "pool_sha256": digest(routes), "samples": samples, "median_total_s": medians,
                "speedup": medians["baseline"] / medians["skip_matrix"],
                "max_paired_numeric_error": max_error, "reference_solutions": reference,
                "scope": "Master replay including model setup and independent audit; excludes graph/pricing/CG/I/O",
            })
        # Skipping the matrix must retain validation at Gurobi synchronization.
        master = GurobiRestrictedMaster(trip_ids=[0, 1], artificial_penalty=1e7, threads=1)
        rejected = 0
        try:
            for invalid in ([], [0, 0], [2]):
                try:
                    master.sync_routes([{"trips": invalid, "cost": 100000.0}])
                except RestrictedMasterInputError:
                    rejected += 1
                else:
                    raise AssertionError("Malformed route accepted")
        finally:
            master.close()
        result["validation"] = {"malformed_routes_rejected": rejected,
                                "cheaper_same_incidence_rewrite_in_each_replay": True,
                                "primal_and_dual_audit_each_solve": True,
                                "global_pricing_certificate_tested": False}
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"microbenchmarks": [{k: v for k, v in row.items() if k != "samples_s"}
                                          for row in result["microbenchmarks"]],
                      "master_replays": [{k: v for k, v in row.items() if k not in {"samples", "reference_solutions"}}
                                         for row in result["master_replays"]]}, indent=2))


if __name__ == "__main__":
    main()

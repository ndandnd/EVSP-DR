"""Deterministic local benchmark for exact pricing and candidate extraction.

This does not run column generation or write result pools.  Every repetition
uses the same dual vector, and the output includes a canonical SHA-256 of all
returned route/cost/charging records for before/after equivalence checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

from audit_giro_known_columns import HORIZON_MIN, build_problem
from config import BUS_COST_KX, CHARGE_RATE_KW, CHARGING_STATIONS
from durable_io import atomic_write_json
from exact_cg_telemetry import peak_rss_bytes
from exact_pricer_expanded import DATA_DIR, ExpandedNetwork
from utils_v2 import load_station_hourly_prices


SCHEMA = "evsp-dr-exact-pricing-benchmark-v1"


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _canonical_routes(routes: list[dict]) -> tuple[str, list[dict]]:
    cleaned = [
        {
            key: value for key, value in route.items()
            if not key.startswith("_")
        }
        for route in routes
    ]
    encoded = json.dumps(
        cleaned, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest(), cleaned


def benchmark_case(args, csv_name: str) -> dict:
    data_dir = args.data_dir.expanduser().resolve()
    prices_data_dir = args.prices_data_dir.expanduser().resolve()
    problem_started = time.perf_counter()
    problem = build_problem(
        data_dir, csv_name,
        max_station_to_trip_wait_min=HORIZON_MIN,
    )
    problem_build_s = time.perf_counter() - problem_started
    prices = load_station_hourly_prices(
        prices_data_dir / args.prices_csv, CHARGING_STATIONS
    )
    network_started = time.perf_counter()
    network = ExpandedNetwork(
        problem,
        prices,
        soc_step=args.soc_step,
        block_min=args.block_min,
        g_kwh=args.g_kwh,
        charge_kw=args.charge_kw,
        reserve_kwh=args.reserve * args.g_kwh,
    )
    network_build_s = time.perf_counter() - network_started
    duals = {trip: float(args.dual_value) for trip in problem.trips}

    for _ in range(args.warmup):
        network.k_best_routes(duals, k=args.columns)
    durations = []
    route_hashes = []
    canonical = None
    for _ in range(args.repeat):
        started = time.perf_counter()
        routes = network.k_best_routes(duals, k=args.columns)
        durations.append(time.perf_counter() - started)
        route_hash, canonical = _canonical_routes(routes)
        route_hashes.append(route_hash)
    if len(set(route_hashes)) != 1:
        raise RuntimeError(
            f"pricing benchmark is nondeterministic for {csv_name}: "
            f"{route_hashes}"
        )
    return {
        "csv": csv_name,
        "trip_count": len(problem.trips),
        "problem_build_s": problem_build_s,
        "network_build_s": network_build_s,
        "network_nodes": len(network.node_meta),
        "network_arcs": network.n_arcs,
        "sink_arcs": sum(
            successor == network.SINK
            for arcs in network.out
            for successor, _cost, _trip in arcs
        ),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "pricing_seconds": durations,
        "pricing_min_s": min(durations),
        "pricing_median_s": statistics.median(durations),
        "returned_routes": len(canonical or []),
        "route_sha256": route_hashes[0],
        "best_route": (canonical or [None])[0],
        "peak_rss_bytes": peak_rss_bytes(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--prices-data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--csv", action="append", required=True)
    parser.add_argument("--prices_csv", default="hourly_prices_flat.csv")
    parser.add_argument("--soc-step", type=float, default=15.0)
    parser.add_argument("--block-min", type=int, default=10)
    parser.add_argument("--g-kwh", type=float, default=300.0)
    parser.add_argument("--charge-kw", type=float, default=CHARGE_RATE_KW)
    parser.add_argument("--reserve", type=float, default=0.0)
    parser.add_argument("--columns", type=int, default=30)
    parser.add_argument("--dual-value", type=float, default=BUS_COST_KX)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.columns <= 0 or args.warmup < 0 or args.repeat <= 0:
        parser.error("columns/repeat must be positive and warmup nonnegative")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    payload = {
        "schema": SCHEMA,
        "configuration": {
            "data_dir": str(args.data_dir.expanduser().resolve()),
            "prices_data_dir": str(
                args.prices_data_dir.expanduser().resolve()
            ),
            "prices_csv": args.prices_csv,
            "soc_step": args.soc_step,
            "block_min": args.block_min,
            "g_kwh": args.g_kwh,
            "charge_kw": args.charge_kw,
            "reserve": args.reserve,
            "columns": args.columns,
            "dual_value": args.dual_value,
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": platform.python_version(),
        },
        "cases": [benchmark_case(args, csv_name) for csv_name in args.csv],
    }
    if args.out:
        atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

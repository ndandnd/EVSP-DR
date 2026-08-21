#!/usr/bin/env python3
"""Certify lexicographic phase-2 fleet LP from an immutable RAW CG pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import exact_pricer_expanded as exact
from durable_io import atomic_write_json, flush_and_fsync
from expanded_path_realization import (
    BLOCK_SCHEDULE_SCHEMA,
    realized_costs,
)
from master_lp_scipy import build_route_incidence
from run_exact_pool_mip import resolve_pool_journal
from target_pool_feasibility import load_bound_pool_bytes


SCHEMA = "evsp-dr-certified-fleet-lp-phase2-v1"
CERTIFICATE_SCHEMA = "evsp-dr-fleet-lp-phase2-certificate-v1"
PRICING_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FleetLP:
    objective: float
    route_values: tuple[float, ...]
    trip_duals: dict[int, float]
    max_row_violation: float
    max_bound_violation: float


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def solve_fleet_master(trips, routes) -> FleetLP:
    incidence = build_route_incidence(
        trips, [route["trips"] for route in routes]
    )
    if not routes:
        raise ValueError("fleet master has no real routes")
    rhs = np.ones(len(trips), dtype=float)
    solved = linprog(
        c=np.ones(len(routes), dtype=float),
        A_eq=incidence,
        b_eq=rhs,
        bounds=(0.0, None),
        method="highs-ds",
        options={"presolve": True},
    )
    if not solved.success or solved.x is None:
        raise RuntimeError(
            "fleet master failed: "
            f"status={solved.status}, message={solved.message}"
        )
    primal = np.asarray(solved.x, dtype=float)
    dual = np.asarray(solved.eqlin.marginals, dtype=float)
    if not np.isfinite(primal).all() or not np.isfinite(dual).all():
        raise RuntimeError("fleet master returned non-finite values")
    row_violation = float(np.max(np.abs(incidence @ primal - rhs)))
    bound_violation = max(0.0, -float(np.min(primal)))
    if row_violation > 1e-7 or bound_violation > 1e-7:
        raise RuntimeError(
            "fleet master primal violation: "
            f"row={row_violation}, bound={bound_violation}"
        )
    return FleetLP(
        objective=float(solved.fun),
        route_values=tuple(map(float, primal)),
        trip_duals={
            trip: float(dual[index])
            for index, trip in enumerate(trips)
        },
        max_row_violation=row_violation,
        max_bound_violation=bound_violation,
    )


def _build_network(status, *, data_dir: Path):
    problem = exact.build_problem(
        data_dir,
        status["csv"],
        max_station_to_trip_wait_min=exact.HORIZON_MIN,
    )
    prices = exact.load_station_hourly_prices(
        data_dir / status["prices_csv"],
        exact.CHARGING_STATIONS,
    )
    kwargs = {
        "soc_step": float(status["soc_step"]),
        "block_min": int(status["block_min"]),
        "g_kwh": float(status["g_kwh"]),
        "charge_kw": float(status["charge_kw"]),
        "reserve_kwh":
            float(status["min_soc_frac"]) * float(status["g_kwh"]),
        "strict_tariff_coverage": bool(
            status.get("strict_tariff_coverage", False)
        ),
    }
    if status.get("time_model", "uniform") == "event":
        from event_pricer_network import EventExpandedNetwork

        network = EventExpandedNetwork(
            problem, prices, arc_mode="lazy", **kwargs
        )
    else:
        network = exact.ExpandedNetwork(problem, prices, **kwargs)
    return problem, prices, network


def _candidate_record(candidate, prices, status, iteration):
    if "_event_record" in candidate:
        record = deepcopy(candidate["_event_record"])
    else:
        temporary = {
            "trips": list(candidate["trips"]),
            "route_nodes": list(candidate["route_nodes"]),
            "charging_stops": deepcopy(candidate["charging_stops"]),
            "expanded_grid_charging_stops": deepcopy(
                candidate["_expanded_grid_charging"]
            ),
            "cost": 0.0,
        }
        costs = realized_costs(
            temporary,
            candidate["_continuous_mapping"],
            station_prices=prices,
        )
        record = {
            **temporary,
            "cost": float(costs["recomputed_expanded_grid_cost"]),
            "expanded_grid_cost":
                float(costs["recomputed_expanded_grid_cost"]),
            "continuous_realized_cost":
                float(costs["continuous_realized_cost"]),
            "continuous_realized_charging_blocks":
                costs["continuous_realized_charging_blocks"],
            "continuous_realized_charging_blocks_json_bytes": len(
                json.dumps(
                    costs["continuous_realized_charging_blocks"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ),
            "cost_semantics": "expanded_grid_cost",
            "master_cost_semantics": "expanded_grid_cost",
            "continuous_cost_pricing_certified": False,
            "physical_realization": {
                key: value
                for key, value in candidate[
                    "_continuous_mapping"
                ].items()
                if key != "trace"
            },
        }
        record["physical_realization"].update({
            "continuous_realized_charging_blocks_sha256":
                costs["continuous_realized_charging_blocks_sha256"],
            "continuous_realized_charging_blocks_schema":
                BLOCK_SCHEDULE_SCHEMA,
        })
    record.update({
        "found_fleet_phase_iteration": iteration,
        "origin": "lexicographic_phase_2_pricing",
        "cost_tariff_sha256":
            (status.get("provenance") or {}).get("prices_sha256"),
    })
    return record


def certify(args) -> dict:
    output = Path(args.out).expanduser().resolve()
    added_journal = Path(str(output) + ".added.columns.jsonl")
    iteration_log = Path(str(output) + ".iters.csv")
    for path in (output, added_journal, iteration_log):
        if os.path.lexists(path):
            raise FileExistsError(path)

    result_path = Path(args.result).expanduser().resolve(strict=True)
    source_status_bytes = result_path.read_bytes()
    source_status = json.loads(source_status_bytes)
    source_journal = resolve_pool_journal(
        result_path, source_status
    ).resolve(strict=True)
    source_journal_bytes = source_journal.read_bytes()
    source_result_sha256 = _sha256_bytes(source_status_bytes)
    source_journal_sha256 = _sha256_bytes(source_journal_bytes)
    if source_status.get("certified_rc_optimal") is not True:
        raise ValueError("source combined-cost CG is not certified")
    final = source_status.get("final") or {}
    if float(final.get("artificials", math.inf)) > 1e-7:
        raise ValueError("source combined-cost CG retains artificials")
    routes, trips = load_bound_pool_bytes(
        source_status, source_journal_bytes
    )
    pool = {}
    for route in routes:
        key = frozenset(route["trips"])
        current = pool.get(key)
        if (
            current is None
            or float(route["cost"]) < float(current["cost"]) - 1e-9
        ):
            pool[key] = route

    data_dir = Path(args.data_dir).expanduser().resolve()
    _problem, prices, network = _build_network(
        source_status, data_dir=data_dir
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stop_reason = "max_iters"
    certified = False
    minimum_reduced_cost = None
    lp = None
    added_count = 0
    with (
        added_journal.open("x") as journal_handle,
        iteration_log.open("x", newline="") as iteration_handle,
    ):
        writer = csv.DictWriter(
            iteration_handle,
            fieldnames=(
                "elapsed_s", "iteration", "fleet_lp",
                "minimum_reduced_cost", "pool_columns",
                "added_columns",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        flush_and_fsync(iteration_handle)
        for iteration in range(1, args.max_iters + 1):
            elapsed = time.perf_counter() - started
            if (
                args.wall_limit_s is not None
                and elapsed >= args.wall_limit_s
            ):
                stop_reason = "wall_limit"
                break
            lp = solve_fleet_master(list(trips), list(pool.values()))
            batch = network.sink_predecessor_route_batch(
                lp.trip_duals,
                limit=args.columns_per_iter,
                objective="fleet-only",
            )
            minimum_reduced_cost = (
                float(batch[0]["rc"]) if batch else None
            )
            added_this_iteration = 0
            if (
                minimum_reduced_cost is not None
                and minimum_reduced_cost < -args.rc_tolerance
            ):
                for candidate in batch:
                    key = frozenset(candidate["trips"])
                    if key in pool:
                        continue
                    record = _candidate_record(
                        candidate, prices, source_status, iteration
                    )
                    pool[key] = record
                    journal_handle.write(json.dumps(record) + "\n")
                    added_this_iteration += 1
                if added_this_iteration:
                    flush_and_fsync(journal_handle)
                    added_count += added_this_iteration
            writer.writerow({
                "elapsed_s": f"{time.perf_counter() - started:.6f}",
                "iteration": iteration,
                "fleet_lp": f"{lp.objective:.12g}",
                "minimum_reduced_cost": (
                    "" if minimum_reduced_cost is None
                    else f"{minimum_reduced_cost:.12g}"
                ),
                "pool_columns": len(pool),
                "added_columns": added_this_iteration,
            })
            flush_and_fsync(iteration_handle)
            if minimum_reduced_cost is None:
                stop_reason = "no_path"
                break
            if minimum_reduced_cost >= -args.rc_tolerance:
                certified = True
                stop_reason = "certified"
                break
            if added_this_iteration == 0:
                stop_reason = "degenerate_stall"
                break
        else:
            iteration = args.max_iters

    wall_s = time.perf_counter() - started
    if lp is None:
        raise RuntimeError("fleet phase stopped before its first master solve")
    dual_lower_bound = (
        sum(lp.trip_duals.values())
        - len(trips) * float(args.rc_tolerance)
        if certified else None
    )
    if (
        certified
        and dual_lower_bound is not None
        and dual_lower_bound > lp.objective + 1e-6
    ):
        raise RuntimeError("fleet dual bound exceeds primal objective")
    if (
        result_path.read_bytes() != source_status_bytes
        or source_journal.read_bytes() != source_journal_bytes
    ):
        raise RuntimeError("source CG artifacts changed during certification")
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "certified": certified,
        "certificate_scope":
            "fleet_lp_lower_bound_in_named_discrete_route_space"
            if certified else "uncertified_fleet_phase",
        "objective_definition":
            "minimize_sum_real_route_variables_each_coefficient_exactly_one",
        "fleet_lp_primal": lp.objective,
        "fleet_lp_lower_bound": dual_lower_bound,
        "primal_dual_gap": (
            lp.objective - dual_lower_bound
            if dual_lower_bound is not None else None
        ),
        "minimum_reduced_cost": minimum_reduced_cost,
        "pricing_tolerance": float(args.rc_tolerance),
        "iterations": iteration,
        "stop_reason": stop_reason,
        "max_row_violation": lp.max_row_violation,
        "max_bound_violation": lp.max_bound_violation,
    }
    certificate["certificate_sha256"] = _sha256_bytes(
        json.dumps(
            certificate,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    result = {
        "schema": SCHEMA,
        "phase": 2,
        "objective": "lexicographic-fleet-phase-2",
        "source_cg": {
            "result": str(result_path),
            "result_sha256": source_result_sha256,
            "journal": str(source_journal),
            "journal_sha256": source_journal_sha256,
            "certified": source_status.get("certified_rc_optimal"),
            "stop_reason": source_status.get("stop_reason"),
            "iterations": source_status.get("iterations"),
            "wall_s": source_status.get("wall_s"),
            "pool_columns": source_status.get("columns"),
        },
        "representation": {
            key: source_status.get(key)
            for key in (
                "time_model", "soc_step", "block_min",
                "g_kwh", "charge_kw", "min_soc_frac",
            )
        },
        "certificate": certificate,
        "fleet_lp_lower_bound": dual_lower_bound,
        "source_unique_pool_columns": len({
            frozenset(route["trips"]) for route in routes
        }),
        "phase_2_pool_columns": len(pool),
        "phase_2_added_columns": added_count,
        "phase_2_added_columns_journal": str(added_journal),
        "phase_2_added_columns_journal_sha256":
            _sha256_bytes(added_journal.read_bytes()),
        "iteration_log": str(iteration_log),
        "iteration_log_sha256":
            _sha256_bytes(iteration_log.read_bytes()),
        "wall_s": wall_s,
        "peak_rss_mb": exact._peak_rss_mb(),
        "network_metrics": (
            network.metrics()
            if hasattr(network, "metrics")
            else {
                "time_model": "uniform",
                "dag_nodes": len(network.node_meta),
                "dag_arcs": network.n_arcs,
            }
        ),
        "producer": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_dirty": bool(_git("status", "--porcelain")),
        },
    }
    atomic_write_json(output, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--columns-per-iter", type=int, default=30)
    parser.add_argument("--wall-limit-s", type=float)
    parser.add_argument(
        "--rc-tolerance", type=float, default=PRICING_TOLERANCE,
    )
    args = parser.parse_args(argv)
    if args.max_iters <= 0 or args.columns_per_iter <= 0:
        parser.error("iteration and column limits must be positive")
    if not 0.0 < args.rc_tolerance <= PRICING_TOLERANCE:
        parser.error("--rc-tolerance must be in (0, 1e-9]")
    result = certify(args)
    print(json.dumps({
        "certified": result["certificate"]["certified"],
        "fleet_lp_lower_bound": result["fleet_lp_lower_bound"],
        "iterations": result["certificate"]["iterations"],
        "wall_s": result["wall_s"],
    }, sort_keys=True))
    return 0 if result["certificate"]["certified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

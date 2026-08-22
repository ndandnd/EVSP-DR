#!/usr/bin/env python3
"""Two-stage RAW finite-pool MIP using SciPy's bundled HiGHS backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack

from exact_cg_telemetry import peak_rss_bytes
from run_exact_pool_mip import (
    deduplicate_pool,
    file_sha256,
    fleet_bound_proves_incumbent,
    greedy_partition_start_indices,
    ordered_pool_sha256,
    prepare_strict_partition_pool,
    resolve_pool_journal,
    singleton_partition_indices,
    validate_final_selected_routes,
    verified_mip_code_identity,
    write_new_json,
)
from target_pool_feasibility import load_bound_pool_bytes


SCHEMA = "evsp-dr-raw-pool-two-stage-highs-v1"
STATUS = {
    0: "OPTIMAL",
    1: "TIME_LIMIT_OR_ITERATION_LIMIT",
    2: "INFEASIBLE",
    3: "UNBOUNDED",
    4: "SOLVER_ERROR",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _incidence(routes, trips):
    trip_position = {trip: index for index, trip in enumerate(trips)}
    rows = []
    columns = []
    for column, route in enumerate(routes):
        for trip in route["trips"]:
            rows.append(trip_position[trip])
            columns.append(column)
    return csr_matrix(
        (
            np.ones(len(rows), dtype=float),
            (np.asarray(rows), np.asarray(columns)),
        ),
        shape=(len(trips), len(routes)),
    )


def _selected_indices(raw):
    if raw.x is None:
        return []
    return [
        index for index, value in enumerate(raw.x)
        if float(value) > 0.5
    ]


def _bound(raw):
    value = getattr(raw, "mip_dual_bound", None)
    return (
        float(value)
        if value is not None and math.isfinite(float(value))
        else None
    )


def _gap(raw):
    value = getattr(raw, "mip_gap", None)
    return (
        float(value)
        if value is not None and math.isfinite(float(value))
        else None
    )


def solve(args) -> dict:
    output = Path(args.out).expanduser().resolve()
    if os.path.lexists(output):
        raise FileExistsError(output)
    source = Path(args.result).expanduser().resolve(strict=True)
    source_status_bytes = source.read_bytes()
    status = json.loads(source_status_bytes)
    journal = resolve_pool_journal(source, status).resolve(strict=True)
    journal_bytes = journal.read_bytes()
    source_status_sha256 = _sha256(source_status_bytes)
    source_journal_sha256 = _sha256(journal_bytes)
    routes, trips = load_bound_pool_bytes(status, journal_bytes)
    preparation_started = time.perf_counter()
    routes, physical_audit = prepare_strict_partition_pool(
        status,
        routes,
        data_dir=args.data_dir,
        reference_data_dir=args.reference_data_dir,
    )
    routes = deduplicate_pool(routes)
    preparation_wall_s = time.perf_counter() - preparation_started
    matrix = _incidence(routes, trips)
    constraints = LinearConstraint(
        matrix,
        np.ones(len(trips)),
        np.ones(len(trips)),
    )
    options = {
        "presolve": True,
        "time_limit": float(args.timelimit),
        "mip_rel_gap": float(args.mipgap),
    }
    integrality = np.ones(len(routes), dtype=np.uint8)
    bounds = Bounds(
        np.zeros(len(routes)),
        np.ones(len(routes)),
    )
    seed_partition = singleton_partition_indices(routes, trips)
    greedy = greedy_partition_start_indices(
        routes, trips, seed_partition
    )
    started = time.perf_counter()
    stage1 = milp(
        c=np.ones(len(routes)),
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    stage1_wall_s = time.perf_counter() - started
    stage1_selected = _selected_indices(stage1)
    incumbent_source = "highs"
    if not stage1_selected and greedy:
        stage1_selected = list(greedy)
        incumbent_source = "greedy_pool_partition"
    stage1_buses = len(stage1_selected) if stage1_selected else None
    stage1_bound = _bound(stage1)
    fleet_proven = (
        fleet_bound_proves_incumbent(
            stage1_buses, stage1_bound, int(stage1.status)
        )
        if stage1_buses is not None else False
    )
    remaining_s = max(
        0.0, float(args.timelimit) - stage1_wall_s
    )
    selected = list(stage1_selected)
    stage2 = None
    stage2_wall_s = 0.0
    if fleet_proven and remaining_s >= 1.0:
        fleet_row = csr_matrix(
            np.ones((1, len(routes)), dtype=float)
        )
        stage2_matrix = vstack([matrix, fleet_row], format="csr")
        lower = np.concatenate([
            np.ones(len(trips)), [float(stage1_buses)]
        ])
        stage2_constraints = LinearConstraint(
            stage2_matrix, lower, lower
        )
        variable_cost = np.asarray([
            float(route["cost"]) - 100000.0 for route in routes
        ])
        stage2_started = time.perf_counter()
        stage2 = milp(
            c=variable_cost,
            integrality=integrality,
            bounds=bounds,
            constraints=stage2_constraints,
            options={
                "presolve": True,
                "time_limit": remaining_s,
                "mip_rel_gap": float(args.mipgap),
            },
        )
        stage2_wall_s = time.perf_counter() - stage2_started
        stage2_selected = _selected_indices(stage2)
        if stage2_selected:
            selected = stage2_selected
            incumbent_source = "highs_stage2"
    selected_routes = [routes[index] for index in selected]
    physical_witness_valid = False
    if selected_routes:
        validate_final_selected_routes(
            status,
            trips,
            selected_routes,
            data_dir=args.data_dir,
            reference_data_dir=args.reference_data_dir,
            physical_pool_audit=physical_audit,
        )
        physical_witness_valid = True
    if (
        source.read_bytes() != source_status_bytes
        or journal.read_bytes() != journal_bytes
    ):
        raise RuntimeError("source CG artifacts changed during MIP")
    stage2_optimal = stage2 is not None and int(stage2.status) == 0
    optimality_scope = (
        "full_pool_lexicographic"
        if fleet_proven and stage2_optimal
        else "fleet_only" if fleet_proven else "none"
    )
    payload = {
        "schema": SCHEMA,
        "backend": "scipy_highs",
        "two_stage": True,
        "requested_timelimit_s": float(args.timelimit),
        "requested_mip_gap": float(args.mipgap),
        "threads_requested": int(args.threads),
        "native_thread_control_available": False,
        "thread_control_note":
            "SciPy milp does not expose HiGHS thread count; all arms use "
            "the same backend and process environment.",
        "status_name": STATUS.get(
            int(stage2.status if stage2 is not None else stage1.status),
            "UNKNOWN",
        ),
        "buses": len(selected) if selected else None,
        "incumbent_found": bool(selected),
        "incumbent_source": incumbent_source,
        "fleet_bound": stage1_bound,
        "fleet_proven": fleet_proven,
        "mip_gap": _gap(stage1),
        "optimality_scope": optimality_scope,
        "physical_witness_valid": physical_witness_valid,
        "selected_route_indices": selected,
        "selected_route_set_sha256": _sha256(json.dumps(
            sorted(
                [sorted(route["trips"]) for route in selected_routes]
            ),
            separators=(",", ":"),
        ).encode()),
        "stage1": {
            "status": int(stage1.status),
            "status_name": STATUS.get(int(stage1.status), "UNKNOWN"),
            "wall_s": stage1_wall_s,
            "buses": stage1_buses,
            "bound": stage1_bound,
            "mip_gap": _gap(stage1),
        },
        "stage2": {
            "executed": stage2 is not None,
            "status": int(stage2.status) if stage2 is not None else None,
            "status_name": (
                STATUS.get(int(stage2.status), "UNKNOWN")
                if stage2 is not None else None
            ),
            "wall_s": stage2_wall_s,
        },
        "runtime_s": stage1_wall_s + stage2_wall_s,
        "physical_pool_preparation_wall_s": preparation_wall_s,
        "peak_rss_mb": peak_rss_bytes() / (1024.0 * 1024.0),
        "pool_columns": len(routes),
        "pool_ordered_sha256": ordered_pool_sha256(routes),
        "physical_pool_audit": physical_audit,
        "source_result": str(source),
        "source_result_sha256": source_status_sha256,
        "source_journal": str(journal),
        "source_journal_sha256": source_journal_sha256,
        "source_cg_certified":
            status.get("certified_rc_optimal") is True,
        "source_cg_stop_reason": status.get("stop_reason"),
        "source_cg_iterations": status.get("iterations"),
        "source_cg_wall_s": status.get("wall_s"),
        "source_cg_peak_rss_mb": status.get("peak_rss_mb"),
        "code_identity": verified_mip_code_identity(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(output, payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--timelimit", type=float, default=1800)
    parser.add_argument("--mipgap", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--reference-data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = solve(args)
    print(json.dumps({
        key: result[key] for key in (
            "status_name", "buses", "fleet_bound", "fleet_proven",
            "optimality_scope", "physical_witness_valid",
        )
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

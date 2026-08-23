#!/usr/bin/env python3
"""Two-stage RAW finite-pool MIP using SciPy's bundled HiGHS backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import highspy
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

NATIVE_STATUS = {
    highspy.HighsModelStatus.kOptimal: 0,
    highspy.HighsModelStatus.kTimeLimit: 1,
    highspy.HighsModelStatus.kIterationLimit: 1,
    highspy.HighsModelStatus.kSolutionLimit: 1,
    highspy.HighsModelStatus.kInfeasible: 2,
    highspy.HighsModelStatus.kUnbounded: 3,
    highspy.HighsModelStatus.kUnboundedOrInfeasible: 4,
}


@dataclass
class NativeResult:
    status: int
    x: np.ndarray | None
    mip_dual_bound: float | None
    mip_gap: float | None
    message: str


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


def _native_milp(
    *,
    objective,
    matrix,
    row_lower,
    row_upper,
    time_limit,
    mip_gap,
    threads,
    start_indices,
) -> NativeResult:
    columns = matrix.tocsc()
    lp = highspy.HighsLp()
    lp.num_col_ = int(columns.shape[1])
    lp.num_row_ = int(columns.shape[0])
    lp.col_cost_ = np.asarray(objective, dtype=np.float64)
    lp.col_lower_ = np.zeros(columns.shape[1], dtype=np.float64)
    lp.col_upper_ = np.ones(columns.shape[1], dtype=np.float64)
    lp.row_lower_ = np.asarray(row_lower, dtype=np.float64)
    lp.row_upper_ = np.asarray(row_upper, dtype=np.float64)
    lp.integrality_ = [
        highspy.HighsVarType.kInteger for _ in range(columns.shape[1])
    ]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = columns.indptr.astype(np.int32)
    lp.a_matrix_.index_ = columns.indices.astype(np.int32)
    lp.a_matrix_.value_ = columns.data.astype(np.float64)
    highspy.Highs.resetGlobalScheduler(True)
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("time_limit", float(time_limit))
    solver.setOptionValue("mip_rel_gap", float(mip_gap))
    solver.setOptionValue("threads", int(threads))
    solver.setOptionValue("random_seed", 0)
    solver.passModel(lp)
    if start_indices:
        indices = np.asarray(start_indices, dtype=np.int32)
        values = np.ones(len(indices), dtype=np.float64)
        solver.setSolution(len(indices), indices, values)
    solver.run()
    model_status = solver.getModelStatus()
    info = solver.getInfo()
    solution = solver.getSolution()
    primal = (
        np.asarray(solution.col_value, dtype=float)
        if solution.value_valid else None
    )
    return NativeResult(
        status=NATIVE_STATUS.get(model_status, 4),
        x=primal,
        mip_dual_bound=(
            float(info.mip_dual_bound)
            if math.isfinite(float(info.mip_dual_bound)) else None
        ),
        mip_gap=(
            float(info.mip_gap)
            if math.isfinite(float(info.mip_gap)) else None
        ),
        message=solver.modelStatusToString(model_status),
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
    if args.solver == "native":
        stage1 = _native_milp(
            objective=np.ones(len(routes)),
            matrix=matrix,
            row_lower=np.ones(len(trips)),
            row_upper=np.ones(len(trips)),
            time_limit=args.timelimit,
            mip_gap=args.mipgap,
            threads=args.threads,
            start_indices=greedy,
        )
    else:
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
        if args.solver == "native":
            stage2 = _native_milp(
                objective=variable_cost,
                matrix=stage2_matrix,
                row_lower=lower,
                row_upper=lower,
                time_limit=remaining_s,
                mip_gap=args.mipgap,
                threads=args.threads,
                start_indices=stage1_selected,
            )
        else:
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
        "backend": (
            "highspy_native" if args.solver == "native"
            else "scipy_highs"
        ),
        "two_stage": True,
        "requested_timelimit_s": float(args.timelimit),
        "requested_mip_gap": float(args.mipgap),
        "threads_requested": int(args.threads),
        "native_thread_control_available": args.solver == "native",
        "thread_control_note": (
            "Native HiGHS threads set explicitly."
            if args.solver == "native"
            else "SciPy milp does not expose HiGHS thread count."
        ),
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
    parser.add_argument(
        "--solver", choices=("native", "scipy"), default="native"
    )
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

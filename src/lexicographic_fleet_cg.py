"""Three-phase exact CG for a certified expanded-grid fleet LP bound."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye, hstack, vstack

from config import BUS_COST_KX
from durable_io import DurableFileError, flush_and_fsync
from expanded_path_realization import BLOCK_SCHEDULE_SCHEMA, realized_costs
from master_lp_scipy import build_route_incidence

SCHEMA = "evsp-dr-lexicographic-fleet-cg-v1"
CERTIFICATE_SCHEMA = "evsp-dr-lexicographic-cg-phase-certificate-v1"
TOLERANCE = 1e-7; PRICING_TOLERANCE = 1e-9

@dataclass(frozen=True)
class PhaseLP:
    objective: float
    route_values: tuple[float, ...]
    artificial_values: tuple[float, ...]
    trip_duals: dict[int, float]
    fleet_dual: float | None
    method: str
    max_row_violation: float
    max_bound_violation: float
    @property
    def route_weight(self):
        return float(sum(self.route_values))
    @property
    def artificial_total(self):
        return float(sum(self.artificial_values))

def _canonical(payload):
    return json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _solve_master(trips, routes, phase, fleet_optimum=None, method="highs-ds"):
    incidence=build_route_incidence(trips,[route["trips"] for route in routes])
    n_trips, n_routes = incidence.shape
    if phase == 1:
        matrix=hstack([incidence,eye(n_trips,format="csr")],format="csr")
        objective=np.concatenate([np.zeros(n_routes),np.ones(n_trips)])
        rhs = np.ones(n_trips)
    elif phase == 2:
        if not n_routes:
            raise RuntimeError("fleet phase has no real routes")
        matrix=incidence;objective=np.ones(n_routes);rhs=np.ones(n_trips)
    elif phase == 3:
        if fleet_optimum is None or not math.isfinite(fleet_optimum):
            raise ValueError("charging phase requires a finite fleet optimum")
        matrix=vstack([incidence,np.ones((1,n_routes))],format="csr")
        objective=np.asarray([float(route["cost"])-BUS_COST_KX for route in routes])
        rhs = np.concatenate([np.ones(n_trips), [fleet_optimum]])
    else:
        raise ValueError(f"unknown lexicographic phase: {phase}")
    solved=linprog(c=objective,A_eq=matrix,b_eq=rhs,bounds=(0.0,None),
                   method=method,options={"presolve":True})
    if not solved.success or solved.x is None:
        raise RuntimeError(
            f"lexicographic master failed in phase {phase}: "
            f"status={solved.status}, message={solved.message}"
        )
    primal = np.asarray(solved.x, dtype=float)
    dual = np.asarray(solved.eqlin.marginals, dtype=float)
    if not np.isfinite(primal).all() or not np.isfinite(dual).all():
        raise RuntimeError("lexicographic master returned non-finite values")
    maximum_bound_violation = max(0.0, -float(np.min(primal)))
    maximum_row_violation = float(np.max(np.abs(matrix @ primal - rhs)))
    if maximum_bound_violation>TOLERANCE or maximum_row_violation>TOLERANCE:
        raise RuntimeError(
            "lexicographic master violated a primal constraint: "
            f"row={maximum_row_violation}, bound={maximum_bound_violation}"
        )
    route_values = tuple(float(value) for value in primal[:n_routes])
    artificial_values=tuple(map(float,primal[n_routes:])) if phase==1 else ()
    return PhaseLP(
        objective=float(solved.fun),
        route_values=route_values,
        artificial_values=artificial_values,
        trip_duals={
            trip: float(dual[index]) for index, trip in enumerate(trips)
        },
        fleet_dual=float(dual[-1]) if phase == 3 else None,
        method=method,
        max_row_violation=maximum_row_violation,
        max_bound_violation=maximum_bound_violation,
    )

def _candidate_record(candidate, prices, tariff_sha, phase, iteration):
    temporary={"trips":candidate["trips"],"cost":0.0,"route_nodes":candidate["route_nodes"],
               "charging_stops":candidate["charging_stops"],
               "expanded_grid_charging_stops":candidate["_expanded_grid_charging"]}
    mapping=candidate["_continuous_mapping"];costs=realized_costs(temporary,mapping,station_prices=prices)
    combined_cost=float(costs["recomputed_expanded_grid_cost"])
    record = {
        **temporary,
        "cost": combined_cost,
        "expanded_grid_cost": combined_cost,
        "continuous_realized_cost": costs["continuous_realized_cost"],
        "continuous_realized_charging_blocks":
            costs["continuous_realized_charging_blocks"],
        "continuous_realized_charging_blocks_json_bytes": len(json.dumps(
            costs["continuous_realized_charging_blocks"],
            sort_keys=True, separators=(",", ":"),
        ).encode()),
        "cost_semantics": "expanded_grid_cost",
        "master_cost_semantics": "expanded_grid_cost",
        "continuous_cost_pricing_certified": False,
        "cost_tariff_sha256": tariff_sha,
        "charges_started": candidate["charges_started"],
        "found_iter": iteration,
        "found_lexicographic_phase": phase,
        "physical_realization": {key:value for key,value in mapping.items() if key!="trace"},
    }
    record["physical_realization"].update({
        "continuous_realized_charging_blocks_sha256":
            costs["continuous_realized_charging_blocks_sha256"],
        "continuous_realized_charging_blocks_schema": BLOCK_SCHEDULE_SCHEMA,
        "continuous_cost_pricing_certified": False,
    })
    return record
def _certificate(phase, lp, *, pricing_certified, min_rc, iterations,
                 columns, fixed_optima, identity, rc_tolerance):
    names={1:"artificial_elimination",2:"fleet_only_minimization",3:"charging_cost_at_fixed_fleet"}
    objectives={1:"minimize_sum_artificial_variables_real_route_coefficient_zero",2:"minimize_sum_real_route_variables_each_coefficient_exactly_one",3:"minimize_expanded_grid_charging_and_charge_start_cost"}
    reduced_costs={1:"0 - sum(trip_duals_on_route)",2:"1 - sum(trip_duals_on_route)",3:"expanded_grid_charging_cost - sum(trip_duals_on_route) - fleet_equality_dual"}
    zero_artificial = lp is not None and lp.artificial_total <= TOLERANCE
    certified = pricing_certified and (phase != 1 or zero_artificial)
    dual_bound = None
    if lp is not None and pricing_certified:
        dual_bound = sum(lp.trip_duals.values()) - len(lp.trip_duals)*rc_tolerance
        if phase == 3:
            fleet = fixed_optima["phase_2_fleet_optimum"]
            dual_bound = sum(lp.trip_duals.values()) + fleet*(lp.fleet_dual-rc_tolerance)
    payload = {
        "schema": CERTIFICATE_SCHEMA,
        "phase": phase,
        "name": names[phase],
        "identity": identity,
        "objective_definition": objectives[phase],
        "fixed_optima": fixed_optima,
        "objective_value": lp.objective if lp is not None else None,
        "dual_objective_lower_bound": dual_bound,
        "primal_dual_gap": lp.objective-dual_bound if dual_bound is not None else None,
        "route_weight": lp.route_weight if lp is not None else None,
        "artificial_mass": lp.artificial_total if lp is not None else None,
        "minimum_reduced_cost": min_rc,
        "pricing_tolerance": rc_tolerance,
        "reduced_cost_definition": reduced_costs[phase],
        "pricing_certified": pricing_certified,
        "certified": certified,
        "zero_artificial_mass_certified":
            bool(pricing_certified and zero_artificial),
        "trip_duals": {
            str(trip): value for trip, value in (lp.trip_duals.items() if lp else ())
        },
        "fleet_equality_dual": lp.fleet_dual if lp is not None else None,
        "master_method": lp.method if lp is not None else None,
        "iterations": iterations,
        "pool_columns": columns,
        "max_row_violation": lp.max_row_violation if lp is not None else None,
        "max_bound_violation": lp.max_bound_violation if lp is not None else None,
        "continuous_cost_pricing_certified": False,
    }
    if phase == 1:
        payload["certificate_scope"] = ("zero_artificial_mass_in_expanded_route_space"
                                        if certified else "uncertified_artificial_elimination")
    elif phase == 2:
        payload.update({
            "certificate_scope": ("fleet_lp_lower_bound_in_expanded_route_space"
                                  if certified else "uncertified_fleet_phase"),
            "real_route_objective_coefficient": 1.0,
            "charging_terms_in_objective": False,
        })
        if certified: payload["fleet_lp_lower_bound"] = dual_bound
    else:
        payload["certificate_scope"] = ("expanded_grid_charging_cost_at_fixed_certified_fleet_lp_optimum"
                                        if certified else "uncertified_charging_phase")
    return payload

def _run_phase(args, phase, trips, pool, net, prices, tariff_sha,
               fleet_optimum, started, journal, iterations, identity):
    preferred_method = 0
    methods = ("highs-ds", "highs-ipm", "highs")
    final_lp = None
    min_rc = None
    pricing_certified = False
    stop_reason = "max_iters"; iteration = 0
    rc_tolerance = min(float(args.rc_eps), PRICING_TOLERANCE)
    for iteration in range(1, args.max_iters + 1):
        if (
            args.wall_limit_s is not None
            and time.perf_counter() - started >= args.wall_limit_s
        ):
            stop_reason = "wall_limit"; iteration -= 1
            break
        routes = list(pool.values())
        final_lp = _solve_master(
            trips, routes, phase, fleet_optimum, methods[preferred_method],
        )
        mode = {
            1: "artificial-elimination",
            2: "fleet-only",
            3: "charging-cost",
        }[phase]
        batch = net.k_best_routes(
            final_lp.trip_duals,
            k=args.columns_per_iter,
            objective=mode,
            route_dual=final_lp.fleet_dual or 0.0,
        )
        min_rc = float(batch[0]["rc"]) if batch else None
        iterations.writerow({
            "phase": phase,
            "iteration": iteration,
            "objective": f"{final_lp.objective:.12g}",
            "route_weight": f"{final_lp.route_weight:.12g}",
            "artificial_mass": f"{final_lp.artificial_total:.12g}",
            "minimum_reduced_cost": (
                f"{min_rc:.12g}" if min_rc is not None else ""
            ),
            "pool_columns": len(pool),
        })
        flush_and_fsync(iterations.writerows_handle)
        if min_rc is None:
            stop_reason = "no_path"
            break
        if min_rc >= -rc_tolerance:
            pricing_certified = True
            stop_reason = "certified"
            break
        added = 0
        for candidate in batch:
            record = _candidate_record(
                candidate, prices, tariff_sha, phase, iteration,
            )
            key = frozenset(record["trips"])
            if (
                key not in pool
                or record["cost"] < pool[key]["cost"] - 1e-9
            ):
                pool[key] = record
                if journal is not None:
                    journal.write(json.dumps(record) + "\n")
                added += 1
        if journal is not None and added:
            flush_and_fsync(journal)
        if added:
            preferred_method = 0
        elif preferred_method + 1 < len(methods):
            preferred_method += 1
        else:
            stop_reason = "degenerate_stall"
            break
    fixed = {}
    if phase >= 2:
        fixed["phase_1_artificial_optimum"] = 0.0
    if phase == 3:
        fixed["phase_2_fleet_optimum"] = fleet_optimum
    certificate = _certificate(
        phase,
        final_lp,
        pricing_certified=pricing_certified,
        min_rc=min_rc,
        iterations=iteration,
        columns=len(pool),
        fixed_optima=fixed,
        identity=identity,
        rc_tolerance=rc_tolerance,
    )
    certificate["stop_reason"] = stop_reason
    certificate["certificate_sha256"] = hashlib.sha256(
        _canonical(certificate)
    ).hexdigest()
    return certificate

class _IterationWriter:
    fieldnames = (
        "phase", "iteration", "objective", "route_weight",
        "artificial_mass", "minimum_reduced_cost", "pool_columns",
    )

    def __init__(self, handle):
        self.writerows_handle = handle
        self.writer = csv.DictWriter(
            handle, fieldnames=self.fieldnames, lineterminator="\n",
        )
        self.writer.writeheader()
        flush_and_fsync(handle)

    def writerow(self, row):
        self.writer.writerow(row)

def run_lexicographic_fleet_cg(args):
    import exact_pricer_expanded as exact

    unsupported = (
        args.master_sense != "partition"
        or args.resume
        or args.diversify_rounds
        or args.snapshot_at_minutes
        or args.phase_telemetry is not None
        or args.stall_window_min is not None
        or getattr(args, "validated_seed_routes", None) is not None
        or not math.isfinite(float(args.rc_eps)) or float(args.rc_eps) < 0.0
    )
    if unsupported:
        raise ValueError(
            "lexicographic-fleet requires partition mode without resume, "
            "snapshots, telemetry, diversification, stalls, or injected seeds"
        )
    output = Path(args.out) if args.out else None
    journal_path = Path(str(output) + ".columns.jsonl") if output else None
    iteration_path = Path(str(output) + ".lexicographic.iters.csv") if output else None
    certificate_path = Path(str(output) + ".phase-certificates.jsonl") if output else None
    persisted = [
        path for path in (output, journal_path, iteration_path, certificate_path)
        if path is not None and path.exists()
    ]
    if persisted:
        raise DurableFileError(
            "refusing to overwrite lexicographic artifacts: "
            + ", ".join(map(str, persisted))
        )
    problem = exact.build_problem(
        exact.DATA_DIR, args.csv,
        max_station_to_trip_wait_min=exact.HORIZON_MIN,
    )
    prices = exact.load_station_hourly_prices(
        exact.DATA_DIR / args.prices_csv, exact.CHARGING_STATIONS,
    )
    net = exact.ExpandedNetwork(
        problem, prices,
        soc_step=args.soc_step,
        block_min=args.block_min,
        g_kwh=args.g_kwh,
        charge_kw=args.charge_kw,
        reserve_kwh=args.min_soc_frac * args.g_kwh,
        strict_tariff_coverage=args.strict_tariff_coverage,
    )
    provenance = exact._provenance(args)
    identity={"git_commit":provenance.get("git_commit"),"git_dirty":provenance.get("git_dirty"),"instance_sha256":provenance["instance_sha256"],"prices_sha256":provenance["prices_sha256"],"reference_sha256":provenance["reference_sha256"],"deadhead_sha256":provenance["deadhead_sha256"],"csv":args.csv,"prices_csv":args.prices_csv,"soc_step":args.soc_step,"block_min":args.block_min,"g_kwh":args.g_kwh,"charge_kw":args.charge_kw,"min_soc_frac":args.min_soc_frac,"strict_tariff_coverage":args.strict_tariff_coverage}
    pool = {}
    from exact_initial_pools import (
        build_heuristic_initial_pool,
        pool_provenance,
    )
    if args.initial_pool == "singletons":
        seeds, missing = exact.direct_singleton_seed_records(
            problem,
            g_kwh=args.g_kwh,
            soc_step=args.soc_step,
            reserve_kwh=args.min_soc_frac * args.g_kwh,
        )
        if missing:
            raise ValueError(
                f"singleton initialization missing {len(missing)} trips"
            )
        for seed in seeds:
            seed["cost_tariff_sha256"] = provenance["prices_sha256"]
        initial_pool_provenance = pool_provenance(
            "singletons",
            seeds,
            generator="exact_pricer_expanded.direct_singleton_seed_records",
        )
    elif args.initial_pool in {"matching", "greedy"}:
        seeds, initial_pool_provenance = build_heuristic_initial_pool(
            problem,
            prices,
            mode=args.initial_pool,
            depot=exact.DEPOT,
            stations=exact.STATIONS,
            g_kwh=args.g_kwh,
            charge_kw=args.charge_kw,
            reserve_kwh=args.min_soc_frac * args.g_kwh,
            soc_step=args.soc_step,
            block_min=args.block_min,
            tariff_sha256=provenance["prices_sha256"],
            instance_sha256=provenance["instance_sha256"],
        )
    else:
        seeds = []
        initial_pool_provenance = pool_provenance(
            "artificial", [], generator="none_artificial_variables_only",
        )
    initial_pool_sha256 = initial_pool_provenance["generated_pool_sha256"]
    identity.update({
        "initial_pool": args.initial_pool,
        "initial_pool_sha256": initial_pool_sha256,
    })
    for seed in seeds:
        seed["found_lexicographic_phase"] = 0
        pool[frozenset(seed["trips"])] = seed
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
    journal_handle = open(journal_path, "x") if journal_path else None
    iteration_handle = (
        open(iteration_path, "x", newline="") if iteration_path
        else open(os.devnull, "w")
    )
    certificate_handle = open(certificate_path, "x") if certificate_path else None
    try:
        if journal_handle:
            for seed in seeds:
                journal_handle.write(json.dumps(seed) + "\n")
            flush_and_fsync(journal_handle)
        iteration_writer = _IterationWriter(iteration_handle)
        started = time.perf_counter()
        certificates = []
        fleet_optimum = None
        for phase in (1, 2, 3):
            certificate = _run_phase(
                args, phase, list(problem.trips), pool, net, prices,
                provenance["prices_sha256"], fleet_optimum, started,
                journal_handle, iteration_writer, identity,
            )
            certificates.append(certificate)
            if certificate_handle:
                certificate_handle.write(_canonical(certificate).decode() + "\n")
                flush_and_fsync(certificate_handle)
            if not certificate["certified"]:
                break
            if phase == 2:
                fleet_optimum = certificate["route_weight"]
        wall_s = time.perf_counter() - started
    finally:
        for handle in (journal_handle, iteration_handle, certificate_handle):
            if handle:
                handle.close()
    result = {
        "schema": SCHEMA,
        "objective": "lexicographic-fleet",
        "csv": args.csv,
        "prices_csv": args.prices_csv,
        "soc_step": args.soc_step,
        "block_min": args.block_min,
        "master_sense": args.master_sense,
        "initial_pool": args.initial_pool,
        "initial_pool_sha256": initial_pool_sha256,
        "initial_pool_provenance": initial_pool_provenance,
        "trip_ids": list(problem.trips),
        "phases": certificates,
        "all_phases_certified": (
            len(certificates) == 3
            and all(record["certified"] for record in certificates)
        ),
        "phase_2_fleet_lp_bound": (
            certificates[1].get("fleet_lp_lower_bound")
            if len(certificates) >= 2 and certificates[1]["certified"] else None
        ),
        "phase_3_charging_cost": (
            certificates[2]["objective_value"]
            if len(certificates) >= 3 and certificates[2]["certified"] else None
        ),
        "columns": len(pool),
        "columns_journal": str(journal_path) if journal_path else None,
        "phase_iteration_log": str(iteration_path) if iteration_path else None,
        "phase_certificate_journal":
            str(certificate_path) if certificate_path else None,
        "wall_s": wall_s,
        "continuous_cost_pricing_certified": False,
        "provenance": provenance,
    }
    if journal_path:
        result["columns_journal_sha256"] = _sha256(journal_path)
    if certificate_path:
        result["phase_certificate_journal_sha256"] = _sha256(certificate_path)
    if iteration_path:
        result["phase_iteration_log_sha256"] = _sha256(iteration_path)
    return result

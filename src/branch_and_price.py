"""Exact Ryan--Foster branch-and-price for the expanded-grid E-VSP.

This adapter leaves the certified pricer unchanged. Active branch decisions
expand into exact required/forbidden shortest paths, with required trips joined
by min-plus DAG passes over all SOC states.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from audit_giro_known_columns import DEPOT, HORIZON_MIN, build_problem
from config import (
    BIG_M_PENALTY,
    CHARGE_RATE_KW,
    CHARGING_STATIONS,
)
from durable_io import atomic_write_json, read_jsonl_records
from exact_pricer_expanded import (
    DATA_DIR,
    ExpandedNetwork,
    _provenance,
    direct_singleton_seed_records,
    load_column_pool,
)
from expanded_path_realization import (
    BLOCK_SCHEDULE_SCHEMA,
    realize_expanded_path,
    realized_costs,
)
from master_lp_scipy import (
    RestrictedMasterSolveError,
    build_route_incidence,
    solve_restricted_master_lp,
)
from run_exact_pool_mip import validate_final_selected_routes
from utils_v2 import load_station_hourly_prices


class BranchAndPriceError(RuntimeError):
    """Base class for fail-closed experiment errors."""
class ValidationGateError(BranchAndPriceError):
    """A named validation gate failed."""
@dataclass(frozen=True, order=True)
class BranchConstraint:
    """One Ryan--Foster route-incidence constraint."""

    kind: str
    left: int
    right: int

    def __post_init__(self) -> None:
        if self.kind not in {"apart", "together"}:
            raise ValueError(f"unknown Ryan--Foster constraint {self.kind!r}")
        if self.left == self.right:
            raise ValueError("Ryan--Foster pair must contain distinct trips")
        if self.right < self.left:
            left, right = self.right, self.left
            object.__setattr__(self, "left", left)
            object.__setattr__(self, "right", right)
@dataclass(frozen=True)
class SearchNode:
    node_id: int
    constraints: tuple[BranchConstraint, ...]
    parent_bound: float | None

    @property
    def depth(self) -> int:
        return len(self.constraints)
@dataclass
class NodeResult:
    certified: bool
    infeasible: bool
    lp: object | None
    routes: list[dict]
    min_rc: float | None
    cg_iterations: int
    reason: str
def route_satisfies(
    route_trips: Iterable[int],
    constraints: Sequence[BranchConstraint],
) -> bool:
    """Whether a route incidence obeys every active branch decision."""
    present = set(route_trips)
    for constraint in constraints:
        left = constraint.left in present
        right = constraint.right in present
        if constraint.kind == "apart" and left and right:
            return False
        if constraint.kind == "together" and left != right:
            return False
    return True
def expand_constraint_assignments(
    constraints: Sequence[BranchConstraint],
) -> list[tuple[frozenset[int], frozenset[int]]]:
    """Expand active decisions into exact required/forbidden subproblems.

    The returned tuples are ``(required, forbidden)``.  Inconsistent and
    duplicate assignments are removed, so the number of actual shortest-path
    solves is at most ``2**len(constraints)``.
    """
    states: set[tuple[frozenset[int], frozenset[int]]] = {
        (frozenset(), frozenset())
    }
    for constraint in constraints:
        choices: tuple[tuple[set[int], set[int]], ...]
        if constraint.kind == "apart":
            choices = (
                (set(), {constraint.left}),
                (set(), {constraint.right}),
            )
        else:
            choices = (
                ({constraint.left, constraint.right}, set()),
                (set(), {constraint.left, constraint.right}),
            )
        next_states: set[tuple[frozenset[int], frozenset[int]]] = set()
        for required, forbidden in states:
            for add_required, add_forbidden in choices:
                new_required = set(required) | add_required
                new_forbidden = set(forbidden) | add_forbidden
                if new_required.isdisjoint(new_forbidden):
                    next_states.add(
                        (frozenset(new_required), frozenset(new_forbidden))
                    )
        states = next_states
        if not states:
            break
    return sorted(
        states,
        key=lambda state: (
            tuple(sorted(state[0])),
            tuple(sorted(state[1])),
        ),
    )
def pair_alphas(
    routes: Sequence[dict],
    route_values: Sequence[float],
    *,
    value_tolerance: float = 1e-10,
) -> dict[tuple[int, int], float]:
    """Compute Ryan--Foster pair co-assignment values."""
    alphas: dict[tuple[int, int], float] = {}
    for route, value in zip(routes, route_values):
        if value <= value_tolerance:
            continue
        trips = sorted(route["trips"])
        for left, right in combinations(trips, 2):
            pair = (left, right)
            alphas[pair] = alphas.get(pair, 0.0) + float(value)
    return alphas
def choose_ryan_foster_pair(
    routes: Sequence[dict],
    route_values: Sequence[float],
    *,
    tolerance: float = 1e-7,
) -> tuple[tuple[int, int], float] | None:
    """Choose the fractional pair nearest 0.5, or return ``None``."""
    fractional = [
        (abs(alpha - 0.5), pair, alpha)
        for pair, alpha in pair_alphas(routes, route_values).items()
        if tolerance < alpha < 1.0 - tolerance
    ]
    if not fractional:
        return None
    _, pair, alpha = min(fractional)
    return pair, alpha
def audit_exact_partition(
    trip_ids: Sequence[int],
    selected_routes: Sequence[dict],
) -> None:
    """Fail unless selected routes cover every trip exactly once."""
    expected = set(trip_ids)
    counts = Counter()
    for ordinal, route in enumerate(selected_routes, start=1):
        trips = list(route.get("trips") or [])
        if not trips or len(trips) != len(set(trips)):
            raise ValidationGateError(
                f"G5: selected route {ordinal} is empty or repeats a trip"
            )
        unknown = set(trips) - expected
        if unknown:
            raise ValidationGateError(
                f"G5: selected route {ordinal} has unknown trips "
                f"{sorted(unknown)[:10]}"
            )
        counts.update(trips)
    bad = {trip: counts[trip] for trip in trip_ids if counts[trip] != 1}
    if bad:
        raise ValidationGateError(
            f"G5: selected routes are not an exact partition: "
            f"{list(bad.items())[:15]}"
        )
def assert_integral_solution(
    trip_ids: Sequence[int],
    routes: Sequence[dict],
    route_values: Sequence[float],
    *,
    tolerance: float = 1e-7,
) -> list[dict]:
    """G4: certify that a pair-integral LP solution is route-integral."""
    nonintegral = [
        (index, value)
        for index, value in enumerate(route_values)
        if abs(float(value) - round(float(value))) > tolerance
    ]
    if nonintegral:
        raise ValidationGateError(
            "G4: every pair alpha is integral but route values are not: "
            f"{nonintegral[:10]}"
        )
    selected = [
        route
        for route, value in zip(routes, route_values)
        if round(float(value)) == 1
    ]
    audit_exact_partition(trip_ids, selected)
    return selected
def assert_child_bound(
    parent_bound: float,
    child_bound: float,
    *,
    tolerance: float = 1e-5,
) -> None:
    """G2 runtime assertion: branch restrictions cannot improve the LP."""
    allowed = max(tolerance, abs(parent_bound) * 1e-10)
    if child_bound < parent_bound - allowed:
        raise ValidationGateError(
            "G2: child LP bound decreased: "
            f"parent={parent_bound:.12f}, child={child_bound:.12f}, "
            f"tolerance={allowed:.3g}"
        )
class ConstrainedDAGPricer:
    """Exact shortest paths under required and forbidden trip sets."""
    def __init__(self, network):
        self.network = network
        self.problem = network.problem
        self.sink = network.SINK
        self.position = {
            node: index for index, node in enumerate(network.topo)
        }
        self.trip_nodes: dict[int, tuple[int, ...]] = {}
        for node, (kind, key, _level) in enumerate(network.node_meta):
            if kind == "trip":
                self.trip_nodes.setdefault(key, []).append(node)
        self.trip_nodes = {
            trip: tuple(nodes) for trip, nodes in self.trip_nodes.items()
        }
    def _ordered_required(self, required: frozenset[int]) -> tuple[int, ...]:
        unknown = required - set(self.problem.trips)
        if unknown:
            raise ValueError(f"required trips are outside the instance: {unknown}")
        trip_position = {
            trip: index for index, trip in enumerate(self.problem.trips)
        }
        return tuple(
            sorted(
                required,
                key=lambda trip: (
                    self.problem.start_min[trip],
                    trip_position[trip],
                ),
            )
        )
    def _relax_segment(
        self,
        starts: dict[int, float],
        *,
        target_trip: int | None,
        required: frozenset[int],
        forbidden: frozenset[int],
        dense_duals: Sequence[float],
        parent: list[int | None],
    ) -> tuple[list[float], dict[int, float]]:
        """Run one DAG segment, stopping at all SOC states of target_trip."""
        inf = float("inf")
        values = [inf] * len(self.network.node_meta)
        for node, value in starts.items():
            values[node] = value
        if not starts:
            return values, {}
        target_nodes = (set(self.trip_nodes.get(target_trip, ()))
                        if target_trip is not None else set())
        if target_trip is not None and not target_nodes:
            return values, {}
        start_position = min(self.position[node] for node in starts)
        end_position = (max(self.position[node] for node in target_nodes)
                        if target_trip is not None
                        else len(self.network.topo) - 2)
        if start_position > end_position:
            return values, {}
        for node in self.network.topo[start_position : end_position + 1]:
            value = values[node]
            if value == inf or (
                target_trip is not None and node in target_nodes
            ):
                continue
            for successor, base_cost, dual_index in self.network.out[node]:
                if successor == self.sink:
                    continue
                kind, key, _level = self.network.node_meta[successor]
                if kind == "trip":
                    if key in forbidden:
                        continue
                    if key in required and key != target_trip:
                        continue
                candidate = (value + float(base_cost)
                             - (dense_duals[dual_index]
                                if dual_index >= 0 else 0.0))
                if candidate < values[successor] - 1e-12:
                    values[successor] = candidate
                    parent[successor] = node
        reached = ({node: values[node] for node in target_nodes
                    if values[node] != inf}
                   if target_trip is not None else {})
        return values, reached
    @staticmethod
    def _reconstruct(parent: Sequence[int | None], endpoint: int) -> list[int]:
        path: list[int] = []
        node = endpoint
        while node != 0:
            path.append(node)
            predecessor = parent[node]
            if predecessor is None:
                raise BranchAndPriceError(
                    f"constrained-pricing parent chain breaks at node {node}"
                )
            node = predecessor
        path.reverse()
        return path
    def shortest_subproblem(
        self,
        alpha: dict[int, float],
        *,
        required: frozenset[int],
        forbidden: frozenset[int],
        max_candidates: int,
    ) -> list[dict]:
        """Solve one required/forbidden shortest-path subproblem exactly."""
        if not required.isdisjoint(forbidden):
            return []
        dense_duals = [
            float(alpha.get(trip, 0.0)) for trip in self.problem.trips
        ]
        parent: list[int | None] = [None] * len(self.network.node_meta)
        starts = {0: 0.0}
        for target in self._ordered_required(required):
            _values, starts = self._relax_segment(
                starts,
                target_trip=target,
                required=required,
                forbidden=forbidden,
                dense_duals=dense_duals,
                parent=parent,
            )
            if not starts:
                return []
        values, _unused = self._relax_segment(
            starts,
            target_trip=None,
            required=required,
            forbidden=forbidden,
            dense_duals=dense_duals,
            parent=parent,
        )
        sink_candidates = sorted(
            (values[node] + float(cost), node)
            for node, cost in self.network.sink_arcs
            if values[node] != float("inf")
        )
        candidates: dict[frozenset[int], dict] = {}
        scan_limit = max(200, 4 * max_candidates)
        for reduced_cost, endpoint in sink_candidates[:scan_limit]:
            path = self._reconstruct(parent, endpoint)
            trips = [
                self.network.node_meta[node][1]
                for node in path
                if self.network.node_meta[node][0] == "trip"
            ]
            trip_set = frozenset(trips)
            if (
                not required <= trip_set
                or trip_set & forbidden
                or not trips
            ):
                raise BranchAndPriceError(
                    "constrained shortest path violates its subproblem"
                )
            current = candidates.get(trip_set)
            if current is None or reduced_cost < current["rc"] - 1e-12:
                candidates[trip_set] = {
                    "rc": float(reduced_cost),
                    "trips": trips,
                    "path_nodes": path,
                }
        return sorted(candidates.values(), key=lambda route: route["rc"])[
            :max_candidates
        ]
    def price(
        self,
        alpha: dict[int, float],
        constraints: Sequence[BranchConstraint],
        *,
        max_candidates: int,
    ) -> tuple[list[dict], int]:
        """Price the union of all exact disjunctive subproblems."""
        assignments = expand_constraint_assignments(constraints)
        combined: dict[frozenset[int], dict] = {}
        for required, forbidden in assignments:
            routes = self.shortest_subproblem(
                alpha,
                required=required,
                forbidden=forbidden,
                max_candidates=max_candidates,
            )
            for route in routes:
                if not route_satisfies(route["trips"], constraints):
                    raise BranchAndPriceError(
                        "enumerated constrained route violates branch decisions"
                    )
                key = frozenset(route["trips"])
                current = combined.get(key)
                if current is None or route["rc"] < current["rc"] - 1e-12:
                    combined[key] = route
        return (
            sorted(combined.values(), key=lambda route: route["rc"])[
                :max_candidates
            ],
            len(assignments),
        )
class ExpandedRouteAdapter:
    """Convert a constrained DAG node path to the standard route record."""
    def __init__(self, network, prices, prices_sha256: str):
        self.network = network
        self.problem = network.problem
        self.prices = prices
        self.prices_sha256 = prices_sha256
    def materialize(self, candidate: dict) -> dict:
        path = candidate["path_nodes"]
        trips: list[int] = []
        runs: list[list] = []
        run = None
        sequence: list[tuple[str, object]] = []
        for node in path:
            kind, key, level = self.network.node_meta[node]
            if kind == "trip":
                if run is not None:
                    runs.append(run)
                    run = None
                trips.append(key)
                sequence.append(("trip", key))
            elif kind == "charge":
                station, block = key
                if (run is not None and run[0] == station
                        and block == run[2] + 1):
                    run[2] = block
                else:
                    if run is not None:
                        runs.append(run)
                    run = [station, block, block, level]
                marker = ("station", station)
                if not sequence or sequence[-1] != marker:
                    sequence.append(marker)
        if run is not None:
            runs.append(run)
        charging = {"stations": [], "cst": [], "cet": [], "kwh": []}
        for station, first_block, last_block, initial_level in runs:
            soc = self.network.grid[initial_level]
            for _block in range(first_block, last_block + 1):
                soc = self.network._charge_result(self.network._floor(soc))
            charging["stations"].append(station)
            charging["cst"].append(first_block * self.network.block_min)
            charging["cet"].append((last_block + 1) * self.network.block_min)
            charging["kwh"].append(
                round(soc - self.network.grid[initial_level], 6)
            )
        route_nodes = [DEPOT] + [value for _kind, value in sequence] + [DEPOT]
        expanded_grid_charging = {key: list(values)
                                  for key, values in charging.items()}
        realized, detail = realize_expanded_path(
            self.problem, {"trips": trips, "charging_stops": charging,
                           "route_nodes": route_nodes},
            g_kwh=self.network.g,
            charge_kw=self.network.charge_kw,
            reserve_kwh=self.network.reserve,
            soc_step=self.network.soc_step,
            block_min=self.network.block_min,
            arc_map=self.network.continuous_arc_map,
        )
        if realized is None:
            raise BranchAndPriceError(
                "constrained expanded path has no continuous realization: "
                f"{detail.get('reason')}"
            )
        return {
            "rc": candidate["rc"], "trips": trips,
            "charging_stops": realized["charging_stops"],
            "route_nodes": route_nodes,
            "charges_started": len(charging["stations"]),
            "_continuous_mapping": detail["mapping"],
            "_expanded_grid_charging": expanded_grid_charging,
        }
    def record(
        self,
        route: dict,
        duals: dict[int, float],
        *,
        found_iter: int,
        origin: str,
    ) -> dict:
        cost = float(route["rc"]) + sum(
            float(duals.get(trip, 0.0)) for trip in route["trips"]
        )
        record = {
            "trips": list(route["trips"]), "cost": cost,
            "route_nodes": list(route["route_nodes"]),
            "charging_stops": route["charging_stops"],
            "expanded_grid_charging_stops":
                route["_expanded_grid_charging"],
            "charges_started": int(route["charges_started"]),
            "found_iter": found_iter,
            "origin": origin,
        }
        mapping = route["_continuous_mapping"]
        costs = realized_costs(record, mapping, station_prices=self.prices)
        record.update({
            "expanded_grid_cost": cost,
            "continuous_realized_cost": costs["continuous_realized_cost"],
            "continuous_realized_charging_blocks":
                costs["continuous_realized_charging_blocks"],
            "continuous_realized_charging_blocks_json_bytes": len(json.dumps(
                costs["continuous_realized_charging_blocks"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()),
            "cost_semantics": "expanded_grid_cost",
            "master_cost_semantics": "expanded_grid_cost",
            "continuous_cost_pricing_certified": False,
            "cost_tariff_sha256": self.prices_sha256,
            "physical_realization": {
                key: value
                for key, value in mapping.items()
                if key != "trace"
            },
        })
        record["physical_realization"][
            "continuous_realized_charging_blocks_sha256"
        ] = costs["continuous_realized_charging_blocks_sha256"]
        record["physical_realization"][
            "continuous_realized_charging_blocks_schema"
        ] = BLOCK_SCHEDULE_SCHEMA
        return record
class BranchAndPriceSolver:
    """Depth-first exact branch-and-price driver."""
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.problem = build_problem(
            DATA_DIR,
            args.csv,
            max_station_to_trip_wait_min=HORIZON_MIN,
        )
        self.trips = list(self.problem.trips)
        self.provenance = _provenance(args)
        self.prices = load_station_hourly_prices(
            DATA_DIR / args.prices_csv,
            CHARGING_STATIONS,
        )
        build_started = time.monotonic()
        self.network = ExpandedNetwork(
            self.problem,
            self.prices,
            soc_step=args.soc_step,
            block_min=args.block_min,
            g_kwh=args.g_kwh,
            charge_kw=args.charge_kw,
            reserve_kwh=args.min_soc_frac * args.g_kwh,
            strict_tariff_coverage=False,
        )
        self.network_build_s = time.monotonic() - build_started
        self.pricer = ConstrainedDAGPricer(self.network)
        self.route_adapter = ExpandedRouteAdapter(
            self.network,
            self.prices,
            self.provenance["prices_sha256"],
        )
        self.pool: dict[frozenset[int], dict] = {}
        self._seed_singletons()
        if args.root_pool_result is not None:
            self._load_root_pool(args.root_pool_result)
        self.pricing_solves = 0
        self.master_solves = 0
        self.bound_assertions = 0
        self.integrality_certificates = 0
        self.integer_audits = 0
        self.nodes_explored = 0
        self.nodes_depth_capped = 0
        self.next_node_id = 1
        self.root_lp = None
        self.root_min_rc = None
        self.incumbent: dict | None = None
    def _seed_singletons(self) -> None:
        records, missing = direct_singleton_seed_records(
            self.problem,
            g_kwh=self.args.g_kwh,
            soc_step=self.args.soc_step,
            reserve_kwh=self.args.min_soc_frac * self.args.g_kwh,
        )
        if missing:
            raise ValidationGateError(
                "G5: direct singleton partition is incomplete: "
                f"{missing[:15]}"
            )
        for record in records:
            record["cost_tariff_sha256"] = self.provenance["prices_sha256"]
            key = frozenset(record["trips"])
            self.pool[key] = record
    def _load_root_pool(self, result_path: Path) -> None:
        path = result_path.expanduser().resolve()
        payload = json.loads(path.read_text())
        identity = ("csv", "prices_csv", "soc_step", "block_min", "g_kwh",
                    "charge_kw", "min_soc_frac")
        mismatches = [key for key in identity
                      if payload.get(key) != getattr(self.args, key)]
        prior_provenance = payload.get("provenance") or {}
        hashes = ("instance_sha256", "prices_sha256",
                  "reference_sha256", "deadhead_sha256")
        mismatches += [f"provenance.{key}" for key in hashes
                       if prior_provenance.get(key) != self.provenance.get(key)]
        if payload.get("root_certified") is not True:
            mismatches.append("root_certified")
        if mismatches:
            raise BranchAndPriceError(
                f"root-pool identity mismatch in {path}: {sorted(mismatches)}"
            )
        journal = Path(payload["columns_journal"])
        if not journal.is_absolute():
            journal = path.parent / journal
        loaded = load_column_pool(
            read_jsonl_records(journal, repair_trailing=False), self.trips
        )
        for key, record in loaded.items():
            current = self.pool.get(key)
            if current is None or record["cost"] < current["cost"] - 1e-9:
                self.pool[key] = record
        print(
            f"[B&P] loaded {len(loaded)} certified-root columns from {journal}",
            flush=True,
        )
    def _remaining_s(self) -> float | None:
        if self.args.wall_limit_s is None:
            return None
        return self.args.wall_limit_s - (time.monotonic() - self.started)
    def _expired(self, reserve_s: float = 0.0) -> bool:
        remaining = self._remaining_s()
        return remaining is not None and remaining <= reserve_s
    def _allowed_routes(
        self,
        constraints: Sequence[BranchConstraint],
    ) -> list[dict]:
        return [
            route
            for route in self.pool.values()
            if route_satisfies(route["trips"], constraints)
        ]
    def _solve_master(self, routes: Sequence[dict], method: str):
        incidence = build_route_incidence(
            self.trips,
            [route["trips"] for route in routes],
        )
        remaining = self._remaining_s()
        time_limit = None if remaining is None else max(0.001, remaining)
        self.master_solves += 1
        return solve_restricted_master_lp(
            trip_ids=self.trips,
            route_incidence=incidence,
            route_costs=[route["cost"] for route in routes],
            artificial_penalty=BIG_M_PENALTY,
            method=method,
            coverage_sense="partition",
            time_limit_s=time_limit,
        )
    def _price(
        self,
        duals: dict[int, float],
        constraints: Sequence[BranchConstraint],
    ) -> list[dict]:
        if not constraints:
            self.pricing_solves += 1
            return self.network.k_best_routes(
                duals,
                k=self.args.columns_per_iter,
            )
        routes, solves = self.pricer.price(
            duals,
            constraints,
            max_candidates=self.args.columns_per_iter,
        )
        self.pricing_solves += solves
        return routes
    def _solve_node(self, node: SearchNode) -> NodeResult:
        preferred_method = "highs-ds"
        stalled_once = False
        last_lp = None
        last_routes: list[dict] = []
        last_rc = None
        for iteration in range(1, self.args.max_cg_iters + 1):
            if self._expired():
                return NodeResult(
                    False, False, last_lp, last_routes, last_rc,
                    iteration - 1, "wall_limit",
                )
            routes = self._allowed_routes(node.constraints)
            try:
                lp = self._solve_master(routes, preferred_method)
            except RestrictedMasterSolveError as exc:
                return NodeResult(
                    False, False, last_lp, last_routes, last_rc,
                    iteration - 1, f"master_failed:{exc}",
                )
            last_lp, last_routes = lp, routes
            candidates = self._price(lp.trip_duals, node.constraints)
            min_rc = candidates[0]["rc"] if candidates else float("inf")
            last_rc = float(min_rc)
            if iteration == 1 or iteration % 25 == 0 or (
                min_rc >= -self.args.rc_eps
            ):
                print(
                    f"[B&P] node={node.node_id} depth={node.depth} "
                    f"cg={iteration} obj={lp.objective:.6f} "
                    f"weight={lp.route_weight:.9f} "
                    f"art={lp.artificial_total:.3g} rc={min_rc:.6g} "
                    f"cols={len(routes)}",
                    flush=True,
                )
            if min_rc >= -self.args.rc_eps:
                return NodeResult(
                    True,
                    lp.artificial_total > self.args.integrality_tol,
                    lp,
                    routes,
                    float(min_rc),
                    iteration,
                    "infeasible" if (
                        lp.artificial_total > self.args.integrality_tol
                    ) else "certified",
                )
            added = 0
            for candidate in candidates:
                if candidate["rc"] >= -self.args.rc_eps:
                    break
                route = (
                    self.route_adapter.materialize(candidate)
                    if "path_nodes" in candidate
                    else candidate
                )
                record = self.route_adapter.record(
                    route,
                    lp.trip_duals,
                    found_iter=iteration,
                    origin=(
                        "branch_constrained_pricing"
                        if node.constraints else "root_pricing"
                    ),
                )
                key = frozenset(record["trips"])
                current = self.pool.get(key)
                if current is None or (
                    record["cost"] < current["cost"] - 1e-9
                ):
                    self.pool[key] = record
                    added += 1
            if not added:
                if stalled_once:
                    raise BranchAndPriceError(
                        "negative constrained reduced cost persisted without "
                        "a new or cheaper incidence under alternate duals"
                    )
                stalled_once = True
                preferred_method = "highs-ipm"
            else:
                stalled_once = False
                preferred_method = "highs-ds"
        return NodeResult(
            False, False, last_lp, last_routes, last_rc,
            self.args.max_cg_iters, "max_cg_iters",
        )
    def _replay_status(self) -> dict:
        return {
            "csv": self.args.csv,
            "prices_csv": self.args.prices_csv,
            "soc_step": self.args.soc_step,
            "block_min": self.args.block_min,
            "g_kwh": self.args.g_kwh,
            "charge_kw": self.args.charge_kw,
            "min_soc_frac": self.args.min_soc_frac,
            "provenance": self.provenance,
        }
    def _update_incumbent(
        self,
        selected: Sequence[dict],
        *,
        source: str,
    ) -> None:
        audit_exact_partition(self.trips, selected)
        cost = float(sum(float(route["cost"]) for route in selected))
        if self.root_lp is not None and (
            cost < self.root_lp.objective - self.args.bound_tolerance
        ):
            raise ValidationGateError(
                "G5: integer candidate is below the certified root LP: "
                f"candidate={cost}, root={self.root_lp.objective}"
            )
        validate_final_selected_routes(
            self._replay_status(),
            self.trips,
            list(selected),
        )
        self.integer_audits += 1
        if self.incumbent is None or (
            cost < self.incumbent["cost"] - 1e-7
        ):
            self.incumbent = {
                "cost": cost,
                "fleet": len(selected),
                "routes": list(selected),
                "source": source,
            }
            print(
                f"[B&P] incumbent={len(selected)} buses "
                f"cost={cost:.6f} source={source}",
                flush=True,
            )
    def _singleton_incumbent(self) -> None:
        selected = []
        for trip in self.trips:
            route = self.pool.get(frozenset({trip}))
            if route is None:
                raise ValidationGateError(
                    f"G5: singleton incumbent lacks trip {trip}"
                )
            selected.append(route)
        self._update_incumbent(selected, source="direct_singletons")
    def _root_pool_mip_incumbent(self, routes: Sequence[dict]) -> None:
        if self.args.root_mip_s <= 0:
            return
        incidence = build_route_incidence(
            self.trips, [route["trips"] for route in routes]
        )
        ones_routes = np.ones(len(routes), dtype=np.uint8)
        ones_trips = np.ones(len(self.trips), dtype=float)
        result = milp(
            c=np.asarray([route["cost"] for route in routes], dtype=float),
            integrality=ones_routes,
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(incidence, ones_trips, ones_trips),
            options={
                "time_limit": float(self.args.root_mip_s),
                "mip_rel_gap": 0.0,
                "presolve": True,
            },
        )
        if result.x is None:
            print(
                f"[B&P] root-pool MIP found no incumbent: {result.message}",
                flush=True,
            )
            return
        selected = [route for route, value in zip(routes, result.x)
                    if value > 0.5]
        try:
            audit_exact_partition(self.trips, selected)
        except ValidationGateError:
            print(
                "[B&P] root-pool MIP returned no auditable integer incumbent",
                flush=True,
            )
            return
        self._update_incumbent(selected, source="root_pool_scipy_milp")
    def _new_child(
        self,
        parent: SearchNode,
        bound: float,
        constraint: BranchConstraint,
    ) -> SearchNode:
        child = SearchNode(
            self.next_node_id,
            parent.constraints + (constraint,),
            bound,
        )
        self.next_node_id += 1
        return child
    def _process_certified_node(
        self,
        node: SearchNode,
        result: NodeResult,
        stack: list[SearchNode],
        frontier_bounds: list[float],
    ) -> None:
        if result.infeasible:
            return
        lp = result.lp
        if node.parent_bound is not None:
            assert_child_bound(
                node.parent_bound,
                lp.objective,
                tolerance=self.args.bound_tolerance,
            )
            self.bound_assertions += 1
        if self.incumbent is not None and (
            lp.objective
            >= self.incumbent["cost"] - self.args.bound_tolerance
        ):
            return
        branch = choose_ryan_foster_pair(
            result.routes,
            lp.route_values,
            tolerance=self.args.integrality_tol,
        )
        if branch is None:
            selected = assert_integral_solution(
                self.trips,
                result.routes,
                lp.route_values,
                tolerance=self.args.integrality_tol,
            )
            self.integrality_certificates += 1
            self._update_incumbent(
                selected,
                source=f"integral_node_{node.node_id}",
            )
            return
        if node.depth >= self.args.max_depth:
            self.nodes_depth_capped += 1
            frontier_bounds.append(lp.objective)
            return
        (left, right), alpha = branch
        together = self._new_child(
            node,
            lp.objective,
            BranchConstraint("together", left, right),
        )
        apart = self._new_child(
            node,
            lp.objective,
            BranchConstraint("apart", left, right),
        )
        preferred, other = (
            (together, apart) if alpha >= 0.5 else (apart, together)
        )
        stack.append(other)
        stack.append(preferred)
    def _base_payload(self) -> dict:
        return {
            "schema": "evsp-dr-exact-branch-and-price-v1",
            "csv": self.args.csv, "prices_csv": self.args.prices_csv,
            "soc_step": self.args.soc_step, "block_min": self.args.block_min,
            "g_kwh": self.args.g_kwh, "charge_kw": self.args.charge_kw,
            "min_soc_frac": self.args.min_soc_frac, "master_sense": "partition",
            "column_pool_treatment": "RAW",
            "target_fleet": self.args.target_fleet, "trip_ids": self.trips,
            "pricing_certificate_scope":
                "conservative_expanded_grid_model_only",
            "provenance": self.provenance,
        }
    def run(self) -> dict:
        root = SearchNode(0, tuple(), None)
        root_result = self._solve_node(root)
        self.nodes_explored = 1
        if not root_result.certified or root_result.infeasible:
            raise ValidationGateError(
                "G1: root column generation did not certify a feasible LP: "
                f"{root_result.reason}"
            )
        self.root_lp = root_result.lp
        self.root_min_rc = root_result.min_rc
        if self.args.expected_root_weight is not None and not math.isclose(
            self.root_lp.route_weight,
            self.args.expected_root_weight,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValidationGateError(
                "G1: certified root route weight mismatch: "
                f"expected={self.args.expected_root_weight:.9f}, "
                f"observed={self.root_lp.route_weight:.9f}"
            )
        print(
            f"[B&P] G1 PASS root weight={self.root_lp.route_weight:.9f} "
            f"objective={self.root_lp.objective:.6f} "
            f"min_rc={self.root_min_rc:.6g}",
            flush=True,
        )
        if self.args.root_only:
            elapsed = time.monotonic() - self.started
            return {
                **self._base_payload(),
                "root_certified": True,
                "root_lp": {
                    "objective": self.root_lp.objective,
                    "route_weight": self.root_lp.route_weight,
                    "min_rc": self.root_min_rc, "columns": len(root_result.routes),
                    "cg_iterations": root_result.cg_iterations,
                },
                "best_integer_fleet": None, "best_integer_cost": None,
                "global_lower_bound": self.root_lp.objective,
                "gap": None, "proven_optimal": False, "nodes_explored": 1,
                "nodes_depth_capped": 0, "pricing_solves": self.pricing_solves,
                "master_solves": self.master_solves, "wall_s": elapsed,
                "network_build_s": self.network_build_s,
                "validation": {"G1": "pass"},
            }

        self._singleton_incumbent()
        self._root_pool_mip_incumbent(root_result.routes)
        stack: list[SearchNode] = []
        frontier_bounds: list[float] = []
        self._process_certified_node(root, root_result, stack, frontier_bounds)
        interrupted_reason = None
        while stack:
            if self.nodes_explored >= self.args.node_limit:
                interrupted_reason = "node_limit"
                break
            if self._expired():
                interrupted_reason = "wall_limit"
                break
            node = stack.pop()
            result = self._solve_node(node)
            self.nodes_explored += 1
            if not result.certified:
                frontier_bounds.append(
                    node.parent_bound
                    if node.parent_bound is not None else self.root_lp.objective
                )
                if result.reason == "wall_limit":
                    interrupted_reason = "wall_limit"
                    break
                continue
            self._process_certified_node(
                node, result, stack, frontier_bounds
            )

        open_bounds = [float(bound) for bound in frontier_bounds] + [
            float(
                node.parent_bound
                if node.parent_bound is not None
                else self.root_lp.objective
            )
            for node in stack
        ]
        tree_closed = (
            not frontier_bounds and not stack and interrupted_reason is None
        )
        proven = bool(tree_closed and self.incumbent is not None)
        if proven:
            global_bound = self.incumbent["cost"]
        elif open_bounds:
            global_bound = min(open_bounds)
        else:
            global_bound = self.root_lp.objective
        best_cost = self.incumbent["cost"] if self.incumbent else None
        gap = (
            max(0.0, best_cost - global_bound) / max(1.0, abs(best_cost))
            if best_cost is not None
            else None
        )
        elapsed = time.monotonic() - self.started
        return {
            **self._base_payload(),
            "root_certified": True,
            "root_lp": {
                "objective": self.root_lp.objective,
                "route_weight": self.root_lp.route_weight,
                "min_rc": self.root_min_rc, "columns": len(root_result.routes),
                "cg_iterations": root_result.cg_iterations,
            },
            "best_integer_fleet": self.incumbent["fleet"] if self.incumbent else None,
            "best_integer_cost": best_cost,
            "best_integer_source": self.incumbent["source"] if self.incumbent else None,
            "best_integer_routes": self.incumbent["routes"] if self.incumbent else [],
            "global_lower_bound": global_bound, "gap": gap,
            "proven_optimal": proven, "nodes_explored": self.nodes_explored,
            "nodes_depth_capped": self.nodes_depth_capped,
            "pricing_solves": self.pricing_solves, "master_solves": self.master_solves,
            "wall_s": elapsed,
            "network_build_s": self.network_build_s,
            "interrupted_reason": interrupted_reason,
            "open_frontier_nodes": len(frontier_bounds) + len(stack),
            "validation": {
                "G1": "pass",
                "G2_bound_assertions": self.bound_assertions,
                "G4_integrality_certificates": self.integrality_certificates,
                "G5_integer_audits": self.integer_audits,
            },
        }

    def write(self, result: dict, out: Path) -> None:
        out = out.expanduser().resolve()
        journal = Path(str(out) + ".columns.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() or journal.exists():
            raise FileExistsError(
                f"refusing to overwrite branch-and-price evidence: "
                f"{out} or {journal}"
            )
        temporary = journal.with_name(f".{journal.name}.tmp.{os.getpid()}")
        with temporary.open("x") as handle:
            for record in self.pool.values():
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, journal)
        result["columns"] = len(self.pool)
        result["columns_journal"] = str(journal)
        atomic_write_json(out, result)


def _validate_scope(csv_path: str, target_fleet: int) -> None:
    match = re.search(r"_k(\d{2})_", Path(csv_path).name)
    if match is None:
        raise ValueError("instance filename must contain _kNN_")
    scale = int(match.group(1))
    if scale not in {2, 3, 5}:
        raise ValueError("branch-and-price scope is restricted to k2/k3/k5")
    if scale != target_fleet:
        raise ValueError(
            f"--target-fleet {target_fleet} disagrees with k{scale:02d} input"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path relative to data/")
    parser.add_argument("--prices_csv", default="hourly_prices_flat.csv")
    parser.add_argument("--target-fleet", type=int, choices=(2, 3, 5), required=True)
    parser.add_argument("--soc-step", type=float, default=15.0)
    parser.add_argument("--block-min", type=int, default=10)
    parser.add_argument("--g-kwh", type=float, default=300.0)
    parser.add_argument("--charge-kw", type=float, default=CHARGE_RATE_KW)
    parser.add_argument("--min-soc-frac", type=float, default=0.0)
    parser.add_argument("--columns-per-iter", type=int, default=30)
    parser.add_argument("--rc-eps", type=float, default=1e-7)
    parser.add_argument("--integrality-tol", type=float, default=1e-7)
    parser.add_argument("--bound-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-cg-iters", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--node-limit", type=int, default=1000)
    parser.add_argument("--wall-limit-s", type=float, default=21600.0)
    parser.add_argument("--root-mip-s", type=float, default=60.0)
    parser.add_argument("--expected-root-weight", type=float)
    parser.add_argument("--root-only", action="store_true")
    parser.add_argument("--root-pool-result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    if os.environ.get("SLURM_JOB_ID"):
        print("[B&P] refusing cluster execution; local runs only", file=sys.stderr)
        return 2
    args = build_parser().parse_args(argv)
    try:
        _validate_scope(args.csv, args.target_fleet)
        if args.max_depth < 0 or args.max_depth > 20:
            raise ValueError("--max-depth must be between 0 and 20")
        if args.node_limit < 1 or args.max_cg_iters < 1:
            raise ValueError("node and CG iteration limits must be positive")
        solver = BranchAndPriceSolver(args)
        result = solver.run()
        solver.write(result, args.out)
    except (BranchAndPriceError, FileExistsError, OSError, ValueError) as exc:
        print(f"[B&P] FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        f"[B&P] wrote {args.out}: fleet={result['best_integer_fleet']} "
        f"LB={result['global_lower_bound']:.6f} "
        f"proven={result['proven_optimal']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

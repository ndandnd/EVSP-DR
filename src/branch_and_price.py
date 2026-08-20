"""Exact Ryan--Foster branch-and-price for the expanded-grid E-VSP."""
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
from audit_giro_known_columns import DEPOT, HORIZON_MIN, build_problem
from config import CHARGE_RATE_KW, CHARGING_STATIONS
from durable_io import read_jsonl_records
from exact_pricer_expanded import (
    DATA_DIR,
    ExpandedNetwork,
    _file_sha256,
    _provenance,
    direct_singleton_seed_records,
    load_column_pool,
)
from expanded_path_realization import realize_expanded_path
from lexicographic_fleet_cg import (
    _candidate_record,
    _solve_master as solve_phase_master,
)
from run_exact_pool_mip import validate_final_selected_routes
from target_pool_feasibility import solve_target_feasibility
from utils_v2 import load_station_hourly_prices
from branch_and_price_state import (
    DurableStateMixin,
    baseline_identity as _state_baseline_identity,
    fleet_bound_closes,
)
class BranchAndPriceError(RuntimeError): pass
class ValidationGateError(BranchAndPriceError): pass
def _baseline_identity(args, provenance):
    return _state_baseline_identity(args, provenance, ValidationGateError)
@dataclass(frozen=True, order=True)
class BranchConstraint:
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
    lower_bound: float
    parent_lp_value: float | None
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
    lower_bound: float | None
    phase_1: dict
    phase_2: dict | None
@dataclass
class PhaseResult:
    certified: bool
    lp: object | None
    routes: list[dict]
    min_rc: float | None
    iterations: int
    reason: str
    stats: dict
def route_satisfies(
    route_trips: Iterable[int],
    constraints: Sequence[BranchConstraint],
) -> bool:
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
    allowed = max(tolerance, abs(parent_bound) * 1e-10)
    if child_bound < parent_bound - allowed:
        raise ValidationGateError(
            "G2: child LP bound decreased: "
            f"parent={parent_bound:.12f}, child={child_bound:.12f}, "
            f"tolerance={allowed:.3g}"
        )
def conservative_dual_lower_bound(lp, trip_count, pricing_tolerance): return sum(lp.trip_duals.values()) - trip_count * pricing_tolerance
class ConstrainedDAGPricer:
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
        objective: str,
    ) -> tuple[list[float], dict[int, float]]:
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
                objective_cost = (
                    0.0 if objective == "artificial-elimination"
                    else 1.0 if objective == "fleet-only" and node == 0
                    else 0.0 if objective == "fleet-only"
                    else float(base_cost)
                )
                candidate = (value + objective_cost
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
        objective: str = "combined-cost",
    ) -> list[dict]:
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
                objective=objective,
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
            objective=objective,
        )
        sink_candidates = sorted(
            (values[node] + (
                float(cost) if objective == "combined-cost" else 0.0
            ), node)
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
        objective: str = "combined-cost",
    ) -> tuple[list[dict], int]:
        assignments = expand_constraint_assignments(constraints)
        combined: dict[frozenset[int], dict] = {}
        for required, forbidden in assignments:
            routes = self.shortest_subproblem(
                alpha,
                required=required,
                forbidden=forbidden,
                max_candidates=max_candidates,
                objective=objective,
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
class BranchAndPriceSolver(DurableStateMixin):
    _checkpoint_fields = (
        "pricing_solves", "pricing_calls", "pricing_wall_s", "master_solves",
        "bound_assertions", "integrality_certificates", "integer_audits",
        "infeasible_certificates", "nodes_explored", "nodes_depth_capped",
        "next_node_id", "root_lower_bound", "root_min_rc",
        "root_phase_stats", "root_record", "root_solved", "slow_nodes",
        "ledger_events", "interrupted_reason",
    )
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.elapsed_offset = 0.0
        self.problem = build_problem(
            DATA_DIR,
            args.csv,
            max_station_to_trip_wait_min=HORIZON_MIN,
        )
        self.trips = list(self.problem.trips)
        self.provenance = _provenance(args)
        self.baseline = _baseline_identity(args, self.provenance)
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
        self.root_pool_source = None
        self._seed_singletons()
        if args.root_pool_result is not None:
            self._load_root_pool(args.root_pool_result)
        self.pricing_solves = 0
        self.pricing_calls = 0
        self.pricing_wall_s = 0.0
        self.master_solves = 0
        self.bound_assertions = 0
        self.integrality_certificates = 0
        self.integer_audits = 0
        self.infeasible_certificates = 0
        self.nodes_explored = 0
        self.nodes_depth_capped = 0
        self.next_node_id = 1
        self.root_lp = None
        self.root_lower_bound = None
        self.root_min_rc = None
        self.root_phase_stats = None
        self.root_record = None
        self.incumbent: dict | None = None
        self.stack = [SearchNode(0, tuple(), 0.0, None)]
        self.frontier_bounds: list[float] = []
        self.root_solved = False
        self.interrupted_reason = None
        self.slow_nodes = 0
        self.ledger_events = 0
        self.run_identity = {
            "schema": "evsp-dr-exact-branch-and-price-identity-v2",
            "git_commit": self.provenance["git_commit"],
            "git_dirty": self.provenance["git_dirty"],
            **{key: self.provenance[key] for key in (
                "instance_sha256", "prices_sha256",
                "reference_sha256", "deadhead_sha256",
            )},
            **{key: getattr(args, key) for key in (
                "csv", "prices_csv", "target_fleet", "soc_step", "block_min",
                "g_kwh", "charge_kw", "min_soc_frac", "rc_eps", "max_depth",
                "columns_per_iter", "integrality_tol",
                "phase_1_positive_tol", "bound_tolerance",
                "pricing_slowdown_limit", "pricing_slow_nodes",
            )},
            "baseline": self.baseline,
            "root_pool_source": self.root_pool_source,
        }
        self._setup_io()
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
        self.root_pool_source = {
            "result": str(path), "result_sha256": _file_sha256(path),
            "journal": str(journal), "journal_sha256": _file_sha256(journal),
        }
    @staticmethod
    def _node_payload(node):
        return {
            "node_id": node.node_id,
            "constraints": [
                {"kind": item.kind, "left": item.left, "right": item.right}
                for item in node.constraints
            ],
            "lower_bound": node.lower_bound,
            "parent_lp_value": node.parent_lp_value,
        }
    @staticmethod
    def _node_from_payload(payload):
        return SearchNode(
            int(payload["node_id"]),
            tuple(BranchConstraint(**item) for item in payload["constraints"]),
            float(payload["lower_bound"]),
            payload.get("parent_lp_value"),
        )
    def _remaining_s(self) -> float | None:
        if self.args.wall_limit_s is None:
            return None
        return self.args.wall_limit_s - self._elapsed_s()
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
    def _solve_master(self, routes, phase, method_index):
        last_error = None
        methods = ("highs-ds", "highs-ipm", "highs")
        for candidate_method in methods[method_index:] + methods[:method_index]:
            self.master_solves += 1
            try:
                return solve_phase_master(
                    self.trips, routes, phase, method=candidate_method
                )
            except RuntimeError as exc:
                last_error = exc
        raise BranchAndPriceError(f"all Phase-{phase} masters failed: {last_error}")
    def _price(self, duals, constraints, objective):
        started = time.perf_counter()
        if not constraints:
            routes, solves = self.network.k_best_routes(
                duals, k=self.args.columns_per_iter, objective=objective,
            ), 1
        else:
            routes, solves = self.pricer.price(
                duals, constraints, max_candidates=self.args.columns_per_iter,
                objective=objective,
            )
        duration = time.perf_counter() - started
        self.pricing_solves += solves
        self.pricing_calls += 1
        self.pricing_wall_s += duration
        return routes, solves, duration
    def _solve_phase(self, node, phase):
        preferred_method = 0
        last_lp = None
        last_routes: list[dict] = []
        last_rc = None
        objective = {
            1: "artificial-elimination", 2: "fleet-only",
        }[phase]
        stats = {"pricing_calls": 0, "pricing_solves": 0, "pricing_wall_s": 0.0}
        pricing_eps = min(float(self.args.rc_eps), 1e-9)
        for iteration in range(1, self.args.max_cg_iters + 1):
            if self._expired():
                return PhaseResult(
                    False, last_lp, last_routes, last_rc, iteration - 1,
                    "wall_limit", stats,
                )
            routes = self._allowed_routes(node.constraints)
            lp = self._solve_master(routes, phase, preferred_method)
            last_lp, last_routes = lp, routes
            candidates, solves, duration = self._price(
                lp.trip_duals, node.constraints, objective,
            )
            stats["pricing_calls"] += 1
            stats["pricing_solves"] += solves
            stats["pricing_wall_s"] += duration
            min_rc = candidates[0]["rc"] if candidates else float("inf")
            last_rc = float(min_rc)
            self._event(
                "pricing_iteration", node_id=node.node_id, depth=node.depth,
                phase=phase, iteration=iteration, wall_s=duration,
                subproblems=solves, minimum_reduced_cost=(
                    last_rc if math.isfinite(last_rc) else None
                ),
                master_objective=lp.objective,
                artificial_mass=lp.artificial_total, pool_columns=len(routes),
            )
            if iteration == 1 or iteration % 25 == 0 or (
                min_rc >= -pricing_eps
            ):
                print(
                    f"[B&P] node={node.node_id} depth={node.depth} phase={phase} "
                    f"cg={iteration} obj={lp.objective:.9f} "
                    f"weight={lp.route_weight:.9f} "
                    f"art={lp.artificial_total:.3g} rc={min_rc:.6g} "
                    f"cols={len(routes)}",
                    flush=True,
                )
            if min_rc >= -pricing_eps:
                return PhaseResult(
                    True, lp, routes, float(min_rc), iteration,
                    "certified", stats,
                )
            added = 0
            for candidate in candidates:
                if candidate["rc"] >= -pricing_eps:
                    break
                route = (
                    self.route_adapter.materialize(candidate)
                    if "path_nodes" in candidate
                    else candidate
                )
                record = _candidate_record(
                    route, self.prices, self.provenance["prices_sha256"],
                    phase, iteration,
                )
                record["origin"] = "branch_and_price_phase_pricing"
                record["found_branch_node"] = node.node_id
                key = frozenset(record["trips"])
                current = self.pool.get(key)
                if current is None or (
                    record["cost"] < current["cost"] - 1e-9
                ):
                    self.pool[key] = record
                    self._append_column(record)
                    added += 1
            if not added:
                if preferred_method == 2:
                    return PhaseResult(
                        False, lp, routes, last_rc, iteration,
                        "degenerate_stall", stats,
                    )
                preferred_method += 1
            else:
                preferred_method = 0
        return PhaseResult(
            False, last_lp, last_routes, last_rc,
            self.args.max_cg_iters, "max_cg_iters", stats,
        )
    def _phase_summary(self, result, phase):
        eps = min(float(self.args.rc_eps), 1e-9)
        dual_bound = (
            conservative_dual_lower_bound(result.lp, len(self.trips), eps)
            if result.certified and result.lp is not None else None
        )
        return {
            "phase": phase, "certified": result.certified,
            "reason": result.reason, "iterations": result.iterations,
            "objective": result.lp.objective if result.lp else None,
            "artificial_mass": result.lp.artificial_total if result.lp else None,
            "minimum_reduced_cost": (
                result.min_rc
                if result.min_rc is not None and math.isfinite(result.min_rc)
                else None
            ),
            "conservative_dual_lower_bound": dual_bound,
            **result.stats,
        }
    def _solve_node(self, node):
        phase_1 = self._solve_phase(node, 1)
        p1 = self._phase_summary(phase_1, 1)
        if not phase_1.certified:
            return NodeResult(
                False, False, phase_1.lp, phase_1.routes, phase_1.min_rc,
                phase_1.iterations, phase_1.reason, None, p1, None,
            )
        mass = phase_1.lp.artificial_total
        if mass > self.args.integrality_tol:
            if p1["conservative_dual_lower_bound"] <= self.args.phase_1_positive_tol:
                return NodeResult(
                    False, False, phase_1.lp, phase_1.routes, phase_1.min_rc,
                    phase_1.iterations, "positive_mass_not_strictly_certified",
                    None, p1, None,
                )
            self._event(
                "node_infeasible_certified", node_id=node.node_id,
                artificial_mass=mass,
                dual_lower_bound=p1["conservative_dual_lower_bound"],
            )
            self.infeasible_certificates += 1
            return NodeResult(
                True, True, phase_1.lp, phase_1.routes, phase_1.min_rc,
                phase_1.iterations, "phase_1_positive_mass_certified",
                None, p1, None,
            )
        phase_2 = self._solve_phase(node, 2)
        p2 = self._phase_summary(phase_2, 2)
        if not phase_2.certified:
            return NodeResult(
                False, False, phase_2.lp, phase_2.routes, phase_2.min_rc,
                phase_1.iterations + phase_2.iterations, phase_2.reason,
                None, p1, p2,
            )
        bound = p2["conservative_dual_lower_bound"]
        if bound > phase_2.lp.objective + self.args.bound_tolerance:
            raise ValidationGateError("Phase-II dual bound exceeds primal")
        return NodeResult(
            True, False, phase_2.lp, phase_2.routes, phase_2.min_rc,
            phase_1.iterations + phase_2.iterations, "certified",
            bound, p1, p2,
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
        fleet = len(selected)
        if (self.root_lower_bound is not None
                and fleet < self.root_lower_bound - self.args.bound_tolerance):
            raise ValidationGateError(
                "G5: integer fleet is below the certified root fleet bound"
            )
        validate_final_selected_routes(
            self._replay_status(),
            self.trips,
            list(selected),
        )
        self.integer_audits += 1
        if (self.incumbent is None
                or fleet < self.incumbent["fleet"]
                or (fleet == self.incumbent["fleet"]
                    and cost < self.incumbent["cost"] - 1e-7)):
            self.incumbent = {
                "cost": cost, "fleet": fleet,
                "routes": list(selected),
                "source": source,
            }
            self._event(
                "incumbent", fleet=fleet, cost=cost, source=source,
                routes=list(selected),
            )
            self._prune_open_by_incumbent()
            print(
                f"[B&P] incumbent={fleet} buses "
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
        attempts = min(6, len(self.trips) - self.args.target_fleet + 1)
        for fleet in range(
            self.args.target_fleet, self.args.target_fleet + attempts
        ):
            solved = solve_target_feasibility(
                routes, self.trips, fleet,
                timelimit=self.args.root_mip_s / attempts,
                threads=1, solver="highs",
            )
            if solved["outcome"] == "FEASIBLE":
                selected = [routes[index] for index in solved["selected_indices"]]
                self._update_incumbent(
                    selected, source=f"root_pool_highs_feasibility_{fleet}"
                )
                return
    def _new_child(self, parent, result, constraint):
        child = SearchNode(
            self.next_node_id,
            parent.constraints + (constraint,),
            result.lower_bound,
            result.lp.objective,
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
        if node.parent_lp_value is not None:
            assert_child_bound(
                node.parent_lp_value,
                lp.objective,
                tolerance=self.args.bound_tolerance,
            )
            self.bound_assertions += 1
        if node.node_id and result.phase_2 and self.root_phase_stats:
            root = self.root_phase_stats["phase_2"]
            root_average = root["pricing_wall_s"] / root["pricing_calls"]
            node_average = (
                result.phase_2["pricing_wall_s"]
                / result.phase_2["pricing_calls"]
            )
            ratio = node_average / max(root_average, 1e-12)
            self._event(
                "node_pricing_cost", node_id=node.node_id,
                pricing_wall_s=result.phase_2["pricing_wall_s"],
                pricing_calls=result.phase_2["pricing_calls"],
                subproblems=result.phase_2["pricing_solves"],
                root_average_s=root_average, slowdown_ratio=ratio,
            )
            if ratio > self.args.pricing_slowdown_limit:
                self.slow_nodes += 1
                if self.slow_nodes >= self.args.pricing_slow_nodes:
                    self.interrupted_reason = "pricing_slowdown_kill_criterion"
        if self.incumbent is not None and (
            fleet_bound_closes(
                result.lower_bound, self.incumbent["fleet"],
                self.args.bound_tolerance,
            )
        ):
            self._event(
                "node_pruned_by_bound", node_id=node.node_id,
                lower_bound=result.lower_bound,
                incumbent_fleet=self.incumbent["fleet"],
            )
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
            frontier_bounds.append(result.lower_bound)
            return
        (left, right), alpha = branch
        together = self._new_child(
            node, result,
            BranchConstraint("together", left, right),
        )
        apart = self._new_child(
            node, result,
            BranchConstraint("apart", left, right),
        )
        preferred, other = (
            (together, apart) if alpha >= 0.5 else (apart, together)
        )
        stack.append(other)
        stack.append(preferred)
    def run(self) -> dict:
        if not self.root_solved:
            root = self.stack[-1]
            self._event("node_started", **self._node_payload(root))
            self._checkpoint()
            root_result = self._solve_node(root)
            if not root_result.certified or root_result.infeasible:
                if root_result.infeasible:
                    raise ValidationGateError("G1 root is Phase-I infeasible")
                self.interrupted_reason = root_result.reason
                return self._checkpoint()
            self.stack.pop()
            self.nodes_explored += 1
            self.root_lp = root_result.lp
            self.root_lower_bound = root_result.lower_bound
            self.root_min_rc = root_result.min_rc
            self.root_phase_stats = {
                "phase_1": root_result.phase_1,
                "phase_2": root_result.phase_2,
            }
            self.root_record = {
                "objective": root_result.lp.objective,
                "route_weight": root_result.lp.route_weight,
                "conservative_dual_lower_bound": root_result.lower_bound,
                "min_rc": root_result.min_rc,
                "columns": len(root_result.routes),
                "cg_iterations": root_result.cg_iterations,
                "phases": self.root_phase_stats,
            }
            expected = (
                self.baseline["route_weight"] if self.baseline else None
            )
            if expected is not None and not math.isclose(
                root_result.lp.route_weight, expected, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValidationGateError(
                    f"G1 expected {expected:.9f}, "
                    f"observed {root_result.lp.route_weight:.9f}"
                )
            self.root_solved = True
            self._event("root_certified", root=self.root_record)
            print(
                f"[B&P] G1 PASS root fleet={root_result.lp.route_weight:.9f} "
                f"LB={root_result.lower_bound:.9f}",
                flush=True,
            )
            if self.incumbent is None:
                self._singleton_incumbent()
                self._root_pool_mip_incumbent(root_result.routes)
            if self.args.root_only:
                self.interrupted_reason = "root_only"
                return self._checkpoint()
            self._process_certified_node(
                root, root_result, self.stack, self.frontier_bounds,
            )
            self._checkpoint()
        while self.stack:
            if self.nodes_explored >= self.args.node_limit:
                self.interrupted_reason = "node_limit"
                break
            if self._expired():
                self.interrupted_reason = "wall_limit"
                break
            if self.interrupted_reason == "pricing_slowdown_kill_criterion":
                break
            node = self.stack[-1]
            self._event("node_started", **self._node_payload(node))
            self._checkpoint()
            result = self._solve_node(node)
            if not result.certified:
                self.interrupted_reason = result.reason
                self._event(
                    "node_left_open", node_id=node.node_id, reason=result.reason,
                    inherited_lower_bound=node.lower_bound,
                )
                break
            self.stack.pop()
            self.nodes_explored += 1
            self._event(
                "node_certified", node_id=node.node_id,
                infeasible=result.infeasible, lower_bound=result.lower_bound,
                phase_1=result.phase_1, phase_2=result.phase_2,
            )
            self._process_certified_node(
                node, result, self.stack, self.frontier_bounds,
            )
            self._checkpoint()
        complete = not self.stack and not self.frontier_bounds
        if complete:
            self.interrupted_reason = None
            self._event("search_complete")
        return self._checkpoint(search_complete=complete)
    def interrupt(self, reason="external_interrupt"):
        self.interrupted_reason = reason
        self._event("interrupted", reason=reason)
        return self._checkpoint()
def _validate_scope(csv_path: str, target_fleet: int) -> None:
    match = re.search(r"_k(\d{2})_", Path(csv_path).name)
    if match is None:
        raise ValueError("instance filename must contain _kNN_")
    scale = int(match.group(1))
    if scale not in {2, 3, 5}:
        raise ValueError("branch-and-price scope is restricted to k2/k3/k5")
    if scale != target_fleet:
        raise ValueError(f"--target-fleet {target_fleet} disagrees with k{scale:02d}")
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
    parser.add_argument("--rc-eps", type=float, default=1e-9)
    parser.add_argument("--integrality-tol", type=float, default=1e-7)
    parser.add_argument("--phase-1-positive-tol", type=float, default=1e-8)
    parser.add_argument("--bound-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-cg-iters", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--node-limit", type=int, default=1000)
    parser.add_argument("--wall-limit-s", type=float, default=21600.0)
    parser.add_argument("--root-mip-s", type=float, default=60.0)
    parser.add_argument("--pricing-slowdown-limit", type=float, default=10.0)
    parser.add_argument("--pricing-slow-nodes", type=int, default=3)
    parser.add_argument("--expected-root-weight", type=float)
    parser.add_argument("--root-only", action="store_true")
    parser.add_argument("--root-pool-result", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser
def main(argv=None) -> int:
    if os.environ.get("SLURM_JOB_ID"):
        print("[B&P] refusing cluster execution; local runs only", file=sys.stderr)
        return 2
    args = build_parser().parse_args(argv)
    solver = None
    try:
        _validate_scope(args.csv, args.target_fleet)
        if args.max_depth < 0 or args.max_depth > 20:
            raise ValueError("--max-depth must be between 0 and 20")
        if args.node_limit < 1 or args.max_cg_iters < 1:
            raise ValueError("node and CG iteration limits must be positive")
        solver = BranchAndPriceSolver(args)
        result = solver.run()
    except KeyboardInterrupt:
        if solver is not None:
            solver.interrupt()
        print("[B&P] interrupted after durable checkpoint", file=sys.stderr)
        return 130
    except (
        BranchAndPriceError, FileExistsError, OSError, RuntimeError, ValueError
    ) as exc:
        if solver is not None:
            solver.interrupt(f"error:{type(exc).__name__}")
        print(f"[B&P] FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if solver is not None:
            solver.close()
    print(f"[B&P] wrote {args.out}: fleet={result['best_integer_fleet']} "
          f"LB={result['global_lower_bound']} proven={result['proven_optimal']}")
    return 0
if __name__ == "__main__": raise SystemExit(main())

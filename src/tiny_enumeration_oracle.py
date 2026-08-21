#!/usr/bin/env python3
"""Exhaustive tiny SOC-time oracle and four-way differential campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from arcflow_oracle import NetworkData, build_model, index_active_arcs, solve
from branch_and_price import (
    BranchConstraint,
    ConstrainedDAGPricer,
    choose_ryan_foster_pair,
    fleet_bound_closes,
    route_satisfies,
)
from config import BUS_COST_KX
from lexicographic_fleet_cg import _solve_master as solve_phase_master


SCHEMA = "evsp-dr-tiny-differential-oracle-v2"
DEFAULT_SEED = 20260821


@dataclass(frozen=True)
class TinySpec:
    case_id: str
    seed: int
    trip_count: int
    station_count: int
    capacity: int
    reserve: int
    start: tuple[int, ...]
    duration: tuple[int, ...]
    energy: tuple[int, ...]
    deadhead_energy: tuple[tuple[int, ...], ...]
    trip_to_station: tuple[tuple[bool, ...], ...]
    station_to_trip: tuple[tuple[bool, ...], ...]
    block_min: int = 5
    charge_step: int = 2
    max_trip_gap: int = 57
    reserve_relax_steps: int = 0
    allow_station_transfer: bool = False
    soc_step: float = 1.0
    delayed_charging: bool = True
    tariff_shape: str = "flat"
    tariff_phase: int = 0


class TinyNetwork:
    SINK = 1

    def __init__(self, spec: TinySpec):
        self.spec = spec
        self.problem = SimpleNamespace(
            trips=tuple(range(spec.trip_count)),
            start_min={i: spec.start[i] for i in range(spec.trip_count)},
            end_min={
                i: spec.start[i] + spec.duration[i]
                for i in range(spec.trip_count)
            },
        )
        self.node_meta = [("source", None, None), ("sink", None, None)]
        self.trip_node = {}
        self.charge_node = {}
        self.out = []
        effective_reserve = max(
            0, spec.reserve - spec.reserve_relax_steps,
        )
        order = []
        for trip in self.problem.trips:
            minimum = spec.energy[trip] + effective_reserve
            for level in range(minimum, spec.capacity + 1):
                node = len(self.node_meta)
                self.node_meta.append(("trip", trip, level))
                self.trip_node[(trip, level)] = node
                order.append((spec.start[trip], 0, node))
        horizon = max(
            spec.start[i] + spec.duration[i]
            for i in self.problem.trips
        ) + 3 * spec.block_min
        self.n_blocks = int(math.ceil(horizon / spec.block_min))
        if spec.tariff_shape not in {"flat", "time_varying"}:
            raise ValueError(f"unknown tariff shape {spec.tariff_shape!r}")
        self.tariff_by_block = tuple(
            0.2 if spec.tariff_shape == "flat"
            else (0.05, 0.4, 0.15, 0.6)[
                (block + spec.tariff_phase) % 4
            ]
            for block in range(self.n_blocks)
        )
        for station in range(spec.station_count):
            for block in range(self.n_blocks):
                for level in range(effective_reserve, spec.capacity + 1):
                    node = len(self.node_meta)
                    self.node_meta.append(
                        ("charge", (station, block), level)
                    )
                    self.charge_node[(station, block, level)] = node
                    order.append((block * spec.block_min, 1, node))
        self.topo = [0] + [
            node for _time, _kind, node in sorted(order)
        ] + [self.SINK]
        self.out = [[] for _ in self.node_meta]

        def add(left, right, cost=0.0, trip=-1):
            self.out[left].append((right, float(cost), int(trip)))

        source_level = spec.capacity - 1
        for trip in self.problem.trips:
            node = self.trip_node.get((trip, source_level))
            if node is not None:
                add(0, node, BUS_COST_KX, trip)
        for (trip, level), left in self.trip_node.items():
            exit_level = level - spec.energy[trip]
            end = self.problem.end_min[trip]
            if exit_level - 1 >= effective_reserve:
                add(left, self.SINK)
            for successor in range(trip + 1, spec.trip_count):
                gap = spec.start[successor] - end
                deadhead = spec.deadhead_energy[trip][successor]
                next_level = exit_level - deadhead
                if (
                    gap >= 1
                    and gap <= spec.max_trip_gap
                    and (successor, next_level) in self.trip_node
                ):
                    add(
                        left, self.trip_node[(successor, next_level)],
                        trip=successor,
                    )
            for station in range(spec.station_count):
                if not spec.trip_to_station[trip][station]:
                    continue
                first = int(math.ceil((end + 1) / spec.block_min))
                blocks = (
                    range(first, self.n_blocks)
                    if spec.delayed_charging else (first,)
                )
                for block in blocks:
                    node = self.charge_node.get(
                        (station, block, exit_level)
                    )
                    if node is not None:
                        add(left, node)
        for (station, block, level), left in self.charge_node.items():
            after = min(spec.capacity, level + spec.charge_step)
            block_end = (block + 1) * spec.block_min
            charge_cost = (
                (after - level) * spec.soc_step
                * self.tariff_by_block[block]
            )
            if block + 1 < self.n_blocks and after > level:
                add(
                    left, self.charge_node[(station, block + 1, after)],
                    charge_cost,
                )
            for trip in self.problem.trips:
                if (
                    spec.station_to_trip[station][trip]
                    and block_end + 1 <= spec.start[trip]
                    and (trip, after) in self.trip_node
                ):
                    add(
                        left, self.trip_node[(trip, after)],
                        charge_cost, trip=trip,
                    )
            if after >= effective_reserve:
                add(left, self.SINK, charge_cost)
            if spec.allow_station_transfer and block + 1 < self.n_blocks:
                for other in range(spec.station_count):
                    if other != station:
                        add(
                            left,
                            self.charge_node[(other, block + 1, after)],
                            charge_cost,
                        )
        self.n_arcs = sum(len(arcs) for arcs in self.out)
        self.sink_arcs = tuple(
            (left, cost) for left, arcs in enumerate(self.out)
            for right, cost, _trip in arcs if right == self.SINK
        )


class TinyExactCGPricer:
    """Unconstrained DAG pricing mirroring the production exact-CG pass."""

    def __init__(self, network: TinyNetwork):
        self.network = network

    def _solve(self, duals, objective):
        infinity = float("inf")
        value = [infinity] * len(self.network.node_meta)
        parent = [None] * len(self.network.node_meta)
        value[0] = 0.0
        for left in self.network.topo:
            if value[left] == infinity:
                continue
            for right, cost, trip in self.network.out[left]:
                if objective == "combined-cost":
                    objective_cost = cost
                elif objective == "fleet-only":
                    objective_cost = 1.0 if left == 0 else 0.0
                elif objective == "artificial-elimination":
                    objective_cost = 0.0
                else:
                    raise ValueError(f"unknown pricing objective {objective!r}")
                candidate = value[left] + objective_cost - (
                    float(duals.get(trip, 0.0)) if trip >= 0 else 0.0
                )
                if candidate < value[right] - 1e-12:
                    value[right] = candidate
                    parent[right] = left
        return value, parent

    def minimum_reduced_cost(self, duals, *, objective="combined-cost"):
        value, _parent = self._solve(duals, objective)
        return value[self.network.SINK]

    def price(self, duals, _constraints, *, max_candidates, objective):
        value, parent = self._solve(duals, objective)

        def record(node):
            trips = []
            while node != 0:
                kind, key, _level = self.network.node_meta[node]
                if kind == "trip":
                    trips.append(key)
                node = parent[node]
            trips.reverse()
            return {"trips": trips, "cost": 1.0}

        candidates = sorted(
            (
                value[left] + (
                    cost if objective == "combined-cost" else 0.0
                ),
                left,
            )
            for left, cost in self.network.sink_arcs
            if value[left] != math.inf
        )
        records, seen = [], set()
        limit = min(max_candidates, 30)
        for _reduced_cost, left in candidates:
            candidate = record(left)
            incidence = frozenset(candidate["trips"])
            if incidence in seen:
                continue
            seen.add(incidence)
            records.append(candidate)
            if len(records) == limit:
                break
        return records, 1


def generate_spec(seed: int, ordinal: int) -> TinySpec:
    rng = random.Random((seed << 20) + ordinal)
    trip_count = rng.randint(8, 14)
    station_count = rng.randint(1, 2)
    soc_step = rng.choice((0.5, 1.0, 2.0))
    reserve = rng.randint(0, 2)
    capacity = rng.randint(max(6, reserve + 4), 9)
    maximum_energy = max(1, capacity - reserve - 2)
    energy = tuple(
        rng.randint(1, min(3, maximum_energy))
        for _ in range(trip_count)
    )
    start, duration, current = [], [], rng.randint(5, 15)
    for _trip in range(trip_count):
        length = rng.randint(2, 5)
        start.append(current)
        duration.append(length)
        current += length + rng.randint(2, 20)
    deadhead = tuple(tuple(
        0 if right <= left else rng.randint(0, 1)
        for right in range(trip_count)
    ) for left in range(trip_count))
    reachability = rng.uniform(0.35, 0.85)
    trip_to_station = tuple(tuple(
        rng.random() < reachability for _ in range(station_count)
    ) for _ in range(trip_count))
    station_to_trip = tuple(tuple(
        rng.random() < reachability for _ in range(trip_count)
    ) for _ in range(station_count))
    return TinySpec(
        case_id=f"random_{ordinal:04d}", seed=seed,
        trip_count=trip_count, station_count=station_count,
        capacity=capacity, reserve=reserve,
        start=tuple(start), duration=tuple(duration), energy=energy,
        deadhead_energy=deadhead,
        trip_to_station=trip_to_station,
        station_to_trip=station_to_trip,
        charge_step=rng.randint(1, 2),
        soc_step=soc_step,
        delayed_charging=bool(rng.getrandbits(1)),
        tariff_shape=rng.choice(("flat", "time_varying")),
        tariff_phase=rng.randrange(4),
    )


def enumerate_route_masks(network: TinyNetwork) -> list[int]:
    trip_position = {
        trip: position for position, trip in enumerate(network.problem.trips)
    }
    stack = [(0, 0)]
    visited = {(0, 0)}
    routes = set()
    while stack:
        node, mask = stack.pop()
        for successor, _cost, trip_index in network.out[node]:
            next_mask = mask
            if trip_index >= 0:
                next_mask |= 1 << trip_index
            if successor == network.SINK:
                if next_mask:
                    routes.add(next_mask)
                continue
            state = (successor, next_mask)
            if state not in visited:
                visited.add(state)
                stack.append(state)
    singleton_masks = {1 << trip_position[t] for t in network.problem.trips}
    if not singleton_masks <= routes:
        raise AssertionError("generator failed to provide every singleton")
    return sorted(routes)


def enumerate_route_costs(network: TinyNetwork) -> dict[int, float]:
    """Enumerate every reachable incidence, retaining its cheapest path."""
    labels = [dict() for _ in network.node_meta]
    labels[0][0] = 0.0
    routes = {}
    for node in network.topo:
        for mask, prefix_cost in tuple(labels[node].items()):
            for successor, arc_cost, trip_index in network.out[node]:
                next_mask = (
                    mask | (1 << trip_index)
                    if trip_index >= 0 else mask
                )
                next_cost = prefix_cost + arc_cost
                if successor == network.SINK:
                    if next_mask and next_cost < routes.get(
                        next_mask, math.inf,
                    ):
                        routes[next_mask] = next_cost
                    continue
                if next_cost < labels[successor].get(next_mask, math.inf):
                    labels[successor][next_mask] = next_cost
    if set(enumerate_route_masks(network)) != set(routes):
        raise AssertionError("cost enumeration changed route incidence set")
    return routes


def sampled_dual_vectors(spec: TinySpec, count: int):
    if count < 1:
        return []
    material = json.dumps(asdict(spec), sort_keys=True).encode()
    derived = int.from_bytes(
        hashlib.sha256(material + b"|arbitrary-duals-v1").digest(),
        "big",
    )
    rng = random.Random(derived)
    vectors = [
        ("exact_tie_zero", tuple(0.0 for _ in range(spec.trip_count))),
        (
            "near_tie_signed",
            tuple(
                (-1.0 if trip % 2 else 1.0) * 1e-8
                for trip in range(spec.trip_count)
            ),
        ),
        (
            "all_negative",
            tuple(
                -rng.uniform(1.0, BUS_COST_KX)
                for _ in range(spec.trip_count)
            ),
        ),
    ][:count]
    while len(vectors) < count:
        sample = len(vectors)
        mode = sample % 3
        if mode == 0:
            values = tuple(
                rng.uniform(-BUS_COST_KX, 1.25 * BUS_COST_KX)
                for _ in range(spec.trip_count)
            )
            label = "wide_signed"
        elif mode == 1:
            center = BUS_COST_KX / max(1, spec.trip_count // 2)
            values = tuple(
                center + rng.uniform(-500.0, 500.0)
                for _ in range(spec.trip_count)
            )
            values = tuple(
                -abs(value) if trip % 5 == 0 else value
                for trip, value in enumerate(values)
            )
            label = "cg_scale_signed"
        else:
            choices = (
                -0.5 * BUS_COST_KX, -1e-8, 0.0, 1e-8,
                0.25 * BUS_COST_KX, 0.5 * BUS_COST_KX,
            )
            values = tuple(
                choices[rng.randrange(len(choices))]
                for _ in range(spec.trip_count)
            )
            label = "quantized_ties"
        vectors.append((f"{label}_{sample:02d}", values))
    return vectors


def evaluate_dual_vector(
    network, route_costs, vector, *, kind, sample_index,
    exact_pricer=None, constrained_pricer=None,
):
    reduced = {
        mask: cost - sum(
            vector[trip] for trip in range(network.spec.trip_count)
            if mask & (1 << trip)
        )
        for mask, cost in route_costs.items()
    }
    expected = min(reduced.values())
    ties = [
        mask for mask, value in reduced.items()
        if abs(value - expected) <= 1e-7
    ]
    duals = {trip: vector[trip] for trip in range(len(vector))}
    exact_pricer = exact_pricer or TinyExactCGPricer(network)
    constrained_pricer = constrained_pricer or ConstrainedDAGPricer(network)
    actual = exact_pricer.minimum_reduced_cost(
        duals, objective="combined-cost",
    )
    constrained_candidates, _solves = constrained_pricer.price(
        duals, (), max_candidates=1, objective="combined-cost",
    )
    constrained_actual = (
        constrained_candidates[0]["rc"]
        if constrained_candidates else math.inf
    )
    error = max(
        abs(actual - expected),
        abs(constrained_actual - expected),
    )
    return {
        "sample_index": sample_index,
        "kind": kind,
        "dual_vector": list(vector),
        "expected_min_reduced_cost": expected,
        "exact_cg_dp_min_reduced_cost": actual,
        "constrained_dp_min_reduced_cost": constrained_actual,
        "exact_cg_dp_absolute_error": abs(actual - expected),
        "constrained_dp_absolute_error": abs(
            constrained_actual - expected
        ),
        "absolute_error": error,
        "exhaustive_minimizer_masks": ties,
    }


def check_arbitrary_duals(
    network: TinyNetwork, route_costs: dict[int, float], count: int,
):
    mismatches = []
    maximum_error = 0.0
    exact_maximum_error = 0.0
    constrained_maximum_error = 0.0
    negative_vectors = 0
    tied_vectors = 0
    exact_pricer = TinyExactCGPricer(network)
    constrained_pricer = ConstrainedDAGPricer(network)
    for sample_index, (kind, vector) in enumerate(
        sampled_dual_vectors(network.spec, count)
    ):
        if any(value < 0.0 for value in vector):
            negative_vectors += 1
        detail = evaluate_dual_vector(
            network, route_costs, vector,
            kind=kind, sample_index=sample_index,
            exact_pricer=exact_pricer,
            constrained_pricer=constrained_pricer,
        )
        tied_vectors += len(detail["exhaustive_minimizer_masks"]) > 1
        maximum_error = max(maximum_error, detail["absolute_error"])
        exact_maximum_error = max(
            exact_maximum_error, detail["exact_cg_dp_absolute_error"],
        )
        constrained_maximum_error = max(
            constrained_maximum_error,
            detail["constrained_dp_absolute_error"],
        )
        if detail["absolute_error"] > 1e-7:
            mismatches.append(detail)
    return {
        "samples": count,
        "agreements": count - len(mismatches),
        "agreement_rate": (
            (count - len(mismatches)) / count if count else None
        ),
        "negative_component_vectors": negative_vectors,
        "near_or_exact_tie_vectors": tied_vectors,
        "max_absolute_error": maximum_error,
        "exact_cg_dp_max_absolute_error": exact_maximum_error,
        "constrained_dp_max_absolute_error": constrained_maximum_error,
        "mismatches": mismatches,
    }


def route_records(masks: list[int], trip_count: int) -> list[dict]:
    return [
        {
            "mask": mask,
            "trips": [
                trip for trip in range(trip_count)
                if mask & (1 << trip)
            ],
            "cost": 1.0,
        }
        for mask in masks
    ]


def exact_cover_fleet(masks: list[int], trip_count: int) -> int:
    full = (1 << trip_count) - 1
    by_trip = [
        [mask for mask in masks if mask & (1 << trip)]
        for trip in range(trip_count)
    ]

    @lru_cache(maxsize=None)
    def search(covered):
        if covered == full:
            return 0
        missing = (~covered) & full
        bit = missing & -missing
        trip = bit.bit_length() - 1
        return min(
            1 + search(covered | route)
            for route in by_trip[trip]
            if not route & covered
        )

    return search(0)


def fleet_lp(records: list[dict], trips: list[int]):
    result = solve_phase_master(trips, records, 2)
    return result.objective, result


def _priced_records(pricer, duals, constraints, objective):
    candidates, solves = pricer.price(
        duals, constraints, max_candidates=64, objective=objective,
    )
    records = [
        {
            "mask": sum(1 << trip for trip in candidate["trips"]),
            "trips": list(candidate["trips"]),
            "cost": 1.0,
        }
        for candidate in candidates
    ]
    return records, solves


def column_generation(
    network, initial_pool=None, constraints=(), *, pricing="exact_cg",
):
    trips = list(network.problem.trips)
    pool = {
        record["mask"]: record for record in (
            initial_pool or route_records(
                [1 << trip for trip in trips], len(trips),
            )
        )
    }
    pricer = (
        TinyExactCGPricer(network)
        if pricing == "exact_cg"
        else ConstrainedDAGPricer(network)
    )

    def phase(number):
        methods = ("highs-ds", "highs-ipm", "highs")
        method = 0
        for iteration in range(1, 1001):
            routes = [
                record for record in pool.values()
                if route_satisfies(record["trips"], constraints)
            ]
            lp = solve_phase_master(
                trips, routes, number, method=methods[method],
            )
            objective = (
                "artificial-elimination" if number == 1 else "fleet-only"
            )
            candidates, _solves = _priced_records(
                pricer, lp.trip_duals, constraints, objective,
            )
            minimum = min(
                (candidate["cost"] - sum(
                    lp.trip_duals[t] for t in candidate["trips"]
                ) if number == 2 else -sum(
                    lp.trip_duals[t] for t in candidate["trips"]
                ))
                for candidate in candidates
            ) if candidates else math.inf
            if minimum >= -1e-9:
                return lp, routes, iteration
            added = 0
            for candidate in candidates:
                reduced = (
                    1.0 if number == 2 else 0.0
                ) - sum(lp.trip_duals[t] for t in candidate["trips"])
                if reduced < -1e-9 and candidate["mask"] not in pool:
                    pool[candidate["mask"]] = candidate
                    added += 1
            if not added:
                method += 1
                if method == len(methods):
                    raise RuntimeError("tiny CG degenerate stall")
            else:
                method = 0
        raise RuntimeError("tiny CG iteration limit")

    phase1, _routes, phase1_iterations = phase(1)
    if phase1.artificial_total > 1e-7:
        return {
            "infeasible": True, "pool": pool,
            "phase1_iterations": phase1_iterations,
        }
    phase2, routes, phase2_iterations = phase(2)
    return {
        "infeasible": False, "lp": phase2, "routes": routes,
        "pool": pool, "lp_bound": phase2.objective,
        "phase1_iterations": phase1_iterations,
        "phase2_iterations": phase2_iterations,
    }


def solve_branch_and_price(network):
    trips = list(network.problem.trips)
    singleton = route_records([1 << trip for trip in trips], len(trips))
    pool = {record["mask"]: record for record in singleton}
    incumbent = len(trips)
    root_bound = None
    nodes = 0
    stack = [tuple()]
    while stack:
        constraints = stack.pop()
        solved = column_generation(
            network, list(pool.values()), constraints, pricing="branch",
        )
        pool.update(solved["pool"])
        nodes += 1
        if solved["infeasible"]:
            continue
        lp, routes = solved["lp"], solved["routes"]
        if root_bound is None:
            root_bound = lp.objective
        if fleet_bound_closes(lp.objective, incumbent, 1e-7):
            continue
        branch = choose_ryan_foster_pair(
            routes, lp.route_values, tolerance=1e-7,
        )
        if branch is None:
            chosen = [
                route for route, value in zip(routes, lp.route_values)
                if round(value) == 1
            ]
            covered = 0
            for route in chosen:
                if covered & route["mask"]:
                    raise AssertionError("B&P integral routes overlap")
                covered |= route["mask"]
            if covered != (1 << len(trips)) - 1:
                raise AssertionError("B&P integral routes omit trips")
            incumbent = min(incumbent, len(chosen))
            stack = [
                active for active in stack
                if not fleet_bound_closes(
                    0.0 if root_bound is None else root_bound,
                    incumbent, 1e-7,
                )
            ]
            continue
        (left, right), alpha = branch
        together = constraints + (BranchConstraint("together", left, right),)
        apart = constraints + (BranchConstraint("apart", left, right),)
        stack.extend(
            (apart, together) if alpha >= 0.5 else (together, apart)
        )
        if nodes > 5000:
            raise RuntimeError("tiny B&P node limit")
    return {"lp_bound": root_bound, "integer": incumbent, "nodes": nodes}


def solve_arcflow(network):
    data = NetworkData(
        "tiny.json", "flat", network.problem, network,
        1.0, network.spec.block_min,
        float(network.spec.capacity), float(network.spec.charge_step * 12),
        float(max(0, network.spec.reserve - network.spec.reserve_relax_steps)),
    )
    arcs = index_active_arcs(network)
    model = build_model(data, arcs)
    lp, _ = solve(model, objective_kind="fleet", integrality="none")
    mip, primal = solve(model, objective_kind="fleet", integrality="all")
    if lp.status != "optimal" or mip.status != "optimal" or primal is None:
        raise RuntimeError("tiny arc-flow failed")
    paths = []
    remaining = np.rint(primal).astype(int)
    while remaining[arcs.tail == 0].sum():
        node, mask = 0, 0
        while node != network.SINK:
            arc = next(
                index for index in range(
                    int(arcs.out_start[node]),
                    int(arcs.out_start[node + 1]),
                ) if remaining[index] > 0
            )
            remaining[arc] -= 1
            if arcs.trip[arc] >= 0:
                mask |= 1 << int(arcs.trip[arc])
            node = int(arcs.head[arc])
        paths.append(mask)
    if remaining.any():
        raise AssertionError("arc-flow decomposition left residual flow")
    covered = 0
    for mask in paths:
        if covered & mask:
            raise AssertionError("arc-flow paths overcover trips")
        covered |= mask
    if covered != (1 << network.spec.trip_count) - 1:
        raise AssertionError("arc-flow paths omit trips")
    return {
        "lp_bound": lp.vehicles,
        "integer": int(round(mip.vehicles)),
        "variables": model.matrix.shape[1],
        "constraints": model.matrix.shape[0],
    }


def compare_spec(spec: TinySpec, *, dual_samples=0) -> dict:
    network = TinyNetwork(spec)
    started = time.perf_counter()
    route_costs = (
        enumerate_route_costs(network) if dual_samples else None
    )
    masks = (
        sorted(route_costs) if route_costs is not None
        else enumerate_route_masks(network)
    )
    full_records = route_records(masks, spec.trip_count)
    brute_integer = exact_cover_fleet(masks, spec.trip_count)
    brute_lp, _lp = fleet_lp(full_records, list(range(spec.trip_count)))
    cg = column_generation(network)
    cg_integer = exact_cover_fleet(
        list(cg["pool"]), spec.trip_count,
    )
    bnp = solve_branch_and_price(network)
    arc = solve_arcflow(network)
    lp_values = {
        "brute_force": brute_lp,
        "exact_cg": cg["lp_bound"],
        "branch_and_price": bnp["lp_bound"],
        "arc_flow": arc["lp_bound"],
    }
    integer_values = {
        "brute_force": brute_integer,
        "exact_cg_pool_mip": cg_integer,
        "branch_and_price": bnp["integer"],
        "arc_flow": arc["integer"],
    }
    pricing = (
        check_arbitrary_duals(network, route_costs, dual_samples)
        if route_costs is not None else {
            "samples": 0, "agreements": 0, "agreement_rate": None,
            "negative_component_vectors": 0,
            "near_or_exact_tie_vectors": 0,
            "max_absolute_error": 0.0,
            "exact_cg_dp_max_absolute_error": 0.0,
            "constrained_dp_max_absolute_error": 0.0,
            "mismatches": [],
        }
    )
    lp_agrees = max(lp_values.values()) - min(lp_values.values()) <= 1e-7
    integer_agrees = len(set(integer_values.values())) == 1
    pricing_agrees = not pricing["mismatches"]
    return {
        "case_id": spec.case_id,
        "trip_count": spec.trip_count,
        "station_count": spec.station_count,
        "soc_step": spec.soc_step,
        "delayed_charging": spec.delayed_charging,
        "tariff_shape": spec.tariff_shape,
        "station_reachability_density": (
            (
                sum(sum(row) for row in spec.trip_to_station)
                + sum(sum(row) for row in spec.station_to_trip)
            )
            / (2 * spec.trip_count * spec.station_count)
        ),
        "route_count": len(masks),
        "lp": lp_values,
        "integer": integer_values,
        "lp_agrees": lp_agrees,
        "integer_agrees": integer_agrees,
        "pricing": pricing,
        "pricing_agrees": pricing_agrees,
        "agreement": lp_agrees and integer_agrees and pricing_agrees,
        "cg_pool_columns": len(cg["pool"]),
        "bnp_nodes": bnp["nodes"],
        "arc_variables": arc["variables"],
        "wall_s": time.perf_counter() - started,
    }


def _disagreement_signature(result):
    brute = result["integer"]["brute_force"]
    return (
        not result["lp_agrees"],
        not result.get("pricing_agrees", True),
        tuple(sorted(
            method for method, value in result["integer"].items()
            if method != "brute_force" and value != brute
        )),
    )


def induced_spec(spec: TinySpec, keep: tuple[int, ...]) -> TinySpec:
    keep = tuple(sorted(keep))
    return replace(
        spec,
        case_id=f"{spec.case_id}_min{len(keep)}",
        trip_count=len(keep),
        start=tuple(spec.start[index] for index in keep),
        duration=tuple(spec.duration[index] for index in keep),
        energy=tuple(spec.energy[index] for index in keep),
        deadhead_energy=tuple(tuple(
            spec.deadhead_energy[left][right] for right in keep
        ) for left in keep),
        trip_to_station=tuple(
            spec.trip_to_station[index] for index in keep
        ),
        station_to_trip=tuple(tuple(
            station[index] for index in keep
        ) for station in spec.station_to_trip),
    )


def induced_stations(spec: TinySpec, keep: tuple[int, ...]) -> TinySpec:
    keep = tuple(sorted(keep))
    return replace(
        spec,
        station_count=len(keep),
        trip_to_station=tuple(tuple(
            row[index] for index in keep
        ) for row in spec.trip_to_station),
        station_to_trip=tuple(
            spec.station_to_trip[index] for index in keep
        ),
    )


def minimize_pricing_disagreement(spec: TinySpec, detail: dict):
    current = spec
    vector = tuple(detail["dual_vector"])
    current_detail = detail
    changed = True
    while changed and current.trip_count > 1:
        changed = False
        for remove in range(current.trip_count):
            keep = tuple(
                index for index in range(current.trip_count)
                if index != remove
            )
            candidate = induced_spec(current, keep)
            candidate_vector = tuple(vector[index] for index in keep)
            network = TinyNetwork(candidate)
            candidate_detail = evaluate_dual_vector(
                network, enumerate_route_costs(network), candidate_vector,
                kind=detail["kind"],
                sample_index=detail["sample_index"],
            )
            if candidate_detail["absolute_error"] > 1e-7:
                current = candidate
                vector = candidate_vector
                current_detail = candidate_detail
                changed = True
                break
    if current.station_count > 1:
        for station in range(current.station_count):
            candidate = induced_stations(current, (station,))
            network = TinyNetwork(candidate)
            candidate_detail = evaluate_dual_vector(
                network, enumerate_route_costs(network), vector,
                kind=detail["kind"],
                sample_index=detail["sample_index"],
            )
            if candidate_detail["absolute_error"] > 1e-7:
                current, current_detail = candidate, candidate_detail
                break
    return current, current_detail


def minimize_disagreement(spec: TinySpec, result: dict):
    signature = _disagreement_signature(result)
    current, current_result = spec, result
    changed = True
    while changed and current.trip_count > 2:
        changed = False
        for remove in range(current.trip_count):
            keep = tuple(
                index for index in range(current.trip_count)
                if index != remove
            )
            candidate = induced_spec(current, keep)
            candidate_result = compare_spec(candidate)
            if _disagreement_signature(candidate_result) == signature:
                current, current_result = candidate, candidate_result
                changed = True
                break
    if current.station_count > 1:
        for station in range(current.station_count):
            candidate = induced_stations(current, (station,))
            candidate_result = compare_spec(candidate)
            if _disagreement_signature(candidate_result) == signature:
                current, current_result = candidate, candidate_result
                break
    return current, current_result


def _zero_matrix(size):
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))


def mutation_specs():
    no_station_trip = ((False, False),)
    gap = TinySpec(
        "mutation_gap57", DEFAULT_SEED, 2, 1, 6, 1,
        (5, 66), (3, 3), (1, 1), _zero_matrix(2),
        ((False,), (False,)), no_station_trip,
    )
    reserve = TinySpec(
        "mutation_reserve", DEFAULT_SEED, 2, 1, 5, 2,
        (5, 15), (3, 3), (1, 1), _zero_matrix(2),
        ((False,), (False,)), no_station_trip,
    )
    transfer = TinySpec(
        "mutation_station_transfer", DEFAULT_SEED, 2, 2, 5, 1,
        (5, 25), (3, 3), (2, 2), _zero_matrix(2),
        ((True, False), (False, False)),
        ((False, False), (False, True)),
    )
    return {
        "gap_58_allowed": (
            gap, replace(gap, case_id="mutation_gap58", max_trip_gap=58)
        ),
        "soc_below_reserve": (
            reserve,
            replace(
                reserve, case_id="mutation_reserve_minus_one",
                reserve_relax_steps=1,
            ),
        ),
        "station_to_station": (
            transfer,
            replace(
                transfer, case_id="mutation_station_transfer_allowed",
                allow_station_transfer=True,
            ),
        ),
    }


def run_campaign(seed, cases, output_dir, dual_samples=32):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, disagreements = [], []
    for ordinal in range(cases):
        spec = generate_spec(seed, ordinal)
        result = compare_spec(spec, dual_samples=dual_samples)
        rows.append(result)
        if not result["agreement"]:
            payload = {
                "schema": SCHEMA,
                "original_spec": asdict(spec), "original_result": result,
            }
            if not result["lp_agrees"] or not result["integer_agrees"]:
                method_result = {
                    **result, "pricing_agrees": True,
                    "agreement": (
                        result["lp_agrees"] and result["integer_agrees"]
                    ),
                }
                minimal_spec, minimal_result = minimize_disagreement(
                    spec, method_result,
                )
                payload.update({
                    "minimal_spec": asdict(minimal_spec),
                    "minimal_result": minimal_result,
                    "minimality": (
                        "greedy one-trip and one-station irreducible"
                    ),
                })
            if result["pricing"]["mismatches"]:
                pricing_reproducers = []
                for detail in result["pricing"]["mismatches"]:
                    minimal_spec, minimal_detail = (
                        minimize_pricing_disagreement(spec, detail)
                    )
                    pricing_reproducers.append({
                        "minimal_spec": asdict(minimal_spec),
                        "minimal_pricing_disagreement": minimal_detail,
                        "minimality": (
                            "greedy one-trip and one-station irreducible"
                        ),
                    })
                payload["minimal_pricing_reproducers"] = pricing_reproducers
            path = output_dir / f"disagreement_{spec.case_id}.json"
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
            disagreements.append(str(path))
    mutations = {}
    for name, (baseline, mutated) in mutation_specs().items():
        baseline_result = compare_spec(baseline)
        mutated_result = compare_spec(mutated)
        changed = (
            baseline_result["integer"]["brute_force"]
            != mutated_result["integer"]["brute_force"]
        )
        mutations[name] = {
            "baseline": baseline_result,
            "mutated": mutated_result,
            "optimum_changed": changed,
        }
    summary = {
        "schema": SCHEMA,
        "seed": seed,
        "cases": cases,
        "agreements": sum(row["agreement"] for row in rows),
        "disagreements": len(disagreements),
        "agreement_rate": sum(row["agreement"] for row in rows) / cases,
        "dual_samples_per_network": dual_samples,
        "pricing_dual_samples": sum(
            row["pricing"]["samples"] for row in rows
        ),
        "pricing_agreements": sum(
            row["pricing"]["agreements"] for row in rows
        ),
        "pricing_agreement_rate": (
            sum(row["pricing"]["agreements"] for row in rows)
            / max(1, sum(row["pricing"]["samples"] for row in rows))
        ),
        "pricing_disagreements": sum(
            len(row["pricing"]["mismatches"]) for row in rows
        ),
        "pricing_max_absolute_error": max(
            row["pricing"]["max_absolute_error"] for row in rows
        ),
        "exact_cg_dp_max_absolute_error": max(
            row["pricing"]["exact_cg_dp_max_absolute_error"]
            for row in rows
        ),
        "constrained_dp_max_absolute_error": max(
            row["pricing"]["constrained_dp_max_absolute_error"]
            for row in rows
        ),
        "pricing_domain_coverage": {
            "soc_step": {
                str(value): sum(row["soc_step"] == value for row in rows)
                for value in sorted({row["soc_step"] for row in rows})
            },
            "station_count": {
                str(value): sum(
                    row["station_count"] == value for row in rows
                )
                for value in sorted({
                    row["station_count"] for row in rows
                })
            },
            "delayed_charging": {
                str(value).lower(): sum(
                    row["delayed_charging"] == value for row in rows
                )
                for value in (False, True)
            },
            "tariff_shape": {
                value: sum(row["tariff_shape"] == value for row in rows)
                for value in ("flat", "time_varying")
            },
            "negative_component_vectors": sum(
                row["pricing"]["negative_component_vectors"]
                for row in rows
            ),
            "near_or_exact_tie_vectors": sum(
                row["pricing"]["near_or_exact_tie_vectors"]
                for row in rows
            ),
            "station_reachability_density": {
                "minimum": min(
                    row["station_reachability_density"] for row in rows
                ),
                "maximum": max(
                    row["station_reachability_density"] for row in rows
                ),
            },
        },
        "disagreement_files": disagreements,
        "mutations": mutations,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "agreement.csv").open("w", newline="") as handle:
        fields = (
            "case_id", "trip_count", "station_count", "route_count",
            "soc_step", "delayed_charging", "tariff_shape",
            "station_reachability_density",
            "lp_agrees", "integer_agrees", "pricing_agrees", "agreement",
            "pricing_samples", "pricing_max_absolute_error",
            "cg_pool_columns", "bnp_nodes", "arc_variables", "wall_s",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["pricing_samples"] = row["pricing"]["samples"]
            flat["pricing_max_absolute_error"] = (
                row["pricing"]["max_absolute_error"]
            )
            writer.writerow({field: flat[field] for field in fields})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cases", type=int, default=240)
    parser.add_argument("--dual-samples", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cases < 1:
        parser.error("--cases must be positive")
    if args.dual_samples < 1:
        parser.error("--dual-samples must be positive")
    summary = run_campaign(
        args.seed, args.cases, args.output_dir, args.dual_samples,
    )
    print(json.dumps({
        "seed": summary["seed"], "cases": summary["cases"],
        "agreements": summary["agreements"],
        "disagreements": summary["disagreements"],
        "pricing_dual_samples": summary["pricing_dual_samples"],
        "pricing_disagreements": summary["pricing_disagreements"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

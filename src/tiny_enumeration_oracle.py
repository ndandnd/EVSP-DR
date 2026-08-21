#!/usr/bin/env python3
"""Exhaustive tiny SOC-time oracle and four-way differential campaign."""

from __future__ import annotations

import argparse
import csv
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


SCHEMA = "evsp-dr-tiny-differential-oracle-v1"
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
                for block in range(first, self.n_blocks):
                    node = self.charge_node.get(
                        (station, block, exit_level)
                    )
                    if node is not None:
                        add(left, node)
        for (station, block, level), left in self.charge_node.items():
            after = min(spec.capacity, level + spec.charge_step)
            block_end = (block + 1) * spec.block_min
            if block + 1 < self.n_blocks and after > level:
                add(left, self.charge_node[(station, block + 1, after)])
            for trip in self.problem.trips:
                if (
                    spec.station_to_trip[station][trip]
                    and block_end + 1 <= spec.start[trip]
                    and (trip, after) in self.trip_node
                ):
                    add(left, self.trip_node[(trip, after)], trip=trip)
            if after >= effective_reserve:
                add(left, self.SINK)
            if spec.allow_station_transfer and block + 1 < self.n_blocks:
                for other in range(spec.station_count):
                    if other != station:
                        add(
                            left,
                            self.charge_node[(other, block + 1, after)],
                        )
        self.n_arcs = sum(len(arcs) for arcs in self.out)
        self.sink_arcs = tuple(
            (left, cost) for left, arcs in enumerate(self.out)
            for right, cost, _trip in arcs if right == self.SINK
        )


def generate_spec(seed: int, ordinal: int) -> TinySpec:
    rng = random.Random((seed << 20) + ordinal)
    trip_count = rng.randint(8, 14)
    station_count = rng.randint(1, 2)
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


def column_generation(network, initial_pool=None, constraints=()):
    trips = list(network.problem.trips)
    pool = {
        record["mask"]: record for record in (
            initial_pool or route_records(
                [1 << trip for trip in trips], len(trips),
            )
        )
    }
    pricer = ConstrainedDAGPricer(network)

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
        solved = column_generation(network, list(pool.values()), constraints)
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


def compare_spec(spec: TinySpec) -> dict:
    network = TinyNetwork(spec)
    started = time.perf_counter()
    masks = enumerate_route_masks(network)
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
    lp_agrees = max(lp_values.values()) - min(lp_values.values()) <= 1e-7
    integer_agrees = len(set(integer_values.values())) == 1
    return {
        "case_id": spec.case_id,
        "trip_count": spec.trip_count,
        "station_count": spec.station_count,
        "route_count": len(masks),
        "lp": lp_values,
        "integer": integer_values,
        "lp_agrees": lp_agrees,
        "integer_agrees": integer_agrees,
        "agreement": lp_agrees and integer_agrees,
        "cg_pool_columns": len(cg["pool"]),
        "bnp_nodes": bnp["nodes"],
        "arc_variables": arc["variables"],
        "wall_s": time.perf_counter() - started,
    }


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
        "mutation_station_transfer", DEFAULT_SEED, 2, 2, 4, 1,
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


def run_campaign(seed, cases, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, disagreements = [], []
    for ordinal in range(cases):
        spec = generate_spec(seed, ordinal)
        result = compare_spec(spec)
        rows.append(result)
        if not result["agreement"]:
            path = output_dir / f"disagreement_{spec.case_id}.json"
            path.write_text(json.dumps({
                "schema": SCHEMA, "spec": asdict(spec), "result": result,
            }, indent=2, sort_keys=True) + "\n")
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
        "disagreement_files": disagreements,
        "mutations": mutations,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "agreement.csv").open("w", newline="") as handle:
        fields = (
            "case_id", "trip_count", "station_count", "route_count",
            "lp_agrees", "integer_agrees", "agreement",
            "cg_pool_columns", "bnp_nodes", "arc_variables", "wall_s",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cases", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cases < 1:
        parser.error("--cases must be positive")
    summary = run_campaign(args.seed, args.cases, args.output_dir)
    print(json.dumps({
        "seed": summary["seed"], "cases": summary["cases"],
        "agreements": summary["agreements"],
        "disagreements": summary["disagreements"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

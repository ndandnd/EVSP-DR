#!/usr/bin/env python3
"""Exact fixed-duty charging optimization on the expanded SOC/time DAG."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from audit_giro_known_columns import DEPOT, HORIZON_MIN, STATIONS
from config import BUS_COST_KX, CHARGE_START_COST
from expanded_path_realization import (
    BLOCK_SCHEDULE_SCHEMA,
    realize_expanded_path,
    realized_costs,
)
from run_exact_pool_mip import validate_injected_route
from tariff_response_core import PHYSICS, canonical_sha
from utils_v2 import base_station_name


CERTIFICATE_SCHEMA = "evsp-dr-fixed-duty-expanded-certificate-v1"
RESULT_SCHEMA = "evsp-dr-fixed-duty-expanded-result-v1"


@dataclass(frozen=True)
class Arc:
    successor: object
    travel_min: float
    deadhead_kwh: float
    kind: str


def _arc_groups(problem):
    groups = {
        "depot_trip": {},
        "trip_depot": {},
        "trip_trip": {},
        "trip_station": {},
        "station_trip": {},
        "station_depot": {},
    }
    for source, arcs in problem.adjacency.items():
        for successor, travel, energy, kind in arcs:
            arc = Arc(
                successor, float(travel), float(energy), str(kind)
            )
            if kind == "depot_trip":
                groups[kind][successor] = arc
            elif kind in {"trip_depot", "station_depot"}:
                groups[kind][source] = arc
            elif kind in {"trip_trip", "trip_station", "station_trip"}:
                groups[kind].setdefault(source, {})[successor] = arc
    return groups


def _floor(grid, soc_step, soc):
    level = min(
        max(int(math.floor((soc + 1e-9) / soc_step)), 0),
        len(grid) - 1,
    )
    if grid[level] > soc + 1e-9:
        level -= 1
    return level


def _action_key(action):
    return (
        action["kind"],
        action.get("station", ""),
        action.get("first_block", -1),
        action.get("last_block", -1),
        action.get("next_trip", -1),
    )


def _accept(states, level, cost, actions):
    current = states.get(level)
    key = (float(cost), tuple(_action_key(action) for action in actions))
    if current is None or key < (
        current[0], tuple(_action_key(action) for action in current[1])
    ):
        states[level] = (float(cost), actions)
        return True
    return False


def optimize_fixed_duty(
    problem,
    trip_sequence,
    station_prices,
    *,
    g_kwh=300.0,
    charge_kw=300.0,
    reserve_kwh=0.0,
    soc_step=15.0,
    block_min=10,
    tariff_id=None,
    tariff_sha256=None,
    instance_sha256=None,
    allow_diagnostic_grid=False,
):
    started = time.perf_counter()
    trips = tuple(int(trip) for trip in trip_sequence)
    if (
        not trips
        or len(trips) != len(set(trips))
        or any(trip not in set(problem.trips) for trip in trips)
    ):
        raise ValueError("fixed duty has invalid or foreign trips")
    physics_tuple = (
        float(g_kwh), float(charge_kw), float(reserve_kwh),
        float(soc_step), int(block_min),
    )
    if not allow_diagnostic_grid and physics_tuple != (
        300.0, 300.0, 0.0, 15.0, 10
    ):
        raise ValueError("fixed-duty pilot physics differ")
    if allow_diagnostic_grid and (
        physics_tuple[:3] != (300.0, 300.0, 0.0)
        or float(soc_step) not in {15.0, 5.0, 2.5, 1.0}
        or int(block_min) not in {10, 5}
        or (int(block_min) == 5 and float(soc_step) != 1.0)
    ):
        raise ValueError("unsupported fixed-duty diagnostic grid")
    g_kwh = float(g_kwh)
    charge_kw = float(charge_kw)
    reserve_kwh = float(reserve_kwh)
    soc_step = float(soc_step)
    block_min = int(block_min)
    required_hours = set(range(int(math.ceil(HORIZON_MIN / 60.0))))
    if any(
        set(curve) != set(range(max(curve) + 1))
        or not required_hours <= set(curve)
        for curve in station_prices.values()
    ):
        raise ValueError("tariff curves are incomplete")
    arcs = _arc_groups(problem)
    problem_identity_sha256 = canonical_sha({
        "trips": list(problem.trips),
        "trip_energy": {
            str(trip): float(problem.trip_energy[trip])
            for trip in problem.trips
        },
        "start_min": {
            str(trip): float(problem.start_min[trip])
            for trip in problem.trips
        },
        "end_min": {
            str(trip): float(problem.end_min[trip])
            for trip in problem.trips
        },
        "adjacency": [
            [
                str(source), str(successor), float(travel),
                float(energy), str(kind),
            ]
            for source, entries in sorted(
                problem.adjacency.items(), key=lambda item: str(item[0])
            )
            for successor, travel, energy, kind in sorted(
                entries, key=lambda item: (
                    str(item[0]), float(item[1]), float(item[2]), str(item[3])
                )
            )
        ],
    })
    grid = [
        round(index * soc_step, 6)
        for index in range(int(g_kwh / soc_step) + 1)
    ]
    block_kwh = charge_kw * block_min / 60.0
    n_blocks = int(HORIZON_MIN) // int(block_min)
    first = arcs["depot_trip"].get(trips[0])
    if first is None or first.travel_min > problem.start_min[trips[0]] + 1e-9:
        return _infeasible(
            trips, tariff_id, tariff_sha256,
            "depot cannot reach first trip", started,
        )
    first_level = _floor(grid, soc_step, g_kwh - first.deadhead_kwh)
    if (
        first_level < 0
        or grid[first_level] + 1e-9
        < problem.trip_energy[trips[0]] + reserve_kwh
    ):
        return _infeasible(
            trips, tariff_id, tariff_sha256,
            "first trip violates SOC", started,
        )
    states = {
        first_level: (
            float(BUS_COST_KX),
            ({
                "kind": "source",
                "next_trip": trips[0],
                "travel_min": first.travel_min,
                "deadhead_kwh": first.deadhead_kwh,
                "waiting_min": (
                    problem.start_min[trips[0]] - first.travel_min
                ),
            },),
        )
    }
    labels = 1
    transitions = 0
    frontier_hashes = []
    for position, trip in enumerate(trips):
        frontier_hashes.append(canonical_sha({
            "position": position,
            "trip": trip,
            "states": [
                [level, value[0]]
                for level, value in sorted(states.items())
            ],
        }))
        final_gap = position == len(trips) - 1
        successor = None if final_gap else trips[position + 1]
        next_states = {}
        terminal = []
        for level, (base_cost, actions) in sorted(states.items()):
            soc_exit = grid[level] - problem.trip_energy[trip]
            depart = problem.end_min[trip]
            direct = (
                arcs["trip_depot"].get(trip)
                if final_gap
                else arcs["trip_trip"].get(trip, {}).get(successor)
            )
            if direct is not None:
                arrival = depart + direct.travel_min
                deadline = (
                    HORIZON_MIN
                    if final_gap else problem.start_min[successor]
                )
                remaining = soc_exit - direct.deadhead_kwh
                next_level = _floor(grid, soc_step, remaining)
                if (
                    arrival <= deadline + 1e-9
                    and remaining >= reserve_kwh - 1e-9
                    and (
                        final_gap
                        or (
                            next_level >= 0
                            and grid[next_level] + 1e-9
                            >= problem.trip_energy[successor] + reserve_kwh
                        )
                    )
                ):
                    action = {
                        "kind": "direct",
                        "from_trip": trip,
                        "next_trip": successor,
                        "travel_min": direct.travel_min,
                        "deadhead_kwh": direct.deadhead_kwh,
                        "waiting_min": (
                            0.0 if final_gap
                            else max(0.0, deadline - arrival)
                        ),
                    }
                    transitions += 1
                    if final_gap:
                        terminal.append((base_cost, actions + (action,)))
                    elif _accept(
                        next_states, next_level, base_cost,
                        actions + (action,),
                    ):
                        labels += 1

            for station, to_station in sorted(
                arcs["trip_station"].get(trip, {}).items()
            ):
                from_station = (
                    arcs["station_depot"].get(station)
                    if final_gap
                    else arcs["station_trip"].get(station, {}).get(successor)
                )
                if from_station is None:
                    continue
                station_arrival = depart + to_station.travel_min
                soc_arrival = soc_exit - to_station.deadhead_kwh
                if soc_arrival < reserve_kwh - 1e-9:
                    continue
                entry_level = _floor(grid, soc_step, soc_arrival)
                if entry_level < 0:
                    continue
                deadline = (
                    HORIZON_MIN
                    if final_gap else problem.start_min[successor]
                )
                first_block = max(
                    0, int(math.ceil(
                        station_arrival / block_min - 1e-9
                    ))
                )
                last_possible = min(
                    n_blocks - 1,
                    int(math.floor(
                        (deadline - from_station.travel_min)
                        / block_min + 1e-9
                    )) - 1,
                )
                curve = station_prices[base_station_name(station)]
                for start_block in range(
                    first_block, last_possible + 1
                ):
                    charge_level = entry_level
                    charging_cost = 0.0
                    for end_block in range(
                        start_block, last_possible + 1
                    ):
                        after_soc = min(
                            g_kwh, grid[charge_level] + block_kwh
                        )
                        after_level = _floor(grid, soc_step, after_soc)
                        gain = grid[after_level] - grid[charge_level]
                        hour = int(end_block * block_min // 60)
                        if hour not in curve:
                            raise ValueError("tariff hour missing in DP")
                        charging_cost += gain * curve[hour]
                        departure = (end_block + 1) * block_min
                        remaining = (
                            grid[after_level] - from_station.deadhead_kwh
                        )
                        next_level = _floor(grid, soc_step, remaining)
                        if (
                            departure + from_station.travel_min
                            <= deadline + 1e-9
                            and remaining >= reserve_kwh - 1e-9
                            and (
                                final_gap
                                or (
                                    next_level >= 0
                                    and grid[next_level] + 1e-9
                                    >= problem.trip_energy[successor]
                                    + reserve_kwh
                                )
                            )
                        ):
                            action = {
                                "kind": "charge",
                                "from_trip": trip,
                                "next_trip": successor,
                                "station": station,
                                "first_block": start_block,
                                "last_block": end_block,
                                "entry_level": entry_level,
                                "exit_level": after_level,
                                "expanded_grid_kwh": (
                                    grid[after_level] - grid[entry_level]
                                ),
                                "travel_min": (
                                    to_station.travel_min
                                    + from_station.travel_min
                                ),
                                "deadhead_kwh": (
                                    to_station.deadhead_kwh
                                    + from_station.deadhead_kwh
                                ),
                                "waiting_min": max(
                                    0.0,
                                    start_block * block_min - station_arrival,
                                ) + (0.0 if final_gap else max(
                                    0.0,
                                    deadline - (
                                        departure
                                        + from_station.travel_min
                                    ),
                                )),
                            }
                            candidate_cost = (
                                base_cost + CHARGE_START_COST
                                + charging_cost
                            )
                            transitions += 1
                            if final_gap:
                                terminal.append((
                                    candidate_cost,
                                    actions + (action,),
                                ))
                            elif _accept(
                                next_states, next_level, candidate_cost,
                                actions + (action,),
                            ):
                                labels += 1
                        if gain <= 1e-9:
                            break
                        charge_level = after_level
        if final_gap:
            if not terminal:
                return _infeasible(
                    trips, tariff_id, tariff_sha256,
                    f"no feasible terminal path after trip {trip}", started,
                )
            best_cost, best_actions = min(
                terminal,
                key=lambda item: (
                    item[0],
                    tuple(_action_key(action) for action in item[1]),
                ),
            )
        else:
            if not next_states:
                return _infeasible(
                    trips, tariff_id, tariff_sha256,
                    f"no fixed-duty transition {trip}->{successor}",
                    started,
                )
            states = next_states

    route_nodes = [DEPOT, trips[0]]
    charging = {"stations": [], "cst": [], "cet": [], "kwh": []}
    for action in best_actions[1:]:
        if action["kind"] == "charge":
            route_nodes.append(action["station"])
            charging["stations"].append(action["station"])
            charging["cst"].append(action["first_block"] * block_min)
            charging["cet"].append(
                (action["last_block"] + 1) * block_min
            )
            charging["kwh"].append(action["expanded_grid_kwh"])
        if action["next_trip"] is not None:
            route_nodes.append(action["next_trip"])
    route_nodes.append(DEPOT)
    record = {
        "trips": list(trips),
        "route_nodes": route_nodes,
        "charging_stops": charging,
        "cost": float(best_cost),
    }
    realized, detail = realize_expanded_path(
        problem,
        record,
        g_kwh=g_kwh,
        charge_kw=charge_kw,
        reserve_kwh=reserve_kwh,
        soc_step=soc_step,
        block_min=block_min,
    )
    if realized is None:
        raise ValueError(
            f"certified DP path failed realization: {detail['reason']}"
        )
    reason = validate_injected_route(
        problem,
        realized,
        g_kwh,
        charge_kw,
        reserve_kwh,
        HORIZON_MIN,
    )
    if reason is not None:
        raise ValueError(f"certified DP path failed replay: {reason}")
    costs = realized_costs(
        {**realized, "cost": float(best_cost)},
        detail["mapping"],
        station_prices=station_prices,
    )
    if not math.isclose(
        costs["recomputed_expanded_grid_cost"],
        best_cost,
        rel_tol=1e-10,
        abs_tol=1e-6,
    ):
        raise ValueError("DP objective differs from expanded replay")
    continuous = costs["continuous_realized_charging_blocks"]
    waiting_min = sum(
        action.get("waiting_min", 0.0) for action in best_actions
    )
    deadhead_min = sum(
        action.get("travel_min", 0.0) for action in best_actions
    )
    deadhead_kwh = sum(
        action.get("deadhead_kwh", 0.0) for action in best_actions
    )
    realized.update({
        "cost": float(best_cost),
        "expanded_grid_cost": float(best_cost),
        "continuous_realized_cost": costs[
            "continuous_realized_cost"
        ],
        "continuous_realized_charging_blocks": continuous,
        "master_cost_semantics": "expanded_grid_cost",
        "cost_tariff_sha256": tariff_sha256,
        "continuous_terminal_soc_kwh": detail["mapping"][
            "continuous_terminal_soc_kwh"
        ],
        "expanded_grid_terminal_soc_kwh": detail["mapping"][
            "expanded_grid_terminal_soc_kwh"
        ],
        "waiting_min": waiting_min,
        "deadhead_min": deadhead_min,
        "deadhead_kwh": deadhead_kwh,
        "physical_realization": {
            **detail["mapping"],
            "status": "validated_continuous_fixed_duty",
            "continuous_realized_charging_blocks_schema":
                BLOCK_SCHEDULE_SCHEMA,
            "continuous_realized_charging_blocks_sha256": costs[
                "continuous_realized_charging_blocks_sha256"
            ],
            "continuous_cost_pricing_certified": False,
        },
    })
    selected_path_sha256 = canonical_sha({
        "actions": list(best_actions),
        "route_nodes": route_nodes,
        "expanded_grid_charging_stops":
            realized["expanded_grid_charging_stops"],
    })
    certificate_payload = {
        "schema": CERTIFICATE_SCHEMA,
        "certified": True,
        "scope":
            "optimal_discretized_charging_for_fixed_trip_sequence",
        "algorithm": "acyclic_dynamic_programming_exhaustive",
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "trip_sequence": list(trips),
        "instance_sha256": instance_sha256,
        "problem_identity_sha256": problem_identity_sha256,
        "selected_path_sha256": selected_path_sha256,
        "state_frontier_sha256": canonical_sha(frontier_hashes),
        "tariff_id": tariff_id,
        "tariff_sha256": tariff_sha256,
        "physics": {
            **PHYSICS,
            "g_kwh": g_kwh,
            "charge_kw": charge_kw,
            "reserve_kwh": reserve_kwh,
            "soc_step": soc_step,
            "block_min": block_min,
        },
        "objective": best_cost,
        "labels_accepted": labels,
        "transitions_evaluated": transitions,
        "continuous_cost_optimality_certified": False,
    }
    certificate_sha256 = canonical_sha(certificate_payload)
    realized["fixed_duty_certificate_sha256"] = certificate_sha256
    return {
        "schema": RESULT_SCHEMA,
        "feasible": True,
        "route": realized,
        "expanded_grid_objective": best_cost,
        "continuous_replay_objective":
            costs["continuous_realized_cost"],
        "waiting_min": waiting_min,
        "deadhead_min": deadhead_min,
        "deadhead_kwh": deadhead_kwh,
        "certificate": {
            **certificate_payload,
            "certificate_sha256": certificate_sha256,
        },
        "physical_replay_status": "validated",
        "runtime_s": time.perf_counter() - started,
    }


def _infeasible(trips, tariff_id, tariff_sha256, reason, started):
    return {
        "schema": RESULT_SCHEMA,
        "feasible": False,
        "trip_sequence": list(trips),
        "tariff_id": tariff_id,
        "tariff_sha256": tariff_sha256,
        "reason": reason,
        "certificate": {
            "schema": CERTIFICATE_SCHEMA,
            "certified": False,
            "scope": "fixed_trip_sequence_infeasible_or_unreachable",
            "continuous_cost_optimality_certified": False,
        },
        "physical_replay_status": "not_available",
        "runtime_s": time.perf_counter() - started,
    }

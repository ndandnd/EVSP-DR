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


def evaluate_fixed_duty_transition(
    problem,
    arcs,
    *,
    trip,
    successor,
    final_gap,
    level,
    base_cost,
    actions,
    grid,
    soc_step,
    block_min,
    g_kwh,
    charge_kw,
    reserve_kwh,
    station_prices,
    n_blocks,
    include_trace=False,
):
    """Pure production transition evaluator for one predecessor label.

    The returned candidates are the only transitions the production DP may
    accept. Diagnostic rows retain every predicate and raw quantity but cannot
    affect candidate generation or ordering.
    """
    candidates = []
    rows = []
    soc_entry = float(grid[level])
    trip_energy = float(problem.trip_energy[trip])
    soc_exit = soc_entry - trip_energy
    depart = float(problem.end_min[trip])
    deadline = (
        float(HORIZON_MIN)
        if final_gap else float(problem.start_min[successor])
    )
    successor_energy = (
        None if final_gap else float(problem.trip_energy[successor])
    )

    def emit(row):
        if include_trace:
            row["failed_predicates"] = sorted(
                key for key, value in row["predicates"].items()
                if value is False
            )
            rows.append(row)

    direct = (
        arcs["trip_depot"].get(trip)
        if final_gap
        else arcs["trip_trip"].get(trip, {}).get(successor)
    )
    if direct is None:
        emit({
            "option_kind": "direct",
            "station": None,
            "predecessor_level": int(level),
            "predecessor_soc_kwh": soc_entry,
            "trip_energy_kwh": trip_energy,
            "soc_after_trip_kwh": soc_exit,
            "deadline_min": deadline,
            "successor_energy_kwh": successor_energy,
            "predicates": {
                "direct_arc_exists": False,
            },
            "accepted": False,
        })
    else:
        arrival = depart + direct.travel_min
        remaining = soc_exit - direct.deadhead_kwh
        next_level = _floor(grid, soc_step, remaining)
        predicates = {
            "direct_arc_exists": True,
            "deadline_satisfied": arrival <= deadline + 1e-9,
            "resulting_soc_meets_reserve":
                remaining >= reserve_kwh - 1e-9,
            "resulting_level_valid": final_gap or next_level >= 0,
            "successor_energy_and_reserve_satisfied": (
                True if final_gap else (
                    next_level >= 0
                    and grid[next_level] + 1e-9
                    >= successor_energy + reserve_kwh
                )
            ),
        }
        accepted = all(predicates.values())
        action = {
            "kind": "direct",
            "from_trip": trip,
            "next_trip": successor,
            "travel_min": direct.travel_min,
            "deadhead_kwh": direct.deadhead_kwh,
            "waiting_min": (
                0.0 if final_gap else max(0.0, deadline - arrival)
            ),
        }
        if accepted:
            candidates.append({
                "cost": float(base_cost),
                "actions": actions + (action,),
                "action": action,
                "next_level": int(next_level),
                "terminal": bool(final_gap),
            })
        emit({
            "option_kind": "direct",
            "station": None,
            "predecessor_level": int(level),
            "predecessor_soc_kwh": soc_entry,
            "trip_energy_kwh": trip_energy,
            "soc_after_trip_kwh": soc_exit,
            "direct_arc_exists": True,
            "direct_arc_type": direct.kind,
            "arrival_min": arrival,
            "deadline_min": deadline,
            "outgoing_deadhead_min": direct.travel_min,
            "outgoing_deadhead_kwh": direct.deadhead_kwh,
            "resulting_soc_before_floor_kwh": remaining,
            "resulting_soc_level": int(next_level),
            "resulting_soc_kwh": (
                grid[next_level] if next_level >= 0 else None
            ),
            "successor_energy_kwh": successor_energy,
            "reserve_kwh": reserve_kwh,
            "predicates": predicates,
            "accepted": accepted,
        })

    block_kwh = charge_kw * block_min / 60.0
    available_to_station = arcs["trip_station"].get(trip, {})
    station_nodes = set(STATIONS) | set(available_to_station)
    for station in sorted(station_nodes):
        to_station = available_to_station.get(station)
        from_station = (
            arcs["station_depot"].get(station)
            if final_gap
            else arcs["station_trip"].get(station, {}).get(successor)
        )
        graph_predicates = {
            "trip_to_station_arc_exists": to_station is not None,
            "station_to_successor_arc_exists": from_station is not None,
        }
        if to_station is None or from_station is None:
            emit({
                "option_kind": "station",
                "station": station,
                "predecessor_level": int(level),
                "predecessor_soc_kwh": soc_entry,
                "trip_energy_kwh": trip_energy,
                "soc_after_trip_kwh": soc_exit,
                "deadline_min": deadline,
                "successor_energy_kwh": successor_energy,
                "trip_to_station_arc_type": (
                    to_station.kind if to_station is not None else None
                ),
                "station_to_successor_arc_type": (
                    from_station.kind if from_station is not None else None
                ),
                "predicates": graph_predicates,
                "accepted": False,
            })
            continue
        station_arrival = depart + to_station.travel_min
        soc_arrival = soc_exit - to_station.deadhead_kwh
        entry_level = _floor(grid, soc_step, soc_arrival)
        entry_soc = grid[entry_level] if entry_level >= 0 else None
        first_block = max(
            0,
            int(math.ceil(station_arrival / block_min - 1e-9)),
        )
        last_possible = min(
            n_blocks - 1,
            int(math.floor(
                (deadline - from_station.travel_min)
                / block_min + 1e-9
            )) - 1,
        )
        common_predicates = {
            **graph_predicates,
            "station_arrival_soc_meets_reserve":
                soc_arrival >= reserve_kwh - 1e-9,
            "station_entry_level_valid": entry_level >= 0,
            "charging_window_has_usable_block":
                last_possible >= first_block,
        }
        if not all(common_predicates.values()):
            emit({
                "option_kind": "station",
                "station": station,
                "predecessor_level": int(level),
                "predecessor_soc_kwh": soc_entry,
                "trip_energy_kwh": trip_energy,
                "soc_after_trip_kwh": soc_exit,
                "station_arrival_min": station_arrival,
                "station_arrival_soc_before_floor_kwh": soc_arrival,
                "station_entry_level": int(entry_level),
                "station_entry_soc_kwh": entry_soc,
                "first_charging_block": int(first_block),
                "last_possible_charging_block": int(last_possible),
                "usable_blocks": max(
                    0, int(last_possible - first_block + 1)
                ),
                "usable_minutes": max(
                    0, int(last_possible - first_block + 1)
                ) * block_min,
                "deadline_min": deadline,
                "outgoing_deadhead_min": from_station.travel_min,
                "outgoing_deadhead_kwh": from_station.deadhead_kwh,
                "trip_to_station_arc_type": to_station.kind,
                "station_to_successor_arc_type": from_station.kind,
                "successor_energy_kwh": successor_energy,
                "reserve_kwh": reserve_kwh,
                "predicates": common_predicates,
                "accepted": False,
            })
            continue
        curve = station_prices[base_station_name(station)]
        for start_block in range(first_block, last_possible + 1):
            charge_level = entry_level
            charging_cost = 0.0
            for end_block in range(start_block, last_possible + 1):
                before_soc = grid[charge_level]
                after_soc_before_floor = min(
                    g_kwh, before_soc + block_kwh
                )
                after_level = _floor(
                    grid, soc_step, after_soc_before_floor
                )
                gain = grid[after_level] - before_soc
                hour = int(end_block * block_min // 60)
                if hour not in curve:
                    raise ValueError("tariff hour missing in DP")
                charging_cost += gain * curve[hour]
                departure = (end_block + 1) * block_min
                remaining = (
                    grid[after_level] - from_station.deadhead_kwh
                )
                next_level = _floor(grid, soc_step, remaining)
                predicates = {
                    **common_predicates,
                    "battery_cap_satisfied":
                        after_soc_before_floor <= g_kwh + 1e-9,
                    "departure_deadline_satisfied": (
                        departure + from_station.travel_min
                        <= deadline + 1e-9
                    ),
                    "resulting_soc_meets_reserve":
                        remaining >= reserve_kwh - 1e-9,
                    "resulting_level_valid":
                        final_gap or next_level >= 0,
                    "successor_energy_and_reserve_satisfied": (
                        True if final_gap else (
                            next_level >= 0
                            and grid[next_level] + 1e-9
                            >= successor_energy + reserve_kwh
                        )
                    ),
                }
                accepted = all(predicates.values())
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
                    ) + (
                        0.0 if final_gap else max(
                            0.0,
                            deadline - (
                                departure + from_station.travel_min
                            ),
                        )
                    ),
                }
                candidate_cost = (
                    base_cost + CHARGE_START_COST + charging_cost
                )
                if accepted:
                    candidates.append({
                        "cost": float(candidate_cost),
                        "actions": actions + (action,),
                        "action": action,
                        "next_level": int(next_level),
                        "terminal": bool(final_gap),
                    })
                emit({
                    "option_kind": "station",
                    "station": station,
                    "predecessor_level": int(level),
                    "predecessor_soc_kwh": soc_entry,
                    "trip_energy_kwh": trip_energy,
                    "soc_after_trip_kwh": soc_exit,
                    "trip_to_station_arc_type": to_station.kind,
                    "station_to_successor_arc_type": from_station.kind,
                    "station_arrival_min": station_arrival,
                    "station_arrival_soc_before_floor_kwh": soc_arrival,
                    "station_entry_level": int(entry_level),
                    "station_entry_soc_kwh": entry_soc,
                    "first_charging_block": int(start_block),
                    "last_charging_block": int(end_block),
                    "last_possible_charging_block": int(last_possible),
                    "usable_blocks": int(end_block - start_block + 1),
                    "usable_minutes":
                        int(end_block - start_block + 1) * block_min,
                    "delayed_charging":
                        bool(start_block > first_block),
                    "charge_gain_before_floor_kwh": (
                        after_soc_before_floor - before_soc
                    ),
                    "charge_gain_after_floor_kwh": gain,
                    "cumulative_grid_charge_gain_kwh": (
                        grid[after_level] - grid[entry_level]
                    ),
                    "battery_cap_kwh": g_kwh,
                    "departure_min": departure,
                    "deadline_min": deadline,
                    "outgoing_deadhead_min": from_station.travel_min,
                    "outgoing_deadhead_kwh":
                        from_station.deadhead_kwh,
                    "resulting_soc_before_floor_kwh": remaining,
                    "resulting_soc_level": int(next_level),
                    "resulting_soc_kwh": (
                        grid[next_level] if next_level >= 0 else None
                    ),
                    "successor_energy_kwh": successor_energy,
                    "reserve_kwh": reserve_kwh,
                    "candidate_cost": float(candidate_cost),
                    "predicates": predicates,
                    "accepted": accepted,
                })
                if gain <= 1e-9:
                    break
                charge_level = after_level
    return candidates, rows


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
    allow_declared_physics=False,
    trace=False,
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
    if allow_declared_physics:
        if (
            not all(math.isfinite(value) for value in physics_tuple[:4])
            or float(g_kwh) <= 0.0
            or float(charge_kw) <= 0.0
            or not 0.0 <= float(reserve_kwh) <= float(g_kwh)
            or not 0.0 < float(soc_step) <= float(g_kwh)
            or int(block_min) <= 0
            or int(HORIZON_MIN) % int(block_min) != 0
        ):
            raise ValueError("invalid plan-declared fixed-duty physics")
    elif not allow_diagnostic_grid and physics_tuple != (
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
    diagnostic_frontiers = []
    diagnostic_candidates = []
    for position, trip in enumerate(trips):
        frontier_hashes.append(canonical_sha({
            "position": position,
            "trip": trip,
            "states": [
                [level, value[0]]
                for level, value in sorted(states.items())
            ],
        }))
        if trace:
            diagnostic_frontiers.extend({
                "position": position,
                "trip": trip,
                "successor": (
                    None if position == len(trips) - 1
                    else trips[position + 1]
                ),
                "level": int(level),
                "soc_kwh": float(grid[level]),
                "cost": float(value[0]),
                "actions": list(value[1]),
            } for level, value in sorted(states.items()))
        final_gap = position == len(trips) - 1
        successor = None if final_gap else trips[position + 1]
        next_states = {}
        terminal = []
        for level, (base_cost, actions) in sorted(states.items()):
            evaluated, trace_rows = evaluate_fixed_duty_transition(
                problem,
                arcs,
                trip=trip,
                successor=successor,
                final_gap=final_gap,
                level=level,
                base_cost=base_cost,
                actions=actions,
                grid=grid,
                soc_step=soc_step,
                block_min=block_min,
                g_kwh=g_kwh,
                charge_kw=charge_kw,
                reserve_kwh=reserve_kwh,
                station_prices=station_prices,
                n_blocks=n_blocks,
                include_trace=trace,
            )
            if trace:
                for row in trace_rows:
                    row.update({
                        "position": position,
                        "trip": trip,
                        "successor": successor,
                    })
                diagnostic_candidates.extend(trace_rows)
            for candidate in evaluated:
                transitions += 1
                if candidate["terminal"]:
                    terminal.append((
                        candidate["cost"], candidate["actions"],
                    ))
                elif _accept(
                    next_states,
                    candidate["next_level"],
                    candidate["cost"],
                    candidate["actions"],
                ):
                    labels += 1
        if final_gap:
            if not terminal:
                return _infeasible(
                    trips, tariff_id, tariff_sha256,
                    f"no feasible terminal path after trip {trip}", started,
                    diagnostic_trace=(
                        {
                            "frontier_states": diagnostic_frontiers,
                            "transition_candidates": diagnostic_candidates,
                            "failed_transition": {
                                "position": position,
                                "trip": trip,
                                "successor": None,
                            },
                        } if trace else None
                    ),
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
                    diagnostic_trace=(
                        {
                            "frontier_states": diagnostic_frontiers,
                            "transition_candidates": diagnostic_candidates,
                            "failed_transition": {
                                "position": position,
                                "trip": trip,
                                "successor": successor,
                            },
                        } if trace else None
                    ),
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
    result = {
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
    if trace:
        result["diagnostic_trace"] = {
            "frontier_states": diagnostic_frontiers,
            "transition_candidates": diagnostic_candidates,
            "failed_transition": None,
        }
    return result


def _infeasible(
    trips,
    tariff_id,
    tariff_sha256,
    reason,
    started,
    *,
    diagnostic_trace=None,
):
    result = {
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
    if diagnostic_trace is not None:
        result["diagnostic_trace"] = diagnostic_trace
    return result

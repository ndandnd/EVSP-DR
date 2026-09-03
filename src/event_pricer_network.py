"""Event-time trip/SOC DAG for exact reduced-cost route pricing."""

from __future__ import annotations

import hashlib
import json
import math
import time
from array import array
from copy import deepcopy

import numpy as np

from audit_giro_known_columns import DEPOT, HORIZON_MIN, STATIONS
from config import BUS_COST_KX, CHARGE_START_COST, charge_cost_premium
from expanded_path_realization import (
    BLOCK_SCHEDULE_SCHEMA,
    charging_block_schedule_sha256,
    normalize_event_station_prices,
    realize_expanded_path,
    realized_costs,
)
from fixed_duty_continuous_optimizer import _segments
from run_exact_pool_mip import validate_injected_route
from utils_v2 import base_station_name


TOL = 1e-9


def _floor_level(grid, step, value):
    level = min(
        max(int(math.floor((value + 1e-9) / step)), 0),
        len(grid) - 1,
    )
    if grid[level] > value + 1e-9:
        level -= 1
    return level


def _event_times(problem, station_prices, uniform_block_min):
    """Return station-specific exact, tariff, and reachable uniform events."""

    arrivals = {station: set() for station in STATIONS}
    deadlines = {station: set() for station in STATIONS}
    for source, arcs in problem.adjacency.items():
        for successor, travel, _energy, kind in arcs:
            if kind == "trip_station" and successor in arrivals:
                arrivals[successor].add(
                    float(problem.end_min[source]) + float(travel)
                )
            elif kind == "station_trip" and source in deadlines:
                deadlines[source].add(
                    float(problem.start_min[successor]) - float(travel)
                )
            elif kind == "station_depot" and source in deadlines:
                deadlines[source].add(float(HORIZON_MIN) - float(travel))
    output = {}
    for station in STATIONS:
        times = set(arrivals[station]) | set(deadlines[station])
        curve = station_prices[base_station_name(station)]
        times.update(
            float(hour * 60)
            for hour in range(max(curve) + 2)
            if 0 <= hour * 60 <= HORIZON_MIN
        )
        if arrivals[station] and deadlines[station]:
            lower = min(arrivals[station])
            upper = max(deadlines[station])
            first = int(math.ceil(lower / uniform_block_min - 1e-9))
            last = int(math.floor(upper / uniform_block_min + 1e-9))
            times.update(
                float(index * uniform_block_min)
                for index in range(first, last + 1)
            )
        output[station] = tuple(sorted(
            value for value in times
            if -TOL <= value <= HORIZON_MIN + TOL
        ))
    return output


def _window_cost(station, start, duration, energy, prices, charge_kw):
    end = start + duration
    segments = _segments(start, end, station, prices, charge_kw)
    remaining = float(energy)
    cost = 0.0
    for segment in segments:
        delivered = min(remaining, segment.capacity_kwh)
        cost += delivered * segment.price_per_kwh
        remaining -= delivered
        if remaining <= 1e-8:
            break
    if remaining > 1e-6:
        raise ValueError("event window cannot deliver target energy")
    return cost


def _best_charge_window(
    station,
    arrival,
    deadline,
    energy,
    *,
    event_times,
    station_prices,
    charge_kw,
):
    duration = float(energy) * 60.0 / float(charge_kw)
    latest_start = float(deadline) - duration
    if latest_start < arrival - TOL:
        return None
    curve = station_prices[base_station_name(station)]
    if len(set(curve.values())) == 1:
        start = float(arrival)
        return (
            float(energy) * float(next(iter(curve.values())))
            * float(charge_cost_premium),
            start,
            start + duration,
        )
    candidates = {float(arrival), float(latest_start)}
    for event in event_times[station]:
        if abs(event / 60.0 - round(event / 60.0)) <= TOL:
            candidates.add(float(event))
            candidates.add(float(event) - duration)
    feasible = sorted(
        value for value in candidates
        if value >= arrival - TOL and value <= latest_start + TOL
    )
    if not feasible:
        return None
    return min(
        (
            _window_cost(
                station, start, duration, energy,
                station_prices, charge_kw,
            ),
            start,
            start + duration,
        )
        for start in feasible
    )


class EventExpandedNetwork:
    """Acyclic event-transition graph with the ExpandedNetwork pricing API."""

    def __init__(
        self,
        problem,
        station_prices,
        *,
        soc_step,
        block_min,
        g_kwh,
        charge_kw,
        reserve_kwh,
        strict_tariff_coverage=False,
        arc_mode="lazy",
    ):
        self.problem = problem
        self.soc_step = float(soc_step)
        self.block_min = int(block_min)
        self.g = float(g_kwh)
        self.charge_kw = float(charge_kw)
        self.reserve = float(reserve_kwh)
        self.strict_tariff_coverage = bool(strict_tariff_coverage)
        if arc_mode not in {"explicit", "lazy"}:
            raise ValueError(f"unsupported event arc mode: {arc_mode}")
        self.arc_mode = arc_mode
        self.station_position = {
            station: index for index, station in enumerate(STATIONS)
        }
        self.prices = normalize_event_station_prices(
            station_prices,
            horizon_min=HORIZON_MIN,
            strict_tariff_coverage=self.strict_tariff_coverage,
        )
        self.grid = [
            round(index * self.soc_step, 9)
            for index in range(int(self.g / self.soc_step) + 1)
        ]
        self.trip_position = {
            trip: index for index, trip in enumerate(problem.trips)
        }
        self.events = _event_times(
            problem, self.prices, self.block_min
        )
        self._window_cache = {}
        self._selected_action_cache = {}
        self._split_arcs()
        self._build_nodes()
        self._build_arcs()

    def __getstate__(self):
        """Return a compact, reconstructable state for durable graph caches."""

        state = dict(self.__dict__)
        # These dictionaries are construction/runtime accelerators, not graph
        # identity.  The window entries can dominate serialized size, and any
        # selected action is cheaply reconstructed only for paths actually
        # returned by pricing.
        state["_window_cache"] = {}
        state["_selected_action_cache"] = {}
        if self.arc_mode == "lazy":
            # NumPy views duplicate their backing arrays when pickled.  Persist
            # the canonical ``array`` buffers only and rebuild zero-copy views
            # after loading.
            for name in (
                "_arc_targets_np", "_arc_costs_np", "_arc_recipes_np",
                "_node_dual_np",
            ):
                state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._window_cache = {}
        self._selected_action_cache = {}
        if self.arc_mode == "lazy":
            self._arc_targets_np = np.frombuffer(
                self._arc_targets, dtype=np.uint32
            )
            self._arc_costs_np = np.frombuffer(
                self._arc_costs, dtype=np.float64
            )
            self._arc_recipes_np = np.frombuffer(
                self._arc_recipes, dtype=np.uint32
            )
            self._node_dual_np = np.full(
                len(self.node_meta), -1, dtype=np.int32
            )
            for (trip, _level), node in self.trip_node.items():
                self._node_dual_np[node] = self.trip_position[trip]

    def _split_arcs(self):
        self.trip_trip = {}
        self.trip_station = {}
        self.station_trip = {}
        self.depot_trip = {}
        self.trip_depot = {}
        self.station_depot = {}
        for source, arcs in self.problem.adjacency.items():
            for successor, travel, energy, kind in arcs:
                value = (float(travel), float(energy))
                if kind == "trip_trip":
                    self.trip_trip.setdefault(source, {})[successor] = value
                elif kind == "trip_station":
                    self.trip_station.setdefault(source, {})[successor] = value
                elif kind == "station_trip":
                    self.station_trip.setdefault(source, {})[successor] = value
                elif kind == "depot_trip":
                    self.depot_trip[successor] = value
                elif kind == "trip_depot":
                    self.trip_depot[source] = value
                elif kind == "station_depot":
                    self.station_depot[source] = value

    def _build_nodes(self):
        self.node_meta = [("source", None, None), ("sink", None, None)]
        self.SINK = 1
        self.trip_node = {}
        order = []
        for trip in self.problem.trips:
            for level, soc in enumerate(self.grid):
                if soc + TOL < self.problem.trip_energy[trip] + self.reserve:
                    continue
                node = len(self.node_meta)
                self.node_meta.append(("trip", trip, level))
                self.trip_node[trip, level] = node
                order.append((float(self.problem.start_min[trip]), node))
        order.sort()
        self.topo = [0] + [node for _time, node in order] + [1]

    def _add(self, source, target, cost, trip, action):
        dual = self.trip_position[trip] if trip is not None else -1
        row = (target, float(cost), dual, action)
        key = (target, dual)
        candidate = (row[1], json.dumps(action, sort_keys=True))
        retained = self._building_arcs.setdefault(source, {})
        current = retained.get(key)
        if current is None or candidate < current[0]:
            retained[key] = (candidate, row)

    def _finalize_source(self, source):
        retained = self._building_arcs.pop(source, {})
        rows = sorted(
            (value[1] for value in retained.values()),
            key=lambda row: (
                row[0], row[1], json.dumps(row[3], sort_keys=True)
            ),
        )
        if self.arc_mode == "explicit":
            self.out[source] = rows
        else:
            start = len(self._arc_targets)
            self._arc_targets.extend(row[0] for row in rows)
            self._arc_costs.extend(row[1] for row in rows)
            self._arc_recipes.extend(
                self._action_recipe(row[3]) for row in rows
            )
            self._arc_slices[source] = (start, len(self._arc_targets))
        self.sink_arcs.extend(
            (
                source,
                cost,
                action if self.arc_mode == "explicit" else None,
            )
            for target, cost, _dual, action in rows
            if target == self.SINK
        )

    def _build_arcs(self):
        self.out = (
            [[] for _node in self.node_meta]
            if self.arc_mode == "explicit" else None
        )
        if self.arc_mode == "lazy":
            self._arc_targets = array("I")
            self._arc_costs = array("d")
            self._arc_recipes = array("I")
            self._arc_slices = [(0, 0) for _node in self.node_meta]
        self._building_arcs = {}
        self.sink_arcs = []
        for target, cost, trip, action in self._source_candidates():
            self._add(0, target, cost, trip, action)
        self._finalize_source(0)
        for (trip, level), source in sorted(self.trip_node.items()):
            soc_exit = self.grid[level] - self.problem.trip_energy[trip]
            depart = float(self.problem.end_min[trip])
            self._direct_arcs(source, trip, soc_exit, depart)
            self._charge_arcs(source, trip, soc_exit, depart)
            self._finalize_source(source)
        del self._building_arcs
        if self.arc_mode == "explicit":
            self.n_arcs = sum(len(arcs) for arcs in self.out)
        else:
            self.n_arcs = len(self._arc_targets)
            self._arc_targets_np = np.frombuffer(
                self._arc_targets, dtype=np.uint32
            )
            self._arc_costs_np = np.frombuffer(
                self._arc_costs, dtype=np.float64
            )
            self._arc_recipes_np = np.frombuffer(
                self._arc_recipes, dtype=np.uint32
            )
            self._node_dual_np = np.full(
                len(self.node_meta), -1, dtype=np.int32
            )
            for (trip, _level), node in self.trip_node.items():
                self._node_dual_np[node] = self.trip_position[trip]

    def _source_candidates(self):
        for trip, (travel, deadhead) in sorted(self.depot_trip.items()):
            if travel > self.problem.start_min[trip] + TOL:
                continue
            level = _floor_level(self.grid, self.soc_step, self.g - deadhead)
            node = self.trip_node.get((trip, level))
            if node is not None:
                yield node, BUS_COST_KX, trip, {
                    "kind": "source", "trip": trip,
                    "travel_min": travel, "deadhead_kwh": deadhead,
                }

    def _action_recipe(self, action):
        if action["kind"] != "charge":
            return 0
        return (
            1
            + self.station_position[action["station"]] * len(self.grid)
            + int(action["exit_level"])
        )

    def _direct_arcs(self, source, trip, soc_exit, depart):
        for target, cost, successor, action in self._direct_candidates(
            trip, soc_exit, depart
        ):
            self._add(source, target, cost, successor, action)

    def _direct_candidates(self, trip, soc_exit, depart):
        options = list(sorted(self.trip_trip.get(trip, {}).items()))
        if trip in self.trip_depot:
            options.append((None, self.trip_depot[trip]))
        for successor, (travel, deadhead) in options:
            deadline = (
                HORIZON_MIN if successor is None
                else self.problem.start_min[successor]
            )
            remaining = soc_exit - deadhead
            level = _floor_level(self.grid, self.soc_step, remaining)
            if (
                depart + travel > deadline + TOL
                or remaining < self.reserve - TOL
                or level < 0
                or (
                    successor is not None
                    and self.grid[level] + TOL
                    < self.problem.trip_energy[successor] + self.reserve
                )
            ):
                continue
            target = (
                self.SINK if successor is None
                else self.trip_node.get((successor, level))
            )
            if target is None:
                continue
            action = {
                "kind": "direct", "from_trip": trip,
                "next_trip": successor, "travel_min": travel,
                "deadhead_kwh": deadhead,
            }
            yield target, 0.0, successor, action

    def _charge_arcs(self, source, trip, soc_exit, depart):
        for target, cost, successor, action in self._charge_candidates(
            trip, soc_exit, depart
        ):
            self._add(source, target, cost, successor, action)

    def _charge_candidates(self, trip, soc_exit, depart):
        for station, (inbound_min, inbound_kwh) in sorted(
            self.trip_station.get(trip, {}).items()
        ):
            arrival = depart + inbound_min
            arrival_soc = soc_exit - inbound_kwh
            entry = _floor_level(self.grid, self.soc_step, arrival_soc)
            if entry < 0 or arrival_soc < self.reserve - TOL:
                continue
            destinations = list(sorted(
                self.station_trip.get(station, {}).items()
            ))
            if station in self.station_depot:
                destinations.append((None, self.station_depot[station]))
            for successor, (outbound_min, outbound_kwh) in destinations:
                deadline = (
                    HORIZON_MIN if successor is None
                    else self.problem.start_min[successor]
                )
                latest = float(deadline) - outbound_min
                for target_level in range(entry + 1, len(self.grid)):
                    energy = self.grid[target_level] - self.grid[entry]
                    cache_key = (
                        station, round(arrival, 9), round(latest, 9),
                        round(energy, 9),
                    )
                    if cache_key not in self._window_cache:
                        self._window_cache[cache_key] = _best_charge_window(
                            station, arrival, latest, energy,
                            event_times=self.events,
                            station_prices=self.prices,
                            charge_kw=self.charge_kw,
                        )
                    selected = self._window_cache[cache_key]
                    if selected is None:
                        break
                    energy_cost, start, end = selected
                    remaining = self.grid[target_level] - outbound_kwh
                    next_level = _floor_level(
                        self.grid, self.soc_step, remaining
                    )
                    if (
                        remaining < self.reserve - TOL
                        or next_level < 0
                        or (
                            successor is not None
                            and self.grid[next_level] + TOL
                            < self.problem.trip_energy[successor] + self.reserve
                        )
                    ):
                        continue
                    target = (
                        self.SINK if successor is None
                        else self.trip_node.get((successor, next_level))
                    )
                    if target is None:
                        continue
                    action = {
                        "kind": "charge", "from_trip": trip,
                        "next_trip": successor, "station": station,
                        "arrival_min": arrival,
                        "deadline_min": latest,
                        "cst": start, "cet": end, "kwh": energy,
                        "inbound_min": inbound_min,
                        "outbound_min": outbound_min,
                        "inbound_kwh": inbound_kwh,
                        "outbound_kwh": outbound_kwh,
                        "entry_level": entry,
                        "exit_level": target_level,
                    }
                    cost = CHARGE_START_COST + energy_cost
                    yield target, cost, successor, action

    def _iter_arcs(self, source):
        if self.arc_mode == "explicit":
            for target, cost, _dual, _action in self.out[source]:
                yield target, cost
            return
        start, end = self._arc_slices[source]
        for offset in range(start, end):
            yield (
                int(self._arc_targets_np[offset]),
                float(self._arc_costs_np[offset]),
            )

    def _edge_action(self, source, target):
        key = (int(source), int(target))
        cached = self._selected_action_cache.get(key)
        if cached is not None:
            return cached
        if self.arc_mode == "explicit":
            for candidate_target, _cost, _dual, action in self.out[source]:
                if candidate_target == target:
                    self._selected_action_cache[key] = action
                    return action
            raise RuntimeError(f"missing explicit event edge {source}->{target}")

        start, end = self._arc_slices[source]
        relative = np.flatnonzero(
            self._arc_targets_np[start:end] == int(target)
        )
        if len(relative) != 1:
            raise RuntimeError(f"missing lazy event edge {source}->{target}")
        recipe = int(self._arc_recipes_np[start + int(relative[0])])
        if source == 0:
            trip = self.node_meta[target][1]
            travel, deadhead = self.depot_trip[trip]
            action = {
                "kind": "source", "trip": trip,
                "travel_min": travel, "deadhead_kwh": deadhead,
            }
        else:
            kind, trip, level = self.node_meta[source]
            if kind != "trip":
                raise RuntimeError(f"invalid event edge source {source}")
            successor = (
                None if target == self.SINK
                else self.node_meta[target][1]
            )
            if recipe == 0:
                travel, deadhead = (
                    self.trip_depot[trip] if successor is None
                    else self.trip_trip[trip][successor]
                )
                action = {
                    "kind": "direct", "from_trip": trip,
                    "next_trip": successor, "travel_min": travel,
                    "deadhead_kwh": deadhead,
                }
            else:
                encoded = recipe - 1
                station_index, target_level = divmod(
                    encoded, len(self.grid)
                )
                station = STATIONS[station_index]
                inbound_min, inbound_kwh = self.trip_station[trip][station]
                outbound_min, outbound_kwh = (
                    self.station_depot[station] if successor is None
                    else self.station_trip[station][successor]
                )
                depart = float(self.problem.end_min[trip])
                arrival = depart + inbound_min
                deadline = (
                    HORIZON_MIN if successor is None
                    else self.problem.start_min[successor]
                )
                latest = float(deadline) - outbound_min
                soc_exit = self.grid[level] - self.problem.trip_energy[trip]
                entry = _floor_level(
                    self.grid, self.soc_step, soc_exit - inbound_kwh
                )
                energy = self.grid[target_level] - self.grid[entry]
                cache_key = (
                    station, round(arrival, 9), round(latest, 9),
                    round(energy, 9),
                )
                selected = self._window_cache.get(cache_key)
                if selected is None:
                    selected = _best_charge_window(
                        station,
                        arrival,
                        latest,
                        energy,
                        event_times=self.events,
                        station_prices=self.prices,
                        charge_kw=self.charge_kw,
                    )
                    if selected is None:
                        raise RuntimeError(
                            "cannot reconstruct event window recipe for "
                            f"{source}->{target}"
                        )
                    self._window_cache[cache_key] = selected
                _energy_cost, charge_start, charge_end = selected
                action = {
                    "kind": "charge", "from_trip": trip,
                    "next_trip": successor, "station": station,
                    "arrival_min": arrival,
                    "deadline_min": latest,
                    "cst": charge_start, "cet": charge_end, "kwh": energy,
                    "inbound_min": inbound_min,
                    "outbound_min": outbound_min,
                    "inbound_kwh": inbound_kwh,
                    "outbound_kwh": outbound_kwh,
                    "entry_level": entry,
                    "exit_level": target_level,
                }
        self._selected_action_cache[key] = action
        return action

    def _record(self, actions):
        trips = [
            action["trip"] for action in actions
            if action["kind"] == "source"
        ] + [
            action["next_trip"] for action in actions
            if action.get("next_trip") is not None
        ]
        route_nodes = [DEPOT, trips[0]]
        charging = {"stations": [], "cst": [], "cet": [], "kwh": []}
        expanded = {"stations": [], "cst": [], "cet": [], "kwh": []}
        continuous_soc = self.g
        continuous_soc -= actions[0]["deadhead_kwh"]
        continuous_soc -= self.problem.trip_energy[trips[0]]
        for action in actions[1:]:
            if action["kind"] == "charge":
                route_nodes.append(action["station"])
                continuous_soc -= action["inbound_kwh"]
                target_soc = self.grid[action["exit_level"]]
                realized_kwh = max(0.0, target_soc - continuous_soc)
                if realized_kwh > action["kwh"] + 1e-6:
                    raise RuntimeError(
                        "event route requires more than grid charge energy"
                    )
                continuous_soc += realized_kwh
                continuous_soc -= action["outbound_kwh"]
                charging["stations"].append(action["station"])
                charging["cst"].append(action["cst"])
                charging["cet"].append(action["cet"])
                charging["kwh"].append(realized_kwh)
                expanded["stations"].append(action["station"])
                expanded["cst"].append(action["cst"])
                expanded["cet"].append(action["cet"])
                expanded["kwh"].append(action["kwh"])
            elif action["kind"] == "direct":
                continuous_soc -= action["deadhead_kwh"]
            if action.get("next_trip") is not None:
                route_nodes.append(action["next_trip"])
                continuous_soc -= self.problem.trip_energy[
                    action["next_trip"]
                ]
            if continuous_soc < self.reserve - 1e-6:
                raise RuntimeError("event route replay drops below reserve")
        route_nodes.append(DEPOT)
        record = {
            "trips": trips,
            "route_nodes": route_nodes,
            "charging_stops": charging,
            "expanded_grid_charging_stops": expanded,
        }
        expanded_record = {
            **record,
            "charging_stops": deepcopy(expanded),
            "expanded_grid_charging_stops": deepcopy(expanded),
            # The authoritative master cost is recomputed from validated
            # tariff blocks below; realization itself only needs the path.
            "cost": 0.0,
        }
        realized, detail = realize_expanded_path(
            self.problem,
            expanded_record,
            g_kwh=self.g,
            charge_kw=self.charge_kw,
            reserve_kwh=self.reserve,
            soc_step=self.soc_step,
            block_min=self.block_min,
            time_model="event",
        )
        if realized is None:
            raise RuntimeError(
                "event route has no continuous realization: "
                f"{detail.get('reason')}"
            )
        record = realized
        costs = realized_costs(
            record,
            detail["mapping"],
            station_prices=self.prices,
        )
        blocks = costs["continuous_realized_charging_blocks"]
        cost = float(costs["recomputed_expanded_grid_cost"])
        reason = validate_injected_route(
            self.problem, record, self.g, self.charge_kw,
            self.reserve, HORIZON_MIN, arrival_grace_min=0.0,
        )
        if reason is not None:
            raise RuntimeError(f"event route failed physical replay: {reason}")
        record.update({
            "cost": cost,
            "expanded_grid_cost": cost,
            "continuous_realized_cost":
                costs["continuous_realized_cost"],
            "continuous_realized_charging_blocks": blocks,
            "continuous_realized_charging_blocks_json_bytes": len(
                json.dumps(blocks, sort_keys=True, separators=(",", ":"))
            ),
            "cost_semantics": "expanded_grid_cost",
            "master_cost_semantics": "expanded_grid_cost",
            "continuous_cost_pricing_certified": False,
            "physical_realization": {
                "status": "valid_event_time_realized",
                "time_model": "event",
                "realization_schema": detail["mapping"]["schema"],
                "realization_mapping_sha256":
                    detail["mapping"]["mapping_sha256"],
                "continuous_realized_charging_blocks_schema":
                    BLOCK_SCHEDULE_SCHEMA,
                "continuous_realized_charging_blocks_sha256":
                    charging_block_schedule_sha256(blocks),
                "continuous_cost_pricing_certified": False,
            },
        })
        return record

    def _walk(self, parent, node, *, terminal_source=None):
        actions = []
        if terminal_source is not None:
            actions.append(self._edge_action(terminal_source, self.SINK))
            node = terminal_source
        while node != 0:
            if self.arc_mode == "explicit":
                previous, action = parent[node]
            else:
                previous = int(parent[node])
                if previous < 0:
                    raise RuntimeError(
                        f"event parent is missing for node {node}"
                    )
                action = self._edge_action(previous, node)
            actions.append(action)
            node = previous
        actions.reverse()
        record = self._record(actions)
        return {
            "trips": record["trips"],
            "charging_stops": record["charging_stops"],
            "route_nodes": record["route_nodes"],
            "charges_started": len(
                record["charging_stops"]["stations"]
            ),
            "_event_record": record,
        }

    def min_reduced_cost_route(
        self,
        alpha,
        *,
        objective="combined-cost",
        route_dual=0.0,
    ):
        if objective not in {
            "combined-cost", "artificial-elimination",
            "fleet-only", "charging-cost",
        }:
            raise ValueError(f"unsupported pricing objective: {objective}")
        dense = [
            float(alpha.get(trip, 0.0)) for trip in self.problem.trips
        ]
        if self.arc_mode == "lazy":
            return self._min_reduced_cost_route_lazy(
                dense, objective=objective, route_dual=route_dual,
            )
        values = [float("inf")] * len(self.node_meta)
        parent = [None] * len(self.node_meta)
        values[0] = 0.0
        for source in self.topo:
            if not math.isfinite(values[source]):
                continue
            for target, cost, dual, action in self.out[source]:
                if objective == "combined-cost" and route_dual == 0.0:
                    candidate = values[source] + cost - (
                        dense[dual] if dual >= 0 else 0.0
                    )
                else:
                    objective_cost = (
                        0.0 if objective == "artificial-elimination"
                        else 1.0
                        if objective == "fleet-only" and source == 0
                        else 0.0 if objective == "fleet-only"
                        else cost - BUS_COST_KX
                        if objective == "charging-cost" and source == 0
                        else cost
                    )
                    candidate = values[source] + objective_cost - (
                        dense[dual] if dual >= 0 else 0.0
                    ) - (route_dual if source == 0 else 0.0)
                if candidate < values[target] - 1e-12:
                    values[target] = candidate
                    parent[target] = (source, action)
        if not math.isfinite(values[self.SINK]):
            return None
        best = self._walk(parent, self.SINK)
        return {
            "rc": values[self.SINK],
            **best,
            "_value": values,
            "_parent": parent,
        }

    def _min_reduced_cost_route_lazy(
        self, dense, *, objective, route_dual
    ):
        values = np.full(len(self.node_meta), np.inf, dtype=np.float64)
        parent = np.full(len(self.node_meta), -1, dtype=np.int32)
        values[0] = 0.0
        dual_by_node = np.zeros(len(self.node_meta), dtype=np.float64)
        trip_nodes = self._node_dual_np >= 0
        dual_by_node[trip_nodes] = np.asarray(
            dense, dtype=np.float64
        )[self._node_dual_np[trip_nodes]]
        for source in self.topo:
            source_value = values[source]
            if not np.isfinite(source_value):
                continue
            start, end = self._arc_slices[source]
            if start == end:
                continue
            targets = self._arc_targets_np[start:end]
            if objective == "combined-cost" and route_dual == 0.0:
                candidates = (
                    source_value
                    + self._arc_costs_np[start:end]
                    - dual_by_node[targets]
                )
            else:
                if objective == "artificial-elimination":
                    objective_costs = 0.0
                elif objective == "fleet-only":
                    objective_costs = 1.0 if source == 0 else 0.0
                elif source == 0:
                    objective_costs = (
                        self._arc_costs_np[start:end] - BUS_COST_KX
                    )
                else:
                    objective_costs = self._arc_costs_np[start:end]
                candidates = (
                    source_value + objective_costs
                    - dual_by_node[targets]
                    - (route_dual if source == 0 else 0.0)
                )
            improved = candidates < values[targets] - 1e-12
            if not np.any(improved):
                continue
            improved_targets = targets[improved]
            values[improved_targets] = candidates[improved]
            parent[improved_targets] = source
        if not np.isfinite(values[self.SINK]):
            return None
        best = self._walk(parent, self.SINK)
        return {
            "rc": float(values[self.SINK]),
            **best,
            "_value": values,
            "_parent": parent,
        }

    def sink_predecessor_route_batch(
        self, alpha, limit=30, *, phase_callback=None,
        objective="combined-cost", route_dual=0.0,
        selection_mode="reduced_cost", diversity_weight=0.5,
        candidate_multiplier=4,
    ):
        """Return the exact best route plus distinct sink-predecessor paths.

        This is a deterministic enrichment heuristic, not k-shortest-path
        enumeration: each sink predecessor contributes only its best prefix.
        """
        if selection_mode not in {"reduced_cost", "complementary"}:
            raise ValueError(
                f"unsupported event column selection: {selection_mode}"
            )
        if not 0.0 <= float(diversity_weight) <= 1.0:
            raise ValueError("diversity_weight must be between zero and one")
        if int(candidate_multiplier) < 1:
            raise ValueError("candidate_multiplier must be positive")
        started = time.perf_counter()
        if objective == "combined-cost" and route_dual == 0.0:
            best = self.min_reduced_cost_route(alpha)
        else:
            best = self.min_reduced_cost_route(
                alpha, objective=objective, route_dual=route_dual,
            )
        if phase_callback is not None:
            phase_callback(
                "pricing_shortest_path",
                time.perf_counter() - started,
                {"path_found": best is not None, "time_model": "event"},
            )
        if best is None:
            return []
        values = best.pop("_value")
        parent = best.pop("_parent")
        candidates = []
        for source, cost, action in self.sink_arcs:
            if math.isfinite(values[source]):
                sink_cost = (
                    cost
                    if objective in {"combined-cost", "charging-cost"}
                    else 0.0
                )
                candidates.append((
                    values[source] + sink_cost, source, action
                ))
        candidates.sort(key=lambda row: (
            row[0], row[1], json.dumps(row[2], sort_keys=True)
        ))
        routes = [best]
        seen = {frozenset(best["trips"])}
        if selection_mode == "reduced_cost":
            scan_limit = limit
        else:
            scan_limit = max(limit, int(candidate_multiplier) * limit)
        eligible = []
        for reduced_cost, source, _action in candidates:
            if reduced_cost >= -1e-9:
                break
            route = self._walk(
                parent, self.SINK, terminal_source=source
            )
            key = frozenset(route["trips"])
            if key in seen:
                continue
            seen.add(key)
            eligible.append({"rc": reduced_cost, **route})
            if (
                selection_mode == "reduced_cost"
                and len(routes) + len(eligible) >= limit
            ):
                break
            if selection_mode == "complementary" and len(eligible) >= scan_limit:
                break

        novelty_trace = []
        if selection_mode == "reduced_cost":
            routes.extend(eligible[: max(0, limit - 1)])
            selected_sets = [frozenset(routes[0]["trips"])]
            for route in routes[1:]:
                incidence = frozenset(route["trips"])
                novelty_trace.append(min(
                    1.0 - len(incidence & selected) / len(incidence | selected)
                    for selected in selected_sets
                ))
                selected_sets.append(incidence)
        else:
            # The exact best route is never displaced.  Remaining slots trade
            # reduced-cost quality against trip-incidence novelty.  This is a
            # column-selection heuristic only: every retained route remains
            # negative under the unmodified master duals, and the exact LP
            # certificate still comes solely from ``best`` above.
            remaining = list(eligible)
            selected_sets = [frozenset(best["trips"])]
            best_rc = float(best["rc"])
            worst_rc = max(
                [float(route["rc"]) for route in remaining] or [best_rc]
            )
            quality_span = max(worst_rc - best_rc, 1e-12)
            while remaining and len(routes) < limit:
                scored = []
                for index, route in enumerate(remaining):
                    incidence = frozenset(route["trips"])
                    novelty = min(
                        1.0 - len(incidence & selected) / len(incidence | selected)
                        for selected in selected_sets
                    )
                    quality = (worst_rc - float(route["rc"])) / quality_span
                    score = (
                        (1.0 - float(diversity_weight)) * quality
                        + float(diversity_weight) * novelty
                    )
                    scored.append((
                        -score,
                        float(route["rc"]),
                        tuple(route["trips"]),
                        index,
                    ))
                _neg_score, _rc, _trips, chosen = min(scored)
                route = remaining.pop(chosen)
                routes.append(route)
                novelty_trace.append(
                    min(
                        1.0
                        - len(frozenset(route["trips"]) & selected)
                        / len(frozenset(route["trips"]) | selected)
                        for selected in selected_sets
                    )
                )
                selected_sets.append(frozenset(route["trips"]))
        if phase_callback is not None:
            phase_callback(
                "pricing_extra_columns",
                time.perf_counter() - started,
                {
                    "sink_candidates": len(candidates),
                    "eligible_distinct_routes": len(eligible),
                    "returned_routes": len(routes),
                    "selection_mode": selection_mode,
                    "diversity_weight": float(diversity_weight),
                    "mean_incremental_novelty": (
                        sum(novelty_trace) / len(novelty_trace)
                        if novelty_trace else None
                    ),
                    "time_model": "event",
                },
            )
        return routes

    def fixed_sequence_record(self, trips):
        """Return the cheapest event route for one fixed trip sequence."""

        trips = tuple(trips)
        if not trips or any(trip not in self.trip_position for trip in trips):
            return None
        frontier = {}
        for target, cost in self._iter_arcs(0):
            if self.node_meta[target][1] == trips[0]:
                frontier[target] = (cost, [(0, target)])
        for successor in (*trips[1:], None):
            following = {}
            for source, (base_cost, edges) in frontier.items():
                for target, cost in self._iter_arcs(source):
                    matches = (
                        target == self.SINK if successor is None
                        else self.node_meta[target][0] == "trip"
                        and self.node_meta[target][1] == successor
                    )
                    if not matches:
                        continue
                    candidate = (
                        base_cost + cost,
                        edges + [(source, target)],
                    )
                    current = following.get(target)
                    if current is None or (
                        candidate[0],
                        candidate[1],
                    ) < (
                        current[0],
                        current[1],
                    ):
                        following[target] = candidate
            frontier = following
            if not frontier:
                return None
        _cost, edges = frontier[self.SINK]
        actions = [
            self._edge_action(source, target)
            for source, target in edges
        ]
        return self._record(actions)

    def metrics(self):
        return {
            "time_model": "event",
            "dag_nodes": len(self.node_meta),
            "dag_arcs": self.n_arcs,
            "arc_mode": self.arc_mode,
            "materialized_python_arc_objects": (
                self.n_arcs if self.arc_mode == "explicit" else 0
            ),
            "packed_arc_bytes": (
                0 if self.arc_mode == "explicit"
                else (
                    len(self._arc_targets) * self._arc_targets.itemsize
                    + len(self._arc_costs) * self._arc_costs.itemsize
                    + len(self._arc_recipes) * self._arc_recipes.itemsize
                )
            ),
            "station_event_times": {
                station: len(times)
                for station, times in self.events.items()
            },
            "event_lattice_sha256": hashlib.sha256(json.dumps(
                {
                    station: list(times)
                    for station, times in sorted(self.events.items())
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()).hexdigest(),
        }

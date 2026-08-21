"""Event-time trip/SOC DAG for exact reduced-cost route pricing."""

from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy

from audit_giro_known_columns import DEPOT, HORIZON_MIN, STATIONS
from config import BUS_COST_KX, CHARGE_START_COST, charge_cost_premium
from expanded_path_realization import (
    BLOCK_SCHEDULE_SCHEMA,
    blocks_from_continuous_stops,
    charging_block_schedule_sha256,
    validate_continuous_charging_blocks,
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
    ):
        self.problem = problem
        self.prices = deepcopy(station_prices)
        self.soc_step = float(soc_step)
        self.block_min = int(block_min)
        self.g = float(g_kwh)
        self.charge_kw = float(charge_kw)
        self.reserve = float(reserve_kwh)
        self.strict_tariff_coverage = bool(strict_tariff_coverage)
        required_hour = int(math.ceil(HORIZON_MIN / 60.0)) - 1
        for curve in self.prices.values():
            if self.strict_tariff_coverage and required_hour not in curve:
                raise ValueError("event tariff coverage is incomplete")
            if curve:
                last = curve[max(curve)]
                for hour in range(required_hour + 1):
                    curve.setdefault(hour, last)
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
        self._split_arcs()
        self._build_nodes()
        self._build_arcs()

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
        self.out[source].append((target, float(cost), dual, action))

    def _build_arcs(self):
        self.out = [[] for _node in self.node_meta]
        self.sink_arcs = []
        for trip, (travel, deadhead) in sorted(self.depot_trip.items()):
            if travel > self.problem.start_min[trip] + TOL:
                continue
            level = _floor_level(self.grid, self.soc_step, self.g - deadhead)
            node = self.trip_node.get((trip, level))
            if node is not None:
                self._add(0, node, BUS_COST_KX, trip, {
                    "kind": "source", "trip": trip,
                    "travel_min": travel, "deadhead_kwh": deadhead,
                })
        for (trip, level), source in sorted(self.trip_node.items()):
            soc_exit = self.grid[level] - self.problem.trip_energy[trip]
            depart = float(self.problem.end_min[trip])
            self._direct_arcs(source, trip, soc_exit, depart)
            self._charge_arcs(source, trip, soc_exit, depart)
        for arcs in self.out:
            arcs.sort(key=lambda row: (
                row[0], row[1], json.dumps(row[3], sort_keys=True)
            ))
        self.n_arcs = sum(len(arcs) for arcs in self.out)

    def _direct_arcs(self, source, trip, soc_exit, depart):
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
            self._add(source, target, 0.0, successor, action)
            if successor is None:
                self.sink_arcs.append((source, 0.0, action))

    def _charge_arcs(self, source, trip, soc_exit, depart):
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
                    self._add(source, target, cost, successor, action)
                    if successor is None:
                        self.sink_arcs.append((source, cost, action))

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
        }
        blocks = blocks_from_continuous_stops(
            expanded_record,
            station_prices=self.prices,
            charge_kw=self.charge_kw,
        )
        remaining_by_stop = list(charging["kwh"])
        for block in blocks:
            stop = int(block["stop_index"])
            realized = min(
                remaining_by_stop[stop],
                (block["end_min"] - block["start_min"])
                * self.charge_kw / 60.0,
            )
            block["realized_kwh"] = realized
            remaining_by_stop[stop] -= realized
        if any(value > 1e-6 for value in remaining_by_stop):
            raise RuntimeError("event route block allocation is incomplete")
        validation = validate_continuous_charging_blocks(
            record,
            blocks,
            station_prices=self.prices,
            charge_kw=self.charge_kw,
        )
        cost = float(validation["recomputed_expanded_grid_cost"])
        reason = validate_injected_route(
            self.problem, record, self.g, self.charge_kw,
            self.reserve, HORIZON_MIN,
        )
        if reason is not None:
            raise RuntimeError(f"event route failed physical replay: {reason}")
        record.update({
            "cost": cost,
            "expanded_grid_cost": cost,
            "continuous_realized_cost":
                validation["continuous_realized_cost"],
            "continuous_realized_charging_blocks": blocks,
            "continuous_realized_charging_blocks_json_bytes": len(
                json.dumps(blocks, sort_keys=True, separators=(",", ":"))
            ),
            "cost_semantics": "expanded_grid_cost",
            "master_cost_semantics": "expanded_grid_cost",
            "continuous_cost_pricing_certified": False,
            "physical_realization": {
                "status": "valid_event_time_mapped",
                "time_model": "event",
                "continuous_realized_charging_blocks_schema":
                    BLOCK_SCHEDULE_SCHEMA,
                "continuous_realized_charging_blocks_sha256":
                    charging_block_schedule_sha256(blocks),
                "continuous_cost_pricing_certified": False,
            },
        })
        return record

    def _walk(self, parent, node):
        actions = []
        while node != 0:
            previous, action = parent[node]
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

    def min_reduced_cost_route(self, alpha):
        dense = [
            float(alpha.get(trip, 0.0)) for trip in self.problem.trips
        ]
        values = [float("inf")] * len(self.node_meta)
        parent = [None] * len(self.node_meta)
        values[0] = 0.0
        for source in self.topo:
            if not math.isfinite(values[source]):
                continue
            for target, cost, dual, action in self.out[source]:
                candidate = values[source] + cost - (
                    dense[dual] if dual >= 0 else 0.0
                )
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

    def k_best_routes(self, alpha, k=30, *, phase_callback=None):
        started = time.perf_counter()
        best = self.min_reduced_cost_route(alpha)
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
                candidates.append((values[source] + cost, source, action))
        candidates.sort(key=lambda row: (
            row[0], row[1], json.dumps(row[2], sort_keys=True)
        ))
        routes = [best]
        seen = {frozenset(best["trips"])}
        for reduced_cost, source, action in candidates:
            if len(routes) >= k or reduced_cost >= -1e-9:
                break
            terminal_parent = list(parent)
            terminal_parent[self.SINK] = (source, action)
            route = self._walk(terminal_parent, self.SINK)
            key = frozenset(route["trips"])
            if key in seen:
                continue
            seen.add(key)
            routes.append({"rc": reduced_cost, **route})
        if phase_callback is not None:
            phase_callback(
                "pricing_extra_columns",
                time.perf_counter() - started,
                {
                    "sink_candidates": len(candidates),
                    "returned_routes": len(routes),
                    "time_model": "event",
                },
            )
        return routes

    def metrics(self):
        return {
            "time_model": "event",
            "dag_nodes": len(self.node_meta),
            "dag_arcs": self.n_arcs,
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

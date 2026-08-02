"""Model-derived minimum-path-cover initialization for column generation.

This initializer is deliberately independent of historical vehicle duties.  It
uses only the active trip set and pricing graph to construct a minimum path
cover, then realizes every path under the pricing model's time, battery, and
charging resources.  Compatibility can include both direct ``trip_trip`` arcs
and conservative, positive-charge ``trip_station``/``station_trip`` bridges.

The path-cover problem ignores route-history battery resources, so a minimum
time-feasible cover is not guaranteed to admit a charging realization on every
instance. ``build_matching_initial_routes`` therefore tries alternate maximum
matchings, then deterministically cuts infeasible relaxed paths into the fewest
contiguous resource-feasible routes. It raises only when even a singleton trip
cannot be realized; it never returns an invalid warm start.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from pricing_dp_og import _successor_boundary_soc_levels


Arc = tuple[Hashable, float, float, str]
Adjacency = Mapping[Hashable, Sequence[Arc]]
ChargingCost = Callable[[Hashable, float, float], float]


class MatchingInitializationError(ValueError):
    """Base error for invalid or infeasible matching initialization."""


class RouteRealizationError(MatchingInitializationError):
    """Raised when a time-path-cover path cannot satisfy route resources."""

    def __init__(
        self,
        message: str,
        *,
        trip_order: Sequence[int],
        failed_transition: tuple[Hashable, Hashable] | None = None,
    ) -> None:
        super().__init__(message)
        self.trip_order = tuple(trip_order)
        self.failed_transition = failed_transition


@dataclass(frozen=True)
class _RouteState:
    soc: float
    charges_used: int
    proxy_cost: float
    route: tuple[Hashable, ...]
    charging_stops: tuple[tuple[Hashable, float, float, float], ...]
    deadhead_kwh: float


def peak_trip_concurrency(
    trips: Iterable[int],
    trip_start_min: Mapping[int, float],
    trip_end_min: Mapping[int, float],
) -> int:
    """Return peak simultaneous trip service using half-open intervals.

    A trip occupies ``[start, end)``.  Consequently, a trip ending at minute
    ``t`` does not overlap a different trip starting at minute ``t``.
    Zero-duration trips occupy no service time.
    """

    active = tuple(trips)
    if len(set(active)) != len(active):
        raise MatchingInitializationError("Active trips must be unique")
    events: dict[float, list[int]] = {}
    for trip in active:
        if trip not in trip_start_min or trip not in trip_end_min:
            raise MatchingInitializationError(
                f"Missing service interval for active trip {trip}"
            )
        start = float(trip_start_min[trip])
        end = float(trip_end_min[trip])
        if end < start - 1e-9:
            raise MatchingInitializationError(
                f"Trip {trip} ends before it starts: {start} -> {end}"
            )
        if end <= start + 1e-9:
            continue
        events.setdefault(start, [0, 0])[1] += 1
        events.setdefault(end, [0, 0])[0] += 1

    concurrent = 0
    peak = 0
    for time_min in sorted(events):
        endings, starts = events[time_min]
        concurrent -= endings
        if concurrent < 0:
            raise MatchingInitializationError(
                "Invalid service intervals produced negative concurrency"
            )
        concurrent += starts
        peak = max(peak, concurrent)
    return peak


def _bridge_supports_positive_charge(
    *,
    previous: int,
    following: int,
    station: Hashable,
    arcs: Mapping[tuple[Hashable, Hashable], Arc],
    trip_start_min: Mapping[int, float],
    trip_end_min: Mapping[int, float],
    trip_energy_kwh: Mapping[int, float],
    battery_capacity_kwh: float,
    charge_rate_kw: float,
    soc_charge_levels: Sequence[float],
    horizon_min: float,
    max_station_to_trip_wait_min: float,
    successor_boundary_soc_target: bool,
    station_waiting_unrestricted: bool,
) -> bool:
    into_station = arcs.get((previous, station))
    out_of_station = arcs.get((station, following))
    if (
        into_station is None
        or out_of_station is None
        or into_station[3] != "trip_station"
        or out_of_station[3] != "station_trip"
    ):
        return False

    _right, into_minutes, into_deadhead, _arc_type = into_station
    _right, out_minutes, out_deadhead, _arc_type = out_of_station
    arrival_time = float(trip_end_min[previous]) + into_minutes
    latest_departure = min(
        float(trip_start_min[following]) - out_minutes,
        horizon_min,
    )
    available_minutes = latest_departure - arrival_time
    if available_minutes <= 1e-9:
        return False

    # Even a vehicle entering the previous trip full cannot reach this station
    # when the upper bound below is negative.
    maximum_arrival_soc = (
        battery_capacity_kwh
        - float(trip_energy_kwh[previous])
        - into_deadhead
    )
    energy_after_station = out_deadhead + float(trip_energy_kwh[following])
    if maximum_arrival_soc < -1e-6 or energy_after_station > battery_capacity_kwh + 1e-6:
        return False

    minimum_charge_minutes = 0.0
    if not station_waiting_unrestricted:
        minimum_charge_minutes = max(
            0.0,
            float(trip_start_min[following])
            - max_station_to_trip_wait_min
            - arrival_time,
        )
    if minimum_charge_minutes > available_minutes + 1e-6:
        return False

    minimum_time_energy = minimum_charge_minutes * charge_rate_kw / 60.0
    maximum_time_energy = available_minutes * charge_rate_kw / 60.0
    positive_epsilon = 1e-7

    # For a fixed target L, arrival SOC is L - charged energy.  Intersect the
    # allowed charging-time interval with arrival SOC in [0, maximum].
    for target_soc in soc_charge_levels:
        if target_soc < energy_after_station - 1e-6:
            continue
        minimum_energy = max(
            positive_epsilon,
            minimum_time_energy,
            target_soc - maximum_arrival_soc,
        )
        maximum_energy = min(maximum_time_energy, target_soc)
        if minimum_energy <= maximum_energy + 1e-9:
            return True

    if successor_boundary_soc_target:
        # A boundary target that does not hit capacity charges through the full
        # available window.  Check whether some physically possible arrival SOC
        # makes that target sufficient for the next trip.
        boundary_energy = maximum_time_energy
        lower_arrival = max(0.0, energy_after_station - boundary_energy)
        upper_arrival = min(
            maximum_arrival_soc,
            battery_capacity_kwh - boundary_energy,
        )
        if (
            boundary_energy >= max(positive_epsilon, minimum_time_energy) - 1e-9
            and lower_arrival <= upper_arrival + 1e-9
        ):
            return True

    return False


def _trip_successors(
    trips: Sequence[int],
    adjacency: Adjacency,
    *,
    stations: Sequence[Hashable] = (),
    trip_start_min: Mapping[int, float] | None = None,
    trip_end_min: Mapping[int, float] | None = None,
    trip_energy_kwh: Mapping[int, float] | None = None,
    battery_capacity_kwh: float = 300.0,
    charge_rate_kw: float = 300.0,
    soc_charge_levels: Sequence[float] = (),
    horizon_min: float = 1560.0,
    max_daily_recharges: int = 15,
    max_station_to_trip_wait_min: float = 220.0,
    successor_boundary_soc_target: bool = False,
    station_waiting_unrestricted: bool = False,
    direct_only: bool = False,
) -> dict[int, tuple[int, ...]]:
    active = set(trips)
    arcs = _arc_lookup(adjacency)
    successors: dict[int, tuple[int, ...]] = {}
    for trip in trips:
        found = {
            successor
            for successor, _minutes, _energy, arc_type in adjacency.get(trip, ())
            if arc_type == "trip_trip" and successor in active
        }
        if not direct_only and stations and max_daily_recharges > 0:
            if (
                trip_start_min is None
                or trip_end_min is None
                or trip_energy_kwh is None
            ):
                raise MatchingInitializationError(
                    "Station-bridge compatibility requires trip start/end/energy data"
                )
            for following in trips:
                if following == trip or following in found:
                    continue
                if any(
                    _bridge_supports_positive_charge(
                        previous=trip,
                        following=following,
                        station=station,
                        arcs=arcs,
                        trip_start_min=trip_start_min,
                        trip_end_min=trip_end_min,
                        trip_energy_kwh=trip_energy_kwh,
                        battery_capacity_kwh=battery_capacity_kwh,
                        charge_rate_kw=charge_rate_kw,
                        soc_charge_levels=soc_charge_levels,
                        horizon_min=horizon_min,
                        max_station_to_trip_wait_min=max_station_to_trip_wait_min,
                        successor_boundary_soc_target=successor_boundary_soc_target,
                        station_waiting_unrestricted=station_waiting_unrestricted,
                    )
                    for station in stations
                ):
                    found.add(following)
        successors[trip] = tuple(
            successor for successor in trips if successor in found
        )
    return successors


def _assert_dag(
    trips: Sequence[int],
    successors: Mapping[int, Sequence[int]],
) -> None:
    indegree = {trip: 0 for trip in trips}
    for following in successors.values():
        for trip in following:
            indegree[trip] += 1

    queue = deque(trip for trip in trips if indegree[trip] == 0)
    visited = 0
    while queue:
        trip = queue.popleft()
        visited += 1
        for following in successors[trip]:
            indegree[following] -= 1
            if indegree[following] == 0:
                queue.append(following)

    if visited != len(trips):
        raise MatchingInitializationError(
            "The active trip compatibility graph is not acyclic"
        )


def _matching_orders(
    trips: Sequence[int],
    *,
    attempt: int,
    order_seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    original = tuple(trips)
    if attempt == 0:
        return original, original, "input"
    if attempt == 1:
        return original, tuple(reversed(original)), "columns_reversed"
    if attempt == 2:
        return tuple(reversed(original)), original, "rows_reversed"
    if attempt == 3:
        return (
            tuple(reversed(original)),
            tuple(reversed(original)),
            "rows_and_columns_reversed",
        )

    def digest_order(role: str) -> tuple[int, ...]:
        def key(trip: int) -> bytes:
            payload = f"{order_seed}|{attempt}|{role}|{trip!r}".encode("utf-8")
            return hashlib.blake2b(payload, digest_size=16).digest()

        return tuple(sorted(original, key=key))

    return (
        digest_order("row"),
        digest_order("column"),
        f"blake2b_seeded_attempt_{attempt}",
    )


def _path_cover_from_successors(
    ordered_trips: Sequence[int],
    successors: Mapping[int, Sequence[int]],
    *,
    trip_start_min: Mapping[int, float] | None,
    matching_attempt: int,
    matching_order_seed: int,
) -> tuple[list[tuple[int, ...]], frozenset[tuple[int, int]], str]:
    row_order, column_order, order_name = _matching_orders(
        ordered_trips,
        attempt=matching_attempt,
        order_seed=matching_order_seed,
    )
    row_index = {trip: position for position, trip in enumerate(row_order)}
    column_index = {
        trip: position for position, trip in enumerate(column_order)
    }
    rows: list[int] = []
    columns: list[int] = []
    for trip, following in successors.items():
        rows.extend([row_index[trip]] * len(following))
        columns.extend(column_index[other] for other in following)

    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)),
        shape=(len(ordered_trips), len(ordered_trips)),
    )
    row_to_column = maximum_bipartite_matching(graph, perm_type="column")

    matched_successor = {
        row_order[row]: column_order[int(column)]
        for row, column in enumerate(row_to_column)
        if column >= 0
    }
    matching_signature = frozenset(matched_successor.items())
    matched_predecessors = set(matched_successor.values())
    starts = [trip for trip in ordered_trips if trip not in matched_predecessors]

    paths: list[tuple[int, ...]] = []
    covered: set[int] = set()
    for start in starts:
        path: list[int] = []
        current = start
        while current not in covered:
            path.append(current)
            covered.add(current)
            if current not in matched_successor:
                break
            current = matched_successor[current]
        paths.append(tuple(path))

    if covered != set(ordered_trips):
        missing = [trip for trip in ordered_trips if trip not in covered]
        raise MatchingInitializationError(
            f"Matching reconstruction failed to cover active trips: {missing[:10]}"
        )

    if trip_start_min is not None:
        original_index = {
            trip: position for position, trip in enumerate(ordered_trips)
        }
        paths.sort(
            key=lambda path: (
                trip_start_min[path[0]],
                original_index[path[0]],
            )
        )
    else:
        original_index = {
            trip: position for position, trip in enumerate(ordered_trips)
        }
        paths.sort(key=lambda path: original_index[path[0]])
    return paths, matching_signature, order_name


def minimum_trip_path_cover(
    trips: Iterable[int],
    adjacency: Adjacency,
    *,
    stations: Iterable[Hashable] = (),
    trip_start_min: Mapping[int, float] | None = None,
    trip_end_min: Mapping[int, float] | None = None,
    trip_energy_kwh: Mapping[int, float] | None = None,
    battery_capacity_kwh: float = 300.0,
    charge_rate_kw: float = 300.0,
    soc_charge_levels: Iterable[float] | None = None,
    horizon_min: float = 1560.0,
    max_daily_recharges: int = 15,
    max_station_to_trip_wait_min: float = 220.0,
    successor_boundary_soc_target: bool = False,
    station_waiting_unrestricted: bool = False,
    direct_only: bool = False,
    matching_attempt: int = 0,
    matching_order_seed: int = 0,
) -> list[tuple[int, ...]]:
    """Return a minimum vertex-disjoint path cover of model connectivity.

    By default, compatibility includes direct trip arcs and conservative
    positive-charge station bridges.  Pass ``direct_only=True`` for the older
    diagnostic graph.  For a DAG, the minimum number of paths equals
    ``|trips| - |maximum matching|`` in its standard bipartite split graph.

    ``matching_attempt`` selects a deterministic alternate vertex ordering;
    it is mainly useful for diagnostics.  The high-level initializer retries
    alternate maximum matchings automatically when resource realization fails.
    """

    ordered_trips = tuple(trips)
    station_nodes = tuple(dict.fromkeys(stations))
    if len(set(ordered_trips)) != len(ordered_trips):
        raise MatchingInitializationError("Active trips must be unique")
    if matching_attempt < 0:
        raise MatchingInitializationError("matching_attempt cannot be negative")
    if not ordered_trips:
        return []

    if soc_charge_levels is None:
        soc_charge_levels = (
            battery_capacity_kwh * index / 10.0 for index in range(1, 11)
        )
    levels = _dedupe_soc_levels(soc_charge_levels, battery_capacity_kwh)
    successors = _trip_successors(
        ordered_trips,
        adjacency,
        stations=station_nodes,
        trip_start_min=trip_start_min,
        trip_end_min=trip_end_min,
        trip_energy_kwh=trip_energy_kwh,
        battery_capacity_kwh=battery_capacity_kwh,
        charge_rate_kw=charge_rate_kw,
        soc_charge_levels=levels,
        horizon_min=horizon_min,
        max_daily_recharges=max_daily_recharges,
        max_station_to_trip_wait_min=max_station_to_trip_wait_min,
        successor_boundary_soc_target=successor_boundary_soc_target,
        station_waiting_unrestricted=station_waiting_unrestricted,
        direct_only=direct_only,
    )
    _assert_dag(ordered_trips, successors)

    if trip_start_min is not None:
        missing = [trip for trip in ordered_trips if trip not in trip_start_min]
        if missing:
            raise MatchingInitializationError(
                f"Missing trip start times for active trips: {missing[:10]}"
            )
        for trip, following in successors.items():
            backwards = [
                other
                for other in following
                if trip_start_min[other] < trip_start_min[trip] - 1e-6
            ]
            if backwards:
                raise MatchingInitializationError(
                    "Compatibility arc moves backwards in time: "
                    f"{trip} -> {backwards[0]}"
                )

    paths, _signature, _order_name = _path_cover_from_successors(
        ordered_trips,
        successors,
        trip_start_min=trip_start_min,
        matching_attempt=matching_attempt,
        matching_order_seed=matching_order_seed,
    )
    return paths


def _arc_lookup(adjacency: Adjacency) -> dict[tuple[Hashable, Hashable], Arc]:
    lookup: dict[tuple[Hashable, Hashable], Arc] = {}
    for left, outgoing in adjacency.items():
        for arc in outgoing:
            if len(arc) != 4:
                raise MatchingInitializationError(
                    f"Malformed adjacency arc from {left!r}: {arc!r}"
                )
            right, minutes, energy, arc_type = arc
            if float(minutes) < -1e-9 or float(energy) < -1e-9:
                raise MatchingInitializationError(
                    f"Negative travel resource on arc {left!r} -> {right!r}"
                )
            key = (left, right)
            normalized = (right, float(minutes), float(energy), str(arc_type))
            if key in lookup and lookup[key] != normalized:
                raise MatchingInitializationError(
                    f"Conflicting duplicate arcs for {left!r} -> {right!r}"
                )
            lookup[key] = normalized
    return lookup


def _dedupe_soc_levels(
    levels: Iterable[float],
    battery_capacity_kwh: float,
) -> tuple[float, ...]:
    valid = {
        min(float(level), battery_capacity_kwh)
        for level in levels
        if float(level) > 1e-9 and float(level) <= battery_capacity_kwh + 1e-6
    }
    return tuple(sorted(valid))


def _pareto(states: Iterable[_RouteState]) -> list[_RouteState]:
    output: list[_RouteState] = []
    ordered = sorted(
        states,
        key=lambda state: (
            state.proxy_cost,
            state.charges_used,
            -state.soc,
            state.deadhead_kwh,
        ),
    )
    for state in ordered:
        if any(
            other.soc >= state.soc - 1e-7
            and other.charges_used <= state.charges_used
            and other.proxy_cost <= state.proxy_cost + 1e-7
            for other in output
        ):
            continue
        output = [
            other
            for other in output
            if not (
                state.soc >= other.soc - 1e-7
                and state.charges_used <= other.charges_used
                and state.proxy_cost <= other.proxy_cost + 1e-7
            )
        ]
        output.append(state)
    return output


def realize_fixed_trip_path(
    trip_order: Iterable[int],
    *,
    adjacency: Adjacency,
    depot: Hashable,
    stations: Iterable[Hashable],
    trip_start_min: Mapping[int, float],
    trip_end_min: Mapping[int, float],
    trip_energy_kwh: Mapping[int, float],
    battery_capacity_kwh: float = 300.0,
    charge_rate_kw: float = 300.0,
    soc_charge_levels: Iterable[float] | None = None,
    horizon_min: float = 1560.0,
    max_daily_recharges: int = 15,
    max_station_to_trip_wait_min: float = 220.0,
    successor_boundary_soc_target: bool = False,
    max_successor_charge_targets: int = 64,
    station_waiting_unrestricted: bool = False,
    charge_start_cost: float = 0.0,
    charging_cost: ChargingCost | None = None,
    deadhead_cost_per_kwh: float = 0.0,
) -> dict[str, Any]:
    """Realize one fixed trip order as an ``R_truck``-compatible route.

    Charging starts immediately on station arrival and targets the supplied
    absolute SOC grid, matching the current pricing resource extension.  When
    requested, the same capped station-successor boundary targets as the DP are
    added.
    ``station_waiting_unrestricted`` permits implicit idle time after charging;
    it does not delay the start of charging.
    """

    trips = tuple(trip_order)
    station_nodes = tuple(dict.fromkeys(stations))
    if not trips:
        raise RouteRealizationError("Trip path is empty", trip_order=trips)
    if battery_capacity_kwh <= 0 or charge_rate_kw <= 0:
        raise MatchingInitializationError(
            "Battery capacity and charge rate must be positive"
        )
    if max_daily_recharges < 0:
        raise MatchingInitializationError("max_daily_recharges cannot be negative")
    if max_station_to_trip_wait_min < 0:
        raise MatchingInitializationError(
            "max_station_to_trip_wait_min cannot be negative"
        )
    if max_successor_charge_targets <= 0:
        raise MatchingInitializationError(
            "max_successor_charge_targets must be positive"
        )

    for name, values in (
        ("start time", trip_start_min),
        ("end time", trip_end_min),
        ("energy", trip_energy_kwh),
    ):
        missing = [trip for trip in trips if trip not in values]
        if missing:
            raise MatchingInitializationError(
                f"Missing trip {name} values: {missing[:10]}"
            )

    if soc_charge_levels is None:
        soc_charge_levels = (
            battery_capacity_kwh * index / 10.0 for index in range(1, 11)
        )
    base_levels = _dedupe_soc_levels(soc_charge_levels, battery_capacity_kwh)
    if not base_levels:
        raise MatchingInitializationError("SOC charge grid has no valid levels")

    if charging_cost is None:
        charging_cost = lambda _station, _start, energy: float(energy)

    arcs = _arc_lookup(adjacency)
    station_successor_deadlines = {}
    if successor_boundary_soc_target:
        for station in station_nodes:
            station_successor_deadlines[station] = sorted({
                float(trip_start_min[successor]) - float(travel_min)
                for successor, travel_min, _energy, arc_type in adjacency.get(
                    station,
                    (),
                )
                if arc_type == "station_trip" and successor in trip_start_min
            })
    first = trips[0]
    first_arc = arcs.get((depot, first))
    if first_arc is None or first_arc[3] != "depot_trip":
        raise RouteRealizationError(
            f"No depot-to-trip arc for first trip {first}",
            trip_order=trips,
            failed_transition=(depot, first),
        )

    _right, first_minutes, first_deadhead, _arc_type = first_arc
    initial_soc = (
        battery_capacity_kwh - first_deadhead - float(trip_energy_kwh[first])
    )
    if (
        first_minutes > float(trip_start_min[first]) + 1e-6
        or initial_soc < -1e-6
    ):
        raise RouteRealizationError(
            f"First trip {first} is not reachable from the depot",
            trip_order=trips,
            failed_transition=(depot, first),
        )

    states = [
        _RouteState(
            soc=initial_soc,
            charges_used=0,
            proxy_cost=first_deadhead * deadhead_cost_per_kwh,
            route=(depot, first),
            charging_stops=(),
            deadhead_kwh=first_deadhead,
        )
    ]

    for previous, following in zip(trips, trips[1:]):
        candidates: list[_RouteState] = []
        direct = arcs.get((previous, following))
        if direct is not None and direct[3] == "trip_trip":
            _right, minutes, deadhead, _arc_type = direct
            for state in states:
                next_soc = (
                    state.soc - deadhead - float(trip_energy_kwh[following])
                )
                if (
                    float(trip_end_min[previous]) + minutes
                    <= float(trip_start_min[following]) + 1e-6
                    and next_soc >= -1e-6
                ):
                    candidates.append(
                        _RouteState(
                            soc=next_soc,
                            charges_used=state.charges_used,
                            proxy_cost=(
                                state.proxy_cost
                                + deadhead * deadhead_cost_per_kwh
                            ),
                            route=state.route + (following,),
                            charging_stops=state.charging_stops,
                            deadhead_kwh=state.deadhead_kwh + deadhead,
                        )
                    )

        for station in station_nodes:
            into_station = arcs.get((previous, station))
            out_of_station = arcs.get((station, following))
            if (
                into_station is None
                or out_of_station is None
                or into_station[3] != "trip_station"
                or out_of_station[3] != "station_trip"
            ):
                continue

            _right, into_minutes, into_deadhead, _arc_type = into_station
            _right, out_minutes, out_deadhead, _arc_type = out_of_station
            arrival_time = float(trip_end_min[previous]) + into_minutes
            latest_departure = float(trip_start_min[following]) - out_minutes

            for state in states:
                if state.charges_used >= max_daily_recharges:
                    continue
                arrival_soc = state.soc - into_deadhead
                if arrival_soc < -1e-6 or arrival_time > horizon_min + 1e-6:
                    continue

                charge_levels = list(base_levels)
                if successor_boundary_soc_target:
                    charge_levels = _successor_boundary_soc_levels(
                        base_levels=base_levels,
                        successor_latest_departures=station_successor_deadlines.get(
                            station,
                            (),
                        ),
                        arrival_soc=arrival_soc,
                        arrival_time_min=arrival_time,
                        G=battery_capacity_kwh,
                        charge_rate_kw=charge_rate_kw,
                        max_successor_targets=max_successor_charge_targets,
                    )

                for target_soc in _dedupe_soc_levels(
                    charge_levels,
                    battery_capacity_kwh,
                ):
                    if target_soc <= arrival_soc + 1e-6:
                        continue
                    charged_kwh = target_soc - arrival_soc
                    charge_end = arrival_time + charged_kwh / charge_rate_kw * 60.0
                    if charge_end > latest_departure + 1e-6:
                        continue
                    if (
                        not station_waiting_unrestricted
                        and float(trip_start_min[following]) - charge_end
                        > max_station_to_trip_wait_min + 1e-6
                    ):
                        continue

                    next_soc = (
                        target_soc
                        - out_deadhead
                        - float(trip_energy_kwh[following])
                    )
                    if next_soc < -1e-6:
                        continue
                    candidates.append(
                        _RouteState(
                            soc=next_soc,
                            charges_used=state.charges_used + 1,
                            proxy_cost=(
                                state.proxy_cost
                                + (into_deadhead + out_deadhead)
                                * deadhead_cost_per_kwh
                                + charge_start_cost
                                + float(
                                    charging_cost(
                                        station,
                                        arrival_time,
                                        charged_kwh,
                                    )
                                )
                            ),
                            route=state.route + (station, following),
                            charging_stops=state.charging_stops
                            + ((station, arrival_time, charge_end, charged_kwh),),
                            deadhead_kwh=(
                                state.deadhead_kwh
                                + into_deadhead
                                + out_deadhead
                            ),
                        )
                    )

        states = _pareto(candidates)
        if not states:
            raise RouteRealizationError(
                "No time/SOC-feasible direct or one-charge transition for "
                f"trips {previous} -> {following}",
                trip_order=trips,
                failed_transition=(previous, following),
            )

    completed: list[_RouteState] = []
    last = trips[-1]
    direct_return = arcs.get((last, depot))
    if direct_return is not None and direct_return[3] == "trip_depot":
        _right, minutes, deadhead, _arc_type = direct_return
        for state in states:
            if (
                float(trip_end_min[last]) + minutes <= horizon_min + 1e-6
                and state.soc - deadhead >= -1e-6
            ):
                completed.append(
                    _RouteState(
                        soc=state.soc - deadhead,
                        charges_used=state.charges_used,
                        proxy_cost=(
                            state.proxy_cost + deadhead * deadhead_cost_per_kwh
                        ),
                        route=state.route + (depot,),
                        charging_stops=state.charging_stops,
                        deadhead_kwh=state.deadhead_kwh + deadhead,
                    )
                )

    for station in station_nodes:
        into_station = arcs.get((last, station))
        out_of_station = arcs.get((station, depot))
        if (
            into_station is None
            or out_of_station is None
            or into_station[3] != "trip_station"
            or out_of_station[3] != "station_depot"
        ):
            continue
        _right, into_minutes, into_deadhead, _arc_type = into_station
        _right, out_minutes, out_deadhead, _arc_type = out_of_station
        arrival_time = float(trip_end_min[last]) + into_minutes
        latest_departure = horizon_min - out_minutes
        for state in states:
            if state.charges_used >= max_daily_recharges:
                continue
            arrival_soc = state.soc - into_deadhead
            if arrival_soc < -1e-6:
                continue
            for target_soc in base_levels:
                if target_soc <= arrival_soc + 1e-6:
                    continue
                charged_kwh = target_soc - arrival_soc
                charge_end = arrival_time + charged_kwh / charge_rate_kw * 60.0
                if charge_end > latest_departure + 1e-6:
                    continue
                final_soc = target_soc - out_deadhead
                if final_soc < -1e-6:
                    continue
                completed.append(
                    _RouteState(
                        soc=final_soc,
                        charges_used=state.charges_used + 1,
                        proxy_cost=(
                            state.proxy_cost
                            + (into_deadhead + out_deadhead)
                            * deadhead_cost_per_kwh
                            + charge_start_cost
                            + float(
                                charging_cost(
                                    station,
                                    arrival_time,
                                    charged_kwh,
                                )
                            )
                        ),
                        route=state.route + (station, depot),
                        charging_stops=state.charging_stops
                        + ((station, arrival_time, charge_end, charged_kwh),),
                        deadhead_kwh=(
                            state.deadhead_kwh + into_deadhead + out_deadhead
                        ),
                    )
                )

    if not completed:
        raise RouteRealizationError(
            f"Last trip {last} cannot return to the depot",
            trip_order=trips,
            failed_transition=(last, depot),
        )

    best = min(
        completed,
        key=lambda state: (
            state.proxy_cost,
            state.charges_used,
            -state.soc,
            state.deadhead_kwh,
        ),
    )
    stations_out = [stop[0] for stop in best.charging_stops]
    starts_out = [stop[1] for stop in best.charging_stops]
    ends_out = [stop[2] for stop in best.charging_stops]
    energy_out = [stop[3] for stop in best.charging_stops]
    trip_set = set(trips)
    description = " -> ".join(
        f"T{node}" if node in trip_set else str(node) for node in best.route
    )
    return {
        "route": list(best.route),
        "charging_stops": {
            "stations": stations_out,
            "cst": starts_out,
            "cet": ends_out,
            "kwh": energy_out,
        },
        "charging_activities": len(stations_out),
        "type": "truck",
        "deadhead_kwh": best.deadhead_kwh,
        "_rc": 0.0,
        "desc": f"Matching path cover: {description}",
    }


def _minimum_contiguous_resource_segments(
    trip_path: Sequence[int],
    *,
    realize_segment: Callable[[tuple[int, ...]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[tuple[int, ...], ...]]:
    """Split one relaxed path into the fewest resource-feasible subpaths.

    The matching path order is retained: only cuts between consecutive trips
    are allowed.  This is a shortest-path dynamic program on the positions
    ``0..len(trip_path)``.  Ties are resolved by the lexicographically smallest
    tuple of cut positions, making the fallback deterministic.
    """

    trips = tuple(trip_path)
    if not trips:
        raise RouteRealizationError("Trip path is empty", trip_order=trips)

    # best[end] is (number of segments, cut positions, trip segments, routes)
    # for the prefix trips[:end].
    best: list[
        tuple[
            int,
            tuple[int, ...],
            tuple[tuple[int, ...], ...],
            tuple[dict[str, Any], ...],
        ]
        | None
    ] = [None] * (len(trips) + 1)
    best[0] = (0, (), (), ())

    for end in range(1, len(trips) + 1):
        for start in range(end):
            prefix = best[start]
            if prefix is None:
                continue
            segment = trips[start:end]
            try:
                route = realize_segment(segment)
            except RouteRealizationError:
                continue

            candidate = (
                prefix[0] + 1,
                prefix[1] + (end,),
                prefix[2] + (segment,),
                prefix[3] + (route,),
            )
            incumbent = best[end]
            if incumbent is None or candidate[:2] < incumbent[:2]:
                best[end] = candidate

    solution = best[-1]
    if solution is None:
        # If every singleton is feasible, the all-singleton partition is always
        # a valid fallback.  Preserve the concrete singleton failure otherwise.
        for trip in trips:
            try:
                realize_segment((trip,))
            except RouteRealizationError as error:
                raise RouteRealizationError(
                    "Contiguous splitting cannot cover the relaxed path because "
                    f"singleton trip {trip} is infeasible: {error}",
                    trip_order=trips,
                    failed_transition=error.failed_transition,
                ) from error
        raise MatchingInitializationError(
            "Contiguous splitting failed although every singleton is feasible"
        )

    return list(solution[3]), solution[2]


def build_matching_initial_routes(
    *,
    trips: Iterable[int],
    adjacency: Adjacency,
    depot: Hashable,
    stations: Iterable[Hashable],
    trip_start_min: Mapping[int, float],
    trip_end_min: Mapping[int, float],
    trip_energy_kwh: Mapping[int, float],
    direct_only: bool = False,
    max_matching_attempts: int = 32,
    matching_order_seed: int = 0,
    **realization_options: Any,
) -> list[dict[str, Any]]:
    """Build and resource-validate a model-derived path-cover seed.

    Every attempted matching has maximum cardinality for the same compatibility
    graph.  If its path partition is resource-infeasible, deterministic row and
    column orderings expose alternate maximum matchings. If no attempted minimum
    cover is resource-feasible, each relaxed path is split into the minimum
    number of contiguous feasible segments. Successful routes carry explicit
    exact-versus-repaired ``_matching_init`` provenance and the same distinction
    in ``desc``.
    """

    active_trips = tuple(trips)
    station_nodes = tuple(dict.fromkeys(stations))
    if len(set(active_trips)) != len(active_trips):
        raise MatchingInitializationError("Active trips must be unique")
    if not active_trips:
        return []
    if max_matching_attempts <= 0:
        raise MatchingInitializationError(
            "max_matching_attempts must be positive"
        )

    battery_capacity_kwh = float(
        realization_options.get("battery_capacity_kwh", 300.0)
    )
    charge_rate_kw = float(realization_options.get("charge_rate_kw", 300.0))
    horizon_min = float(realization_options.get("horizon_min", 1560.0))
    max_daily_recharges = int(
        realization_options.get("max_daily_recharges", 15)
    )
    max_wait = float(
        realization_options.get("max_station_to_trip_wait_min", 220.0)
    )
    successor_boundary = bool(
        realization_options.get("successor_boundary_soc_target", False)
    )
    max_successor_targets = int(
        realization_options.get("max_successor_charge_targets", 64)
    )
    if max_successor_targets <= 0:
        raise MatchingInitializationError(
            "max_successor_charge_targets must be positive"
        )
    station_waiting = bool(
        realization_options.get("station_waiting_unrestricted", False)
    )
    configured_levels = realization_options.get("soc_charge_levels")
    if configured_levels is None:
        configured_levels = [
            battery_capacity_kwh * index / 10.0 for index in range(1, 11)
        ]
    else:
        # The same iterable is needed by compatibility and every realization.
        configured_levels = list(configured_levels)
        realization_options["soc_charge_levels"] = configured_levels
    levels = _dedupe_soc_levels(configured_levels, battery_capacity_kwh)

    successors = _trip_successors(
        active_trips,
        adjacency,
        stations=station_nodes,
        trip_start_min=trip_start_min,
        trip_end_min=trip_end_min,
        trip_energy_kwh=trip_energy_kwh,
        battery_capacity_kwh=battery_capacity_kwh,
        charge_rate_kw=charge_rate_kw,
        soc_charge_levels=levels,
        horizon_min=horizon_min,
        max_daily_recharges=max_daily_recharges,
        max_station_to_trip_wait_min=max_wait,
        successor_boundary_soc_target=successor_boundary,
        station_waiting_unrestricted=station_waiting,
        direct_only=direct_only,
    )
    _assert_dag(active_trips, successors)

    seen_matchings: set[frozenset[tuple[int, int]]] = set()
    last_error: RouteRealizationError | None = None
    unique_matchings_tried = 0
    attempts_considered = 0
    matching_candidates: list[
        tuple[int, list[tuple[int, ...]], str]
    ] = []
    realization_cache: dict[
        tuple[int, ...], dict[str, Any] | RouteRealizationError
    ] = {}

    def realize_path(path: tuple[int, ...]) -> dict[str, Any]:
        cached = realization_cache.get(path)
        if isinstance(cached, RouteRealizationError):
            raise cached
        if cached is not None:
            return cached
        try:
            route = realize_fixed_trip_path(
                path,
                adjacency=adjacency,
                depot=depot,
                stations=station_nodes,
                trip_start_min=trip_start_min,
                trip_end_min=trip_end_min,
                trip_energy_kwh=trip_energy_kwh,
                **realization_options,
            )
        except RouteRealizationError as error:
            realization_cache[path] = error
            raise
        realization_cache[path] = route
        return route

    def finalize_routes(
        routes: list[dict[str, Any]],
        *,
        selected_attempt: int,
        selected_order: str,
        relaxed_path_count: int,
        repair_mode: str,
    ) -> list[dict[str, Any]]:
        active_set = set(active_trips)
        covered = [
            node
            for route in routes
            for node in route["route"]
            if node in active_set
        ]
        if len(covered) != len(active_trips) or set(covered) != active_set:
            raise MatchingInitializationError(
                "Constructed routes do not cover every active trip exactly once"
            )

        exact_minimum = repair_mode == "none"
        final_path_count = len(routes)
        provenance = {
            "compatibility_mode": "direct_only" if direct_only else "full",
            "matching_attempt_index": selected_attempt,
            "matching_retry_count": unique_matchings_tried - 1,
            "matching_attempts_considered": attempts_considered,
            "unique_matchings_tried": unique_matchings_tried,
            "matching_order_seed": int(matching_order_seed),
            "matching_order": selected_order,
            "matching_cardinality": len(active_trips) - relaxed_path_count,
            "path_count": final_path_count,
            "relaxed_minimum_path_count": relaxed_path_count,
            "resource_feasible_path_count": final_path_count,
            "resource_repair_mode": repair_mode,
            "is_exact_minimum_path_cover": exact_minimum,
            "contiguous_splits_added": final_path_count - relaxed_path_count,
            "max_successor_charge_targets": max_successor_targets,
        }
        provenance_text = (
            f"mode={provenance['compatibility_mode']}, "
            f"retry={provenance['matching_retry_count']}, "
            f"seed={matching_order_seed}, order={selected_order}, "
            f"repair={repair_mode}"
        )
        for route in routes:
            route["_matching_init"] = dict(provenance)
            route["desc"] = (
                f"Matching path cover [{provenance_text}]: "
                + route["desc"].removeprefix("Matching path cover: ")
            )
        return routes

    for attempt in range(max_matching_attempts):
        attempts_considered = attempt + 1
        paths, signature, order_name = _path_cover_from_successors(
            active_trips,
            successors,
            trip_start_min=trip_start_min,
            matching_attempt=attempt,
            matching_order_seed=matching_order_seed,
        )
        if signature in seen_matchings:
            continue
        seen_matchings.add(signature)
        unique_matchings_tried += 1
        matching_candidates.append((attempt, paths, order_name))

        routes: list[dict[str, Any]] = []
        failed = False
        for path_index, path in enumerate(paths):
            try:
                route = realize_path(path)
            except RouteRealizationError as error:
                last_error = RouteRealizationError(
                    f"Minimum path-cover route {path_index} is infeasible: {error}",
                    trip_order=error.trip_order,
                    failed_transition=error.failed_transition,
                )
                failed = True
                break
            routes.append(route)

        if failed:
            continue

        return finalize_routes(
            routes,
            selected_attempt=attempt,
            selected_order=order_name,
            relaxed_path_count=len(paths),
            repair_mode="none",
        )

    # Every attempted maximum-cardinality matching contained at least one path
    # that the exact resource extension could not realize.  Retain each path's
    # trip order but cut it into a minimum number of contiguous feasible routes.
    # Compare the repaired versions of all unique matchings already attempted;
    # route count is primary and the earlier deterministic attempt breaks ties.
    best_repaired: tuple[
        tuple[int, int],
        list[dict[str, Any]],
        int,
        str,
        int,
    ] | None = None
    fallback_error: RouteRealizationError | None = None
    for candidate_attempt, candidate_paths, candidate_order in matching_candidates:
        repaired_routes: list[dict[str, Any]] = []
        try:
            for path in candidate_paths:
                segment_routes, _segments = _minimum_contiguous_resource_segments(
                    path,
                    realize_segment=realize_path,
                )
                repaired_routes.extend(segment_routes)
        except RouteRealizationError as error:
            fallback_error = error
            continue

        key = (len(repaired_routes), candidate_attempt)
        if best_repaired is None or key < best_repaired[0]:
            best_repaired = (
                key,
                repaired_routes,
                candidate_attempt,
                candidate_order,
                len(candidate_paths),
            )
        # Since every exact realization failed, one extra route is the best
        # possible repaired count for any attempted relaxed minimum cover.
        if len(repaired_routes) == len(candidate_paths) + 1:
            break

    if best_repaired is not None:
        _, repaired_routes, selected_attempt, selected_order, relaxed_count = (
            best_repaired
        )
        return finalize_routes(
            repaired_routes,
            selected_attempt=selected_attempt,
            selected_order=selected_order,
            relaxed_path_count=relaxed_count,
            repair_mode="contiguous_split",
        )

    if fallback_error is not None:
        raise RouteRealizationError(
            "No resource-feasible route cover after maximum-matching retries and "
            f"contiguous splitting; last failure: {fallback_error}",
            trip_order=fallback_error.trip_order,
            failed_transition=fallback_error.failed_transition,
        ) from fallback_error
    if last_error is None:
        raise RouteRealizationError(
            "No resource-distinct maximum matching was found across "
            f"{max_matching_attempts} deterministic orderings",
            trip_order=(),
        )
    raise RouteRealizationError(
        "No resource-feasible maximum matching after "
        f"{max_matching_attempts} deterministic orderings "
        f"({unique_matchings_tried} unique matchings); last failure: {last_error}",
        trip_order=last_error.trip_order,
        failed_transition=last_error.failed_transition,
    ) from last_error

#!/usr/bin/env python3
"""Event-based fixed-duty charging optimization with exact physical replay.

The ordered trip sequence is fixed.  A small HiGHS MILP chooses one restricted
graph option in every inter-trip gap, continuous charging energy in tariff-hour
segments, and the number of contiguous charge events.  Charging can start
anywhere in a station layover; no time or SOC lattice is used.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from audit_giro_known_columns import DEPOT, HORIZON_MIN, STATIONS
from config import (
    BUS_COST_KX,
    CHARGE_START_COST,
    charge_cost_premium,
)
from fixed_duty_expanded_optimizer import (
    _arc_groups,
    _floor,
    evaluate_fixed_duty_transition,
    optimize_fixed_duty,
)
from run_exact_pool_mip import validate_injected_route
from tariff_response_core import canonical_sha
from utils_v2 import base_station_name


RESULT_SCHEMA = "evsp-dr-fixed-duty-continuous-result-v1"
CERTIFICATE_SCHEMA = "evsp-dr-fixed-duty-continuous-certificate-v1"
REPLAY_SCHEMA = "evsp-dr-fixed-duty-continuous-replay-v1"
PHYSICS_CAVEAT = (
    "Constant power to 100% is a standard linear E-VSP idealization and is "
    "mildly optimistic because real charging tapers above about 80% SOC."
)
CAPACITY_CAVEAT = (
    "Charger capacity is unlimited. Load and concurrency are reported, but "
    "this solution does not license a peak-shaving claim."
)
TOL = 1e-7


@dataclass(frozen=True)
class Segment:
    start_min: float
    end_min: float
    tariff_hour: int
    price_per_kwh: float
    capacity_kwh: float


@dataclass(frozen=True)
class GapOption:
    gap: int
    kind: str
    station: str | None
    inbound_min: float
    outbound_min: float
    inbound_kwh: float
    outbound_kwh: float
    arrival_min: float
    latest_departure_min: float
    segments: tuple[Segment, ...]

    @property
    def deadhead_kwh(self) -> float:
        return self.inbound_kwh + self.outbound_kwh


def _normal_terminal_policy(value: str) -> str:
    aliases = {
        "free": "free",
        "reserve": ">= reserve",
        ">= reserve": ">= reserve",
        "start": ">= start",
        ">= start": ">= start",
        "priced": "priced terminal energy",
        "priced terminal energy": "priced terminal energy",
    }
    try:
        return aliases[str(value).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "terminal_soc_policy must be free, >= reserve, >= start, or "
            "priced terminal energy"
        ) from exc


def _segments(start, end, station, station_prices, charge_kw):
    curve = station_prices.get(base_station_name(station))
    if not curve:
        raise ValueError(f"tariff missing station {base_station_name(station)}")
    result = []
    cursor = float(start)
    while cursor < float(end) - 1e-9:
        hour = int(cursor // 60)
        if hour not in curve:
            raise ValueError(f"tariff hour {hour} missing at {station}")
        segment_end = min(float(end), float((hour + 1) * 60))
        price = float(curve[hour]) * float(charge_cost_premium)
        if not math.isfinite(price):
            raise ValueError("tariff contains a non-finite price")
        result.append(Segment(
            start_min=cursor,
            end_min=segment_end,
            tariff_hour=hour,
            price_per_kwh=price,
            capacity_kwh=(segment_end - cursor) * charge_kw / 60.0,
        ))
        cursor = segment_end
    return tuple(result)


def _gap_options(problem, trips, station_prices, charge_kw):
    arcs = _arc_groups(problem)
    options = []
    for gap, trip in enumerate(trips):
        final = gap == len(trips) - 1
        successor = None if final else trips[gap + 1]
        deadline = HORIZON_MIN if final else float(problem.start_min[successor])
        direct = (
            arcs["trip_depot"].get(trip)
            if final else arcs["trip_trip"].get(trip, {}).get(successor)
        )
        current = []
        if (
            direct is not None
            and float(problem.end_min[trip]) + direct.travel_min
            <= deadline + 1e-9
        ):
            current.append(GapOption(
                gap, "direct", None, direct.travel_min, 0.0,
                direct.deadhead_kwh, 0.0,
                float(problem.end_min[trip]) + direct.travel_min,
                deadline, (),
            ))
        inbound_by_station = arcs["trip_station"].get(trip, {})
        for station in sorted(set(STATIONS) | set(inbound_by_station)):
            inbound = inbound_by_station.get(station)
            outbound = (
                arcs["station_depot"].get(station)
                if final
                else arcs["station_trip"].get(station, {}).get(successor)
            )
            if inbound is None or outbound is None:
                continue
            arrival = float(problem.end_min[trip]) + inbound.travel_min
            latest = deadline - outbound.travel_min
            if arrival > latest + 1e-9:
                continue
            current.append(GapOption(
                gap, "station", station,
                inbound.travel_min, outbound.travel_min,
                inbound.deadhead_kwh, outbound.deadhead_kwh,
                arrival, latest,
                _segments(
                    arrival, latest, station, station_prices, charge_kw
                ),
            ))
        if not current:
            return None, f"gap after trip {trip} has no feasible graph option"
        options.append(tuple(current))
    return tuple(options), None


class _Model:
    def __init__(self):
        self.cost = []
        self.lb = []
        self.ub = []
        self.integrality = []
        self.rows = []
        self.cols = []
        self.values = []
        self.con_lb = []
        self.con_ub = []

    def variable(self, *, cost=0.0, lb=0.0, ub=np.inf, integer=False):
        index = len(self.cost)
        self.cost.append(float(cost))
        self.lb.append(float(lb))
        self.ub.append(float(ub))
        self.integrality.append(1 if integer else 0)
        return index

    def constraint(self, entries, lower=-np.inf, upper=np.inf):
        row = len(self.con_lb)
        for column, value in entries:
            if abs(value) > 0.0:
                self.rows.append(row)
                self.cols.append(column)
                self.values.append(float(value))
        self.con_lb.append(float(lower))
        self.con_ub.append(float(upper))

    def solve(self, *, time_limit_s=None):
        matrix = csr_matrix(
            (self.values, (self.rows, self.cols)),
            shape=(len(self.con_lb), len(self.cost)),
        )
        options = {"presolve": True}
        if time_limit_s is not None:
            options["time_limit"] = float(time_limit_s)
        return milp(
            c=np.asarray(self.cost),
            integrality=np.asarray(self.integrality),
            bounds=Bounds(self.lb, self.ub),
            constraints=LinearConstraint(
                matrix, self.con_lb, self.con_ub
            ),
            options=options,
        )


def _build_model(
    problem,
    trips,
    options,
    *,
    g_kwh,
    reserve_kwh,
    charge_start_cost,
    timing_mode,
    terminal_policy,
    terminal_energy_price,
):
    model = _Model()
    y_idx, energy_idx, on_idx, start_idx = {}, {}, {}, {}
    for gap, gap_options in enumerate(options):
        for option_index, option in enumerate(gap_options):
            y_idx[gap, option_index] = model.variable(ub=1, integer=True)
            previous_on = None
            for segment_index, segment in enumerate(option.segments):
                key = (gap, option_index, segment_index)
                energy_idx[key] = model.variable(
                    cost=segment.price_per_kwh,
                    ub=segment.capacity_kwh,
                )
                on_idx[key] = model.variable(ub=1, integer=True)
                start_idx[key] = model.variable(
                    cost=charge_start_cost, ub=1, integer=True
                )
                energy = energy_idx[key]
                on = on_idx[key]
                start = start_idx[key]
                model.constraint(
                    [(energy, 1), (on, -segment.capacity_kwh)],
                    upper=0,
                )
                model.constraint(
                    [(on, 1), (y_idx[gap, option_index], -1)],
                    upper=0,
                )
                model.constraint([(start, 1), (on, -1)], upper=0)
                if previous_on is None:
                    model.constraint([(start, 1), (on, -1)], lower=0)
                else:
                    model.constraint(
                        [(start, 1), (on, -1), (previous_on, 1)],
                        lower=0,
                    )
                    model.constraint(
                        [(start, 1), (previous_on, 1)], upper=1
                    )
                previous_on = on
            for segment_index in range(1, len(option.segments) - 1):
                key = (gap, option_index, segment_index)
                previous = on_idx[gap, option_index, segment_index - 1]
                following = on_idx[gap, option_index, segment_index + 1]
                model.constraint(
                    [
                        (energy_idx[key], 1),
                        (previous, -option.segments[
                            segment_index
                        ].capacity_kwh),
                        (on_idx[key], -option.segments[
                            segment_index
                        ].capacity_kwh),
                        (following, -option.segments[
                            segment_index
                        ].capacity_kwh),
                    ],
                    lower=-2 * option.segments[
                        segment_index
                    ].capacity_kwh,
                )
            if timing_mode == "arrival":
                for segment_index in range(1, len(option.segments)):
                    prior = option.segments[segment_index - 1]
                    model.constraint(
                        [
                            (
                                energy_idx[
                                    gap, option_index, segment_index - 1
                                ],
                                1,
                            ),
                            (
                                on_idx[gap, option_index, segment_index],
                                -prior.capacity_kwh,
                            ),
                        ],
                        lower=0,
                    )
        model.constraint(
            [
                (y_idx[gap, option_index], 1)
                for option_index in range(len(gap_options))
            ],
            lower=1,
            upper=1,
        )

    soc_idx = {
        index: model.variable(lb=reserve_kwh, ub=g_kwh)
        for index in range(len(trips))
    }
    terminal_lb = (
        g_kwh if terminal_policy == ">= start" else reserve_kwh
    )
    terminal_cost = (
        -terminal_energy_price
        if terminal_policy == "priced terminal energy" else 0.0
    )
    terminal_idx = model.variable(
        cost=terminal_cost, lb=terminal_lb, ub=g_kwh
    )

    first_arc = _arc_groups(problem)["depot_trip"].get(trips[0])
    if first_arc is None:
        return None, "depot cannot reach first trip"
    if (
        first_arc.travel_min
        > float(problem.start_min[trips[0]]) + 1e-9
    ):
        return None, "depot cannot reach first trip in time"
    first_soc = (
        g_kwh - first_arc.deadhead_kwh
        - float(problem.trip_energy[trips[0]])
    )
    if first_soc < reserve_kwh - TOL:
        return None, "first trip violates reserve SOC"
    model.constraint([(soc_idx[0], 1)], lower=first_soc, upper=first_soc)

    for gap, gap_options in enumerate(options):
        final = gap == len(trips) - 1
        target = terminal_idx if final else soc_idx[gap + 1]
        entries = [(target, 1), (soc_idx[gap], -1)]
        for option_index, option in enumerate(gap_options):
            y = y_idx[gap, option_index]
            entries.append((y, option.deadhead_kwh))
            option_energy = [
                energy_idx[gap, option_index, segment_index]
                for segment_index in range(len(option.segments))
            ]
            entries.extend((index, -1) for index in option_energy)
            if option.kind == "station":
                model.constraint(
                    [(soc_idx[gap], 1), (y, -g_kwh)],
                    lower=(
                        reserve_kwh + option.inbound_kwh - g_kwh
                    ),
                )
                model.constraint(
                    [(soc_idx[gap], 1), (y, g_kwh)]
                    + [(index, 1) for index in option_energy],
                    upper=2 * g_kwh + option.inbound_kwh,
                )
        draw = 0.0 if final else float(
            problem.trip_energy[trips[gap + 1]]
        )
        model.constraint(entries, lower=-draw, upper=-draw)
    return {
        "model": model,
        "y": y_idx,
        "energy": energy_idx,
        "on": on_idx,
        "start": start_idx,
        "soc": soc_idx,
        "terminal": terminal_idx,
        "first_arc": first_arc,
    }, None


def _selected_events(option, option_energy, option_on, charge_kw, timing_mode):
    runs = []
    cursor = 0
    while cursor < len(option.segments):
        if option_on[cursor] <= 0.5:
            cursor += 1
            continue
        end = cursor
        while end + 1 < len(option.segments) and option_on[end + 1] > 0.5:
            end += 1
        positive = [
            index for index in range(cursor, end + 1)
            if option_energy[index] > TOL
        ]
        if positive:
            first, last = positive[0], positive[-1]
            blocks = []
            for index in range(first, last + 1):
                segment = option.segments[index]
                energy = max(0.0, float(option_energy[index]))
                duration = energy * 60.0 / charge_kw
                if first == last or (
                    index == first and timing_mode == "arrival"
                ):
                    start = segment.start_min
                    finish = start + duration
                elif index == first:
                    finish = segment.end_min
                    start = finish - duration
                else:
                    start = segment.start_min
                    finish = start + duration
                blocks.append({
                    "station": option.station,
                    "start_min": start,
                    "end_min": finish,
                    "delivered_kwh": energy,
                    "tariff_hour": segment.tariff_hour,
                    "price_per_kwh": segment.price_per_kwh,
                    "energy_cost": energy * segment.price_per_kwh,
                })
            runs.append({
                "station": option.station,
                "station_arrival_min": option.arrival_min,
                "latest_departure_min": option.latest_departure_min,
                "start_min": blocks[0]["start_min"],
                "end_min": blocks[-1]["end_min"],
                "duration_min": (
                    blocks[-1]["end_min"] - blocks[0]["start_min"]
                ),
                "delivered_kwh": sum(
                    block["delivered_kwh"] for block in blocks
                ),
                "energy_cost": sum(block["energy_cost"] for block in blocks),
                "blocks": blocks,
            })
        cursor = end + 1
    return runs


def _replay(
    problem,
    trips,
    options,
    selected,
    events_by_gap,
    station_prices,
    *,
    g_kwh,
    charge_kw,
    reserve_kwh,
    charge_start_cost,
    terminal_policy,
    terminal_energy_price,
    reported_objective,
):
    soc = g_kwh
    trace = []
    first = _arc_groups(problem)["depot_trip"][trips[0]]
    soc -= first.deadhead_kwh
    if soc < reserve_kwh - TOL:
        raise ValueError("replay violates reserve before first trip")
    soc -= float(problem.trip_energy[trips[0]])
    if soc < reserve_kwh - TOL:
        raise ValueError("replay violates reserve after first trip")
    trace.append({"event": "trip", "trip": trips[0], "soc_after_kwh": soc})
    route_nodes = [DEPOT, trips[0]]
    stops = {"stations": [], "cst": [], "cet": [], "kwh": []}
    all_events, blocks = [], []
    for gap, option_index in enumerate(selected):
        option = options[gap][option_index]
        final = gap == len(trips) - 1
        if option.kind == "direct":
            soc -= option.deadhead_kwh
            trace.append({
                "event": "deadhead", "gap": gap, "kind": "direct",
                "soc_after_kwh": soc,
            })
        else:
            soc -= option.inbound_kwh
            if soc < reserve_kwh - TOL:
                raise ValueError("replay violates reserve on station arrival")
            events = events_by_gap[gap]
            if not events:
                route_nodes.append(option.station)
                stops["stations"].append(option.station)
                stops["cst"].append(option.arrival_min)
                stops["cet"].append(option.arrival_min)
                stops["kwh"].append(0.0)
            previous_end = option.arrival_min
            for event_index, event in enumerate(events):
                if event["start_min"] < previous_end - TOL:
                    raise ValueError("replay charging events overlap")
                event_blocks = []
                for block in event["blocks"]:
                    duration = block["end_min"] - block["start_min"]
                    if not math.isclose(
                        block["delivered_kwh"],
                        duration * charge_kw / 60.0,
                        abs_tol=1e-6,
                    ):
                        raise ValueError("replay charging block violates power")
                    curve = station_prices[base_station_name(option.station)]
                    expected_price = (
                        float(curve[block["tariff_hour"]])
                        * float(charge_cost_premium)
                    )
                    if not math.isclose(
                        expected_price, block["price_per_kwh"],
                        abs_tol=1e-12,
                    ):
                        raise ValueError("replay charging tariff mismatch")
                    soc += block["delivered_kwh"]
                    if soc > g_kwh + TOL:
                        raise ValueError("replay charging exceeds battery capacity")
                    enriched = {
                        **block, "gap": gap, "event_index": event_index,
                        "soc_after_kwh": soc,
                    }
                    blocks.append(enriched)
                    event_blocks.append(enriched)
                emitted = {
                    **event,
                    "gap": gap,
                    "event_index": event_index,
                    "charge_start_cost": charge_start_cost,
                    "event_cost": event["energy_cost"] + charge_start_cost,
                    "delayed_start": (
                        event["start_min"] > option.arrival_min + TOL
                    ),
                    "blocks": event_blocks,
                }
                all_events.append(emitted)
                route_nodes.append(option.station)
                stops["stations"].append(option.station)
                stops["cst"].append(event["start_min"])
                stops["cet"].append(event["end_min"])
                stops["kwh"].append(event["delivered_kwh"])
                previous_end = event["end_min"]
            soc -= option.outbound_kwh
            trace.append({
                "event": "station_gap", "gap": gap,
                "station": option.station, "soc_after_kwh": soc,
            })
        if soc < reserve_kwh - TOL:
            raise ValueError("replay violates reserve after deadhead")
        if not final:
            next_trip = trips[gap + 1]
            route_nodes.append(next_trip)
            soc -= float(problem.trip_energy[next_trip])
            if soc < reserve_kwh - TOL:
                raise ValueError("replay violates reserve after trip")
            trace.append({
                "event": "trip", "trip": next_trip, "soc_after_kwh": soc,
            })
    route_nodes.append(DEPOT)
    if terminal_policy == ">= start" and soc < g_kwh - TOL:
        raise ValueError("replay violates start-SOC terminal policy")
    electricity_cost = sum(block["energy_cost"] for block in blocks)
    starts_cost = len(all_events) * charge_start_cost
    credit = (
        terminal_energy_price * soc
        if terminal_policy == "priced terminal energy" else 0.0
    )
    objective = BUS_COST_KX + electricity_cost + starts_cost - credit
    if not math.isclose(objective, reported_objective, abs_tol=1e-6):
        raise ValueError(
            f"replay cost {objective} differs from objective "
            f"{reported_objective}"
        )
    route = {
        "trips": list(trips),
        "route_nodes": route_nodes,
        "charging_stops": stops,
    }
    reason = validate_injected_route(
        problem, route, g_kwh, charge_kw, reserve_kwh, HORIZON_MIN
    )
    if reason is not None:
        raise ValueError(f"continuous replay rejected: {reason}")
    return {
        "schema": REPLAY_SCHEMA,
        "ok": True,
        "route": route,
        "events": all_events,
        "blocks": blocks,
        "trace": trace,
        "terminal_soc_kwh": soc,
        "electricity_cost": electricity_cost,
        "charge_start_cost": starts_cost,
        "terminal_energy_credit": credit,
        "replayed_objective": objective,
        "peak_kw": charge_kw if blocks else 0.0,
        "charger_concurrency_max": 1 if blocks else 0,
    }


def optimize_fixed_duty_continuous(
    problem,
    trip_sequence,
    station_prices,
    *,
    g_kwh=240.0,
    charge_kw=240.0,
    reserve_kwh=0.0,
    charge_start_cost=CHARGE_START_COST,
    terminal_soc_policy="free",
    terminal_energy_price=None,
    timing_mode="optimized",
    tariff_id=None,
    tariff_sha256=None,
    instance_sha256=None,
    time_limit_s=None,
):
    """Optimize one fixed ordered trip sequence and replay the exact schedule."""
    started = time.perf_counter()
    trips = tuple(int(trip) for trip in trip_sequence)
    if (
        not trips
        or len(trips) != len(set(trips))
        or any(trip not in set(problem.trips) for trip in trips)
    ):
        raise ValueError("fixed duty has invalid or foreign trips")
    values = (g_kwh, charge_kw, reserve_kwh, charge_start_cost)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("physics and costs must be finite")
    g_kwh = float(g_kwh)
    charge_kw = float(charge_kw)
    reserve_kwh = float(reserve_kwh)
    charge_start_cost = float(charge_start_cost)
    if (
        g_kwh <= 0 or charge_kw <= 0
        or not 0 <= reserve_kwh <= g_kwh
        or charge_start_cost < 0
    ):
        raise ValueError("invalid fixed-duty physics or start cost")
    if timing_mode not in {"optimized", "arrival"}:
        raise ValueError("timing_mode must be optimized or arrival")
    terminal_policy = _normal_terminal_policy(terminal_soc_policy)
    if terminal_policy == "priced terminal energy":
        if terminal_energy_price is None or not math.isfinite(
            float(terminal_energy_price)
        ):
            raise ValueError(
                "priced terminal energy requires terminal_energy_price"
            )
        terminal_energy_price = float(terminal_energy_price)
    elif terminal_energy_price is not None:
        raise ValueError(
            "terminal_energy_price is only valid for priced terminal energy"
        )
    else:
        terminal_energy_price = 0.0
    options, reason = _gap_options(
        problem, trips, station_prices, charge_kw
    )
    if reason is not None:
        return _infeasible(
            trips, reason, terminal_policy, tariff_id, tariff_sha256, started
        )
    built, reason = _build_model(
        problem, trips, options,
        g_kwh=g_kwh,
        reserve_kwh=reserve_kwh,
        charge_start_cost=charge_start_cost,
        timing_mode=timing_mode,
        terminal_policy=terminal_policy,
        terminal_energy_price=terminal_energy_price,
    )
    if reason is not None:
        return _infeasible(
            trips, reason, terminal_policy, tariff_id, tariff_sha256, started
        )
    solved = built["model"].solve(time_limit_s=time_limit_s)
    if not solved.success or solved.x is None:
        return _infeasible(
            trips,
            f"HiGHS status {solved.status}: {solved.message}",
            terminal_policy,
            tariff_id,
            tariff_sha256,
            started,
            solver_status=int(solved.status),
        )
    x = solved.x
    selected = []
    events_by_gap = {}
    for gap, gap_options in enumerate(options):
        chosen = [
            option_index for option_index in range(len(gap_options))
            if x[built["y"][gap, option_index]] > 0.5
        ]
        if len(chosen) != 1:
            raise ValueError("MILP did not select exactly one gap option")
        option_index = chosen[0]
        selected.append(option_index)
        option = gap_options[option_index]
        energy = [
            x[built["energy"][gap, option_index, segment_index]]
            for segment_index in range(len(option.segments))
        ]
        on = [
            x[built["on"][gap, option_index, segment_index]]
            for segment_index in range(len(option.segments))
        ]
        events_by_gap[gap] = _selected_events(
            option, energy, on, charge_kw, timing_mode
        )
    objective = float(BUS_COST_KX + solved.fun)
    replay = _replay(
        problem, trips, options, selected, events_by_gap, station_prices,
        g_kwh=g_kwh,
        charge_kw=charge_kw,
        reserve_kwh=reserve_kwh,
        charge_start_cost=charge_start_cost,
        terminal_policy=terminal_policy,
        terminal_energy_price=terminal_energy_price,
        reported_objective=objective,
    )
    event_count = len(replay["events"])
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "certified": True,
        "scope": "optimal_continuous_charging_for_fixed_trip_sequence",
        "algorithm": "event_based_milp",
        "solver": "scipy.optimize.milp/HiGHS",
        "scipy_version": scipy.__version__,
        "solver_status": int(solved.status),
        "solver_message": str(solved.message),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "trip_sequence": list(trips),
        "instance_sha256": instance_sha256,
        "tariff_id": tariff_id,
        "tariff_sha256": tariff_sha256,
        "physics": {
            "g_kwh": g_kwh,
            "charge_kw": charge_kw,
            "reserve_kwh": reserve_kwh,
            "charge_start_cost": charge_start_cost,
            "chargers": "unlimited",
            "deadhead": "zone_centroid",
            "all_buses_start": "full",
        },
        "timing_mode": timing_mode,
        "terminal_soc_policy": terminal_policy,
        "terminal_energy_price": (
            terminal_energy_price
            if terminal_policy == "priced terminal energy" else None
        ),
        "objective": objective,
        "replay_sha256": canonical_sha(replay),
    }
    certificate["certificate_sha256"] = canonical_sha(certificate)
    return {
        "schema": RESULT_SCHEMA,
        "feasible": True,
        "trip_sequence": list(trips),
        "objective": objective,
        "charging_cost": (
            replay["electricity_cost"] + replay["charge_start_cost"]
        ),
        "electricity_cost": replay["electricity_cost"],
        "charge_events": event_count,
        "delayed_starts": sum(
            event["delayed_start"] for event in replay["events"]
        ),
        "terminal_soc_kwh": replay["terminal_soc_kwh"],
        "terminal_soc_policy": terminal_policy,
        "timing_mode": timing_mode,
        "peak_kw": replay["peak_kw"],
        "charger_concurrency_max": replay["charger_concurrency_max"],
        "route": replay["route"],
        "charging_events": replay["events"],
        "charging_blocks": replay["blocks"],
        "physical_replay": replay,
        "physical_replay_status": "validated",
        "certificate": certificate,
        "physics_caveat": PHYSICS_CAVEAT,
        "charger_capacity_caveat": CAPACITY_CAVEAT,
        "runtime_s": time.perf_counter() - started,
    }


def _infeasible(
    trips,
    reason,
    terminal_policy,
    tariff_id,
    tariff_sha256,
    started,
    *,
    solver_status=None,
):
    return {
        "schema": RESULT_SCHEMA,
        "feasible": False,
        "trip_sequence": list(trips),
        "reason": reason,
        "terminal_soc_policy": terminal_policy,
        "tariff_id": tariff_id,
        "tariff_sha256": tariff_sha256,
        "solver_status": solver_status,
        "physical_replay_status": "not_available",
        "physics_caveat": PHYSICS_CAVEAT,
        "charger_capacity_caveat": CAPACITY_CAVEAT,
        "runtime_s": time.perf_counter() - started,
    }


def validate_lattice_reproduction(
    problem,
    trip_sequence,
    station_prices,
    *,
    tariff_id=None,
    tariff_sha256=None,
    instance_sha256=None,
):
    """Solve the legacy lattice transition graph as a MILP and compare its DP."""
    trips = tuple(int(trip) for trip in trip_sequence)
    g_kwh, charge_kw, reserve_kwh = 300.0, 300.0, 0.0
    soc_step, block_min = 15.0, 10
    grid = [
        index * soc_step for index in range(int(g_kwh / soc_step) + 1)
    ]
    arcs = _arc_groups(problem)
    first = arcs["depot_trip"].get(trips[0])
    if first is None:
        raise ValueError("lattice source arc missing")
    first_level = _floor(grid, soc_step, g_kwh - first.deadhead_kwh)
    reachable = {first_level}
    transitions = []
    for position, trip in enumerate(trips):
        final = position == len(trips) - 1
        successor = None if final else trips[position + 1]
        next_reachable = set()
        for level in sorted(reachable):
            candidates, _trace = evaluate_fixed_duty_transition(
                problem,
                arcs,
                trip=trip,
                successor=successor,
                final_gap=final,
                level=level,
                base_cost=0.0,
                actions=(),
                grid=grid,
                soc_step=soc_step,
                block_min=block_min,
                g_kwh=g_kwh,
                charge_kw=charge_kw,
                reserve_kwh=reserve_kwh,
                station_prices=station_prices,
                n_blocks=int(HORIZON_MIN) // block_min,
            )
            cheapest = {}
            for candidate in candidates:
                target = None if final else candidate["next_level"]
                current = cheapest.get(target)
                key = (
                    candidate["cost"],
                    canonical_sha(candidate["action"]),
                )
                if current is None or key < current[0]:
                    cheapest[target] = (key, candidate)
            for target, (_key, candidate) in cheapest.items():
                transitions.append({
                    "position": position,
                    "source": level,
                    "target": target,
                    "cost": candidate["cost"],
                })
                if target is not None:
                    next_reachable.add(target)
        if not final and not next_reachable:
            break
        reachable = next_reachable
    reference = optimize_fixed_duty(
        problem,
        trips,
        station_prices,
        tariff_id=tariff_id,
        tariff_sha256=tariff_sha256,
        instance_sha256=instance_sha256,
    )
    if not reference["feasible"]:
        return {
            "feasible": False,
            "reference_feasible": False,
            "matches": True,
            "reason": reference["reason"],
        }
    model = _Model()
    variable = [
        model.variable(cost=row["cost"], ub=1, integer=True)
        for row in transitions
    ]
    nodes = {
        (row["position"], row["source"]) for row in transitions
    } | {
        (row["position"] + 1, row["target"])
        for row in transitions if row["target"] is not None
    }
    for position, level in sorted(nodes):
        incoming = [
            (variable[index], 1)
            for index, row in enumerate(transitions)
            if row["position"] == position - 1 and row["target"] == level
        ]
        outgoing = [
            (variable[index], -1)
            for index, row in enumerate(transitions)
            if row["position"] == position and row["source"] == level
        ]
        supply = -1.0 if (position, level) == (0, first_level) else 0.0
        model.constraint(incoming + outgoing, lower=supply, upper=supply)
    terminal = [
        (variable[index], 1)
        for index, row in enumerate(transitions)
        if row["position"] == len(trips) - 1 and row["target"] is None
    ]
    model.constraint(terminal, lower=1, upper=1)
    solved = model.solve()
    if not solved.success:
        raise ValueError(f"lattice MILP failed: {solved.message}")
    objective = BUS_COST_KX + float(solved.fun)
    difference = objective - float(reference["expanded_grid_objective"])
    return {
        "feasible": True,
        "reference_feasible": True,
        "milp_objective": objective,
        "dp_objective": float(reference["expanded_grid_objective"]),
        "difference": difference,
        "matches": math.isclose(difference, 0.0, abs_tol=1e-7),
        "transition_count": len(transitions),
        "scope": "legacy_300kWh_300kW_15kWh_10min_lattice",
    }

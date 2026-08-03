"""Audit historical GIRO duties against a saved column-generation route pool.

This is a diagnostic, not a CHEAT warm start.  It reconstructs the saved
restricted master with SciPy/HiGHS, obtains one optimal dual vector, and asks:

* what reduced cost would each historical duty have at that dual; and
* can the current restricted DP resources realize the same ordered trip list?

Only the tracked, derived CSV inputs are read.  The original GIRO workbooks are
not required.  Per-duty reduced costs can change across equally optimal dual
vectors; aggregate and feasibility results should therefore be interpreted
alongside the dual-degeneracy warning emitted in the JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config import (
    BIG_M_PENALTY,
    BUS_COST_KX,
    CHARGE_RATE_KW,
    CHARGE_START_COST,
)
from master_lp_scipy import build_route_incidence, solve_restricted_master_lp
from matching_init import RouteRealizationError, realize_fixed_trip_path
from pricing_dp_og import build_dag
from utils_v2 import calculate_truck_route_cost_accurate, load_station_hourly_prices


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEPOT = "PARX_0"
STATIONS = (
    "2190L_0",
    "4808_0",
    "PARX_1",
    "3127L_0",
    "7880C_0",
    "JON_A_0",
)
STATION_NODE_BY_BASE = {
    "2190L": "2190L_0",
    "4808": "4808_0",
    "PARX": "PARX_1",
    "3127L": "3127L_0",
    "7880C": "7880C_0",
    "JON_A": "JON_A_0",
}
GRID_SOC_LEVELS = tuple(30.0 * index for index in range(1, 11))
MAX_DAILY_RECHARGES = 15
MAX_STATION_TO_TRIP_WAIT_MIN = 220.0
HORIZON_MIN = 1560.0


@dataclass(frozen=True)
class ProblemData:
    frame: pd.DataFrame
    trips: tuple[int, ...]
    adjacency: dict[Any, list[tuple[Any, float, float, str]]]
    start_min: dict[int, float]
    end_min: dict[int, float]
    trip_energy: dict[int, float]


@dataclass(frozen=True)
class DutyState:
    soc: float
    charges_used: int
    variable_cost: float
    route: tuple[Any, ...]
    charges: tuple[dict[str, float | str | int], ...]


def _total_minutes(value: Any) -> int:
    hour, minute = map(int, str(value).split(":"))
    return hour * 60 + minute


def _normal_token(value: Any) -> str | None:
    if pd.isna(value):
        return None
    result = str(value).strip()
    if not result:
        return None
    if result.endswith(".0"):
        result = result[:-2]
    return result


def _normal_ref(value: Any) -> str | None:
    result = _normal_token(value)
    if result is None:
        return None
    try:
        return str(int(float(result)))
    except (TypeError, ValueError):
        return result


def _base_station(value: Any) -> str:
    result = str(value).strip()
    if "_" in result:
        left, right = result.rsplit("_", 1)
        if right.isdigit():
            return left
    return result


def _ordered_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_reference_lookup(data_dir: Path):
    ref_frame = pd.read_csv(data_dir / "Ref_dict.csv")
    location_to_ref: dict[str, str] = {}
    for _, row in ref_frame.iterrows():
        location = _normal_token(row["Location"])
        reference = _normal_ref(row["Ref"])
        if location is not None and reference is not None:
            location_to_ref[location] = reference
    for location, reference in tuple(location_to_ref.items()):
        location_to_ref.setdefault(_base_station(location), reference)

    deadhead_frame = pd.read_csv(data_dir / "par_ref_dhd.csv")
    ref_pairs: dict[tuple[str, str], tuple[float, float]] = {}
    for _, row in deadhead_frame.iterrows():
        left = _normal_ref(row.iloc[0])
        right = _normal_ref(row.iloc[1])
        duration = pd.to_numeric(row.iloc[2], errors="coerce")
        energy = pd.to_numeric(row.iloc[3], errors="coerce")
        if (
            left is None
            or right is None
            or left == right
            or pd.isna(duration)
            or pd.isna(energy)
        ):
            continue
        key = _ordered_pair(left, right)
        candidate = (float(duration), float(energy))
        if key not in ref_pairs or candidate[0] < ref_pairs[key][0]:
            ref_pairs[key] = candidate
    known_refs = {reference for pair in ref_pairs for reference in pair}
    return location_to_ref, ref_pairs, known_refs


def build_problem(
    data_dir: Path,
    csv_name: str,
    *,
    max_trip2trip_min: float = 57.0,
    max_station_to_trip_wait_min: float = MAX_STATION_TO_TRIP_WAIT_MIN,
) -> ProblemData:
    """Reconstruct the exact issue20 restricted graph for one trip instance."""

    frame = pd.read_csv(data_dir / csv_name).rename(
        columns={
            "From1": "SL",
            "Start1": "ST",
            "End1": "ET",
            "To1": "EL",
            "Usage kWh": "Energy used",
        }
    ).reset_index(drop=True)
    required = {"SL", "ST", "ET", "EL", "Energy used", "Ordered_Trip_ID"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_name} is missing required columns: {sorted(missing)}")

    trips = tuple(range(len(frame)))
    start_loc = dict(zip(trips, frame["SL"]))
    end_loc = dict(zip(trips, frame["EL"]))
    start_min = {trip: _total_minutes(value) for trip, value in enumerate(frame["ST"])}
    end_min = {trip: _total_minutes(value) for trip, value in enumerate(frame["ET"])}
    trip_energy = {
        trip: float(value)
        for trip, value in enumerate(frame["Energy used"])
    }
    start_block = {trip: start_min[trip] + 1 for trip in trips}
    end_block = {trip: end_min[trip] + 1 for trip in trips}

    location_to_ref, ref_pairs, known_refs = _load_reference_lookup(data_dir)

    def resolve_ref(value: Any) -> str | None:
        raw = _normal_token(value)
        if raw is None:
            return None
        base = _base_station(raw)
        for candidate in (raw, base, _normal_ref(raw), _normal_ref(base)):
            if candidate in location_to_ref:
                return location_to_ref[candidate]
            if candidate in known_refs:
                return candidate
        return None

    def arc_from_to(left: Any, right: Any):
        left = _normal_token(left)
        right = _normal_token(right)
        if left == right:
            return 0, 0.0, 0.0
        left_ref = resolve_ref(left)
        right_ref = resolve_ref(right)
        if left_ref is None or right_ref is None:
            return None
        if left_ref == right_ref:
            return 0, 0.0, 0.0
        pair = ref_pairs.get(_ordered_pair(left_ref, right_ref))
        if pair is None:
            return None
        duration, energy = pair
        return int(math.ceil(duration)), duration, energy

    tau: dict[tuple[Any, Any], int] = {}
    tau_min: dict[tuple[Any, Any], float] = {}
    deadhead: dict[tuple[Any, Any], float] = {}

    def put(left: Any, right: Any, arc):
        if arc is not None:
            tau[left, right], tau_min[left, right], deadhead[left, right] = arc

    for trip in trips:
        put(DEPOT, trip, arc_from_to(DEPOT, start_loc[trip]))
        put(trip, DEPOT, arc_from_to(end_loc[trip], DEPOT))
        for following in trips:
            if trip != following:
                put(trip, following, arc_from_to(end_loc[trip], start_loc[following]))
        for station in STATIONS:
            put(trip, station, arc_from_to(end_loc[trip], station))
            put(station, trip, arc_from_to(station, start_loc[trip]))
    for station in STATIONS:
        put(DEPOT, station, arc_from_to(DEPOT, station))
        put(station, DEPOT, arc_from_to(station, DEPOT))

    adjacency = build_dag(
        T=list(trips),
        S_use=list(STATIONS),
        DEPOT=DEPOT,
        tau=tau,
        d=deadhead,
        st=start_block,
        et=end_block,
        sl=start_loc,
        el=end_loc,
        epsilon=trip_energy,
        TB_MIN=1,
        bar_t=int(HORIZON_MIN),
        tau_min=tau_min,
        st_min=start_min,
        et_min=end_min,
        max_trip2trip=max_trip2trip_min,
        max_trip2charge=61,
        max_charge2trip=int(max_station_to_trip_wait_min),
    )
    return ProblemData(
        frame=frame,
        trips=trips,
        adjacency=adjacency,
        start_min=start_min,
        end_min=end_min,
        trip_energy=trip_energy,
    )


def _match_station(raw: Any) -> str:
    raw_string = str(raw)
    if raw_string in STATION_NODE_BY_BASE:
        return raw_string
    for station in STATION_NODE_BY_BASE:
        if raw_string in station or station in raw_string:
            return station
    raise ValueError(f"Historical recharge station {raw!r} is not recognized")


def reconstruct_historical_duties(
    data_dir: Path,
    csv_name: str,
) -> list[dict[str, Any]]:
    """Reproduce the current CHEAT importer using only tracked derived CSVs."""

    instance = pd.read_csv(data_dir / csv_name)
    master = pd.read_csv(data_dir / "Par_VehicleDetails_Updated.csv")
    master["VehicleTask_Str"] = master["VehicleTask"].astype(str)
    ordered_to_local = {
        int(ordered): local
        for local, ordered in enumerate(instance["Ordered_Trip_ID"])
    }
    regular = master[master["Identifier"].eq("Regular")]
    task_by_ordered = regular.set_index("Ordered_Trip_ID")["VehicleTask"]
    buses = (
        instance["Ordered_Trip_ID"]
        .map(task_by_ordered)
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    deadhead_identifiers = {"Deadhead", "Pull-out", "Pull-in", "Prep-out", "Prep-in"}
    duties = []
    for bus in buses:
        nodes: list[Any] = [DEPOT]
        stations: list[str] = []
        charge_starts: list[int] = []
        charge_ends: list[int] = []
        charge_kwh: list[float] = []
        deadhead_kwh = 0.0
        for _, row in master[master["VehicleTask_Str"].eq(bus)].iterrows():
            identifier = row["Identifier"]
            if identifier in deadhead_identifiers and pd.notna(row["Usage kWh"]):
                deadhead_kwh += float(row["Usage kWh"])
            if identifier == "Regular" and pd.notna(row["Ordered_Trip_ID"]):
                nodes.append(ordered_to_local[int(row["Ordered_Trip_ID"])])
            elif identifier == "Recharge":
                station = STATION_NODE_BY_BASE[_match_station(row["From1"])]
                if nodes[-1] != station:
                    nodes.append(station)
                    stations.append(station)
                    charge_starts.append(_total_minutes(row["Start1"]))
                    charge_ends.append(_total_minutes(row["End1"]))
                    charge_kwh.append(
                        float(row["Recharge kWh"])
                        if pd.notna(row["Recharge kWh"])
                        else 0.0
                    )
        nodes.append(DEPOT)
        duties.append({
            "bus": bus,
            "route": nodes,
            "charging_stops": {
                "stations": stations,
                "cst": charge_starts,
                "cet": charge_ends,
                "kwh": charge_kwh,
            },
            "charging_activities": len(stations),
            "deadhead_kwh": deadhead_kwh,
            "type": "truck",
        })
    return duties


def charge_target_levels(
    *,
    arrival_soc: float,
    arrival_time: float,
    latest_departure: float,
    include_successor_boundary: bool,
    grid_levels: Iterable[float] = GRID_SOC_LEVELS,
) -> tuple[float, ...]:
    """Return current SOC targets plus an optional successor-boundary target.

    The counterfactual target is the maximum SOC reachable while still leaving
    in time for a *specific successor*.  Passing the overall 26-hour horizon as
    ``latest_departure`` normally produces 300 kWh, which is already on the
    current grid and therefore does not repair the observed 13303 duty.
    """

    levels = {float(level) for level in grid_levels}
    if include_successor_boundary:
        max_reachable = arrival_soc + max(0.0, latest_departure - arrival_time) * (
            CHARGE_RATE_KW / 60.0
        )
        levels.add(min(300.0, max_reachable))
    return tuple(sorted(level for level in levels if 0.0 < level <= 300.0))


def _pareto(states: Iterable[DutyState]) -> list[DutyState]:
    output: list[DutyState] = []
    for state in sorted(states, key=lambda item: (item.variable_cost, -item.soc, item.charges_used)):
        if any(
            other.soc >= state.soc - 1e-7
            and other.charges_used <= state.charges_used
            and other.variable_cost <= state.variable_cost + 1e-7
            for other in output
        ):
            continue
        output = [
            other
            for other in output
            if not (
                state.soc >= other.soc - 1e-7
                and state.charges_used <= other.charges_used
                and state.variable_cost <= other.variable_cost + 1e-7
            )
        ]
        output.append(state)
    return output


def realize_fixed_trip_order(
    problem: ProblemData,
    trip_order: Iterable[int],
    *,
    include_successor_boundary_target: bool = False,
    allow_station_waiting: bool = False,
    flat_charge_price: float = 0.0992,
) -> dict[str, Any]:
    """Find a current-resource charging realization for one fixed trip order."""

    trips = tuple(trip_order)
    if not trips:
        return {"feasible": False, "reason": "empty_trip_order"}
    arc = {
        (left, right): (minutes, energy, arc_type)
        for left, successors in problem.adjacency.items()
        for right, minutes, energy, arc_type in successors
    }
    first = trips[0]
    first_arc = arc.get((DEPOT, first))
    states: list[DutyState] = []
    if first_arc is not None:
        minutes, deadhead_kwh, _ = first_arc
        soc = 300.0 - deadhead_kwh - problem.trip_energy[first]
        if minutes <= problem.start_min[first] + 1e-6 and soc >= -1e-6:
            states = [DutyState(soc, 0, 0.0, (DEPOT, first), ())]

    failure: dict[str, Any] | None = None
    for previous, following in zip(trips, trips[1:]):
        states_before = states
        candidates: list[DutyState] = []
        direct = arc.get((previous, following))
        if direct is not None:
            minutes, deadhead_kwh, _ = direct
            for state in states:
                new_soc = state.soc - deadhead_kwh - problem.trip_energy[following]
                if (
                    problem.end_min[previous] + minutes
                    <= problem.start_min[following] + 1e-6
                    and new_soc >= -1e-6
                ):
                    candidates.append(DutyState(
                        new_soc,
                        state.charges_used,
                        state.variable_cost,
                        state.route + (following,),
                        state.charges,
                    ))

        for station in STATIONS:
            into_station = arc.get((previous, station))
            out_of_station = arc.get((station, following))
            if into_station is None or out_of_station is None:
                continue
            in_minutes, in_kwh, _ = into_station
            out_minutes, out_kwh, _ = out_of_station
            arrival_time = problem.end_min[previous] + in_minutes
            latest_departure = problem.start_min[following] - out_minutes
            for state in states:
                arrival_soc = state.soc - in_kwh
                if arrival_soc < -1e-6 or state.charges_used >= MAX_DAILY_RECHARGES:
                    continue
                targets = charge_target_levels(
                    arrival_soc=arrival_soc,
                    arrival_time=arrival_time,
                    latest_departure=latest_departure,
                    include_successor_boundary=include_successor_boundary_target,
                )
                for target in targets:
                    if target <= arrival_soc + 1e-6:
                        continue
                    charged_kwh = target - arrival_soc
                    charge_end = arrival_time + charged_kwh / (CHARGE_RATE_KW / 60.0)
                    departure = (
                        max(
                            charge_end,
                            problem.start_min[following] - MAX_STATION_TO_TRIP_WAIT_MIN,
                        )
                        if allow_station_waiting
                        else charge_end
                    )
                    if departure > latest_departure + 1e-6:
                        continue
                    if (
                        problem.start_min[following] - departure
                        > MAX_STATION_TO_TRIP_WAIT_MIN + 1e-6
                    ):
                        continue
                    new_soc = target - out_kwh - problem.trip_energy[following]
                    if new_soc < -1e-6:
                        continue
                    charge = {
                        "from_trip": previous,
                        "to_trip": following,
                        "station": station,
                        "arrival_min": arrival_time,
                        "charge_end_min": charge_end,
                        "departure_min": departure,
                        "charged_kwh": charged_kwh,
                        "target_soc_kwh": target,
                    }
                    candidates.append(DutyState(
                        new_soc,
                        state.charges_used + 1,
                        state.variable_cost
                        + CHARGE_START_COST
                        + flat_charge_price * charged_kwh,
                        state.route + (station, following),
                        state.charges + (charge,),
                    ))
        states = _pareto(candidates)
        if not states:
            direct_required = None
            if direct is not None:
                direct_required = direct[1] + problem.trip_energy[following]
            station_windows = []
            for station in STATIONS:
                into_station = arc.get((previous, station))
                out_of_station = arc.get((station, following))
                if into_station is None or out_of_station is None:
                    continue
                arrival_time = problem.end_min[previous] + into_station[0]
                latest_immediate_end = None
                for state in states_before:
                    arrival_soc = state.soc - into_station[1]
                    if arrival_soc < -1e-6:
                        continue
                    full_charge_end = arrival_time + (300.0 - arrival_soc) / (
                        CHARGE_RATE_KW / 60.0
                    )
                    latest_immediate_end = (
                        full_charge_end
                        if latest_immediate_end is None
                        else max(latest_immediate_end, full_charge_end)
                    )
                station_windows.append({
                    "station": station,
                    "arrival_min": arrival_time,
                    "earliest_departure_from_220_limit": (
                        problem.start_min[following] - MAX_STATION_TO_TRIP_WAIT_MIN
                    ),
                    "latest_departure_to_reach_trip": (
                        problem.start_min[following] - out_of_station[0]
                    ),
                    "latest_immediate_full_charge_end": latest_immediate_end,
                })
            max_soc = max((state.soc for state in states_before), default=None)
            failure = {
                "previous_trip": previous,
                "following_trip": following,
                "previous_end_min": problem.end_min[previous],
                "following_start_min": problem.start_min[following],
                "raw_gap_min": problem.start_min[following] - problem.end_min[previous],
                "max_soc_after_previous": max_soc,
                "direct_energy_required": direct_required,
                "direct_energy_shortfall": (
                    direct_required - max_soc
                    if direct_required is not None and max_soc is not None
                    else None
                ),
                "station_windows": station_windows,
            }
            break

    completed: list[DutyState] = []
    if states:
        last = trips[-1]
        direct_return = arc.get((last, DEPOT))
        if direct_return is not None:
            minutes, deadhead_kwh, _ = direct_return
            for state in states:
                if (
                    problem.end_min[last] + minutes <= HORIZON_MIN + 1e-6
                    and state.soc - deadhead_kwh >= -1e-6
                ):
                    completed.append(state)
        for station in STATIONS:
            into_station = arc.get((last, station))
            out_of_station = arc.get((station, DEPOT))
            if into_station is None or out_of_station is None:
                continue
            in_minutes, in_kwh, _ = into_station
            out_minutes, out_kwh, _ = out_of_station
            arrival_time = problem.end_min[last] + in_minutes
            latest_departure = HORIZON_MIN - out_minutes
            for state in states:
                arrival_soc = state.soc - in_kwh
                if arrival_soc < -1e-6 or state.charges_used >= MAX_DAILY_RECHARGES:
                    continue
                for target in charge_target_levels(
                    arrival_soc=arrival_soc,
                    arrival_time=arrival_time,
                    latest_departure=latest_departure,
                    include_successor_boundary=include_successor_boundary_target,
                ):
                    if target <= arrival_soc + 1e-6:
                        continue
                    charged_kwh = target - arrival_soc
                    charge_end = arrival_time + charged_kwh / (
                        CHARGE_RATE_KW / 60.0
                    )
                    if charge_end > latest_departure + 1e-6:
                        continue
                    if target - out_kwh < -1e-6:
                        continue
                    charge = {
                        "from_trip": last,
                        "to_trip": "DEPOT",
                        "station": station,
                        "arrival_min": arrival_time,
                        "charge_end_min": charge_end,
                        "departure_min": charge_end,
                        "charged_kwh": charged_kwh,
                        "target_soc_kwh": target,
                    }
                    completed.append(DutyState(
                        target - out_kwh,
                        state.charges_used + 1,
                        state.variable_cost
                        + CHARGE_START_COST
                        + flat_charge_price * charged_kwh,
                        state.route + (station,),
                        state.charges + (charge,),
                    ))
    if not completed:
        return {"feasible": False, "failure": failure}
    best = min(completed, key=lambda state: state.variable_cost)
    return {
        "feasible": True,
        "variable_cost": best.variable_cost,
        "total_route_cost": BUS_COST_KX + best.variable_cost,
        "charges_used": best.charges_used,
        "route": list(best.route + (DEPOT,)),
        "charges": list(best.charges),
    }


def realize_current_runner_trip_order(
    problem: ProblemData,
    trip_order: Iterable[int],
    *,
    flat_charge_price: float = 0.0992,
) -> dict[str, Any]:
    """Validate one known duty under the current runner's exact action set."""

    trips = tuple(trip_order)
    try:
        route = realize_fixed_trip_path(
            trips,
            adjacency=problem.adjacency,
            depot=DEPOT,
            stations=STATIONS,
            trip_start_min=problem.start_min,
            trip_end_min=problem.end_min,
            trip_energy_kwh=problem.trip_energy,
            battery_capacity_kwh=300.0,
            charge_rate_kw=CHARGE_RATE_KW,
            soc_charge_levels=GRID_SOC_LEVELS,
            horizon_min=HORIZON_MIN,
            max_daily_recharges=MAX_DAILY_RECHARGES,
            max_station_to_trip_wait_min=HORIZON_MIN,
            successor_boundary_soc_target=True,
            max_successor_charge_targets=64,
            station_waiting_unrestricted=True,
            charge_start_cost=CHARGE_START_COST,
            charging_cost=(
                lambda _station, _start, energy: flat_charge_price * float(energy)
            ),
            deadhead_cost_per_kwh=0.0,
        )
    except RouteRealizationError as error:
        return {
            "feasible": False,
            "reason": str(error),
            "failed_transition": error.failed_transition,
        }
    return {
        "feasible": True,
        "total_route_cost": BUS_COST_KX + _route_variable_cost(route, flat_charge_price),
        "charges_used": route["charging_activities"],
        "route": route,
    }


def _route_variable_cost(route: dict[str, Any], flat_charge_price: float) -> float:
    stops = route.get("charging_stops", {}) or {}
    return sum(float(value) for value in stops.get("kwh", ())) * flat_charge_price + (
        CHARGE_START_COST * len(stops.get("stations", ()))
    )


def _route_trips(route: dict[str, Any]) -> list[int]:
    return [node for node in route.get("route", ()) if isinstance(node, int)]


def _route_cost(
    route: dict[str, Any],
    hourly_prices: dict[int, float],
    station_prices: dict[str, dict[int, float]],
) -> float:
    return calculate_truck_route_cost_accurate(
        route,
        BUS_COST_KX,
        hourly_prices,
        charge_rate_kw=CHARGE_RATE_KW,
        station_hourly_prices=station_prices,
        charge_start_cost=CHARGE_START_COST,
    )


def audit_pool(pool_path: Path, data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    with pool_path.open() as handle:
        pool = json.load(handle)
    run_arguments = pool.get("run_arguments") or {}
    # Older saved pools predate the exposed option and are known to have used
    # the runner's hard-coded 57-minute value.
    max_trip2trip_min = float(run_arguments.get("max_trip2trip", 57.0))
    if (
        pool.get("max_trip2trip") is not None
        and float(pool["max_trip2trip"]) != max_trip2trip_min
    ):
        raise ValueError(
            "Saved top-level max_trip2trip disagrees with "
            "run_arguments.max_trip2trip"
        )
    csv_name = pool["csv_name"]
    instance_path = data_dir / csv_name
    if not instance_path.exists():
        raise FileNotFoundError(instance_path)
    expected_instance_hash = pool.get("instance_sha256")
    actual_instance_hash = _sha256(instance_path)
    if expected_instance_hash and actual_instance_hash != expected_instance_hash:
        raise ValueError(
            f"Instance SHA-256 mismatch for {instance_path}: "
            f"expected {expected_instance_hash}, found {actual_instance_hash}"
        )

    price_name = Path(pool["prices_csv"]).name
    price_path = data_dir / price_name
    expected_price_hash = pool.get("price_sha256")
    actual_price_hash = _sha256(price_path)
    if expected_price_hash and actual_price_hash != expected_price_hash:
        raise ValueError(
            f"Price SHA-256 mismatch for {price_path}: "
            f"expected {expected_price_hash}, found {actual_price_hash}"
        )
    station_prices = load_station_hourly_prices(
        price_path,
        tuple(STATION_NODE_BY_BASE),
    )
    hourly_prices = station_prices["PARX"]
    observed_prices = {
        float(price)
        for curve in station_prices.values()
        for price in curve.values()
    }
    if len(observed_prices) != 1:
        raise ValueError(
            "Known-duty feasibility audit currently requires a flat price file; "
            f"found {len(observed_prices)} distinct prices in {price_path.name}"
        )
    flat_charge_price = next(iter(observed_prices))

    trip_ids = tuple(pool["trip_ids"])
    routes = pool["routes"]
    route_trip_ids = [_route_trips(route) for route in routes]
    incidence = build_route_incidence(trip_ids, route_trip_ids)
    route_costs = [
        _route_cost(route, hourly_prices, station_prices)
        for route in routes
    ]
    master = solve_restricted_master_lp(
        trip_ids=trip_ids,
        route_incidence=incidence,
        route_costs=route_costs,
        artificial_penalty=BIG_M_PENALTY,
        method="highs-ds",
    )
    release_baseline_problem = build_problem(data_dir, csv_name)
    current_runner_problem = build_problem(
        data_dir,
        csv_name,
        max_trip2trip_min=max_trip2trip_min,
        max_station_to_trip_wait_min=HORIZON_MIN,
    )
    duties = reconstruct_historical_duties(data_dir, csv_name)

    rows = []
    for duty in duties:
        trips = _route_trips(duty)
        imported_cost = _route_cost(duty, hourly_prices, station_prices)
        dual_sum = sum(master.trip_duals[trip] for trip in trips)
        release_baseline = realize_fixed_trip_order(
            release_baseline_problem,
            trips,
            flat_charge_price=flat_charge_price,
        )
        successor_boundary = realize_fixed_trip_order(
            release_baseline_problem,
            trips,
            include_successor_boundary_target=True,
            flat_charge_price=flat_charge_price,
        )
        waiting = realize_fixed_trip_order(
            release_baseline_problem,
            trips,
            allow_station_waiting=True,
            flat_charge_price=flat_charge_price,
        )
        both = realize_fixed_trip_order(
            release_baseline_problem,
            trips,
            include_successor_boundary_target=True,
            allow_station_waiting=True,
            flat_charge_price=flat_charge_price,
        )
        current_runner = realize_current_runner_trip_order(
            current_runner_problem,
            trips,
            flat_charge_price=flat_charge_price,
        )
        baseline_cost = release_baseline.get("total_route_cost")
        current_runner_cost = current_runner.get("total_route_cost")
        rows.append({
            "vehicle_task": duty["bus"],
            "trip_count": len(trips),
            "historical_charge_count": duty["charging_activities"],
            "historical_import_cost_diagnostic": imported_cost,
            "dual_sum": dual_sum,
            "historical_import_reduced_cost_diagnostic": imported_cost - dual_sum,
            "release_baseline_fixed_trip_order_feasible": release_baseline["feasible"],
            "release_baseline_realization_cost": baseline_cost,
            "release_baseline_realization_reduced_cost": (
                baseline_cost - dual_sum if baseline_cost is not None else None
            ),
            "release_baseline_realization_charge_count": release_baseline.get("charges_used"),
            "release_baseline_failure": release_baseline.get("failure"),
            "counterfactual_successor_boundary_target_feasible": successor_boundary["feasible"],
            "counterfactual_successor_boundary_charges": successor_boundary.get("charges"),
            "counterfactual_waiting_feasible": waiting["feasible"],
            "counterfactual_boundary_and_waiting_feasible": both["feasible"],
            "current_runner_fixed_trip_order_feasible": current_runner["feasible"],
            "current_runner_realization_cost": current_runner_cost,
            "current_runner_realization_reduced_cost": (
                current_runner_cost - dual_sum
                if current_runner_cost is not None
                else None
            ),
            "current_runner_realization_charge_count": current_runner.get("charges_used"),
            "current_runner_failure": (
                None if current_runner["feasible"] else current_runner
            ),
        })

    return {
        "audit_version": 2,
        "pool_path": str(pool_path.resolve()),
        "instance_csv": csv_name,
        "instance_sha256": actual_instance_hash,
        "price_csv": price_name,
        "price_sha256": actual_price_hash,
        "saved_pool_mode": pool.get("mode"),
        "saved_pool_git": pool.get("git"),
        "current_runner_max_trip2trip_min": max_trip2trip_min,
        "dual_warning": (
            "The release does not save Gurobi duals. This audit reconstructs one "
            "optimal SciPy/HiGHS dual vector. Individual reduced costs can change "
            "under dual degeneracy; feasibility and aggregate-cost gaps do not."
        ),
        "restricted_master": {
            "objective": master.objective,
            "route_weight": master.route_weight,
            "artificial_total": master.artificial_total,
            "solver": master.backend.solver,
            "method": master.backend.method,
            "scipy_version": master.backend.scipy_version,
        },
        "historical_duties": rows,
        "aggregate": {
            "historical_duty_count": len(rows),
            "release_baseline_feasible_duty_count": sum(
                row["release_baseline_fixed_trip_order_feasible"] for row in rows
            ),
            "current_runner_feasible_duty_count": sum(
                row["current_runner_fixed_trip_order_feasible"] for row in rows
            ),
            "sum_historical_import_cost_diagnostic": sum(
                row["historical_import_cost_diagnostic"] for row in rows
            ),
            "sum_historical_import_reduced_cost_diagnostic": sum(
                row["historical_import_reduced_cost_diagnostic"] for row in rows
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        required=True,
        help="Saved routes_colgen_final_*.json file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Tracked derived data directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON destination; stdout is always concise and JSON-safe.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit_pool(args.pool, args.data_dir)
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(args.output.resolve())
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

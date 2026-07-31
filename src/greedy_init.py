"""
Greedy EVSP route construction for column-generation warm starts.

The goal is deliberately modest: create a small set of feasible truck routes
that covers every trip at least once, without using the historical
VehicleTask solution. These routes are only initial columns for the RMP; the
DP pricing routine is still responsible for generating improving columns.

Important implementation detail:
The pricing graph in run_ex_unicorn.py is keyed by trip ids and station-copy
nodes, not by raw start/end location names. For example:

    (DEPOT, trip_i), (trip_i, trip_j), (trip_i, station_h), (station_h, trip_j)

This module uses exactly those same keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _Arc:
    minutes: float
    kwh: float


@dataclass(frozen=True)
class _ChargePlan:
    station: Any
    cst: float
    cet: float
    kwh: float
    depart_soc: float
    first_arc_kwh: float
    second_arc_kwh: float

    @property
    def deadhead_kwh(self) -> float:
        return self.first_arc_kwh + self.second_arc_kwh


@dataclass(frozen=True)
class _ClosePlan:
    nodes: tuple[Any, ...]
    charge: _ChargePlan | None
    deadhead_kwh: float


def build_greedy_routes(
    *,
    T: list[int],
    S_use: list[Any],
    DEPOT: Any,
    tau: dict[tuple[Any, Any], int],
    tau_min: dict[tuple[Any, Any], float] | None,
    d: dict[tuple[Any, Any], float],
    st: dict[int, int],
    et: dict[int, int],
    st_min: dict[int, float] | None,
    et_min: dict[int, float] | None,
    sl: dict[int, str] | None = None,
    el: dict[int, str] | None = None,
    epsilon: dict[int, float] | None = None,
    G: float = 300.0,
    bar_t: int = 1560,
    TB_MIN: float = 1.0,
    CHARGE_RATE_KW: float = 300.0,
    max_trip2trip: float = 57.0,
    max_trip2charge: float = 61.0,
    max_charge2trip: float = 220.0,
    min_soc_fraction: float = 0.0,
    recharge_to_fraction: float = 1.0,
    max_daily_recharges: int = 15,
) -> list[dict[str, Any]]:
    """
    Build greedy routes until every trip in T is covered.

    Heuristic:
      1. Sort trips by scheduled start time.
      2. Start a new route at the earliest uncovered trip reachable from DEPOT.
      3. Extend to the earliest uncovered feasible next trip, directly if
         possible, otherwise with one charging stop inserted.
      4. Only accept an extension if the partial route can still return to the
         depot, either directly or through a charging stop.
      5. If no extension is possible, close the route and start another one.

    The function raises ValueError if a trip cannot be covered by even a
    singleton route. That is preferable to silently seeding an infeasible or
    misleading RMP.
    """

    if not T:
        return []

    epsilon = epsilon or {}
    horizon_min = float(bar_t) * float(TB_MIN)
    reserve_kwh = max(0.0, float(G) * float(min_soc_fraction))
    recharge_target_kwh = min(float(G), max(reserve_kwh, float(G) * float(recharge_to_fraction)))
    charge_rate_kw = float(CHARGE_RATE_KW)
    if charge_rate_kw <= 0:
        raise ValueError("CHARGE_RATE_KW must be positive for greedy initialization")

    def trip_start(i: int) -> float:
        if st_min is not None and i in st_min:
            return float(st_min[i])
        return float(st[i]) * float(TB_MIN)

    def trip_end(i: int) -> float:
        if et_min is not None and i in et_min:
            return float(et_min[i])
        return float(et[i]) * float(TB_MIN)

    def trip_energy(i: int) -> float:
        return float(epsilon.get(i, 0.0))

    def arc(u: Any, v: Any) -> _Arc | None:
        key = (u, v)
        if tau_min is not None and key in tau_min:
            minutes = float(tau_min[key])
        elif key in tau:
            minutes = float(tau[key]) * float(TB_MIN)
        else:
            return None
        return _Arc(minutes=minutes, kwh=float(d.get(key, 0.0)))

    def enough_soc(soc: float, required_kwh: float) -> bool:
        return soc - required_kwh >= reserve_kwh - 1e-6

    def charge_between(
        *,
        from_node: Any,
        station: Any,
        to_node: Any,
        current_time: float,
        current_soc: float,
        latest_arrival_to_node: float,
        required_after_second_arc: float,
        enforce_trip2charge_limit: bool,
        enforce_charge2trip_limit: bool,
    ) -> _ChargePlan | None:
        first = arc(from_node, station)
        second = arc(station, to_node)
        if first is None or second is None:
            return None
        if enforce_trip2charge_limit and first.minutes > max_trip2charge + 1e-6:
            return None

        arrival_station = current_time + first.minutes
        if arrival_station > horizon_min + 1e-6:
            return None

        soc_at_station = current_soc - first.kwh
        if soc_at_station < reserve_kwh - 1e-6:
            return None

        latest_depart_station = latest_arrival_to_node - second.minutes
        if latest_depart_station < arrival_station - 1e-6:
            return None

        earliest_depart_station = arrival_station
        if enforce_charge2trip_limit:
            earliest_depart_station = max(
                earliest_depart_station,
                latest_arrival_to_node - max_charge2trip,
            )

        min_depart_soc = second.kwh + required_after_second_arc + reserve_kwh
        if min_depart_soc > float(G) + 1e-6:
            return None

        min_target_by_energy = max(soc_at_station, min_depart_soc)
        min_target_by_time = soc_at_station + max(
            0.0,
            earliest_depart_station - arrival_station,
        ) * charge_rate_kw / 60.0
        target_lower = max(min_target_by_energy, min_target_by_time)

        max_target_by_time = soc_at_station + max(
            0.0,
            latest_depart_station - arrival_station,
        ) * charge_rate_kw / 60.0
        target_upper = min(float(G), max_target_by_time)

        if target_lower > target_upper + 1e-6:
            return None

        # Charge at least to the preferred target if time allows. This keeps
        # the greedy route from bouncing into stations repeatedly.
        target_soc = min(target_upper, max(target_lower, recharge_target_kwh))
        charge_kwh = max(0.0, target_soc - soc_at_station)

        # DP pricing does not create zero-charge station labels. Keep the warm
        # start in the same spirit: only insert a station if it actually charges.
        if charge_kwh <= 1e-6:
            return None

        charge_minutes = charge_kwh / charge_rate_kw * 60.0
        depart_station = arrival_station + charge_minutes

        if depart_station > latest_depart_station + 1e-6:
            return None
        if enforce_charge2trip_limit and latest_arrival_to_node - depart_station > max_charge2trip + 1e-6:
            return None
        if target_soc - second.kwh - required_after_second_arc < reserve_kwh - 1e-6:
            return None

        return _ChargePlan(
            station=station,
            cst=arrival_station,
            cet=depart_station,
            kwh=charge_kwh,
            depart_soc=target_soc,
            first_arc_kwh=first.kwh,
            second_arc_kwh=second.kwh,
        )

    def close_plan(last_trip: int, soc: float, time_now: float, charges_used: int) -> _ClosePlan | None:
        direct = arc(last_trip, DEPOT)
        if direct is not None:
            if time_now + direct.minutes <= horizon_min + 1e-6 and enough_soc(soc, direct.kwh):
                return _ClosePlan(nodes=(DEPOT,), charge=None, deadhead_kwh=direct.kwh)

        if charges_used >= max_daily_recharges:
            return None

        best: tuple[tuple[float, float, str], _ClosePlan] | None = None
        for station in S_use:
            plan = charge_between(
                from_node=last_trip,
                station=station,
                to_node=DEPOT,
                current_time=time_now,
                current_soc=soc,
                latest_arrival_to_node=horizon_min,
                required_after_second_arc=0.0,
                enforce_trip2charge_limit=True,
                enforce_charge2trip_limit=False,
            )
            if plan is None:
                continue
            candidate = _ClosePlan(
                nodes=(plan.station, DEPOT),
                charge=plan,
                deadhead_kwh=plan.deadhead_kwh,
            )
            score = (plan.kwh, plan.cet, str(plan.station))
            if best is None or score < best[0]:
                best = (score, candidate)

        return None if best is None else best[1]

    def direct_extension(last_trip: int, next_trip: int, soc: float, time_now: float) -> tuple[float, float] | None:
        a = arc(last_trip, next_trip)
        if a is None:
            return None
        gap = trip_start(next_trip) - time_now
        if gap < -1e-6 or gap > max_trip2trip + 1e-6:
            return None
        if time_now + a.minutes > trip_start(next_trip) + 1e-6:
            return None
        required = a.kwh + trip_energy(next_trip)
        if not enough_soc(soc, required):
            return None
        return soc - required, a.kwh

    def charged_extension(
        last_trip: int,
        next_trip: int,
        soc: float,
        time_now: float,
        charges_used: int,
    ) -> tuple[_ChargePlan, float] | None:
        if charges_used >= max_daily_recharges:
            return None

        best: tuple[tuple[float, float, str], _ChargePlan, float] | None = None
        for station in S_use:
            plan = charge_between(
                from_node=last_trip,
                station=station,
                to_node=next_trip,
                current_time=time_now,
                current_soc=soc,
                latest_arrival_to_node=trip_start(next_trip),
                required_after_second_arc=trip_energy(next_trip),
                enforce_trip2charge_limit=True,
                enforce_charge2trip_limit=True,
            )
            if plan is None:
                continue
            new_soc = plan.depart_soc - plan.second_arc_kwh - trip_energy(next_trip)
            score = (trip_start(next_trip), plan.kwh, str(plan.station))
            if best is None or score < best[0]:
                best = (score, plan, new_soc)

        return None if best is None else (best[1], best[2])

    sorted_trips = sorted(T, key=lambda i: (trip_start(i), i))
    uncovered = set(T)
    routes: list[dict[str, Any]] = []

    while uncovered:
        first_trip = None
        first_soc = None
        first_deadhead = None

        for i in sorted_trips:
            if i not in uncovered:
                continue
            depot_arc = arc(DEPOT, i)
            if depot_arc is None:
                continue
            if depot_arc.minutes > trip_start(i) + 1e-6:
                continue
            soc_after_i = float(G) - depot_arc.kwh - trip_energy(i)
            if soc_after_i < reserve_kwh - 1e-6:
                continue
            if close_plan(i, soc_after_i, trip_end(i), charges_used=0) is None:
                continue
            first_trip = i
            first_soc = soc_after_i
            first_deadhead = depot_arc.kwh
            break

        if first_trip is None:
            sample = sorted(uncovered, key=lambda i: (trip_start(i), i))[:10]
            raise ValueError(
                "[GREEDY] Could not build a feasible singleton route for the "
                f"earliest uncovered trips: {sample}. Check depot arcs, horizon, "
                "battery capacity, and charger reachability."
            )

        route_nodes: list[Any] = [DEPOT, first_trip]
        charging = {"stations": [], "cst": [], "cet": [], "kwh": []}
        total_deadhead = float(first_deadhead)
        soc = float(first_soc)
        last_trip = first_trip
        charges_used = 0
        uncovered.remove(first_trip)

        while uncovered:
            chosen: tuple[str, int, Any, float, float] | None = None

            for candidate in sorted_trips:
                if candidate not in uncovered:
                    continue
                if trip_start(candidate) < trip_end(last_trip) - 1e-6:
                    continue

                direct = direct_extension(last_trip, candidate, soc, trip_end(last_trip))
                if direct is not None:
                    next_soc, direct_deadhead = direct
                    if close_plan(candidate, next_soc, trip_end(candidate), charges_used) is not None:
                        chosen = ("direct", candidate, None, next_soc, direct_deadhead)
                        break

                charged = charged_extension(last_trip, candidate, soc, trip_end(last_trip), charges_used)
                if charged is not None:
                    plan, next_soc = charged
                    if close_plan(candidate, next_soc, trip_end(candidate), charges_used + 1) is not None:
                        chosen = ("charge", candidate, plan, next_soc, plan.deadhead_kwh)
                        break

            if chosen is None:
                break

            kind, next_trip, plan, next_soc, added_deadhead = chosen
            if kind == "charge":
                assert plan is not None
                route_nodes.append(plan.station)
                charging["stations"].append(plan.station)
                charging["cst"].append(plan.cst)
                charging["cet"].append(plan.cet)
                charging["kwh"].append(plan.kwh)
                charges_used += 1

            route_nodes.append(next_trip)
            total_deadhead += added_deadhead
            soc = next_soc
            last_trip = next_trip
            uncovered.remove(next_trip)

        final_close = close_plan(last_trip, soc, trip_end(last_trip), charges_used)
        if final_close is None:
            raise ValueError(
                f"[GREEDY] Internal error: route ending at trip {last_trip} "
                "was not closeable despite extension checks."
            )

        if final_close.charge is not None:
            plan = final_close.charge
            route_nodes.append(plan.station)
            charging["stations"].append(plan.station)
            charging["cst"].append(plan.cst)
            charging["cet"].append(plan.cet)
            charging["kwh"].append(plan.kwh)
            charges_used += 1
        route_nodes.extend(n for n in final_close.nodes if n != route_nodes[-1])
        total_deadhead += final_close.deadhead_kwh

        trip_count = sum(1 for n in route_nodes if isinstance(n, int))
        routes.append(
            {
                "route": route_nodes,
                "charging_stops": charging,
                "charging_activities": len(charging["stations"]),
                "type": "truck",
                "deadhead_kwh": total_deadhead,
                "_rc": 0.0,
                "desc": f"[GREEDY] route {len(routes) + 1}: {trip_count} trips",
            }
        )

    covered_once = {
        n
        for route in routes
        for n in route["route"]
        if isinstance(n, int)
    }
    missing = sorted(set(T) - covered_once, key=lambda i: (trip_start(i), i))
    if missing:
        raise ValueError(f"[GREEDY] Failed to cover {len(missing)} trips: {missing[:20]}")

    return routes

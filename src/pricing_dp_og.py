"""

pricing_dp.py  –  SPPRC Labeling Algorithm for the EVSP Pricing Problem

=========================================================================



Replaces `solve_pricing_fast` (Gurobi MIP) with a forward‐labeling Dynamic

Programming algorithm inspired by Desaulniers et al.'s methodology for

Shortest Path Problems with Resource Constraints (SPPRC).



Network nodes

─────────────

  • DEPOT (source & sink, e.g. "PARX_0")

  • Trips   – integer indices from T

  • Charging station copies – strings from S_use  (e.g. "3127L_0")



Arcs (type → variable name in MIP)

───────────────────────────────────

  wA        :  DEPOT  → Trip  |  DEPOT → Station

  x         :  Trip   → Trip

  y         :  Trip   → Station

  z         :  Station → Trip

  wOmega    :  Trip   → DEPOT |  Station → DEPOT



Label resources

───────────────

  (reduced_cost, time, soc, path)



The algorithm performs forward extension from the DEPOT source, propagates

labels through trips and charging stations, applies dominance pruning at

every node, and collects the K‑best negative‑reduced‑cost paths that

return to DEPOT.



Design decisions

────────────────

  • Charging at a station is discretised into a set of candidate SOC

    levels to charge *up to* (e.g. 25 %, 50 %, 75 %, 100 % of G).

    For each candidate level the charging duration, cost (time‑of‑day

    price × energy), and resulting departure SOC are computed exactly.

  • Dominance: L1 dominates L2 at the same node when

        rc1 ≤ rc2  AND  time1 ≤ time2  AND  soc1 ≥ soc2

    (lower cost, earlier departure, more remaining energy).

  • The DAG adjacency list is pre‑computed once per CG iteration so that

    inner DP loops do no dictionary look‑ups for arc feasibility.



Public API

──────────

  build_dag(...)           →  adjacency dict

  solve_pricing_dp(...)    →  list of route dicts  (drop‑in for solve_pricing_fast)

"""



from __future__ import annotations



import math

import heapq

import bisect

import time

from dataclasses import dataclass, field

from typing import Any, Mapping



# ──────────────────────────────────────────────────────────────────────

#  LABEL DATA STRUCTURE

# ──────────────────────────────────────────────────────────────────────



@dataclass(slots=True, eq=False)

class Label:

    """

    A single DP label (partial path state) in the SPPRC.



    Attributes

    ----------

    rc   : float   – accumulated reduced cost  (lower is better)

    time : float   – current time in *minutes* (continuous)

    soc  : float   – current state‑of‑charge in kWh

    node : object  – current node  (int for trips, str for stations/depot)

    path : tuple   – sequence of visited nodes so far

    trips_visited : frozenset – set of trip‑indices on this path

                                (for elementarity – each trip at most once)

    charging_stops : tuple  – list of (station, cst_min, cet_min, kwh_charged)

    deadhead_kwh : float  – total deadhead energy consumed on this path

    """

    rc:    float

    time:  float

    soc:   float

    node:  object

    path:  tuple

    trips_visited: frozenset = field(default_factory=frozenset)

    charging_stops: tuple = field(default_factory=tuple)

    deadhead_kwh: float = 0.0

    # False once dominated/evicted from its node pool; stale heap entries
    # check this flag in O(1) instead of scanning the pool.
    alive: bool = True



    # For heap ordering: sort by reduced cost (most negative first)

    def __lt__(self, other):

        return self.rc < other.rc



@dataclass(frozen=True, slots=True)
class PricingRunStats:
    """Search effort from one DP pricing call."""

    queue_order: str
    labels_expanded: int
    completed_routes: int
    negative_completed: int
    best_reduced_cost: float
    label_cap_evictions: int
    exhaustive: bool
    timed_out: bool
    elapsed_s: float


_VALID_QUEUE_ORDERS = frozenset({"time", "reduced_cost", "reduced_cost_bound"})


def _validate_queue_order(queue_order: str) -> str:
    if queue_order not in _VALID_QUEUE_ORDERS:
        raise ValueError(
            "queue_order must be one of "
            f"{sorted(_VALID_QUEUE_ORDERS)}, found {queue_order!r}"
        )
    return queue_order


def _make_label_priority_key(queue_order: str, remaining_dual_bound=None):
    """Bind the configured two-part priority used for queues and cap retention."""

    order = _validate_queue_order(queue_order)
    if order == "time":
        def _priority(label: Label):
            return (label.time, label.rc)
    elif order == "reduced_cost":
        def _priority(label: Label):
            return (label.rc, label.time)
    else:
        if remaining_dual_bound is None:
            raise ValueError(
                "queue_order='reduced_cost_bound' requires a remaining-dual bound"
            )

        def _priority(label: Label):
            return (
                label.rc - remaining_dual_bound(label.time),
                label.rc,
            )
    return _priority


def _make_label_queue_entry(queue_order: str, remaining_dual_bound=None):
    """Bind one validated heap-entry function for an entire pricing call."""

    priority = _make_label_priority_key(queue_order, remaining_dual_bound)

    def _entry(label: Label, unique_id: int):
        first, second = priority(label)
        return (first, second, unique_id, label)

    return _entry


def _cap_label_pool(
    labels: list[Label],
    *,
    max_labels: int,
    priority_key,
) -> tuple[list[Label], int]:
    """Apply hysteretic cap retention and report nondominated label evictions."""

    if max_labels <= 0:
        raise ValueError("max_labels must be positive")
    threshold = max_labels + max(50, max_labels // 10)
    if len(labels) <= threshold:
        return labels, 0

    ordered = sorted(labels, key=priority_key)
    evicted = ordered[max_labels:]
    for label in evicted:
        label.alive = False
    return ordered[:max_labels], len(evicted)


def _build_remaining_dual_bound(
    alpha: dict[int, float],
    T: list[int],
    trip_start_min: dict[int, float],
    trip_end_min: dict[int, float],
):
    """Return an optimistic bound on positive dual still collectible after a time.

    The main term is a weighted-interval-scheduling suffix DP over fixed trip
    intervals.  It deliberately ignores travel, energy, gap, and elementarity
    restrictions, so it can overestimate but cannot underestimate what a
    feasible continuation may collect.  Zero-duration intervals are
    accumulated separately: allowing all of them is loose, but avoids the
    self-indexing ambiguity that ``end == start`` creates in the WIS recurrence.
    """

    intervals: list[tuple[float, float, float]] = []
    point_events: list[tuple[float, float]] = []
    for trip in T:
        weight = max(0.0, float(alpha.get(trip, 0.0)))
        if weight <= 0.0:
            continue
        start = float(trip_start_min[trip])
        end = float(trip_end_min[trip])
        if not (math.isfinite(start) and math.isfinite(end) and math.isfinite(weight)):
            raise ValueError(f"trip {trip!r} has non-finite pricing-bound data")
        if end < start - 1e-6:
            raise ValueError(f"trip {trip!r} ends before it starts")
        if end <= start + 1e-6:
            point_events.append((start, weight))
        else:
            intervals.append((start, end, weight))

    intervals.sort(key=lambda item: (item[0], item[1]))
    starts = [item[0] for item in intervals]
    suffix_best = [0.0] * (len(intervals) + 1)
    for index in range(len(intervals) - 1, -1, -1):
        _, end, weight = intervals[index]
        following = bisect.bisect_left(starts, end, lo=index + 1)
        suffix_best[index] = max(
            suffix_best[index + 1],
            weight + suffix_best[following],
        )

    point_events.sort(key=lambda item: item[0])
    point_starts = [item[0] for item in point_events]
    point_suffix = [0.0] * (len(point_events) + 1)
    for index in range(len(point_events) - 1, -1, -1):
        point_suffix[index] = point_suffix[index + 1] + point_events[index][1]

    def _bound(time_min: float) -> float:
        # Pricing accepts arrivals up to 1e-6 minutes after a nominal start.
        # Querying from time-tolerance keeps this ordering bound optimistic too.
        threshold = float(time_min) - 1e-6
        interval_index = bisect.bisect_left(starts, threshold)
        point_index = bisect.bisect_left(point_starts, threshold)
        return suffix_best[interval_index] + point_suffix[point_index]

    return _bound


def _dedupe_soc_levels(levels, *, G: float, tol: float = 1e-6) -> list[float]:
    """Return sorted in-capacity SOC targets with near-duplicates collapsed."""

    valid = sorted(
        float(level)
        for level in levels
        if math.isfinite(float(level)) and 0.0 < float(level) <= G + tol
    )
    deduped: list[float] = []
    for level in valid:
        level = min(float(G), level)
        if not deduped or level - deduped[-1] > tol:
            deduped.append(level)
        elif level > deduped[-1]:
            deduped[-1] = level
    return deduped


def _successor_boundary_soc_levels(
    *,
    base_levels,
    successor_latest_departures,
    arrival_soc: float,
    arrival_time_min: float,
    G: float,
    charge_rate_kw: float,
    max_successor_targets: int,
) -> list[float]:
    """Augment a SOC grid with charge-to-the-latest-departure boundaries.

    Each station-to-trip successor supplies its own latest station departure
    (trip start minus deadhead time).  Charging exactly to the maximum SOC
    reachable at that boundary captures feasible partial charges that a fixed
    absolute SOC grid can miss.
    """

    if max_successor_targets <= 0:
        raise ValueError("max_successor_targets must be positive")

    boundary_levels = []
    if charge_rate_kw > 0.0:
        charge_kwh_per_minute = charge_rate_kw / 60.0
        deadlines = successor_latest_departures
        first = bisect.bisect_left(deadlines, arrival_time_min - 1e-6)
        full_charge_deadline = arrival_time_min + max(
            0.0,
            G - arrival_soc,
        ) / charge_kwh_per_minute
        first_full = bisect.bisect_left(deadlines, full_charge_deadline, lo=first)
        relevant_deadlines = list(deadlines[first:first_full])
        if first_full < len(deadlines):
            # Every later successor yields the same capacity target.
            relevant_deadlines.append(deadlines[first_full])
        for latest_departure in relevant_deadlines:
            reachable = arrival_soc + max(
                0.0,
                latest_departure - arrival_time_min,
            ) * charge_kwh_per_minute
            boundary_levels.append(min(float(G), reachable))

    boundary_levels = _dedupe_soc_levels(boundary_levels, G=G)
    if len(boundary_levels) > max_successor_targets:
        if max_successor_targets == 1:
            boundary_levels = [boundary_levels[-1]]
        else:
            last = len(boundary_levels) - 1
            selected = {
                round(index * last / (max_successor_targets - 1))
                for index in range(max_successor_targets)
            }
            boundary_levels = [boundary_levels[index] for index in sorted(selected)]

    return _dedupe_soc_levels([*base_levels, *boundary_levels], G=G)





# ──────────────────────────────────────────────────────────────────────

#  DAG PRE‑COMPUTATION

# ──────────────────────────────────────────────────────────────────────



def build_dag(

    T: list[int],

    S_use: list[str],

    DEPOT: str,

    tau: dict,          # (u,v) → travel time in time‑blocks

    d: dict,            # (u,v) → deadhead energy kWh

    st: dict,           # trip i → start time‑block

    et: dict,           # trip i → end time‑block

    sl: dict,           # trip i → start location name

    el: dict,           # trip i → end location name

    epsilon: dict,      # trip i → energy consumed by trip i (kWh)

    TB_MIN: int,        # minutes per time‑block

    bar_t: int,         # horizon (max time‑block)

    tau_min=None, st_min=None, et_min=None,
    *,

    max_trip2trip: int = 57,       # max minute gap for trip→trip

    max_trip2charge: int = 61,     # max minute gap for trip→station

    max_charge2trip: int = 220,    # max minute gap for station→trip

) -> dict:

    """

    Pre‑compute a directed adjacency list for the pricing sub‑problem.



    Returns

    -------

    adj : dict[node] → list of (successor, travel_time_min, deadhead_kwh, arc_type)



    arc_type is one of:

        'depot_trip', 'depot_station',

        'trip_trip', 'trip_station', 'trip_depot',

        'station_trip', 'station_depot'

    """



    adj: dict[Any, list] = {DEPOT: [], **{i: [] for i in T}, **{h: [] for h in S_use}}



    def _tb_to_min(tb: int) -> float:

        """Convert a 1‑based time‑block index to minutes from midnight."""

        return (tb - 1) * TB_MIN



    def _arc_min(u, v) -> float | None:

        """Return travel time in minutes for arc (u,v), or None if arc doesn't exist."""

        if (u, v) not in tau:

            return None

        return tau[(u, v)] * TB_MIN



    def _arc_kwh(u, v) -> float:

        return d.get((u, v), 0.0)



    # ── DEPOT → Trip ──

    for i in T:

        if (DEPOT, i) in tau:

            travel_min = tau[(DEPOT, i)] * TB_MIN

            dh_kwh = _arc_kwh(DEPOT, i)

            adj[DEPOT].append((i, travel_min, dh_kwh, 'depot_trip'))



    # ── DEPOT → Station ──

    for h in S_use:

        if (DEPOT, h) in tau:

            travel_min = tau[(DEPOT, h)] * TB_MIN

            dh_kwh = _arc_kwh(DEPOT, h)

            adj[DEPOT].append((h, travel_min, dh_kwh, 'depot_station'))



    # ── Trip → Trip ──

    for i in T:

        for j in T:

            if i == j:

                continue

            if (i, j) not in tau:

                continue

            # Feasibility: vehicle finishes trip i, travels, arrives before trip j starts

            # if et[i] + tau[(i, j)] > st[j]:

            #     continue

            ### fix
            if tau_min is not None and st_min is not None and et_min is not None:
                # Exact minute check: prevents block rounding from severing valid arcs
                if et_min[i] + tau_min.get((i, j), 0) > st_min[j]:
                    continue
            else:
                # Fallback to block check if minute data not provided
                if et[i] + tau[(i, j)] > st[j]:
                    continue



            # Pruning: don't allow excessively long idle gaps

            if st_min is not None and et_min is not None:
                trip_gap_min = st_min[j] - et_min[i]
            else:
                trip_gap_min = (st[j] - et[i]) * TB_MIN

            if trip_gap_min > max_trip2trip:

                continue

            travel_min = tau[(i, j)] * TB_MIN

            dh_kwh = _arc_kwh(i, j)

            adj[i].append((j, travel_min, dh_kwh, 'trip_trip'))



    # ── Trip → Station ──

    for i in T:

        for h in S_use:

            if (i, h) not in tau:

                continue

            # if et[i] + tau[(i, h)] > bar_t:

            #     continue
            ### fix

            if tau_min is not None and et_min is not None:
                # Must reach station before end of day (bar_t * TB_MIN)
                if et_min[i] + tau_min.get((i, h), 0) > (bar_t * TB_MIN):
                    continue
                trip2charge_gap_min = tau_min.get((i, h), tau[(i, h)] * TB_MIN)
            else:
                if et[i] + tau[(i, h)] > bar_t:
                    continue
                trip2charge_gap_min = tau[(i, h)] * TB_MIN

            if trip2charge_gap_min > max_trip2charge:
                continue


            travel_min = tau[(i, h)] * TB_MIN

            dh_kwh = _arc_kwh(i, h)

            adj[i].append((h, travel_min, dh_kwh, 'trip_station'))



    # ── Station → Trip ──

    for h in S_use:

        for i in T:

            if (h, i) not in tau:

                continue

            # Station departure must allow reaching trip start

            # if tau[(h, i)] > st[i]:

            #     continue
            ### fix
            if tau_min is not None and st_min is not None:
                # departure from station at time 0, must reach trip i by st_min[i]
                if tau_min.get((h, i), 0) > st_min[i]:
                    continue
            else:
                if tau[(h, i)] > st[i]:
                    continue

            travel_min = tau[(h, i)] * TB_MIN

            dh_kwh = _arc_kwh(h, i)

            adj[h].append((i, travel_min, dh_kwh, 'station_trip'))



    # ── Trip → DEPOT ──

    for i in T:

        if (i, DEPOT) in tau:

            travel_min = tau[(i, DEPOT)] * TB_MIN

            if et_min is not None:
                if et_min[i] + travel_min > bar_t * TB_MIN:
                    continue
            else:
                if et[i] + tau[(i, DEPOT)] > bar_t:
                    continue

            dh_kwh = _arc_kwh(i, DEPOT)

            adj[i].append((DEPOT, travel_min, dh_kwh, 'trip_depot'))



    # ── Station → DEPOT ──

    for h in S_use:

        if (h, DEPOT) in tau:

            travel_min = tau[(h, DEPOT)] * TB_MIN

            dh_kwh = _arc_kwh(h, DEPOT)

            adj[h].append((DEPOT, travel_min, dh_kwh, 'station_depot'))



    return adj





# ──────────────────────────────────────────────────────────────────────

#  CHARGING COST HELPERS

# ──────────────────────────────────────────────────────────────────────



def _compute_charging_cost(

    start_min: float,

    energy_kwh: float,

    charge_rate_kw: float,

    hourly_prices: dict,

    charge_cost_premium: float,

) -> float:

    """

    Compute the time‑of‑day electricity cost for charging `energy_kwh`

    starting at `start_min` minutes from midnight.



    The charging duration is  energy_kwh / charge_rate_kw  hours.

    We split that duration across hour boundaries and price each

    segment at the corresponding hourly rate.



    Parameters

    ----------

    start_min        : float – charge start (minutes from midnight)

    energy_kwh       : float – total energy to charge

    charge_rate_kw   : float – charger power (kW)

    hourly_prices    : dict  – {hour_index: $/kWh}

    charge_cost_premium : float – multiplicative mark‑up



    Returns

    -------

    cost : float – total electricity cost ($)

    """

    if energy_kwh <= 1e-9:

        return 0.0



    duration_hours = energy_kwh / charge_rate_kw

    duration_min = duration_hours * 60.0



    end_min = start_min + duration_min

    max_hour = max(hourly_prices.keys()) if hourly_prices else 23



    cost = 0.0

    cursor_min = start_min



    while cursor_min < end_min - 1e-9:

        # Which hour‑bucket does `cursor_min` fall in?

        hour_idx = int(cursor_min // 60)

        hour_idx_clamped = min(hour_idx, max_hour)



        # End of this hour bucket (in minutes)

        next_hour_min = (hour_idx + 1) * 60.0



        # Segment end: whichever comes first – next hour or charging end

        seg_end = min(next_hour_min, end_min)

        seg_duration_hours = (seg_end - cursor_min) / 60.0



        # Energy charged in this segment (at constant power)

        seg_kwh = charge_rate_kw * seg_duration_hours



        # Price for this segment

        price = hourly_prices.get(hour_idx_clamped, hourly_prices.get(hour_idx_clamped % 24, 0.0))

        cost += price * seg_kwh * charge_cost_premium



        cursor_min = seg_end



    return cost





def _generate_charge_options(

    arrival_soc: float,

    arrival_time_min: float,

    departure_deadline_min: float,

    G: float,

    charge_rate_kw: float,

    hourly_prices: dict,

    charge_cost_premium: float,

    soc_levels: list[float] | None = None,

    charge_start_cost=0.0   #NEW

) -> list[tuple[float, float, float, float]]:

    """

    Enumerate discrete charging options at a station.



    For each target SOC level that is above `arrival_soc`, compute:

      – energy to charge  (target − arrival_soc)

      – duration in minutes

      – departure time

      – charging cost



    If the departure time exceeds the deadline, skip that option.



    Parameters

    ----------

    arrival_soc         : float – SOC on arrival (kWh)

    arrival_time_min    : float – time of arrival (minutes)

    departure_deadline_min : float – latest permissible departure (minutes)

    G                   : float – battery capacity (kWh)

    charge_rate_kw      : float – charger power (kW)

    hourly_prices       : dict  – {hour: $/kWh}

    charge_cost_premium : float

    soc_levels          : list[float] | None – target SOC levels to consider

                          (absolute kWh).  Default: [25%, 50%, 75%, 100%] of G.



    Returns

    -------

    options : list of (departure_soc, departure_time_min, charge_cost, energy_kwh)

              Sorted by energy ascending (cheapest / fastest first).

    """

    if soc_levels is None:

        # Default: try charging to 25 %, 50 %, 75 %, 100 % of capacity

        soc_levels = [0.25 * G, 0.50 * G, 0.75 * G, G]



    options = []



    # Option 0: No charging at all (just pass through with some dwell)

    # We still allow a "zero‑charge" option that adds a small dwell time

    # (the minimum dwell is handled by the caller via travel times).

    # Actually, no‑charge means the station is simply not visited,

    # so we omit it here – the caller decides whether to extend

    # through a station or directly to the next trip.



    for target_soc in soc_levels:

        if target_soc <= arrival_soc + 1e-6:

            continue  # nothing to charge

        if target_soc > G + 1e-6:

            continue  # can't exceed capacity



        energy_kwh = target_soc - arrival_soc

        duration_min = (energy_kwh / charge_rate_kw) * 60.0



        departure_time = arrival_time_min + duration_min

        if departure_time > departure_deadline_min + 1e-6:

            continue  # would miss the deadline



        charge_cost = _compute_charging_cost(

            start_min=arrival_time_min,

            energy_kwh=energy_kwh,

            charge_rate_kw=charge_rate_kw,

            hourly_prices=hourly_prices,

            charge_cost_premium=charge_cost_premium,

        ) + charge_start_cost #New



        options.append((target_soc, departure_time, charge_cost, energy_kwh))



    return options





# ──────────────────────────────────────────────────────────────────────

#  DOMINANCE CHECK

# ──────────────────────────────────────────────────────────────────────


def _dominates(
    other: Label,
    label: Label,
    tol: float = 1e-4,
    *,
    station_pool: bool = False,
    station_waiting_unrestricted: bool = False,
) -> bool:
    # At a station, an earlier departure is not always more useful: the
    # restricted graph also imposes a maximum station-to-trip wait.  Without
    # an explicit waiting action, two different station completion times can
    # therefore have different feasible successor sets and are incomparable.
    time_dominates = (
        abs(other.time - label.time) <= tol
        if station_pool and not station_waiting_unrestricted
        else other.time <= label.time + tol
    )
    return (
        other.rc <= label.rc + tol
        and time_dominates
        and other.soc >= label.soc - tol
        and len(other.charging_stops) <= len(label.charging_stops)
        and len(other.trips_visited) >= len(label.trips_visited)
    )


def _is_dominated(
    label: Label,
    label_pool: list[Label],
    *,
    station_pool: bool = False,
    station_waiting_unrestricted: bool = False,
) -> bool:
    """
    Check whether `label` is dominated.
    Because our network moves forward in time, cycles are impossible.
    We do NOT need to check trips_visited for elementarity!

    Recharge count and the minimum completed-trip requirement are resources.
    A label with more charging stops has fewer remaining extensions, while a
    label with fewer visited trips may be unable to satisfy
    MIN_TRIPS_PER_ROUTE at the depot. Both must participate in dominance.
    """

    for other in label_pool:
        if _dominates(
            other,
            label,
            station_pool=station_pool,
            station_waiting_unrestricted=station_waiting_unrestricted,
        ):
            return True
    return False





def _prune_dominated(
    label_pool: list[Label],
    *,
    station_pool: bool = False,
    station_waiting_unrestricted: bool = False,
) -> list[Label]:

    """

    Remove mutually dominated labels from a pool.

    Returns a new list with only non‑dominated labels.

    """

    if len(label_pool) <= 1:

        return label_pool



    kept: list[Label] = []

    for lab in label_pool:

        if not _is_dominated(
            lab,
            kept,
            station_pool=station_pool,
            station_waiting_unrestricted=station_waiting_unrestricted,
        ):

            # Also remove any labels in `kept` that the new label dominates

            kept = [
                k for k in kept
                if not _dominates(
                    lab,
                    k,
                    station_pool=station_pool,
                    station_waiting_unrestricted=station_waiting_unrestricted,
                )
            ]

            kept.append(lab)

    return kept





# ──────────────────────────────────────────────────────────────────────

#  MAIN SPPRC LABELING ALGORITHM

# ──────────────────────────────────────────────────────────────────────



def solve_pricing_dp(

    # ── Duals from master ──

    alpha: dict[int, float],

    beta: dict | None = None,       # station‑capacity duals (future use)

    gamma: dict | None = None,      # discharge duals (future use)



    # ── Problem data ──

    T: list[int] = None,

    S_use: list[str] = None,

    DEPOT: str = "PARX_0",

    adj: dict | None = None,        # pre‑built DAG  (from build_dag)



    # ── Arc / trip data ──

    tau: dict | None = None,

    d: dict | None = None,

    st: dict | None = None,

    et: dict | None = None,

    sl: dict | None = None,

    el: dict | None = None,

    epsilon: dict | None = None,

    tau_min=None,
    st_min=None,
    et_min=None,



    # ── Parameters ──

    G: float = 300.0,

    TB_MIN: int = 1,

    bar_t: int = 1560,

    bus_cost: float = 1e5,

    charge_rate_kw: float = 300.0,


    hourly_prices: dict | None = None,
    charge_cost_premium: float = 0.0,
    travel_cost_factor: float = 1.0,
    station_hourly_prices: dict | None = None,   # NEW
    charge_start_cost: float = 0.0,              # NEW

    RC_EPSILON: float = 1.0,

    EARLY_EXIT_FLOOR_S: float = 5.0,   # don't early-exit before this many seconds


    # ── Algorithm tuning ──

    K_BEST: int = 50,

    MAX_LABELS_PER_NODE: int = 200,

    soc_charge_levels: list[float] | None = None,

    successor_charge_targets: bool = False,

    max_successor_charge_targets: int = 64,

    MIN_TRIPS_PER_ROUTE: int = 1,

    MAX_DAILY_RECHARGES: int = 8,

    max_trip2trip: int = 57,

    max_trip2charge: int = 61,

    max_charge2trip: int = 220,

    time_limit: float | None = None,

    queue_order: str = "time",

    existing_trip_set_costs: Mapping[frozenset, float] | None = None,

    existing_cost_epsilon: float = 1e-6,

    return_stats: bool = False,

) -> (
    tuple[list[dict], bool]
    | tuple[list[dict], bool, PricingRunStats]
):

    """

    Solve the EVSP pricing sub‑problem via forward‑labeling SPPRC.



    This is a drop‑in replacement for ``solve_pricing_fast``.

    It returns a list of route dictionaries (with ``_rc`` field)

    sorted by reduced cost, compatible with ``R_truck`` append logic.



    Parameters

    ----------

    alpha : dict[int, float]

        Dual values for trip‑coverage constraints  (α_i).

    beta, gamma : dict (optional, for future extensions)

    T     : list of trip indices

    S_use : list of charging station copy names

    DEPOT : depot node name

    adj   : pre‑computed DAG adjacency list (from ``build_dag``).

            If None, it will be built from tau/d/st/et/etc.

    G     : battery capacity (kWh)

    TB_MIN : minutes per time‑block

    bar_t : time horizon in time‑blocks

    bus_cost : fixed cost of using one vehicle

    charge_rate_kw : charger power (kW)

    hourly_prices : {hour_index: $/kWh}

    charge_cost_premium : cost multiplier on charging

    travel_cost_factor : cost per kWh of deadhead travel

    RC_EPSILON : threshold – only return routes with rc < −RC_EPSILON

    K_BEST : max number of routes to return

    MAX_LABELS_PER_NODE : cap on un‑dominated labels kept per node

    soc_charge_levels : list of target SOC levels (kWh) to try at stations.

                        Default = [25 %, 50 %, 75 %, 100 % of G].

    MIN_TRIPS_PER_ROUTE : minimum trips a route must cover

    MAX_DAILY_RECHARGES : max number of charging stops per route

    queue_order : ``"time"`` preserves the historical chronological heap;

                  ``"reduced_cost"`` expands the most negative label first;

                  ``"reduced_cost_bound"`` also subtracts an optimistic

                  weighted-interval bound on future positive trip duals.

    existing_trip_set_costs : optional best master cost for each trip-incidence

                              pattern already present in the restricted master.

                              When supplied, dominated existing patterns are

                              skipped *before* applying the K-best output cap.

    return_stats : if true, append a ``PricingRunStats`` object to the

                   otherwise unchanged ``(routes, timed_out)`` return tuple.



    Returns

    -------

    routes : list[dict]

        Each dict has keys compatible with ``R_truck``:

          route, charging_stops, charging_activities, type, deadhead_kwh,

          _rc, desc

    """

    if hourly_prices is None:

        hourly_prices = {}

    if beta is None:

        beta = {}

    if gamma is None:

        gamma = {}

    if max_successor_charge_targets <= 0:
        raise ValueError("max_successor_charge_targets must be positive")
    if MAX_LABELS_PER_NODE <= 0:
        raise ValueError("MAX_LABELS_PER_NODE must be positive")
    if existing_cost_epsilon < 0:
        raise ValueError("existing_cost_epsilon must be nonnegative")
    incumbent_cost_by_trip_set = {
        frozenset(key): float(value)
        for key, value in (existing_trip_set_costs or {}).items()
    }
    if any(not math.isfinite(value) for value in incumbent_cost_by_trip_set.values()):
        raise ValueError("existing_trip_set_costs must contain finite costs")



    # ── Build DAG if not provided ──

    if adj is None:

        adj = build_dag(

            T=T, S_use=S_use, DEPOT=DEPOT,

            tau=tau, d=d, st=st, et=et, sl=sl, el=el,

            epsilon=epsilon, TB_MIN=TB_MIN, bar_t=bar_t,

            tau_min=tau_min, st_min=st_min, et_min=et_min,

            max_trip2trip=max_trip2trip,

            max_trip2charge=max_trip2charge,

            max_charge2trip=max_charge2trip,

        )



    if soc_charge_levels is None:

        soc_charge_levels = [0.25 * G, 0.50 * G, 0.75 * G, G]



    # Resolve the price curve per station ONCE. Station nodes are copy names
    # ("2190L_0") while the price table is keyed by base names ("2190L");
    # without stripping the copy suffix every lookup silently falls back to
    # the depot's hourly_prices curve.
    def _base_station(name) -> str:
        s = str(name)
        if "_" in s:
            left, right = s.rsplit("_", 1)
            if right.isdigit():
                return left
        return s

    station_prices = {
        h: (station_hourly_prices or {}).get(_base_station(h), hourly_prices)
        for h in (S_use or [])
    }



    horizon_min = bar_t * TB_MIN    # total horizon in minutes



    # ── Helper: convert time‑block to minutes ──

    def tb2min(tb: int) -> float:

        return (tb - 1) * TB_MIN



    # Trip time windows in minutes

    if st_min is not None:
        trip_start_min = {i: st_min[i] for i in T}
    else:
        trip_start_min = {i: tb2min(st[i]) for i in T}

    if et_min is not None:
        trip_end_min = {i: et_min[i] for i in T}
    else:
        trip_end_min = {i: tb2min(et[i]) for i in T}

    remaining_dual_bound = None
    if queue_order == "reduced_cost_bound":
        remaining_dual_bound = _build_remaining_dual_bound(
            alpha,
            T,
            trip_start_min,
            trip_end_min,
        )
    label_priority = _make_label_priority_key(queue_order, remaining_dual_bound)
    queue_entry = _make_label_queue_entry(queue_order, remaining_dual_bound)

    # When the station-to-trip wait cap spans the entire horizon, an earlier
    # station label has every temporal successor available to a later one.
    # Ordinary earlier-time dominance is then safe and substantially tighter.
    station_waiting_unrestricted = max_charge2trip >= horizon_min - 1e-6

    station_successor_deadlines = {}
    if successor_charge_targets:
        for station in S_use:
            station_successor_deadlines[station] = sorted({
                float(trip_start_min[successor]) - float(travel_min)
                for successor, travel_min, _, arc_type in adj.get(station, ())
                if arc_type == "station_trip"
            })



    # ──────────────────────────────────────────────────────────────

    # STEP 1:  Initialise source label at DEPOT

    # ──────────────────────────────────────────────────────────────

    # The vehicle departs the depot at time 0 with full SOC.

    # The fixed bus_cost is added to the reduced cost at initialisation

    # so that it appears in every route.



    source_label = Label(

        rc=bus_cost,

        time=0.0,          # earliest possible departure (minute 0)

        soc=float(G),      # full battery at the depot, no initial charging cost

        node=DEPOT,

        path=(DEPOT,),

        trips_visited=frozenset(),

        charging_stops=(),

        deadhead_kwh=0.0,

    )



    # ──────────────────────────────────────────────────────────────

    # STEP 2:  Label‑setting / label‑correcting forward pass

    # ──────────────────────────────────────────────────────────────

    #

    # The priority queue can use the historical chronological order or

    # reduced-cost-first order.  This is a label-correcting approach since

    # the graph may have negative arc costs (due to dual subtraction).

    #

    # node_labels[v] stores the list of non‑dominated labels at node v.



    node_labels: dict[Any, list[Label]] = {DEPOT: [source_label]}

    for i in T:

        node_labels[i] = []

    for h in S_use:

        node_labels[h] = []



    # Keep only the cheapest negative completion for each trip-incidence
    # pattern.  The current master has trip-cover rows only, so two routes with
    # the same trip set differ only through their objective cost.  Retaining
    # every charging/path realization here used to consume large amounts of
    # memory before a post-search de-duplication pass.

    best_negative_by_trip_set: dict[frozenset, Label] = {}



    # Priority queue entries are constructed centrally so every push follows

    # the configured ordering.

    pq: list[tuple[float, float, int, Label]] = []

    _uid = 0

    heapq.heappush(pq, queue_entry(source_label, _uid))

    _uid += 1



    # ── Main loop ──
    dp_start_time = time.time()
    hit_timelimit = False
    early_exit_kbest = False
    labels_expanded = 0
    n_completed = 0
    n_neg_completed = 0   # running count of completed labels with rc < -RC_EPSILON
    label_cap_evictions = 0

    while pq:

        _, _, _, label = heapq.heappop(pq)

        elapsed = time.time() - dp_start_time
        if time_limit and elapsed > time_limit:
            print(f"[DP-PRICER] Time limit ({time_limit}s) reached! Halting DP early.")
            hit_timelimit = True
            break


        ## this might be the problem.
        # if (n_neg_completed >= K_BEST) and (elapsed >= EARLY_EXIT_FLOOR_S):
        #     print(f"[DP-PRICER] Early exit: {n_neg_completed} neg-RC routes found in {elapsed:.1f}s "
        #           f"(K_BEST={K_BEST}, floor={EARLY_EXIT_FLOOR_S}s)")
        #     early_exit_kbest = True
        #     break



        cur = label.node



        # Skip if this label was dominated/evicted after being enqueued.
        # O(1) flag check instead of a linear (deep-equality) pool scan.
        if not label.alive:
            continue

        labels_expanded += 1



        # ── Extend to each successor ──

        successors = adj.get(cur, [])



        for (succ, travel_min, dh_kwh, arc_type) in successors:



            # ────────────────────────────────────────────────

            # A)  Extension to a TRIP node

            # ────────────────────────────────────────────────

            if arc_type in ('depot_trip', 'trip_trip', 'station_trip'):

                # succ is a trip index (int)

                trip_idx = succ



                # Elementarity: skip if already visited

                if trip_idx in label.trips_visited:

                    continue



                # ── Time feasibility ──

                # Earliest arrival at trip start location

                earliest_arrival = label.time + travel_min

                # Trip has a fixed start time; we must arrive by then

                trip_start = trip_start_min[trip_idx]

                if arc_type == 'station_trip':
                    charge2trip_gap_min = trip_start - label.time
                    if charge2trip_gap_min > max_charge2trip:
                        continue

                if earliest_arrival > trip_start + 1e-6:

                    continue  # too late



                # Actual departure from trip = trip end time

                trip_end = trip_end_min[trip_idx]



                # ── SOC feasibility ──

                # Energy consumed: deadhead travel + trip service

                energy_needed = dh_kwh + epsilon[trip_idx]

                new_soc = label.soc - energy_needed

                if new_soc < -1e-6:

                    continue  # insufficient battery



                # ── Reduced cost ──

                # Subtract dual α_i for covering trip i

                dual_val = alpha.get(trip_idx, 0.0)

                arc_travel_cost = dh_kwh * travel_cost_factor

                new_rc = label.rc + arc_travel_cost - dual_val



                # ── Create new label ──

                new_label = Label(

                    rc=new_rc,

                    time=trip_end,

                    soc=new_soc,

                    node=trip_idx,

                    path=label.path + (trip_idx,),

                    trips_visited=label.trips_visited | {trip_idx},

                    charging_stops=label.charging_stops,

                    deadhead_kwh=label.deadhead_kwh + dh_kwh,

                )



                # ── Dominance check & insertion ──

                # pool = node_labels[trip_idx]

                # if not _is_dominated(new_label, pool):

                #     # Remove labels dominated by the new one

                #     node_labels[trip_idx] = [

                #         lb for lb in pool

                #         if not (new_label.rc   <= lb.rc   and

                #                 new_label.time <= lb.time and

                #                 #new_label.soc  >= lb.soc  and

                #                 new_label.trips_visited <= lb.trips_visited and

                #                 (new_label.rc < lb.rc or new_label.time < lb.time or

                #                  #new_label.soc > lb.soc or

                #                  new_label.trips_visited < lb.trips_visited))

                #     ]

                #     node_labels[trip_idx].append(new_label)



                #     # Cap the label pool size (keep best by rc)

                #     if len(node_labels[trip_idx]) > MAX_LABELS_PER_NODE:

                #         node_labels[trip_idx].sort(key=lambda lb: lb.rc)

                #         node_labels[trip_idx] = node_labels[trip_idx][:MAX_LABELS_PER_NODE]



                #     heapq.heappush(pq, (new_label.rc, _uid, new_label))

                #    _uid += 1
                pool = node_labels[trip_idx]
                if not _is_dominated(new_label, pool):
                    # Prune labels that are strictly worse than our new label,
                    # marking them dead so their stale heap entries skip in O(1)
                    kept = []
                    for lb in pool:
                        if _dominates(new_label, lb):
                            lb.alive = False
                        else:
                            kept.append(lb)
                    kept.append(new_label)

                    # Cap with hysteresis (instead of sorting the whole pool on
                    # every insertion once at the cap)
                    kept, evicted = _cap_label_pool(
                        kept,
                        max_labels=MAX_LABELS_PER_NODE,
                        priority_key=label_priority,
                    )
                    label_cap_evictions += evicted
                    node_labels[trip_idx] = kept

                    if new_label.alive:
                        heapq.heappush(
                            pq,
                            queue_entry(new_label, _uid),
                        )
                        _uid += 1



            # ────────────────────────────────────────────────

            # B)  Extension to a CHARGING STATION node

            # ────────────────────────────────────────────────

            elif arc_type in ('depot_station', 'trip_station'):

                station = succ  # string



                # Limit number of charging stops

                if len(label.charging_stops) >= MAX_DAILY_RECHARGES:

                    continue



                # ── Time feasibility ──

                arrival_time = label.time + travel_min

                if arrival_time > horizon_min:

                    continue



                # ── SOC at station arrival (after deadhead) ──

                soc_at_station = label.soc - dh_kwh

                if soc_at_station < -1e-6:

                    continue



                # ── Enumerate charging options ──

                # Departure deadline: we need to leave early enough to

                # reach at least one more trip or return to depot.

                # Use horizon as upper bound.

                departure_deadline = horizon_min



                charge_levels = soc_charge_levels
                if successor_charge_targets:
                    charge_levels = _successor_boundary_soc_levels(
                        base_levels=soc_charge_levels,
                        successor_latest_departures=station_successor_deadlines.get(
                            station,
                            (),
                        ),
                        arrival_soc=soc_at_station,
                        arrival_time_min=arrival_time,
                        G=G,
                        charge_rate_kw=charge_rate_kw,
                        max_successor_targets=max_successor_charge_targets,
                    )

                charge_options = _generate_charge_options(
                    arrival_soc=soc_at_station,
                    arrival_time_min=arrival_time,
                    departure_deadline_min=departure_deadline,
                    G=G,
                    charge_rate_kw=charge_rate_kw,
                    hourly_prices=station_prices.get(station, hourly_prices),
                    charge_cost_premium=charge_cost_premium,
                    soc_levels=charge_levels,
                    charge_start_cost=charge_start_cost,   # NEW
                )
                # charge_options = _generate_charge_options(

                #     arrival_soc=soc_at_station,

                #     arrival_time_min=arrival_time,

                #     departure_deadline_min=departure_deadline,

                #     G=G,

                #     charge_rate_kw=charge_rate_kw,

                #     hourly_prices=station_hourly_prices.get(station, hourly_prices),

                #     charge_cost_premium=charge_cost_premium,

                #     soc_levels=soc_charge_levels,

                # )



                if not charge_options:

                    continue



                arc_travel_cost = dh_kwh * travel_cost_factor



                for (dep_soc, dep_time, charge_cost, energy_kwh) in charge_options:



                    # ── Reduced cost ──

                    new_rc = label.rc + arc_travel_cost + charge_cost

                    # Future: subtract β_{h,t} dual for station capacity



                    # Record charging stop details

                    new_charging_stops = label.charging_stops + (

                        (station, arrival_time, dep_time, energy_kwh),

                    )



                    new_label = Label(

                        rc=new_rc,

                        time=dep_time,

                        soc=dep_soc,

                        node=station,

                        path=label.path + (station,),

                        trips_visited=label.trips_visited,

                        charging_stops=new_charging_stops,

                        deadhead_kwh=label.deadhead_kwh + dh_kwh,

                    )



                    # ── Dominance check ──

                    # pool = node_labels[station]

                    # if not _is_dominated(new_label, pool):

                    #     node_labels[station] = [

                    #         lb for lb in pool

                    #         if not (new_label.rc   <= lb.rc   and

                    #                 new_label.time <= lb.time and

                    #                 new_label.soc  >= lb.soc  and

                    #                 new_label.trips_visited <= lb.trips_visited and

                    #                 (new_label.rc < lb.rc or new_label.time < lb.time or

                    #                  new_label.soc > lb.soc or

                    #                  new_label.trips_visited < lb.trips_visited))

                    #     ]

                    #     node_labels[station].append(new_label)



                    #     if len(node_labels[station]) > MAX_LABELS_PER_NODE:

                    #         node_labels[station].sort(key=lambda lb: lb.rc)

                    #         node_labels[station] = node_labels[station][:MAX_LABELS_PER_NODE]



                    #     heapq.heappush(pq, (new_label.rc, _uid, new_label))

                    #     _uid += 1

                    pool = node_labels[station]
                    if not _is_dominated(
                        new_label,
                        pool,
                        station_pool=True,
                        station_waiting_unrestricted=station_waiting_unrestricted,
                    ):
                        # Prune labels that are strictly worse than our new label,
                        # marking them dead so their stale heap entries skip in O(1)
                        kept = []
                        for lb in pool:
                            if _dominates(
                                new_label,
                                lb,
                                station_pool=True,
                                station_waiting_unrestricted=station_waiting_unrestricted,
                            ):
                                lb.alive = False
                            else:
                                kept.append(lb)
                        kept.append(new_label)

                        kept, evicted = _cap_label_pool(
                            kept,
                            max_labels=MAX_LABELS_PER_NODE,
                            priority_key=label_priority,
                        )
                        label_cap_evictions += evicted
                        node_labels[station] = kept

                        if new_label.alive:
                            heapq.heappush(
                                pq,
                                queue_entry(new_label, _uid),
                            )
                            _uid += 1



            # ────────────────────────────────────────────────

            # C)  Extension to DEPOT (sink – route completion)

            # ────────────────────────────────────────────────

            elif arc_type in ('trip_depot', 'station_depot'):



                # ── Time feasibility ──

                arrival_depot = label.time + travel_min

                if arrival_depot > horizon_min + 1e-6:

                    continue



                # ── SOC feasibility ──

                new_soc = label.soc - dh_kwh

                if new_soc < -1e-6:

                    continue  # can't make it back



                # ── Minimum trips requirement ──

                if len(label.trips_visited) < MIN_TRIPS_PER_ROUTE:

                    continue



                # ── Reduced cost ──

                arc_travel_cost = dh_kwh * travel_cost_factor

                final_rc = label.rc + arc_travel_cost



                completed_label = Label(

                    rc=final_rc,

                    time=arrival_depot,

                    soc=new_soc,

                    node=DEPOT,

                    path=label.path + (DEPOT,),

                    trips_visited=label.trips_visited,

                    charging_stops=label.charging_stops,

                    deadhead_kwh=label.deadhead_kwh + dh_kwh,

                )



                n_completed += 1

                if completed_label.rc < -RC_EPSILON:
                    n_neg_completed += 1
                    trip_set = completed_label.trips_visited
                    incumbent = best_negative_by_trip_set.get(trip_set)
                    if incumbent is None or completed_label.rc < incumbent.rc:
                        best_negative_by_trip_set[trip_set] = completed_label

    # ──────────────────────────────────────────────────────────────

    # STEP 3:  Collect K‑best routes with negative reduced cost

    # ──────────────────────────────────────────────────────────────



    # Negative routes are already unique by trip-incidence pattern.

    neg_routes = list(best_negative_by_trip_set.values())



    # Sort by reduced cost (most negative first)

    neg_routes.sort(key=lambda lb: lb.rc)



    unique_routes: list[Label] = []

    for lab in neg_routes:

        key = lab.trips_visited

        incumbent_cost = incumbent_cost_by_trip_set.get(key)
        # The DP reduced cost is current master cost minus the trip-cover
        # duals.  Recovering the cost this way lets us discard an equal or more
        # expensive realization already represented in the master while still
        # admitting a genuinely cheaper route with the same incidence.
        candidate_master_cost = lab.rc + sum(float(alpha.get(i, 0.0)) for i in key)
        if (
            incumbent_cost is not None
            and candidate_master_cost >= incumbent_cost - existing_cost_epsilon
        ):
            continue

        unique_routes.append(lab)

        if len(unique_routes) >= K_BEST:

            break



    # ──────────────────────────────────────────────────────────────

    # STEP 4:  Format output as R_truck‑compatible dictionaries

    # ──────────────────────────────────────────────────────────────



    results: list[dict] = []



    for lab in unique_routes:

        # Build route node list (same format as MIP extractor)

        route_nodes = list(lab.path)



        # Build charging_stops sub‑dict

        cs_stations = []

        cs_cst = []

        cs_cet = []

        cs_kwh = []

        for (station, cst_min, cet_min, kwh) in lab.charging_stops:

            cs_stations.append(station)

            cs_cst.append(cst_min)

            cs_cet.append(cet_min)

            cs_kwh.append(kwh)



        # Build description string

        desc_parts = []

        for node in route_nodes:

            part = str(node)

            # If it's a trip, show which trip

            if isinstance(node, int):

                part = f"T{node}"

            # If it's a station with charging info, annotate

            for (stn, cst_m, cet_m, kwh) in lab.charging_stops:

                if stn == node:

                    h_s, m_s = divmod(int(cst_m), 60)

                    h_e, m_e = divmod(int(cet_m), 60)

                    part += f" [Charge {kwh:.1f}kWh @ {h_s:02d}:{m_s:02d}-{h_e:02d}:{m_e:02d}]"

                    break

            desc_parts.append(part)



        route_dict = {

            "route": route_nodes,

            "charging_stops": {

                "stations": cs_stations,

                "cst": cs_cst,

                "cet": cs_cet,

                "kwh": cs_kwh,

            },

            "charging_activities": len(cs_stations),

            "type": "truck",

            "deadhead_kwh": lab.deadhead_kwh,

            "_rc": lab.rc,

            "desc": " -> ".join(desc_parts),

        }

        results.append(route_dict)



    pricing_stats = PricingRunStats(
        queue_order=queue_order,
        labels_expanded=labels_expanded,
        completed_routes=n_completed,
        negative_completed=n_neg_completed,
        best_reduced_cost=(neg_routes[0].rc if neg_routes else float("inf")),
        label_cap_evictions=label_cap_evictions,
        exhaustive=(
            not hit_timelimit
            and not early_exit_kbest
            and label_cap_evictions == 0
        ),
        timed_out=hit_timelimit,
        elapsed_s=time.time() - dp_start_time,
    )
    if return_stats:
        return results, hit_timelimit, pricing_stats
    return results, hit_timelimit




# ──────────────────────────────────────────────────────────────────────

#  CONVENIENCE WRAPPER  (matches solve_pricing_fast call signature)

# ──────────────────────────────────────────────────────────────────────



def make_dp_pricer(

    T, S_use, DEPOT, tau, d, st, et, sl, el, epsilon,

    G, TB_MIN, bar_t,


    bus_cost, charge_rate_kw,

    hourly_prices, charge_cost_premium, travel_cost_factor,

    RC_EPSILON, K_BEST,
    st_min=None, et_min=None, tau_min=None, #New


    MAX_LABELS_PER_NODE=200,

    soc_charge_levels=None,

    successor_charge_targets=False,

    max_successor_charge_targets=64,

    MIN_TRIPS_PER_ROUTE=1,

    MAX_DAILY_RECHARGES=4,

    max_trip2trip=57,

    max_trip2charge=61,

    max_charge2trip=220,

    station_hourly_prices=None,   # NEW
    charge_start_cost=0.0,        # NEW

    queue_order="time",

    adj=None,


):

    """

    Factory that returns a callable with the same interface as

    ``solve_pricing_fast(alpha, beta, gamma, mode, ...)``.



    Usage in run_ex_unicorn.py

    ---------------------------

    >>> from pricing_dp_og import make_dp_pricer

    >>> dp_price = make_dp_pricer(T=T, S_use=S_use, ...)

    >>> # Inside CG loop:

    >>> new_routes = dp_price(alpha, beta_dual, gamma_dual)



    The returned function pre‑builds the DAG once on the first call and

    caches it for subsequent calls (the graph structure doesn't change

    across CG iterations – only the duals change).

    """

    default_queue_order = _validate_queue_order(queue_order)

    if max_successor_charge_targets <= 0:
        raise ValueError("max_successor_charge_targets must be positive")



    # ── Pre-build the DAG (topology is fixed) ──
    # Matching initialization can construct this same graph before the pricer.
    # Accept it here so large instances do not pay the O(|T|^2) build twice.
    if adj is None:
        adj = build_dag(

            T=T, S_use=S_use, DEPOT=DEPOT,

            tau=tau, d=d, st=st, et=et, sl=sl, el=el,

            epsilon=epsilon, TB_MIN=TB_MIN, bar_t=bar_t,

            tau_min=tau_min, st_min=st_min, et_min=et_min, #

            max_trip2trip=max_trip2trip,

            max_trip2charge=max_trip2charge,

            max_charge2trip=max_charge2trip,

        )



    _n_arcs = sum(len(v) for v in adj.values())

    print(f"[DP-PRICER] DAG built: {len(adj)} nodes, {_n_arcs} arcs")



    def _solve(
        alpha,
        beta=None,
        gamma=None,
        mode=1,
        num_fast_cols=None,
        time_limit=None,
        max_labels=None,
        existing_trip_set_costs=None,
        **kwargs,
    ):

        """

        Solve the pricing problem via DP.



        `mode` and `num_fast_cols` are accepted for API compatibility but

        ignored. `time_limit` (seconds) and `max_labels` (labels per node)

        override the factory defaults per call, so one pricer (and one DAG)

        can serve every escalation tier of every CG iteration.

        Returns a list of route dicts (same as solve_pricing_fast output,

        but without the Gurobi model – just the routes).

        """
        routes, hit_timelimit, pricing_stats = solve_pricing_dp(

            alpha=alpha,

            beta=beta,

            gamma=gamma,

            T=T,

            S_use=S_use,

            DEPOT=DEPOT,

            adj=adj,

            tau=tau, d=d,

            st=st, et=et, sl=sl, el=el,

            epsilon=epsilon,

            tau_min=tau_min,
            st_min=st_min,
            et_min=et_min,

            G=G,

            TB_MIN=TB_MIN,

            bar_t=bar_t,

            bus_cost=bus_cost,

            charge_rate_kw=charge_rate_kw,

            hourly_prices=hourly_prices,

            charge_cost_premium=charge_cost_premium,

            travel_cost_factor=travel_cost_factor,

            RC_EPSILON=RC_EPSILON,

            K_BEST=K_BEST,

            MAX_LABELS_PER_NODE=int(max_labels) if max_labels else MAX_LABELS_PER_NODE,

            soc_charge_levels=soc_charge_levels,

            successor_charge_targets=successor_charge_targets,

            max_successor_charge_targets=max_successor_charge_targets,

            MIN_TRIPS_PER_ROUTE=MIN_TRIPS_PER_ROUTE,

            MAX_DAILY_RECHARGES=MAX_DAILY_RECHARGES,
            max_trip2trip=max_trip2trip,
            max_trip2charge=max_trip2charge,
            max_charge2trip=max_charge2trip,
            time_limit=time_limit,

            station_hourly_prices=station_hourly_prices,   # NEW
            charge_start_cost=charge_start_cost,           # NEW

            queue_order=default_queue_order,
            existing_trip_set_costs=existing_trip_set_costs,
            return_stats=True,

        )

        _solve.last_stats = pricing_stats


        n_neg = len(routes)
        best_overall_rc = pricing_stats.best_reduced_cost
        print(
            f"[DP-PRICER] Found {n_neg} negative-RC routes "
            f"(best_rc={best_overall_rc:.2f}, hit_timelimit={hit_timelimit}, "
            f"label_cap_evictions={pricing_stats.label_cap_evictions}, "
            f"exhaustive={pricing_stats.exhaustive})"
        )

        return routes, best_overall_rc, hit_timelimit


    _solve.last_stats = None
    _solve.queue_order = default_queue_order
    _solve.successor_charge_targets = bool(successor_charge_targets)
    _solve.max_successor_charge_targets = int(max_successor_charge_targets)
    _solve.adjacency = adj
    return _solve





# ──────────────────────────────────────────────────────────────────────

#  INTEGRATION EXAMPLE  (run_ex_unicorn.py CG loop)

# ──────────────────────────────────────────────────────────────────────

#

#  # ── Before the CG loop (after building S_use, tau, d, etc.) ──

#  from pricing_dp_og import make_dp_pricer

#

#  S_use = sorted(...)   # same S_use as in build_pricing

#

#  dp_price = make_dp_pricer(

#      T=T, S_use=S_use, DEPOT=DEPOT,

#      tau=tau, d=d, st=st, et=et, sl=sl, el=el, epsilon=epsilon,

#      G=G, TB_MIN=TB_MIN, bar_t=bar_t,

#      bus_cost=bus_cost,

#      charge_rate_kw=CHARGE_RATE_KW,

#      hourly_prices=hourly_prices,

#      charge_cost_premium=charge_cost_premium,

#      travel_cost_factor=TRAVEL_COST_FACTOR,

#      RC_EPSILON=RC_EPSILON,

#      K_BEST=K_BEST,

#      MIN_TRIPS_PER_ROUTE=MIN_TRIPS_PER_ROUTE,

#      MAX_DAILY_RECHARGES=MAX_DAILY_RECHARGES,

#  )

#

#  # ── Inside the CG loop (replaces solve_pricing_fast call) ──

#  new_trucks = dp_price(alpha, beta_dual, gamma_dual)

#

#  # new_trucks is already a list of route dicts with _rc field.

#  # Filter duplicates and add to master as before:

#  for route in new_trucks:

#      if _route_key(route) not in seen_keys_existing:

#          R_truck.append(route)

#          cost = calculate_truck_route_cost(route, bus_cost, hourly_prices)

#          col = Column()

#          for node in route["route"]:

#              if isinstance(node, int):

#                  col.addTerms(1.0, trip_cov[node])

#          idx = len(R_truck) - 1

#          a[idx] = rmp.addVar(obj=cost, lb=0, ub=1, ...)

#      ...

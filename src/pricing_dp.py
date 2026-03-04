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

from dataclasses import dataclass, field

from typing import Any



# ──────────────────────────────────────────────────────────────────────

#  LABEL DATA STRUCTURE

# ──────────────────────────────────────────────────────────────────────



@dataclass(slots=True)

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



    # For heap ordering: sort by reduced cost (most negative first)

    def __lt__(self, other):

        return self.rc < other.rc





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

    *,

    max_trip2trip: int = 15,       # max time‑block gap for trip→trip

    max_trip2charge: int = 60,     # max time‑block gap for trip→station

    max_charge2trip: int = 60,     # max time‑block gap for station→trip

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

            if et[i] + tau[(i, j)] > st[j]:

                continue

            # Pruning: don't allow excessively long idle gaps

            if (st[j] - et[i]) > max_trip2trip:

                continue

            travel_min = tau[(i, j)] * TB_MIN

            dh_kwh = _arc_kwh(i, j)

            adj[i].append((j, travel_min, dh_kwh, 'trip_trip'))



    # ── Trip → Station ──

    for i in T:

        for h in S_use:

            if (i, h) not in tau:

                continue

            if et[i] + tau[(i, h)] > bar_t:

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

            if tau[(h, i)] > st[i]:

                continue

            travel_min = tau[(h, i)] * TB_MIN

            dh_kwh = _arc_kwh(h, i)

            adj[h].append((i, travel_min, dh_kwh, 'station_trip'))



    # ── Trip → DEPOT ──

    for i in T:

        if (i, DEPOT) in tau:

            travel_min = tau[(i, DEPOT)] * TB_MIN

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

        )



        options.append((target_soc, departure_time, charge_cost, energy_kwh))



    return options





# ──────────────────────────────────────────────────────────────────────

#  DOMINANCE CHECK

# ──────────────────────────────────────────────────────────────────────


def _is_dominated(label: Label, label_pool: list[Label]) -> bool:
    """
    Check whether `label` is dominated.
    Because our network moves forward in time, cycles are impossible.
    We do NOT need to check trips_visited for elementarity!
    """
    
    for other in label_pool:
        # Added tiny 1e-4 tolerance to prevent floating point misses
        if (other.rc   <= label.rc + 1e-4 and
            other.time <= label.time + 1e-4
            and  other.soc  >= label.soc - 1e-4):
            return True
    return False





def _prune_dominated(label_pool: list[Label]) -> list[Label]:

    """

    Remove mutually dominated labels from a pool.

    Returns a new list with only non‑dominated labels.

    """

    if len(label_pool) <= 1:

        return label_pool



    kept: list[Label] = []

    for lab in label_pool:

        if not _is_dominated(lab, kept):

            # Also remove any labels in `kept` that the new label dominates

            kept = [k for k in kept

                    if not (lab.rc   <= k.rc   and

                            lab.time <= k.time and

                            lab.soc  >= k.soc  and

                            lab.trips_visited <= k.trips_visited and

                            (lab.rc < k.rc or lab.time < k.time or

                             lab.soc > k.soc or lab.trips_visited < k.trips_visited))]

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



    # ── Parameters ──

    G: float = 300.0,

    TB_MIN: int = 1,

    bar_t: int = 1560,

    bus_cost: float = 1e5,

    charge_rate_kw: float = 300.0,

    hourly_prices: dict | None = None,

    charge_cost_premium: float = 0.0,

    travel_cost_factor: float = 1.0,

    RC_EPSILON: float = 1.0,



    # ── Algorithm tuning ──

    K_BEST: int = 50,

    MAX_LABELS_PER_NODE: int = 200,

    soc_charge_levels: list[float] | None = None,

    MIN_TRIPS_PER_ROUTE: int = 0,

    MAX_DAILY_RECHARGES: int = 4,

) -> list[dict]:

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



    # ── Build DAG if not provided ──

    if adj is None:

        adj = build_dag(

            T=T, S_use=S_use, DEPOT=DEPOT,

            tau=tau, d=d, st=st, et=et, sl=sl, el=el,

            epsilon=epsilon, TB_MIN=TB_MIN, bar_t=bar_t,

        )



    if soc_charge_levels is None:

        soc_charge_levels = [0.25 * G, 0.50 * G, 0.75 * G, G]



    horizon_min = bar_t * TB_MIN    # total horizon in minutes



    # ── Helper: convert time‑block to minutes ──

    def tb2min(tb: int) -> float:

        return (tb - 1) * TB_MIN



    # Trip time windows in minutes

    trip_start_min = {i: tb2min(st[i]) for i in T}

    trip_end_min   = {i: tb2min(et[i]) for i in T}



    # ──────────────────────────────────────────────────────────────

    # STEP 1:  Initialise source label at DEPOT

    # ──────────────────────────────────────────────────────────────

    # The vehicle departs the depot at time 0 with full SOC.

    # The fixed bus_cost is added to the reduced cost at initialisation

    # so that it appears in every route.



    source_label = Label(

        rc=bus_cost,

        time=0.0,          # earliest possible departure (minute 0)

        soc=G,             # full battery

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

    # We use a priority queue (min‑heap on reduced cost) so that

    # labels with the most promising (lowest) reduced cost are

    # extended first.  This is a label‑correcting approach since

    # the graph may have negative arc costs (due to dual subtraction).

    #

    # node_labels[v] stores the list of non‑dominated labels at node v.



    node_labels: dict[Any, list[Label]] = {DEPOT: [source_label]}

    for i in T:

        node_labels[i] = []

    for h in S_use:

        node_labels[h] = []



    # Completed routes (labels that reached DEPOT as sink)

    completed: list[Label] = []



    # Priority queue:  (rc, unique_id, label)

    pq: list[tuple[float, int, Label]] = []

    _uid = 0

    #heapq.heappush(pq, (source_label.rc, _uid, source_label))
    heapq.heappush(pq, (source_label.time, source_label.rc, _uid, source_label))
    
    _uid += 1



    # ── Main loop ──

    while pq:

        _, _, _, label = heapq.heappop(pq)



        cur = label.node



        # Skip if this label has been dominated since it was enqueued

        if cur != DEPOT or label is source_label:

            # Check it's still in the pool (quick identity check)

            if cur in node_labels and label not in node_labels.get(cur, []):

                # It was pruned; skip

                if cur != DEPOT:

                    continue



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
                    # Prune labels that are strictly worse than our new label
                    node_labels[trip_idx] = [
                        lb for lb in pool
                        if not (new_label.rc <= lb.rc and new_label.time <= lb.time and new_label.soc >= lb.soc)
                    ]
                    node_labels[trip_idx].append(new_label)

                    if len(node_labels[trip_idx]) > MAX_LABELS_PER_NODE:
                        node_labels[trip_idx].sort(key=lambda lb: lb.rc)
                        node_labels[trip_idx] = node_labels[trip_idx][:MAX_LABELS_PER_NODE]

                    # NEW: Push sorted by TIME first!
                    heapq.heappush(pq, (new_label.time, new_label.rc, _uid, new_label))
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



                charge_options = _generate_charge_options(

                    arrival_soc=soc_at_station,

                    arrival_time_min=arrival_time,

                    departure_deadline_min=departure_deadline,

                    G=G,

                    charge_rate_kw=charge_rate_kw,

                    hourly_prices=hourly_prices,

                    charge_cost_premium=charge_cost_premium,

                    soc_levels=soc_charge_levels,

                )



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
                    if not _is_dominated(new_label, pool):
                        # Prune labels that are strictly worse than our new label
                        node_labels[station] = [
                            lb for lb in pool
                            if not (new_label.rc <= lb.rc and new_label.time <= lb.time and new_label.soc >= lb.soc)
                        ]
                        node_labels[station].append(new_label)

                        if len(node_labels[station]) > MAX_LABELS_PER_NODE:
                            node_labels[station].sort(key=lambda lb: lb.rc)
                            node_labels[station] = node_labels[station][:MAX_LABELS_PER_NODE]

                        # NEW: Push sorted by TIME first!
                        heapq.heappush(pq, (new_label.time, new_label.rc, _uid, new_label))
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



                completed.append(completed_label)



    # ──────────────────────────────────────────────────────────────

    # STEP 3:  Collect K‑best routes with negative reduced cost

    # ──────────────────────────────────────────────────────────────



    # Filter to negative reduced cost

    neg_routes = [lab for lab in completed if lab.rc < -RC_EPSILON]



    # Sort by reduced cost (most negative first)

    neg_routes.sort(key=lambda lb: lb.rc)



    # De‑duplicate by trip‑set (keep best rc per unique trip set)

    seen_trip_sets: set[frozenset] = set()

    unique_routes: list[Label] = []

    for lab in neg_routes:

        key = lab.trips_visited

        if key not in seen_trip_sets:

            seen_trip_sets.add(key)

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



    return results





# ──────────────────────────────────────────────────────────────────────

#  CONVENIENCE WRAPPER  (matches solve_pricing_fast call signature)

# ──────────────────────────────────────────────────────────────────────



def make_dp_pricer(

    T, S_use, DEPOT, tau, d, st, et, sl, el, epsilon,

    G, TB_MIN, bar_t, bus_cost, charge_rate_kw,

    hourly_prices, charge_cost_premium, travel_cost_factor,

    RC_EPSILON, K_BEST,

    MAX_LABELS_PER_NODE=200,

    soc_charge_levels=None,

    MIN_TRIPS_PER_ROUTE=0,

    MAX_DAILY_RECHARGES=4,

    max_trip2trip=15,

    max_trip2charge=60,

    max_charge2trip=60,

):

    """

    Factory that returns a callable with the same interface as

    ``solve_pricing_fast(alpha, beta, gamma, mode, ...)``.



    Usage in run_experiments.py

    ---------------------------

    >>> from pricing_dp import make_dp_pricer

    >>> dp_price = make_dp_pricer(T=T, S_use=S_use, ...)

    >>> # Inside CG loop:

    >>> new_routes = dp_price(alpha, beta_dual, gamma_dual)



    The returned function pre‑builds the DAG once on the first call and

    caches it for subsequent calls (the graph structure doesn't change

    across CG iterations – only the duals change).

    """



    # ── Pre‑build the DAG (topology is fixed) ──

    adj = build_dag(

        T=T, S_use=S_use, DEPOT=DEPOT,

        tau=tau, d=d, st=st, et=et, sl=sl, el=el,

        epsilon=epsilon, TB_MIN=TB_MIN, bar_t=bar_t,

        max_trip2trip=max_trip2trip,

        max_trip2charge=max_trip2charge,

        max_charge2trip=max_charge2trip,

    )



    _n_arcs = sum(len(v) for v in adj.values())

    print(f"[DP-PRICER] DAG built: {len(adj)} nodes, {_n_arcs} arcs")



    def _solve(alpha, beta=None, gamma=None, mode=1,

               num_fast_cols=None, time_limit=None, **kwargs):

        """

        Solve the pricing problem via DP.



        Parameters `mode`, `num_fast_cols`, `time_limit` are accepted

        for API compatibility but ignored (DP doesn't need them).

        Returns a list of route dicts (same as solve_pricing_fast output,

        but without the Gurobi model – just the routes).

        """

        routes = solve_pricing_dp(

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

            MAX_LABELS_PER_NODE=MAX_LABELS_PER_NODE,

            soc_charge_levels=soc_charge_levels,

            MIN_TRIPS_PER_ROUTE=MIN_TRIPS_PER_ROUTE,

            MAX_DAILY_RECHARGES=MAX_DAILY_RECHARGES,

        )



        n_neg = len(routes)
        best_overall_rc = routes[0]["_rc"] if routes else float("inf")
        print(f"[DP-PRICER] Found {n_neg} negative-RC routes (best_rc={best_overall_rc:.2f})")
        
        return routes, best_overall_rc



    return _solve





# ──────────────────────────────────────────────────────────────────────

#  INTEGRATION EXAMPLE  (paste into run_experiments.py CG loop)

# ──────────────────────────────────────────────────────────────────────

#

#  # ── Before the CG loop (after building S_use, tau, d, etc.) ──

#  from pricing_dp import make_dp_pricer

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


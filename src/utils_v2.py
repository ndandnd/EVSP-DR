import re

import numpy as np

import pandas as pd

import string

# import matplotlib.pyplot as plt



# use the shared constant from config; do NOT redefine locally

from config import time_blocks, charge_cost_premium, TIMEBLOCKS_PER_HOUR, CHARGE_PER_BLOCK, TRAVEL_COST_FACTOR





def make_locs(n: int):

    if not 1 <= n <= len(string.ascii_uppercase):

        raise ValueError(f"points must be 1..{len(string.ascii_uppercase)}")

    return list(string.ascii_uppercase[:n])





# def plot_net_with_delta(net, delta, time_blocks, solar_mult, mode_name, base_eps, points):

#     times = sorted(time_blocks)

#     net_vals = [net.get(t, 0) * 100 for t in times]

#     delta_vals = [-delta.get(t, 0) * 100 for t in times]



#     plt.figure()

#     plt.bar(times, net_vals, align='center', label='Net (Dis)charge')

#     plt.plot(times, delta_vals, marker='o', label='Net Generation')

#     plt.axhline(0, linewidth=0.8)

#     plt.xlabel('Hour')

#     plt.ylabel('Power (kW)')

#     plt.title('Net Generation Over Time')

#     plt.legend()

#     plt.grid(True, linestyle='--', linewidth=0.5)

#     filename = f"net_generation_s{solar_mult}_m{mode_name}_e{base_eps}_p{points}.png"

#     plt.savefig(filename, bbox_inches='tight')

#     plt.close()







# ---------- pricing costs (uses time-varying price table) ----------

def base_station_name(name: str) -> str:

    s = str(name).strip()

    if "_" in s:

        left, right = s.rsplit("_", 1)

        if right.isdigit():   # only strip copy suffix

            return left

    return s


def load_station_hourly_prices(csv_path, stations):

    """Load either temporal-only or station-specific hourly prices.

    Accepted schemas are ``time_block,cost`` (replicated across every physical
    station) and ``time_block,station,cost``. The latter must contain every
    requested physical station; missing stations are an error rather than a
    silent fallback to the depot curve.
    """

    df = pd.read_csv(csv_path)
    required = {"time_block", "cost"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must contain time_block,cost; found {list(df.columns)}"
        )

    work = df.copy()
    work["time_block"] = pd.to_numeric(work["time_block"], errors="raise")
    work["cost"] = pd.to_numeric(work["cost"], errors="raise")
    if work[["time_block", "cost"]].isna().any().any():
        raise ValueError(f"{csv_path} contains missing time blocks or costs")
    if not np.allclose(work["time_block"], np.round(work["time_block"])):
        raise ValueError(f"{csv_path} contains non-integer hourly time_block values")
    work["time_block"] = work["time_block"].astype(int)
    if work.empty:
        raise ValueError(f"{csv_path} contains no price rows")

    def _validate_contiguous_hours(hours):
        hour_set = set(hours)
        expected = set(range(0, max(hour_set) + 1))
        if hour_set != expected:
            missing_hours = sorted(expected - hour_set)
            raise ValueError(
                f"{csv_path} must contain contiguous hourly time_block values "
                f"starting at 0; missing={missing_hours}"
            )

    station_bases = list(dict.fromkeys(base_station_name(s) for s in stations))
    if not station_bases:
        raise ValueError("At least one station is required to load price curves")

    if "station" not in work.columns:
        if work["time_block"].duplicated().any():
            raise ValueError(f"{csv_path} contains duplicate time_block rows")
        curve = work.set_index("time_block")["cost"].astype(float).to_dict()
        _validate_contiguous_hours(curve)
        return {station: dict(curve) for station in station_bases}

    work["station"] = work["station"].map(base_station_name)
    if work[["station", "time_block"]].duplicated().any():
        raise ValueError(f"{csv_path} contains duplicate station/time_block rows")

    result = {
        station: group.set_index("time_block")["cost"].astype(float).to_dict()
        for station, group in work.groupby("station", sort=False)
    }
    missing = sorted(set(station_bases) - set(result))
    if missing:
        raise ValueError(f"{csv_path} has no price curve for stations: {missing}")

    expected_hours = set(result[station_bases[0]])
    _validate_contiguous_hours(expected_hours)
    inconsistent = [
        station for station in station_bases
        if set(result[station]) != expected_hours
    ]
    if inconsistent:
        raise ValueError(
            f"{csv_path} has inconsistent time blocks for stations: {inconsistent}"
        )
    return {station: result[station] for station in station_bases}


def select_unique_station_copies(stations, depot):

    """Choose one pricing node per physical station without reusing DEPOT.

    The source/sink depot node and a charging node must have distinct graph
    identities. For the depot's physical station, prefer another configured
    copy (for example PARX_1 rather than the source node PARX_0).
    """

    grouped = {}
    for station in stations:
        grouped.setdefault(base_station_name(station), []).append(station)

    selected = []
    depot_base = base_station_name(depot)
    for base, copies in grouped.items():
        candidates = [station for station in copies if station != depot]
        if base == depot_base and not candidates:
            raise ValueError(
                f"Depot charging station {base!r} needs a node distinct from {depot!r}"
            )
        selected.append(candidates[0] if base == depot_base else copies[0])

    if depot in selected:
        raise AssertionError("Pricing station selection reused the source depot node")
    return selected


def route_column_key(route, ndigits=6):

    """Identify a column by both its node path and charging realization."""

    charging = route.get("charging_stops", {}) or {}

    def _rounded(values):
        return tuple(round(float(value), ndigits) for value in (values or []))

    return (
        tuple(route.get("route", [])),
        tuple(str(value) for value in charging.get("stations", []) or []),
        _rounded(charging.get("cst", [])),
        _rounded(charging.get("cet", [])),
        _rounded(charging.get("kwh", [])),
    )




# ---------- COST CALCULATOR (UPDATED FOR DICT) ----------

def calc_cost_distance_only(route, truck_cost):

    # Use the route's built-in deadhead and travel, ignore charging

    total = float(truck_cost)

    deadhead_amount = route.get("deadhead_kwh", 0.0)

    total += deadhead_amount * TRAVEL_COST_FACTOR

    return total



#def calculate_truck_route_cost(route, truck_cost, hourly_prices: dict) -> float:
def calculate_truck_route_cost(
    route,
    truck_cost,
    hourly_prices: dict,
    station_hourly_prices: dict = None,   # NEW
    charge_start_cost: float = 0.0,       # NEW
) -> float:
    """

    Calculates cost using hourly_prices dict {0: 0.10, 1: 0.15...}

    and the route's cst/cet values.

    """

    total = float(truck_cost)

    # --- NEW: Add Travel Cost from Route Description ---

    # We use .get(0) so it doesn't crash on old pickles or dummy routes

    deadhead_amount = route.get("deadhead_kwh", 0.0)

    total += deadhead_amount * TRAVEL_COST_FACTOR

    # ---------------------------------------------------



    stops = route.get("charging_stops", {})

    stations = stops.get("stations", [])

    csts = stops.get("cst", [])

    cets = stops.get("cet", [])



    # Safety check

    if not (len(stations) == len(csts) == len(cets)):

        # If lengths mismatch, we might have a data issue, return base cost or warn

        return total



    for i, station in enumerate(stations):

        start_min = csts[i]

        end_min   = cets[i]



        # Logic: Price is based on the START hour

        hour_idx = int(start_min // 60)



        # Fallback to key 0 or max key if hour is out of bounds (e.g. 25th hour)

        # Assuming hourly_prices has keys 0..23 or similar


        # Use station-specific prices if available, fall back to base hourly_prices.
        # Station nodes are copy names ("2190L_0"); price tables use base names.
        prices_to_use = (station_hourly_prices or {}).get(base_station_name(station), hourly_prices)
        price = prices_to_use.get(hour_idx, 0.0)
        if price == 0.0 and prices_to_use:
            price = prices_to_use.get(hour_idx % 24, prices_to_use.get(0, 0.0))

        # price = hourly_prices.get(hour_idx, 0.0)

        # if price == 0.0 and hourly_prices:

        #      # Try modulo 24 if your dict is 0-23 but time goes to 26h

        #      price = hourly_prices.get(hour_idx % 24, hourly_prices.get(0, 0.0))



        duration_min = end_min - start_min



        # Energy = Duration * Rate (kW/min)

        energy_kwh = duration_min * CHARGE_PER_BLOCK



        total += price * energy_kwh * charge_cost_premium
        total += charge_start_cost  # Add flat cost per charging activity


    return total





def calculate_truck_route_cost_accurate(

    route,

    truck_cost,

    hourly_prices: dict,

    charge_rate_kw: float,

    travel_cost_factor: float | None = None,
    station_hourly_prices=None,    # NEW — optional
    charge_start_cost=0.0,         # NEW

) -> float:

    """

    Calculates charging cost by splitting across hour boundaries (matching DP's _compute_charging_cost logic).



    This is more accurate than calculate_truck_route_cost because it properly accounts for

    charging that spans multiple hours with different prices.



    Parameters

    ----------

    route : dict

        Route dictionary with "charging_stops", "deadhead_kwh" keys

    truck_cost : float

        Fixed truck cost component

    hourly_prices : dict

        {hour_index: $/kWh}

    charge_rate_kw : float

        Charger power in kW (used to compute duration from energy)

    travel_cost_factor : float | None

        Cost per kWh of deadhead. If None, uses imported TRAVEL_COST_FACTOR



    Returns

    -------

    total_cost : float

        Total route cost (truck + deadhead travel + multi-hour charging)

    """

    if travel_cost_factor is None:

        travel_cost_factor = TRAVEL_COST_FACTOR



    total = float(truck_cost)



    # Add travel cost for deadhead

    deadhead_amount = route.get("deadhead_kwh", 0.0)

    total += deadhead_amount * travel_cost_factor



    # Add charging cost (split across hour boundaries)

    stops = route.get("charging_stops", {})

    stations = stops.get("stations", [])

    csts = stops.get("cst", [])

    cets = stops.get("cet", [])

    kwhs = stops.get("kwh", [])



    # Safety check

    if not (len(stations) == len(csts) == len(cets)):

        raise ValueError(

            "charging_stops stations/cst/cet must have matching lengths"

        )

    # Older route dicts may omit per-stop kWh; fall back to duration * rate.

    if not kwhs and stations:

        kwhs = [max(0.0, (cets[i] - csts[i])) * charge_rate_kw / 60.0

                for i in range(len(stations))]

    elif len(kwhs) != len(stations):

        raise ValueError(

            "charging_stops.kwh must be empty or have one entry per station"

        )



    for i, station in enumerate(stations):

        # Station-specific curve when available (keyed by base station name)

        prices_to_use = (station_hourly_prices or {}).get(

            base_station_name(station), hourly_prices

        )

        # Compute charging cost for this segment (split across hours)

        cost = _compute_charging_cost_accurate(

            start_min=csts[i],

            energy_kwh=kwhs[i],

            charge_rate_kw=charge_rate_kw,

            hourly_prices=prices_to_use,

            charge_cost_premium=charge_cost_premium,

        )



        total += cost + charge_start_cost



    return total





def _compute_charging_cost_accurate(

    start_min: float,

    energy_kwh: float,

    charge_rate_kw: float,

    hourly_prices: dict,

    charge_cost_premium: float,

) -> float:

    """

    Compute time-of-day electricity cost by splitting across hour boundaries.



    This matches the DP implementation's _compute_charging_cost logic exactly.

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

        # Which hour-bucket does cursor_min fall in?

        hour_idx = int(cursor_min // 60)

        hour_idx_clamped = min(hour_idx, max_hour)



        # End of this hour bucket (in minutes)

        next_hour_min = (hour_idx + 1) * 60.0



        # Segment end: whichever comes first – next hour or charging end

        seg_end = min(next_hour_min, end_min)

        seg_duration_hours = (seg_end - cursor_min) / 60.0



        # Energy charged in this segment

        seg_kwh = charge_rate_kw * seg_duration_hours



        # Price for this segment

        price = hourly_prices.get(hour_idx_clamped, hourly_prices.get(hour_idx_clamped % 24, 0.0))

        cost += price * seg_kwh * charge_cost_premium



        cursor_min = seg_end



    return cost





def calculate_battery_route_cost(route, batt_cost, charging_cost_data: pd.DataFrame) -> float:

    total = float(batt_cost)



    kwh_per_block = CHARGE_PER_BLOCK



    cs = route.get("charging_stops", {})

    for (h, t) in cs.get("chi_plus", []):

        total += kwh_per_block *float(charging_cost_data.at[int(t), str(h)]) * charge_cost_premium

    for (h, t) in cs.get("chi_minus", []):

        total -= kwh_per_block * float(charging_cost_data.at[int(t), str(h)])

    return total







# ---------- duals from master ----------



def extract_duals(rmp):

    alpha, beta, gamma = {}, {}, {}

    for c in rmp.getConstrs():

        cname = c.ConstrName

        dual = c.Pi

        if cname.startswith("trip_coverage_"):

            alpha[int(cname.split("_")[-1])] = dual

        elif cname.startswith("freecharge_"):

            beta[int(cname.split("_")[-1])] = dual

        elif cname.startswith("discharge_"):

            gamma[int(cname.split("_")[-1])] = dual

    return alpha, beta, gamma





# ---------- pricing solution → route dicts ----------



def _safe_X(model, name):

    v = model.getVarByName(name)

    return 0.0 if v is None else float(v.X)





def extract_batt_route_from_solution(model, bar_t, h="h1"):

    route = {

        "route": ["PARX", h, "PARX"],

        "charging_stops": {

            "stations": [h],

            "cst": [], "cet": [],

            "chi_plus_free": [], "chi_minus_free": [],

            "chi_minus": [], "chi_plus": [], "chi_zero": []

        },

        "charging_activities": 1,

        "type": "batt"

    }



    cst = None

    cet = None

    for t in range(1, bar_t + 1):

        if _safe_X(model, f"chi_plus_free[{t}]") > 0.5:

            route["charging_stops"]["chi_plus_free"].append((h, t)); cst = cst or t; cet = t

        if _safe_X(model, f"chi_plus[{t}]") > 0.5:

            route["charging_stops"]["chi_plus"].append((h, t)); cst = cst or t; cet = t

        if _safe_X(model, f"chi_minus[{t}]") > 0.5:

            route["charging_stops"]["chi_minus"].append((h, t)); cst = cst or t; cet = t

        if _safe_X(model, f"chi_zero[{t}]") > 0.5:

            route["charging_stops"]["chi_zero"].append((h, t)); cst = cst or t; cet = t

        if _safe_X(model, f"chi_minus_free[{t}]") > 0.5:

            route["charging_stops"]["chi_minus_free"].append((h, t)); cst = cst or t; cet = t



    route["charging_stops"]["cst"].append(cst if cst is not None else 1)

    route["charging_stops"]["cet"].append(cet if cet is not None else bar_t)

    return route

# In utils.py





def extract_route_from_solution(vars_dict, T, S, bar_t, depot="PARX", value_getter=lambda v: v.X):

    def _has(varname, key):

        if varname not in vars_dict: return False

        return key in vars_dict[varname]



    def _get_val(varname, key, default=0.0):

        if _has(varname, key):

            return value_getter(vars_dict[varname][key])

        return default



    # 1. GRAPH TRAVERSAL (Find path)

    route_nodes = [depot]

    first = None



    # Find start

    for i in T:

        if _get_val("wA_trip", i) > 0.5: first = i; break

    if first is None:

        for h in S:

            if _get_val("wA_station", h) > 0.5: first = h; break



    if first is None:

        return {"route": [], "dummy": True, "type": "empty", "desc": "Empty Route"}



    route_nodes.append(first)

    cur = first

    seen = set([depot, first])



    while True:

        nxt = None

        # From Trip

        if cur in T:

            for j in T:

                if j != cur and _get_val("x", (cur, j)) > 0.5: nxt = j; break

            if nxt is None:

                for h in S:

                    if _get_val("y", (cur, h)) > 0.5: nxt = h; break

            if nxt is None and _get_val("wOmega_trip", cur) > 0.5: nxt = depot

        # From Station

        else:

            for i in T:

                if _get_val("z", (cur, i)) > 0.5: nxt = i; break

            if nxt is None and _get_val("wOmega_station", cur) > 0.5: nxt = depot



        if nxt is None: break

        route_nodes.append(nxt)

        if nxt == depot: break

        if nxt in seen: break

        seen.add(nxt)

        cur = nxt



    # 2. EXTRACT DETAILS (Charging & SoC)

    route_data = {

        "route": route_nodes,

        "charging_stops": {"stations": [], "cst": [], "cet": [], "kwh": []},

        "charging_activities": 0

    }



    # Build a "Rich Description" string for debugging

    # Format: PARX -> T1(SoC:280) -> S1(Charge:50 @ 600-630) -> ...

    desc_parts = []



    for node in route_nodes:

        part_str = str(node)



        # Try to get SoC at arrival (g variables)

        # Assuming g[i] exists in pricing model

        soc = _get_val("g", node, default=-1)

        if soc >= 0:

            part_str += f"(SoC:{soc:.0f})"



        # If it's a station, get charging info

        if node in S:

            cst = _get_val("cst", node)

            cet = _get_val("cet", node)

            amt = _get_val("v_amt", node) # <--- THIS IS WHAT YOU MISSED



            if amt > 0.1:

                route_data["charging_stops"]["stations"].append(node)

                route_data["charging_stops"]["cst"].append(cst)

                route_data["charging_stops"]["cet"].append(cet)

                route_data["charging_stops"]["kwh"].append(amt)

                route_data["charging_activities"] += 1



                # Add to description

                # Convert minutes to HH:MM for readability

                h_start, m_start = divmod(int(cst), 60)

                h_end, m_end = divmod(int(cet), 60)

                time_str = f"{h_start:02d}:{m_start:02d}-{h_end:02d}:{m_end:02d}"



                part_str += f" [Charge {amt:.1f}kWh @ {time_str}]"



        desc_parts.append(part_str)



    route_data["desc"] = " -> ".join(desc_parts)

    route_data["type"] = "truck"



    return route_data







# ---------- price curve loader ----------



def load_price_curve(csv_path, time_blocks, stations, timeblocks_per_hour=1, clamp_to_csv=True):

    df = pd.read_csv(csv_path)

    if not {'time_block', 'cost'}.issubset(df.columns):

        raise ValueError("CSV must have columns: time_block,cost")



    df['time_block'] = df['time_block'].astype(int)

    df['cost'] = df['cost'].astype(float)



    # Build hourly map from CSV

    price_map_hour = dict(zip(df['time_block'], df['cost']))

    max_hour_in_csv = max(price_map_hour.keys())



    k = int(timeblocks_per_hour)

    if k <= 0:

        raise ValueError(f"timeblocks_per_hour must be positive, got {k}")



    # Map fine block -> hour index (ceil(t/k))

    def block_to_hour(t: int) -> int:

        h = (int(t) + k - 1) // k

        if clamp_to_csv:

            return min(h, max_hour_in_csv)

        return h



    # Validate *hours* needed exist (not fine blocks)

    needed_hours = {block_to_hour(t) for t in time_blocks}

    missing_hours = sorted(h for h in needed_hours if h not in price_map_hour)

    if missing_hours:

        raise ValueError(f"CSV missing prices for HOURS needed: {missing_hours}")



    # Expand to per-(fine block) price

    price_map_block = {t: price_map_hour[block_to_hour(t)] for t in time_blocks}



    # Build station x time table (still same price for all stations in your current setup)

    data = []

    for t in time_blocks:

        cost = price_map_block[t]

        row = [cost for _ in stations]

        data.append(row)



    charging_cost_data = pd.DataFrame(data, index=time_blocks, columns=stations)

    avg_cost = float(np.mean([price_map_block[t] for t in time_blocks]))

    return charging_cost_data, avg_cost

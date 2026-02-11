import re
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt

# use the shared constant from config; do NOT redefine locally
from config import time_blocks, charge_cost_premium, TIMEBLOCKS_PER_HOUR, CHARGE_PER_BLOCK


def make_locs(n: int):
    if not 1 <= n <= len(string.ascii_uppercase):
        raise ValueError(f"points must be 1..{len(string.ascii_uppercase)}")
    return list(string.ascii_uppercase[:n])


def plot_net_with_delta(net, delta, time_blocks, solar_mult, mode_name, base_eps, points):
    times = sorted(time_blocks)
    net_vals = [net.get(t, 0) * 100 for t in times]
    delta_vals = [-delta.get(t, 0) * 100 for t in times]

    plt.figure()
    plt.bar(times, net_vals, align='center', label='Net (Dis)charge')
    plt.plot(times, delta_vals, marker='o', label='Net Generation')
    plt.axhline(0, linewidth=0.8)
    plt.xlabel('Hour')
    plt.ylabel('Power (kW)')
    plt.title('Net Generation Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    filename = f"net_generation_s{solar_mult}_m{mode_name}_e{base_eps}_p{points}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()



# ---------- pricing costs (uses time-varying price table) ----------
def base_station_name(name: str) -> str:
    s = str(name).strip()
    if "_" in s:
        left, right = s.rsplit("_", 1)
        if right.isdigit():   # only strip copy suffix
            return left
    return s


# ---------- COST CALCULATOR (UPDATED FOR DICT) ----------
def calculate_truck_route_cost(route, truck_cost, hourly_prices: dict) -> float:
    """
    Calculates cost using hourly_prices dict {0: 0.10, 1: 0.15...}
    and the route's cst/cet values.
    """
    total = float(truck_cost)
    
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
        price = hourly_prices.get(hour_idx, 0.0)
        if price == 0.0 and hourly_prices:
             # Try modulo 24 if your dict is 0-23 but time goes to 26h
             price = hourly_prices.get(hour_idx % 24, hourly_prices.get(0, 0.0))

        duration_min = end_min - start_min
        
        # Energy = Duration * Rate (kW/min)
        energy_kwh = duration_min * CHARGE_PER_BLOCK
        
        total += price * energy_kwh * charge_cost_premium

    return total


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

# def extract_route_from_solution(vars_dict, T, S, bar_t, depot="PARX", value_getter=lambda v: v.X):
#     def _has(varname, key):
#         # SAFEGUARD: Check if varname exists in dict first!
#         if varname not in vars_dict:
#             return False
#         return key in vars_dict[varname]

#     # 1. GRAPH TRAVERSAL (Find the path of nodes)
#     route_nodes = [depot]
#     first = None
    
#     # Find start arc
#     for i in T:
#         if _has("wA_trip", i) and value_getter(vars_dict["wA_trip"][i]) > 0.5:
#             first = i; break
#     if first is None:
#         for h in S:
#             if _has("wA_station", h) and value_getter(vars_dict["wA_station"][h]) > 0.5:
#                 first = h; break
    
#     if first is None:
#         # If no start found, check if it's a "stay at depot" or empty route (dummy)
#         return {"route": [], "dummy": True, "charging_stops": {}, "type": "empty"}

#     route_nodes.append(first)
#     cur = first
#     seen = set([depot, first])

#     # Follow arcs
#     while True:
#         nxt = None
#         # From Trip
#         if cur in T:
#             for j in T:
#                 if j != cur and _has("x", (cur, j)) and value_getter(vars_dict["x"][(cur, j)]) > 0.5:
#                     nxt = j; break
#             if nxt is None:
#                 for h in S:
#                     if _has("y", (cur, h)) and value_getter(vars_dict["y"][(cur, h)]) > 0.5:
#                         nxt = h; break
#             if nxt is None and _has("wOmega_trip", cur) and value_getter(vars_dict["wOmega_trip"][cur]) > 0.5:
#                 nxt = depot
#         # From Station
#         else:
#             for i in T:
#                 if _has("z", (cur, i)) and value_getter(vars_dict["z"][(cur, i)]) > 0.5:
#                     nxt = i; break
#             if nxt is None and _has("wOmega_station", cur) and value_getter(vars_dict["wOmega_station"][cur]) > 0.5:
#                 nxt = depot

#         if nxt is None:
#             break 

#         route_nodes.append(nxt)
#         if nxt == depot:
#             break
#         if nxt in seen:
#             break # Cycle detected
#         seen.add(nxt)
#         cur = nxt

#     # 2. EXTRACT CHARGING DETAILS (Updated for cst/cet)
#     route = {
#         "route": route_nodes,
#         "charging_stops": {
#             "stations": [], "cst": [], "cet": []
#         },
#         "charging_activities": 0
#     }

#     # Iterate through the path and grab cst/cet for any station nodes
#     for node in route_nodes[1:-1]:
#         if node in S:
#             route["charging_stops"]["stations"].append(node)
            
#             start_val = 0
#             end_val = 0
            
#             # Look for cst/cet in vars_dict safely
#             if "cst" in vars_dict and node in vars_dict["cst"]:
#                 start_val = value_getter(vars_dict["cst"][node])
            
#             if "cet" in vars_dict and node in vars_dict["cet"]:
#                 end_val = value_getter(vars_dict["cet"][node])
                
#             route["charging_stops"]["cst"].append(start_val)
#             route["charging_stops"]["cet"].append(end_val)
            
#             # Count if meaningful charge occurred
#             if end_val - start_val > 0.1:
#                 route["charging_activities"] += 1

#     route["type"] = "truck"
#     return route


# ---------- price curve loader ----------

# def load_price_curve(csv_path, time_blocks, stations):
#     df = pd.read_csv(csv_path)
#     if not {'time_block', 'cost'}.issubset(df.columns):
#         raise ValueError("CSV must have columns: time_block,cost")

#     df['time_block'] = df['time_block'].astype(int)
#     df['cost'] = df['cost'].astype(float)

#     missing = [t for t in time_blocks if t not in set(df['time_block'])]
#     if missing:
#         raise ValueError(f"CSV missing prices for time blocks: {missing}")

#     price_map = dict(zip(df['time_block'], df['cost']))



#     #data = [[price_map[t] for _ in stations] for t in time_blocks]
    
    

#     data = []
#     for t in time_blocks:
#         row = []
#         for s in stations:
#             # 1. Parse base name to handle copies (e.g., "2190L_1" -> "2190L")
#             if "_" in s and s.split("_")[-1].isdigit():
#                 base_name = s.rsplit("_", 1)[0]
#             else:
#                 base_name = s

#             # 2. Get base price (using base_name logic if you had station-specific columns)
#             # Since we use a single price_map currently:
#             cost = price_map[t]
            
#             row.append(cost)
#         data.append(row)
    
    
#     charging_cost_data = pd.DataFrame(data, index=time_blocks, columns=stations)
#     avg_cost = float(np.mean([price_map[t] for t in time_blocks]))
#     return charging_cost_data, avg_cost


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
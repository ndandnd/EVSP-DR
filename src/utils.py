import re
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt

# use the shared constant from config; do NOT redefine locally
from config import time_blocks, charge_cost_premium


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

def calculate_truck_route_cost(route, truck_cost, charging_cost_data: pd.DataFrame) -> float:
    total = float(truck_cost)
    cs = route.get("charging_stops", {})
    for (h, t) in cs.get("chi_plus", []):
        total += float(charging_cost_data.at[int(t), str(h)]) * charge_cost_premium
    for (h, t) in cs.get("chi_minus", []):
        total -= float(charging_cost_data.at[int(t), str(h)])
    return total

def calculate_battery_route_cost(route, batt_cost, charging_cost_data: pd.DataFrame) -> float:
    total = float(batt_cost)
    cs = route.get("charging_stops", {})
    for (h, t) in cs.get("chi_plus", []):
        total += float(charging_cost_data.at[int(t), str(h)]) * charge_cost_premium
    for (h, t) in cs.get("chi_minus", []):
        total -= float(charging_cost_data.at[int(t), str(h)])
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


def extract_route_from_solution(vars_dict, T, S, bar_t, depot="PARX", value_getter=lambda v: v.X):
    def _has(varname, key):
        return key in vars_dict[varname]

    route_nodes = [depot]
    first = None
    for i in T:
        if value_getter(vars_dict["wA_trip"][i]) > 0.5:
            first = i; break
    if first is None:
        for h in S:
            if value_getter(vars_dict["wA_station"][h]) > 0.5:
                first = h; break
    if first is None:
        raise RuntimeError("No node found leaving the depot!")

    route_nodes.append(first)
    cur = first
    seen = set([depot, first])

    while True:
        nxt = None
        if cur in T:
            for j in T:
                if j != cur and _has("x", (cur, j)) and value_getter(vars_dict["x"][(cur, j)]) > 0.5:
                    nxt = j; break
            if nxt is None:
                for h in S:
                    if _has("y", (cur, h)) and value_getter(vars_dict["y"][(cur, h)]) > 0.5:
                        nxt = h; break
            if nxt is None and value_getter(vars_dict["wOmega_trip"][cur]) > 0.5:
                nxt = depot
        else:
            for i in T:
                if _has("z", (cur, i)) and value_getter(vars_dict["z"][(cur, i)]) > 0.5:
                    nxt = i; break
            if nxt is None and value_getter(vars_dict["wOmega_station"][cur]) > 0.5:
                nxt = depot

        if nxt is None:
            raise RuntimeError(f"No outgoing arc from {cur}")

        route_nodes.append(nxt)
        if nxt == depot:
            break
        if nxt in seen:
            raise RuntimeError(f"Cycle detected at {nxt}")
        seen.add(nxt)
        cur = nxt

    route = {
        "route": route_nodes,
        "charging_stops": {k: [] for k in [
            "stations", "cst", "cet",
            "chi_plus_free", "chi_minus_free", "chi_minus", "chi_plus", "chi_zero"
        ]},
        "charging_activities": 0
    }

    for node in route_nodes[1:-1]:
        if node in S:
            route["charging_stops"]["stations"].append(node)
            if "cst" in vars_dict and node in vars_dict["cst"]:
                route["charging_stops"]["cst"].append(value_getter(vars_dict["cst"][node]))
            if "cet" in vars_dict and node in vars_dict["cet"]:
                route["charging_stops"]["cet"].append(value_getter(vars_dict["cet"][node]))
            for t in range(1, bar_t + 1):
                if _has("chi_plus_free",  (node, t)) and value_getter(vars_dict["chi_plus_free"][(node, t)])  > 0.5:
                    route["charging_stops"]["chi_plus_free"].append((node, t))
                if _has("chi_minus_free", (node, t)) and value_getter(vars_dict["chi_minus_free"][(node, t)]) > 0.5:
                    route["charging_stops"]["chi_minus_free"].append((node, t))
                if _has("chi_minus",      (node, t)) and value_getter(vars_dict["chi_minus"][(node, t)])      > 0.5:
                    route["charging_stops"]["chi_minus"].append((node, t))
                if _has("chi_plus",       (node, t)) and value_getter(vars_dict["chi_plus"][(node, t)])       > 0.5:
                    route["charging_stops"]["chi_plus"].append((node, t))
                if _has("chi_zero",       (node, t)) and value_getter(vars_dict["chi_zero"][(node, t)])       > 0.5:
                    route["charging_stops"]["chi_zero"].append((node, t))
            route["charging_activities"] += 1

    route["type"] = "truck" if any(n in T for n in route_nodes) else "batt"
    if "g_return" in vars_dict:
        try:
            route["remaining_soc"] = float(value_getter(vars_dict["g_return"]))
        except Exception:
            pass
    return route


# ---------- price curve loader ----------

def load_price_curve(csv_path, time_blocks, stations):
    df = pd.read_csv(csv_path)
    if not {'time_block', 'cost'}.issubset(df.columns):
        raise ValueError("CSV must have columns: time_block,cost")

    df['time_block'] = df['time_block'].astype(int)
    df['cost'] = df['cost'].astype(float)

    missing = [t for t in time_blocks if t not in set(df['time_block'])]
    if missing:
        raise ValueError(f"CSV missing prices for time blocks: {missing}")

    price_map = dict(zip(df['time_block'], df['cost']))



    #data = [[price_map[t] for _ in stations] for t in time_blocks]
    
    

    data = []
    for t in time_blocks:
        row = []
        for s in stations:
            # 1. Parse base name to handle copies (e.g., "2190L_1" -> "2190L")
            if "_" in s and s.split("_")[-1].isdigit():
                base_name = s.rsplit("_", 1)[0]
            else:
                base_name = s

            # 2. Get base price (using base_name logic if you had station-specific columns)
            # Since we use a single price_map currently:
            cost = price_map[t]
            
            row.append(cost)
        data.append(row)
    
    
    charging_cost_data = pd.DataFrame(data, index=time_blocks, columns=stations)
    avg_cost = float(np.mean([price_map[t] for t in time_blocks]))
    return charging_cost_data, avg_cost

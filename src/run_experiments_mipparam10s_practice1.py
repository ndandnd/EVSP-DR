#%%
# EVSP only (no solar, no V2G/V2V) on your real data.
# Speedups:
#  - Smaller pool + RC cutoff + K-best columns per CG iteration
#  - Pricing timelimit + gap; master LP timelimit; stagnation early-stop
#  - Nodefile spill to avoid OOM kills; auto TMP detection
#  - Final MIP with warm-start and relaxed gap

import os
import time
import math
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from gurobipy import Model, Column, GRB, quicksum
# from collections import Counter, defaultdict

from config import (
    n_fast_cols, n_exact_cols, tolerance,
    bar_t, time_blocks, TIMEBLOCKS_PER_HOUR,
    DEPOT_NAME, G, CHARGE_PER_BLOCK, CHARGE_RATE_KW,
    charge_cost_premium,
    BUS_COST_KX, 
    CHARGING_STATIONS,
    STATION_COPIES,

    # NEW
    RC_EPSILON, K_BEST,
    MAX_CG_ITERS, STAGNATION_ITERS, MASTER_IMPROVE_THRESHOLD,
    THREADS, NODEFILE_START, NODEFILE_DIR,
    MASTER_TIMELIMIT, PRICING_TIMELIMIT, PRICING_GAP
)

from utils import (
    load_price_curve, extract_duals, extract_route_from_solution,
    calculate_truck_route_cost
)
from master import init_master, solve_master, build_master


stopwatch_start = time.time()



# ------------------------------ Output dirs ------------------------------
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)
CKPT = OUTDIR / f"ckpt_{RUN_ID}"
CKPT.mkdir(parents=True, exist_ok=True)

# ------------------------------ Helpers ------------------------------

TB_MIN   = int(round(60 / TIMEBLOCKS_PER_HOUR))  # minutes per block (60, 30, 15…)
TB_HOURS = 1.0 / TIMEBLOCKS_PER_HOUR             # hours per block (1.0, 0.5, 0.25…)

# def energy_to_events(kwh: float) -> int:
#     # “event” = BLOCK_KWH kWh regardless of granularity
#     return int(math.ceil(float(kwh) / float(BLOCK_KWH)))

def _total_minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)

def _floor_block(hhmm: str) -> int:
    m = _total_minutes(hhmm)
    blk0 = m // TB_MIN            # 0-based
    return max(1, min(int(bar_t), blk0 + 1))

def _ceil_block(hhmm: str) -> int:
    m = _total_minutes(hhmm)
    blk0 = (m + TB_MIN - 1) // TB_MIN   # 0-based
    return max(1, min(int(bar_t), blk0 + 1))

def ceil_blocks_from_minutes(m: float) -> int:
    return int(math.ceil(float(m) / float(TB_MIN)))

def _detect_tmp():
    if NODEFILE_DIR:
        return NODEFILE_DIR
    for k in ("SLURM_TMPDIR", "TMPDIR", "TMP"):
        v = os.environ.get(k)
        if v:
            return v
    return "/tmp"

# ------------------------------ Data ingest ------------------------------

DATA_DIR = ROOT.parent / "data"

routes_csv    = DATA_DIR / "Practice_1bus.csv"
ref_dhd_csv   = DATA_DIR / "par_ref_dhd.csv"
ref_dict_csv  = DATA_DIR / "Ref_dict.csv"
prices_csv    = DATA_DIR / "hourly_prices.csv"
# details_csv    = DATA_DIR / "Par_VehicleDetails.csv"
# vd_csv = DATA_DIR / "Par_VehicleDetails.csv"  # or VehicleDetails.csv

# routes_csv    = DATA_DIR / "Toy_Routes.csv"
# ref_dhd_csv   = DATA_DIR / "Toy_Ref_DHD.csv"
# ref_dict_csv  = DATA_DIR / "Toy_Ref_dict.csv"
# prices_csv    = DATA_DIR / "Toy_Prices.csv"


if not routes_csv.exists():
    raise FileNotFoundError(f"Missing {routes_csv}")
if not ref_dhd_csv.exists():
    raise FileNotFoundError(f"Missing {ref_dhd_csv}")
if not ref_dict_csv.exists():
    raise FileNotFoundError(f"Missing {ref_dict_csv}")
if not prices_csv.exists():
    raise FileNotFoundError(f"Missing {prices_csv} (needed for charging prices)")

df_trips = pd.read_csv(routes_csv)



# # ------------------------------
# # Build loc -> reference map from VehicleDetails
# # ------------------------------
# def _norm_loc(x):
#     if pd.isna(x):
#         return None
#     return str(x).strip()

# def _norm_ref(x):
#     if pd.isna(x):
#         return None
#     s = str(x).strip()
#     # "13410.0" -> "13410"
#     if s.endswith(".0"):
#         s = s[:-2]
#     return s

# def build_loc_to_ref(vehicle_details_df: pd.DataFrame):
#     pairs = []

#     if {"From1", "Refer."}.issubset(vehicle_details_df.columns):
#         tmp = vehicle_details_df[["From1", "Refer."]]
#         for loc, ref in zip(tmp["From1"], tmp["Refer."]):
#             loc, ref = _norm_loc(loc), _norm_ref(ref)
#             if loc and ref:
#                 pairs.append((loc, ref))

#     if {"To1", "Refer.1"}.issubset(vehicle_details_df.columns):
#         tmp = vehicle_details_df[["To1", "Refer.1"]]
#         for loc, ref in zip(tmp["To1"], tmp["Refer.1"]):
#             loc, ref = _norm_loc(loc), _norm_ref(ref)
#             if loc and ref:
#                 pairs.append((loc, ref))

#     counts = defaultdict(Counter)
#     for loc, ref in pairs:
#         counts[loc][ref] += 1

#     loc_to_ref = {}
#     ambiguous = {}
#     for loc, ctr in counts.items():
#         best_ref, best_ct = ctr.most_common(1)[0]
#         loc_to_ref[loc] = best_ref
#         if len(ctr) > 1:
#             ambiguous[loc] = ctr

#     print(f"[map] loc_to_ref size = {len(loc_to_ref)}")
#     if ambiguous:
#         print(f"[map] ambiguous locations = {len(ambiguous)} (showing up to 20)")
#         for loc, ctr in list(ambiguous.items())[:20]:
#             print(" ", loc, dict(ctr))

#     return loc_to_ref

# # ------------------------------
# # Load VehicleDetails.csv (preprocess once)
# # ------------------------------

# loc_to_ref = {}
# df_vd = pd.read_csv(vd_csv, low_memory=False)
# loc_to_ref = build_loc_to_ref(df_vd)


# trip_col_map = {"SL": None, "ST": None, "ET": None, "EL": None, "Energy used": None}
trip_col_map = {
    "SL": "Start_Loc",
    "ST": "Start_Time",
    "ET": "End_Time",
    "EL": "End_Loc",
    "Energy used": "Energy"
}
for want in list(trip_col_map.keys()):
    if want in df_trips.columns:
        trip_col_map[want] = want
        continue
    if want == "Energy used" and "Energy_used" in df_trips.columns:
        trip_col_map[want] = "Energy_used"
        continue
    for c in df_trips.columns:
        if c.strip().lower() == want.lower():
            trip_col_map[want] = c
            break
# define the map: { "Name in CSV" : "Name Code Expects" }
column_renaming = {
    "From1": "SL",
    "Start1": "ST",
    "End1": "ET",
    "To1": "EL",
    "Usage kWh": "Energy used"
}
df_trips = df_trips.rename(columns=column_renaming)

missing = [k for k, v in trip_col_map.items() if v is None]
if missing:
    raise ValueError(f"{routes_csv.name} must have columns {{'SL','ST','ET','EL','Energy used'}}, "
                     f"could not find: {missing}. Found: {set(df_trips.columns)}")

df_trips = df_trips.rename(columns={trip_col_map[k]: k for k in trip_col_map})

df_ref_dict = pd.read_csv(ref_dict_csv)
df_ref_dhd = pd.read_csv(ref_dhd_csv)


def _norm_token(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _normalize_ref(x):
    s = _norm_token(x)
    if s is None:
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return s


# def _is_copy_node(name: str) -> bool:
#     s = str(name).strip()
#     if "_" not in s:
#         return False
#     return s.rsplit("_", 1)[1].isdigit()


def _ordered_ref_pair(a: str, b: str):
    return (a, b) if a <= b else (b, a)


def strip_copy_suffix(name: str) -> str:
    s = str(name).strip()
    if "_" in s:
        left, right = s.rsplit("_", 1)
        if right.isdigit():
            return left
    return s


def expand_station_copies(base_names, station_copies):
    out = []
    for b in base_names:
        c = station_copies.get(b, 1)
        for k in range(c):
            out.append(f"{b}_{k}")
    return out


# copies only
CHARGERS = expand_station_copies(CHARGING_STATIONS, STATION_COPIES)

# choose a canonical depot copy
DEPOT_BASE = DEPOT_NAME
DEPOT_NAME = f"{DEPOT_BASE}_0"

print(f"[INFO] New Depot Name: {DEPOT_NAME}")
print(f"[INFO] Expanded Chargers (copies only): {CHARGERS}")


# ------------------------------ Ref dict: location -> ref ------------------------------
loc_col = next((c for c in df_ref_dict.columns if str(c).strip().lower() == "location"), None)
ref_col = next((c for c in df_ref_dict.columns if str(c).strip().lower() == "ref"), None)
if loc_col is None or ref_col is None:
    raise ValueError(
        f"{ref_dict_csv.name} must include columns 'Location' and 'Ref'. "
        f"Found: {list(df_ref_dict.columns)}"
    )

loc_to_ref = {}
for _, row in df_ref_dict.iterrows():
    loc = _norm_token(row[loc_col])
    ref = _normalize_ref(row[ref_col])
    if loc is None or ref is None:
        continue
    loc_to_ref[loc] = ref

# also register base-name aliases (e.g., 2190L_0 -> 2190L)
for loc, ref in list(loc_to_ref.items()):
    loc_to_ref.setdefault(strip_copy_suffix(loc), ref)


# ------------------------------ Ref deadheads: symmetric pair lookup ------------------------------
if len(df_ref_dhd.columns) < 4:
    raise ValueError(
        f"{ref_dhd_csv.name} must have at least 4 columns "
        f"(Start Place, End Place, Base Duration, Energy used). "
        f"Found: {list(df_ref_dhd.columns)}"
    )

start_ref_col, end_ref_col, dur_col, en_col = list(df_ref_dhd.columns[:4])

ref_pair_minutes_kwh = {}
duplicate_rows = 0

for _, row in df_ref_dhd.iterrows():
    ref_a = _normalize_ref(row[start_ref_col])
    ref_b = _normalize_ref(row[end_ref_col])
    dur_min = pd.to_numeric(row[dur_col], errors="coerce")
    eng_kwh = pd.to_numeric(row[en_col], errors="coerce")

    if ref_a is None or ref_b is None or pd.isna(dur_min) or pd.isna(eng_kwh):
        continue
    if ref_a == ref_b:
        continue

    key = _ordered_ref_pair(ref_a, ref_b)
    val = (float(dur_min), float(eng_kwh))
    prev = ref_pair_minutes_kwh.get(key)

    # Keep shortest duration variant if duplicate pairs exist.
    if prev is None or val[0] < prev[0]:
        if prev is not None:
            duplicate_rows += 1
        ref_pair_minutes_kwh[key] = val
    else:
        duplicate_rows += 1

known_refs = set()
for a_ref, b_ref in ref_pair_minutes_kwh.keys():
    known_refs.add(a_ref)
    known_refs.add(b_ref)

print(
    f"[INFO] Ref deadhead pairs loaded: {len(ref_pair_minutes_kwh)} "
    f"(ignored duplicate rows: {duplicate_rows})"
)


def resolve_ref(node_name):
    raw = _norm_token(node_name)
    if raw is None:
        return None

    base = strip_copy_suffix(raw)
    candidates = [raw, base, _normalize_ref(raw), _normalize_ref(base)]

    for c in candidates:
        if c is None:
            continue
        if c in loc_to_ref:
            return loc_to_ref[c]
        if c in known_refs:
            return c
    return None


def arc_from_to(from_node: str, to_node: str):
    a = _norm_token(from_node)
    b = _norm_token(to_node)
    if a is None or b is None:
        return None
    if a == b:
        return (0, 0.0)

    # # Avoid copy-to-copy teleporting for the same physical station.
    # if _is_copy_node(a) and _is_copy_node(b):
    #     if strip_copy_suffix(a) == strip_copy_suffix(b):
    #         return None

    ref_a = resolve_ref(a)
    ref_b = resolve_ref(b)
    if ref_a is None or ref_b is None:
        return None
    if ref_a == ref_b:
        return (0, 0.0)

    pair = ref_pair_minutes_kwh.get(_ordered_ref_pair(ref_a, ref_b))
    if pair is None:
        return None

    dur_min, eng_kwh = pair
    return (ceil_blocks_from_minutes(dur_min), float(eng_kwh))


# ------------------------------ Build trip set ------------------------------

df_trips = df_trips.reset_index(drop=True).copy()
df_trips["Trip"] = df_trips.index

df_trips["st_blk"] = df_trips["ST"].astype(str).map(_floor_block)
df_trips["et_blk"] = df_trips["ET"].astype(str).map(_ceil_block)

df_trips["eps_kwh"] = df_trips["Energy used"].astype(float)

T = list(df_trips["Trip"].tolist())

sl = df_trips.set_index("Trip")["SL"].to_dict()
el = df_trips.set_index("Trip")["EL"].to_dict()
st = df_trips.set_index("Trip")["st_blk"].to_dict()
et = df_trips.set_index("Trip")["et_blk"].to_dict()
epsilon = df_trips.set_index("Trip")["eps_kwh"].to_dict()

# Arc costs/times are now resolved by ref lookup via arc_from_to().

# ------------------------------ Globals for pricing ------------------------------

S = CHARGERS[:]  # stations set (includes depot name too)
# G = energy_to_events(G_KWH)  
DEPOT = DEPOT_NAME

nodes_to_check = set(S + [DEPOT] + list(sl.values()) + list(el.values()))
missing_ref_nodes = sorted([n for n in nodes_to_check if resolve_ref(n) is None])
if missing_ref_nodes:
    print(f"[WARN] Missing Ref_dict mapping for {len(missing_ref_nodes)} nodes:")
    print(missing_ref_nodes[:30])

tau = {}
d   = {}  # deadhead energy (kWh) for each pricing arc

# Depot <-> trip
for i in T:
    pair = arc_from_to(DEPOT, sl[i])
    if pair is not None:
        tau[(DEPOT, i)] = pair[0]; d[(DEPOT, i)] = pair[1]
    pair = arc_from_to(el[i], DEPOT)
    if pair is not None:
        tau[(i, DEPOT)] = pair[0]; d[(i, DEPOT)] = pair[1]

# Trip -> Trip
for i in T:
    for j in T:
        if i == j: continue
        pair = arc_from_to(el[i], sl[j])
        if pair is not None:
            tau[(i, j)] = pair[0]; d[(i, j)] = pair[1]

# Trip <-> Station
for i in T:
    for h in S:
        pair1 = arc_from_to(el[i], h)
        if pair1 is not None:
            tau[(i, h)] = pair1[0]; d[(i, h)] = pair1[1]
        pair2 = arc_from_to(h, sl[i])
        if pair2 is not None:
            tau[(h, i)] = pair2[0]; d[(h, i)] = pair2[1]

# Zero-hop station-trip links are implicit when location refs match in arc_from_to().


# Station <-> Depot
for h in S:
    pair1 = arc_from_to(DEPOT, h)
    if pair1 is not None:
        tau[(DEPOT, h)] = pair1[0]; d[(DEPOT, h)] = pair1[1]
    pair2 = arc_from_to(h, DEPOT)
    if pair2 is not None:
        tau[(h, DEPOT)] = pair2[0]; d[(h, DEPOT)] = pair2[1]

def has_depot_pull(i):
    return ((DEPOT, i) in tau) and ((i, DEPOT) in tau)

unseedable = [i for i in T if not has_depot_pull(i)]
if unseedable:
    print("[WARN] Trips lacking depot pull-out or pull-in in DHD (cannot seed O->i->O):", unseedable)

# ------------------------------ Price curve ------------------------------

# price curve should be indexed by PHYSICAL station names (base), not copies
STATION_BASES = sorted(set(strip_copy_suffix(s) for s in S))

charging_cost_data, avg_cost_per_kwh = load_price_curve(
    str(prices_csv), time_blocks, STATION_BASES
)

bus_cost  = BUS_COST_KX # * avg_cost_per_kwh #??

print(f"[INFO] Capacity G: {G} kWh")
print(f"[INFO] Avg price/kWh={avg_cost_per_kwh:.3f}")
print(f"[INFO] bus_cost={bus_cost:.2f}")
print(f"[INFO] Trips={len(T)}  Stations={len(S)}")

# ------------------------------ Seed routes ------------------------------

R_truck = []


### instead of depot-trip-depot seeding, just make dummy routes for all trips
# for i in T:
#     if has_depot_pull(i):
#         g_ret = max(0, G - (d[(DEPOT, i)] + d[(i, DEPOT)] + epsilon[i]))
#         R_truck.append({
#             "route": [DEPOT, i, DEPOT],
#             "charging_stops": {
#                 "stations": [], "cst": [], "cet": [],
#                 "chi_plus_free": [], "chi_minus": [], "chi_minus_free": [],
#                 "chi_plus": [], "chi_zero": []
#             },
#             "charging_activities": 0,
#             "type": "truck",
#             "remaining_soc": g_ret
#         })

# # Dummy columns (BIG-M) for trips we couldn't seed
# BIG_M = 1e7
# dummy_count = 0
# for i in T:
#     if not has_depot_pull(i):
#         R_truck.append({
#             "route": [i],
#             "charging_stops": {
#                 "stations": [], "cst": [], "cet": [],
#                 "chi_plus_free": [], "chi_minus": [], "chi_minus_free": [],
#                 "chi_plus": [], "chi_zero": []
#             },
#             "charging_activities": 0,
#             "type": "truck",
#             "dummy": True,
#             "dummy_cost": BIG_M
#         })
#         dummy_count += 1

# print(f"[INFO] Seeded {len([r for r in R_truck if not r.get('dummy')])} real routes + {dummy_count} dummy routes (for uncovered trips).")

# ------------------------------ Build & solve master once ------------------------------

rmp, a, trip_cov = init_master(
    R_truck=R_truck,
    T=T,
    charging_cost_data=charging_cost_data,
    bus_cost=bus_cost,
    binary=False
)
# LP params for the RMP (per-iteration)
rmp.Params.Threads = THREADS
rmp.Params.NodefileStart = NODEFILE_START
rmp.Params.NodefileDir = _detect_tmp()
rmp.Params.Method = 1
rmp.Params.TimeLimit = MASTER_TIMELIMIT
rmp.optimize()

# ------------------------------ Pricing model (no layering, dwell-linked charge) ------------------------------

def build_pricing(alpha, beta, gamma, mode):
    pricing_model = Model("EV_Routing")
    pricing_model.Params.OutputFlag = 1
    pricing_model.Params.Threads = THREADS
    pricing_model.Params.NodefileStart = NODEFILE_START
    pricing_model.Params.NodefileDir = _detect_tmp()
    pricing_model.Params.Method = 1  # simplex in root
    pricing_model.Params.TimeLimit = PRICING_TIMELIMIT
    pricing_model.Params.MIPGap = PRICING_GAP
    pricing_model.Params.MIPFocus = 1
    pricing_model.Params.Heuristics = 0.8
    pricing_model.Params.Cuts = 0

    # --------- helper: strong pre-pruning for feasibility + short deadheads ---------
    # MAX_TAU = 2  # at most 2 hour-blocks between nodes in pricing graph

    def tt_ok(i, j):
        return (
            (i, j) in tau
            and (et[i] + tau[(i, j)] <= st[j])
            #and (tau[(i, j)] <= MAX_TAU)
        )

    def ih_ok(i, h):
        return (
            (i, h) in tau
            and (et[i] + tau[(i, h)] <= bar_t)
            #and (tau[(i, h)] <= MAX_TAU)
        )

    def hi_ok(h, i):
        return (
            (h, i) in tau
            and (tau[(h, i)] <= st[i])
            #and (tau[(h, i)] <= MAX_TAU)
        )

    # --------- sparse key sets (heavily pruned) ---------
    x_keys = [(i, j) for (i, j) in tau.keys()
              if isinstance(i, int) and isinstance(j, int) and tt_ok(i, j)]
    y_keys = [(i, h) for (i, h) in tau.keys()
              if isinstance(i, int) and isinstance(h, str) and (h in S) and ih_ok(i, h)]
    z_keys = [(h, i) for (h, i) in tau.keys()
              if isinstance(h, str) and (h in S) and isinstance(i, int) and hi_ok(h, i)]

    # Stations that actually appear in the pruned graph, plus any with depot links
    S_use = set(h for (_, h) in y_keys) | set(h for (h, _) in z_keys)
    for h in S:
        if (DEPOT, h) in d or (h, DEPOT) in d:
            S_use.add(h)
    S_use = sorted(S_use)

    forbid_start = {h for h in S_use if (DEPOT, h) not in d}
    forbid_end   = {h for h in S_use if (h, DEPOT) not in d}

    print(f"[PRICING] |S_use|={len(S_use)}  |x|={len(x_keys)}  |y|={len(y_keys)}  |z|={len(z_keys)}")

    # --------- variables ----------
    wA_trip        = pricing_model.addVars(T, vtype=GRB.BINARY, name="wA_trip")
    wOmega_trip    = pricing_model.addVars(T, vtype=GRB.BINARY, name="wOmega_trip")
    wA_station     = pricing_model.addVars(S_use, vtype=GRB.BINARY, name="wA")
    wOmega_station = pricing_model.addVars(S_use, vtype=GRB.BINARY, name="wOmega")

    x = pricing_model.addVars(x_keys, vtype=GRB.BINARY, name="x")
    y = pricing_model.addVars([(i, h) for (i, h) in y_keys if h in S_use], vtype=GRB.BINARY, name="y")
    z = pricing_model.addVars([(h, i) for (h, i) in z_keys if h in S_use], vtype=GRB.BINARY, name="z")

    cst = pricing_model.addVars(S_use, lb=0, ub=bar_t, vtype=GRB.INTEGER, name="cst")
    cet = pricing_model.addVars(S_use, lb=0, ub=bar_t, vtype=GRB.INTEGER, name="cet")
    chi_plus       = pricing_model.addVars(S_use, time_blocks, vtype=GRB.BINARY, name="chi_plus")
    chi_plus_free  = pricing_model.addVars(S_use, time_blocks, vtype=GRB.BINARY, name="chi_plus_free")
    chi_minus      = pricing_model.addVars(S_use, time_blocks, vtype=GRB.BINARY, name="chi_minus")
    chi_minus_free = pricing_model.addVars(S_use, time_blocks, vtype=GRB.BINARY, name="chi_minus_free")
    chi_zero       = pricing_model.addVars(S_use, time_blocks, vtype=GRB.BINARY, name="chi_zero")
    charge         = pricing_model.addVars(S_use, time_blocks, vtype=GRB.BINARY, name="charge")

    t_in  = {}
    t_out = {}
    for i in T:
        t_in[i]  = pricing_model.addVar(lb=1, ub=bar_t, vtype=GRB.INTEGER, name=f"tin_trip_{i}")
        t_out[i] = pricing_model.addVar(lb=1, ub=bar_t, vtype=GRB.INTEGER, name=f"tout_trip_{i}")
    for h in S_use:
        t_in[h]  = pricing_model.addVar(lb=0, ub=bar_t, vtype=GRB.INTEGER, name=f"tin_stat_{h}")
        t_out[h] = pricing_model.addVar(lb=0, ub=bar_t, vtype=GRB.INTEGER, name=f"tout_stat_{h}")
    t_in[DEPOT]  = pricing_model.addVar(lb=0, ub=bar_t, vtype=GRB.INTEGER, name="tin_depot")
    t_out[DEPOT] = pricing_model.addVar(lb=0, ub=bar_t, vtype=GRB.INTEGER, name="tout_depot")

    v_amt = {h: pricing_model.addVar(lb=-G, ub=G, vtype=GRB.CONTINUOUS, name=f"v_{h}") for h in S_use}
    if DEPOT in S_use:
        g = pricing_model.addVars(T + S_use, lb=0, ub=G, vtype=GRB.CONTINUOUS, name="g")
    else:
        g = pricing_model.addVars(T + S_use + [DEPOT], lb=0, ub=G, vtype=GRB.CONTINUOUS, name="g")
    g_return = pricing_model.addVar(lb=0, ub=G, vtype=GRB.CONTINUOUS, name="g_return")

    vars_dict = dict(
        wA_trip=wA_trip, wOmega_trip=wOmega_trip,
        wA_station=wA_station, wOmega_station=wOmega_station,
        x=x, y=y, z=z, cst=cst, cet=cet,
        chi_plus=chi_plus, chi_plus_free=chi_plus_free,
        chi_minus=chi_minus, chi_minus_free=chi_minus_free, chi_zero=chi_zero,
        g=g, g_return=g_return
    )

    pullout = quicksum(wA_station[h] for h in S_use) + quicksum(wA_trip[i] for i in T)
    pullin  = quicksum(wOmega_station[h] for h in S_use) + quicksum(wOmega_trip[i] for i in T)
    pricing_model.addConstr(pullout == pullin, name="start_end_balance")
    pricing_model.addConstr(pullout == 1, name="start_end_once_value")

    for h in forbid_start:
        pricing_model.addConstr(wA_station[h] == 0, name=f"no_start_{h}")
    for h in forbid_end:
        pricing_model.addConstr(wOmega_station[h] == 0, name=f"no_end_{h}")

    for h in S_use:
        in_h  = wA_station[h]  + quicksum(y[(i,h)] for (i,hk) in y.keys() if hk == h)
        out_h = wOmega_station[h] + quicksum(z[(h,i)] for (hk,i) in z.keys() if hk == h)
        pricing_model.addConstr(in_h == out_h, name=f"flow_charge_balance_{h}")

    for i in T:
        in_x  = quicksum(x[(j,i)] for (j,i2) in x.keys() if i2 == i)
        out_x = quicksum(x[(i,j)] for (i2,j) in x.keys() if i2 == i)
        in_z  = quicksum(z[(h,i)] for (h,i2) in z.keys() if i2 == i)
        out_y = quicksum(y[(i,h)] for (i2,h) in y.keys() if i2 == i)
        pricing_model.addConstr(
            wA_trip[i] + in_x + in_z ==
            wOmega_trip[i] + out_x + out_y,
            name=f"flow_trip_{i}"
        )

    for i in T:
        pricing_model.addConstr(t_in[i]  == st[i], name=f"tin_trip_fix_{i}")
        pricing_model.addConstr(t_out[i] == et[i], name=f"tout_trip_fix_{i}")

    for h in S_use:
        pricing_model.addConstr(cst[h] >= t_in[h],  name=f"cst_ge_tin_{h}")
        pricing_model.addConstr(cet[h] <= t_out[h], name=f"cet_le_tout_{h}")
        pricing_model.addConstr(cst[h] <= cet[h],   name=f"cst_le_cet_{h}")
        for t in time_blocks:
            pricing_model.addGenConstrIndicator(charge[h,t], 1, cst[h] - t, GRB.LESS_EQUAL, 0, name=f"ind_charge_cst_{h}_{t}")
            pricing_model.addGenConstrIndicator(charge[h,t], 1, cet[h] - t, GRB.GREATER_EQUAL, 0, name=f"ind_charge_cet_{h}_{t}")
            pricing_model.addConstr(chi_plus[h,t] + chi_zero[h,t] == charge[h,t], name=f"mode1_charge_modes_{h}_{t}")
            pricing_model.addConstr(chi_plus_free[h,t] == 0, name=f"mode1_no_free_{h}_{t}")
            pricing_model.addConstr(chi_minus[h,t]     == 0, name=f"mode1_no_dis_{h}_{t}")
            pricing_model.addConstr(chi_minus_free[h,t]== 0, name=f"mode1_no_disfree_{h}_{t}")



    # depot in out
    pricing_model.addConstr(t_in[DEPOT] <= t_out[DEPOT], name="depot_time_order")

    for i in T:
        if (DEPOT, i) in tau:
            pricing_model.addGenConstrIndicator(
                wA_trip[i], 1, t_in[i] - (t_in[DEPOT] + tau[(DEPOT, i)]), GRB.GREATER_EQUAL, 0,
                name=f"ind_time_O2trip_{i}"
            )
        if (i, DEPOT) in tau:
            pricing_model.addGenConstrIndicator(
                wOmega_trip[i], 1, t_out[DEPOT] - (t_out[i] + tau[(i, DEPOT)]), GRB.GREATER_EQUAL, 0,
                name=f"ind_time_trip2O_{i}"
            )
    for h in S_use:
        if (DEPOT, h) in tau:
            pricing_model.addGenConstrIndicator(
                wA_station[h], 1, t_in[h] - (t_in[DEPOT] + tau[(DEPOT, h)]), GRB.GREATER_EQUAL, 0,
                name=f"ind_time_O2stat_{h}"
            )
        if (h, DEPOT) in tau:
            pricing_model.addGenConstrIndicator(
                wOmega_station[h], 1, t_out[DEPOT] - (t_out[h] + tau[(h, DEPOT)]), GRB.GREATER_EQUAL, 0,
                name=f"ind_time_stat2O_{h}"
            )

    for (ii,jj) in x.keys():
        pricing_model.addGenConstrIndicator(
            x[(ii,jj)], 1, t_in[jj] - (t_out[ii] + tau[(ii, jj)]), GRB.GREATER_EQUAL, 0,
            name=f"ind_time_tt_{ii}_{jj}"
        )
    for (ii,h) in y.keys():
        pricing_model.addGenConstrIndicator(
            y[(ii,h)], 1, t_in[h] - (t_out[ii] + tau[(ii, h)]), GRB.GREATER_EQUAL, 0,
            name=f"ind_time_ts_{ii}_{h}"
        )
    for (h,ii) in z.keys():
        pricing_model.addGenConstrIndicator(
            z[(h,ii)], 1, t_in[ii] - (t_out[h] + tau[(h, ii)]), GRB.GREATER_EQUAL, 0,
            name=f"ind_time_st_{h}_{ii}"
        )

    for h in S_use:
        pricing_model.addConstr(
            v_amt[h] == CHARGE_PER_BLOCK *
                quicksum(chi_plus[(h, t)] for t in time_blocks),
             name=f"amt_charged_{h}"
        )

    pricing_model.addConstr(g[DEPOT] == G, name="initial_soc")

    for h in S_use:
        pricing_model.addConstr(g[h] + v_amt[h] <= G, name=f"soc_sum_ub_{h}")
        pricing_model.addConstr(g[h] + v_amt[h] >= 0, name=f"soc_sum_lb_{h}")
        if (DEPOT, h) in d:
            pricing_model.addGenConstrIndicator(
                wA_station[h], 1, g[h] - (g[DEPOT] - d[(DEPOT, h)]), GRB.EQUAL, 0,
                name=f"ind_soc_depot2station_{h}"
            )
        if (h, DEPOT) in d:
            pricing_model.addGenConstrIndicator(
                wOmega_station[h], 1, g[h] + v_amt[h] - d[(h, DEPOT)], GRB.GREATER_EQUAL, 0,
                name=f"ind_suff_stat2O_{h}"
            )
    for i in T:
        if (DEPOT, i) in d:
            pricing_model.addGenConstrIndicator(
                wA_trip[i], 1, g[i] - (g[DEPOT] - d[(DEPOT, i)]), GRB.EQUAL, 0,
                name=f"ind_soc_depot2trip_{i}"
            )
        if (i, DEPOT) in d:
            pricing_model.addGenConstrIndicator(
                wOmega_trip[i], 1, g[i] - (d[(i, DEPOT)] + epsilon[i]), GRB.GREATER_EQUAL, 0,
                name=f"ind_suff_trip2O_{i}"
            )

        for (ii,jj) in [key for key in x.keys() if key[0] == i]:
            pricing_model.addGenConstrIndicator(
                x[(ii,jj)], 1, g[jj] - (g[i] - epsilon[i] - d[(ii,jj)]), GRB.EQUAL, 0,
                name=f"ind_soc_trip2trip_{ii}_{jj}"
            )
        for (ii,h) in [key for key in y.keys() if key[0] == i]:
            pricing_model.addGenConstrIndicator(
                y[(ii,h)], 1, g[h] - (g[i] - epsilon[i] - d[(ii,h)]), GRB.EQUAL, 0,
                name=f"ind_soc_trip2station_{ii}_{h}"
            )
        for (h,ii) in [key for key in z.keys() if key[1] == i]:
            pricing_model.addGenConstrIndicator(
                z[(h,ii)], 1, g[i] - (g[h] + v_amt[h] - d[(h,ii)]), GRB.EQUAL, 0,
                name=f"ind_soc_station2trip_{h}_{ii}"
            )

    obj = bus_cost
    for h in S_use:
        if h in charging_cost_data.columns:
            for t in time_blocks:
                if t in charging_cost_data.index:
                    price_kwh = float(charging_cost_data.at[t, h])

                    obj += price_kwh * CHARGE_PER_BLOCK * chi_plus[h, t] * charge_cost_premium

    for i in T:
        a_i = alpha.get(i, 0.0)
        cov_expr = wA_trip[i] \
                   + quicksum(x[(j, i)] for (j, i2) in x.keys() if i2 == i) \
                   + quicksum(z[(h, i)] for (h, i2) in z.keys() if i2 == i)
        obj -= a_i * cov_expr


    # ---------------------------------------------------------
    # NEW: Symmetry Breaking (cst_0 <= cst_1 <= ...)
    # ---------------------------------------------------------
    # 1. Group stations by their base name
    station_groups = {}
    for h in S_use:
        # Check for suffix pattern like "Name_0", "Name_1"
        if "_" in str(h) and str(h).split("_")[-1].isdigit():
            base = str(h).rsplit("_", 1)[0]
            if base not in station_groups:
                station_groups[base] = []
            station_groups[base].append(h)
    
    # 2. Add constraints for each group
    symmetry_count = 0
    for base, nodes in station_groups.items():
        # Sort by suffix index to ensure correct order (_0, then _1, then _2)
        nodes.sort(key=lambda x: int(x.split("_")[-1]))
        
        for k in range(len(nodes) - 1):
            h_curr = nodes[k]
            h_next = nodes[k+1]
            
            # Constraint: cst[h_curr] <= cst[h_next]
            pricing_model.addConstr(
                cst[h_curr] <= cst[h_next],
                name=f"sym_break_{h_curr}_{h_next}"
            )
            symmetry_count += 1
            
    print(f"[PRICING] Added {symmetry_count} symmetry breaking constraints.")
    # ---------------------------------------------------------

    pricing_model.setObjective(obj, GRB.MINIMIZE)
    return pricing_model, vars_dict




# --- CONFIGURATION ---
PRICING_TLIM_INIT = 10
PRICING_TLIM_MAX  = 300
PRICING_TLIM_GROW = 2.0   # multiply TL when we retry
PRICING_TLIM_DECAY = 0.8  # shrink TL after success (optional)
PRICING_POOL_CAP_MAX = 30 # keep pool small for CG; 10–30 is usually plenty

# --- HELPER FUNCTIONS ---
def solve_pricing_fast(alpha, beta, gamma, mode, num_fast_cols=10, time_limit=10, *,
                       best_obj_stop=None):
    """
    Solves pricing using aggressive heuristics and Barrier method.
    Optimized for finding ANY negative reduced cost columns quickly.
    """
    cap = min(int(num_fast_cols), PRICING_POOL_CAP_MAX)
    m, vars_dict = build_pricing(alpha, beta, gamma, mode)

    # 1. Hard runtime cap for this call
    m.Params.TimeLimit = int(time_limit)

    # 2. Root-LP speed (Critical for large models)
    m.Params.Method = -1           # Barrier method 2
    m.Params.Crossover = 0        # Disable crossover (saves ~30% time, we don't need a basis)

    # 3. Incumbent-hunting (CG wants good columns, not tight bounds)
    m.Params.MIPFocus   = 1       # Focus on Feasibility
    m.Params.Heuristics = 0.7     # 60% time on heuristics
    m.Params.Cuts       = 0       # Disable cuts (saves time)

    # 4. NoRel Heuristic (Great for finding initial solutions quickly)
    if time_limit >= 20:
        m.Params.NoRelHeurTime = min(5, int(time_limit) - 1)
    else:
        m.Params.NoRelHeurTime = 0

    # 5. Early stop (Optional)
    if best_obj_stop is not None:
        m.Params.BestObjStop = float(best_obj_stop)

    # 6. Pool settings
    m.Params.PoolSearchMode = 1
    m.Params.PoolSolutions  = cap
    m.Params.SolutionLimit  = 2000 
    
    # [FIX] REMOVED Invalid Parameter PoolObjBound
    # Filtering happens in the extraction phase instead.

    m.optimize()
    return m, vars_dict






def solve_pricing_exact(alpha, beta, gamma, mode, num_exact_cols=10, time_limit=60):
    """
    Solves pricing exactly to prove optimality or find hard-to-reach columns.
    """
    m, vars_dict = build_pricing(alpha, beta, gamma, mode)

    m.Params.TimeLimit = int(time_limit)
    
    # Use Barrier for the root node speedup
    m.Params.Method = 2
    m.Params.Crossover = 0 

    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions  = int(num_exact_cols)
    
    # [FIX] REMOVED Invalid Parameter PoolObjBound

    m.optimize()
    return m, vars_dict


def _collect_candidates_from_pool(pricing_model, vars_pr, *, T, bar_t, DEPOT, RC_EPSILON):
    """
    Safely extracts routes from the Gurobi pool.
    Returns: candidates(list), best_rc(float), neg_in_pool(int), extract_fail(int)
    """
    candidates = []
    best_rc = float("inf")
    neg_in_pool = 0
    extract_fail = 0

    # Get the list of station keys once
    station_list = list(vars_pr["wA_station"].keys())

    for sol in range(pricing_model.SolCount):
        pricing_model.Params.SolutionNumber = sol
        rc = pricing_model.PoolObjVal
        
        if rc < best_rc:
            best_rc = rc

        if rc < -RC_EPSILON:
            neg_in_pool += 1
            try:
                # Use your existing extraction logic
                truck = extract_route_from_solution(
                    vars_pr, T, station_list, bar_t,
                    depot=DEPOT,
                    value_getter=lambda v: v.Xn
                )
                truck["_rc"] = rc
                candidates.append(truck)
            except Exception as e:
                # Log error but don't crash
                # print(f"[WARN] Failed to extract candidate {sol}: {e}")
                extract_fail += 1

    return candidates, best_rc, neg_in_pool, extract_fail

# ------------------------------ DIAGNOSTICS: list missing depot arcs ------------------------------
diag_dir = OUTDIR / f"diag_{RUN_ID}"
diag_dir.mkdir(parents=True, exist_ok=True)
missing_pullout = [i for i in T if arc_from_to(DEPOT, sl[i]) is None]
missing_pulluin = [i for i in T if arc_from_to(el[i], DEPOT) is None]
print(f"[DIAG] Trips missing PARX -> SL: {len(missing_pullout)}")
print(f"[DIAG] Trips missing EL -> PARX: {len(missing_pulluin)}")
pd.DataFrame({"Trip": missing_pullout, "SL": [sl[i] for i in missing_pullout]}).to_csv(diag_dir / "missing_pullout.csv", index=False)
pd.DataFrame({"Trip": missing_pulluin, "EL": [el[i] for i in missing_pulluin]}).to_csv(diag_dir / "missing_pulluin.csv", index=False)
print(f"[WRITE] Diagnostics saved under {diag_dir}")

#%%
# ------------------------------ CG loop ------------------------------

iteration = 0
new_pricing_obj = -1.0
max_iter = MAX_CG_ITERS

if len(R_truck) == 0:
    print("[WARN] No initial seed routes; master may be infeasible if some trips lack any coverable pattern.")

master_times = []
pricing_times = []
iter_rows = []

def _route_key(route):
    return tuple(route["route"])

best_master = float("inf")

PRICING_TLIM_INIT = 15
PRICING_TLIM_MAX  = 300

STAGNATION_LIMIT = 3
MIN_IMPROVEMENT = 1.5  # master obj improvement must be at least

stagnant_counter = 0
last_master_obj = None   #start as None
current_pricing_timelimit = PRICING_TLIM_INIT


# For deduplication of routes

# seen_patterns = set()
# for r in R_truck:
#     pattern = frozenset([x for x in r["route"] if isinstance(x, int)])
#     seen_patterns.add(pattern)

while iteration < max_iter:
    iteration += 1
    print(f"\n--- Iteration {iteration} ---")

    # 1) SOLVE MASTER
    t0 = time.time()
    rmp.Params.TimeLimit = MASTER_TIMELIMIT
    rmp.optimize()
    master_times.append(time.time() - t0)
    print(f" Master obj: {rmp.ObjVal:.2f}")

    # Check stagnation (optional, keep your existing logic here)
    # ...

    current_obj = rmp.ObjVal
    # compute improvement safely
    if last_master_obj is None:
        improvement = float("inf")
    else:
        improvement = last_master_obj - current_obj

    print(f" Master obj: {current_obj:.2f} (Impv: {improvement:.4f})")

    # B. STAGNATION CHECK
    if improvement < MIN_IMPROVEMENT:
        stagnant_counter += 1
        print(f"   [WARN] Stagnant {stagnant_counter}/{STAGNATION_LIMIT}")
        if stagnant_counter >= STAGNATION_LIMIT:
            print("[STOP] Master stabilized. Converged.")
            break
    else:
        stagnant_counter = 0
        
    last_master_obj = current_obj

    # Extract Duals
    alpha, beta_dual, gamma_dual = extract_duals(rmp)

    # 2) SOLVE PRICING (Adaptive Retry Loop)
    new_trucks = []
    best_rc_iter = float("inf")
    timed_out_any = False

    # For deduplication
    seen_keys_existing = {_route_key(r) for r in R_truck}

    # Configuration for this iteration
    BEST_OBJ_STOP = None         # Set to e.g. -5000.0 if you want early stops
    
    while True:
        # time limit cap
        current_pricing_timelimit = min(current_pricing_timelimit, PRICING_TLIM_MAX)

        print(f"   > FAST pricing (TimeLimit={current_pricing_timelimit}s)")

        t0_price = time.time()
        pricing_model, vars_pr = solve_pricing_fast(
            alpha, beta_dual, gamma_dual,
            mode=1,
            num_fast_cols=n_fast_cols,
            time_limit=current_pricing_timelimit,
            best_obj_stop=BEST_OBJ_STOP
            
        )

        pricing_times.append(time.time() - t0_price)

        # check status
        timed_out_fast = (pricing_model.Status == GRB.TIME_LIMIT)
        timed_out_any |= timed_out_fast

        # Collect candidates
        candidates, best_rc_fast, neg_in_pool, extract_fail = _collect_candidates_from_pool(
            pricing_model, vars_pr, T=T, bar_t=bar_t, DEPOT=DEPOT, RC_EPSILON=RC_EPSILON
        )
        best_rc_iter = min(best_rc_iter, best_rc_fast)

        # Sort and filter unique candidates
        candidates.sort(key=lambda r: r["_rc"])
        seen_new = set()
        for t_route in candidates:
            k = _route_key(t_route)
            if (k not in seen_keys_existing) and (k not in seen_new):
                new_trucks.append(t_route)
                seen_new.add(k)
            if len(new_trucks) >= K_BEST:
                break

        # A) SUCCESS: We found columns
        if new_trucks:
            print(f"   [SUCCESS] Found {len(new_trucks)} cols (best_rc={best_rc_iter:.1f})")
            # Decay time limit slightly to keep next iter fast
            current_pricing_timelimit = max(PRICING_TLIM_INIT, int(current_pricing_timelimit * PRICING_TLIM_DECAY))
            break

        # B) FALLBACK: Fast pricing found nothing, but maybe Exact will?
        # Only run if fast saw "potential" (neg_in_pool > 0) but failed to extract, 
        # OR if we want to be exhaustive.
        ran_exact = False
        if (best_rc_fast < -RC_EPSILON) and (n_exact_cols > 0) and not new_trucks:
            ran_exact = True
            print(f"   > EXACT pricing (TimeLimit={current_pricing_timelimit}s) "
                  f"[neg_in_pool={neg_in_pool}, extract_fail={extract_fail}]")

            pricing_model2, vars_pr2 = solve_pricing_exact(
                alpha, beta_dual, gamma_dual,
                mode=1,
                num_exact_cols=n_exact_cols,
                time_limit=current_pricing_timelimit
            )
            timed_out_exact = (pricing_model2.Status == GRB.TIME_LIMIT)
            timed_out_any |= timed_out_exact

            candidates2, best_rc_exact, neg_in_pool2, extract_fail2 = _collect_candidates_from_pool(
                pricing_model2, vars_pr2, T=T, bar_t=bar_t, DEPOT=DEPOT, RC_EPSILON=RC_EPSILON
            )
            best_rc_iter = min(best_rc_iter, best_rc_exact)

            # Process exact candidates
            candidates2.sort(key=lambda r: r["_rc"])
            seen_new = set()
            for t_route in candidates2:
                k = _route_key(t_route)
                if (k not in seen_keys_existing) and (k not in seen_new):
                    new_trucks.append(t_route)
                    seen_new.add(k)
                if len(new_trucks) >= K_BEST:
                    break
            
            if new_trucks:
                print(f"   [SUCCESS] Found {len(new_trucks)} cols after EXACT (best_rc={best_rc_iter:.1f})")
                current_pricing_timelimit = max(PRICING_TLIM_INIT, int(current_pricing_timelimit * PRICING_TLIM_DECAY))
                break

        # C) OPTIMAL STOP: No timeout occurred, and best RC is non-negative
        if (not timed_out_any) and (best_rc_iter >= -RC_EPSILON):
            print(f"   [RC-OPT] Pricing solved (no timeout) and best_rc={best_rc_iter:.1f} >= -RC_EPSILON")
            break

        # D) TIMEOUT / GIVE UP: We hit max time limit
        if current_pricing_timelimit >= PRICING_TLIM_MAX:
            print(f"   [GIVE UP] Hit max pricing TL={PRICING_TLIM_MAX}s; continuing without new cols.")
            break

        # E) RETRY: We timed out with no columns. Double time and loop again.
        new_tlim = min(PRICING_TLIM_MAX, int(current_pricing_timelimit * PRICING_TLIM_GROW))
        print(f"   [RETRY] No cols found (best_rc={best_rc_iter:.1f}, timeout={timed_out_any}). "
              f"Increasing TimeLimit: {current_pricing_timelimit}s -> {new_tlim}s")
        current_pricing_timelimit = new_tlim
        # Loop continues...

    # 3) ADD COLUMNS TO MASTER
    for route in new_trucks:
        print(f"[ADD] column rc={route.get('_rc', float('nan')):.1f}  route={route['route']}")
        R_truck.append(route)
        cost = calculate_truck_route_cost(route, bus_cost, charging_cost_data)

        col = Column()
        for node in route["route"]:
            if isinstance(node, int):
                col.addTerms(1.0, trip_cov[node])

        idx = len(R_truck) - 1
        a[idx] = rmp.addVar(obj=cost, lb=0, ub=1, vtype=GRB.CONTINUOUS, column=col, name=f"a[{idx}]")
    rmp.update()

    # 4) CHECK TERMINATION
    # Only stop if we truly found nothing AND we didn't time out
    if (not new_trucks) and (not timed_out_any) and (best_rc_iter >= -RC_EPSILON):
        print(f"[STOP] Reduced-cost optimal (best_rc={best_rc_iter:.1f}).")
        break
#%%
# ---------------- DIAGNOSTIC START ----------------
# Check which trips are still using dummy variables in the LP solution

print("\n--- Solving RMP one last time for diagnostics ---")
rmp.optimize()  # <--- ADD THIS LINE. It restores .X values.

print("\n--- Uncovered Trips Diagnostic ---")
uncovered_trips = []
for i in T:
    q_var = rmp.getVarByName(f"q_{i}")
    if q_var and q_var.X > 0.01:  # If slack is non-zero
        uncovered_trips.append(i)

if uncovered_trips:
    print(f"[WARN] The following {len(uncovered_trips)} trips are covered by DUMMY variables (q_i=1):")
    print(uncovered_trips)
    print("These trips likely have no valid incoming/outgoing arcs in the pricing graph.")
else:
    print("[SUCCESS] All trips are covered by real vehicle routes.")
# ---------------- DIAGNOSTIC END ------------------

#%%
# ------------------------------ Final solve (LP then MIP warm-start) ------------------------------

rmp_lp, a_lp = solve_master(
    R_truck=R_truck,
    T=T,
    charging_cost_data=charging_cost_data,
    bus_cost=bus_cost,
    binary=False
)
final_LP_obj = rmp_lp.ObjVal

rmp_final, a_final, trip_cov_final = build_master(
    R_truck=R_truck,
    T=T,
    charging_cost_data=charging_cost_data,
    bus_cost=bus_cost,
    binary=True
)


# ---------------- ENFORCE DUMMY =0  ----------------
# print("[FINAL] Locking out dummy variables (forcing q_i = 0)...")
# locked_count = 0
# for i in T:
#     # Retrieve the slack variable by name
#     q_var = rmp_final.getVarByName(f"q_{i}")
#     if q_var is not None:
#         # Force it to 0. The solver MUST cover trip i with a real vehicle OR return Infeasible.
#         q_var.UB = 0.0 
#         locked_count += 1

# print(f"[FINAL] Locked {locked_count} dummy variables.")
# ---------------- ENFORCE DUMMY =0  ----------------



for idx, var in a_final.items():
    if idx in a_lp:
        var.start = a_lp[idx].X

# safer final MIP params
rmp_final.Params.Threads = THREADS
rmp_final.Params.NodefileStart = NODEFILE_START
rmp_final.Params.NodefileDir = _detect_tmp()
rmp_final.Params.MIPFocus = 1
rmp_final.Params.Heuristics = 0.5
rmp_final.Params.Cuts = 1
rmp_final.Params.MIPGap = 0.03
rmp_final.Params.TimeLimit = 600  # brief polish
rmp_final.Params.LPWarmStart = 2

rmp_final.optimize()
final_MIP_obj = rmp_final.ObjVal
 
print("\n=== Selected truck routes ===")
used_routes = []
for r in range(len(R_truck)):
    if r in a_final and a_final[r].X > 0.5:
        used_routes.append(r)
        print(f"Route {r}: a[{r}]={a_final[r].X:.0f}  -> {R_truck[r]}")

print("\n Master LP obj:", final_LP_obj)
print(" Master MIP obj:", final_MIP_obj)
print(f" Buses used: {len(used_routes)}")

pd.DataFrame(iter_rows).to_csv(OUTDIR / f"iterations_{RUN_ID}.csv", index=False)
print(f"[WRITE] {OUTDIR / ('iterations_' + RUN_ID + '.csv')}")
try:
    rmp_final.write(str(OUTDIR / f"solution_{RUN_ID}.sol"))
except Exception:
    pass

dummy_used = [r for r in used_routes if R_truck[r].get("dummy", False)]
real_used  = [r for r in used_routes if not R_truck[r].get("dummy", False)]
print(f" Dummy routes used: {len(dummy_used)} / {len(used_routes)}")
print(f" Real routes used : {len(real_used)} / {len(used_routes)}")


# arc stats
# print(f"[arc stats] direct={direct_hits} fallback(ref->ref)={fallback_hits} mixed={mixed_hits} misses={misses}")



stopwatch_end = time.time()
elapsed = stopwatch_end - stopwatch_start
print(f"\n=== CG Loop Completed in {elapsed:.1f} seconds ===")

# %%

# # --- PHASE 2: TARGETING UNCOVERED TRIPS ---

# print("\n\n>>> STARTING PHASE 2: CLEANING UP UNCOVERED TRIPS <<<\n")

# # Configuration for Phase 2
# PHASE2_MAX_ITERS = 200      # Safety cap so it doesn't run forever
# TARGET_MAX_ALPHA = 900.0   # 
# current_pricing_timelimit = 15 # Start slightly higher for stubborn trips

# # We continue using the EXISTING R_truck and rmp from memory
# # No need to rebuild the RMP from scratch unless you closed the object

# iteration_p2 = 0
# while iteration_p2 < PHASE2_MAX_ITERS:
#     iteration_p2 += 1
#     print(f"\n--- Phase 2 Iteration {iteration_p2} ---")

#     # 1. SOLVE MASTER (Reuse existing model)
#     t0 = time.time()
#     rmp.Params.TimeLimit = MASTER_TIMELIMIT
#     rmp.optimize()
#     print(f" Master obj: {rmp.ObjVal:.2f}")

#     # 2. CHECK STATUS (Duals & Dummies)
#     alpha, beta_dual, gamma_dual = extract_duals(rmp)
    
#     # Calculate max alpha specifically for TRIP constraints
#     # (Assuming alpha is a dict/list corresponding to trip indices)
#     if isinstance(alpha, dict):
#         max_alpha = max(alpha.values()) if alpha else 0
#     else:
#         max_alpha = max(alpha) if len(alpha) > 0 else 0

#     # Count active dummies
#     # (Assuming you have a list/dict of dummy variables 'q' or can infer from slack)
#     # A simple way to check coverage is checking if 'alpha' is close to your Big-M penalty
#     # But checking the RMP variables is safer if you have the handles:
#     # active_dummies = sum(1 for v in dummy_vars if v.X > 0.5) 
    
#     print(f"   [STATUS] Max Alpha: {max_alpha:.1f} (Target: {TARGET_MAX_ALPHA})")
    
#     # STOP CONDITION
#     if max_alpha <= TARGET_MAX_ALPHA:
#         print(f"[STOP] Success! Max alpha {max_alpha:.1f} is below threshold {TARGET_MAX_ALPHA}.")
#         break

#     # 3. SOLVE PRICING (Standard Adaptive Logic)
#     new_trucks = []
#     best_rc_iter = float("inf")
#     timed_out_any = False
#     seen_keys_existing = {_route_key(r) for r in R_truck}
    
#     # Configuration for "stubborn" trips
#     # We allow exact pricing to run earlier if fast pricing fails
#     BEST_OBJ_STOP = -100.0 # Stop fast pricing if we find ANY saving
    

#     while True:
#         print(f"   > Pricing (TimeLimit={current_pricing_timelimit}s)")
        
#         # ... [Reuse your existing solve_pricing_fast call] ...
#         pricing_model, vars_pr = solve_pricing_fast(
#             alpha, beta_dual, gamma_dual,
#             mode=1,
#             num_fast_cols=n_fast_cols,
#             time_limit=current_pricing_timelimit,
#             best_obj_stop=BEST_OBJ_STOP
#         )
        
#         timed_out_fast = (pricing_model.Status == GRB.TIME_LIMIT)
#         timed_out_any |= timed_out_fast
        
#         candidates, best_rc_fast, neg_in_pool, extract_fail = _collect_candidates_from_pool(
#             pricing_model, vars_pr, T=T, bar_t=bar_t, DEPOT=DEPOT, RC_EPSILON=RC_EPSILON
#         )
#         best_rc_iter = min(best_rc_iter, best_rc_fast)

#         # Standard extraction logic...
#         candidates.sort(key=lambda r: r["_rc"])
#         seen_new = set()
#         for t_route in candidates:
#             k = _route_key(t_route)
#             if (k not in seen_keys_existing) and (k not in seen_new):
#                 new_trucks.append(t_route)
#                 seen_new.add(k)
#             if len(new_trucks) >= K_BEST: break

#         # Success?
#         if new_trucks:
#             print(f"   [SUCCESS] Found {len(new_trucks)} cols (best_rc={best_rc_iter:.1f})")
#             current_pricing_timelimit = max(15, int(current_pricing_timelimit * 0.8)) # Decay
#             break
            
#         # Fallback to Exact? (Run this aggressively in Phase 2)
#         ran_exact = False
#         if not new_trucks and (neg_in_pool > 0 or n_exact_cols > 0):
#             print(f"   > Trying EXACT pricing (TimeLimit={current_pricing_timelimit}s)...")
#             pricing_model2, vars_pr2 = solve_pricing_exact(
#                 alpha, beta_dual, gamma_dual,
#                 mode=1,
#                 num_exact_cols=n_exact_cols,
#                 time_limit=current_pricing_timelimit
#             )
#             # ... [Extract candidates logic same as before] ...
#             # (If you need the full exact block pasted here let me know, 
#             # otherwise assume it's the same logic as your main loop)
#             # ...
            
#             # [Shortened for brevity - insert extraction logic here]
#             candidates2, _, _, _ = _collect_candidates_from_pool(
#                 pricing_model2, vars_pr2, T=T, bar_t=bar_t, DEPOT=DEPOT, RC_EPSILON=RC_EPSILON
#             )
#             for t_route in candidates2:
#                  k = _route_key(t_route)
#                  if (k not in seen_keys_existing) and (k not in seen_new):
#                      new_trucks.append(t_route)
#                      seen_new.add(k)
            
#             if new_trucks:
#                 print(f"   [SUCCESS] Found {len(new_trucks)} cols via Exact.")
#                 break

#         # Check for IMPOSSIBILITY
#         if not new_trucks and not timed_out_any and best_rc_iter >= -RC_EPSILON:
#             print(f"[STOP] Pricing is Optimal (rc >= 0). No more columns exist.")
#             print(f"[WARNING] We still have Max Alpha={max_alpha:.1f}. This means the remaining trips are INFEASIBLE to cover.")
#             iteration_p2 = PHASE2_MAX_ITERS # Force exit
#             break

#         # Retry Logic
#         if current_pricing_timelimit >= 300: # Max 5 mins
#             print("[GIVE UP] Max timer hit.")
#             break
            
#         current_pricing_timelimit = int(current_pricing_timelimit * 2.0)
#         print(f"   [RETRY] bumping time to {current_pricing_timelimit}s")

#     # 4. ADD COLUMNS (Same as before)
#     for route in new_trucks:
#         # ... [Add to R_truck and RMP logic] ...
#         R_truck.append(route)
#         cost = calculate_truck_route_cost(route, bus_cost, charging_cost_data)
#         col = Column()
#         for node in route["route"]:
#             if isinstance(node, int):
#                 col.addTerms(1.0, trip_cov[node])
#         idx = len(R_truck) - 1
#         a[idx] = rmp.addVar(obj=cost, lb=0, ub=1, vtype=GRB.CONTINUOUS, column=col, name=f"a[{idx}]")
    
#     rmp.update()


# stopwatch_end2 = time.time()
# elapsed = stopwatch_end2 - stopwatch_end
# print(f"\n=== 2nd CG Loop Completed in {elapsed:.1f} seconds ===")
# # %%



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
from gurobipy import Model, Column, GRB, quicksum, LinExpr
# from collections import Counter, defaultdict

from config import (
    n_fast_cols, n_exact_cols, tolerance,
    bar_t, # time_blocks,
    TIMEBLOCKS_PER_HOUR,
    DEPOT_NAME, G, CHARGE_PER_BLOCK, CHARGE_RATE_KW,
    charge_cost_premium,
    BUS_COST_KX, 
    CHARGING_STATIONS,
    STATION_COPIES,
    TRAVEL_COST_FACTOR, # new


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
MASTER_FILE = "Par_VehicleDetails_Updated.csv"
def parse_time_to_minutes(t_str):
    """Converts 'HH:MM' or 'H:MM' string to minutes from midnight."""
    if pd.isna(t_str): return 99999 
    try:
        parts = str(t_str).split(':')
        h = int(parts[0])
        m = int(parts[1])
        return h * 60 + m
    except:
        return 99999

def generate_specific_buses_instance(target_bus_ids, output_filename=None):
    """
    Creates a clean optimization dataset for a specific list of bus IDs.
    
    Args:
        target_bus_ids (list): List of vehicle IDs (e.g. [13405, 13411])
        output_filename (str): Optional custom filename.
    """
    # 1. Load Master Data
    df = pd.read_csv(DATA_DIR / MASTER_FILE)
    
    # Standardize types to string for comparison (some IDs might be int, some str)
    df['VehicleTask_Str'] = df['VehicleTask'].astype(str)
    target_ids_str = [str(x) for x in target_bus_ids]
    
    # 2. Filter for Regular trips AND the specific buses
    mask = (df['Identifier'] == 'Regular') & (df['VehicleTask_Str'].isin(target_ids_str))
    subset_df = df[mask].copy()
    
    if len(subset_df) == 0:
        print(f"[ERROR] No regular trips found for buses: {target_bus_ids}")
        return

    print(f"--- Generating Instance for {len(target_bus_ids)} Specific Buses ---")
    print(f"Targets: {target_bus_ids}")
    
    # 3. SORT CHRONOLOGICALLY (Crucial)
    # Create temporary minutes column to sort correctly
    subset_df['Sort_Time'] = subset_df['Start1'].apply(parse_time_to_minutes)
    subset_df = subset_df.sort_values('Sort_Time')
    subset_df = subset_df.drop(columns=['Sort_Time', 'VehicleTask_Str']) 
    
    # 4. RESET IDs
    subset_df['count_trip_id'] = range(len(subset_df))
    
    # 5. Save
    if output_filename is None:
        output_filename = f"Practice_Custom_{len(target_bus_ids)}buses.csv"
        
    subset_df.to_csv(DATA_DIR / output_filename, index=False)
    
    print(f"Created: {output_filename}")
    print(f"Total Trips: {len(subset_df)}")
    print(f"Unique Buses Found: {subset_df['VehicleTask'].nunique()}")
    print("-" * 30)

# ==========================================
# EXECUTE YOUR REQUEST
# ==========================================

# Create a file with exactly these 3 buses
# generate_specific_buses_instance([13405, 13411], "Practice_Selected_2buses.csv")
# generate_specific_buses_instance([13320, 13311, 13307 , 13314], "Practice_Selected_4bus.csv")

MAX_DAILY_RECHARGES = 4  # Buffer above observed max of 13
MIN_TRIPS_PER_ROUTE = 14  # Based on observed distribution (allowing some flexibility below the historical min of 17)

# generate_specific_buses_instance([13405, 13411, 13413], "Practice_Selected_3buses.csv")


#%%



routes_csv    = DATA_DIR / "Practice_Selected_3bus.csv"
ref_dhd_csv   = DATA_DIR / "par_ref_dhd.csv"
ref_dict_csv  = DATA_DIR / "Ref_dict.csv"
prices_csv    = DATA_DIR / "hourly_prices.csv"

# Create a dynamic name based on the input file (e.g., "Practice_Selected_4buses")
DATA_NAME = routes_csv.stem 
RUN_NAME = f"{DATA_NAME}_{RUN_ID}"

# Create a dedicated directory for EVERYTHING from this run
RUN_DIR = OUTDIR / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] All outputs and logs will be saved to: {RUN_DIR}")


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


df_p = pd.read_csv(prices_csv)
hourly_prices = df_p.set_index('time_block')['cost'].to_dict()
MAX_HOUR = int(max(hourly_prices.keys()))

# charging_cost_data, avg_cost_per_kwh = load_price_curve(
#     str(prices_csv), time_blocks, STATION_BASES
# )

bus_cost  = BUS_COST_KX # * avg_cost_per_kwh #??
print(f"[INFO] Loaded {len(hourly_prices)} hourly price points.")
print(f"[INFO] Max Hour: {MAX_HOUR}")

#%%
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
#%%
rmp, a, trip_cov = init_master(
    R_truck=R_truck,
    T=T,
    charging_cost_data=hourly_prices,
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
#%%
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
    pricing_model.Params.Heuristics = 0.5
    pricing_model.Params.Cuts = 0

    # --------- helper: strong pre-pruning for feasibility + short deadheads ---------
    # MAX_TAU = 2  # at most 2 hour-blocks between nodes in pricing graph
    max_trip2trip = 15

    # 2. Trip -> Charge (Historical max was 43)
    # Set to 60 to allow slightly more flexibility than history
    max_trip2charge = 60 

    # 3. Charge -> Trip (Historical max was 8)
    # 8 is very tight. 60 is safer to allow charging well before a trip starts.
    max_charge2trip = 60

    def tt_ok(i, j):
        return (
            (i,j) in tau and
            et[i] + tau[(i, j)] <= st[j] and
            (st[j] - et[i]) <= max_trip2trip
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
    # chi_plus       = pricing_model.addVars(S_use, time_blocks, vtype=GRB.BINARY, name="chi_plus")
    # chi_plus_free  = pricing_model.addVars(S_use, time_blocks, vtype=GRB.BINARY, name="chi_plus_free")
    # chi_minus      = pricing_model.addVars(S_use, time_blocks, vtype=GRB.BINARY, name="chi_minus")
    # chi_minus_free = pricing_model.addVars(S_use, time_blocks, vtype=GRB.BINARY, name="chi_minus_free")
    # chi_zero       = pricing_model.addVars(S_use, time_blocks, vtype=GRB.BINARY, name="chi_zero")
    # charge         = pricing_model.addVars(S_use, time_blocks, vtype=GRB.BINARY, name="charge")

    # --- ADD NEW VARIABLES ---
    hours = range(MAX_HOUR + 1)
    
    # u_hour[h, k] = 1 if station h starts charging in hour k
    u_hour = pricing_model.addVars(S_use, hours, vtype=GRB.BINARY, name="u_hour")

    # q_hour[h, k] = Amount (kWh) charged in hour k
    q_hour = pricing_model.addVars(S_use, hours, lb=0, ub=G, vtype=GRB.CONTINUOUS, name="q_hour")






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
        # chi_plus=chi_plus, chi_plus_free=chi_plus_free,
        # chi_minus=chi_minus, chi_minus_free=chi_minus_free, chi_zero=chi_zero,
        g=g, g_return=g_return,
        v_amt=v_amt
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
            
        # 2. Pick exactly one start hour bin
        pricing_model.addConstr(quicksum(u_hour[h, k] for k in hours) == 1, name=f"one_start_hour_{h}")

        # 3. Link 'cst' to the chosen hour (e.g. if u=1 for hour 10, cst must be 600-659)
        pricing_model.addConstr(cst[h] >= quicksum(k * 60 * u_hour[h, k] for k in hours), name=f"cst_lb_{h}")
        pricing_model.addConstr(cst[h] <= quicksum((k * 60 + 59) * u_hour[h, k] for k in hours), name=f"cst_ub_{h}")

        # 4. Define Amount based on Duration (Continuous)
        #    Assumes CHARGE_RATE_KW is adapted to your time units (e.g. kW/min)
        pricing_model.addConstr(v_amt[h] == (cet[h] - cst[h]) * CHARGE_PER_BLOCK, name=f"def_amt_{h}")

        # 5. Linearize Cost: Link q_hour to v_amt
        pricing_model.addConstr(quicksum(q_hour[h, k] for k in hours) == v_amt[h], name=f"sum_q_hour_{h}")
        
        for k in hours:
            # If hour k is NOT chosen, q_hour[k] must be 0
            pricing_model.addConstr(q_hour[h, k] <= G * u_hour[h, k], name=f"link_q_u_{h}_{k}")
            
        
        # for t in time_blocks:
        #     pricing_model.addGenConstrIndicator(charge[h,t], 1, cst[h] - t, GRB.LESS_EQUAL, 0, name=f"ind_charge_cst_{h}_{t}")
        #     pricing_model.addGenConstrIndicator(charge[h,t], 1, cet[h] - t, GRB.GREATER_EQUAL, 0, name=f"ind_charge_cet_{h}_{t}")
        #     pricing_model.addConstr(chi_plus[h,t] + chi_zero[h,t] == charge[h,t], name=f"mode1_charge_modes_{h}_{t}")
        #     pricing_model.addConstr(chi_plus_free[h,t] == 0, name=f"mode1_no_free_{h}_{t}")
        #     pricing_model.addConstr(chi_minus[h,t]     == 0, name=f"mode1_no_dis_{h}_{t}")
        #     pricing_model.addConstr(chi_minus_free[h,t]== 0, name=f"mode1_no_disfree_{h}_{t}")



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




    trips_covered = (
        quicksum(wA_trip[i] for i in T) + 
        quicksum(x[(i,j)] for (i,j) in x.keys()) + 
        quicksum(z[(h,i)] for (h,i) in z.keys())
    )
    
    pricing_model.addConstr(
        trips_covered >= MIN_TRIPS_PER_ROUTE,
        name="force_quality_route"
    )



    pricing_model.addConstr(
        quicksum(y[k] for k in y.keys()) + quicksum(wA_station[h] for h in S_use) <= MAX_DAILY_RECHARGES,
        name="max_recharge_count"
    )






    for h in S_use:
        # v_amt = Duration (minutes) * Rate (Energy/minute)
        pricing_model.addConstr(
            v_amt[h] == (cet[h] - cst[h]) * CHARGE_PER_BLOCK, 
            name=f"def_amt_{h}"
        )
        # pricing_model.addConstr(
        #     v_amt[h] == CHARGE_PER_BLOCK *
        #         quicksum(chi_plus[(h, t)] for t in time_blocks),
        #      name=f"amt_charged_{h}"
        # )








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

    # --- NEW: Add Travel (Deadhead) Cost ---
    # Sum d[u,v] * var[u,v] for all active arcs
    travel_expr = LinExpr()
    
    # 1. Trip -> Trip (x)
    for (i, j) in x.keys():
        if (i, j) in d: travel_expr += d[(i, j)] * x[(i, j)]
            
    # 2. Trip -> Station (y)
    for (i, h) in y.keys():
        if (i, h) in d: travel_expr += d[(i, h)] * y[(i, h)]
            
    # 3. Station -> Trip (z)
    for (h, i) in z.keys():
        if (h, i) in d: travel_expr += d[(h, i)] * z[(h, i)]
            
    # 4. Depot Connections (wA, wOmega)
    for i in T:
        if (DEPOT, i) in d: travel_expr += d[(DEPOT, i)] * wA_trip[i]
        if (i, DEPOT) in d: travel_expr += d[(i, DEPOT)] * wOmega_trip[i]
        
    for h in S_use:
        if (DEPOT, h) in d: travel_expr += d[(DEPOT, h)] * wA_station[h]
        if (h, DEPOT) in d: travel_expr += d[(h, DEPOT)] * wOmega_station[h]

    # Add to main objective
    obj += travel_expr * TRAVEL_COST_FACTOR


    for h in S_use:
        for k in hours:
            # Look up price for hour k
            price = hourly_prices.get(k, 100.0) 
            
            # Add cost: Price * Amount_in_that_bucket * Premium
            # Note: CHARGE_PER_BLOCK is NOT needed here if v_amt is already energy units (kWh).
            # Double check if your prices are $/kWh. If so:
            obj += price * q_hour[h, k] * charge_cost_premium


        # if h in charging_cost_data.columns:
        #     for t in time_blocks:
        #         if t in charging_cost_data.index:
        #             price_kwh = float(charging_cost_data.at[t, h])

        #             obj += price_kwh * CHARGE_PER_BLOCK * chi_plus[h, t] * charge_cost_premium

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
PRICING_POOL_CAP_MAX = 50 # keep pool small for CG; 10–30 is usually plenty

# --- HELPER FUNCTIONS ---
def solve_pricing_fast(alpha, beta, gamma, mode, num_fast_cols=10, time_limit=10, *,
                       best_obj_stop=None):
    """
    Solves pricing using aggressive heuristics and Barrier method.
    Optimized for finding ANY negative reduced cost columns quickly.
    """
    cap = min(int(num_fast_cols), PRICING_POOL_CAP_MAX)
    m, vars_dict = build_pricing(alpha, beta, gamma, mode)


    m.Params.LogFile = str(RUN_DIR / "pricing_fast.log")


    # 1. Hard runtime cap for this call
    m.Params.TimeLimit = int(time_limit)

    # 2. Root-LP speed (Critical for large models)
    m.Params.Method = 3           # Barrier method 2
    m.Params.Crossover = 0        # Disable crossover (saves ~30% time, we don't need a basis)

    # 3. Incumbent-hunting (CG wants good columns, not tight bounds)
    m.Params.MIPFocus   = 1       # Focus on Feasibility
    m.Params.Heuristics = 0.8     # % time on heuristics
    m.Params.NoRelHeurTime = 5
    #m.Params.Cuts       = 0       # Disable cuts (saves time)

    # 4. NoRel Heuristic (Great for finding initial solutions quickly)
    if time_limit >= 20:
        m.Params.NoRelHeurTime = min(5, int(time_limit) - 1)
    else:
        m.Params.NoRelHeurTime = 0

    # # 5. Early stop (Optional)
    # if best_obj_stop is not None:
    #     m.Params.BestObjStop = float(best_obj_stop)

    # 6. Pool settings
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions  = cap
    #m.Params.SolutionLimit  = 10 # no limit
    
    # [FIX] REMOVED Invalid Parameter PoolObjBound
    # Filtering happens in the extraction phase instead.

    m.optimize()
    return m, vars_dict






def solve_pricing_exact(alpha, beta, gamma, mode, num_exact_cols=10, time_limit=120):
    """
    Solves pricing exactly to prove optimality or find hard-to-reach columns.
    """
    m, vars_dict = build_pricing(alpha, beta, gamma, mode)

    m.Params.TimeLimit = int(time_limit)
    
    # Use Barrier for the root node speedup
    m.Params.Method = 2
    m.Params.Crossover = 0 
    m.Params.MIPGap = 1e-1
    m.Params.Presolve = 2

    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions  = int(num_exact_cols)
    
    # [FIX] REMOVED Invalid Parameter PoolObjBound


    m.Params.MIPFocus = 2
    m.Params.Heuristics = 0.4


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
                # --- NEW: Calculate & Store Total Deadhead ---
                # We traverse the extracted node list and sum 'd' for the edges
                route_nodes = truck["route"]
                total_d = 0.0
                for k in range(len(route_nodes) - 1):
                    u, v = route_nodes[k], route_nodes[k+1]
                    # Check if arc exists in global d
                    if (u, v) in d:
                        total_d += d[(u, v)]
                
                truck["deadhead_kwh"] = total_d
                # ---------------------------------------------

                truck["_rc"] = rc
                candidates.append(truck)
            except Exception as e:
                # Log error but don't crash
                # print(f"[WARN] Failed to extract candidate {sol}: {e}")
                extract_fail += 1
                print(f"[ERROR] Extraction failed: {e}")
                continue

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

STAGNATION_LIMIT = 10
MIN_IMPROVEMENT = 1.5  # master obj improvement must be at least

stagnant_counter = 0
last_master_obj = None   #start as None
current_pricing_timelimit = PRICING_TLIM_INIT




# For deduplication of routes

# seen_patterns = set()
# for r in R_truck:
#     pattern = frozenset([x for x in r["route"] if isinstance(x, int)])
#     seen_patterns.add(pattern)

cg_stats = []
# Create a unique filename based on the run ID so you don't overwrite previous experiments
stats_csv_path = RUN_DIR / f"pricing_stats.csv"
print(f"Saving real-time stats to: {stats_csv_path}")

#%%
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





    ####### stopping criteria for practice ######
        # --- TARGET CONFIGURATION ---
    TARGET_NUM_BUSES = 1  # Set to 1 for your '2minus1' file, or 2 for '2bus' file
    TARGET_OBJ = (TARGET_NUM_BUSES * BUS_COST_KX)

    print(f"temp Goal: Run until Master Objective <= {TARGET_OBJ:.2f}")

    # 2. CHECK TARGET CRITERIA
    # If we are effectively using the target number of buses (plus reasonable energy), STOP.
    if current_obj <= TARGET_OBJ:
        break


    # compute improvement safely
    if last_master_obj is None:
        improvement = float("inf")
    else:
        improvement = last_master_obj - current_obj

    print(f" Master obj: {current_obj:.2f} (Impv: {improvement:.4f})")

    
    # compute improvement safely
    if last_master_obj is None:
        improvement = float("inf")
    else:
        improvement = last_master_obj - current_obj

    print(f" Master obj: {current_obj:.2f} (Impv: {improvement:.4f})")

    stagnant_counter = 0 # Keep this here just so your cg_stats dictionary doesn't throw a NameError
        
    last_master_obj = current_obj

    # Extract Duals
    alpha, beta_dual, gamma_dual = extract_duals(rmp)


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
    
    t0_pricing_total = time.time()
    # --- PRICING RETRY LOOP ---
    while True:
        # Cap max time
        current_pricing_timelimit = min(current_pricing_timelimit, 120) 

        print(f"   > FAST pricing (TimeLimit={current_pricing_timelimit}s)")
        
        # Call your existing fast pricing function
        pricing_model, vars_pr = solve_pricing_fast(
            alpha, beta_dual, gamma_dual,
            mode=1,
            num_fast_cols=n_fast_cols,
            time_limit=current_pricing_timelimit
        )

        # Collect (We keep RC_EPSILON small here so we don't accidentally ignore good heuristics)
        candidates, best_rc_fast, neg_in_pool, _ = _collect_candidates_from_pool(
            pricing_model, vars_pr, T=T, bar_t=bar_t, DEPOT=DEPOT, RC_EPSILON=RC_EPSILON
        )
        best_rc_iter = min(best_rc_iter, best_rc_fast)

        # Filter duplicates
        candidates.sort(key=lambda r: r["_rc"])
        seen_new = set()
        for t_route in candidates:
            k = _route_key(t_route)
            if (k not in seen_keys_existing) and (k not in seen_new):
                new_trucks.append(t_route)
                seen_new.add(k)
            if len(new_trucks) >= K_BEST:
                break
        
        # 1. SUCCESS?
        if new_trucks:
            print(f"   [SUCCESS] Found {len(new_trucks)} cols (best_rc={best_rc_iter:.1f})")
            # Decay time slightly to stay aggressive for next iteration
            current_pricing_timelimit = max(15, int(current_pricing_timelimit * 0.9))
            break
            
        # 2. ESCALATING EXACT PRICING SCHEDULE
        print(f"   > FAST pricing found no columns. Switching to EXACT pricing schedule...")
        
        exact_time_schedule = [30, 60, 120, 240, 480]
        exact_success = False
        
        # We will require a reduced cost to be at least -1.0 to be considered "meaningful"
        SIGNIFICANT_RC_THRESHOLD = 1.0 

        for exact_tlim in exact_time_schedule:
            print(f"      [EXACT] Trying TimeLimit={exact_tlim}s...")
            pricing_model2, vars_pr2 = solve_pricing_exact(
                alpha, beta_dual, gamma_dual,
                mode=1,
                num_exact_cols=n_exact_cols,
                time_limit=exact_tlim
            )
            
            # Pass our SIGNIFICANT_RC_THRESHOLD so we ignore noise like -0.67
            candidates2, best_rc_exact, _, _ = _collect_candidates_from_pool(
                pricing_model2, vars_pr2, T=T, bar_t=bar_t, DEPOT=DEPOT, RC_EPSILON=SIGNIFICANT_RC_THRESHOLD
            )
            best_rc_iter = min(best_rc_iter, best_rc_exact)

            candidates2.sort(key=lambda r: r["_rc"])
            for t_route in candidates2:
                k = _route_key(t_route)
                if (k not in seen_keys_existing) and (k not in seen_new):
                    new_trucks.append(t_route)
                    seen_new.add(k)
                if len(new_trucks) >= K_BEST:
                    break
            
            if new_trucks:
                print(f"   [SUCCESS] Found {len(new_trucks)} cols after EXACT at {exact_tlim}s (best_rc={best_rc_iter:.2f})")
                exact_success = True
                break
            else:
                print(f"      [EXACT FAIL] No meaningful columns found (< -{SIGNIFICANT_RC_THRESHOLD}) at {exact_tlim}s. Best was {best_rc_exact:.2f}.")
        
        if exact_success:
            # We break out of the while True loop and go back to the Master LP
            break
        else:
            # 3. OPTIMAL STOP
            print(f"   [RC-OPT / STOP] EXACT pricing exhausted schedule up to {exact_time_schedule[-1]}s. Mathematical convergence reached.")
            break
    # --------------------------

    # --- [INSERT 2: Collect Metrics (After Retry Loop Ends)] ---
    pricing_dur_total = time.time() - t0_pricing_total
    
    current_stat = {
        "Iteration": iteration,
        "Master_Obj": current_obj,
        "Master_Improvement": improvement if last_master_obj is not None else 0,
        "Pricing_Time_s": pricing_dur_total,  # Total time for ALL pricing attempts this iter
        "Cols_Added": len(new_trucks),
        "Best_RC": best_rc_iter,
        "Timed_Out": timed_out_any,
        "Pricing_TimeLimit_Used": current_pricing_timelimit,
        "Stagnant_Counter": stagnant_counter,
        "Total_Runtime_s": time.time() - stopwatch_start
    }
    cg_stats.append(current_stat)
    
    # Force Save (Overwrites file every iteration)
    pd.DataFrame(cg_stats).to_csv(stats_csv_path, index=False)
    # -----------------------------------------------------------

    # 3) ADD COLUMNS TO MASTER
    for route in new_trucks:
        # print(f"[ADD] column rc={route.get('_rc', float('nan')):.1f}  route={route['route']}")
        # print(f"[ADD] rc={route.get('_rc', 0):.1f} | Path: {route['desc']}")
        R_truck.append(route)
        cost = calculate_truck_route_cost(route, bus_cost, hourly_prices)

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
    charging_cost_data=hourly_prices,
    bus_cost=bus_cost,
    binary=False
)
final_LP_obj = rmp_lp.ObjVal
#%%
rmp_final, a_final, trip_cov_final = build_master(
    R_truck=R_truck,
    T=T,
    charging_cost_data=hourly_prices,
    bus_cost=bus_cost,
    binary=True
)
rmp_final.Params.LogFile = str(RUN_DIR / "final_mip.log")



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

try:
    rmp_final.write(str(RUN_DIR / f"solution_{RUN_ID}.sol"))
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

#%%

import pickle
import datetime



# Save list of routes to a pickle file
backup_filename = f"R_truck_backup_{datetime.datetime.now().strftime('%H%M%S')}.pkl"
with open(backup_filename, "wb") as f:
    pickle.dump(R_truck, f)

print(f"SAFEGUARD: Saved {len(R_truck)} routes to {backup_filename}")




def solve_pricing_deep_dive(alpha, beta, gamma, mode, time_limit=300):
    """
    Configuration to find the mathematically optimal column, 
    no matter how hard the tree search is.
    """
    m, vars_dict = build_pricing(alpha, beta, gamma, mode)

    # 1. TIME: Give it a long leash. 
    # For 50 trips, 300s is plenty. For 1000 trips, this would hang forever.
    m.Params.TimeLimit = time_limit 

    # 2. FOCUS: Optimality
    # 1=Feasibility (Fast), 2=Optimality (Deep), 3=Bound (Proof)
    m.Params.MIPFocus = 2  

    # 3. METHOD: Auto
    # Let Gurobi decide between Primal/Dual/Barrier. 
    # Barrier is fast for roots, but Dual Simplex is better for deep tree searches.
    m.Params.Method = -1 
    
    # 4. GAP: Tight
    # We don't want a "good enough" route. We want THE route.
    m.Params.MIPGap = 0.01  # 0.1% gap (standard default is 0.0001)

    # 5. HEURISTICS: Balanced
    # Don't spend 100% on heuristics. We need the Branch-and-Bound tree to prove 
    # that no better column exists.
    m.Params.Heuristics = 0.2 

    # 6. POOL: Optimization
    # Mode 0 = Just find the optimal solution. 
    # We don't need 20 mediocre columns; we need the 1 perfect one.
    m.Params.PoolSearchMode = 0 
    
    # 7. CROSSOVER: Auto (Default)
    # We need accurate bases now.
    m.Params.Crossover = -1

    m.optimize()
    return m, vars_dict

print("\n--- INITIATING DEEP DIVE FOR 1-BUS SOLUTION ---")

# 1. Update Duals one last time
rmp.optimize()
alpha, beta_d, gamma_d = extract_duals(rmp)
print(f"Current Master LP Obj: {rmp.ObjVal:.2f}")

# 2. Run Deep Dive Pricing
# We set a high time limit because we expect this to solve the puzzle.
pricing_model, vars_pr = solve_pricing_deep_dive(alpha, beta_d, gamma_d, mode=1, time_limit=600)

#%%
# 3. Process Result
if pricing_model.SolCount > 0:
    # [FIX] Corrected arguments: Remove pricing_model, Add S (stations)
    # Ensure S is defined in your scope (it usually is from config/globals)
    route = extract_route_from_solution(vars_pr, T, S, bar_t, DEPOT)
    
    print(f"\n[DEEP DIVE SUCCESS] Found column with RC: {pricing_model.ObjVal:.2f}")
    
    # Add to Master
    R_truck.append(route)
    
    # Calculate cost and add variable
    cost = calculate_truck_route_cost(route, bus_cost, hourly_prices)
    col = Column()
    for node in route["route"]:
        if isinstance(node, int):
            col.addTerms(1.0, trip_cov[node])
    
    idx = len(R_truck) - 1
    a[idx] = rmp.addVar(obj=cost, lb=0, ub=1, vtype=GRB.CONTINUOUS, column=col, name=f"a[{idx}]")
    
    # 4. Solve Master ONE LAST TIME as MIP
    rmp.update()
    
    print("\n--- SOLVING MASTER AS MIP (Integrality Check) ---")
    # Switch all variables to binary to test for integer solution
    for var in rmp.getVars():
        var.VType = GRB.BINARY 
    
    rmp.optimize()
    
    if rmp.SolCount > 0:
        print(f"Final Integer Objective: {rmp.ObjVal:.2f}")
        if rmp.ObjVal < 20000:
             print("VICTORY: 1 BUS SOLUTION FOUND!")
        else:
             print(f"Result: Best Integer Solution is {rmp.ObjVal:.2f} (likely >1 bus).")
    else:
        print("Master MIP Infeasible (The current columns cannot form a valid integer set).")

else:
    print("[DEEP DIVE FAILED] No negative reduced cost columns found. The current LP relaxation is likely the true bound.")
#%%


# --- 1. ADJUST PARAMETERS FOR THE "EXTENSION" RUN ---
# We want to push past the stagnation. 
# Allow 30 more iterations of "low gain" before quitting, and look for smaller gains.

MAX_CG_ITERS = iteration + 50     # Add 200 more iterations to the current count
STAGNATION_IMPROVEMENT_LIMIT = 10  # Be much more patient (was 8 or 10)
MIN_MEANINGFUL_IMPROVEMENT = 1  # Lower threshold (was 40 or 100) based on K-Means
PRICING_TIMELIMIT = 180             # Reset pricing to be fast/aggressive

# Reset the internal counter so it doesn't stop on the very next step
stagnant_improvement_counter = 0   

# --- 2. RE-SYNC MASTER PROBLEM ---
# We rebuild the RMP to ensure it perfectly matches the 'R_truck' list in memory.
print(f"Re-building Master Problem with {len(R_truck)} existing columns...")
rmp, a, trip_cov = init_master(
    R_truck=R_truck,
    T=T,
    charging_cost_data=hourly_prices,
    bus_cost=bus_cost,
    binary=False
)

# --- 3. RESUME LOOP ---
print(f"Resuming Optimization from Iteration {iteration} to {MAX_CG_ITERS}...")
#%%

MAX_RUNTIME_SECONDS = 3600 * 4 # Let it run for 1 hour maximum
loop_start_time = time.time()

while iteration < MAX_CG_ITERS:
    iteration += 1
    print(f"\n--- Iteration {iteration} (RESUMED) ---")



    # A. SOLVE MASTER
    t0 = time.time()
    if t0 - loop_start_time > MAX_RUNTIME_SECONDS:
        print(f"[STOP] Max runtime of {MAX_RUNTIME_SECONDS}s reached. Stopping safely.")
        break
    
    rmp.Params.TimeLimit = MASTER_TIMELIMIT
    rmp.optimize()
    
    current_obj = rmp.ObjVal
    
    # Calculate improvement
    # Note: last_master_obj might be from the old run, which is fine
    if last_master_obj is None:
        improvement = float("inf")
    else:
        improvement = last_master_obj - current_obj

    print(f" Master obj: {current_obj:.2f} (Impv: {improvement:.4f})")

    # B. STAGNATION CHECK (Updated Settings)
    if improvement < MIN_MEANINGFUL_IMPROVEMENT:
        stagnant_improvement_counter += 1
        print(f"   [INFO] Low improvement (< {MIN_MEANINGFUL_IMPROVEMENT}). "
              f"Stagnation count: {stagnant_improvement_counter}/{STAGNATION_IMPROVEMENT_LIMIT}")
        
        # Safety Valve: Don't stop if we are finding huge RC columns (-1000 or less)
        # We check best_rc_iter from previous loop (initialized to -inf for safety first time)
        if stagnant_improvement_counter >= STAGNATION_IMPROVEMENT_LIMIT:
             print(f"[STOP] Stagnated for {STAGNATION_IMPROVEMENT_LIMIT} iterations. Stopping.")
             break
    else:
        if stagnant_improvement_counter > 0:
            print(f"   [RESET] Improvement {improvement:.2f}. Resetting counter.")
        stagnant_improvement_counter = 0
        current_pricing_timelimit = PRICING_TIMELIMIT # Keep it fast
    
    last_master_obj = current_obj

    # C. EXTRACT DUALS
    alpha, beta_dual, gamma_dual = extract_duals(rmp)

    # D. SOLVE PRICING
    new_trucks = []
    best_rc_iter = float("inf")
    timed_out_any = False
    
    # Track existing routes to avoid duplicates
    seen_keys_existing = {_route_key(r) for r in R_truck}
    
    t0_pricing_total = time.time()

    # --- PRICING RETRY LOOP ---
    while True:
        # Cap max time
        current_pricing_timelimit = min(current_pricing_timelimit, 120) 

        print(f"   > FAST pricing (TimeLimit={current_pricing_timelimit}s)")
        
        # Call your existing fast pricing function
        pricing_model, vars_pr = solve_pricing_fast(
            alpha, beta_dual, gamma_dual,
            mode=1,
            num_fast_cols=n_fast_cols,
            time_limit=current_pricing_timelimit
        )
        
        timed_out_fast = (pricing_model.Status == GRB.TIME_LIMIT)
        timed_out_any |= timed_out_fast

        # Collect
        candidates, best_rc_fast, neg_in_pool, _ = _collect_candidates_from_pool(
            pricing_model, vars_pr, T=T, bar_t=bar_t, DEPOT=DEPOT, RC_EPSILON=RC_EPSILON
        )
        best_rc_iter = min(best_rc_iter, best_rc_fast)

        # Filter duplicates
        candidates.sort(key=lambda r: r["_rc"])
        seen_new = set()
        for t_route in candidates:
            k = _route_key(t_route)
            if (k not in seen_keys_existing) and (k not in seen_new):
                new_trucks.append(t_route)
                seen_new.add(k)
            if len(new_trucks) >= K_BEST:
                break
        
        # Success?
        if new_trucks:
            print(f"   [SUCCESS] Found {len(new_trucks)} cols (best_rc={best_rc_iter:.1f})")
            # Decay time slightly to stay aggressive
            current_pricing_timelimit = max(20, int(current_pricing_timelimit * 0.9))
            break
            
        # Fallback to EXACT if FAST failed but showed promise?
        # Note: You can re-enable exact pricing here if you wish, 
        # but for now, let's stick to the retry logic to keep it simple.
        
        # Stop if optimal
        if (not timed_out_any) and (best_rc_iter >= -RC_EPSILON):
            print(f"   [RC-OPT] Pricing optimal. Stopping.")
            break
            
        # Retry with more time
        if current_pricing_timelimit >= 180:
             print("   [GIVE UP] Max time reached.")
             break
             
        new_tlim = int(current_pricing_timelimit * 2.0)
        print(f"   [RETRY] No cols. Increasing time: {current_pricing_timelimit} -> {new_tlim}")
        current_pricing_timelimit = new_tlim
    # --------------------------

    # E. RECORD STATS (Append to existing list)
    pricing_dur_total = time.time() - t0_pricing_total
    current_stat = {
        "Iteration": iteration,
        "Master_Obj": current_obj,
        "Master_Improvement": improvement,
        "Pricing_Time_s": pricing_dur_total,
        "Cols_Added": len(new_trucks),
        "Best_RC": best_rc_iter,
        "Stagnant_Counter": stagnant_improvement_counter,
        "Total_Runtime_s": time.time() - stopwatch_start
    }
    cg_stats.append(current_stat)
    
    # Update CSV in place
    pd.DataFrame(cg_stats).to_csv(stats_csv_path, index=False)

    # F. ADD COLUMNS
    if not new_trucks and best_rc_iter >= -RC_EPSILON:
        print("Done.")
        break
        
    for route in new_trucks:
        print(f"[ADD] rc={route.get('_rc', 0):.1f} | Path: {route['desc']}")
        R_truck.append(route)
        cost = calculate_truck_route_cost(route, bus_cost, hourly_prices)

        col = Column()
        for node in route["route"]:
            if isinstance(node, int):
                col.addTerms(1.0, trip_cov[node])

        idx = len(R_truck) - 1
        a[idx] = rmp.addVar(obj=cost, lb=0, ub=1, vtype=GRB.CONTINUOUS, column=col, name=f"a[{idx}]")
    rmp.update()
# %%

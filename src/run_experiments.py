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
    DEPOT_NAME, G_KWH, ENERGY_PER_BLOCK_KWH, BLOCK_KWH, 
    charge_mult, charge_cost_premium,
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
#%%

begin = time.time()


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

def energy_to_events(kwh: float) -> int:
    # “event” = BLOCK_KWH kWh regardless of granularity
    return int(math.ceil(float(kwh) / float(BLOCK_KWH)))

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

routes_csv    = DATA_DIR / "Par_Routes_Overhauled.csv"
# routes_csv = DATA_DIR / "Par_Routes_for_code.csv"
deadheads_csv = DATA_DIR / "Par_DHD_for_code.csv"
prices_csv    = DATA_DIR / "hourly_prices.csv"

# details_csv    = DATA_DIR / "Par_VehicleDetails.csv"
# vd_csv = DATA_DIR / "Par_VehicleDetails.csv"  # or VehicleDetails.csv

vd_csv = DATA_DIR / "Par_VehicleDetails_Updated.csv"

# routes_csv    = DATA_DIR / "Toy_Routes.csv"
# deadheads_csv = DATA_DIR / "Toy_DHD.csv"
# prices_csv    = DATA_DIR / "Toy_Prices.csv"


if not routes_csv.exists():
    raise FileNotFoundError(f"Missing {routes_csv}")
if not deadheads_csv.exists():
    raise FileNotFoundError(f"Missing {deadheads_csv}")
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


trip_col_map = {"SL": None, "ST": None, "ET": None, "EL": None, "Energy used": None}
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

missing = [k for k, v in trip_col_map.items() if v is None]
if missing:
    raise ValueError(f"{routes_csv.name} must have columns {{'SL','ST','ET','EL','Energy used'}}, "
                     f"could not find: {missing}. Found: {set(df_trips.columns)}")

df_trips = df_trips.rename(columns={trip_col_map[k]: k for k in trip_col_map})

df_dh = pd.read_csv(deadheads_csv)

dh_col_map = {"From": None, "To": None, "Duration": None, "Energy used": None}
for want in list(dh_col_map.keys()):
    if want in df_dh.columns:
        dh_col_map[want] = want
        continue
    if want == "Energy used" and "Energy_use" in df_dh.columns:
        dh_col_map[want] = "Energy_use"
        continue
    for c in df_dh.columns:
        if c.strip().lower() == want.lower():
            dh_col_map[want] = c
            break
missing_dh = [k for k, v in dh_col_map.items() if v is None]
if missing_dh:
    raise ValueError(f"{deadheads_csv.name} must include columns {{'From','To','Duration','Energy used'}}, "
                     f"could not find: {missing_dh}. Found: {set(df_dh.columns)}")
df_dh = df_dh.rename(columns={dh_col_map[k]: k for k in dh_col_map})


# ==============================================================================
# Graph Node Explosion (Station Copies) -- COPIES ONLY, underscore-safe
# ==============================================================================
def strip_copy_suffix(name: str) -> str:
    s = str(name).strip()
    if "_" in s:
        left, right = s.rsplit("_", 1)
        if right.isdigit():
            return left
    return s

def variants_copies_only(node_name, station_copies):
    base = strip_copy_suffix(node_name)
    if base in station_copies:
        c = station_copies[base]
        return [f"{base}_{k}" for k in range(c)]
    else:
        return [str(node_name).strip()]

def explode_graph_nodes(df, station_copies):
    """
    Replaces any station base node (e.g. DEPOT, CHARGER1) by its copies
    (e.g. DEPOT_0, DEPOT_1, ...) everywhere in the deadhead arcs.
    Does NOT keep base nodes in the graph.
    """
    new_rows = []
    print("[INFO] Generating exploded graph arcs (copies only)...")

    for _, row in df.iterrows():
        sources = variants_copies_only(row["From"], station_copies)
        targets = variants_copies_only(row["To"], station_copies)

        for s in sources:
            for t in targets:
                if s == t:
                    continue

                # Prevent copy-to-copy "teleporting" within the same physical station
                s_base = strip_copy_suffix(s)
                t_base = strip_copy_suffix(t)
                if (s_base == t_base) and (s_base in station_copies) and (s != t):
                    continue

                new_r = row.copy()
                new_r["From"] = s
                new_r["To"]   = t
                new_rows.append(new_r)

    return pd.DataFrame(new_rows)


df_dh = explode_graph_nodes(df_dh, STATION_COPIES)



def expand_station_copies(base_names, station_copies):
    out = []
    for b in base_names:
        c = station_copies.get(b, 1)
        for k in range(c):
            out.append(f"{b}_{k}")
    return out

# copies only
CHARGERS = expand_station_copies(CHARGING_STATIONS, STATION_COPIES)
DEPOT_BASE = DEPOT_NAME

DEPOT_NAME = f"{DEPOT_BASE}_0"

DEPOT = DEPOT_NAME


print(f"[INFO] New Depot Name: {DEPOT_NAME}")
print(f"[INFO] Expanded Chargers (copies only): {CHARGERS}")

assert all(h.endswith("_0") or h.endswith("_1") or h.rsplit("_",1)[1].isdigit() for h in CHARGERS)
assert "CHARGER1" not in CHARGERS  # base should NOT appear

# # ----------------------------------------------------------------------
# # SINGLE canonical CHARGERS definition (use config + depot)
# # ----------------------------------------------------------------------
# CHARGERS = sorted(set(CHARGING_STATIONS + [DEPOT_NAME]))

# ----------------------------------------------------------------------
# OPTIONAL: auto-augment deadheads with one-hop compositions
# ----------------------------------------------------------------------


def augment_deadheads(df, chargers):
    df = df.copy()
    df["From"] = df["From"].astype(str).str.strip()
    df["To"]   = df["To"].astype(str).str.strip()
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
    df["Energy used"] = pd.to_numeric(df["Energy used"], errors="coerce")

    # join (X→mid) + (mid→charger) → (X→charger)
    mid_to_chg = df[df["To"].isin(chargers)][["From","To","Duration","Energy used"]]
    mid_to_chg = mid_to_chg.rename(columns={"From":"mid","To":"charger"})
    x_to_mid   = df.rename(columns={"From":"X","To":"mid"})

    fwd = x_to_mid.merge(mid_to_chg, on="mid", how="inner")
    fwd["From"] = fwd["X"]
    fwd["To"]   = fwd["charger"]
    fwd["Duration"]    = fwd["Duration_x"] + fwd["Duration_y"]
    fwd["Energy used"] = fwd["Energy used_x"] + fwd["Energy used_y"]
    fwd = fwd[["From","To","Duration","Energy used"]]

    # backward: (charger→mid) + (mid→X) → (charger→X)
    chg_to_mid = df[df["From"].isin(chargers)][["From","To","Duration","Energy used"]]
    chg_to_mid = chg_to_mid.rename(columns={"From":"charger","To":"mid"})
    mid_to_x   = df.rename(columns={"From":"mid","To":"X"})

    bwd = chg_to_mid.merge(mid_to_x, on="mid", how="inner")
    bwd["From"] = bwd["charger"]
    bwd["To"]   = bwd["X"]
    bwd["Duration"]    = bwd["Duration_x"] + bwd["Duration_y"]
    bwd["Energy used"] = bwd["Energy used_x"] + bwd["Energy used_y"]
    bwd = bwd[["From","To","Duration","Energy used"]]

    composed = pd.concat([fwd, bwd], ignore_index=True)
    composed = composed[composed["From"] != composed["To"]]

    # remove duplicates and already-existing arcs
    key = set(zip(df["From"], df["To"]))
    composed = composed[~composed.apply(lambda r: (r["From"], r["To"]) in key, axis=1)]

    # --- duration cap for indirect arcs (minutes) ---
    MAX_INDIRECT_MIN = 40
    composed = composed[composed["Duration"] <= MAX_INDIRECT_MIN]

    # combine and keep only the shortest version of each (From, To)
    df = pd.concat([df, composed], ignore_index=True)
    df = df.sort_values(["From","To","Duration","Energy used"])
    df = df.drop_duplicates(subset=["From","To"], keep="first")

    print(f"[augment] added {len(composed)} indirect arcs (after duration filter ≤ {MAX_INDIRECT_MIN} min)")
    return df

df_dh = augment_deadheads(df_dh, CHARGERS)

# ---- Add direct depot arcs via 1-hop composition: PARX→mid→Y and X→mid→PARX ----
def augment_depot_bridging(df, depot):
    df = df.copy()
    for col in ["From", "To"]:
        df[col] = df[col].astype(str).str.strip()
    df["Duration"]    = pd.to_numeric(df["Duration"], errors="coerce")
    df["Energy used"] = pd.to_numeric(df["Energy used"], errors="coerce")
    df = df.dropna(subset=["Duration","Energy used"])

    # (depot → mid) + (mid → Y) → (depot → Y)
    dep_to_mid = df[df["From"] == depot][["From","To","Duration","Energy used"]].rename(
        columns={"To":"mid","Duration":"d0","Energy used":"e0"}
    )
    mid_to_Y   = df.rename(columns={"From":"mid","To":"Y","Duration":"d1","Energy used":"e1"})[["mid","Y","d1","e1"]]
    fwd = dep_to_mid.merge(mid_to_Y, on="mid", how="inner")
    fwd = fwd.assign(
        From=depot,
        To=fwd["Y"],
        Duration=fwd["d0"] + fwd["d1"],
        **{"Energy used": fwd["e0"] + fwd["e1"]}
    )[["From","To","Duration","Energy used"]]

    # (X → mid) + (mid → depot) → (X → depot)
    X_to_mid = df.rename(columns={"To":"mid","Duration":"d0","Energy used":"e0"})[["From","mid","d0","e0"]]
    mid_to_dep = df[df["To"] == depot][["From","To","Duration","Energy used"]].rename(
        columns={"From":"mid","Duration":"d1","Energy used":"e1"}
    )
    bwd = X_to_mid.merge(mid_to_dep, on="mid", how="inner")
    bwd = bwd.assign(
        From=bwd["From"],
        To=depot,
        Duration=bwd["d0"] + bwd["d1"],
        **{"Energy used": bwd["e0"] + bwd["e1"]}
    )[["From","To","Duration","Energy used"]]

    composed = pd.concat([fwd, bwd], ignore_index=True)
    composed = composed[
        (composed["From"] != composed["To"]) &
        (composed["Duration"] >= 0) &
        (composed["Energy used"] >= 0)
    ]

    # drop ones that already exist
    existing = set(zip(df["From"], df["To"]))
    composed = composed[~composed.apply(lambda r: (r["From"], r["To"]) in existing, axis=1)]

    # --- duration cap for these indirect arcs (minutes) ---
    MAX_INDIRECT_MIN = 40
    composed = composed[composed["Duration"] <= MAX_INDIRECT_MIN]

    # combine and keep only the shortest version of each (From, To)
    out = pd.concat([df, composed], ignore_index=True)
    out = out.sort_values(["From","To","Duration","Energy used"])
    out = out.drop_duplicates(subset=["From","To"], keep="first")

    print(f"[augment/depot] added {len(composed)} depot-bridging arcs (≤ {MAX_INDIRECT_MIN} min)")
    return out

# DEPOT_CHARGERS = [f"PARX_{k}" for k in range(STATION_COPIES["PARX"])]

# rows = []
# for h in DEPOT_CHARGERS:
#     rows += [
#         {"From": DEPOT_NAME, "To": h, "Duration": 0, "Energy used": 0},
#         {"From": h, "To": DEPOT_NAME, "Duration": 0, "Energy used": 0},
#     ]
# df_dh = pd.concat([df_dh, pd.DataFrame(rows)], ignore_index=True)

df_dh = augment_depot_bridging(df_dh, DEPOT_NAME)

# ---- Ensure depot↔trip endpoint arcs exist (allow ANY mid, not just chargers) ----
def ensure_depot_endpoint_arcs(df, depot, sl_map, el_map, max_dur_min=None):
    """
    Ensure depot↔trip endpoint arcs exist by composing via ANY mid node.
    If max_dur_min is not None, discard composed arcs whose total Duration exceeds this cap (minutes).
    """
    df = df.copy()
    df["From"] = df["From"].astype(str).str.strip()
    df["To"]   = df["To"].astype(str).str.strip()
    df["Duration"]    = pd.to_numeric(df["Duration"], errors="coerce")
    df["Energy used"] = pd.to_numeric(df["Energy used"], errors="coerce")
    df = df.dropna(subset=["Duration","Energy used"])

    sl_set = set(sl_map.values())
    el_set = set(el_map.values())
    existing = set(zip(df["From"], df["To"]))

    # Helper tables (ANY mid)
    dep_to_mid = df[df["From"] == depot][["From","To","Duration","Energy used"]].rename(
        columns={"To":"mid","Duration":"d0","Energy used":"e0"}
    )[["mid","d0","e0"]]
    mid_to_any = df.rename(columns={"From":"mid","To":"Y","Duration":"d1","Energy used":"e1"})[
        ["mid","Y","d1","e1"]
    ]
    any_to_mid = df.rename(columns={"From":"X","To":"mid","Duration":"d0","Energy used":"e0"})[
        ["X","mid","d0","e0"]
    ]
    mid_to_dep = df[df["To"] == depot][["From","To","Duration","Energy used"]].rename(
        columns={"From":"mid","Duration":"d1","Energy used":"e1"}
    )[["mid","d1","e1"]]

    new_rows = []

    # ---- Missing pull-out: depot -> SL(i)
    missing_pullout_targets = {s for s in sl_set if (depot, s) not in existing}
    if missing_pullout_targets and not dep_to_mid.empty and not mid_to_any.empty:
        fwd = dep_to_mid.merge(mid_to_any, on="mid", how="inner")
        fwd = fwd[fwd["Y"].isin(missing_pullout_targets)]
        # Compose totals
        fwd["Duration_tot"] = fwd["d0"] + fwd["d1"]
        fwd["Energy_tot"]   = fwd["e0"] + fwd["e1"]
        # Apply duration cap if requested
        if max_dur_min is not None:
            fwd = fwd[fwd["Duration_tot"] <= float(max_dur_min)]
        for _, r in fwd.iterrows():
            From, To = depot, r["Y"]
            if (From, To) not in existing:
                new_rows.append({
                    "From": From,
                    "To": To,
                    "Duration": float(r["Duration_tot"]),
                    "Energy used": float(r["Energy_tot"]),
                })

    # ---- Missing pull-in: EL(i) -> depot
    missing_pulluin_sources = {e for e in el_set if (e, depot) not in existing}
    if missing_pulluin_sources and not any_to_mid.empty and not mid_to_dep.empty:
        bwd = any_to_mid.merge(mid_to_dep, on="mid", how="inner")
        bwd = bwd[bwd["X"].isin(missing_pulluin_sources)]
        bwd["Duration_tot"] = bwd["d0"] + bwd["d1"]
        bwd["Energy_tot"]   = bwd["e0"] + bwd["e1"]
        if max_dur_min is not None:
            bwd = bwd[bwd["Duration_tot"] <= float(max_dur_min)]
        for _, r in bwd.iterrows():
            From, To = r["X"], depot
            if (From, To) not in existing:
                new_rows.append({
                    "From": From,
                    "To": To,
                    "Duration": float(r["Duration_tot"]),
                    "Energy used": float(r["Energy_tot"]),
                })

    if new_rows:
        # Keep the best (shortest) per (From, To)
        add_df = (
            pd.DataFrame(new_rows)
              .sort_values(["From","To","Duration","Energy used"])
              .drop_duplicates(subset=["From","To"], keep="first")
        )
        print(f"[augment/endpoints-targeted] adding {len(add_df)} depot↔endpoint arcs"
              + (f" (≤ {max_dur_min} min)" if max_dur_min is not None else ""))
        df = pd.concat([df, add_df], ignore_index=True)
    else:
        note = f" with duration cap ≤ {max_dur_min} min" if max_dur_min is not None else ""
        print(f"[augment/endpoints-targeted] no missing endpoint arcs found to add{note}")

    return df


# ------------------------------ Build trip set ------------------------------

df_trips = df_trips.reset_index(drop=True).copy()
df_trips["Trip"] = df_trips.index

df_trips["st_blk"] = df_trips["ST"].astype(str).map(_floor_block)
df_trips["et_blk"] = df_trips["ET"].astype(str).map(_ceil_block)

df_trips["eps_events"] = df_trips["Energy used"].map(energy_to_events)



# ==============================================================================
# NEW: Diagnostic - Check Time Block Discretization
# ==============================================================================
print(f"\n[DIAGNOSTIC] Time Discretization Check (TBPH={TIMEBLOCKS_PER_HOUR})")
print(f"Horizon: {bar_t} blocks")
# Print a sample of trips to verify rounding
print(df_trips[["Trip", "ST", "st_blk", "ET", "et_blk"]].head(20).to_string())

# Check for late trips that might be clamped
late_trips = df_trips[df_trips['et_blk'] >= bar_t]
if not late_trips.empty:
    print(f"\n[WARN] {len(late_trips)} trips end at or beyond the time horizon limit ({bar_t}):")
    print(late_trips[["Trip", "ST", "st_blk", "ET", "et_blk"]].head(5).to_string())
#==============================================================================



#%%



T = list(df_trips["Trip"].tolist())

sl = df_trips.set_index("Trip")["SL"].to_dict()
el = df_trips.set_index("Trip")["EL"].to_dict()
st = df_trips.set_index("Trip")["st_blk"].to_dict()
et = df_trips.set_index("Trip")["et_blk"].to_dict()
epsilon = df_trips.set_index("Trip")["eps_events"].to_dict()

# ensure endpoints after we know sl/el
df_dh = ensure_depot_endpoint_arcs(df_dh, DEPOT_NAME, sl, el, max_dur_min=50)

# ------------------------------ Allowed deadhead graph ------------------------------

def _energy_to_events(kwh: float) -> int:
    return energy_to_events(kwh)

allowed_arcs = {}
for _, row in df_dh.iterrows():
    f = str(row["From"]).strip()
    t = str(row["To"]).strip()
    dur_min = float(row["Duration"])
    eng_kwh = float(row["Energy used"])
    tau_blk = ceil_blocks_from_minutes(dur_min)
    d_evt   = _energy_to_events(eng_kwh)
    allowed_arcs[(f, t)] = (tau_blk, d_evt)


# arc_from_to before we allowed ref to reference.
def arc_from_to(from_node: str, to_node: str):
    if str(from_node).strip() == str(to_node).strip():

        return (0, 0)

    return allowed_arcs.get((str(from_node).strip(), str(to_node).strip()), None)

# # ------------------------------
# # Arc lookup with stats
# # ------------------------------
# fallback_hits = 0
# direct_hits = 0
# mixed_hits = 0
# misses = 0


# def arc_from_to(from_node, to_node):
#     global fallback_hits, direct_hits, mixed_hits, misses

#     f = str(from_node).strip()
#     t = str(to_node).strip()
#     if f == t:
#         return (0, 0)

#     # 1) direct (location->location)
#     pair = allowed_arcs.get((f, t), None)
#     if pair is not None:
#         direct_hits += 1
#         return pair

#     if not loc_to_ref:
#         misses += 1
#         return None

#     # 2) fallback using refs (use base names for station copies)
#     base_f = strip_copy_suffix(f)
#     base_t = strip_copy_suffix(t)

#     f_ref = loc_to_ref.get(base_f, base_f)
#     t_ref = loc_to_ref.get(base_t, base_t)

#     # try ref->ref
#     pair = allowed_arcs.get((f_ref, t_ref), None)
#     if pair is not None:
#         fallback_hits += 1
#         return pair

#     # try mixed (helps when only one side is missing in DHD)
#     pair = allowed_arcs.get((f_ref, t), None)
#     if pair is not None:
#         mixed_hits += 1
#         return pair

#     pair = allowed_arcs.get((f, t_ref), None)
#     if pair is not None:
#         mixed_hits += 1
#         return pair

#     misses += 1
#     return None


#%%
# ------------------------------ Globals for pricing ------------------------------

S = CHARGERS[:]  # stations set 
# S = [h for h in S if h != DEPOT] # make sure not to include depot
G = energy_to_events(G_KWH)  


tau = {}
d   = {}

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

# ---- Zero-hop arcs when the trip endpoint is itself a charger ----
# If SL(i) is a charger h, allow h -> i with 0 time and 0 energy.
# If EL(i) is a charger h, allow i -> h with 0 time and 0 energy.
added_zero = 0
for i in T:
    # pre-trip charge at SL if it’s a station
    if sl[i] in S and (sl[i], i) not in tau:
        tau[(sl[i], i)] = 0
        d[(sl[i], i)]   = 0
        added_zero += 1
    # post-trip charge at EL if it’s a station
    if el[i] in S and (i, el[i]) not in tau:
        tau[(i, el[i])] = 0
        d[(i, el[i])]   = 0
        added_zero += 1

print(f"[augment/zero-hop] added {added_zero} station↔trip zero-hop arcs")


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
charging_cost_data = charging_cost_data * float(ENERGY_PER_BLOCK_KWH)
avg_cost_per_event = float(avg_cost_per_kwh) * float(ENERGY_PER_BLOCK_KWH)

bus_cost  = BUS_COST_KX* avg_cost_per_event

print(f"[INFO] Capacity G(events): {G} (each={ENERGY_PER_BLOCK_KWH} kWh)")
print(f"[INFO] Avg price/kWh={avg_cost_per_kwh:.3f} → per-event={avg_cost_per_event:.3f}")
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
    pricing_model.Params.Heuristics = 0.25
    pricing_model.Params.Cuts = 0

    # --------- helper: strong pre-pruning for feasibility + short deadheads ---------
    MAX_TAU = TIMEBLOCKS_PER_HOUR // 2 # at most 30MIN deadheads between nodes in pricing graph

    def tt_ok(i, j):
        return (
            (i, j) in tau
            and (et[i] + tau[(i, j)] <= st[j])
            and (tau[(i, j)] <= MAX_TAU)
        )

    def ih_ok(i, h):
        return (
            (i, h) in tau
            and (et[i] + tau[(i, h)] <= bar_t)
            and (tau[(i, h)] <= MAX_TAU)
        )

    def hi_ok(h, i):
        return (
            (h, i) in tau
            and (tau[(h, i)] <= st[i])
            and (tau[(h, i)] <= MAX_TAU)
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


    ## charging updates
    for h in S_use:
        pricing_model.addConstr(
            v_amt[h] == charge_mult * (1.0 / TIMEBLOCKS_PER_HOUR) *
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

    # # Define g_return based on which node is the last node (end marker)
    # for i in T:
    #     if (i, DEPOT) in d:
    #         pricing_model.addGenConstrIndicator(
    #             wOmega_trip[i], 1,
    #             g_return - (g[i] - epsilon[i] - d[(i, DEPOT)]),
    #             GRB.EQUAL, 0,
    #             name=f"ind_greturn_endtrip_{i}"
    #         )

    # for h in S_use:
    #     if (h, DEPOT) in d:
    #         pricing_model.addGenConstrIndicator(
    #             wOmega_station[h], 1,
    #             g_return - (g[h] + v_amt[h] - d[(h, DEPOT)]),
    #             GRB.EQUAL, 0,
    #             name=f"ind_greturn_endstat_{h}"
    #         )


    obj = bus_cost
    for h in S_use:
        if h in charging_cost_data.columns:
            for t in time_blocks:
                if t in charging_cost_data.index:
                    price_evt = float(charging_cost_data.at[t, h])
                    obj += price_evt * chi_plus[h, t] * charge_cost_premium

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

def solve_pricing_fast(alpha, beta, gamma, mode, num_fast_cols=10):
    cap = min(int(num_fast_cols), 400)
    m, vars_dict = build_pricing(alpha, beta, gamma, mode)
    m.Params.PoolSearchMode = 1
    m.Params.PoolSolutions  = cap
    m.Params.SolutionLimit  = cap
    m.optimize()
    return m, vars_dict

def solve_pricing_exact(alpha, beta, gamma, mode, num_exact_cols=10):
    m, vars_dict = build_pricing(alpha, beta, gamma, mode)
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions  = int(num_exact_cols)
    m.optimize()
    return m, vars_dict

# ------------------------------ DIAGNOSTICS: list missing depot arcs ------------------------------
diag_dir = OUTDIR / f"diag_{RUN_ID}"
diag_dir.mkdir(parents=True, exist_ok=True)
missing_pullout = [i for i in T if (DEPOT, sl[i]) not in allowed_arcs]
missing_pulluin = [i for i in T if (el[i], DEPOT) not in allowed_arcs]
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
stagnant = 0

while new_pricing_obj < -tolerance and iteration < max_iter:
# for _ in range(1):
    iteration += 1
    print(f"\n--- Iteration {iteration} ---")
    t0 = time.time()
    rmp.Params.TimeLimit = MASTER_TIMELIMIT
    rmp.optimize()
    master_times.append(time.time() - t0)
    print(" Master obj:", rmp.ObjVal)

    # stagnation check
    rel_impr = (best_master - rmp.ObjVal) / max(1.0, abs(best_master)) if math.isfinite(best_master) else 1.0
    if rmp.SolCount > 0 and rmp.ObjVal < best_master - 1e-9 and rel_impr >= MASTER_IMPROVE_THRESHOLD:
        best_master = rmp.ObjVal
        stagnant = 0
    else:
        stagnant += 1
        print(f" [CG] no meaningful improve (stagnant={stagnant})")

    if stagnant >= STAGNATION_ITERS:
        print(f"[STOP] Stagnation for {STAGNATION_ITERS} iters (rel_impr<{MASTER_IMPROVE_THRESHOLD:.4%}).")
        break

    alpha, beta_dual, gamma_dual = extract_duals(rmp)

    t0 = time.time()
    pricing_model, vars_pr = solve_pricing_fast(
        alpha, beta_dual, gamma_dual,
        mode=1,
        num_fast_cols=n_fast_cols
    )
    pricing_times.append(time.time() - t0)

    new_trucks = []
    new_pricing_obj = float('inf')
    seen_keys_existing = {_route_key(r) for r in R_truck}

    # Scan pool, collect K_BEST with rc < -RC_EPSILON
    candidates = []
    for sol in range(pricing_model.SolCount):
        pricing_model.Params.SolutionNumber = sol
        rc = pricing_model.PoolObjVal
        new_pricing_obj = min(new_pricing_obj, rc)
        if rc < -RC_EPSILON:
            try:
                truck = extract_route_from_solution(
                    vars_pr, T, list(vars_pr["wA_station"].keys()), bar_t,
                    depot=DEPOT,
                    value_getter=lambda v: v.Xn
                )
                truck["_rc"] = rc
                candidates.append(truck)
            except Exception as e:
                print(f"[SKIP] bad pricing solution in pool: {e}")

    # sort by rc and keep K_BEST unique routes
    candidates.sort(key=lambda r: r["_rc"])
    seen_new = set()
    for t_route in candidates:
        k = _route_key(t_route)
        if (k not in seen_keys_existing) and (k not in seen_new):
            new_trucks.append(t_route)
            seen_new.add(k)
        if len(new_trucks) >= K_BEST:
            break

    # exact if needed (still negative but no candidates under epsilon)
    if not new_trucks and new_pricing_obj < -RC_EPSILON and n_exact_cols > 0:
        print(" No fast cols → exact pricing")
        pricing_model, vars_pr = solve_pricing_exact(
            alpha, beta_dual, gamma_dual,
            mode=1,
            num_exact_cols=n_exact_cols
        )
        candidates = []
        for sol in range(pricing_model.SolCount):
            pricing_model.Params.SolutionNumber = sol
            rc = pricing_model.PoolObjVal
            new_pricing_obj = min(new_pricing_obj, rc)
            if rc < -RC_EPSILON:
                try:
                    truck = extract_route_from_solution(
                        vars_pr, T, list(vars_pr["wA_station"].keys()), bar_t,
                        depot=DEPOT,
                        value_getter=lambda v: v.Xn
                    )
                    truck["_rc"] = rc
                    candidates.append(truck)
                except Exception as e:
                    print(f"[SKIP] bad pricing solution in pool: {e}")
        candidates.sort(key=lambda r: r["_rc"])
        seen_new = set()
        for t_route in candidates:
            k = _route_key(t_route)
            if (k not in seen_keys_existing) and (k not in seen_new):
                new_trucks.append(t_route)
                seen_new.add(k)
            if len(new_trucks) >= K_BEST:
                break

    # Add columns (append to R_truck so final rebuilds see them)
    for route in new_trucks:
        print(f"[ADD] column rc={route.get('_rc', float('nan')):.1f}  route={route['route']}")
        R_truck.append(route)  # keep master copy in sync
        cost = calculate_truck_route_cost(route, bus_cost, charging_cost_data)
        col = Column()
        for node in route["route"]:
            if isinstance(node, int):
                col.addTerms(1.0, trip_cov[node])
        idx = len(R_truck) - 1  # index corresponds to appended route
        a[idx] = rmp.addVar(
            obj=cost, lb=0, ub=1, vtype=GRB.CONTINUOUS, column=col, name=f"a[{idx}]"
        )
    rmp.update()

    # checkpoint each iter
    try:
        rmp.write(str(CKPT / f"iter_{iteration:03d}.lp"))
    except Exception:
        pass

    iter_rows.append({
        "run_id": RUN_ID,
        "iteration": iteration,
        "master_time_s": master_times[-1] if master_times else None,
        "pricing_time_s": pricing_times[-1] if pricing_times else None,
        "pricing_mode": "fast" if new_trucks else ("exact" if new_pricing_obj < -RC_EPSILON else "none"),
        "best_rc": float(new_pricing_obj)
    })
    print(f"[ITER] {iteration}: master {master_times[-1]:.2f}s, pricing {pricing_times[-1]:.2f}s, "
          f"new_cols={len(new_trucks)}, best_rc={new_pricing_obj:.1f}")

    if new_pricing_obj >= -RC_EPSILON:
        print(f"[STOP] Reduced-cost optimal (best_rc={new_pricing_obj:.1f} ≥ -RC_EPSILON={-RC_EPSILON}).")
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


# %%

#thisfile has unsaved changes on symm breaking a1 \leq a2
end = time.time()
print(f"\n=== TOTAL TIME: {end - begin:.2f} seconds ===")









# %%
# ==============================================================================
# DIAGNOSTIC: Check GIRO Solution Reduced Cost (With Fuzzy Matching)
# ==============================================================================

# def check_giro_route_fuzzy(alpha_duals, charging_cost_data, bus_cost_val, df_trips):
#     print("\n" + "="*60)
#     print(" CHECKING GIRO SOLUTION REDUCED COST (Fuzzy Match)")
#     print("="*60)

#     # 1. LOAD MAPPING DATA (RouteID -> Trip Attributes -> Internal Index)
#     # ---------------------------------------------------------
    
#     # A. Parse VehicleDetails to get RouteID -> List of (Start, From)
    
#     #vd_csv = DATA_DIR / "VehicleDetails.csv" # Ensure filename is correct
#     vd_csv = DATA_DIR / "Par_VehicleDetails_Updated.csv"
#     if not vd_csv.exists():
#         # Fallback to Par_VehicleDetails if just renamed
#         vd_csv = DATA_DIR / "Par_VehicleDetails.xlsx - Data.csv"
        
#     if not vd_csv.exists():
#         print(f"[ERROR] VehicleDetails file not found. Cannot map Route IDs.")
#         return

#     print("Building Route ID map from VehicleDetails...")
#     df_vd = pd.read_csv(vd_csv, low_memory=False)
    
#     # Helper to normalize time "4:45" -> "04:45"
#     def norm_time(t):
#         try:
#             h, m = map(int, str(t).split(':'))
#             return f"{h}:{m:02d}"
#         except: return str(t)
    
#     # Map: RouteID (e.g. 5518) -> List of specific trip details [(4:45, JON_A), (6:00, ...)]
#     # Because one RouteID might be used for multiple trips in a day
#     giro_trip_details = {} 
    
#     # Filter for Regular trips
#     for _, row in df_vd[df_vd['Identifier'] == 'Regular'].iterrows():
#         try:
#             r_id = int(float(row['Route']))
#             st = norm_time(row['Start'])
#             sl = str(row['From']).strip()
            
#             if r_id not in giro_trip_details:
#                 giro_trip_details[r_id] = []
#             giro_trip_details[r_id].append({'st': st, 'sl': sl})
#         except: pass

#     # B. Map Internal Index -> (ST, SL) for fast lookup
#     # We allow some tolerance on time matching if needed, but exact string match first
#     internal_trips_map = {}
#     for idx, row in df_trips.iterrows():
#         st = norm_time(row['ST'])
#         sl = str(row['SL']).strip()
#         # Key by (ST, SL) -> Index
#         internal_trips_map[(st, sl)] = idx

#     # 2. DEFINE THE ROUTE (GIRO IDs + Stations)
#     # ---------------------------------------------------------
#     # Route 13301
#     giro_route_raw = [
#         'PARX', 5518, 5513, 5513, 5514, '7880C_0', 5514, 5518, 5501, 'PARX', 
#         'PARX_1', 5515, 5515, 5515, 5515, 5518, 5519, 5517, 5517, 5518, 
#         'JON_A_0', 5518, 5513, '3127L_0', 5513, 5510, 5513, 5510, '3127L_1', 
#         5510, 5519, 'JON_A_1', 5519, 5515, 5515, 5519, 5519, 5510, '3127L_2', 
#         5510, 5515, 5515, 5515, 5515, 5515, 'PARX'
#     ]

#     # 3. MATCHING LOGIC
#     # ---------------------------------------------------------
#     internal_route = []
#     covered_indices = []
    
#     print("Matching Route items to Internal Trips...")
    
#     # We need to track "current time" to disambiguate if 5518 appears twice?
#     # Actually, the sequence in the list likely matches the sequence in VehicleDetails
#     # But VehicleDetails isn't sorted by bus? 
#     # Let's try simple exact matching. If 5518 has multiple entries, we need to pick the "next" one.
    
#     # Queue of available trips for each RouteID
#     # We sort them by time to pick them in order
#     import collections
#     trip_queues = {}
#     for r_id, trips in giro_trip_details.items():
#         # Sort by start time
#         sorted_trips = sorted(trips, key=lambda x: int(x['st'].split(':')[0])*60 + int(x['st'].split(':')[1]))
#         trip_queues[r_id] = collections.deque(sorted_trips)

#     for item in giro_route_raw:
#         if isinstance(item, str):
#             # Station
#             if item == "PARX": internal_route.append(f"{DEPOT_NAME}_0")
#             else: internal_route.append(item)
#         else:
#             # Trip ID (int)
#             r_id = item
#             if r_id in trip_queues and trip_queues[r_id]:
#                 # Pop the next occurrence of this route
#                 details = trip_queues[r_id].popleft()
#                 key = (details['st'], details['sl'])
                
#                 if key in internal_trips_map:
#                     idx = internal_trips_map[key]
#                     internal_route.append(idx)
#                     covered_indices.append(idx)
#                     # print(f"  Mapped {r_id} -> Trip {idx} ({details['st']} {details['sl']})")
#                 else:
#                     print(f"  [FAIL] Route {r_id} ({details['st']} {details['sl']}) not found in df_trips!")
#             else:
#                 print(f"  [FAIL] Route {r_id} has no more scheduled trips in VehicleDetails or unknown ID.")

#     # 4. CALCULATE REDUCED COST
#     # ---------------------------------------------------------
#     test_route = {
#         "route": internal_route,
#         "charging_stops": {
#             'chi_plus': [
#                 ('7880C_0', 142), ('7880C_0', 143), ('7880C_0', 144), 
#                 ('PARX_1', 238), ('PARX_1', 239), ('PARX_1', 240),
#                 ('JON_A_0', 614), ('3127L_0', 668), ('3127L_1', 802),
#                 ('JON_A_1', 865), ('3127L_2', 1006)
#             ],
#             'chi_minus': []
#         },
#         "type": "truck"
#     }

#     try:
#         real_cost = calculate_truck_route_cost(test_route, bus_cost_val, charging_cost_data)
        
#         dual_sum = 0.0
#         for i in covered_indices:
#             if i in alpha_duals:
#                 dual_sum += alpha_duals[i]
#             else:
#                  # If i is not in alpha, it might be a trip that was filtered out or dummy?
#                  pass
        
#         rc = real_cost - dual_sum
        
#         print(f"\n--- RESULTS ---")
#         print(f"Route Length: {len(internal_route)} nodes")
#         print(f"Trips Covered: {len(covered_indices)}")
#         print(f"Real Cost: {real_cost:,.2f}")
#         print(f"Dual Sum:  {dual_sum:,.2f}")
#         print(f"Reduced Cost: {rc:,.2f}")
        
#         if rc < -1e-3:
#             print("✅ VALID: Negative Reduced Cost. The solver SHOULD find this.")
#         else:
#             print("❌ INVALID: Positive Reduced Cost.")
            
#     except Exception as e:
#         print(f"[ERROR] Calculation failed: {e}")
# check_giro_route_fuzzy(alpha, charging_cost_data, bus_cost, df_trips)
# # %%
# from collections import Counter

# def pricing_obj_for_route(
#     route_dict,
#     alpha,
#     charging_cost_data,
#     bus_cost,
#     charge_cost_premium=1.0,
#     # if your route uses GIRO IDs (e.g., 5518) but alpha uses 0..986,
#     # pass a mapping {giro_trip_id -> internal_trip_index}
#     trip_id_to_alpha_key=None,
#     # if you have actual chi_plus values, pass {(h,t): value}; else we assume 1.0 for each (h,t) listed
#     chi_plus_value=None,
#     # if True: count coverage multiplicity (if the same trip appears multiple times)
#     count_multiplicity=False,
# ):
#     obj = float(bus_cost)

#     # ---- 1) charging term ----
#     missing_prices = []
#     for (h, t) in route_dict["charging_stops"].get("chi_plus", []):
#         if (h in charging_cost_data.columns) and (t in charging_cost_data.index):
#             price_evt = float(charging_cost_data.at[t, h])
#             qty = 1.0 if chi_plus_value is None else float(chi_plus_value.get((h, t), 0.0))
#             obj += price_evt * qty * float(charge_cost_premium)
#         else:
#             missing_prices.append((h, t))

#     # ---- 2) alpha coverage term ----
#     # interpret "covered trips" as the integer nodes in the route.
#     # IMPORTANT: if these ints are GIRO IDs, you MUST map them to the alpha index space (0..986).
#     route_ints = [n for n in route_dict["route"] if isinstance(n, int)]

#     if count_multiplicity:
#         counts = Counter(route_ints)
#         covered = counts.items()
#     else:
#         covered = [(n, 1) for n in set(route_ints)]

#     unmapped = []
#     for trip_id, mult in covered:
#         key = trip_id
#         if trip_id_to_alpha_key is not None:
#             if trip_id in trip_id_to_alpha_key:
#                 key = trip_id_to_alpha_key[trip_id]
#             else:
#                 unmapped.append(trip_id)
#                 continue  # can't subtract alpha if we don't know the key
#         obj -= float(alpha.get(key, 0.0)) * float(mult)

#     return obj, {
#         "missing_prices": missing_prices,
#         "unmapped_trip_ids": unmapped,
#         "num_chi_plus_terms": len(route_dict["charging_stops"].get("chi_plus", [])),
#         "num_route_int_nodes": len(route_ints),
#         "num_covered_trips_used": len(covered),
#     }

# obj_val, debug = pricing_obj_for_route(
#     route_dict={'route': ['PARX', 5518, 5513, 5513, 5514, '7880C_0', 5514, 5518, 5501, 'PARX', 'PARX_1', 5515, 5515, 5515, 5515, 5518, 5519, 5517, 5517, 5518, 'JON_A_0', 5518, 5513, '3127L_0', 5513, 5510, 5513, 5510, '3127L_1', 5510, 5519, 'JON_A_1', 5519, 5515, 5515, 5519, 5519, 5510, '3127L_2', 5510, 5515, 5515, 5515, 5515, 5515, 'PARX'], 'charging_stops': {'stations': ['7880C_0', 'PARX_1', 'JON_A_0', '3127L_0', '3127L_1', 'JON_A_1', '3127L_2'], 'cst': [142.0, 238.0, 614.0, 668.0, 802.0, 865.0, 1006.0], 'cet': [153.0, 359.0, 624.0, 677.0, 813.0, 873.0, 1031.0], 'chi_plus_free': [], 'chi_minus_free': [], 'chi_minus': [], 'chi_plus': [('7880C_0', 142), ('7880C_0', 143), ('7880C_0', 144), ('7880C_0', 145), ('7880C_0', 146), ('7880C_0', 147), ('7880C_0', 148), ('7880C_0', 149), ('7880C_0', 150), ('7880C_0', 151), ('7880C_0', 152), ('7880C_0', 153), ('PARX_1', 238), ('PARX_1', 239), ('PARX_1', 240), ('PARX_1', 241), ('PARX_1', 242), ('PARX_1', 243), ('PARX_1', 244), ('PARX_1', 245), ('PARX_1', 246), ('PARX_1', 247), ('PARX_1', 248), ('PARX_1', 249), ('PARX_1', 250), ('PARX_1', 251), ('PARX_1', 252), ('PARX_1', 253), ('PARX_1', 254), ('PARX_1', 255), ('PARX_1', 256), ('PARX_1', 257), ('PARX_1', 258), ('PARX_1', 259), ('PARX_1', 260), ('PARX_1', 261), ('PARX_1', 262), ('PARX_1', 263), ('PARX_1', 264), ('PARX_1', 265), ('PARX_1', 266), ('PARX_1', 267), ('PARX_1', 268), ('PARX_1', 269), ('PARX_1', 270), ('PARX_1', 271), ('PARX_1', 272), ('PARX_1', 273), ('PARX_1', 274), ('PARX_1', 275), ('PARX_1', 276), ('PARX_1', 277), ('PARX_1', 278), ('PARX_1', 279), ('PARX_1', 280), ('PARX_1', 281), ('PARX_1', 282), ('PARX_1', 283), ('PARX_1', 284), ('PARX_1', 285), ('PARX_1', 286), ('PARX_1', 287), ('PARX_1', 288), ('PARX_1', 289), ('PARX_1', 290), ('PARX_1', 291), ('PARX_1', 292), ('PARX_1', 293), ('PARX_1', 294), ('PARX_1', 295), ('PARX_1', 296), ('PARX_1', 297), ('PARX_1', 298), ('PARX_1', 299), ('PARX_1', 300), ('PARX_1', 301), ('PARX_1', 302), ('PARX_1', 303), ('PARX_1', 304), ('PARX_1', 305), ('PARX_1', 306), ('PARX_1', 307), ('PARX_1', 308), ('PARX_1', 309), ('PARX_1', 310), ('PARX_1', 311), ('PARX_1', 312), ('PARX_1', 313), ('PARX_1', 314), ('PARX_1', 315), ('PARX_1', 316), ('PARX_1', 317), ('PARX_1', 318), ('PARX_1', 319), ('PARX_1', 320), ('PARX_1', 321), ('PARX_1', 322), ('PARX_1', 323), ('PARX_1', 324), ('PARX_1', 325), ('PARX_1', 326), ('PARX_1', 327), ('PARX_1', 328), ('PARX_1', 329), ('PARX_1', 330), ('PARX_1', 331), ('PARX_1', 332), ('PARX_1', 333), ('PARX_1', 334), ('PARX_1', 335), ('PARX_1', 336), ('PARX_1', 337), ('PARX_1', 338), ('PARX_1', 339), ('PARX_1', 340), ('PARX_1', 341), ('PARX_1', 342), ('PARX_1', 343), ('PARX_1', 344), ('PARX_1', 345), ('PARX_1', 346), ('PARX_1', 347), ('PARX_1', 348), ('PARX_1', 349), ('PARX_1', 350), ('PARX_1', 351), ('PARX_1', 352), ('PARX_1', 353), ('PARX_1', 354), ('PARX_1', 355), ('PARX_1', 356), ('PARX_1', 357), ('PARX_1', 358), ('PARX_1', 359), ('JON_A_0', 614), ('JON_A_0', 615), ('JON_A_0', 616), ('JON_A_0', 617), ('JON_A_0', 618), ('JON_A_0', 619), ('JON_A_0', 620), ('JON_A_0', 621), ('JON_A_0', 622), ('JON_A_0', 623), ('JON_A_0', 624), ('3127L_0', 668), ('3127L_0', 669), ('3127L_0', 670), ('3127L_0', 671), ('3127L_0', 672), ('3127L_0', 673), ('3127L_0', 674), ('3127L_0', 675), ('3127L_0', 676), ('3127L_0', 677), ('3127L_1', 802), ('3127L_1', 803), ('3127L_1', 804), ('3127L_1', 805), ('3127L_1', 806), ('3127L_1', 807), ('3127L_1', 808), ('3127L_1', 809), ('3127L_1', 810), ('3127L_1', 811), ('3127L_1', 812), ('3127L_1', 813), ('JON_A_1', 865), ('JON_A_1', 866), ('JON_A_1', 867), ('JON_A_1', 868), ('JON_A_1', 869), ('JON_A_1', 870), ('JON_A_1', 871), ('JON_A_1', 872), ('JON_A_1', 873), ('3127L_2', 1006), ('3127L_2', 1007), ('3127L_2', 1008), ('3127L_2', 1009), ('3127L_2', 1010), ('3127L_2', 1011), ('3127L_2', 1012), ('3127L_2', 1013), ('3127L_2', 1014), ('3127L_2', 1015), ('3127L_2', 1016), ('3127L_2', 1017), ('3127L_2', 1018), ('3127L_2', 1019), ('3127L_2', 1020), ('3127L_2', 1021), ('3127L_2', 1022), ('3127L_2', 1023), ('3127L_2', 1024), ('3127L_2', 1025), ('3127L_2', 1026), ('3127L_2', 1027), ('3127L_2', 1028), ('3127L_2', 1029), ('3127L_2', 1030), ('3127L_2', 1031)], 'chi_zero': []}, 'charging_activities': 7, 'type': 'truck', 'remaining_soc': 15.8902826, '_rc': -1000000000.0},              # your GIRO route dict
#     alpha=alpha,                       # your alpha dict
#     charging_cost_data=charging_cost_data,
#     bus_cost=bus_cost,
#     charge_cost_premium=charge_cost_premium,
#     #trip_id_to_alpha_key=trip_id_to_alpha_key,  # <-- if needed
#     chi_plus_value=None,               # set to dict if you have actual amounts
#     count_multiplicity=False,          # usually False for "covers trip once"
# )

# print("pricing obj:", obj_val)
# print("debug:", debug)
# # %%
# from collections import Counter

# def default_station_normalizer(h: str) -> str:
#     # strips a trailing "_<digits>" (e.g., "7880C_0" -> "7880C")
#     if isinstance(h, str) and "_" in h:
#         base, suf = h.rsplit("_", 1)
#         if suf.isdigit():
#             return base
#     return h

# def pricing_obj_for_route_breakdown(
#     route_dict,
#     alpha,
#     charging_cost_data,
#     bus_cost,
#     charge_cost_premium=1.0,
#     trip_id_to_alpha_key=None,     # dict GIRO_trip_id -> internal i in T (0..986)
#     time_map=None,                 # function t_route -> t_df (e.g., minute -> 15-min block)
#     station_map=None,              # function station_name -> df column name
#     chi_plus_qty=None,             # dict (h,t)->amount; if None assumes 1.0 each (h,t)
#     count_multiplicity=False,      # usually False: each trip covered once
# ):
#     # ---- trips covered by this route (integers in route sequence) ----
#     seq = route_dict.get("route", [])
#     trip_ids = [n for n in seq if isinstance(n, int)]
#     if not count_multiplicity:
#         trip_ids = sorted(set(trip_ids))

#     # map to alpha keys
#     alpha_keys = []
#     unmapped_trips = []
#     for tid in trip_ids:
#         k = trip_id_to_alpha_key[tid] if trip_id_to_alpha_key is not None else tid
#         if k is None:
#             unmapped_trips.append(tid)
#         else:
#             alpha_keys.append(k)

#     alpha_credit = sum(float(alpha.get(k, 0.0)) for k in alpha_keys)

#     # ---- charging cost ----
#     chi_plus = route_dict.get("charging_stops", {}).get("chi_plus", [])
#     if station_map is None:
#         station_map = default_station_normalizer
#     if time_map is None:
#         time_map = lambda t: t

#     charge_cost = 0.0
#     missing = []
#     missing_station = Counter()
#     missing_time = Counter()

#     for (h, t) in chi_plus:
#         col = station_map(h)
#         tt = time_map(t)
#         qty = 1.0 if chi_plus_qty is None else float(chi_plus_qty.get((h, t), 0.0))

#         has_col = col in charging_cost_data.columns
#         has_t = tt in charging_cost_data.index

#         if has_col and has_t:
#             price = float(charging_cost_data.at[tt, col])
#             charge_cost += price * qty * float(charge_cost_premium)
#         else:
#             missing.append((h, t, col, tt))
#             if not has_col:
#                 missing_station[col] += 1
#             if not has_t:
#                 missing_time[tt] += 1

#     obj = float(bus_cost) + float(charge_cost) - float(alpha_credit)

#     debug = {
#         "bus_cost": float(bus_cost),
#         "charge_cost": float(charge_cost),
#         "alpha_credit": float(alpha_credit),
#         "obj": float(obj),
#         "num_trips_used": len(trip_ids),
#         "num_chi_plus_terms": len(chi_plus),
#         "unmapped_trips": unmapped_trips[:20],
#         "missing_count": len(missing),
#         "missing_station_top": missing_station.most_common(10),
#         "missing_time_top": missing_time.most_common(10),
#         "missing_examples": missing[:10],
#     }
#     return obj, debug

# obj_val, debug = pricing_obj_for_route(
#     route_dict={'route': ['PARX', 5518, 5513, 5513, 5514, '7880C_0', 5514, 5518, 5501, 'PARX', 'PARX_1', 5515, 5515, 5515, 5515, 5518, 5519, 5517, 5517, 5518, 'JON_A_0', 5518, 5513, '3127L_0', 5513, 5510, 5513, 5510, '3127L_1', 5510, 5519, 'JON_A_1', 5519, 5515, 5515, 5519, 5519, 5510, '3127L_2', 5510, 5515, 5515, 5515, 5515, 5515, 'PARX'], 'charging_stops': {'stations': ['7880C_0', 'PARX_1', 'JON_A_0', '3127L_0', '3127L_1', 'JON_A_1', '3127L_2'], 'cst': [142.0, 238.0, 614.0, 668.0, 802.0, 865.0, 1006.0], 'cet': [153.0, 359.0, 624.0, 677.0, 813.0, 873.0, 1031.0], 'chi_plus_free': [], 'chi_minus_free': [], 'chi_minus': [], 'chi_plus': [('7880C_0', 142), ('7880C_0', 143), ('7880C_0', 144), ('7880C_0', 145), ('7880C_0', 146), ('7880C_0', 147), ('7880C_0', 148), ('7880C_0', 149), ('7880C_0', 150), ('7880C_0', 151), ('7880C_0', 152), ('7880C_0', 153), ('PARX_1', 238), ('PARX_1', 239), ('PARX_1', 240), ('PARX_1', 241), ('PARX_1', 242), ('PARX_1', 243), ('PARX_1', 244), ('PARX_1', 245), ('PARX_1', 246), ('PARX_1', 247), ('PARX_1', 248), ('PARX_1', 249), ('PARX_1', 250), ('PARX_1', 251), ('PARX_1', 252), ('PARX_1', 253), ('PARX_1', 254), ('PARX_1', 255), ('PARX_1', 256), ('PARX_1', 257), ('PARX_1', 258), ('PARX_1', 259), ('PARX_1', 260), ('PARX_1', 261), ('PARX_1', 262), ('PARX_1', 263), ('PARX_1', 264), ('PARX_1', 265), ('PARX_1', 266), ('PARX_1', 267), ('PARX_1', 268), ('PARX_1', 269), ('PARX_1', 270), ('PARX_1', 271), ('PARX_1', 272), ('PARX_1', 273), ('PARX_1', 274), ('PARX_1', 275), ('PARX_1', 276), ('PARX_1', 277), ('PARX_1', 278), ('PARX_1', 279), ('PARX_1', 280), ('PARX_1', 281), ('PARX_1', 282), ('PARX_1', 283), ('PARX_1', 284), ('PARX_1', 285), ('PARX_1', 286), ('PARX_1', 287), ('PARX_1', 288), ('PARX_1', 289), ('PARX_1', 290), ('PARX_1', 291), ('PARX_1', 292), ('PARX_1', 293), ('PARX_1', 294), ('PARX_1', 295), ('PARX_1', 296), ('PARX_1', 297), ('PARX_1', 298), ('PARX_1', 299), ('PARX_1', 300), ('PARX_1', 301), ('PARX_1', 302), ('PARX_1', 303), ('PARX_1', 304), ('PARX_1', 305), ('PARX_1', 306), ('PARX_1', 307), ('PARX_1', 308), ('PARX_1', 309), ('PARX_1', 310), ('PARX_1', 311), ('PARX_1', 312), ('PARX_1', 313), ('PARX_1', 314), ('PARX_1', 315), ('PARX_1', 316), ('PARX_1', 317), ('PARX_1', 318), ('PARX_1', 319), ('PARX_1', 320), ('PARX_1', 321), ('PARX_1', 322), ('PARX_1', 323), ('PARX_1', 324), ('PARX_1', 325), ('PARX_1', 326), ('PARX_1', 327), ('PARX_1', 328), ('PARX_1', 329), ('PARX_1', 330), ('PARX_1', 331), ('PARX_1', 332), ('PARX_1', 333), ('PARX_1', 334), ('PARX_1', 335), ('PARX_1', 336), ('PARX_1', 337), ('PARX_1', 338), ('PARX_1', 339), ('PARX_1', 340), ('PARX_1', 341), ('PARX_1', 342), ('PARX_1', 343), ('PARX_1', 344), ('PARX_1', 345), ('PARX_1', 346), ('PARX_1', 347), ('PARX_1', 348), ('PARX_1', 349), ('PARX_1', 350), ('PARX_1', 351), ('PARX_1', 352), ('PARX_1', 353), ('PARX_1', 354), ('PARX_1', 355), ('PARX_1', 356), ('PARX_1', 357), ('PARX_1', 358), ('PARX_1', 359), ('JON_A_0', 614), ('JON_A_0', 615), ('JON_A_0', 616), ('JON_A_0', 617), ('JON_A_0', 618), ('JON_A_0', 619), ('JON_A_0', 620), ('JON_A_0', 621), ('JON_A_0', 622), ('JON_A_0', 623), ('JON_A_0', 624), ('3127L_0', 668), ('3127L_0', 669), ('3127L_0', 670), ('3127L_0', 671), ('3127L_0', 672), ('3127L_0', 673), ('3127L_0', 674), ('3127L_0', 675), ('3127L_0', 676), ('3127L_0', 677), ('3127L_1', 802), ('3127L_1', 803), ('3127L_1', 804), ('3127L_1', 805), ('3127L_1', 806), ('3127L_1', 807), ('3127L_1', 808), ('3127L_1', 809), ('3127L_1', 810), ('3127L_1', 811), ('3127L_1', 812), ('3127L_1', 813), ('JON_A_1', 865), ('JON_A_1', 866), ('JON_A_1', 867), ('JON_A_1', 868), ('JON_A_1', 869), ('JON_A_1', 870), ('JON_A_1', 871), ('JON_A_1', 872), ('JON_A_1', 873), ('3127L_2', 1006), ('3127L_2', 1007), ('3127L_2', 1008), ('3127L_2', 1009), ('3127L_2', 1010), ('3127L_2', 1011), ('3127L_2', 1012), ('3127L_2', 1013), ('3127L_2', 1014), ('3127L_2', 1015), ('3127L_2', 1016), ('3127L_2', 1017), ('3127L_2', 1018), ('3127L_2', 1019), ('3127L_2', 1020), ('3127L_2', 1021), ('3127L_2', 1022), ('3127L_2', 1023), ('3127L_2', 1024), ('3127L_2', 1025), ('3127L_2', 1026), ('3127L_2', 1027), ('3127L_2', 1028), ('3127L_2', 1029), ('3127L_2', 1030), ('3127L_2', 1031)], 'chi_zero': []}, 'charging_activities': 7, 'type': 'truck', 'remaining_soc': 15.8902826, '_rc': -1000000000.0},              # your GIRO route dict
#     alpha=alpha,                       # your alpha dict
#     charging_cost_data=charging_cost_data,
#     bus_cost=bus_cost,
#     charge_cost_premium=charge_cost_premium,
#     #trip_id_to_alpha_key=trip_id_to_alpha_key,  # <-- if needed
#     chi_plus_value=None,               # set to dict if you have actual amounts
#     count_multiplicity=False,          # usually False for "covers trip once"
# )

# print("pricing obj:", obj_val)
# print("debug:", debug)
# %%

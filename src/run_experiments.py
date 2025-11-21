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

from config import (
    n_fast_cols, n_exact_cols, tolerance,
    bar_t, time_blocks,
    DEPOT_NAME, SOC_CAPACITY_KWH, ENERGY_PER_BLOCK_KWH,
    CHARGING_POWER_KW, CHARGE_EFFICIENCY,
    charge_mult, charge_cost_premium,
    BUS_COST_SCALAR, ALLOW_ONLY_LISTED_DEADHEADS, MODE_EVS_ONLY,
    CHARGING_STATIONS,

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

# ------------------------------ Output dirs ------------------------------
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)
CKPT = OUTDIR / f"ckpt_{RUN_ID}"
CKPT.mkdir(parents=True, exist_ok=True)

# ------------------------------ Helpers ------------------------------

def _nearest_hour_block(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    hh = int(hh); mm = int(mm)
    blk = hh + (1 if mm >= 30 else 0)
    blk = max(1, min(24, blk if blk > 0 else 1))
    return blk

def _ceil_hour_block(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    hh = int(hh); mm = int(mm)
    blk = hh if mm == 0 else hh + 1
    blk = max(1, min(24, blk))
    return blk

def ceil_hours_from_minutes(m: float) -> int:
    return int(math.ceil(float(m) / 60.0))

def energy_to_events(kwh: float) -> int:
    return int(math.ceil(float(kwh) / float(ENERGY_PER_BLOCK_KWH)))

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

routes_csv    = DATA_DIR / "Par_Routes_For_Code.csv"
deadheads_csv = DATA_DIR / "Par_DHD_for_code.csv"
prices_csv    = DATA_DIR / "hourly_prices.csv"

if not routes_csv.exists():
    raise FileNotFoundError(f"Missing {routes_csv}")
if not deadheads_csv.exists():
    raise FileNotFoundError(f"Missing {deadheads_csv}")
if not prices_csv.exists():
    raise FileNotFoundError(f"Missing {prices_csv} (needed for charging prices)")

df_trips = pd.read_csv(routes_csv)



# #### filter out trips ####

# BAD_TRIPS_INDICES = [13, 14, 19, 23, 34, 43, 96, 99, 101, 103, 120, 155, 164, 181, 187, 207, 210, 212, 251, 272, 292, 304, 326, 346, 361, 365, 368, 369, 406, 407, 423, 425, 437, 438, 460, 526, 527, 530, 531, 546, 576, 589, 601, 671, 721, 724, 725, 762, 763, 800, 801, 812, 821, 866, 869, 886, 902, 903, 934, 942, 974]


# print(f"[FILTER] Original trip count: {len(df_trips)}")
# print(f"[FILTER] Removing {len(BAD_TRIPS_INDICES)} trips known to be uncovered...")

# # Filter out rows by index
# df_trips = df_trips.drop(BAD_TRIPS_INDICES, errors='ignore')

# # IMPORTANT: Reset index so the resulting T list is contiguous (0, 1, 2...)
# # This ensures the rest of your code (which assumes Trip ID maps to range(N)) remains valid.
# df_trips = df_trips.reset_index(drop=True)

# print(f"[FILTER] New trip count: {len(df_trips)}")


# #### filter out trips ####



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

# ----------------------------------------------------------------------
# SINGLE canonical CHARGERS definition (use config + depot)
# ----------------------------------------------------------------------
CHARGERS = sorted(set(CHARGING_STATIONS + [DEPOT_NAME]))

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

df_trips["st_blk"] = df_trips["ST"].astype(str).map(_nearest_hour_block)
df_trips["et_blk"] = df_trips["ET"].astype(str).map(_ceil_hour_block)

df_trips["eps_events"] = df_trips["Energy used"].map(energy_to_events)

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
    tau_blk = ceil_hours_from_minutes(dur_min)
    d_evt   = _energy_to_events(eng_kwh)
    allowed_arcs[(f, t)] = (tau_blk, d_evt)

def arc_from_to(from_node: str, to_node: str):
    return allowed_arcs.get((from_node, to_node), None)

# ------------------------------ Globals for pricing ------------------------------

S = CHARGERS[:]  # stations set (includes depot name too)
G = energy_to_events(SOC_CAPACITY_KWH)
DEPOT = DEPOT_NAME

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

STATION_BASES = sorted(set(S))
charging_cost_data, avg_cost_per_kwh = load_price_curve(
    str(prices_csv), time_blocks, STATION_BASES
)
charging_cost_data = charging_cost_data * float(ENERGY_PER_BLOCK_KWH)
avg_cost_per_event = float(avg_cost_per_kwh) * float(ENERGY_PER_BLOCK_KWH)

bus_cost  = BUS_COST_SCALAR * avg_cost_per_event

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
    MAX_TAU = 2  # at most 2 hour-blocks between nodes in pricing graph

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

    v_amt = {h: pricing_model.addVar(lb=-G, ub=G, vtype=GRB.INTEGER, name=f"v_{h}") for h in S_use}
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

    for i in T:
        if (DEPOT, i) in tau:
            pricing_model.addGenConstrIndicator(
                wA_trip[i], 1, t_in[i] - (t_out[DEPOT] + tau[(DEPOT, i)]), GRB.GREATER_EQUAL, 0,
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
                wA_station[h], 1, t_in[h] - (t_out[DEPOT] + tau[(DEPOT, h)]), GRB.GREATER_EQUAL, 0,
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
            v_amt[h] == charge_mult * quicksum(chi_plus[(h,t)] for t in time_blocks),
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
                    price_evt = float(charging_cost_data.at[t, h])
                    obj += price_evt * chi_plus[h, t] * charge_cost_premium

    for i in T:
        a_i = alpha.get(i, 0.0)
        cov_expr = wA_trip[i] \
                   + quicksum(x[(j, i)] for (j, i2) in x.keys() if i2 == i) \
                   + quicksum(z[(h, i)] for (h, i2) in z.keys() if i2 == i)
        obj -= a_i * cov_expr

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
print("[FINAL] Locking out dummy variables (forcing q_i = 0)...")
locked_count = 0
for i in T:
    # Retrieve the slack variable by name
    q_var = rmp_final.getVarByName(f"q_{i}")
    if q_var is not None:
        # Force it to 0. The solver MUST cover trip i with a real vehicle OR return Infeasible.
        q_var.UB = 0.0 
        locked_count += 1

print(f"[FINAL] Locked {locked_count} dummy variables.")
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

# %%

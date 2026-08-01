# EVSP only (no solar, no V2G/V2V) on your real data.
# Speedups:
#  - Smaller pool + RC cutoff + K-best columns per CG iteration
#  - Pricing timelimit + gap; master LP timelimit; stagnation early-stop
#  - Nodefile spill to avoid OOM kills; auto TMP detection
#  - Final MIP with warm-start and relaxed gap
#%%

import os
import time
import math
import datetime
from pathlib import Path
import json
import sys
import argparse
import collections
import hashlib
import subprocess

import pandas as pd
import gurobipy as gp
from gurobipy import Column, GRB
# from collections import Counter, defaultdict

from config import (
    bar_t, TIMEBLOCKS_PER_HOUR,
    DEPOT_NAME, CHARGE_RATE_KW,
    charge_cost_premium, BUS_COST_KX,
    CHARGING_STATIONS, STATION_COPIES,
    RC_EPSILON,
    THREADS, NODEFILE_START, NODEFILE_DIR,
    MASTER_TIMELIMIT,
    CHARGE_START_COST
)


from pricing_dp_og import make_dp_pricer

from utils_v2 import (
    extract_duals,
    calculate_truck_route_cost_accurate,
    load_station_hourly_prices, route_column_key,
    select_unique_station_copies,
)


from master import init_master, solve_master, build_master

import re

stopwatch_start = time.time()
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
ROOT = Path(__file__).resolve().parent


# ==========================================
# 1. COMMAND LINE ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description="EVSP Column Generation")
parser.add_argument("--csv", type=str, required=True, help="Input CSV (e.g., Practice_10bus.csv)")
parser.add_argument("--G", type=int, required=True, help="Battery capacity (300 for EVSP, 9999 for VSP/infcharge)")
parser.add_argument("--kbest", type=int, default=150, help="Number of columns to add per iteration")
parser.add_argument("--max_labels", type=int, default=200000, help="DP max labels per node")
parser.add_argument('--stagnation_window', type=int, default=50, help='Lookback window for rolling stagnation')
parser.add_argument('--improvement_bound', type=float, default=5.0, help='Minimum rolling improvement over the window')
parser.add_argument('--no_resume', action='store_true', help='Force a fresh start, ignoring RESUME_CKPT')
parser.add_argument(
    "--prices_csv",
    type=str,
    default="spatiotemporal_prices.csv",
    help="Spatiotemporal price CSV. Relative paths are resolved inside data/.",
)
parser.add_argument(
    "--price_tag",
    type=str,
    default=None,
    help="Short label added to output directory names, e.g. peak08.",
)
parser.add_argument(
    "--milestones_hours",
    type=str,
    default="3,6",
    help="Comma-separated active-compute-hour snapshots, e.g. '3,6'. Empty disables milestones.",
)
parser.add_argument(
    "--active_time_limit_hours",
    type=float,
    default=6.0,
    help="Column-generation active-compute limit in hours; 0 disables the Python-side limit.",
)
parser.add_argument(
    "--pricing_tiers",
    type=str,
    default=None,
    help="Comma-separated MAX_LABELS:SECONDS tiers. Default uses --max_labels at 500s and 3000s.",
)
parser.add_argument(
    "--pricing_wall_per_iter",
    type=int,
    default=6000,
    help="Maximum total pricing wall time per CG iteration in seconds.",
)
parser.add_argument(
    "--master_time_limit",
    type=float,
    default=MASTER_TIMELIMIT,
    help="Per-iteration master-LP time limit in seconds.",
)
parser.add_argument(
    "--target_master_obj",
    type=float,
    default=None,
    help="Optional experimental LP objective stop. Disabled by default; final fleet count comes from the MIP.",
)
parser.add_argument(
    "--run_tag",
    type=str,
    default=None,
    help="Experiment identifier included in the result directory and checkpoint names.",
)
parser.add_argument(
    "--min_trips_per_route",
    type=int,
    default=1,
    help="Minimum trips in a DP column. Default 1 avoids excluding feasible routes.",
)
parser.add_argument(
    "--allow_unsafe_resume",
    action="store_true",
    help="Allow a checkpoint from another commit/config (unsafe; provenance is retained).",
)
parser.add_argument(
    "--results_root",
    type=str,
    default=None,
    help="Optional results directory. Relative paths are resolved from the repository root.",
)
parser.add_argument(
    "--final_mip_timelimit",
    type=int,
    default=3600,
    help="Final MIP time limit in seconds after column generation stops.",
)
parser.add_argument(
    "--skip_final_mip",
    action="store_true",
    help="Stop after column generation/checkpoint snapshots; do not build or solve the final MIP.",
)
parser.add_argument(
    "--cheat", action="store_true",
    help="If set, warm-start R_truck with the buses' actual driven routes "
         "from Par_VehicleDetails_Updated.csv (bus IDs inferred from the input CSV's "
         "VehicleTask column). If not set, R_truck starts empty."
)
parser.add_argument(
    "--greedy",
    action="store_true",
    help="Warm-start R_truck with a depot/charging-aware greedy route cover "
         "instead of historical VehicleTask routes."
)
args = parser.parse_args()

if args.cheat and args.greedy:
    raise SystemExit("ERROR: --cheat and --greedy are mutually exclusive.")
if args.kbest <= 0:
    raise ValueError("--kbest must be positive")
if args.max_labels <= 0:
    raise ValueError("--max_labels must be positive")
if args.stagnation_window <= 0:
    raise ValueError("--stagnation_window must be positive")
if args.pricing_wall_per_iter <= 30:
    raise ValueError("--pricing_wall_per_iter must be greater than 30 seconds")
if args.master_time_limit <= 0:
    raise ValueError("--master_time_limit must be positive")
if args.min_trips_per_route <= 0:
    raise ValueError("--min_trips_per_route must be positive")
if args.final_mip_timelimit <= 0:
    raise ValueError("--final_mip_timelimit must be positive")

csv_name = args.csv
G_PARAM = args.G
K_BEST = args.kbest
MAX_LABELS_PER_NODE = args.max_labels
MAX_CG_ITERS = 99999
PRICING_WALL_PER_ITER = int(args.pricing_wall_per_iter)
MASTER_TIME_LIMIT = float(args.master_time_limit)


def _parse_pricing_tiers(spec):
    if not spec:
        return [(int(MAX_LABELS_PER_NODE), 500), (int(MAX_LABELS_PER_NODE), 3000)]
    tiers = []
    for raw in spec.split(","):
        labels, seconds = raw.strip().split(":", 1)
        labels, seconds = int(labels), int(seconds)
        if labels <= 0 or seconds <= 0:
            raise ValueError("Pricing tier labels and seconds must be positive")
        tiers.append((labels, seconds))
    if not tiers:
        raise ValueError("--pricing_tiers did not contain any tiers")
    return tiers


ESCALATION_SCHEDULE = _parse_pricing_tiers(args.pricing_tiers)

TARGET_MILESTONES_HOURS = sorted(
    float(x.strip()) for x in args.milestones_hours.split(",") if x.strip()
)
if any(value <= 0 for value in TARGET_MILESTONES_HOURS):
    raise ValueError("--milestones_hours entries must be positive")
if args.active_time_limit_hours < 0:
    raise ValueError("--active_time_limit_hours must be nonnegative")
ACTIVE_TIME_LIMIT_HOURS = float(args.active_time_limit_hours)

# ==========================================
# 2. FILE PATHS
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
if args.results_root:
    OUTDIR = Path(args.results_root)
    if not OUTDIR.is_absolute():
        OUTDIR = ROOT_DIR / OUTDIR
else:
    OUTDIR = ROOT_DIR / "src" / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)

def _resolve_data_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return DATA_DIR / path

def _safe_tag(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("_")
    return cleaned or "untagged"


def _hours_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p") + "h"


def _git_state():
    def _run(*git_args):
        result = subprocess.run(
            ["git", *git_args], cwd=ROOT_DIR, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("branch", "--show-current"),
        "dirty": bool(_run("status", "--porcelain")),
    }

# Dynamically point to the correct CSVs in the data folder
routes_csv = DATA_DIR / csv_name
ref_dhd_csv = DATA_DIR / "par_ref_dhd.csv"
ref_dict_csv = DATA_DIR / "Ref_dict.csv"
master_csv = DATA_DIR / "Par_VehicleDetails_Updated.csv"
prices_csv = _resolve_data_path(args.prices_csv)
price_tag = _safe_tag(args.price_tag or prices_csv.stem)
run_tag = _safe_tag(args.run_tag) if args.run_tag else None
git_state = _git_state()
runtime_versions = {
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "gurobi": ".".join(str(part) for part in gp.gurobi.version()),
}

print(f"[INIT] Using trip data: {routes_csv.name}")
print(f"[INIT] Battery Capacity parameter: {G_PARAM}")
print(f"[INIT] Git state: {git_state}")
print(f"[INIT] Runtime versions: {runtime_versions}")

# ==========================================
# 3. DYNAMIC TARGET & VSP MODE OVERRIDE
# ==========================================
if G_PARAM >= 9000:
    SAFE_G = 300
else:
    SAFE_G = G_PARAM

G = SAFE_G  # ADD THIS LINE — override config G with the correct value

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

MAX_DAILY_RECHARGES = 15  # Buffer above observed max of 13
MIN_TRIPS_PER_ROUTE = args.min_trips_per_route


#%%



# if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
#     csv_name = sys.argv[1]
# else:
#     csv_name = "Practice_20bus.csv"

routes_csv = DATA_DIR / csv_name

ref_dhd_csv   = DATA_DIR / "par_ref_dhd.csv"
ref_dict_csv  = DATA_DIR / "Ref_dict.csv"

# Create a dynamic name based on the input file (e.g., "Practice_Selected_4buses")
# Create a dedicated directory for EVERYTHING from this run
DATA_NAME = routes_csv.stem  # e.g., "Inst_10B_G01_13301_13310"
if args.cheat:
    run_mode = "CHEAT"
    mode_suffix = "_CHEAT"
elif args.greedy:
    run_mode = "GREEDY"
    mode_suffix = "_GREEDY"
else:
    run_mode = "NO_CHEAT"
    mode_suffix = "_NO_CHEAT"
mode_suffix += f"_stag{args.stagnation_window}_imp{args.improvement_bound}"
bus_label = f"{DATA_NAME}{mode_suffix}_{price_tag}"
if run_tag:
    bus_label += f"_{run_tag}"

RUN_DIR = OUTDIR / f"{bus_label}_g{G_PARAM}_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] All outputs and logs will be saved to: {RUN_DIR}")
print(f"[INFO] Price CSV: {prices_csv}")
print(f"[INFO] Price tag: {price_tag}")
print(f"[INFO] Milestones: {TARGET_MILESTONES_HOURS}")
print(f"[INFO] Active-time limit: {ACTIVE_TIME_LIMIT_HOURS:g}h"
      if ACTIVE_TIME_LIMIT_HOURS else "[INFO] Active-time limit: disabled")
print(f"[INFO] Pricing tiers: {ESCALATION_SCHEDULE}")

if not routes_csv.exists():
    raise FileNotFoundError(f"Missing {routes_csv}")
if not ref_dhd_csv.exists():
    raise FileNotFoundError(f"Missing {ref_dhd_csv}")
if not ref_dict_csv.exists():
    raise FileNotFoundError(f"Missing {ref_dict_csv}")
if not prices_csv.exists():
    raise FileNotFoundError(f"Missing {prices_csv} (needed for charging prices)")

instance_sha256 = hashlib.sha256(routes_csv.read_bytes()).hexdigest()
price_sha256 = hashlib.sha256(prices_csv.read_bytes()).hexdigest()

df_trips = pd.read_csv(routes_csv)


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
        return (0, 0.0, 0.0)

    # # Avoid copy-to-copy teleporting for the same physical station.
    # if _is_copy_node(a) and _is_copy_node(b):
    #     if strip_copy_suffix(a) == strip_copy_suffix(b):
    #         return None

    ref_a = resolve_ref(a)
    ref_b = resolve_ref(b)
    if ref_a is None or ref_b is None:
        return None
    if ref_a == ref_b:
        return (0, 0.0, 0.0)

    pair = ref_pair_minutes_kwh.get(_ordered_ref_pair(ref_a, ref_b))
    if pair is None:
        return None

    dur_min, eng_kwh = pair
    return (ceil_blocks_from_minutes(dur_min), float(dur_min), float(eng_kwh))


# ------------------------------ Build trip set ------------------------------

df_trips = df_trips.reset_index(drop=True).copy()
df_trips["Trip"] = df_trips.index

df_trips["st_blk"] = df_trips["ST"].astype(str).map(_floor_block)
df_trips["et_blk"] = df_trips["ET"].astype(str).map(_ceil_block)

df_trips["eps_kwh"] = df_trips["Energy used"].astype(float)

df_trips["st_min"] = df_trips["ST"].astype(str).map(_total_minutes)
df_trips["et_min"] = df_trips["ET"].astype(str).map(_total_minutes)

T = list(df_trips["Trip"].tolist())

# Map global Ordered_Trip_ID -> local Trip index for cheat warm-start routes.
ordered_to_local = {}
if "Ordered_Trip_ID" in df_trips.columns:
    ordered_to_local = dict(zip(
        df_trips["Ordered_Trip_ID"].dropna().astype(int),
        df_trips["Trip"],
    ))
    print(f"[INFO] Built Ordered_Trip_ID -> local Trip map: {len(ordered_to_local)} entries")

sl = df_trips.set_index("Trip")["SL"].to_dict()
el = df_trips.set_index("Trip")["EL"].to_dict()
st = df_trips.set_index("Trip")["st_blk"].to_dict()
et = df_trips.set_index("Trip")["et_blk"].to_dict()
epsilon = df_trips.set_index("Trip")["eps_kwh"].to_dict()

st_min = {i: int(df_trips.set_index("Trip")["st_min"].to_dict()[i]) for i in T}
et_min = {i: int(df_trips.set_index("Trip")["et_min"].to_dict()[i]) for i in T}
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
tau_min = {}
d   = {}  # deadhead energy (kWh) for each pricing arc

# Depot <-> trip
for i in T:
    pair = arc_from_to(DEPOT, sl[i])
    if pair is not None:
        tau[(DEPOT, i)] = pair[0]; tau_min[(DEPOT, i)] = pair[1]; d[(DEPOT, i)] = pair[2]
    pair = arc_from_to(el[i], DEPOT)
    if pair is not None:
        tau[(i, DEPOT)] = pair[0]; tau_min[(i, DEPOT)] = pair[1]; d[(i, DEPOT)] = pair[2]

# Trip -> Trip
for i in T:
    for j in T:
        if i == j: continue
        pair = arc_from_to(el[i], sl[j])
        if pair is not None:
            tau[(i, j)] = pair[0]; tau_min[(i, j)] = pair[1]; d[(i, j)] = pair[2]

# Trip <-> Station
for i in T:
    for h in S:
        pair1 = arc_from_to(el[i], h)
        if pair1 is not None:
            tau[(i, h)] = pair1[0]; tau_min[(i, h)] = pair1[1]; d[(i, h)] = pair1[2]
        pair2 = arc_from_to(h, sl[i])
        if pair2 is not None:
            tau[(h, i)] = pair2[0]; tau_min[(h, i)] = pair2[1]; d[(h, i)] = pair2[2]

# Zero-hop station-trip links are implicit when location refs match in arc_from_to().


# Station <-> Depot
for h in S:
    pair1 = arc_from_to(DEPOT, h)
    if pair1 is not None:
        tau[(DEPOT, h)] = pair1[0]; tau_min[(DEPOT, h)] = pair1[1]; d[(DEPOT, h)] = pair1[2]
    pair2 = arc_from_to(h, DEPOT)
    if pair2 is not None:
        tau[(h, DEPOT)] = pair2[0]; tau_min[(h, DEPOT)] = pair2[1]; d[(h, DEPOT)] = pair2[2]

def has_depot_pull(i):
    return ((DEPOT, i) in tau) and ((i, DEPOT) in tau)

unseedable = [i for i in T if not has_depot_pull(i)]
if unseedable:
    print("[WARN] Trips lacking depot pull-out or pull-in in DHD (cannot seed O->i->O):", unseedable)

# ------------------------------ Price curve ------------------------------

# price curve should be indexed by PHYSICAL station names (base), not copies
STATION_BASES = sorted(set(strip_copy_suffix(s) for s in S))


station_hourly_prices = load_station_hourly_prices(prices_csv, STATION_BASES)
hourly_prices = station_hourly_prices[DEPOT_BASE]
MAX_HOUR = int(max(hourly_prices.keys()))
bus_cost = BUS_COST_KX
print(f"[INFO] Loaded spatiotemporal prices for: {list(station_hourly_prices.keys())}")
print(f"[INFO] Max Hour: {MAX_HOUR}")

VALID_STATIONS = ["2190L", "4808", "3127L", "7880C", "JON_A", "PARX"]


def match_station_name(raw_name):
    """Prioritize exact station matches, then fall back to substring matches."""
    raw_str = str(raw_name)
    if raw_str in VALID_STATIONS:
        return raw_str

    for valid in VALID_STATIONS:
        if raw_str in valid or valid in raw_str:
            return valid
    return raw_str


def get_initial_routes_from_csv(
    vehicle_details_path,
    target_bus_ids,
    *,
    depot,
    station_node_by_base,
):
    df = pd.read_csv(vehicle_details_path)
    df["VehicleTask_Str"] = df["VehicleTask"].astype(str)
    target_ids_str = [str(x) for x in target_bus_ids]

    mask_regular = (df["Identifier"] == "Regular") & (df["VehicleTask_Str"].isin(target_ids_str))
    regular_df = df[mask_regular].copy()
    regular_df = regular_df[regular_df["Ordered_Trip_ID"].notna()].copy()
    regular_df["Ordered_Trip_ID"] = regular_df["Ordered_Trip_ID"].astype(int)

    deadhead_identifiers = {"Deadhead", "Pull-out", "Pull-in", "Prep-out", "Prep-in"}

    routes = []
    for bus_id in target_ids_str:
        bus_df = df[df["VehicleTask_Str"] == bus_id].copy()

        route_nodes = [depot]
        stations, csts, cets, kwhs = [], [], [], []
        deadhead_kwh = 0.0

        for _, row in bus_df.iterrows():
            identifier = row["Identifier"]

            if identifier in deadhead_identifiers:
                usage = row["Usage kWh"]
                if pd.notna(usage):
                    deadhead_kwh += float(usage)

            if identifier == "Regular":
                ord_id = row["Ordered_Trip_ID"]
                if pd.notna(ord_id):
                    route_nodes.append(int(ord_id))

            elif identifier == "Recharge":
                matched_loc = match_station_name(row["From1"])
                station_node = station_node_by_base.get(matched_loc)
                if station_node is None:
                    raise ValueError(
                        f"Historical recharge station {matched_loc!r} is not in the pricing graph"
                    )

                if route_nodes[-1] != station_node:
                    route_nodes.append(station_node)
                    stations.append(station_node)
                    csts.append(parse_time_to_minutes(row["Start1"]))
                    cets.append(parse_time_to_minutes(row["End1"]))
                    kwhs.append(float(row["Recharge kWh"]) if pd.notna(row["Recharge kWh"]) else 0.0)

        if len(route_nodes) > 1 and route_nodes[-1] != depot:
            route_nodes.append(depot)

        trip_count = sum(1 for n in route_nodes if isinstance(n, int))
        if trip_count == 0:
            print(f"Warning: Bus {bus_id} produced 0 trips - skipping")
            continue

        routes.append({
            "route": route_nodes,
            "charging_stops": {"stations": stations, "cst": csts, "cet": cets, "kwh": kwhs},
            "charging_activities": len(stations),
            "type": "truck",
            "deadhead_kwh": deadhead_kwh,
            "_rc": 0.0,
            "desc": f"Imported full-day route for bus {bus_id}",
        })

    return routes, regular_df

# ------------------------------ Charging station subset for DP/greedy ------------------------------

S_use_set = set()
for i in T:
    for h in S:
        if (i, h) in tau and (et[i] + tau[(i, h)] <= bar_t):
            S_use_set.add(h)
        if (h, i) in tau and (tau[(h, i)] <= st[i]):
            S_use_set.add(h)
for h in S:
    if (DEPOT, h) in d or (h, DEPOT) in d:
        S_use_set.add(h)
S_use = sorted(list(S_use_set))
print(f"[INFO] Expanded reachable station copies: {len(S_use)}")

# Copies are interchangeable because this model has no charger-capacity
# constraints. Keep one per physical station, but never reuse the source/sink
# depot node as the PARX charging node.
S_price = select_unique_station_copies(S_use, DEPOT)
print(f"[INFO] Pricing/greedy station set: {len(S_use)} copies -> {len(S_price)} physical nodes")

#%%
# ------------------------------ Seed routes ------------------------------

R_truck = []
start_iteration = 0
prev_cum_master = 0.0
prev_cum_pricing = 0.0
resumed_stats_csv = None
resume_count = 0
seed_route_count = 0
dp_columns_generated = 0
resume_history = []
seed_route_validation = "not_applicable"

RESUME_CKPT = os.environ.get("RESUME_CKPT", "")
data = {}
recent_improvements = collections.deque(maxlen=args.stagnation_window)
last_master_obj = None
is_resuming = False
termination_reason = "unknown"
milestones_passed = []

if RESUME_CKPT and Path(RESUME_CKPT).exists() and not args.no_resume:
    with open(RESUME_CKPT, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        required_metadata = (
            "csv_name",
            "instance_sha256",
            "price_sha256",
            "mode",
            "battery_kwh",
            "trip_ids",
            "git",
            "run_arguments",
        )
        missing_metadata = [
            key for key in required_metadata
            if data.get(key) is None
        ]
        if missing_metadata and not args.allow_unsafe_resume:
            raise ValueError(
                "RESUME_CKPT lacks required provenance fields "
                f"{missing_metadata}. Use a current checkpoint or explicitly pass "
                "--allow_unsafe_resume for a legacy pool."
            )

        resume_expectations = {
            "csv_name": csv_name,
            "instance_sha256": instance_sha256,
            "price_sha256": price_sha256,
            "mode": run_mode,
            "battery_kwh": G_PARAM,
        }
        mismatches = {
            key: (data.get(key), expected)
            for key, expected in resume_expectations.items()
            if data.get(key) is not None and data.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"RESUME_CKPT does not match this run: {mismatches}")

        checkpoint_git = data.get("git") or {}
        checkpoint_commit = checkpoint_git.get("commit")
        current_commit = git_state.get("commit")
        if not checkpoint_commit and not args.allow_unsafe_resume:
            raise ValueError(
                "RESUME_CKPT does not record git.commit. Use a current checkpoint "
                "or explicitly pass --allow_unsafe_resume for a legacy pool."
            )
        if (
            checkpoint_commit
            and current_commit
            and checkpoint_commit != current_commit
            and not args.allow_unsafe_resume
        ):
            raise ValueError(
                "RESUME_CKPT was generated by Git commit "
                f"{checkpoint_commit}, but this checkout is {current_commit}. "
                "Use the original commit, start a fresh run tag, or explicitly pass "
                "--allow_unsafe_resume."
            )

        saved_args = data.get("run_arguments") or {}
        resume_critical_args = (
            "kbest",
            "max_labels",
            "pricing_tiers",
            "pricing_wall_per_iter",
            "min_trips_per_route",
            "stagnation_window",
            "improvement_bound",
            "cheat",
            "greedy",
        )
        config_mismatches = {
            key: (saved_args.get(key), vars(args).get(key))
            for key in resume_critical_args
            if key in saved_args and saved_args.get(key) != vars(args).get(key)
        }
        missing_config = [
            key for key in resume_critical_args
            if key not in saved_args
        ]
        if missing_config and not args.allow_unsafe_resume:
            raise ValueError(
                "RESUME_CKPT lacks critical run arguments "
                f"{missing_config}. Use a current checkpoint or explicitly pass "
                "--allow_unsafe_resume for a legacy pool."
            )
        if config_mismatches and not args.allow_unsafe_resume:
            raise ValueError(
                "RESUME_CKPT uses different algorithm arguments: "
                f"{config_mismatches}. Use the original settings, a fresh run tag, "
                "or explicitly pass --allow_unsafe_resume."
            )

        R_truck = data["routes"]
        start_iteration = data.get("iteration", 0)
        prev_cum_master = data.get(
            "cum_master_time", data.get("cumulative_master_time_s", 0.0)
        )
        prev_cum_pricing = data.get(
            "cum_pricing_time", data.get("cumulative_pricing_time_s", 0.0)
        )
        resumed_stats_csv = data.get("stats_csv_path")
        resume_count = data.get("resume_count", 0)
        seed_route_count = data.get("seed_route_count")
        dp_columns_generated = data.get("dp_columns_generated")
        if seed_route_count is None or dp_columns_generated is None:
            # Old checkpoints did not distinguish seeds from DP columns.
            seed_route_count = len(R_truck)
            dp_columns_generated = 0
            print("[RESUME WARN] Old checkpoint lacks seed/DP column counts; "
                  "treating the existing pool as seed provenance.")
        resume_history = list(data.get("resume_history", []))
        seed_route_validation = data.get("seed_route_validation", "unknown")
        resume_history.append({
            "source_checkpoint": str(RESUME_CKPT),
            "source_git": checkpoint_git,
            "resumed_with_git": git_state,
            "cross_commit": bool(
                checkpoint_commit and current_commit and checkpoint_commit != current_commit
            ),
        })

        last_master_obj = data.get("last_master_obj", None)
        recent_improvements = collections.deque(
            data.get("recent_improvements", []),
            maxlen=args.stagnation_window,
        )
        termination_reason = data.get("termination_reason", "unknown")
        milestones_passed = data.get("milestones_passed", [])
        print(f"[RESUME] Restored {len(recent_improvements)} entries in stagnation window")
        if milestones_passed:
            print(f"[RESUME] Already saved milestones: {milestones_passed}")

        if "run_dir" in data:
            recorded_run_dir = Path(data["run_dir"])
            RUN_DIR = recorded_run_dir if recorded_run_dir.exists() else Path(RESUME_CKPT).parent
    else:
        if not args.allow_unsafe_resume:
            raise ValueError(
                "Legacy list-only RESUME_CKPT has no instance/price/code provenance. "
                "Explicitly pass --allow_unsafe_resume to use it."
            )
        R_truck = data # Compatibility for old list-only format
        seed_route_count = len(R_truck)
        seed_route_validation = "unknown_legacy_checkpoint"

    print(f"[RESUME] Loaded {len(R_truck)} routes. Resuming from Iteration {start_iteration + 1}")
    is_resuming = True

if not is_resuming:
    if args.no_resume:
        print("\n[START] --no_resume flag active. Forcing fresh start.")
    else:
        print("\n[START] No checkpoint found. Fresh start.")

    if args.cheat:
        _raw = pd.read_csv(routes_csv)
        if not ordered_to_local:
            raise ValueError(f"--cheat requires an 'Ordered_Trip_ID' column in {csv_name}")

        if "VehicleTask" in _raw.columns:
            inferred_tasks = _raw["VehicleTask"]
        else:
            # The tracked Practice_10bus/15bus inputs intentionally contain a
            # compact model schema. Recover their historical block labels from
            # the tracked, ordered derived table without rewriting the input.
            master_tasks = pd.read_csv(master_csv)
            master_tasks = master_tasks[master_tasks["Identifier"] == "Regular"].copy()
            master_tasks["Ordered_Trip_ID"] = pd.to_numeric(
                master_tasks["Ordered_Trip_ID"], errors="raise"
            ).astype(int)
            if master_tasks["Ordered_Trip_ID"].duplicated().any():
                raise ValueError("Par_VehicleDetails_Updated.csv has duplicate Ordered_Trip_ID values")
            task_by_ordered = master_tasks.set_index("Ordered_Trip_ID")["VehicleTask"]
            input_ordered = pd.to_numeric(_raw["Ordered_Trip_ID"], errors="raise").astype(int)
            inferred_tasks = input_ordered.map(task_by_ordered)
            if inferred_tasks.isna().any():
                missing_ids = input_ordered[inferred_tasks.isna()].tolist()
                raise ValueError(
                    f"Cannot infer VehicleTask for Ordered_Trip_ID values: {missing_ids[:20]}"
                )
            print(f"[CHEAT] Inferred VehicleTask from {master_csv.name}; input CSV was compact.")

        cheat_target_buses = inferred_tasks.dropna().astype(str).unique().tolist()
        print(f"[CHEAT] Inferred {len(cheat_target_buses)} buses from {csv_name}: {cheat_target_buses}")

        csv_routes, _ = get_initial_routes_from_csv(
            vehicle_details_path=DATA_DIR / "Par_VehicleDetails_Updated.csv",
            target_bus_ids=cheat_target_buses,
            depot=DEPOT,
            station_node_by_base={strip_copy_suffix(s): s for s in S_price},
        )

        for route in csv_routes:
            bus = route["desc"].split()[-1]
            new_route_nodes = []
            unknown_trip = None

            for node in route["route"]:
                if isinstance(node, int):
                    local_id = ordered_to_local.get(node)
                    if local_id is None:
                        unknown_trip = node
                        break
                    new_route_nodes.append(int(local_id))
                else:
                    new_route_nodes.append(node)

            if unknown_trip is not None:
                print(f"[CHEAT -] Skipped bus {bus}: trip Ordered_Trip_ID={unknown_trip} "
                      f"is not in this instance.")
                continue

            route["route"] = new_route_nodes
            R_truck.append(route)
            print(f"[CHEAT +] Added warm-start route for bus {bus} "
                  f"({sum(1 for n in new_route_nodes if isinstance(n, int))} trips)")

        print(f"[CHEAT] Seeded R_truck with {len(R_truck)} warm-start routes.")
        seed_route_validation = "historical_coverage_import_not_time_soc_validated"
    elif args.greedy:
        from greedy_init import build_greedy_routes

        print("[GREEDY] Constructing warm-start routes from trip/depot/station arcs...")
        greedy_routes = build_greedy_routes(
            T=T,
            S_use=S_price,
            DEPOT=DEPOT,
            tau=tau,
            tau_min=tau_min,
            d=d,
            st=st,
            et=et,
            st_min=st_min,
            et_min=et_min,
            sl=sl,
            el=el,
            epsilon=epsilon,
            G=G,
            bar_t=bar_t,
            TB_MIN=TB_MIN,
            CHARGE_RATE_KW=CHARGE_RATE_KW,
            max_trip2trip=57,
            max_trip2charge=61,
            max_charge2trip=220,
            min_soc_fraction=0.0,
            recharge_to_fraction=1.0,
            max_daily_recharges=MAX_DAILY_RECHARGES,
        )
        R_truck.extend(greedy_routes)

        covered = {
            n
            for route in R_truck
            for n in route.get("route", [])
            if isinstance(n, int)
        }
        print(
            f"[GREEDY] Seeded R_truck with {len(greedy_routes)} routes "
            f"covering {len(covered)}/{len(T)} unique trips."
        )
        seed_route_validation = "constructed_under_current_time_soc_rules"
    else:
        print("[NO_CHEAT] R_truck initialized empty.")

    seed_route_count = len(R_truck)

# Statistics trackers for CURRENT run
master_times = []
pricing_times = []

# ------------------------------ Build & solve master once ------------------------------
#%%
rmp, a, trip_cov = init_master(
    R_truck=R_truck,
    T=T,
    charging_cost_data=hourly_prices,
    bus_cost=bus_cost,
    binary=False,
    station_hourly_prices=station_hourly_prices,
)


# LP params for the RMP (per-iteration)
rmp.Params.Threads = THREADS
rmp.Params.NodefileStart = NODEFILE_START
rmp.Params.NodefileDir = _detect_tmp()
rmp.Params.Method = 1
rmp.Params.TimeLimit = MASTER_TIME_LIMIT
# The first CG iteration performs and times the first solve.  Optimizing here
# would solve the same unchanged RMP twice and evade the active-time budget.


# ------------------------------ DIAGNOSTICS: list missing depot arcs ------------------------------
diag_dir = RUN_DIR / "diagnostics"
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

iteration = start_iteration

completed_active_hours = (prev_cum_master + prev_cum_pricing) / 3600.0
skip_cg_loop = bool(
    ACTIVE_TIME_LIMIT_HOURS
    and completed_active_hours >= ACTIVE_TIME_LIMIT_HOURS
)
if skip_cg_loop:
    print(f"[RESUME] Active-time limit {ACTIVE_TIME_LIMIT_HOURS:g}h already reached; "
          "skipping column generation and proceeding to final MIP.")

if len(R_truck) == 0:
    print("[WARN] No initial seed routes; master may be infeasible if some trips lack any coverable pattern.")


def _route_key(route):
    return route_column_key(route)

# Determine CSV path
if is_resuming and resumed_stats_csv and Path(resumed_stats_csv).exists():
    resumed_stats_path = Path(resumed_stats_csv)
    header = resumed_stats_path.read_text().splitlines()[0] if resumed_stats_path.stat().st_size else ""
    if "Pricing_Labels_Used" in header and "Highest_Tier_Reached" in header:
        stats_csv_path = resumed_stats_path
        print(f"[RESUME] Appending stats to original CSV: {stats_csv_path}")
    else:
        stats_csv_path = RUN_DIR / f"pricing_{bus_label}_{K_BEST}cols_instrumented.csv"
        print(f"[RESUME] Existing stats CSV has old schema; writing new stats CSV: {stats_csv_path}")
else:
    stats_csv_path = RUN_DIR / f"pricing_{bus_label}_{K_BEST}cols.csv"


def _write_iteration_checkpoint(iteration_number, reason):
    iteration_state = {
        "iteration": iteration_number,
        "cum_master_time": sum(master_times) + prev_cum_master,
        "cum_pricing_time": sum(pricing_times) + prev_cum_pricing,
        "stats_csv_path": str(stats_csv_path),
        "recent_improvements": list(recent_improvements),
        "last_master_obj": last_master_obj,
        "resume_count": (resume_count + 1) if is_resuming else 0,
        "run_dir": str(RUN_DIR),
        "routes": R_truck,
        "seed_route_count": seed_route_count,
        "dp_columns_generated": dp_columns_generated,
        "seed_route_validation": seed_route_validation,
        "csv_name": csv_name,
        "trip_ids": T,
        "instance_sha256": instance_sha256,
        "price_sha256": price_sha256,
        "mode": run_mode,
        "battery_kwh": G_PARAM,
        "git": git_state,
        "runtime_versions": runtime_versions,
        "resume_history": resume_history,
        "run_arguments": vars(args),
        "price_tag": price_tag,
        "prices_csv": str(prices_csv),
        "termination_reason": reason,
        "milestones_passed": milestones_passed,
    }
    ckpt_path = RUN_DIR / f"ckpt_latest_{bus_label}_g{G_PARAM}_{K_BEST}cols.json"
    tmp_path = ckpt_path.with_suffix(".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(iteration_state, handle)
    os.replace(tmp_path, ckpt_path)
    return ckpt_path


granularity = 10

# Build the DP pricer ONCE: the DAG topology is fixed across CG iterations,
# only the duals change. time_limit / max_labels are passed per call.
dp_price = make_dp_pricer(
    T=T, S_use=S_price, DEPOT=DEPOT,
    tau=tau, d=d, st=st, et=et, sl=sl, el=el, epsilon=epsilon,
    tau_min=tau_min, st_min=st_min, et_min=et_min,
    G=G, TB_MIN=TB_MIN, bar_t=bar_t,
    bus_cost=bus_cost,
    charge_rate_kw=CHARGE_RATE_KW,
    hourly_prices=hourly_prices,
    charge_cost_premium=charge_cost_premium,
    travel_cost_factor=0,
    RC_EPSILON=RC_EPSILON,
    K_BEST=K_BEST,
    MAX_LABELS_PER_NODE=int(ESCALATION_SCHEDULE[0][0]),
    soc_charge_levels=[G * i * (1 / granularity) for i in range(1, 1 + granularity)],
    MIN_TRIPS_PER_ROUTE=MIN_TRIPS_PER_ROUTE,
    MAX_DAILY_RECHARGES=MAX_DAILY_RECHARGES,
    max_trip2trip=57,
    max_trip2charge=61,
    max_charge2trip=220,
    station_hourly_prices=station_hourly_prices,
    charge_start_cost=CHARGE_START_COST,
)
#%%
# max_iter += 100
#%%
while not skip_cg_loop and iteration < MAX_CG_ITERS:
    iteration += 1
    termination_reason = "running"
    print(f"\n--- Iteration {iteration} ---")

    # 1) SOLVE MASTER
    t0 = time.time()
    rmp.Params.TimeLimit = MASTER_TIME_LIMIT
    rmp.optimize()

    master_iter_time = time.time() - t0
    master_times.append(master_iter_time)

    if rmp.Status != GRB.OPTIMAL:
        # Column generation requires valid optimal LP duals. Pricing an
        # incumbent/basis from a time-limited master can create a false
        # reduced-cost-optimal stop, so fail loudly and preserve the previous
        # completed checkpoint instead.
        raise RuntimeError(
            f"Master LP status {rmp.Status} is not OPTIMAL "
            f"(TimeLimit={MASTER_TIME_LIMIT:g}s). Rerun from the last checkpoint "
            "with a larger --master_time_limit."
        )

    print(f" Master obj: {rmp.ObjVal:.2f}")

    current_obj = rmp.ObjVal

    if args.target_master_obj is not None:
        print(f" Experimental objective stop: {args.target_master_obj:.2f}")

    if args.target_master_obj is not None and current_obj <= args.target_master_obj:
        termination_reason = "target_obj_reached"
        last_master_obj = current_obj
        _write_iteration_checkpoint(iteration, termination_reason)
        break

    # compute improvement safely
    if last_master_obj is None:
        improvement = 0.0
        is_first_iter = True
    else:
        improvement = last_master_obj - current_obj
        is_first_iter = False

    print(f" Master obj: {current_obj:.2f} (Impv: {improvement:.4f})")

    # --- Rolling-window stagnation detection ---
    if not is_first_iter:
        recent_improvements.append(improvement)

    if len(recent_improvements) >= args.stagnation_window:
        window_sum = sum(recent_improvements)
        if window_sum <= args.improvement_bound:
            print(f"[STOP] Stagnation: last {args.stagnation_window} iters summed "
                  f"improvement = {window_sum:.4f} <= {args.improvement_bound}")
            print(f"       Master obj settled at {current_obj:.2f}")
            termination_reason = "stagnation_rolling_window"
            last_master_obj = current_obj
            _write_iteration_checkpoint(iteration, termination_reason)
            break

    last_master_obj = current_obj

    # Extract Duals
    alpha, beta_dual, gamma_dual = extract_duals(rmp)


    # 2) SOLVE PRICING (Dynamic Programming)
    new_trucks = []
    best_rc_iter = float("inf")
    timed_out_any = False
    deepest_tier_timed_out = False
    current_max_labels_used = 0
    highest_tier_reached = 0
    tier_stats = []
    active_budget_exhausted = False
    milestone_boundary_reached = False
    milestone_boundary_hour = None

    # For deduplication against what's already in the Master Problem
    seen_keys_existing = {_route_key(r) for r in R_truck}
    t0_pricing_total = time.time()

    for tier_idx, (current_max_labels, current_time_limit) in enumerate(ESCALATION_SCHEDULE, start=1):
        elapsed_pricing = time.time() - t0_pricing_total
        remaining_wall = PRICING_WALL_PER_ITER - elapsed_pricing
        if remaining_wall <= 30:
            print(f"   [WALL] Pricing wall budget exhausted ({elapsed_pricing:.0f}s). Stopping escalation.")
            break

        active_used = (
            prev_cum_master + prev_cum_pricing
            + sum(master_times) + sum(pricing_times)
            + elapsed_pricing
        )

        # Finish the current iteration at each requested milestone instead of
        # letting a long pricing tier overshoot it by hundreds of seconds.
        crossed_unsaved = [
            milestone for milestone in TARGET_MILESTONES_HOURS
            if milestone not in milestones_passed
            and active_used >= milestone * 3600.0
        ]
        if crossed_unsaved:
            milestone_boundary_reached = True
            milestone_boundary_hour = min(crossed_unsaved)
            print(f"   [MILESTONE] Active time crossed {milestone_boundary_hour:g}h; "
                  "checkpointing before another pricing tier.")
            break

        pending_milestones = [
            milestone for milestone in TARGET_MILESTONES_HOURS
            if milestone not in milestones_passed
            and milestone * 3600.0 > active_used
        ]
        next_milestone = min(pending_milestones) if pending_milestones else None
        remaining_milestone = (
            next_milestone * 3600.0 - active_used
            if next_milestone is not None
            else float("inf")
        )

        remaining_active = float("inf")
        if ACTIVE_TIME_LIMIT_HOURS:
            remaining_active = ACTIVE_TIME_LIMIT_HOURS * 3600.0 - active_used
            if remaining_active <= 1:
                print("   [ACTIVE LIMIT] No active-compute budget remains for another pricing tier.")
                active_budget_exhausted = True
                break

        # Do not request more DP time than remains in either budget.
        time_limits = [current_time_limit, int(remaining_wall)]
        if ACTIVE_TIME_LIMIT_HOURS:
            time_limits.append(max(1, int(remaining_active)))
        if next_milestone is not None:
            time_limits.append(max(1, int(remaining_milestone)))
        effective_time_limit = min(time_limits)
        current_max_labels_used = current_max_labels
        highest_tier_reached = tier_idx
        print(f"   > DP pricing tier {tier_idx}: "
              f"MAX_LABELS={current_max_labels}, TIME_LIMIT={effective_time_limit}s...")

        tier_t0 = time.time()
        raw_new_trucks, best_rc_iter, tier_timed_out = dp_price(
            alpha, beta_dual, gamma_dual,
            time_limit=effective_time_limit,
            max_labels=int(current_max_labels),
        )
        tier_time = time.time() - tier_t0
        timed_out_any = timed_out_any or tier_timed_out
        deepest_tier_timed_out = tier_timed_out

        seen_new = set()
        accepted_this_tier = 0
        for t_route in raw_new_trucks:
            k = _route_key(t_route)
            if (k not in seen_keys_existing) and (k not in seen_new):
                new_trucks.append(t_route)
                seen_new.add(k)
                accepted_this_tier += 1
            if len(new_trucks) >= K_BEST:
                break

        tier_stats.append({
            "tier": tier_idx,
            "max_labels": current_max_labels,
            "time_limit_s": effective_time_limit,
            "time_s": tier_time,
            "hit_timelimit": tier_timed_out,
            "returned": len(raw_new_trucks),
            "accepted": accepted_this_tier,
            "found_zero": accepted_this_tier == 0,
        })

        active_after_tier = active_used + tier_time
        if (
            next_milestone is not None
            and active_after_tier >= next_milestone * 3600.0 - 1.0
        ):
            milestone_boundary_reached = True
            milestone_boundary_hour = next_milestone

        if new_trucks:
            print(f"   [SUCCESS tier {tier_idx}] DP accepted {accepted_this_tier} new cols "
                  f"from {len(raw_new_trucks)} returned routes "
                  f"(best_rc={best_rc_iter:.1f}, timed_out={tier_timed_out}) "
                  f"after {time.time()-t0_pricing_total:.0f}s of pricing")
            break
        elif milestone_boundary_reached:
            print(f"   [MILESTONE] Pricing reached the {milestone_boundary_hour:g}h boundary; "
                  "saving the pool before escalation.")
            break
        else:
            print(f"   [FAILED tier {tier_idx}] 0 accepted cols "
                  f"from {len(raw_new_trucks)} returned routes, timed_out={tier_timed_out}. "
                  f"Escalating...")

    pricing_dur_total = time.time() - t0_pricing_total
    pricing_times.append(pricing_dur_total)  # <--- NEW: Track cumulative pricing





    tier_map = {ts["tier"]: ts for ts in tier_stats}

    artificial_values = [
        float(q_var.X)
        for i in T
        if (q_var := rmp.getVarByName(f"q_{i}")) is not None and q_var.X > 1e-6
    ]
    lp_route_weight = sum(float(var.X) for var in a.values() if var.X > 1e-9)

    # --- Collect Metrics ---
    current_stat = {
        "Iteration": iteration,
        "Master_Obj": current_obj,
        "Master_Improvement": improvement,
        "Master_Time_s": master_iter_time,
        "LP_Route_Weight": lp_route_weight,
        "Artificial_Trips": len(artificial_values),
        "Artificial_Total": sum(artificial_values),
        "Pricing_Time_s": pricing_dur_total,
        "Cumulative_Master_Time_s": prev_cum_master + sum(master_times),
        "Cumulative_Pricing_Time_s": prev_cum_pricing + sum(pricing_times),
        "Cols_Added": len(new_trucks),
        "Best_RC": best_rc_iter,
        "Timed_Out": timed_out_any,
        "Deepest_Tier_Hit_Timelimit": deepest_tier_timed_out,
        "Pricing_Labels_Used": current_max_labels_used,
        "Highest_Tier_Reached": highest_tier_reached,
        "Tier1_Time_s": tier_map.get(1, {}).get("time_s"),
        "Tier1_Hit_Timelimit": tier_map.get(1, {}).get("hit_timelimit"),
        "Tier1_Returned": tier_map.get(1, {}).get("returned"),
        "Tier1_Accepted": tier_map.get(1, {}).get("accepted"),
        "Tier1_Found_Zero": tier_map.get(1, {}).get("found_zero"),
        "Tier2_Time_s": tier_map.get(2, {}).get("time_s"),
        "Tier2_Hit_Timelimit": tier_map.get(2, {}).get("hit_timelimit"),
        "Tier2_Returned": tier_map.get(2, {}).get("returned"),
        "Tier2_Accepted": tier_map.get(2, {}).get("accepted"),
        "Tier2_Found_Zero": tier_map.get(2, {}).get("found_zero"),
        "Tier3_Time_s": tier_map.get(3, {}).get("time_s"),
        "Tier3_Hit_Timelimit": tier_map.get(3, {}).get("hit_timelimit"),
        "Tier3_Returned": tier_map.get(3, {}).get("returned"),
        "Tier3_Accepted": tier_map.get(3, {}).get("accepted"),
        "Tier3_Found_Zero": tier_map.get(3, {}).get("found_zero"),
        "Recent_Window_Sum": (sum(recent_improvements)
                              if len(recent_improvements) >= args.stagnation_window
                              else None),
        "Total_Runtime_s": time.time() - stopwatch_start,
    }

    pd.DataFrame([current_stat]).to_csv(
        stats_csv_path,
        mode='a',
        header=not stats_csv_path.exists(),
        index=False
    )

    # 3) ADD COLUMNS TO MASTER
    for route in new_trucks:
        R_truck.append(route)

        # Hour-split charging cost — must match the DP pricer's rc computation
        # (same function build_master uses, so resumes recompute identical costs)
        cost = calculate_truck_route_cost_accurate(
            route, bus_cost, hourly_prices,
            charge_rate_kw=CHARGE_RATE_KW,
            station_hourly_prices=station_hourly_prices,
            charge_start_cost=CHARGE_START_COST,
        )

        col = Column()
        for node in route["route"]:
            if isinstance(node, int):
                col.addTerms(1.0, trip_cov[node])

        idx = len(R_truck) - 1
        # No ub in the LP: a bounded column at ub can price negative via its
        # bound dual, making the DP "rediscover" it forever (see master.py)
        a[idx] = rmp.addVar(obj=cost, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, column=col, name=f"a[{idx}]")
    rmp.update()
    dp_columns_generated += len(new_trucks)


    stop_after_iteration = False
    if not new_trucks:
        if milestone_boundary_reached:
             print("   [CONTINUE] Milestone checkpoint boundary reached; pricing will resume next iteration.")
             termination_reason = "running"
        elif active_budget_exhausted:
             print("   [STOP] Active-compute budget exhausted.")
             termination_reason = "active_time_limit_reached"
        elif (best_rc_iter >= -RC_EPSILON) and not deepest_tier_timed_out:
             # NOTE: optimal only within the RESTRICTED pricing graph (57/61/220
             # min gap limits, min-trips filter, discrete SOC grid, label cap) —
             # not an LP-optimality certificate for the full EVSP.
             print("   [RC-OPT / STOP] Deepest pricing tier exhausted with no negative RC "
                   "columns (restricted pricing graph).")
             termination_reason = "rc_optimal_restricted"
        elif deepest_tier_timed_out:
             print("   [STOP] Deepest pricing tier hit its time limit with no accepted new columns.")
             termination_reason = "pricing_timed_out_no_new_columns"
        else:
             print("   [STOP] Pricing returned negative routes, but none were new to the master.")
             termination_reason = "no_new_columns"
        if not milestone_boundary_reached:
            stop_after_iteration = True

    # --- MILESTONE SNAPSHOTS ---
    # Active compute time is cumulative master + pricing time across resumes.
    # Snapshots are taken after the current iteration's columns have been added.
    total_master_time_s = sum(master_times) + prev_cum_master
    total_pricing_time_s = sum(pricing_times) + prev_cum_pricing
    total_active_time_s = total_master_time_s + total_pricing_time_s
    total_active_hours = total_active_time_s / 3600.0

    if ACTIVE_TIME_LIMIT_HOURS and total_active_hours >= ACTIVE_TIME_LIMIT_HOURS:
        print(f"\n[STOP] Reached {ACTIVE_TIME_LIMIT_HOURS:g} hour active-compute limit. "
              "Halting column generation.")
        termination_reason = "active_time_limit_reached"
        stop_after_iteration = True

    for milestone in TARGET_MILESTONES_HOURS:
        if total_active_hours >= milestone and milestone not in milestones_passed:
            print(f"\n[MILESTONE] Crossed {milestone:g} hours of active compute time.")
            snapshot_path = RUN_DIR / f"routes_{_hours_tag(milestone)}_snapshot_{bus_label}.json"
            tmp_snapshot_path = snapshot_path.with_suffix(".tmp")
            milestones_passed.append(milestone)

            with open(tmp_snapshot_path, "w") as f:
                json.dump({
                    "iteration": iteration,
                    "milestone_hours": milestone,
                    "active_time_hours": total_active_hours,
                    "active_time_s": total_active_time_s,
                    "cumulative_master_time_s": total_master_time_s,
                    "cumulative_pricing_time_s": total_pricing_time_s,
                    "master_obj": current_obj,
                    "best_rc": best_rc_iter,
                    "cols_added_this_iteration": len(new_trucks),
                    "num_routes": len(R_truck),
                    "run_dir": str(RUN_DIR),
                    "seed_route_count": seed_route_count,
                    "dp_columns_generated": dp_columns_generated,
                    "seed_route_validation": seed_route_validation,
                    "csv_name": csv_name,
                    "bus_label": bus_label,
                    "price_tag": price_tag,
                    "prices_csv": str(prices_csv),
                    "price_sha256": price_sha256,
                    "instance_sha256": instance_sha256,
                    "trip_ids": T,
                    "mode": run_mode,
                    "battery_kwh": G_PARAM,
                    "git": git_state,
                    "runtime_versions": runtime_versions,
                    "resume_history": resume_history,
                    "run_arguments": vars(args),
                    "termination_reason": termination_reason,
                    "milestones_passed": milestones_passed,
                    "stats_csv_path": str(stats_csv_path),
                    "routes": R_truck,
                }, f)
            os.replace(tmp_snapshot_path, snapshot_path)

            print(f"            Saved {len(R_truck)} routes to {snapshot_path.name}")

    # --- SAVE STATE (Every Iteration) ---
    _write_iteration_checkpoint(iteration, termination_reason)


    # 4) CHECK TERMINATION
    if stop_after_iteration:
        break

if iteration >= MAX_CG_ITERS and termination_reason == "running":
    termination_reason = "max_iterations_reached"

#%%
print("\n=== COLUMN GENERATION TIME SUMMARY ===")
print(f"Total Iterations:       {iteration}")
print(f"Cumulative Master Time: {sum(master_times):.1f}s")
print(f"Cumulative Pricing Time:{sum(pricing_times):.1f}s")
print(f"Total Loop Runtime:     {time.time() - stopwatch_start:.1f}s")
print("========================================\n")
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

if args.skip_final_mip:
    stopwatch_end = time.time()
    elapsed = stopwatch_end - stopwatch_start
    final_routes_path = RUN_DIR / f"routes_colgen_final_{bus_label}.json"
    tmp_final_routes_path = final_routes_path.with_suffix(".tmp")

    with open(tmp_final_routes_path, "w") as f_out:
        json.dump({
            "iteration": iteration,
            "active_time_s": sum(master_times) + prev_cum_master + sum(pricing_times) + prev_cum_pricing,
            "cumulative_master_time_s": sum(master_times) + prev_cum_master,
            "cumulative_pricing_time_s": sum(pricing_times) + prev_cum_pricing,
            "num_routes": len(R_truck),
            "run_dir": str(RUN_DIR),
            "seed_route_count": seed_route_count,
            "dp_columns_generated": dp_columns_generated,
            "seed_route_validation": seed_route_validation,
            "csv_name": csv_name,
            "bus_label": bus_label,
            "price_tag": price_tag,
            "prices_csv": str(prices_csv),
            "price_sha256": price_sha256,
            "instance_sha256": instance_sha256,
            "trip_ids": T,
            "mode": run_mode,
            "battery_kwh": G_PARAM,
            "git": git_state,
            "runtime_versions": runtime_versions,
            "resume_history": resume_history,
            "run_arguments": vars(args),
            "termination_reason": termination_reason,
            "milestones_passed": milestones_passed,
            "routes": R_truck,
        }, f_out)
    os.replace(tmp_final_routes_path, final_routes_path)

    result = {
        "LP_Obj": rmp.ObjVal,
        "MIP_Obj": None,
        "Skipped_Final_MIP": True,
        "Total_Time_s": elapsed,
        "CG_Iterations": iteration,
        "Columns_In_Pool": len(R_truck),
        "Seed_Routes": seed_route_count,
        "DP_Columns_Generated": dp_columns_generated,
        "Seed_Route_Validation": seed_route_validation,
        "Instance_CSV": csv_name,
        "Instance_SHA256": instance_sha256,
        "Mode": run_mode,
        "Git": git_state,
        "Runtime_Versions": runtime_versions,
        "Run_Arguments": vars(args),
        "Artificial_Trips_LP": len(uncovered_trips),
        "Price_Tag": price_tag,
        "Prices_CSV": str(prices_csv),
        "Milestones_Hours": TARGET_MILESTONES_HOURS,
        "Active_Time_s": (
            prev_cum_master + sum(master_times)
            + prev_cum_pricing + sum(pricing_times)
        ),
        "Termination_Reason": termination_reason,
        "Final_Routes_JSON": str(final_routes_path),
    }

    summary_path = RUN_DIR / f"colgen_summary_{bus_label}.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[CG-ONLY] Saved final column pool: {final_routes_path}")
    print(f"[CG-ONLY] Saved summary: {summary_path}")
    print("[CG-ONLY] --skip_final_mip active. Exiting before final MIP.")
    sys.exit(0)

#%%
# ------------------------------ Final solve (LP then MIP warm-start) ------------------------------

rmp_lp, a_lp = solve_master(
    R_truck=R_truck,
    T=T,
    charging_cost_data=hourly_prices,
    bus_cost=bus_cost,
    binary=False,
    station_hourly_prices=station_hourly_prices,
)
if rmp_lp.SolCount == 0:
    raise RuntimeError(f"Final LP produced no solution (status={rmp_lp.Status})")
final_LP_obj = rmp_lp.ObjVal
#%%
rmp_final, a_final, trip_cov_final = build_master(
    R_truck=R_truck,
    T=T,
    charging_cost_data=hourly_prices,
    bus_cost=bus_cost,
    binary=True,
    station_hourly_prices=station_hourly_prices,
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
        # LP vars are unbounded above; clamp so the binary warm start is valid
        var.start = min(a_lp[idx].X, 1.0)


# ==========================================
# Config W: Winning reference + binary fix + modest heuristic boost
# Matches the 2h reference that found the best incumbent on this pool,
# fixes the binary bug, and gives RINS/ImproveStart extra traction.
# ==========================================
for v in rmp_final.getVars():
    if v.VarName.startswith("a["):
        v.VType = gp.GRB.BINARY
rmp_final.update()

rmp_final.setParam('TimeLimit', args.final_mip_timelimit)
# rmp_final.setParam('Threads', 4)              # match reference; run ONLY this one
# rmp_final.setParam('MIPFocus', 1)             # incumbent-focused, not bound
# rmp_final.setParam('Heuristics', 0.5)         # match reference
# rmp_final.setParam('Cuts', 1)                 # default-level, not aggressive
# rmp_final.setParam('RINS', 15)                # neighborhood search every 15 nodes
# rmp_final.setParam('ImproveStartTime', 1200)  # give it 20 min to explore, then go all-in on incumbent
# rmp_final.setParam('ImproveStartGap', 0.30)

rmp_final.optimize()
has_final_solution = rmp_final.SolCount > 0
final_MIP_obj = float(rmp_final.ObjVal) if has_final_solution else None

print("\n=== Selected truck routes ===")
used_routes = []
if has_final_solution:
    for r in range(len(R_truck)):
        if r in a_final and a_final[r].X > 0.5:
            used_routes.append(r)
            print(f"Route {r}: a[{r}]={a_final[r].X:.0f}  -> {R_truck[r]}")
else:
    print("No final-MIP incumbent was found.")

print("\n Master LP obj:", final_LP_obj)
print(" Master MIP obj:", final_MIP_obj if has_final_solution else "no incumbent")
print(f" Buses used: {len(used_routes)}" if has_final_solution else " Buses used: unavailable")

try:
    rmp_final.write(str(RUN_DIR / f"solution_{RUN_ID}.sol"))
except Exception:
    pass

dummy_used = [r for r in used_routes if R_truck[r].get("dummy", False)]
real_used  = [r for r in used_routes if not R_truck[r].get("dummy", False)]
if has_final_solution:
    print(f" Dummy routes used: {len(dummy_used)} / {len(used_routes)}")
    print(f" Real routes used : {len(real_used)} / {len(used_routes)}")

# The q_i artificial variables are NOT in R_truck: audit them directly, else a
# "0 dummy routes" report can hide trips covered only by BIG-M slacks.
q_used = [i for i in T
          if has_final_solution
          and (qv := rmp_final.getVarByName(f"q_{i}")) is not None and qv.X > 0.5]
print(f" Artificial q_i used in final MIP: {len(q_used)}"
      + (f" -> trips {q_used}" if q_used else ""))

# Set covering allows over-coverage; count trips driven 'empty' at least once
_cov = collections.Counter(
    n for r in used_routes for n in R_truck[r].get("route", []) if isinstance(n, int))
_over = {i: c for i, c in _cov.items() if c > 1}
print(f" Trips covered more than once: {len(_over)}")


# arc stats
# print(f"[arc stats] direct={direct_hits} fallback(ref->ref)={fallback_hits} mixed={mixed_hits} misses={misses}")



stopwatch_end = time.time()
elapsed = stopwatch_end - stopwatch_start
print(f"\n=== CG Loop Completed in {elapsed:.1f} seconds ===")


#%%


result = {
        "LP_Obj": final_LP_obj,
        "MIP_Obj": final_MIP_obj,
        "Total_Time_s": elapsed,
        "CG_Iterations": iteration,
        "Columns_In_Pool": len(R_truck),
        "Seed_Routes": seed_route_count,
        "DP_Columns_Generated": dp_columns_generated,
        "Seed_Route_Validation": seed_route_validation,
        "Instance_CSV": csv_name,
        "Instance_SHA256": instance_sha256,
        "Mode": run_mode,
        "Git": git_state,
        "Runtime_Versions": runtime_versions,
        "Run_Arguments": vars(args),
        "Artificial_Trips_MIP": len(q_used) if has_final_solution else None,
        "Overcovered_Trips_MIP": len(_over) if has_final_solution else None,
        "Price_Tag": price_tag,
        "Prices_CSV": str(prices_csv),
        "Milestones_Hours": TARGET_MILESTONES_HOURS,
        "Final_MIP_Timelimit_s": args.final_mip_timelimit,
        "Skipped_Final_MIP": False,
        "Active_Time_s": (
            prev_cum_master + sum(master_times)
            + prev_cum_pricing + sum(pricing_times)
        ),
        "Termination_Reason": termination_reason,
        "MIP_Status": int(rmp_final.Status),
        "MIP_Has_Solution": has_final_solution,
        "Buses_Used": len(used_routes) if has_final_solution else None,
    }

# Save run-local metadata; array tasks must never overwrite shared files.
summary_path = RUN_DIR / f"colgen_and_mip_summary_{bus_label}.json"
with open(summary_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n[META] Saved summary to {summary_path}. Exiting script gracefully.")

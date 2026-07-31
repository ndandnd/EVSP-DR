"""
run_final_mip.py
Standalone script: loads a CG checkpoint JSON and solves the final MIP.
Usage:
    python run_final_mip.py --ckpt path/to/ckpt_latest_*.json --timelimit 86400
"""

import os
import sys
import json
import time
import math
import argparse
import datetime
from pathlib import Path

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from config import (
    BUS_COST_KX,
    TIMEBLOCKS_PER_HOUR,
    THREADS, NODEFILE_START, NODEFILE_DIR,
)
from master import build_master, solve_master
from utils_v2 import calculate_truck_route_cost

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="EVSP Final MIP from checkpoint")
parser.add_argument("--ckpt",      type=str, required=True,  help="Path to checkpoint JSON")
parser.add_argument("--timelimit", type=int, default=86400,  help="MIP time limit in seconds (default 24h)")
parser.add_argument("--mipgap",    type=float, default=1e-4, help="MIP optimality gap (default 0.01%%)")
parser.add_argument("--threads",   type=int, default=None,   help="Gurobi threads (default: from config)")
parser.add_argument(
    "--prices_csv",
    type=str,
    default=None,
    help="Optional spatiotemporal price CSV. Defaults to prices_csv stored in checkpoint, then data/spatiotemporal_prices.csv.",
)
args = parser.parse_args()

stopwatch = time.time()
RUN_ID    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Load checkpoint ────────────────────────────────────────────────────────────
ckpt_path = Path(args.ckpt)
if not ckpt_path.exists():
    print(f"[ERROR] Checkpoint not found: {ckpt_path}")
    sys.exit(1)

print(f"[INFO] Loading checkpoint: {ckpt_path}")
with open(ckpt_path, "r") as f:
    data = json.load(f)

R_truck        = data["routes"]
iteration_done = data.get("iteration", "?")
run_dir        = Path(data.get("run_dir", ckpt_path.parent))
stats_csv_path = Path(data.get("stats_csv_path", run_dir / "pricing_stats.csv"))

print(f"[INFO] Loaded {len(R_truck)} routes from iteration {iteration_done}")
print(f"[INFO] Results will be saved to: {run_dir}")
if "run_dir" not in data:
    print("[INFO] Snapshot JSON has no run_dir field; using snapshot parent directory.")

# ── Reconstruct T from routes ─────────────────────────────────────────────────
# T must exactly match what the CG used.  Every integer node across all routes
# is a trip; collect them in sorted order.
T = sorted({node for r in R_truck for node in r["route"] if isinstance(node, int)})
print(f"[INFO] Reconstructed T: {len(T)} trips  (range {min(T)}..{max(T)})")

# ── Price curve ───────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
if args.prices_csv:
    prices_csv = Path(args.prices_csv)
    if not prices_csv.is_absolute():
        prices_csv = DATA_DIR / prices_csv
elif data.get("prices_csv"):
    prices_csv = Path(data["prices_csv"])
else:
    prices_csv = DATA_DIR / "spatiotemporal_prices.csv"

print(f"[INFO] Using price CSV: {prices_csv}")
st_df      = pd.read_csv(prices_csv)
station_hourly_prices = {
    station: grp.set_index("time_block")["cost"].to_dict()
    for station, grp in st_df.groupby("station")
}
hourly_prices = station_hourly_prices["PARX"]   # flat fallback
bus_cost      = BUS_COST_KX

# ── Final LP (warm-start for MIP) ─────────────────────────────────────────────
print("\n[STEP 1] Solving final LP relaxation for MIP warm-start ...")
rmp_lp, a_lp = solve_master(
    R_truck=R_truck,
    T=T,
    charging_cost_data=hourly_prices,
    bus_cost=bus_cost,
    binary=False,
    station_hourly_prices=station_hourly_prices,
)
final_LP_obj = rmp_lp.ObjVal
print(f"[INFO]  LP obj: {final_LP_obj:,.2f}")

# ── Diagnostics: check for dummy variable usage ────────────────────────────────
uncovered = [i for i in T
             if (q := rmp_lp.getVarByName(f"q_{i}")) is not None and q.X > 0.01]
if uncovered:
    print(f"[WARN]  {len(uncovered)} trips still covered by dummy vars: {uncovered[:20]}")
else:
    print("[INFO]  All trips covered by real routes in LP. Good to proceed.")

# ── Build binary MIP ──────────────────────────────────────────────────────────
print("\n[STEP 2] Building binary MIP ...")
rmp_final, a_final, trip_cov_final = build_master(
    R_truck=R_truck,
    T=T,
    charging_cost_data=hourly_prices,
    bus_cost=bus_cost,
    binary=True,
    station_hourly_prices=station_hourly_prices,
)

# Force all route variables to binary (belt-and-suspenders)
for v in rmp_final.getVars():
    if v.VarName.startswith("a["):
        v.VType = gp.GRB.BINARY
rmp_final.update()

# Warm-start from LP solution
for idx, var in a_final.items():
    if idx in a_lp:
        var.start = a_lp[idx].X

# ── Gurobi params ─────────────────────────────────────────────────────────────
n_threads = args.threads if args.threads else THREADS
rmp_final.Params.Threads       = n_threads
rmp_final.Params.NodefileStart = NODEFILE_START
rmp_final.Params.NodefileDir   = NODEFILE_DIR or "/tmp"
rmp_final.Params.TimeLimit     = args.timelimit
rmp_final.Params.MIPGap        = args.mipgap
rmp_final.Params.LogFile       = str(run_dir / f"final_mip_{RUN_ID}.log")

print(f"[INFO]  Time limit : {args.timelimit}s  ({args.timelimit/3600:.1f}h)")
print(f"[INFO]  MIP gap    : {args.mipgap*100:.4f}%")
print(f"[INFO]  Threads    : {n_threads}")
print(f"[INFO]  Log file   : {rmp_final.Params.LogFile}")

# ── Solve ─────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Optimizing ...")
rmp_final.optimize()

# ── Results ───────────────────────────────────────────────────────────────────
status      = rmp_final.Status
gap         = rmp_final.MIPGap
best_bound  = rmp_final.ObjBound

try:
    final_MIP_obj = rmp_final.ObjVal
except:
    final_MIP_obj = float("inf")

print("\n=== FINAL MIP RESULTS ===")
print(f"Status       : {status}  ({'OPTIMAL' if status == GRB.OPTIMAL else 'TIME_LIMIT/OTHER'})")
print(f"LP  obj      : {final_LP_obj:,.2f}")
print(f"MIP obj      : {final_MIP_obj:,.2f}")
print(f"Best bound   : {best_bound:,.2f}")
print(f"MIP gap      : {gap*100:.4f}%")
print(f"Proven opt   : {status == GRB.OPTIMAL}")

used_routes = []
if final_MIP_obj < float("inf"):
    print("\n=== Selected routes ===")
    for r in range(len(R_truck)):
        if r in a_final and a_final[r].X > 0.5:
            used_routes.append(r)
            print(f"Route {r}: a[{r}]=1  -> {R_truck[r]}")

    dummy_used = [r for r in used_routes if R_truck[r].get("dummy", False)]
    real_used  = [r for r in used_routes if not R_truck[r].get("dummy", False)]
    print(f"\nBuses used       : {len(used_routes)}")
    print(f"Real routes      : {len(real_used)}")
    print(f"Dummy routes     : {len(dummy_used)}")

    # Save .sol file
    sol_path = run_dir / f"final_mip_{RUN_ID}.sol"
    try:
        rmp_final.write(str(sol_path))
        print(f"Solution written : {sol_path}")
    except Exception as e:
        print(f"[WARN] Could not write .sol: {e}")

# ── Save summary JSON ─────────────────────────────────────────────────────────
elapsed = time.time() - stopwatch
summary = {
    "ckpt_source"      : str(ckpt_path),
    "CG_iterations"    : iteration_done,
    "columns_in_pool"  : len(R_truck),
    "T_size"           : len(T),
    "LP_obj"           : final_LP_obj,
    "MIP_obj"          : final_MIP_obj,
    "MIP_bound"        : best_bound,
    "MIP_gap_pct"      : gap * 100,
    "proven_optimal"   : status == GRB.OPTIMAL,
    "buses_used"       : len(used_routes),
    "real_routes_used" : len(real_used) if final_MIP_obj < float("inf") else None,
    "dummy_routes_used": len(dummy_used) if final_MIP_obj < float("inf") else None,
    "mip_timelimit_s"  : args.timelimit,
    "mip_runtime_s"    : elapsed,
    "run_id"           : RUN_ID,
    "prices_csv"       : str(prices_csv),
    "price_tag"        : data.get("price_tag"),
}
summary_path = run_dir / f"final_mip_summary_{RUN_ID}.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved    : {summary_path}")
print(f"Total runtime    : {elapsed:.1f}s  ({elapsed/3600:.2f}h)")
print("=========================")

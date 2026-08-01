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
import argparse
import datetime
import hashlib
import subprocess
from pathlib import Path

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from config import (
    BUS_COST_KX,
    CHARGING_STATIONS,
    THREADS, NODEFILE_START, NODEFILE_DIR,
)
from master import build_master, solve_master
from utils_v2 import load_station_hourly_prices

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
parser.add_argument(
    "--allow_unsafe_checkpoint",
    action="store_true",
    help="Allow missing/mismatched checkpoint provenance (unsafe for benchmarking).",
)
parser.add_argument(
    "--allow_restricted_pool_reprice",
    action="store_true",
    help="Allow diagnostic re-selection under new prices without regenerating columns.",
)
args = parser.parse_args()

if args.timelimit <= 0:
    raise ValueError("--timelimit must be positive")
if not 0 <= args.mipgap <= 1:
    raise ValueError("--mipgap must be between 0 and 1")
if args.threads is not None and args.threads <= 0:
    raise ValueError("--threads must be positive")

stopwatch = time.time()
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
if os.environ.get("SLURM_JOB_ID"):
    RUN_ID += f"_j{os.environ['SLURM_JOB_ID']}"


def _git_state():
    repo_root = Path(__file__).resolve().parent.parent

    def _run(*git_args):
        result = subprocess.run(
            ["git", *git_args], cwd=repo_root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("branch", "--show-current"),
        "dirty": bool(_run("status", "--porcelain")),
    }

# ── Load checkpoint ────────────────────────────────────────────────────────────
ckpt_path = Path(args.ckpt)
if not ckpt_path.exists():
    print(f"[ERROR] Checkpoint not found: {ckpt_path}")
    sys.exit(1)

print(f"[INFO] Loading checkpoint: {ckpt_path}")
with open(ckpt_path, "r") as f:
    data = json.load(f)
if not isinstance(data, dict):
    raise ValueError(
        "Legacy list-only column pools do not record the complete trip set, inputs, "
        "or code provenance. Convert them to a metadata checkpoint before final MIP use."
    )
required_checkpoint_fields = (
    "routes",
    "trip_ids",
    "csv_name",
    "instance_sha256",
    "price_sha256",
    "mode",
    "git",
)
missing_checkpoint_fields = [
    key for key in required_checkpoint_fields
    if data.get(key) is None
]
if missing_checkpoint_fields and not args.allow_unsafe_checkpoint:
    raise ValueError(
        "Checkpoint lacks required provenance fields "
        f"{missing_checkpoint_fields}. Use a current checkpoint or pass "
        "--allow_unsafe_checkpoint only for an explicitly labeled legacy diagnostic."
    )

current_git = _git_state()
runtime_versions = {
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "gurobi": ".".join(str(part) for part in gp.gurobi.version()),
}
checkpoint_git = data.get("git") or {}
if not checkpoint_git.get("commit") and not args.allow_unsafe_checkpoint:
    raise ValueError(
        "Checkpoint does not record git.commit. Use a current checkpoint or pass "
        "--allow_unsafe_checkpoint only for an explicitly labeled legacy diagnostic."
    )
if (
    checkpoint_git.get("commit")
    and current_git.get("commit")
    and checkpoint_git["commit"] != current_git["commit"]
    and not args.allow_unsafe_checkpoint
):
    raise ValueError(
        f"Checkpoint commit {checkpoint_git['commit']} differs from current "
        f"commit {current_git['commit']}. Check out the generating commit or "
        "explicitly pass --allow_unsafe_checkpoint."
    )

R_truck        = data["routes"]
iteration_done = data.get("iteration", "?")
recorded_run_dir = Path(data.get("run_dir", ckpt_path.parent))
run_dir = recorded_run_dir if recorded_run_dir.exists() else ckpt_path.parent

print(f"[INFO] Loaded {len(R_truck)} routes from iteration {iteration_done}")
print(f"[INFO] Results will be saved to: {run_dir}")
if "run_dir" not in data or run_dir != recorded_run_dir:
    print("[INFO] Recorded run_dir is unavailable; using snapshot parent directory.")

# ── Reconstruct T ─────────────────────────────────────────────────────────────
# Prefer the instance CSV recorded in the checkpoint. Deriving T only from the
# saved routes silently DROPS every trip that never received a column, so the
# MIP would "solve" a smaller problem without any warning.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
T_from_routes = sorted({node for r in R_truck for node in r["route"] if isinstance(node, int)})
T = None
if data.get("trip_ids") is not None:
    T = sorted(int(i) for i in data["trip_ids"])
    print(f"[INFO] T restored directly from checkpoint metadata: {len(T)} trips")
ckpt_csv_name = data.get("csv_name")
if T is None and ckpt_csv_name:
    instance_csv = DATA_DIR / ckpt_csv_name
    if instance_csv.exists():
        T = list(range(len(pd.read_csv(instance_csv))))
        print(f"[INFO] T rebuilt from instance CSV {ckpt_csv_name}: {len(T)} trips")
    else:
        print(f"[WARN] Checkpoint names {ckpt_csv_name}, not found under {DATA_DIR}")
if T is None:
    T = T_from_routes
    print("[WARN] T reconstructed from saved routes only; "
          "trips that never received a column are invisible to this MIP.")
if not T:
    raise ValueError("Checkpoint contains an empty trip set")

missing_column_trips = sorted(set(T) - set(T_from_routes))
if missing_column_trips:
    print(f"[ERROR] {len(missing_column_trips)} trips have NO covering column in this pool: "
          f"{missing_column_trips[:20]}{'...' if len(missing_column_trips) > 20 else ''}")
    print("        They can only be covered by BIG-M artificial q_i variables below.")
print(f"[INFO] T: {len(T)} trips (range {min(T)}..{max(T)}); "
      f"{len(T_from_routes)} of them appear in columns")

# ── Price curve ───────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
price_value = args.prices_csv or data.get("prices_csv") or "spatiotemporal_prices.csv"
prices_csv = Path(price_value)
if not prices_csv.is_absolute():
    prices_csv = DATA_DIR / prices_csv
elif not prices_csv.exists():
    portable_candidate = DATA_DIR / prices_csv.name
    if portable_candidate.exists():
        print(f"[INFO] Remapped unavailable recorded price path to {portable_candidate}")
        prices_csv = portable_candidate
if not prices_csv.exists():
    raise FileNotFoundError(f"Price CSV not found: {prices_csv}")

price_sha256 = hashlib.sha256(prices_csv.read_bytes()).hexdigest()
checkpoint_price_sha256 = data.get("price_sha256")
if not checkpoint_price_sha256 and not args.allow_unsafe_checkpoint:
    raise ValueError(
        "Checkpoint does not record price_sha256. Use a current checkpoint or pass "
        "--allow_unsafe_checkpoint only for an explicitly labeled legacy diagnostic."
    )
if checkpoint_price_sha256 and checkpoint_price_sha256 != price_sha256:
    if not args.prices_csv:
        raise ValueError(
            "Resolved price CSV does not match checkpoint price_sha256: "
            f"{price_sha256} != {checkpoint_price_sha256}"
        )
    if not args.allow_restricted_pool_reprice:
        raise ValueError(
            "--prices_csv changes the checkpoint price hash, but this pool lacks "
            "columns that would become attractive under the new prices. Rerun column "
            "generation for a valid scenario, or explicitly pass "
            "--allow_restricted_pool_reprice for a diagnostic only."
        )
    print("[REPRICE DIAGNOSTIC] Re-selecting only from the existing old-price pool. "
          "This is not a valid price-scenario optimum or savings result.")

print(f"[INFO] Using price CSV: {prices_csv}")
station_hourly_prices = load_station_hourly_prices(prices_csv, CHARGING_STATIONS)
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
if rmp_lp.SolCount == 0:
    raise RuntimeError(f"Final LP produced no solution (status={rmp_lp.Status})")
if rmp_lp.Status != GRB.OPTIMAL:
    print(f"[WARN] Final LP status is {rmp_lp.Status}, not OPTIMAL; using its incumbent as a MIP start.")
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
rmp_final, a_final, _ = build_master(
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

# Warm-start from LP solution (LP vars are unbounded above; clamp for binaries)
for idx, var in a_final.items():
    if idx in a_lp:
        var.start = min(a_lp[idx].X, 1.0)

# ── Gurobi params ─────────────────────────────────────────────────────────────
n_threads = args.threads if args.threads else THREADS
nodefile_dir = NODEFILE_DIR or os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
Path(nodefile_dir).mkdir(parents=True, exist_ok=True)
rmp_final.Params.Threads       = n_threads
rmp_final.Params.NodefileStart = NODEFILE_START
rmp_final.Params.NodefileDir   = nodefile_dir
rmp_final.Params.TimeLimit     = args.timelimit
rmp_final.Params.MIPGap        = args.mipgap
rmp_final.Params.LogFile       = str(run_dir / f"final_mip_{RUN_ID}.log")

print(f"[INFO]  Time limit : {args.timelimit}s  ({args.timelimit/3600:.1f}h)")
print(f"[INFO]  MIP gap    : {args.mipgap*100:.4f}%")
print(f"[INFO]  Threads    : {n_threads}")
print(f"[INFO]  Log file   : {rmp_final.Params.LogFile}")

# ── Solve ─────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Optimizing ...")
mip_stopwatch = time.time()
rmp_final.optimize()
mip_runtime = time.time() - mip_stopwatch

# ── Results ───────────────────────────────────────────────────────────────────
status = rmp_final.Status
has_solution = rmp_final.SolCount > 0
best_bound = float(rmp_final.ObjBound)
final_MIP_obj = float(rmp_final.ObjVal) if has_solution else None
gap = float(rmp_final.MIPGap) if has_solution else None

print("\n=== FINAL MIP RESULTS ===")
print(f"Status       : {status}  ({'OPTIMAL' if status == GRB.OPTIMAL else 'TIME_LIMIT/OTHER'})")
print(f"LP  obj      : {final_LP_obj:,.2f}")
print(f"MIP obj      : {final_MIP_obj:,.2f}" if has_solution else "MIP obj      : no incumbent")
print(f"Best bound   : {best_bound:,.2f}")
print(f"MIP gap      : {gap*100:.4f}%" if gap is not None else "MIP gap      : unavailable")
print(f"Proven opt   : {status == GRB.OPTIMAL}")

used_routes = []
dummy_used = []
real_used = []
q_used = []
overcovered = {}
if has_solution:
    print("\n=== Selected routes ===")
    for r in range(len(R_truck)):
        if r in a_final and a_final[r].X > 0.5:
            used_routes.append(r)
            print(f"Route {r}: a[{r}]=1  -> {R_truck[r]}")

    dummy_used = [r for r in used_routes if R_truck[r].get("dummy", False)]
    real_used  = [r for r in used_routes if not R_truck[r].get("dummy", False)]
    # q_i artificials are NOT route dicts — audit them directly
    q_used = [i for i in T
              if (qv := rmp_final.getVarByName(f"q_{i}")) is not None and qv.X > 0.5]
    from collections import Counter
    _cov = Counter(n for r in used_routes
                   for n in R_truck[r].get("route", []) if isinstance(n, int))
    overcovered = {i: c for i, c in _cov.items() if c > 1}
    print(f"\nBuses used       : {len(used_routes)}")
    print(f"Real routes      : {len(real_used)}")
    print(f"Dummy routes     : {len(dummy_used)}")
    print(f"Artificial q_i   : {len(q_used)}" + (f" -> trips {q_used}" if q_used else ""))
    print(f"Trips overcovered: {len(overcovered)}")

    # Save .sol file
    sol_path = run_dir / f"final_mip_{RUN_ID}.sol"
    try:
        rmp_final.write(str(sol_path))
        print(f"Solution written : {sol_path}")
    except Exception as e:
        print(f"[WARN] Could not write .sol: {e}")

# ── Save summary JSON ─────────────────────────────────────────────────────────
elapsed = time.time() - stopwatch
snapshot_active_time_s = data.get("active_time_s")
if snapshot_active_time_s is None:
    snapshot_active_time_s = (
        float(data.get("cum_master_time", data.get("cumulative_master_time_s", 0.0)) or 0.0)
        + float(data.get("cum_pricing_time", data.get("cumulative_pricing_time_s", 0.0)) or 0.0)
    )
summary = {
    "ckpt_source"      : str(ckpt_path),
    "CG_iterations"    : iteration_done,
    "columns_in_pool"  : len(R_truck),
    "T_size"           : len(T),
    "LP_obj"           : final_LP_obj,
    "LP_status"        : int(rmp_lp.Status),
    "LP_sol_count"     : int(rmp_lp.SolCount),
    "LP_optimal"       : rmp_lp.Status == GRB.OPTIMAL,
    "MIP_obj"          : final_MIP_obj,
    "MIP_bound"        : best_bound,
    "MIP_gap_pct"      : gap * 100 if gap is not None else None,
    "MIP_status"       : int(status),
    "MIP_sol_count"    : int(rmp_final.SolCount),
    "has_solution"     : has_solution,
    "proven_optimal"   : status == GRB.OPTIMAL,
    "buses_used"       : len(used_routes) if has_solution else None,
    "real_routes_used" : len(real_used) if has_solution else None,
    "dummy_routes_used": len(dummy_used) if has_solution else None,
    "artificial_trips_used": len(q_used) if has_solution else None,
    "overcovered_trips": len(overcovered) if has_solution else None,
    "missing_column_trips": len(missing_column_trips),
    "mip_timelimit_s"  : args.timelimit,
    "mip_runtime_s"    : mip_runtime,
    "total_runtime_s"  : elapsed,
    "run_id"           : RUN_ID,
    "prices_csv"       : str(prices_csv),
    "price_sha256"     : price_sha256,
    "checkpoint_price_sha256": checkpoint_price_sha256,
    "repriced"         : bool(
        args.prices_csv
        and checkpoint_price_sha256
        and checkpoint_price_sha256 != price_sha256
    ),
    "restricted_pool_reprice": bool(
        args.allow_restricted_pool_reprice
        and checkpoint_price_sha256
        and checkpoint_price_sha256 != price_sha256
    ),
    "unsafe_checkpoint_override": bool(args.allow_unsafe_checkpoint),
    "checkpoint_missing_fields": missing_checkpoint_fields,
    "price_tag"        : data.get("price_tag"),
    "instance_csv"     : ckpt_csv_name,
    "instance_sha256"  : data.get("instance_sha256"),
    "mode"             : data.get("mode"),
    "checkpoint_git"   : checkpoint_git,
    "solver_git"       : current_git,
    "solver_runtime_versions": runtime_versions,
    "checkpoint_runtime_versions": data.get("runtime_versions"),
    "snapshot_active_time_s": snapshot_active_time_s,
    "snapshot_milestone_hours": data.get("milestone_hours"),
    "CG_termination_reason": data.get("termination_reason"),
    "seed_route_count": data.get("seed_route_count"),
    "DP_columns_generated": data.get("dp_columns_generated"),
    "seed_route_validation": data.get("seed_route_validation"),
}
summary_path = run_dir / f"final_mip_summary_{RUN_ID}.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved    : {summary_path}")
print(f"Total runtime    : {elapsed:.1f}s  ({elapsed/3600:.2f}h)")
print("=========================")

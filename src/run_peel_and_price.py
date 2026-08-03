"""Dive-and-price at the instance level ("peeling").

Motivation (two-duty falsification, 2026-08-03): pricing from uniform/artificial
duals reliably produces deep routes (NO_CHEAT recovers 37/42 exact single
duties; the two-duty NO_CHEAT pool reaches 38-trip columns), while pricing from
sparse degenerate seed duals does not. This driver exploits that asymmetry:

  1. run a short column-generation burst on the current residual instance;
  2. solve the restricted master LP over the returned pool (SciPy/HiGHS);
  3. FIX one route (highest LP weight by default), delete its trips;
  4. recurse on the residual instance until every trip is fixed.

Each stage prices a fresh instance whose master starts artificial-only, i.e.
exactly the dual landscape where the DP is empirically strongest. The result is
an integral cover built from genuinely DP-generated columns, plus per-stage
provenance. This mirrors truncated column generation with node removal
(de Vos, van Lieshout & Dollevoet, Transportation Science / arXiv:2207.13734),
which reported faster AND better solutions than single-shot price-and-branch.

This is an experiment driver: it shells out to the maintained runner per stage
and never modifies tracked inputs. Residual instances are written under
``data/peel_tmp/`` (gitignored territory; do not commit them).

Usage (from src/):

    python run_peel_and_price.py \
        --csv Practice_Custom_TwoDuty_13301_13302.csv \
        --initializer nocheat --pick lp \
        --stage-active-hours 0.04 --pricing-seconds 60 \
        --run-tag peel_twoduty
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from config import (
    BIG_M_PENALTY,
    BUS_COST_KX,
    CHARGE_RATE_KW,
    CHARGE_START_COST,
    CHARGING_STATIONS,
)
from master_lp_scipy import build_route_incidence, solve_restricted_master_lp
from utils_v2 import calculate_truck_route_cost_accurate, load_station_hourly_prices

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
DATA_DIR = REPO_ROOT / "data"
PEEL_TMP = DATA_DIR / "peel_tmp"


def _route_trips(route: dict) -> list[int]:
    return [n for n in route.get("route", []) if isinstance(n, int)]


def _minutes(hhmm: str) -> int:
    hh, mm = str(hhmm).split(":")
    return int(hh) * 60 + int(mm)  # Hastus times may exceed 24:00; hh*60 is correct


def _peak_concurrency(starts, ends) -> int:
    """Max number of trips active at one instant — a valid fleet lower bound."""
    events = sorted([(s, 1) for s in starts] + [(e, -1) for e in ends],
                    key=lambda x: (x[0], x[1]))
    peak = cur = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def _stage_runner_cmd(args, stage_csv_rel: str, stage_tag: str, results_root: Path) -> list[str]:
    cmd = [
        sys.executable, "-u", str(SRC_DIR / "run_ex_unicorn.py"),
        "--csv", stage_csv_rel,
        "--G", "300",
        "--master_backend", "scipy",
        "--skip_final_mip",
        "--queue_order", args.queue_order,
        "--pricing_output_selection", "diversified",
        "--max_charge2trip", "1560",
        "--pricing_tiers", f"{args.max_labels}:{args.pricing_seconds}",
        "--pricing_wall_per_iter", str(args.pricing_seconds + 15),
        "--active_time_limit_hours", str(args.stage_active_hours),
        "--milestones_hours", "",
        "--prices_csv", args.prices_csv,
        "--price_tag", args.price_tag,
        "--run_tag", stage_tag,
        "--results_root", str(results_root),
        "--no_resume",
    ]
    if args.initializer == "greedy":
        cmd.append("--greedy")
    elif args.initializer == "matching":
        cmd.append("--matching")
    return cmd


def _load_stage_pool(results_root: Path) -> dict:
    pools = sorted(results_root.glob("*/routes_colgen_final_*.json"),
                   key=lambda p: p.stat().st_mtime)
    if not pools:
        raise RuntimeError(f"stage produced no final pool under {results_root}")
    with open(pools[-1]) as fh:
        return json.load(fh)


def _pick_route(args, routes: list[dict], costs: list[float],
                stage_df: pd.DataFrame) -> tuple[int, dict]:
    """Return (index, diagnostics) of the route to fix this stage."""
    n_trips = len(stage_df)
    if args.pick == "max_trips":
        idx = max(range(len(routes)),
                  key=lambda i: (len(_route_trips(routes[i])), -costs[i]))
        return idx, {"pick": "max_trips"}

    trip_ids = list(range(n_trips))
    incidence = build_route_incidence(
        trip_ids=trip_ids,
        route_trip_ids=[_route_trips(r) for r in routes],
    )
    lp = solve_restricted_master_lp(
        trip_ids=trip_ids,
        route_incidence=incidence,
        route_costs=costs,
        artificial_penalty=BIG_M_PENALTY,
    )

    if args.pick == "lookahead":
        # Anti-straddling pick: greedily minimize 1 + LB(residual fleet), where
        # LB is the residual's peak trip concurrency. A route that mixes trips
        # from what should be two different buses leaves a residual whose LB is
        # unchanged, scoring worse than a "duty-shaped" route that lowers it.
        st = stage_df["Start1"].map(_minutes).to_numpy()
        et = stage_df["End1"].map(_minutes).to_numpy()

        def residual_lb(i: int) -> int:
            keep = set(range(n_trips)) - set(_route_trips(routes[i]))
            if not keep:
                return 0
            keep = sorted(keep)
            return _peak_concurrency(st[keep], et[keep])

        scored = [(1 + residual_lb(i), -lp.route_values[i],
                   -len(_route_trips(routes[i])), costs[i], i)
                  for i in range(len(routes))]
        best = min(scored)
        idx = best[4]
        diag = {
            "pick": "lookahead",
            "stage_lp_route_weight": lp.route_weight,
            "stage_lp_artificial_total": lp.artificial_total,
            "picked_lp_value": lp.route_values[idx],
            "picked_residual_lb": best[0] - 1,
        }
        return idx, diag

    # Highest LP weight; break ties toward deeper routes so the residual shrinks.
    idx = max(range(len(routes)),
              key=lambda i: (lp.route_values[i], len(_route_trips(routes[i]))))
    diag = {
        "pick": "lp",
        "stage_lp_route_weight": lp.route_weight,
        "stage_lp_artificial_total": lp.artificial_total,
        "picked_lp_value": lp.route_values[idx],
    }
    return idx, diag


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True,
                        help="Instance CSV relative to data/.")
    parser.add_argument("--initializer", choices=("nocheat", "greedy", "matching"),
                        default="nocheat",
                        help="Per-stage warm start. nocheat gives the uniform-dual "
                             "landscape where deep pricing is empirically strongest.")
    parser.add_argument("--pick", choices=("lookahead", "lp", "max_trips"),
                        default="lookahead",
                        help="lookahead minimizes 1 + residual peak-concurrency "
                             "LB (anti-straddling); lp fixes the max-LP-weight "
                             "column; max_trips is the naive depth pick.")
    parser.add_argument("--queue_order", default="reduced_cost_bound")
    parser.add_argument("--stage-active-hours", type=float, default=0.04)
    parser.add_argument("--pricing-seconds", type=int, default=60)
    parser.add_argument("--max_labels", type=int, default=200000)
    parser.add_argument("--prices_csv", default="hourly_prices_flat.csv")
    parser.add_argument("--price_tag", default="flat")
    parser.add_argument("--max-stages", type=int, default=60)
    parser.add_argument("--run-tag", default="peel")
    parser.add_argument("--out", type=Path, default=None,
                        help="Summary JSON destination (default: results dir).")
    args = parser.parse_args(argv)

    prices_path = DATA_DIR / args.prices_csv
    station_prices = load_station_hourly_prices(prices_path, CHARGING_STATIONS)
    depot_curve = station_prices["PARX"]

    def master_cost(route: dict) -> float:
        return calculate_truck_route_cost_accurate(
            route, BUS_COST_KX, depot_curve,
            charge_rate_kw=CHARGE_RATE_KW,
            station_hourly_prices=station_prices,
            charge_start_cost=CHARGE_START_COST,
        )

    PEEL_TMP.mkdir(parents=True, exist_ok=True)
    batch_root = SRC_DIR / "results" / f"peel_{args.run_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    batch_root.mkdir(parents=True, exist_ok=True)

    stage_df = pd.read_csv(DATA_DIR / args.csv)
    if "Ordered_Trip_ID" not in stage_df.columns:
        raise SystemExit("Instance CSV must carry Ordered_Trip_ID for provenance.")
    original_trips = len(stage_df)

    fixed = []
    t0 = time.time()
    stage = 0
    stage_csv_rel = args.csv

    while len(stage_df) > 0 and stage < args.max_stages:
        stage += 1
        stage_tag = f"{args.run_tag}_s{stage:02d}"
        stage_results = batch_root / f"stage_{stage:02d}"
        stage_results.mkdir(parents=True, exist_ok=True)

        print(f"\n[PEEL] stage {stage}: {len(stage_df)} trips remain "
              f"({stage_csv_rel})", flush=True)
        log_path = stage_results / "runner.log"
        with open(log_path, "w") as log:
            rc = subprocess.run(
                _stage_runner_cmd(args, stage_csv_rel, stage_tag, stage_results),
                cwd=SRC_DIR, stdout=log, stderr=subprocess.STDOUT,
            ).returncode
        if rc != 0:
            raise RuntimeError(f"stage {stage} runner failed (rc={rc}); see {log_path}")

        pool = _load_stage_pool(stage_results)
        routes = [r for r in pool["routes"] if _route_trips(r)]
        if not routes:
            raise RuntimeError(f"stage {stage} pool holds no real trip-covering route")
        costs = [master_cost(r) for r in routes]

        idx, diag = _pick_route(args, routes, costs, stage_df)
        picked = routes[idx]
        picked_local = sorted(set(_route_trips(picked)))
        picked_ordered_ids = stage_df.iloc[picked_local]["Ordered_Trip_ID"].tolist()

        fixed.append({
            "stage": stage,
            "trips": len(picked_local),
            "cost": costs[idx],
            "ordered_trip_ids": picked_ordered_ids,
            "route_nodes": picked["route"],
            "charging_stops": picked.get("charging_stops", {}),
            "stage_final_lp_route_weight": pool.get("final_lp_route_weight"),
            **diag,
        })
        print(f"[PEEL] stage {stage}: fixed a {len(picked_local)}-trip route, "
              f"cost {costs[idx]:.2f} "
              f"(stage LP weight {pool.get('final_lp_route_weight')}, {diag.get('pick')})",
              flush=True)

        stage_df = stage_df.drop(stage_df.index[picked_local]).reset_index(drop=True)
        stage_df["count_trip_id"] = range(len(stage_df))
        if len(stage_df) > 0:
            next_csv = PEEL_TMP / f"{args.run_tag}_stage{stage + 1:02d}.csv"
            stage_df.to_csv(next_csv, index=False)
            stage_csv_rel = str(next_csv.relative_to(DATA_DIR))

    covered = sum(f["trips"] for f in fixed)
    summary = {
        "instance_csv": args.csv,
        "initializer": args.initializer,
        "pick": args.pick,
        "queue_order": args.queue_order,
        "stage_active_hours": args.stage_active_hours,
        "pricing_seconds": args.pricing_seconds,
        "prices_csv": args.prices_csv,
        "original_trips": original_trips,
        "trips_covered": covered,
        "complete_partition": covered == original_trips and len(stage_df) == 0,
        "buses_used": len(fixed),
        "total_cost": sum(f["cost"] for f in fixed),
        "wall_time_s": time.time() - t0,
        "stages": fixed,
    }
    out = args.out or (batch_root / "peel_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\n[PEEL] DONE: {len(fixed)} buses for {original_trips} trips "
          f"(complete={summary['complete_partition']}), "
          f"total cost {summary['total_cost']:.2f}, "
          f"{summary['wall_time_s']:.0f}s wall. Summary: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

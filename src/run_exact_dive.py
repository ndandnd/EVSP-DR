"""Truncated exact CG with column fixing and node removal ("exact diving").

Motivation (2026-08-05): pool MIPs over one-shot exact-CG pools show large
integrality gaps at scale (k=30 LP 29-30 -> MIP ~45; k=40 LP 39.1-39.8 ->
MIP ~55). This is the documented price-and-branch failure mode; de Vos,
van Lieshout & Dollevoet fixed it with truncated column generation plus
fixing and network shrinking (their 512-trip case: faster AND better).
This driver is that loop built on the exact pricer:

  1. run a wall-limited exact CG round on the current residual instance;
  2. from the final LP, FIX a disjoint set of high-value routes
     (LP value >= theta; if none qualifies, the single highest-value route);
  3. delete the fixed trips, write the residual instance, recurse;
  4. when the residual is empty, emit the complete integral schedule
     (all fixed routes with charging events) plus per-round provenance.

Every fixed route is a genuine priced column with its full charging
realization; the result is an integral cover by construction — no final
MIP required (though running one over the union of round pools can only
improve it, and the driver records that union).

Usage (from src/):

    python run_exact_dive.py --csv duty_unions_big/Practice_Custom_DutyUnion_k40_r1.csv \
        --round-wall-s 7200 --theta 0.7 --rc-eps 5 --run-tag dive_k40r1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / "data"
DIVE_TMP = DATA_DIR / "exact_dive_tmp"


def _round_cmd(args, stage_csv_rel: str, out_json: Path) -> list[str]:
    cmd = [
        sys.executable, "-u", str(SRC_DIR / "exact_pricer_expanded.py"),
        "--csv", stage_csv_rel,
        "--prices_csv", args.prices_csv,
        "--soc-step", str(args.soc_step),
        "--block-min", str(args.block_min),
        "--g-kwh", str(args.g_kwh),
        "--charge-kw", str(args.charge_kw),
        "--min-soc-frac", str(args.min_soc_frac),
        "--rc-eps", str(args.rc_eps),
        "--max-iters", str(args.max_iters),
        "--wall-limit-s", str(args.round_wall_s),
        "--checkpoint-every", "25",
        "--resume",
        "--out", str(out_json),
    ]
    if args.master_sense:
        cmd += ["--master-sense", args.master_sense]
    return cmd


def _load_journal(out_json: Path) -> dict:
    pool = {}
    journal = Path(str(out_json) + ".columns.jsonl")
    if journal.exists():
        with open(journal) as fh:
            for line in fh:
                rec = json.loads(line)
                key = frozenset(rec["trips"])
                if key not in pool or rec["cost"] < pool[key]["cost"] - 1e-9:
                    pool[key] = rec
    return pool


def pick_fixes(status: dict, pool: dict, theta: float) -> list[dict]:
    """Disjoint high-LP-value routes (>= theta), greedy by value; fallback: best one."""
    positive = (status.get("final_lp") or {}).get("positive_routes") or []
    positive = sorted(positive, key=lambda r: -r["value"])
    chosen, covered = [], set()
    for cand in positive:
        if cand["value"] < theta:
            break
        trips = set(cand["trips"])
        if covered & trips:
            continue
        rec = pool.get(frozenset(cand["trips"]))
        if rec is None:
            continue
        chosen.append(rec)
        covered |= trips
    if not chosen and positive:
        rec = pool.get(frozenset(positive[0]["trips"]))
        if rec is not None:
            chosen = [rec]
    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Instance CSV relative to data/.")
    parser.add_argument("--prices_csv", default="hourly_prices_flat.csv")
    parser.add_argument("--soc-step", type=float, default=15.0)
    parser.add_argument("--block-min", type=int, default=10)
    parser.add_argument("--g-kwh", type=float, default=300.0)
    parser.add_argument("--charge-kw", type=float, default=300.0)
    parser.add_argument("--min-soc-frac", type=float, default=0.0)
    parser.add_argument("--rc-eps", type=float, default=5.0,
                        help="Per-round pricing tolerance (production dial; "
                             "rounds are tolerance-terminated, not certificates).")
    parser.add_argument("--theta", type=float, default=0.7,
                        help="Minimum LP value for fixing (de Vos used 0.7).")
    parser.add_argument("--round-wall-s", type=int, default=3600)
    parser.add_argument("--max-iters", type=int, default=200000)
    parser.add_argument("--master-sense", default=None,
                        help="Forwarded to the pricer when set (e.g. partition).")
    parser.add_argument("--max-rounds", type=int, default=120)
    parser.add_argument("--global-wall-s", type=int, default=None,
                        help="Stop starting new rounds after this many seconds; "
                             "the partial summary (complete_partition=false) "
                             "is written either way.")
    parser.add_argument("--run-tag", default="dive")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    t0 = time.time()
    DIVE_TMP.mkdir(parents=True, exist_ok=True)
    batch_root = SRC_DIR / "results" / f"exact_dive_{args.run_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    batch_root.mkdir(parents=True, exist_ok=True)

    stage_df = pd.read_csv(DATA_DIR / args.csv)
    if "Ordered_Trip_ID" not in stage_df.columns:
        raise SystemExit("Instance CSV must carry Ordered_Trip_ID for provenance.")
    original_trips = len(stage_df)

    fixed, rounds = [], []
    stage_csv_rel = args.csv
    out = args.out or (batch_root / "dive_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    def _write_summary(complete: bool):
        summary = {
            "instance_csv": args.csv, "prices_csv": args.prices_csv,
            "soc_step": args.soc_step, "block_min": args.block_min,
            "g_kwh": args.g_kwh, "charge_kw": args.charge_kw,
            "min_soc_frac": args.min_soc_frac, "theta": args.theta,
            "rc_eps": args.rc_eps, "round_wall_s": args.round_wall_s,
            "original_trips": original_trips,
            "trips_remaining": len(stage_df),
            "complete_partition": complete,
            "buses_used": len(fixed),
            "total_cost": sum(f["cost"] for f in fixed),
            "rounds": rounds,
            "wall_s": time.time() - t0,
            "fixed_routes": fixed,
        }
        tmp = Path(str(out) + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(summary, fh, indent=1)
        tmp.replace(out)
        return summary

    for rnd in range(1, args.max_rounds + 1):
        if args.global_wall_s and time.time() - t0 > args.global_wall_s:
            print(f"[DIVE] global wall {args.global_wall_s}s reached — "
                  "stopping with partial partition saved", flush=True)
            break
        out_json = batch_root / f"round_{rnd:03d}.json"
        print(f"\n[DIVE] round {rnd}: {len(stage_df)} trips remain "
              f"({stage_csv_rel})", flush=True)
        log_path = batch_root / f"round_{rnd:03d}.log"
        with open(log_path, "w") as log:
            rc = subprocess.run(_round_cmd(args, stage_csv_rel, out_json),
                                cwd=SRC_DIR, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise SystemExit(f"[DIVE] round {rnd} pricer failed (rc={rc}); "
                             f"see {log_path}")
        with open(out_json) as fh:
            status = json.load(fh)
        pool = _load_journal(out_json)
        fixes = pick_fixes(status, pool, args.theta)
        if not fixes:
            raise SystemExit(f"[DIVE] round {rnd}: no fixable route in the LP "
                             "support — inspect the round JSON.")

        local_to_ordered = stage_df["Ordered_Trip_ID"].tolist()
        removed_local = set()
        for rec in fixes:
            fixed.append({
                "round": rnd,
                "trips_local": rec["trips"],
                "ordered_trip_ids": [local_to_ordered[t] for t in rec["trips"]],
                "cost": rec["cost"],
                "route_nodes": rec["route_nodes"],
                "charging_stops": rec["charging_stops"],
            })
            removed_local |= set(rec["trips"])
        rounds.append({
            "round": rnd,
            "trips_before": len(stage_df),
            "fixed_routes": len(fixes),
            "fixed_trips": len(removed_local),
            "round_lp_weight": (status.get("final") or {}).get("route_weight"),
            "round_stop_reason": status.get("stop_reason"),
            "round_wall_s": status.get("wall_s"),
        })
        print(f"[DIVE] round {rnd}: fixed {len(fixes)} routes covering "
              f"{len(removed_local)} trips "
              f"(round LP weight {(status.get('final') or {}).get('route_weight')}, "
              f"{status.get('stop_reason')})", flush=True)

        keep = [i for i in range(len(stage_df)) if i not in removed_local]
        stage_df = stage_df.iloc[keep].reset_index(drop=True)
        stage_df["count_trip_id"] = range(len(stage_df))
        _write_summary(complete=len(stage_df) == 0)
        if len(stage_df) == 0:
            break
        next_csv = DIVE_TMP / f"{args.run_tag}_round{rnd + 1:03d}.csv"
        stage_df.to_csv(next_csv, index=False)
        stage_csv_rel = str(next_csv.relative_to(DATA_DIR))

    summary = _write_summary(complete=len(stage_df) == 0)
    print(f"\n[DIVE] DONE: {len(fixed)} buses for {original_trips} trips "
          f"(complete={summary['complete_partition']}) in {len(rounds)} rounds, "
          f"{summary['wall_s']:.0f}s wall. Summary: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

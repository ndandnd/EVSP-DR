"""Adaptive overnight MIP orchestrator (Scaglione).

Loops over its shard of instances; for each, finds the journaled pool
(300 kWh repool/exact_big arms AND 240 kWh realism arms), ensures a MATCHING
cover exists (building one with a short runner call if needed), checks
whether any previous match-MIP already closed the cell (buses <= target),
and otherwise runs a strict partition MIP with escalating budgets until the
global wall runs out. Every decision is logged; reruns are idempotent.

    python run_matchmip_escalate.py --shard 1 --shards 3 --global-wall-s 82800
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
DATA = SRC.parent / "data"


def instance_table():
    rows = []
    for k, reps, soc, block, dirname in (
        (8, 6, 5, 5, "duty_unions"),
        (13, 6, 5, 5, "duty_unions"),
        (15, 6, 15, 10, "duty_unions_big"),
    ):
        for r in range(1, reps + 1):
            name = f"Practice_Custom_DutyUnion_k{k:02d}_r{r}"
            rows.append({
                "name": name, "k": k,
                "csv": f"{dirname}/{name}.csv",
                "pools": [f"results/repool_small/{name}_soc{soc}_b{block}.json"],
            })
    for k, reps in ((30, 6), (40, 4)):
        for r in range(1, reps + 1):
            name = f"Practice_Custom_DutyUnion_k{k}_r{r}"
            base = f"results/exact_big/{name}_soc15_b10"
            rows.append({
                "name": name, "k": k,
                "csv": f"duty_unions_big/{name}.csv",
                "pools": [f"{base}_g300_res0.0.json", f"{base}_g300.json",
                          f"{base}.json"],
            })
            rows.append({
                "name": name + " [240kWh]", "k": k,
                "csv": f"duty_unions_big/{name}.csv",
                "pools": [f"{base}_g240_res0.2.json"],
            })
    return rows


def find_pool(row):
    for cand in row["pools"]:
        if Path(cand).exists():
            return Path(cand)
    return None


def find_or_build_cover(row, dry):
    name = row["csv"].split("/")[-1][:-4]
    cover_dir = SRC / "results" / "matching_covers" / name
    hits = sorted(cover_dir.glob("*/routes_colgen_final_*.json")) \
        if cover_dir.exists() else []
    if hits:
        return hits[0]
    if dry:
        return None
    cover_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(SRC / "run_ex_unicorn.py"),
           "--csv", row["csv"], "--G", "300",
           "--master_backend", "scipy", "--skip_final_mip", "--matching",
           "--queue_order", "reduced_cost_bound", "--max_charge2trip", "1560",
           "--pricing_tiers", "200000:30", "--pricing_wall_per_iter", "40",
           "--active_time_limit_hours", "0.02", "--milestones_hours", "",
           "--prices_csv", "hourly_prices_flat.csv", "--price_tag", "flat",
           "--run_tag", f"matchseed_{name}", "--results_root", str(cover_dir),
           "--no_resume"]
    log = cover_dir / "matchseed.log"
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=SRC, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"[ORCH] cover build FAILED for {name} (rc={rc}); see {log}",
              flush=True)
        return None
    hits = sorted(cover_dir.glob("*/routes_colgen_final_*.json"))
    return hits[0] if hits else None


def best_existing(pool: Path):
    best = None
    for out in glob.glob(str(pool).replace(".json", "_match_mip*.json")):
        try:
            with open(out) as fh:
                buses = json.load(fh).get("buses")
            if buses and (best is None or buses < best):
                best = buses
        except Exception:
            continue
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--global-wall-s", type=int, default=82800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.time()
    mine = [row for i, row in enumerate(instance_table())
            if i % args.shards == args.shard - 1]
    print(f"[ORCH] shard {args.shard}/{args.shards}: {len(mine)} instances",
          flush=True)

    budgets = {True: (3600, 7200, 10800),    # k <= 15
               False: (7200, 14400, 21600)}  # k >= 30
    for pass_no in range(3):
        for row in mine:
            if time.time() - t0 > args.global_wall_s:
                print("[ORCH] global wall reached; stopping cleanly", flush=True)
                return 0
            pool = find_pool(row)
            if pool is None:
                if pass_no == 0:
                    print(f"[ORCH] {row['name']}: no journaled pool yet — skip",
                          flush=True)
                continue
            target = row["k"]
            got = best_existing(pool)
            if got is not None and got <= target:
                if pass_no == 0:
                    print(f"[ORCH] {row['name']}: CLOSED at {got} buses "
                          f"(target {target}) — skip", flush=True)
                continue
            budget = budgets[row["k"] <= 15][pass_no]
            if args.dry_run:
                print(f"[ORCH] would run {row['name']} pass {pass_no + 1} "
                      f"budget {budget}s (best so far: {got})", flush=True)
                continue
            cover = find_or_build_cover(row, args.dry_run)
            if cover is None:
                print(f"[ORCH] {row['name']}: no cover available — skip",
                      flush=True)
                continue
            out = Path(str(pool).replace(".json",
                                         f"_match_mip_p{pass_no + 1}.json"))
            if out.exists():
                continue
            print(f"[ORCH] {row['name']} pass {pass_no + 1}: MIP {budget}s "
                  f"(best so far: {got})", flush=True)
            cmd = [sys.executable, "-u", str(SRC / "run_exact_pool_mip.py"),
                   "--result", str(pool), "--extra-routes", str(cover),
                   "--timelimit", str(budget), "--threads", "8",
                   "--out", str(out)]
            rc = subprocess.run(cmd, cwd=SRC).returncode
            if rc != 0:
                print(f"[ORCH] {row['name']}: MIP rc={rc}", flush=True)
    print(f"[ORCH] all passes done in {(time.time() - t0) / 3600:.1f}h",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

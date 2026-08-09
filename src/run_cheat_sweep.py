"""CHEAT sweep: improve GIRO's own plan under each tariff (Scaglione driver).

For every journaled pool of the target instances (flat pools from
repool_small plus each tariff pool from tariff_matrix as they land), this
driver ensures the instance's GIRO seed exists (one route per constituent
duty, exact partition by construction), then solves the strict partition MIP
over pool + seed. Because the seed is a feasible incumbent, every result is
"GIRO's fleet or fewer, GIRO's cost or better" under that tariff — the
Tier-2 rows of the decomposition experiment, measured against the Tier-0
repriced baseline.

    python run_cheat_sweep.py --shard 1 --shards 2 --global-wall-s 82800
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

INSTANCES = (
    [(f"Practice_Custom_DutyUnion_k08_r{r}", "duty_unions") for r in range(1, 7)]
    + [(f"Practice_Custom_DutyUnion_k13_r{r}", "duty_unions") for r in range(1, 7)]
    + [(f"Practice_Custom_DutyUnion_k15_r{r}", "duty_unions_big") for r in range(1, 7)]
)
TARIFF_POOLS = {
    "flat": "results/repool_small/{name}_soc*_b*.json",
    "peak08": "results/tariff_matrix/{name}_peak08.json",
    "peak12": "results/tariff_matrix/{name}_peak12.json",
    "peak18": "results/tariff_matrix/{name}_peak18.json",
    "sek": "results/tariff_matrix/{name}_sek.json",
}
TARIFF_PRICES = {
    "flat": "hourly_prices_flat.csv",
    "peak08": "hourly_prices_single_peak_08.csv",
    "peak12": "hourly_prices_single_peak_12.csv",
    "peak18": "hourly_prices_single_peak_18.csv",
    "sek": "hourly_prices_transdev_sek.csv",
}


def find_pool(pattern: str):
    hits = [Path(p) for p in sorted(glob.glob(pattern))
            if ".snapshot." not in p and not p.endswith("_mip.json")
            and "certified_backup" not in p]
    return hits[0] if hits else None


def ensure_seed(name: str, dirname: str, dry: bool):
    seed = SRC / "results" / "giro_seeds" / f"{name}_giro_seed.json"
    if seed.exists():
        return seed
    if dry:
        return None
    rc = subprocess.run(
        [sys.executable, "-u", str(SRC / "make_giro_seed_routes.py"),
         "--instance", f"{dirname}/{name}.csv", "--out", str(seed)],
        cwd=SRC).returncode
    return seed if rc == 0 and seed.exists() else None


def ensure_rerealized_seed(seed: Path, pool: Path, tariff: str):
    """Re-optimize the seed's charging under the pool's physics + tariff.

    Recorded GIRO plans fail injection (Hastus rounding, missing arcs,
    physics mismatch); the re-realized seed is injection-valid by
    construction. Exit code 3 = some duties provably infeasible under this
    physics; the valid subset is still a partial backbone worth injecting.
    """
    try:
        with open(pool) as fh:
            g_kwh = float(json.load(fh).get("g_kwh", 300.0)) or 300.0
    except Exception:
        g_kwh = 300.0
    out = seed.with_name(f"{seed.stem}_rrz_g{int(g_kwh)}_{tariff}.json")
    if out.exists():
        return out
    rc = subprocess.run(
        [sys.executable, "-u", str(SRC / "rerealize_routes.py"),
         "--routes", str(seed), "--physics-from", str(pool),
         "--prices", TARIFF_PRICES[tariff], "--out", str(out)],
        cwd=SRC).returncode
    if rc not in (0, 3) or not out.exists():
        print(f"[CHEAT] re-realization failed (rc={rc}) — falling back to "
              f"raw seed", flush=True)
        return seed
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=1)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--timelimit", type=int, default=2700)
    parser.add_argument("--global-wall-s", type=int, default=82800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.time()
    cells = [(name, dirname, tariff, pattern.format(name=name))
             for name, dirname in INSTANCES
             for tariff, pattern in TARIFF_POOLS.items()]
    mine = [c for i, c in enumerate(cells) if i % args.shards == args.shard - 1]
    print(f"[CHEAT] shard {args.shard}/{args.shards}: {len(mine)} cells",
          flush=True)

    for name, dirname, tariff, pattern in mine:
        if time.time() - t0 > args.global_wall_s:
            print("[CHEAT] global wall reached; stopping cleanly", flush=True)
            return 0
        pool = find_pool(pattern)
        if pool is None:
            print(f"[CHEAT] {name}/{tariff}: pool not landed yet — skip",
                  flush=True)
            continue
        # _rrz suffix: keeps pre-re-realization results as evidence and lets
        # every cell rerun over the repaired (injection-valid) seed.
        out = Path(str(pool).replace(".json", "_cheat_mip_rrz.json"))
        if out.exists():
            continue
        if args.dry_run:
            print(f"[CHEAT] would solve {name}/{tariff} over {pool.name}",
                  flush=True)
            continue
        seed = ensure_seed(name, dirname, args.dry_run)
        if seed is None:
            print(f"[CHEAT] {name}: seed generation failed — skip", flush=True)
            continue
        seed = ensure_rerealized_seed(seed, pool, tariff)
        print(f"[CHEAT] {name}/{tariff}: MIP {args.timelimit}s over "
              f"{pool.name} + {seed.name}", flush=True)
        rc = subprocess.run(
            [sys.executable, "-u", str(SRC / "run_exact_pool_mip.py"),
             "--result", str(pool), "--extra-routes", str(seed),
             "--timelimit", str(args.timelimit), "--threads", "8",
             "--out", str(out)], cwd=SRC).returncode
        if rc != 0:
            print(f"[CHEAT] {name}/{tariff}: MIP rc={rc}", flush=True)
    print(f"[CHEAT] shard done in {(time.time() - t0) / 3600:.1f}h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

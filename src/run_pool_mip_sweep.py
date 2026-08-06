"""Sequential strict-MIP sweep over every journaled exact-CG pool.

Small pool MIPs solve in minutes, so one job sweeping N pools beats N
allocations. For each result JSON matching the given globs (that has a
readable column journal), runs the strict fleet-first two-stage partition
MIP unless its output already exists.

    python run_pool_mip_sweep.py \
        "results/repool_small/*.json" "results/exact_peaks/*.json" \
        --timelimit 900 --threads 4
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import run_exact_pool_mip


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("globs", nargs="+")
    parser.add_argument("--timelimit", type=int, default=900,
                        help="Per-pool MIP time limit (two-stage splits it).")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--redo", action="store_true",
                        help="Re-run pools whose MIP output already exists.")
    args = parser.parse_args(argv)

    results = sorted({Path(p) for pattern in args.globs
                      for p in glob.glob(pattern)})
    # Skip our own outputs and snapshots' derived MIP files.
    results = [p for p in results
               if not p.name.endswith(("_mip.json",)) and p.suffix == ".json"]
    print(f"[SWEEP] {len(results)} candidate result files")

    done, skipped, failed = [], [], []
    for res in results:
        out = Path(str(res).replace(".json", "_mip.json"))
        if out.exists() and not args.redo:
            skipped.append(res.name)
            continue
        t0 = time.time()
        try:
            run_exact_pool_mip.main([
                "--result", str(res),
                "--two-stage",
                "--timelimit", str(args.timelimit),
                "--threads", str(args.threads),
                "--out", str(out),
            ])
            with open(out) as fh:
                summary = json.load(fh)
            done.append((res.name, summary.get("buses"),
                         summary.get("status_name"), time.time() - t0))
            print(f"[SWEEP] OK {res.name}: buses={summary.get('buses')} "
                  f"({summary.get('status_name')}, {time.time() - t0:.0f}s)",
                  flush=True)
        except SystemExit as exc:  # no journal / uncovered trips / infeasible
            failed.append((res.name, str(exc)[:120]))
            print(f"[SWEEP] SKIP {res.name}: {exc}", flush=True)
        except Exception as exc:
            failed.append((res.name, repr(exc)[:120]))
            print(f"[SWEEP] FAIL {res.name}: {exc!r}", flush=True)

    print(f"\n[SWEEP] done={len(done)} pre-existing={len(skipped)} "
          f"failed/skipped={len(failed)}")
    for name, buses, status_name, dt in done:
        print(f"  {name}: buses={buses} {status_name} {dt:.0f}s")
    for name, why in failed:
        print(f"  UNSOLVED {name}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

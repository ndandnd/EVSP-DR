"""One-screen summary of the post-fix overnight correctness campaign."""

from __future__ import annotations

import glob
import json
from pathlib import Path


SRC = Path(__file__).resolve().parent


def load_many(pattern):
    rows = []
    for name in sorted(glob.glob(str(SRC / pattern))):
        try:
            with open(name) as fh:
                rows.append((Path(name), json.load(fh)))
        except (OSError, ValueError):
            continue
    return rows


def main() -> int:
    audits = load_many("results/master_audit/*.master_audit.json")
    audit_ok = [(path, data) for path, data in audits
                if data.get("any_method_succeeded")]
    print("=== raw master audits ===")
    print(f"landed={len(audits)} passed={len(audit_ok)} "
          f"failed={len(audits) - len(audit_ok)}")
    for path, data in audits:
        successes = [row for row in data.get("methods", [])
                     if row.get("success")]
        if successes:
            row = min(successes, key=lambda item: item.get("runtime_s", 1e99))
            print(f"  {path.stem:<58} OK  cols={data.get('pool_columns', 0):>6} "
                  f"w={row.get('route_weight', 0):>8.4f} "
                  f"art={row.get('artificial_total', 0):>7.2g} "
                  f"row_res={row.get('max_row_violation', 0):.2g} "
                  f"bound_res={row.get('max_bound_violation', 0):.2g}")
        else:
            errors = " | ".join(row.get("error", "unknown")
                                for row in data.get("methods", []))
            print(f"  {path.stem:<58} FAIL {errors[:100]}")

    print("\n=== snapshot MIP curve ===")
    mips = load_many("results/stopping_mip/*.json")
    print(f"landed={len(mips)}")
    for path, data in mips:
        stage = data.get("two_stage") or {}
        source_cg_wall_s = data.get("source_cg_wall_s")
        source_cg_age = (
            f"{float(source_cg_wall_s) / 3600:.2f}h"
            if source_cg_wall_s is not None else "NA"
        )
        print(f"  {path.stem:<58} buses={data.get('buses')} "
              f"cg_age={source_cg_age} "
              f"cg_iter={data.get('source_cg_iterations')} "
              f"fleet_proven={stage.get('fleet_proven')} "
              f"fleet_bound={stage.get('stage1_bound')} "
              f"cost_gap={stage.get('stage2_absolute_gap')} "
              f"scope={data.get('optimal_scope')} "
              f"wall={float(data.get('runtime_s', 0)) / 60:.1f}m "
              f"status={data.get('status_name')}")

    print("\n=== no-stall 72h controls ===")
    controls = load_many("results/stopping_controls/*_nostall.json")
    print(f"landed={len(controls)}")
    for path, data in controls:
        final = data.get("final_lp") or data.get("final") or {}
        print(f"  {path.stem:<58} stop={data.get('stop_reason')} "
              f"hours={float(data.get('wall_s', 0)) / 3600:.1f} "
              f"w={final.get('route_weight')} "
              f"art={final.get('artificial_total', final.get('artificials'))} "
              f"source={data.get('final_lp_source')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

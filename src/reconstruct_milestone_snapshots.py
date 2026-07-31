"""
Reconstruct milestone route snapshots from pricing stats + latest checkpoint.

This is for runs where milestone snapshots were not written live. It relies on
R_truck being append-only: the latest checkpoint has all routes, and the pricing
CSV tells us how many routes were appended in each iteration.

Default output names include "reconstructed" so running jobs will not overwrite
them with late live milestone snapshots.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import pandas as pd


MILESTONES_HOURS = [3.0, 10.0, 24.0]


def _bus_label_from_checkpoint_path(path: Path) -> str:
    match = re.search(r"ckpt_latest_(.+)_g\d+_\d+cols", path.stem)
    return match.group(1) if match else path.parent.name


def _stats_path_for_checkpoint(data: dict, run_dir: Path) -> Path | None:
    raw = data.get("stats_csv_path")
    if raw and Path(raw).exists():
        return Path(raw)

    candidates = sorted(run_dir.glob("pricing_*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _active_seconds_by_row(df: pd.DataFrame) -> pd.Series:
    master = pd.to_numeric(df.get("Master_Time_s", 0.0), errors="coerce").fillna(0.0)
    pricing = pd.to_numeric(df.get("Pricing_Time_s", 0.0), errors="coerce").fillna(0.0)
    return master + pricing


def reconstruct_one(ckpt_path: Path, canonical_names: bool, overwrite: bool) -> None:
    run_dir = ckpt_path.parent

    try:
        data = json.loads(ckpt_path.read_text())
    except Exception as exc:
        print(f"SKIP unreadable checkpoint: {ckpt_path}: {exc}")
        return

    routes = data.get("routes", [])
    if not isinstance(routes, list):
        print(f"SKIP checkpoint has no route list: {ckpt_path}")
        return

    stats_path = _stats_path_for_checkpoint(data, run_dir)
    if stats_path is None:
        print(f"SKIP no pricing CSV found: {run_dir}")
        return

    try:
        df = pd.read_csv(stats_path)
    except Exception as exc:
        print(f"SKIP unreadable stats CSV: {stats_path}: {exc}")
        return

    if df.empty or "Cols_Added" not in df.columns:
        print(f"SKIP stats CSV lacks rows or Cols_Added: {stats_path}")
        return

    cols_added = pd.to_numeric(df["Cols_Added"], errors="coerce").fillna(0).astype(int)
    total_cols_in_stats = int(cols_added.sum())
    initial_route_count = len(routes) - total_cols_in_stats

    if initial_route_count < 0:
        print(
            f"SKIP inconsistent checkpoint/stats, more CSV columns than checkpoint routes: "
            f"{run_dir.name}"
        )
        return

    per_row_active = _active_seconds_by_row(df)
    csv_active_s = float(per_row_active.sum())
    ckpt_active_s = float(data.get("cum_master_time", 0.0)) + float(data.get("cum_pricing_time", 0.0))
    time_offset_s = max(0.0, ckpt_active_s - csv_active_s)
    active_s_by_row = time_offset_s + per_row_active.cumsum()
    route_count_by_row = initial_route_count + cols_added.cumsum()

    bus_label = data.get("bus_label") or _bus_label_from_checkpoint_path(ckpt_path)

    print(
        f"\n{run_dir.name}\n"
        f"  stats={stats_path.name}\n"
        f"  routes={len(routes)}, initial_routes={initial_route_count}, "
        f"ckpt_active={ckpt_active_s / 3600.0:.2f}h, csv_window={csv_active_s / 3600.0:.2f}h"
    )

    for milestone_h in MILESTONES_HOURS:
        milestone_s = milestone_h * 3600.0

        if ckpt_active_s < milestone_s:
            print(f"  {milestone_h:g}h: not reached")
            continue

        if milestone_s <= time_offset_s:
            print(
                f"  {milestone_h:g}h: cannot reconstruct exactly from this CSV "
                f"(milestone was before this stats window)"
            )
            continue

        crossed = active_s_by_row[active_s_by_row >= milestone_s]
        if crossed.empty:
            print(f"  {milestone_h:g}h: not found in stats rows")
            continue

        row_idx = int(crossed.index[0])
        route_count = int(route_count_by_row.loc[row_idx])

        if route_count > len(routes):
            print(
                f"  {milestone_h:g}h: inconsistent route count {route_count} > "
                f"{len(routes)}; skip and rerun after current iteration finishes"
            )
            continue

        prefix = routes[:route_count]
        stem = (
            f"routes_{int(milestone_h)}h_snapshot_{bus_label}.json"
            if canonical_names
            else f"routes_{int(milestone_h)}h_reconstructed_snapshot_{bus_label}.json"
        )
        out_path = run_dir / stem

        if out_path.exists() and not overwrite:
            print(f"  {milestone_h:g}h: exists {out_path.name}")
            continue

        next_master_obj = None
        if row_idx + 1 < len(df):
            next_master_obj = pd.to_numeric(
                pd.Series([df.iloc[row_idx + 1].get("Master_Obj")]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(next_master_obj):
                next_master_obj = None

        row = df.iloc[row_idx]
        payload = {
            "iteration": int(row.get("Iteration")),
            "milestone_hours": milestone_h,
            "active_time_hours": float(active_s_by_row.loc[row_idx]) / 3600.0,
            "active_time_s": float(active_s_by_row.loc[row_idx]),
            "cumulative_master_time_s": None,
            "cumulative_pricing_time_s": None,
            "master_obj_before_pricing": row.get("Master_Obj"),
            "master_obj_after_added_columns": next_master_obj,
            "cols_added_this_iteration": int(row.get("Cols_Added", 0)),
            "num_routes": len(prefix),
            "initial_route_count": initial_route_count,
            "csv_name": data.get("csv_name"),
            "bus_label": bus_label,
            "stats_csv_path": str(stats_path),
            "checkpoint_path": str(ckpt_path),
            "routes": prefix,
            "reconstructed_from_stats": True,
            "exact_route_prefix": True,
        }

        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, out_path)
        print(
            f"  {milestone_h:g}h: wrote {out_path.name} "
            f"(iter={payload['iteration']}, routes={len(prefix)}, "
            f"active={payload['active_time_hours']:.2f}h)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(Path.home() / "demandresponse" / "src" / "results"),
    )
    parser.add_argument(
        "--pattern",
        default="*/ckpt_latest_*stag999999*.json",
        help="Glob under results-dir selecting checkpoints to process.",
    )
    parser.add_argument(
        "--canonical-names",
        action="store_true",
        help="Write routes_3h_snapshot_*.json names instead of reconstructed names.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ckpts = sorted(glob.glob(str(Path(args.results_dir) / args.pattern)))
    print(f"Found {len(ckpts)} checkpoints.")
    for ckpt in ckpts:
        reconstruct_one(Path(ckpt), args.canonical_names, args.overwrite)


if __name__ == "__main__":
    main()

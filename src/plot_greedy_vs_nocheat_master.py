#!/usr/bin/env python3
"""
Compare master objective trajectories for GREEDY, NO_CHEAT, and CHEAT price-scenario runs.

Inputs are the local Downloads folders:
  - /Users/nadan/Downloads/evsp_final_results_greedy_stag999996
  - /Users/nadan/Downloads/evsp_final_results

Outputs:
  - output/greedy_vs_nocheat_master/master_log_by_instance_peak.png
  - output/greedy_vs_nocheat_master/master_log_median.png
  - output/greedy_vs_nocheat_master/master_log_distribution_bands.png
  - output/greedy_vs_nocheat_master/master_log_distribution_bands_zoom_8_10h.png
  - output/greedy_vs_nocheat_master/master_log_distribution_bands_all3.png
  - output/greedy_vs_nocheat_master/master_log_distribution_bands_all3_zoom_8_10h.png
  - output/greedy_vs_nocheat_master/master_log_distribution_bands_10B.png
  - output/greedy_vs_nocheat_master/master_log_distribution_bands_15B.png
  - output/greedy_vs_nocheat_master/master_log_distribution_bands_zoom_8_10h_10B.png
  - output/greedy_vs_nocheat_master/master_log_distribution_bands_zoom_8_10h_15B.png
  - output/greedy_vs_nocheat_master/master_log_distribution_bands_all3_10B.png
  - output/greedy_vs_nocheat_master/master_log_distribution_bands_all3_15B.png
  - output/greedy_vs_nocheat_master/master_log_distribution_bands_all3_zoom_8_10h_10B.png
  - output/greedy_vs_nocheat_master/master_log_distribution_bands_all3_zoom_8_10h_15B.png
  - output/greedy_vs_nocheat_master/matched_pricing_runs.csv
  - output/greedy_vs_nocheat_master/matched_pricing_runs_all3.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_GREEDY_ROOT = Path("/Users/nadan/Downloads/evsp_final_results_greedy_stag999996")
DEFAULT_NOCHEAT_ROOT = Path("/Users/nadan/Downloads/evsp_final_results")
DEFAULT_CHEAT_ROOT = Path("/Users/nadan/Downloads/evsp_final_results")
DEFAULT_OUTPUT_DIR = Path("/Users/nadan/Documents/projects/demandresponse/output/greedy_vs_nocheat_master")

RUN_RE = re.compile(
    r"Inst_(?P<fleet>10B|15B)_RND(?P<rnd>\d{3})_"
    r"(?P<mode>NO_CHEAT|GREEDY|CHEAT)_.*?_(?P<peak>peak\d{2})(?:_[^_]+)?_g300_"
)

PEAKS = ["peak08", "peak12", "peak18"]
PEAK_COLORS = {
    "peak08": "#4c78a8",
    "peak12": "#f58518",
    "peak18": "#54a24b",
}
MODE_COLORS = {
    "GREEDY": "#1f77b4",
    "NO_CHEAT": "#d62728",
    "CHEAT": "#2ca02c",
}


def discover_pricing_csvs(root: Path, expected_mode: str) -> dict[tuple[str, str, str], Path]:
    out: dict[tuple[str, str, str], Path] = {}
    for csv_path in sorted(root.glob("*/pricing_*.csv")):
        m = RUN_RE.search(csv_path.parent.name)
        if not m:
            continue
        meta = m.groupdict()
        if meta["mode"] != expected_mode:
            continue
        rnd_num = int(meta["rnd"])
        if rnd_num < 1 or rnd_num > 4:
            continue
        key = (meta["fleet"], meta["rnd"], meta["peak"])
        old = out.get(key)
        if old is None or csv_path.stat().st_mtime > old.stat().st_mtime:
            out[key] = csv_path
    return out


def load_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Iteration", "Master_Obj"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")

    if {"Cumulative_Master_Time_s", "Cumulative_Pricing_Time_s"} <= set(df.columns):
        seconds = df["Cumulative_Master_Time_s"].fillna(0) + df["Cumulative_Pricing_Time_s"].fillna(0)
    elif "Total_Runtime_s" in df.columns:
        seconds = df["Total_Runtime_s"].fillna(0)
    else:
        seconds = df["Iteration"].astype(float)

    out = pd.DataFrame(
        {
            "iteration": df["Iteration"].astype(float),
            "active_hours": seconds.astype(float) / 3600.0,
            "master_obj": pd.to_numeric(df["Master_Obj"], errors="coerce"),
        }
    ).dropna()
    out = out[out["master_obj"] > 0].copy()
    return out


def build_long_table(greedy: dict, nocheat: dict) -> pd.DataFrame:
    rows = []
    all_keys = sorted(set(greedy) | set(nocheat))
    for key in all_keys:
        fleet, rnd, peak = key
        for mode, mapping in [("GREEDY", greedy), ("NO_CHEAT", nocheat)]:
            path = mapping.get(key)
            if path is None:
                rows.append(
                    {
                        "fleet": fleet,
                        "rnd": rnd,
                        "peak": peak,
                        "mode": mode,
                        "path": None,
                        "status": "missing",
                        "final_active_hours": np.nan,
                        "final_master_obj": np.nan,
                        "iterations": 0,
                    }
                )
                continue
            curve = load_curve(path)
            rows.append(
                {
                    "fleet": fleet,
                    "rnd": rnd,
                    "peak": peak,
                    "mode": mode,
                    "path": str(path),
                    "status": "ok",
                    "final_active_hours": curve["active_hours"].max() if not curve.empty else np.nan,
                    "final_master_obj": curve["master_obj"].iloc[-1] if not curve.empty else np.nan,
                    "iterations": len(curve),
                }
            )
    return pd.DataFrame(rows)


def build_long_table_multi(mappings: dict[str, dict]) -> pd.DataFrame:
    rows = []
    all_keys = sorted(set().union(*(set(mapping) for mapping in mappings.values())))
    for key in all_keys:
        fleet, rnd, peak = key
        for mode, mapping in mappings.items():
            path = mapping.get(key)
            if path is None:
                rows.append(
                    {
                        "fleet": fleet,
                        "rnd": rnd,
                        "peak": peak,
                        "mode": mode,
                        "path": None,
                        "status": "missing",
                        "final_active_hours": np.nan,
                        "final_master_obj": np.nan,
                        "iterations": 0,
                    }
                )
                continue
            curve = load_curve(path)
            rows.append(
                {
                    "fleet": fleet,
                    "rnd": rnd,
                    "peak": peak,
                    "mode": mode,
                    "path": str(path),
                    "status": "ok",
                    "final_active_hours": curve["active_hours"].max() if not curve.empty else np.nan,
                    "final_master_obj": curve["master_obj"].iloc[-1] if not curve.empty else np.nan,
                    "iterations": len(curve),
                }
            )
    return pd.DataFrame(rows)


def plot_by_instance(greedy: dict, nocheat: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True, sharey=False)
    fleets = ["10B", "15B"]
    rnds = ["001", "002", "003", "004"]

    for r, fleet in enumerate(fleets):
        for c, rnd in enumerate(rnds):
            ax = axes[r, c]
            for peak in PEAKS:
                key = (fleet, rnd, peak)
                color = PEAK_COLORS[peak]
                if key in nocheat:
                    curve = load_curve(nocheat[key])
                    ax.plot(
                        curve["active_hours"],
                        curve["master_obj"],
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.75,
                    )
                if key in greedy:
                    curve = load_curve(greedy[key])
                    ax.plot(
                        curve["active_hours"],
                        curve["master_obj"],
                        color=color,
                        linestyle="-",
                        linewidth=1.7,
                        alpha=0.95,
                    )
            ax.set_title(f"{fleet} RND{rnd}")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.set_xlim(left=0)
            if c == 0:
                ax.set_ylabel("Master objective (log)")
            if r == 1:
                ax.set_xlabel("Active compute hours")

    peak_handles = [
        plt.Line2D([0], [0], color=PEAK_COLORS[p], lw=2, label=p)
        for p in PEAKS
    ]
    mode_handles = [
        plt.Line2D([0], [0], color="black", lw=2, linestyle="-", label="GREEDY"),
        plt.Line2D([0], [0], color="black", lw=2, linestyle="--", label="NO_CHEAT"),
    ]
    fig.suptitle("Master LP Objective: GREEDY vs NO_CHEAT", y=0.985, fontsize=16)
    fig.legend(
        handles=peak_handles + mode_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "master_log_by_instance_peak.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def interpolate_curves(mapping: dict, grid: np.ndarray) -> list[np.ndarray]:
    curves = []
    for path in mapping.values():
        c = load_curve(path).sort_values("active_hours")
        c = c.drop_duplicates("active_hours", keep="last")
        if len(c) < 2:
            continue
        max_x = c["active_hours"].max()
        valid_grid = grid[grid <= max_x]
        y = np.full_like(grid, np.nan, dtype=float)
        y[: len(valid_grid)] = np.interp(valid_grid, c["active_hours"], c["master_obj"])
        curves.append(y)
    return curves


def common_horizon_hours(*mappings: dict, cap: float = 12.0) -> float:
    max_hours = []
    for mapping in mappings:
        for path in mapping.values():
            c = load_curve(path)
            if not c.empty:
                max_hours.append(float(c["active_hours"].max()))
    if not max_hours:
        return cap
    return min(cap, min(max_hours))


def filter_by_fleet(mapping: dict, fleet: str) -> dict:
    return {key: path for key, path in mapping.items() if key[0] == fleet}


def plot_median(greedy: dict, nocheat: dict, out_dir: Path) -> Path:
    max_hour = common_horizon_hours(greedy, nocheat, cap=12.0)
    grid = np.linspace(0, max_hour, 241)
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, mapping, color in [
        ("GREEDY", greedy, "#1f77b4"),
        ("NO_CHEAT", nocheat, "#d62728"),
    ]:
        curves = interpolate_curves(mapping, grid)
        if not curves:
            continue
        arr = np.vstack(curves)
        med = np.nanmedian(arr, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        ax.plot(grid, med, color=color, lw=2.5, label=f"{label} median")
        ax.fill_between(grid, q25, q75, color=color, alpha=0.16, linewidth=0)

    ax.set_yscale("log")
    ax.set_xlim(0, max_hour)
    ax.set_xlabel("Active compute hours")
    ax.set_ylabel("Master objective (log)")
    ax.set_title("Master LP Objective Across Matched Price-Scenario Runs")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "master_log_median.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_distribution_bands_pair(
    greedy: dict,
    nocheat: dict,
    out_dir: Path,
    filename: str,
    title: str,
    start_hour: float = 0.0,
    end_hour: float = 12.0,
) -> Path:
    max_hour = common_horizon_hours(greedy, nocheat, cap=end_hour)
    if max_hour <= start_hour:
        raise ValueError(f"Not enough active-hour coverage for {start_hour}-{end_hour}h plot.")

    grid = np.linspace(start_hour, max_hour, 241)
    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    for label, mapping, color in [
        ("GREEDY", greedy, MODE_COLORS["GREEDY"]),
        ("NO_CHEAT", nocheat, MODE_COLORS["NO_CHEAT"]),
    ]:
        curves = interpolate_curves(mapping, grid)
        if not curves:
            continue
        arr = np.vstack(curves)

        for y in arr:
            ax.plot(grid, y, color=color, alpha=0.11, lw=0.85)

        q10 = np.nanpercentile(arr, 10, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q50 = np.nanpercentile(arr, 50, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        q90 = np.nanpercentile(arr, 90, axis=0)

        ax.fill_between(grid, q10, q90, color=color, alpha=0.09, linewidth=0)
        ax.fill_between(grid, q25, q75, color=color, alpha=0.20, linewidth=0)
        ax.plot(grid, q50, color=color, lw=2.9, label=f"{label} median")

    ax.set_yscale("log")
    ax.set_xlim(start_hour, max_hour)
    ax.set_xlabel("Active compute hours")
    ax.set_ylabel("Master objective (log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        0.03,
        "Faint lines = individual runs; dark band = 25-75%; light band = 10-90%",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / filename
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_distribution_bands(greedy: dict, nocheat: dict, out_dir: Path) -> Path:
    max_hour = common_horizon_hours(greedy, nocheat, cap=12.0)
    grid = np.linspace(0, max_hour, 241)
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for label, mapping, color in [
        ("GREEDY", greedy, "#1f77b4"),
        ("NO_CHEAT", nocheat, "#d62728"),
    ]:
        curves = interpolate_curves(mapping, grid)
        if not curves:
            continue
        arr = np.vstack(curves)

        for y in arr:
            ax.plot(grid, y, color=color, alpha=0.10, lw=0.8)

        q10 = np.nanpercentile(arr, 10, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q50 = np.nanpercentile(arr, 50, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        q90 = np.nanpercentile(arr, 90, axis=0)

        ax.fill_between(grid, q10, q90, color=color, alpha=0.10, linewidth=0)
        ax.fill_between(grid, q25, q75, color=color, alpha=0.22, linewidth=0)
        ax.plot(grid, q50, color=color, lw=2.8, label=f"{label} median")

    ax.set_yscale("log")
    ax.set_xlim(0, max_hour)
    ax.set_xlabel("Active compute hours")
    ax.set_ylabel("Master objective (log)")
    ax.set_title("Master LP Objective: GREEDY vs NO_CHEAT")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        0.03,
        "Faint lines = individual runs; dark band = 25-75%; light band = 10-90%",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "master_log_distribution_bands.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_distribution_bands_zoom(
    greedy: dict,
    nocheat: dict,
    out_dir: Path,
    start_hour: float = 8.0,
    end_hour: float = 10.0,
) -> Path:
    max_hour = common_horizon_hours(greedy, nocheat, cap=end_hour)
    if max_hour <= start_hour:
        raise ValueError(f"Not enough active-hour coverage for {start_hour}-{end_hour}h zoom.")

    grid = np.linspace(start_hour, max_hour, 241)
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for label, mapping, color in [
        ("GREEDY", greedy, "#1f77b4"),
        ("NO_CHEAT", nocheat, "#d62728"),
    ]:
        curves = interpolate_curves(mapping, grid)
        if not curves:
            continue
        arr = np.vstack(curves)

        for y in arr:
            ax.plot(grid, y, color=color, alpha=0.12, lw=0.9)

        q10 = np.nanpercentile(arr, 10, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q50 = np.nanpercentile(arr, 50, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        q90 = np.nanpercentile(arr, 90, axis=0)

        ax.fill_between(grid, q10, q90, color=color, alpha=0.10, linewidth=0)
        ax.fill_between(grid, q25, q75, color=color, alpha=0.24, linewidth=0)
        ax.plot(grid, q50, color=color, lw=3.0, label=f"{label} median")

    ax.set_yscale("log")
    ax.set_xlim(start_hour, max_hour)
    ax.set_xlabel("Active compute hours")
    ax.set_ylabel("Master objective (log)")
    ax.set_title(f"Master LP Objective Zoom: {start_hour:g}-{max_hour:g} Active Hours")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        0.03,
        "Faint lines = individual runs; dark band = 25-75%; light band = 10-90%",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "master_log_distribution_bands_zoom_8_10h.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_distribution_bands_multi(
    mappings: dict[str, dict],
    out_dir: Path,
    filename: str,
    title: str,
    start_hour: float = 0.0,
    end_hour: float = 12.0,
) -> Path:
    max_hour = common_horizon_hours(*mappings.values(), cap=end_hour)
    if max_hour <= start_hour:
        raise ValueError(f"Not enough active-hour coverage for {start_hour}-{end_hour}h plot.")

    grid = np.linspace(start_hour, max_hour, 241)
    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    for label, mapping in mappings.items():
        color = MODE_COLORS[label]
        curves = interpolate_curves(mapping, grid)
        if not curves:
            continue
        arr = np.vstack(curves)

        for y in arr:
            ax.plot(grid, y, color=color, alpha=0.09, lw=0.8)

        q10 = np.nanpercentile(arr, 10, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q50 = np.nanpercentile(arr, 50, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        q90 = np.nanpercentile(arr, 90, axis=0)

        ax.fill_between(grid, q10, q90, color=color, alpha=0.08, linewidth=0)
        ax.fill_between(grid, q25, q75, color=color, alpha=0.18, linewidth=0)
        ax.plot(grid, q50, color=color, lw=2.9, label=f"{label} median")

    ax.set_yscale("log")
    ax.set_xlim(start_hour, max_hour)
    ax.set_xlabel("Active compute hours")
    ax.set_ylabel("Master objective (log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        0.03,
        "Faint lines = individual runs; dark band = 25-75%; light band = 10-90%",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / filename
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-root", type=Path, default=DEFAULT_GREEDY_ROOT)
    parser.add_argument("--nocheat-root", type=Path, default=DEFAULT_NOCHEAT_ROOT)
    parser.add_argument("--cheat-root", type=Path, default=DEFAULT_CHEAT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    greedy = discover_pricing_csvs(args.greedy_root, "GREEDY")
    nocheat = discover_pricing_csvs(args.nocheat_root, "NO_CHEAT")
    cheat = discover_pricing_csvs(args.cheat_root, "CHEAT")
    matched = build_long_table(greedy, nocheat)
    mappings_all3 = {"GREEDY": greedy, "NO_CHEAT": nocheat, "CHEAT": cheat}
    matched_all3 = build_long_table_multi(mappings_all3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched_path = args.output_dir / "matched_pricing_runs.csv"
    matched_all3_path = args.output_dir / "matched_pricing_runs_all3.csv"
    matched.to_csv(matched_path, index=False)
    matched_all3.to_csv(matched_all3_path, index=False)
    by_instance_path = plot_by_instance(greedy, nocheat, args.output_dir)
    median_path = plot_median(greedy, nocheat, args.output_dir)
    bands_path = plot_distribution_bands(greedy, nocheat, args.output_dir)
    zoom_path = plot_distribution_bands_zoom(greedy, nocheat, args.output_dir)
    all3_path = plot_distribution_bands_multi(
        mappings_all3,
        args.output_dir,
        "master_log_distribution_bands_all3.png",
        "Master LP Objective: GREEDY vs NO_CHEAT vs CHEAT",
    )
    all3_zoom_path = plot_distribution_bands_multi(
        mappings_all3,
        args.output_dir,
        "master_log_distribution_bands_all3_zoom_8_10h.png",
        "Master LP Objective Zoom: GREEDY vs NO_CHEAT vs CHEAT",
        start_hour=8.0,
        end_hour=10.0,
    )
    fleet_paths = []
    for fleet in ["10B", "15B"]:
        greedy_f = filter_by_fleet(greedy, fleet)
        nocheat_f = filter_by_fleet(nocheat, fleet)
        cheat_f = filter_by_fleet(cheat, fleet)
        mappings_f = {"GREEDY": greedy_f, "NO_CHEAT": nocheat_f, "CHEAT": cheat_f}

        fleet_paths.append(
            plot_distribution_bands_pair(
                greedy_f,
                nocheat_f,
                args.output_dir,
                f"master_log_distribution_bands_{fleet}.png",
                f"{fleet} Master LP Objective: GREEDY vs NO_CHEAT",
            )
        )
        fleet_paths.append(
            plot_distribution_bands_pair(
                greedy_f,
                nocheat_f,
                args.output_dir,
                f"master_log_distribution_bands_zoom_8_10h_{fleet}.png",
                f"{fleet} Master LP Objective Zoom: GREEDY vs NO_CHEAT",
                start_hour=8.0,
                end_hour=10.0,
            )
        )
        fleet_paths.append(
            plot_distribution_bands_multi(
                mappings_f,
                args.output_dir,
                f"master_log_distribution_bands_all3_{fleet}.png",
                f"{fleet} Master LP Objective: GREEDY vs NO_CHEAT vs CHEAT",
            )
        )
        fleet_paths.append(
            plot_distribution_bands_multi(
                mappings_f,
                args.output_dir,
                f"master_log_distribution_bands_all3_zoom_8_10h_{fleet}.png",
                f"{fleet} Master LP Objective Zoom: GREEDY vs NO_CHEAT vs CHEAT",
                start_hour=8.0,
                end_hour=10.0,
            )
        )

    n_g = len(greedy)
    n_n = len(nocheat)
    n_c = len(cheat)
    common = len(set(greedy) & set(nocheat))
    common_all3 = len(set(greedy) & set(nocheat) & set(cheat))
    print(f"GREEDY pricing CSVs : {n_g}")
    print(f"NO_CHEAT pricing CSVs: {n_n}")
    print(f"CHEAT pricing CSVs  : {n_c}")
    print(f"Matched keys, 2-way : {common}")
    print(f"Matched keys, all 3 : {common_all3}")
    print(f"Wrote: {matched_path}")
    print(f"Wrote: {matched_all3_path}")
    print(f"Wrote: {by_instance_path}")
    print(f"Wrote: {median_path}")
    print(f"Wrote: {bands_path}")
    print(f"Wrote: {zoom_path}")
    print(f"Wrote: {all3_path}")
    print(f"Wrote: {all3_zoom_path}")
    for path in fleet_paths:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()

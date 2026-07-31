from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CLEAN_ROOT = Path("/Users/nadan/Downloads/demandresponse_csvs_clean")
DP_ROOT = Path("/Users/nadan/Downloads/demandresponse_csvs_dp")
OUT_DIR = Path("/Users/nadan/Downloads/demandresponse_10B_dp_vs_clean_plots")

HOUR_LIMIT = 3.0


def infer_key(run_dir: Path) -> tuple[str, str] | None:
    """Return (RNDxxx, CHEAT/NO_CHEAT) from a run folder name."""
    name = run_dir.name
    if "Inst_10B" not in name:
        return None

    rnd_match = re.search(r"RND\d{3}", name)
    if not rnd_match:
        return None
    rnd = rnd_match.group(0)

    if "NO_CHEAT" in name:
        mode = "NO_CHEAT"
    elif "CHEAT" in name:
        mode = "CHEAT"
    else:
        # Older clean runs were unlabeled; those were no-cheat.
        mode = "NO_CHEAT"

    return rnd, mode


def find_pricing_csv(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("pricing_*.csv"))
    if not candidates:
        return None

    # Prefer the normal pricing stats file if there are multiple.
    normal = [p for p in candidates if "instrumented" not in p.name]
    return normal[0] if normal else candidates[0]


def load_one(csv_path: Path, algo: str, rnd: str, mode: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Algo"] = algo
    df["RND"] = rnd
    df["Mode"] = mode
    df["RunDir"] = csv_path.parent.name
    df["CSV"] = str(csv_path)

    numeric_cols = [
        "Iteration",
        "Master_Obj",
        "Master_Improvement",
        "Master_Time_s",
        "Pricing_Time_s",
        "Cumulative_Master_Time_s",
        "Cumulative_Pricing_Time_s",
        "Cols_Added",
        "Best_RC",
        "Pricing_Labels_Used",
        "Highest_Tier_Reached",
        "Total_Runtime_s",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"Cumulative_Master_Time_s", "Cumulative_Pricing_Time_s"} <= set(df.columns):
        df["Active_Time_s"] = df["Cumulative_Master_Time_s"] + df["Cumulative_Pricing_Time_s"]
    elif "Total_Runtime_s" in df.columns:
        df["Active_Time_s"] = df["Total_Runtime_s"]
    else:
        raise ValueError(f"No cumulative time columns in {csv_path}")

    df["Active_Hours"] = df["Active_Time_s"] / 3600.0
    df = df.sort_values("Active_Hours").reset_index(drop=True)
    return df


def load_all(root: Path, algo: str) -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        key = infer_key(run_dir)
        if key is None:
            continue

        csv_path = find_pricing_csv(run_dir)
        if csv_path is None:
            print(f"[WARN] no pricing CSV in {run_dir}")
            continue

        rnd, mode = key
        df = load_one(csv_path, algo=algo, rnd=rnd, mode=mode)

        # If duplicate keys exist, keep the one with more active time.
        old = out.get(key)
        if old is None or df["Active_Hours"].max() > old["Active_Hours"].max():
            out[key] = df

    return out


def truncate_at_hours(df: pd.DataFrame, hours: float) -> pd.DataFrame:
    """Keep rows up to the limit plus the first crossing row, clipped to x=hours."""
    if df.empty:
        return df.copy()

    before = df[df["Active_Hours"] <= hours]
    after = df[df["Active_Hours"] > hours]

    if after.empty:
        return before.copy()

    first_cross = after.iloc[[0]].copy()
    first_cross["Active_Hours"] = hours
    return pd.concat([before, first_cross], ignore_index=True)


def plot_metric(ax, frames: dict[str, pd.DataFrame], ycol: str, title: str, ylabel: str):
    for label, df in frames.items():
        if ycol not in df.columns:
            continue
        use = df.dropna(subset=["Active_Hours", ycol])
        if use.empty:
            continue
        ax.plot(use["Active_Hours"], use[ycol], marker=".", linewidth=1.4, markersize=3, label=label)

    ax.set_title(title)
    ax.set_xlabel("Active compute hours")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, HOUR_LIMIT)


def plot_one_comparison(key: tuple[str, str], clean_df: pd.DataFrame, dp_df: pd.DataFrame):
    rnd, mode = key
    clean_3h = truncate_at_hours(clean_df, HOUR_LIMIT)
    dp_3h = truncate_at_hours(dp_df, HOUR_LIMIT)
    frames = {"clean": clean_3h, "dp": dp_3h}

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Inst_10B_{rnd}_{mode}: clean vs dp, first {HOUR_LIMIT:g} active hours", fontsize=15)

    plot_metric(axes[0, 0], frames, "Master_Obj", "Master LP objective", "Master_Obj")
    plot_metric(axes[0, 1], frames, "Best_RC", "Most negative pricing reduced cost", "Best_RC")
    plot_metric(axes[0, 2], frames, "Cols_Added", "Columns added per iteration", "Cols_Added")
    plot_metric(axes[1, 0], frames, "Pricing_Time_s", "Pricing time per iteration", "Pricing_Time_s")
    plot_metric(axes[1, 1], frames, "Highest_Tier_Reached", "Highest pricing tier reached", "Tier")

    ax = axes[1, 2]
    for label, df in frames.items():
        if "Best_RC" not in df.columns:
            continue
        vals = df["Best_RC"].dropna()
        if vals.empty:
            continue
        ax.hist(vals, bins=30, alpha=0.45, label=label)
    ax.set_title("Distribution of Best_RC")
    ax.set_xlabel("Best_RC")
    ax.set_ylabel("Iteration count")
    ax.grid(True, alpha=0.25)
    ax.legend()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / f"compare_10B_{rnd}_{mode}_first_{int(HOUR_LIMIT)}h.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def final_snapshot(df: pd.DataFrame, algo: str) -> dict:
    d3 = truncate_at_hours(df, HOUR_LIMIT)
    last = d3.iloc[-1]
    return {
        f"{algo}_iters": int(last["Iteration"]) if pd.notna(last.get("Iteration")) else None,
        f"{algo}_active_h": float(last["Active_Hours"]),
        f"{algo}_master_obj": float(last["Master_Obj"]) if pd.notna(last.get("Master_Obj")) else None,
        f"{algo}_best_rc": float(last["Best_RC"]) if pd.notna(last.get("Best_RC")) else None,
        f"{algo}_routes_added": int(d3["Cols_Added"].fillna(0).sum()) if "Cols_Added" in d3 else None,
        f"{algo}_pricing_time_s": float(d3["Pricing_Time_s"].fillna(0).sum()) if "Pricing_Time_s" in d3 else None,
    }


def plot_summary(all_clean: dict[tuple[str, str], pd.DataFrame], all_dp: dict[tuple[str, str], pd.DataFrame]):
    for mode in ["CHEAT", "NO_CHEAT"]:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
        fig.suptitle(f"10B {mode}: all instances, first {HOUR_LIMIT:g} active hours")

        for rnd in [f"RND{i:03d}" for i in range(1, 7)]:
            key = (rnd, mode)
            if key in all_clean:
                df = truncate_at_hours(all_clean[key], HOUR_LIMIT)
                axes[0].plot(df["Active_Hours"], df["Master_Obj"], linewidth=1.2, label=rnd)
            if key in all_dp:
                df = truncate_at_hours(all_dp[key], HOUR_LIMIT)
                axes[1].plot(df["Active_Hours"], df["Master_Obj"], linewidth=1.2, label=rnd)

        for ax, title in zip(axes, ["clean", "dp"]):
            ax.set_title(title)
            ax.set_xlabel("Active compute hours")
            ax.set_ylabel("Master_Obj")
            ax.set_xlim(0, HOUR_LIMIT)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = OUT_DIR / f"summary_10B_{mode}_master_curves_first_{int(HOUR_LIMIT)}h.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_clean = load_all(CLEAN_ROOT, "clean")
    all_dp = load_all(DP_ROOT, "dp")

    wanted = [(f"RND{i:03d}", mode) for i in range(1, 7) for mode in ["CHEAT", "NO_CHEAT"]]

    rows = []
    made = []
    for key in wanted:
        clean_df = all_clean.get(key)
        dp_df = all_dp.get(key)

        if clean_df is None or dp_df is None:
            print(f"[MISSING] {key}: clean={clean_df is not None}, dp={dp_df is not None}")
            continue

        made.append(plot_one_comparison(key, clean_df, dp_df))

        row = {"RND": key[0], "Mode": key[1]}
        row.update(final_snapshot(clean_df, "clean"))
        row.update(final_snapshot(dp_df, "dp"))
        if row["clean_master_obj"] is not None and row["dp_master_obj"] is not None:
            row["dp_minus_clean_master_obj"] = row["dp_master_obj"] - row["clean_master_obj"]
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / f"summary_first_{int(HOUR_LIMIT)}h.csv"
    summary.to_csv(summary_path, index=False)

    plot_summary(all_clean, all_dp)

    print(f"\nWrote {len(made)} instance comparison plots to:")
    print(f"  {OUT_DIR}")
    print(f"\nSummary CSV:")
    print(f"  {summary_path}")
    if not summary.empty:
        print("\n3h summary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

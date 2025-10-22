import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Files
results_dir = Path(__file__).resolve().parent / "results"

summary_files = sorted(results_dir.glob("summary_*.csv"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
iters_files   = sorted(results_dir.glob("iterations_*.csv"),
                       key=lambda p: p.stat().st_mtime, reverse=True)

if summary_files:
    m = re.search(r"summary_(\d{8}_\d{6}_\d+)\.csv$", summary_files[0].name)
    if not m:
        raise RuntimeError(f"Could not parse RUN_ID from {summary_files[0].name}")
    RUN_ID = m.group(1)
else:
    if not iters_files:
        raise FileNotFoundError(f"No summary_*.csv or iterations_*.csv in {results_dir}")
    m = re.search(r"iterations_(\d{8}_\d{6}_\d+)\.csv$", iters_files[0].name)
    if not m:
        raise RuntimeError(f"Could not parse RUN_ID from {iters_files[0].name}")
    RUN_ID = m.group(1)

summary_path  = results_dir / f"summary_{RUN_ID}.csv"
iters_path    = results_dir / f"iterations_{RUN_ID}.csv"
timeline_path = results_dir / f"timeline_{RUN_ID}.csv"

print(f"[plot] results_dir = {results_dir}")
print(f"[plot] RUN_ID      = {RUN_ID}")

# Load
df_summary  = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
df_iters    = pd.read_csv(iters_path) if iters_path.exists() else pd.DataFrame()
df_timeline = pd.read_csv(timeline_path) if timeline_path.exists() else pd.DataFrame()

# Output dir
outdir = results_dir / "plots" / RUN_ID
outdir.mkdir(parents=True, exist_ok=True)

# ---------------- 1) Metrics vs Solar (single panel, lines by mode) ----------------
if not df_summary.empty:
    # Sort by solar multiplier
    df_summary = df_summary.sort_values("solar_mult")

    # Provide a compact subtitle from first row (same across rows in a run)
    subtitle_bits = []
    if "n_trips" in df_summary.columns and pd.notna(df_summary["n_trips"].iloc[0]):
        subtitle_bits.append(f"{int(df_summary['n_trips'].iloc[0])} trips")
    if "n_stations" in df_summary.columns and pd.notna(df_summary["n_stations"].iloc[0]):
        subtitle_bits.append(f"{int(df_summary['n_stations'].iloc[0])} stations")
    subtitle = " | ".join(subtitle_bits) if subtitle_bits else ""

    for metric, ylabel, fname, title in [
        ("trucks", "Number of Vehicles", "trucks_vs_solar.png", "Vehicles vs Solar Multiplier"),
        ("fuel_used", "Net Energy (kWh)", "energy_vs_solar.png", "Net Energy vs Solar Multiplier"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        if "mode" in df_summary.columns:
            for mode_name, grp in df_summary.groupby("mode"):
                ax.plot(grp["solar_mult"], grp[metric], marker="o", label=str(mode_name))
            ax.legend(title="Mode")
        else:
            ax.plot(df_summary["solar_mult"], df_summary[metric], marker="o")

        ax.set_xlabel("Solar Multiplier")
        ax.set_ylabel(ylabel)
        ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(outdir / fname, dpi=200)
        plt.close(fig)

# ---------------- 2) Iteration timing (simple scatter) ----------------
if not df_iters.empty and {"iteration", "pricing_time_s"}.issubset(df_iters.columns):
    df_iters = df_iters.sort_values("iteration")
    fig, ax = plt.subplots(figsize=(8, 5))

    if "master_time_s" in df_iters.columns:
        ax.scatter(df_iters["iteration"], df_iters["master_time_s"], label="Master", s=20)

    if "pricing_mode" in df_iters.columns:
        mode = df_iters["pricing_mode"].fillna("").str.lower()
        fast  = mode == "fast"
        exact = mode == "exact"
        if fast.any():
            ax.scatter(df_iters.loc[fast, "iteration"], df_iters.loc[fast, "pricing_time_s"],
                       label="Pricing (fast)", s=20)
        if exact.any():
            ax.scatter(df_iters.loc[exact, "iteration"], df_iters.loc[exact, "pricing_time_s"],
                       label="Pricing (exact)", s=20)
        if not fast.any() and not exact.any():
            ax.scatter(df_iters["iteration"], df_iters["pricing_time_s"], label="Pricing", s=20)
    else:
        ax.scatter(df_iters["iteration"], df_iters["pricing_time_s"], label="Pricing", s=20)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time (s)")
    ax.set_title("Iteration Timing")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(outdir / "iteration_timing.png", dpi=200)
    plt.close(fig)

# ---------------- 3) Timeline (Gantt-like, compact) ----------------
need = {"route_label", "event_type", "time_block"}
if not df_timeline.empty and need.issubset(df_timeline.columns):
    # Order by appearance
    order = pd.Categorical(df_timeline["route_label"],
                           categories=pd.unique(df_timeline["route_label"]),
                           ordered=True)
    df_timeline = df_timeline.assign(_ord=order).sort_values(["_ord", "time_block"])

    color_map = {
        "paid_charge": "red",
        "free_charge": "green",
        "v2v_discharge": "cyan",
        "v2g_discharge": "blue",
    }

    labels = list(order.categories)
    fig, ax = plt.subplots(figsize=(10, max(2, int(len(labels) * 0.4))))
    for idx, lab in enumerate(labels):
        sub = df_timeline[df_timeline["route_label"] == lab]
        for _, row in sub.iterrows():
            ax.barh(idx, 1, left=row["time_block"], height=0.8,
                    color=color_map.get(row["event_type"], "gray"), alpha=0.7)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time Block")
    ax.set_title("Charging / Discharging Timeline")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Legend
    handles = [plt.Rectangle((0,0), 1, 1, color=c, alpha=0.7, label=lbl)
               for lbl, c in [("Paid charge", "red"), ("Free charge", "green"),
                              ("V2V discharge", "cyan"), ("V2G discharge", "blue")]]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(outdir / "timeline_gantt.png", dpi=200)
    plt.close(fig)

print(f"Saved plots to: {outdir}")

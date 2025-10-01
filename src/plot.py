import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Always resolve results/ relative to this file (…/src/results)
results_dir = Path(__file__).resolve().parent / "results"

# --- pick latest run id by newest summary_* file ---
summary_files = sorted(
    results_dir.glob("summary_*.csv"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not summary_files:
    raise FileNotFoundError(f"No summary_*.csv in {results_dir}")

m = re.search(r"summary_(\d{8}_\d{6}_\d+)\.csv$", summary_files[0].name)
if not m:
    raise RuntimeError(f"Could not parse RUN_ID from {summary_files[0].name}")
RUN_ID = m.group(1)

summary_path  = results_dir / f"summary_{RUN_ID}.csv"
iters_path    = results_dir / f"iterations_{RUN_ID}.csv"
timeline_path = results_dir / f"timeline_{RUN_ID}.csv"

print(f"[plot] results_dir = {results_dir}")
print(f"[plot] using RUN_ID  = {RUN_ID}")
print(f"[plot] summary       = {summary_path.name}")
print(f"[plot] iterations    = {iters_path.name}")
print(f"[plot] timeline      = {timeline_path.name}")

# Load data
df_summary  = pd.read_csv(summary_path)
df_iters    = pd.read_csv(iters_path)
df_timeline = pd.read_csv(timeline_path)

# Output folder for figures
outdir = results_dir / "plots" / RUN_ID
outdir.mkdir(parents=True, exist_ok=True)

# ---------------- 1) Metrics plots (vehicles & energy) ----------------
if not df_summary.empty:
    points_levels = sorted(df_summary['points'].dropna().unique().tolist())
    metrics = ['trucks', 'fuel_used']
    ylabels = {'trucks': 'Number of Vehicles', 'fuel_used': 'Net Energy (kWh)'}

    for metric in metrics:
        ncols = max(1, len(points_levels))
        fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4), sharey=True)
        if ncols == 1:
            axes = [axes]

        for ax, pts in zip(axes, points_levels):
            sub = df_summary[df_summary['points'] == pts]
            eps_vals = sorted(sub['epsilon'].dropna().unique().tolist())
            for eps in eps_vals:
                grp = sub[sub['epsilon'] == eps].sort_values('solar_mult')
                if grp.empty:
                    continue
                ax.plot(grp['solar_mult'], grp[metric],
                        marker='o', linestyle='-', label=f'ε={eps}')
            ax.set_title(f'{pts} Locations')
            ax.set_xlabel('Solar Multiplier')
            ax.set_ylabel(ylabels[metric])
            ax.grid(True, linestyle='--', alpha=0.5)
            if eps_vals:
                ax.legend(title='ε')

        fig.suptitle(ylabels[metric], fontsize=14)
        plt.tight_layout(rect=[0,0,1,0.95])
        fig.savefig(outdir / f"{metric}_vs_solar_mult.png", dpi=200)
        plt.close(fig)

# ---------------- 2) Iteration timing scatter ----------------
if not df_iters.empty:
    df_iters = df_iters.sort_values('iteration')
    iters = df_iters['iteration'].tolist()

    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(iters, df_iters['master_time_s'], marker='o', label='Master Time')

    mode = df_iters['pricing_mode'] if 'pricing_mode' in df_iters.columns else None
    if mode is not None:
        fast  = mode.fillna('').str.lower() == 'fast'
        exact = mode.fillna('').str.lower() == 'exact'
        if fast.any():
            ax.scatter(df_iters.loc[fast,'iteration'], df_iters.loc[fast,'pricing_time_s'],
                       marker='o', label='Pricing (fast)')
        if exact.any():
            ax.scatter(df_iters.loc[exact,'iteration'], df_iters.loc[exact,'pricing_time_s'],
                       marker='o', label='Pricing (exact)')
        if not fast.any() and not exact.any():
            ax.scatter(iters, df_iters['pricing_time_s'], marker='o', label='Pricing')
    else:
        ax.scatter(iters, df_iters['pricing_time_s'], marker='o', label='Pricing')

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time (s)")
    ax.set_title("Iteration Timing")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(outdir / "iteration_timing.png", dpi=200)
    plt.close(fig)

# ---------------- 3) Timeline (Gantt-like) ----------------
need = {"route_label","event_type","time_block"}
if (not df_timeline.empty) and need.issubset(df_timeline.columns):
    order = pd.Categorical(
        df_timeline['route_label'],
        categories=pd.unique(df_timeline['route_label']),
        ordered=True
    )
    df_timeline = df_timeline.assign(_ord=order).sort_values(['_ord','time_block'])
    colors = {'paid_charge':'red','free_charge':'green',
              'v2v_discharge':'cyan','v2g_discharge':'blue'}
    labels = list(order.categories)
    n = len(labels)
    height = 0.8
    times = sorted(df_timeline['time_block'].unique().tolist())

    fig, ax = plt.subplots(figsize=(10, max(1, int(n*0.4))))
    for idx, lab in enumerate(labels):
        sub = df_timeline[df_timeline['route_label'] == lab]
        for _, row in sub.iterrows():
            ax.barh(idx, 1, left=row['time_block'], height=height,
                    color=colors.get(row['event_type'],'gray'), alpha=0.7)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(times)
    ax.set_xlabel("Time Block")
    ax.set_title("Charging/Discharging Timeline")
    legend_handles = [plt.Rectangle((0,0),1,1,color=c,alpha=0.7,label=l)
                      for l,c in [('Paid charge','red'),('Free charge','green'),
                                  ('V2V discharge','cyan'),('V2G discharge','blue')]]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02,1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(outdir / "timeline_gantt.png", dpi=200)
    plt.close(fig)

print(f"Saved plots to: {outdir}")

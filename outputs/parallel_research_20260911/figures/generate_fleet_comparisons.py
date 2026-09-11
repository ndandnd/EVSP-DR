#!/usr/bin/env python3
"""Generate current fresh partition/cover and matched fresh/warm cover figures.

All observations are read from the saved experiment register and the checked
warm-chain status tables. Missing covering cells are deliberately left blank.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/parallel_research_20260911/figures"
OUT.mkdir(parents=True, exist_ok=True)
REGISTER = ROOT / "outputs/research_register/register.csv"
SNAPSHOT = ROOT / "outputs/post_meeting_20260910/monitor/20260911T020803Z.json"
WARM_CG = ROOT / "outputs/post_meeting_20260910/warm_chain_status/cg_status.csv"
WARM_MIP = ROOT / "outputs/post_meeting_20260910/warm_chain_status/mip_status.csv"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10, "figure.dpi": 160, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def boolish(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite_float(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def status_code(row: pd.Series) -> str:
    status = str(row.get("mip_status", "")).strip().upper()
    scope = str(row.get("optimal_scope", "")).strip().lower()
    if status == "OPTIMAL" and scope == "full_pool_lexicographic":
        return "OPT"
    if boolish(row.get("fleet_proven", False)):
        return "F"
    return "~"


register = pd.read_csv(REGISTER, low_memory=False)
register["target_k_num"] = pd.to_numeric(register["target_k"], errors="coerce")
register["chain_num"] = pd.to_numeric(register["chain"], errors="coerce")


def mip_rows(campaign: str) -> pd.DataFrame:
    x = register[
        (register["campaign_id"] == campaign)
        & (register["stage"].astype(str).str.lower() == "mip")
        & (register["artifact_status"].astype(str).str.lower() == "result")
        & register["target_k_num"].isin([5, 8, 10])
        & register["chain_num"].isin([1, 2, 3, 4, 5, 6])
    ].copy()
    x = x.sort_values(["target_k_num", "chain_num", "source_path"])
    return x.drop_duplicates(["target_k_num", "chain_num"], keep="first")

partition = mip_rows("nested84")
cover = mip_rows("covering_rerun9")

# Export exact source rows, including explicit missing covering cells.
matrix_rows = []
for formulation, x in [("partition", partition), ("cover", cover)]:
    lookup = {(int(r.target_k_num), int(r.chain_num)): r for r in x.itertuples()}
    for k in [5, 8, 10]:
        for chain in [1, 2, 3, 4, 5, 6]:
            r = lookup.get((k, chain))
            if r is None:
                matrix_rows.append({
                    "formulation": formulation, "target_k": k, "chain": chain,
                    "available": False, "fleet": np.nan, "fleet_excess": np.nan,
                    "status_code": "MISSING", "mip_status": "", "fleet_proven": "",
                    "optimal_scope": "", "mip_bound": np.nan, "overcovered_trips": np.nan,
                    "input_sha256": "", "source_path": "",
                })
            else:
                fleet = finite_float(r.mip_incumbent_fleet)
                bound = finite_float(r.mip_bound_fleet)
                matrix_rows.append({
                    "formulation": formulation, "target_k": k, "chain": chain,
                    "available": True, "fleet": fleet, "fleet_excess": fleet-k,
                    "status_code": status_code(pd.Series(r._asdict())),
                    "mip_status": r.mip_status, "fleet_proven": r.fleet_proven,
                    "optimal_scope": r.optimal_scope, "mip_bound": bound,
                    "overcovered_trips": finite_float(r.overcovered_trips),
                    "input_sha256": r.input_sha256, "source_path": r.source_path,
                })
matrix_df = pd.DataFrame(matrix_rows)
matrix_df.to_csv(OUT / "fresh_fleet_matrix.csv", index=False)

# ---- Figure 1: paired matrix -----------------------------------------------
colors = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027", "#7f0000"]
cmap = ListedColormap(colors)
norm = BoundaryNorm([-0.5, 0.5, 2.5, 5.5, 10.5, 20.5, 100.0], cmap.N)
fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), sharex=True, sharey=True)
fig.subplots_adjust(left=0.10, right=0.98, top=0.76, bottom=0.16, wspace=0.16)

for ax, formulation, title in zip(
    axes, ["partition", "cover"], ["Fresh partition master", "Fresh covering master"]
):
    sub = matrix_df[matrix_df.formulation == formulation]
    values = np.full((6, 3), np.nan)
    for _, r in sub.iterrows():
        if r.available:
            values[int(r.chain)-1, [5, 8, 10].index(int(r.target_k))] = r.fleet_excess
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm,
              aspect="auto", interpolation="none")
    missing = np.ma.masked_where(~np.isnan(values), np.ones_like(values))
    ax.imshow(missing, cmap=ListedColormap(["#d9d9d9"]), vmin=0, vmax=1,
              aspect="auto", interpolation="none", alpha=1.0)
    ax.set_title(title, fontweight="bold", pad=5)
    ax.set_xticks(range(3), ["k=5", "k=8", "k=10"])
    ax.set_yticks(range(6), [f"chain {i}" for i in range(1, 7)])
    ax.set_xlabel("Target number of buses, k")
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(5.5, -0.5)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for _, r in sub.iterrows():
        x = [5, 8, 10].index(int(r.target_k)); y = int(r.chain)-1
        if not r.available:
            label, color = "—", "#555555"
        else:
            label, color = f"{int(round(r.fleet))}\n{r.status_code}", "#111111"
        ax.text(x, y, label, ha="center", va="center", fontsize=11,
                fontweight="bold" if r.available and r.status_code in {"OPT", "F"} else "normal",
                color=color, linespacing=1.05)

axes[0].set_ylabel("Random nested chain")
legend = [
    Patch(facecolor=colors[0], label="0 excess"), Patch(facecolor=colors[1], label="1–2 excess"),
    Patch(facecolor=colors[2], label="3–5 excess"), Patch(facecolor=colors[3], label="6–10 excess"),
    Patch(facecolor=colors[4], label="11–20 excess"), Patch(facecolor=colors[5], label=">20 excess"),
    Patch(facecolor="#d9d9d9", edgecolor="#888888", label="covering pending (810454)"),
]
fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.93),
           ncol=4, frameon=False, columnspacing=1.2, handlelength=1.2)
fig.savefig(OUT / "fresh_partition_vs_cover_fleet_matrix.png", bbox_inches="tight")
fig.savefig(OUT / "fresh_partition_vs_cover_fleet_matrix.pdf", bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: fresh vs inherited covering on exact matched chain 3 --------
cg = pd.read_csv(WARM_CG)
mip = pd.read_csv(WARM_MIP)
ks = [5, 8, 10]
cg_rows = []
for k in ks:
    f = cg[(cg.arm == "fresh") & (cg.chain == 3) & (cg.k == k)]
    w = cg[(cg.arm == "warm_p3") & (cg.chain == 3) & (cg.k == k)]
    if len(f) != 1 or len(w) != 1:
        raise RuntimeError(f"expected one fresh and warm CG row for p3 k={k}")
    f = f.iloc[0]; w = w.iloc[0]
    rf = register[(register.campaign_id == "covering_rerun9") & (register.stage == "cg") &
                  (register.chain_num == 3) & (register.target_k_num == k)].iloc[0]
    rw = register[(register.campaign_id == "warm_chain_p3_k2_10") & (register.stage == "cg") &
                  (register.chain_num == 3) & (register.target_k_num == k)].iloc[0]
    if str(rf.input_sha256) != str(rw.input_sha256):
        raise RuntimeError(f"input SHA mismatch at p3 k={k}")
    mf = mip[(mip.arm == "fresh") & (mip.chain == 3) & (mip.k == k)].iloc[0]
    mw = mip[(mip.arm == "warm_p3") & (mip.chain == 3) & (mip.k == k)].iloc[0]
    total = float(w.wall_min); import_min = float(w.import_min)
    cg_rows.append({
        "chain": 3, "target_k": k, "trip_count": int(f.trips),
        "input_sha256": str(rf.input_sha256),
        "fresh_cg_total_min": float(f.wall_min), "fresh_import_min": float(f.import_min),
        "fresh_post_import_min": float(f.wall_min - f.import_min),
        "warm_cg_total_min": total, "warm_import_min": import_min,
        "warm_post_import_min": total - import_min,
        "fresh_pool_columns": int(f.pool_columns), "warm_pool_columns": int(w.pool_columns),
        "fresh_cg_iterations": int(f.cg_iterations), "warm_cg_iterations": int(w.cg_iterations),
        "fresh_lp_route_weight": float(f.route_weight), "warm_lp_route_weight": float(w.route_weight),
        "fresh_mip_buses": int(mf.buses), "warm_mip_buses": int(mw.buses),
        "fresh_mip_bound": float(mf.fleet_bound), "warm_mip_bound": float(mw.fleet_bound),
        "fresh_fleet_proven": bool(mf.fleet_proven), "warm_fleet_proven": bool(mw.fleet_proven),
        "fresh_mip_status": str(mf.status), "warm_mip_status": str(mw.status),
        "fresh_optimal_scope": str(mf.optimal_scope), "warm_optimal_scope": str(mw.optimal_scope),
        "fresh_overcovered_trips": int(mf.overcovered_trips), "warm_overcovered_trips": int(mw.overcovered_trips),
        "fresh_cg_source": str(f.source_json), "warm_cg_source": str(w.source_json),
        "fresh_mip_source": str(mf.source_json), "warm_mip_source": str(mw.source_json),
    })
matched = pd.DataFrame(cg_rows)
matched.to_csv(OUT / "fresh_vs_warm_cover_chain3.csv", index=False)

fig, (ax_time, ax_fleet) = plt.subplots(2, 1, figsize=(12.8, 8.4), sharex=True,
                                        gridspec_kw={"height_ratios": [1.15, 1]})
fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.14, hspace=0.32)
x = np.arange(len(ks)); width = 0.28
ax_time.bar(x - width/2, matched.fresh_cg_total_min, width=width,
            color="#377eb8", label="Fresh covering CG: total")
ax_time.bar(x + width/2, matched.warm_import_min, width=width,
            color="#e69f00", label="Warm CG: inherited-pool import")
ax_time.bar(x + width/2, matched.warm_post_import_min, width=width,
            bottom=matched.warm_import_min, color="#56b4e9", label="Warm CG: after import")
for i, r in matched.iterrows():
    ymax = max(matched.warm_cg_total_min)
    ax_time.text(x[i]-width/2, r.fresh_cg_total_min+ymax*0.025, f"{r.fresh_cg_total_min:.1f}",
                 ha="center", va="bottom", fontsize=10, color="#1f4f7a")
    ax_time.text(x[i]+width/2, r.warm_cg_total_min+ymax*0.025, f"{r.warm_cg_total_min:.1f}",
                 ha="center", va="bottom", fontsize=10, color="#245b73")
    ax_time.text(x[i]+width/2, r.warm_import_min/2, f"{r.warm_import_min:.1f}\nimport",
                 ha="center", va="center", fontsize=8.5, color="#4b3300")
ax_time.set_ylabel("CG wall time (minutes)")
ax_time.set_title("Matched chain 3: fresh versus previous-k inherited covering", fontweight="bold", pad=10)
ax_time.grid(axis="y", color="#dddddd", linewidth=0.8); ax_time.set_axisbelow(True)
ax_time.legend(loc="upper left", ncol=3, frameon=False)
ax_fleet.bar(x - width/2, matched.fresh_mip_buses, width=width, color="#377eb8", label="Fresh pool MIP")
ax_fleet.bar(x + width/2, matched.warm_mip_buses, width=width, color="#56b4e9", label="Inherited pool MIP")
ax_fleet.plot(x, ks, color="#222222", marker="_", markersize=18, linewidth=1.4, label="Target k")
for i, r in matched.iterrows():
    fs = status_code(pd.Series({"mip_status": r.fresh_mip_status,
                                 "fleet_proven": r.fresh_fleet_proven,
                                 "optimal_scope": r.fresh_optimal_scope}))
    ws = status_code(pd.Series({"mip_status": r.warm_mip_status,
                                 "fleet_proven": r.warm_fleet_proven,
                                 "optimal_scope": r.warm_optimal_scope}))
    ax_fleet.text(x[i]-width/2, r.fresh_mip_buses+0.22, f"{int(r.fresh_mip_buses)} {fs}",
                  ha="center", va="bottom", fontsize=9.5)
    ax_fleet.text(x[i]+width/2, r.warm_mip_buses+0.22, f"{int(r.warm_mip_buses)} {ws}",
                  ha="center", va="bottom", fontsize=9.5)
ax_fleet.set_ylabel("Selected buses"); ax_fleet.set_xlabel("Target k (same chain-3 input at each k)")
ax_fleet.set_xticks(x, [f"k={k}" for k in ks])
ax_fleet.set_ylim(0, max(matched.warm_mip_buses.max(), matched.fresh_mip_buses.max()) + 2)
ax_fleet.grid(axis="y", color="#dddddd", linewidth=0.8); ax_fleet.set_axisbelow(True)
ax_fleet.legend(loc="upper left", ncol=3, frameon=False)
fig.savefig(OUT / "fresh_vs_inherited_covering_chain3.png", bbox_inches="tight")
fig.savefig(OUT / "fresh_vs_inherited_covering_chain3.pdf", bbox_inches="tight")
plt.close(fig)

source_hashes = {str(p): sha256(p) for p in [REGISTER, SNAPSHOT, WARM_CG, WARM_MIP]}
prov = {
    "register_sha256": source_hashes[str(REGISTER)], "snapshot_sha256": source_hashes[str(SNAPSHOT)],
    "warm_cg_status_sha256": source_hashes[str(WARM_CG)], "warm_mip_status_sha256": source_hashes[str(WARM_MIP)],
    "register_snapshot_time_utc": str(register["snapshot_time_utc"].dropna().iloc[0]) if register["snapshot_time_utc"].notna().any() else "unknown",
    "matched_chain": 3, "matched_k": ks, "same_input_hash_verified": True,
    "source_commit": "21fbecba826824c44f897feef038fcf51c532582",
    "physics": "240 kWh / 240 kW / zero reserve; event 2.5 kWh SOC step / 5-minute blocks; flat tariff",
    "figure_1_data": "fresh_fleet_matrix.csv", "figure_2_data": "fresh_vs_warm_cover_chain3.csv",
}
(OUT / "figure_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

print("Generated figure assets under", OUT)
print("Matrix rows:", len(matrix_df), "matched rows:", len(matched))
print(matched[["target_k", "fresh_cg_total_min", "warm_cg_total_min", "warm_import_min", "warm_post_import_min", "fresh_mip_buses", "warm_mip_buses"]].to_string(index=False))

#!/usr/bin/env python3
"""Render the current matched charging and warm-pool timing figures."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "document_cleanup_20260912"
MONITOR = ROOT / "outputs" / "post_meeting_20260910" / "monitor" / "20260912T041951Z.json"
SNAPSHOT = ROOT / "outputs" / "overnight_extension_20260912" / "cluster_snapshot.json"



def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def charging_figure(
    data: dict,
    filename: str = "charging_current.png",
    figsize: tuple[float, float] = (8.5, 5.2),
) -> dict:
    order = {"peak08": 0, "peak12": 1, "peak18": 2}
    records = sorted(
        data["campaigns"]["terminal_energy_fair_mip_retry_5cdb813"]["comparisons"],
        key=lambda c: order[c["result"]["cell"]["tariff"]],
    )
    peaks = ["08:00", "12:00", "18:00"]
    fixed = [c["result"]["fixed_duties_optimized"]["physical_charging_cost"] for c in records]
    joint = [c["result"]["joint_pool_optimized"]["physical_charging_cost"] for c in records]
    lo = [c["result"]["original_giro"]["charging_cost_lower"] for c in records]
    hi = [c["result"]["original_giro"]["charging_cost_upper"] for c in records]
    mid = [(a + b) / 2 for a, b in zip(lo, hi)]
    yerr = np.array([[m - a for m, a in zip(mid, lo)], [b - m for b, m in zip(hi, mid)]])
    x = np.arange(3, dtype=float)
    width = 0.30

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.bar(x - width / 2, fixed, width, color="#4C78A8", label="Fixed duty (physical replay)")
    ax.bar(x + width / 2, joint, width, color="#F58518", label="Joint (physical replay)")
    ax.errorbar(
        x,
        mid,
        yerr=yerr,
        fmt="o",
        color="#222222",
        ecolor="#222222",
        elinewidth=1.6,
        capsize=5,
        capthick=1.6,
        markersize=5,
        label="Original GIRO (repriced interval)",
    )
    ax.set_xticks(x, peaks)
    ax.set_xlabel("Tariff peak")
    ax.set_ylabel("Charging cost (model currency)")
    ax.set_ylim(0, 600)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / filename, dpi=220, facecolor="white")
    plt.close(fig)
    return {
        "source_records": [c["sha256"] for c in records],
        "tariff_peaks": peaks,
        "fixed_physical_replay": fixed,
        "joint_physical_replay": joint,
        "original_giro_interval": [[a, b] for a, b in zip(lo, hi)],
    }


def import_figure(
    snapshot: dict,
    filename: str = "import_time_current.png",
    figsize: tuple[float, float] = (8.5, 4.8),
) -> dict:
    specs = [
        ("Chain 3, k=8", "warm_chain_p3_k2_10", 8),
        ("Chain 3, k=10", "warm_chain_p3_k2_10", 10),
        ("Chain 5, k=5", "warm_chain_p5_k2_10", 5),
        ("Chain 5, k=6", "warm_chain_p5_k2_10", 6),
        ("Chain 5, k=10", "warm_chain_p5_k2_10", 10),
    ]
    rows = []
    for label, campaign, k in specs:
        campaign_data = snapshot["campaigns"][campaign]
        for record, phases in zip(campaign_data["cg"], campaign_data["phases"]):
            if f"_k{k:02d}_" in Path(record["csv"]).name:
                imported = phases["duration_s_by_phase"].get("inherited_event_pool_import", 0.0)
                other = record["wall_s"] - imported
                rows.append(
                    {
                        "label": label,
                        "import_and_validate_minutes": imported / 60.0,
                        "other_cg_minutes": other / 60.0,
                        "status": record["stop_reason"],
                        "source_path": record["path"],
                        "instance_sha256": record["provenance"]["instance_sha256"],
                        "inherited_pool_status_sha256": record["provenance"]["inherited_event_pool_status_sha256"],
                    }
                )
                break
        else:
            raise RuntimeError(f"Missing {label}")

    labels = [r["label"] for r in rows]
    imported = np.array([r["import_and_validate_minutes"] for r in rows])
    other = np.array([r["other_cg_minutes"] for r in rows])
    y = np.arange(len(rows), dtype=float)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.barh(y, imported, color="#4C78A8", label="Import and validate columns")
    ax.barh(y, other, left=imported, color="#BDBDBD", label="Other CG work")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Minutes")
    ax.set_xlim(0, float(np.max(imported + other) * 1.08))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    fig.savefig(OUT / filename, dpi=220, facecolor="white")
    plt.close(fig)
    return {"rows": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    monitor = json.loads(MONITOR.read_text())
    snapshot = json.loads(SNAPSHOT.read_text())
    provenance = {
        "generator": {
            "script": str(Path(__file__).resolve()),
            "python": sys.executable,
            "matplotlib": matplotlib.__version__,
            "figures_have_titles": False,
            "figures_have_embedded_captions": False,
        },
        "inputs": {
            "monitor": {"path": str(MONITOR), "sha256": sha256(MONITOR)},
            "cluster_snapshot": {"path": str(SNAPSHOT), "sha256": sha256(SNAPSHOT)},
        },
        "charging": charging_figure(monitor),
        "charging_slide": charging_figure(monitor, "charging_current_slide.png", (10.5, 4.3)),
        "import_timing": import_figure(snapshot),
        "import_timing_slide": import_figure(snapshot, "import_time_current_slide.png", (10.5, 4.3)),
    }
    provenance["outputs"] = {
        name: {"path": str(OUT / name), "sha256": sha256(OUT / name)}
        for name in (
            "charging_current.png",
            "charging_current_slide.png",
            "import_time_current.png",
            "import_time_current_slide.png",
        )
    }
    (OUT / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["outputs"], indent=2))


if __name__ == "__main__":
    main()

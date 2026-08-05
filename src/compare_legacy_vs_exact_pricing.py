"""Compare preserved legacy DP pricing traces with the exact-pricer logs.

The old CSVs record per-iteration cumulative pricing and master times. The
exact-pricer logs record objective snapshots every ten iterations and one
total wall time, but not per-iteration timing. Consequently, the comparison
plots convergence against iteration; the CSV reports the timing totals.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
LEGACY = {
    "20-bus": ROOT / "Practice_20bus_g300_20260429_015514" / "pricing_20bus_200cols.csv",
    "30-bus": ROOT / "Practice_30bus_g300_20260429_015514" / "pricing_30bus_200cols.csv",
    "43-bus": ROOT / "Practice_43bus_g300_20260429_015514" / "pricing_43bus_200cols.csv",
}

ITERATION = re.compile(
    r"^\[EXACT\] it\s+(?P<iteration>\d+): obj=(?P<objective>[\d,.]+) "
    r"weight=(?P<weight>[\d.]+) art=(?P<artificials>[\d.]+) min_rc=(?P<min_rc>[-\d,.]+)$",
    re.MULTILINE,
)
START = re.compile(r"^\[SLURM \d+\] (?P<csv>\S+).*?G=(?P<g>\d+) kWh .*?reserve=(?P<reserve>[\d.]+)", re.MULTILINE)
DONE = re.compile(r"^\[EXACT\] DONE: \{.*?\} certified=.*? columns=(?P<columns>\d+) wall=(?P<wall>\d+)s$", re.MULTILINE)


def load_legacy() -> tuple[dict[str, pd.DataFrame], list[dict]]:
    curves = {}
    summaries = []
    for label, path in LEGACY.items():
        frame = pd.read_csv(path)
        curves[label] = frame
        last = frame.iloc[-1]
        summaries.append(
            {
                "family": "legacy_dp",
                "case": label,
                "iterations": int(last["Iteration"]),
                "final_objective": float(last["Master_Obj"]),
                "route_weight": None,
                "min_rc": float(last["Best_RC"]),
                "columns": int(last["Iteration"] * last["Cols_Added"]),
                "master_hours": float(last["Cumulative_Master_Time_s"]) / 3600,
                "pricing_hours": float(last["Cumulative_Pricing_Time_s"]) / 3600,
                "wall_hours": float(last["Total_Runtime_s"]) / 3600,
            }
        )
    return curves, summaries


def load_exact(archive_path: Path) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    curves = {}
    summaries = []
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.name.endswith(".out") or "/logs/EXACTBIG_" not in member.name:
                continue
            text = archive.extractfile(member).read().decode(errors="replace")
            start = START.search(text)
            done = DONE.search(text)
            if start is None or done is None:
                continue
            records = []
            for match in ITERATION.finditer(text):
                records.append(
                    {
                        "iteration": int(match["iteration"]),
                        "objective": float(match["objective"].replace(",", "")),
                        "weight": float(match["weight"]),
                        "artificials": float(match["artificials"]),
                        "min_rc": float(match["min_rc"].replace(",", "")),
                    }
                )
            if not records:
                continue
            csv_name = Path(start["csv"]).stem
            variant = f"G={start['g']}, reserve={start['reserve']}"
            label = f"{csv_name} ({variant})"
            curves[label] = pd.DataFrame(records)
            last = records[-1]
            summaries.append(
                {
                    "family": "exact_pricer",
                    "case": csv_name,
                    "iterations": last["iteration"],
                    "final_objective": last["objective"],
                    "route_weight": last["weight"],
                    "min_rc": last["min_rc"],
                    "columns": int(done["columns"]),
                    "master_hours": None,
                    "pricing_hours": None,
                    "wall_hours": int(done["wall"]) / 3600,
                    "variant": variant,
                }
            )
    return curves, summaries


def plot(legacy: dict[str, pd.DataFrame], exact: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for label, frame in legacy.items():
        axes[0].plot(frame["Iteration"], frame["Master_Obj"], label=label)
    axes[0].set_title("Legacy DP pricing")
    axes[0].set_xlabel("column-generation iteration")
    axes[0].set_ylabel("restricted-master objective")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    selected = [item for item in exact.items() if "_r1" in item[0]]
    for label, frame in selected:
        axes[1].plot(frame["iteration"], frame["objective"], label=label.replace("Practice_Custom_DutyUnion_", ""))
    axes[1].set_title("Exact-pricer runs: r1 examples")
    axes[1].set_xlabel("exact-pricing iteration")
    axes[1].set_ylabel("restricted-master objective")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Convergence traces are not instance-matched; compare shape, not objective level.")
    fig.tight_layout()
    fig.savefig(output_dir / "legacy_vs_exact_convergence.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "legacy_vs_exact")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    legacy_curves, legacy_rows = load_legacy()
    exact_curves, exact_rows = load_exact(args.exact_archive)
    plot(legacy_curves, exact_curves, args.output_dir)
    pd.DataFrame(legacy_rows + exact_rows).to_csv(args.output_dir / "timing_and_endpoint_summary.csv", index=False)
    print(f"Wrote {args.output_dir / 'legacy_vs_exact_convergence.png'}")
    print(f"Wrote {args.output_dir / 'timing_and_endpoint_summary.csv'}")


if __name__ == "__main__":
    main()

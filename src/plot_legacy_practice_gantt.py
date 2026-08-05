"""Render charging Gantts for the preserved April practice MIP solutions.

The legacy solutions select route variables named ``a[i]`` and their route
pools live in checkpoint JSON files, unlike the later snapshot format used by
``plot_charging_gantt_from_solutions.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "legacy_practice_gantt"

CASES = [
    (
        "Practice 30-bus, g=300",
        ROOT / "Practice_30bus_g300_20260429_015514" / "ckpt_latest_30bus_g300_200cols.json",
        ROOT / "Practice_30bus_g300_20260429_015514" / "solution_20260429_015514.sol",
    ),
    (
        "Practice 43-bus, g=300",
        ROOT / "Practice_43bus_g300_20260429_015514" / "ckpt_latest_43bus_g300_200cols.json",
        ROOT / "Practice_43bus_g300_20260429_015514" / "solution_20260429_015514.sol",
    ),
]

STATION_COLORS = {
    "PARX": "#4c78a8",
    "JON_A": "#f58518",
    "2190L": "#54a24b",
    "4808": "#e45756",
    "3127L": "#72b7b2",
    "7880C": "#b279a2",
}


def station_base(station: str) -> str:
    for name in STATION_COLORS:
        if station.startswith(name):
            return name
    return station.split("_", 1)[0]


def parse_solution(solution_path: Path) -> tuple[float | None, list[int], list[str]]:
    objective = None
    selected = []
    dummy_trips = []
    route_pattern = re.compile(r"^a\[(\d+)\]\s+([-+0-9.eE]+)$")
    dummy_pattern = re.compile(r"^q_(\d+)\s+([-+0-9.eE]+)$")

    for line in solution_path.read_text(errors="replace").splitlines():
        if line.startswith("# Objective value ="):
            try:
                objective = float(line.split("=", 1)[1].strip())
            except ValueError:
                pass
        match = route_pattern.match(line.strip())
        if match and float(match.group(2)) > 0.5:
            selected.append(int(match.group(1)))
        match = dummy_pattern.match(line.strip())
        if match and float(match.group(2)) > 0.5:
            dummy_trips.append(match.group(1))
    return objective, selected, dummy_trips


def event_rows(routes: list[dict], selected: list[int], label: str) -> list[dict]:
    rows = []
    for order, index in enumerate(selected):
        route = routes[index]
        stops = route.get("charging_stops", {})
        stations = stops.get("stations", [])
        starts = stops.get("cst", [])
        ends = stops.get("cet", [])
        energy = stops.get("kwh", [])
        for stop_index, station in enumerate(stations):
            if stop_index >= len(starts) or stop_index >= len(ends):
                continue
            start, end = float(starts[stop_index]), float(ends[stop_index])
            if end <= start:
                continue
            rows.append(
                {
                    "case": label,
                    "route_order": order,
                    "route_index": index,
                    "station": str(station),
                    "station_base": station_base(str(station)),
                    "start_min": start,
                    "end_min": end,
                    "kwh": float(energy[stop_index]) if stop_index < len(energy) else None,
                }
            )
    return rows


def render_case(label: str, checkpoint_path: Path, solution_path: Path, output_dir: Path) -> tuple[Path, list[dict]]:
    checkpoint = json.loads(checkpoint_path.read_text())
    routes = checkpoint["routes"]
    objective, selected, dummy_trips = parse_solution(solution_path)
    if any(index >= len(routes) for index in selected):
        raise ValueError(f"{label}: selected solution index is outside checkpoint route pool")

    rows = event_rows(routes, selected, label)
    events = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(16, max(5, len(selected) * 0.28 + 2.5)))

    for row in rows:
        ax.barh(
            row["route_order"],
            (row["end_min"] - row["start_min"]) / 60,
            left=row["start_min"] / 60,
            height=0.72,
            color=STATION_COLORS.get(row["station_base"], "#777777"),
            edgecolor="white",
            linewidth=0.4,
        )

    total_kwh = events["kwh"].sum() if not events.empty else 0.0
    obj_text = f" | MIP objective={objective:,.0f}" if objective is not None else ""
    dummy_text = f" | WARNING: {len(dummy_trips)} dummy trips ({', '.join(dummy_trips)})" if dummy_trips else ""
    ax.set_title(
        f"{label}{obj_text} | real selected routes={len(selected)} | charge events={len(rows)} | kWh={total_kwh:,.1f}{dummy_text}"
    )
    ax.set_xlabel("time of day")
    ax.set_ylabel("selected route")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{hour:02d}:00" for hour in range(0, 25, 2)])
    ax.set_ylim(-1, max(len(selected), 1))
    ax.set_yticks(range(len(selected)))
    ax.grid(True, axis="x", alpha=0.25)

    handles = [Line2D([0], [0], color=color, lw=8, label=name) for name, color in STATION_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    output_path = output_dir / f"{stem}_charging_gantt.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    all_rows = []
    for label, checkpoint_path, solution_path in CASES:
        output_path, rows = render_case(label, checkpoint_path, solution_path, args.output_dir)
        all_rows.extend(rows)
        print(f"Wrote {output_path}")

    pd.DataFrame(all_rows).to_csv(args.output_dir / "charging_events.csv", index=False)


if __name__ == "__main__":
    main()

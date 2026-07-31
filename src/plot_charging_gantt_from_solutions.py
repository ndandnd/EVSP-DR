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


DEFAULT_RESULTS_ROOT = Path("/Users/nadan/Downloads/evsp_final_results")
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "charging_gantt_plots"
DEFAULT_DATA_DIR = Path("/Users/nadan/Documents/projects/demandresponse/data")

STATION_COLORS = {
    "PARX": "#4c78a8",
    "JON_A": "#f58518",
    "2190L": "#54a24b",
    "4808": "#e45756",
    "3127L": "#72b7b2",
    "7880C": "#b279a2",
}


def parse_run_key(run_dir: Path) -> dict | None:
    name = run_dir.name
    m = re.search(
        r"Inst_(?P<fleet>10B|15B)_RND(?P<rnd>\d{3})_(?P<mode>NO_CHEAT|CHEAT|GREEDY)_.*?_(?P<peak>peak\d{2})(?:_[^_]+)?_g300_",
        name,
    )
    if not m:
        return None
    return m.groupdict()


def station_base(station: str) -> str:
    s = str(station)
    if s.startswith("JON_A"):
        return "JON_A"
    if s.startswith("7880C"):
        return "7880C"
    if s.startswith("3127L"):
        return "3127L"
    if s.startswith("2190L"):
        return "2190L"
    if s.startswith("4808"):
        return "4808"
    if s.startswith("PARX"):
        return "PARX"
    return s.split("_")[0]


def minutes_to_hhmm(minutes: float) -> str:
    minutes = int(round(minutes))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def find_latest(patterns: list[str], run_dir: Path) -> Path | None:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(run_dir.glob(pattern))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def snapshot_patterns(snapshot_kind: str) -> list[str]:
    if snapshot_kind == "auto":
        return [
            "routes_12h_snapshot_*.json",
            "routes_10h_snapshot_*.json",
            "routes_3h_snapshot_*.json",
            "routes_colgen_final_*.json",
        ]
    return [f"routes_{snapshot_kind}_snapshot_*.json"]


def solution_for_snapshot(run_dir: Path, snapshot_path: Path) -> Path | None:
    """Find the .sol whose matching summary/log says it solved this snapshot."""
    candidates = sorted(run_dir.glob("final_mip_*.sol"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None

    snapshot_name = snapshot_path.name
    for sol_path in candidates:
        suffix = sol_path.stem.replace("final_mip_", "", 1)
        summary_path = run_dir / f"final_mip_summary_{suffix}.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                if Path(summary.get("ckpt_source", "")).name == snapshot_name:
                    return sol_path
            except Exception:
                pass

        log_path = run_dir / f"final_mip_{suffix}.log"
        if not log_path.exists():
            continue
        try:
            if snapshot_name in log_path.read_text(errors="replace"):
                return sol_path
        except OSError:
            continue

    # If summaries exist but none match, do not guess. This prevents pairing a
    # 3h snapshot with a 12h MIP solution in folders that contain both.
    if list(run_dir.glob("final_mip_summary_*.json")):
        return None

    # Backward-compatible fallback for older runs without summaries/log context.
    return candidates[0]


def parse_objective(sol_path: Path) -> float | None:
    with sol_path.open() as f:
        for line in f:
            if line.startswith("# Objective value ="):
                try:
                    return float(line.split("=", 1)[1].strip())
                except ValueError:
                    return None
    return None


def parse_selected_routes(sol_path: Path) -> list[int]:
    selected = []
    pat = re.compile(r"^a\[(\d+)\]\s+([-+0-9.eE]+)")
    with sol_path.open() as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            idx = int(m.group(1))
            val = float(m.group(2))
            if val > 0.5:
                selected.append(idx)
    return selected


def load_selected_routes(snapshot_path: Path, selected_indices: list[int]) -> dict[int, dict]:
    """Stream a large route snapshot and decode only the selected route objects."""
    wanted = set(selected_indices)
    if not wanted:
        return {}

    marker = '"routes"'
    routes_started = False
    carry = ""
    route_idx = -1
    depth = 0
    in_string = False
    escape = False
    collecting = False
    obj_chars: list[str] = []
    selected_routes: dict[int, dict] = {}

    with snapshot_path.open() as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break

            if not routes_started:
                text = carry + chunk
                pos = text.find(marker)
                if pos < 0:
                    carry = text[-len(marker):]
                    continue
                bracket = text.find("[", pos)
                if bracket < 0:
                    carry = text[pos:]
                    continue
                routes_started = True
                chunk = text[bracket + 1:]

            for ch in chunk:
                if depth == 0:
                    if ch == "{":
                        route_idx += 1
                        depth = 1
                        collecting = route_idx in wanted
                        obj_chars = ["{"] if collecting else []
                    elif ch == "]":
                        return selected_routes
                    continue

                if collecting:
                    obj_chars.append(ch)

                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        if collecting:
                            selected_routes[route_idx] = json.loads("".join(obj_chars))
                            if wanted <= selected_routes.keys():
                                return selected_routes
                        collecting = False
                        obj_chars = []

    return selected_routes


def load_events(run_dir: Path, snapshot_kind: str) -> tuple[pd.DataFrame, dict]:
    meta = parse_run_key(run_dir)
    if meta is None:
        raise ValueError(f"Cannot parse run metadata from {run_dir.name}")

    snapshot_path = find_latest(snapshot_patterns(snapshot_kind), run_dir)
    if snapshot_path is None:
        raise FileNotFoundError(f"Missing .sol or route snapshot in {run_dir}")
    sol_path = solution_for_snapshot(run_dir, snapshot_path)
    if sol_path is None:
        raise FileNotFoundError(f"Missing .sol or route snapshot in {run_dir}")

    selected = parse_selected_routes(sol_path)
    objective = parse_objective(sol_path)
    routes = load_selected_routes(snapshot_path, selected)

    rows = []
    for y_pos, route_idx in enumerate(selected):
        route = routes.get(route_idx)
        if route is None:
            continue
        stops = route.get("charging_stops") or {}
        stations = stops.get("stations") or []
        cst = stops.get("cst") or []
        cet = stops.get("cet") or []
        kwh = stops.get("kwh") or []

        for stop_idx, station in enumerate(stations):
            if stop_idx >= len(cst) or stop_idx >= len(cet):
                continue
            start = float(cst[stop_idx])
            end = float(cet[stop_idx])
            if end <= start:
                continue
            energy = float(kwh[stop_idx]) if stop_idx < len(kwh) else None
            rows.append({
                **meta,
                "run_dir": str(run_dir),
                "sol_path": str(sol_path),
                "snapshot_path": str(snapshot_path),
                "objective": objective,
                "route_idx": route_idx,
                "selected_route_order": y_pos,
                "station": str(station),
                "station_base": station_base(str(station)),
                "start_min": start,
                "end_min": end,
                "duration_min": end - start,
                "kwh": energy,
                "route_desc": route.get("desc", ""),
            })

    return pd.DataFrame(rows), {
        **meta,
        "run_dir": str(run_dir),
        "sol_path": str(sol_path),
        "snapshot_path": str(snapshot_path),
        "snapshot_kind": snapshot_kind,
        "objective": objective,
        "selected_routes": selected,
    }


def load_price_curve(data_dir: Path, peak: str) -> pd.DataFrame | None:
    peak_num = peak.replace("peak", "")
    path = data_dir / f"spatiotemporal_single_peak_{peak_num}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if {"time_block", "cost"} - set(df.columns):
        return None
    # Same prices at every station in this experiment; average is robust if duplicated.
    return df.groupby("time_block", as_index=False)["cost"].mean().sort_values("time_block")


def add_price_background(ax, data_dir: Path, peak: str):
    price = load_price_curve(data_dir, peak)
    if price is None or price.empty:
        return

    ax2 = ax.twinx()
    ax2.plot(price["time_block"], price["cost"], color="black", linewidth=1.0, alpha=0.45)
    ax2.set_ylabel("price", fontsize=8)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.grid(False)


def plot_group(group_key: tuple[str, str, str], events_by_peak: dict[str, pd.DataFrame], meta_by_peak: dict[str, dict], out_dir: Path, data_dir: Path):
    fleet, rnd, mode = group_key
    peaks = ["peak08", "peak12", "peak18"]
    available = [p for p in peaks if p in events_by_peak]
    if not available:
        return None

    fig, axes = plt.subplots(len(peaks), 1, figsize=(16, 4.2 * len(peaks)), sharex=True)
    if len(peaks) == 1:
        axes = [axes]

    snapshot_label = ""
    for meta in meta_by_peak.values():
        if meta.get("snapshot_kind"):
            snapshot_label = f" ({meta['snapshot_kind']})"
            break

    fig.suptitle(f"Charging Gantt: Inst_{fleet}_RND{rnd}_{mode}{snapshot_label}", fontsize=16)

    for ax, peak in zip(axes, peaks):
        df = events_by_peak.get(peak)
        meta = meta_by_peak.get(peak)
        if df is None or df.empty:
            ax.set_title(f"{peak}: missing solution or no charging events")
            ax.set_xlim(0, 24)
            ax.set_ylim(-1, 1)
            ax.grid(True, axis="x", alpha=0.25)
            continue

        for _, row in df.iterrows():
            color = STATION_COLORS.get(row["station_base"], "#777777")
            ax.barh(
                y=row["selected_route_order"],
                width=row["duration_min"] / 60.0,
                left=row["start_min"] / 60.0,
                height=0.72,
                color=color,
                edgecolor="white",
                linewidth=0.4,
            )

        obj_txt = ""
        if meta and meta.get("objective") is not None:
            obj_txt = f" | obj={meta['objective']:,.0f}"
        total_kwh = df["kwh"].dropna().sum()
        ax.set_title(f"{peak}{obj_txt} | charges={len(df)} | kWh={total_kwh:,.1f}")
        ax.set_ylabel("selected route")
        ax.grid(True, axis="x", alpha=0.25)
        max_route_order = int(df["selected_route_order"].max())
        ax.set_ylim(-1, max(max_route_order + 1, 1))
        ax.set_yticks(range(max_route_order + 1))

        add_price_background(ax, data_dir, peak)

    axes[-1].set_xlabel("time of day")
    axes[-1].set_xlim(0, 24)
    axes[-1].set_xticks(range(0, 25, 2))
    axes[-1].set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])

    handles = [
        Line2D([0], [0], color=color, lw=8, label=station)
        for station, color in STATION_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.94, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{snapshot_label.strip(' ()')}" if snapshot_label else ""
    out = out_dir / f"charging_gantt_Inst_{fleet}_RND{rnd}_{mode}{suffix}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create charging Gantt plots from final_mip .sol files and route snapshots.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fleet", choices=["10B", "15B"], help="Optional filter, e.g. 10B.")
    parser.add_argument("--rnd", help="Optional filter, e.g. 001.")
    parser.add_argument("--mode", choices=["CHEAT", "NO_CHEAT", "GREEDY"], help="Optional filter.")
    parser.add_argument(
        "--snapshot-kind",
        choices=["auto", "3h", "10h", "12h", "24h"],
        default="12h",
        help="Which route snapshot to pair with the matching final_mip solution.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each run as it is loaded.")
    args = parser.parse_args()

    all_events = []
    meta_rows = []
    grouped_events: dict[tuple[str, str, str], dict[str, pd.DataFrame]] = {}
    grouped_meta: dict[tuple[str, str, str], dict[str, dict]] = {}

    for run_dir in sorted(p for p in args.results_root.iterdir() if p.is_dir()):
        meta = parse_run_key(run_dir)
        if meta is None:
            continue
        if args.fleet and meta["fleet"] != args.fleet:
            continue
        if args.rnd and meta["rnd"] != args.rnd:
            continue
        if args.mode and meta["mode"] != args.mode:
            continue
        if args.verbose:
            print(f"[LOAD] {run_dir.name}", flush=True)
        try:
            events, run_meta = load_events(run_dir, args.snapshot_kind)
        except Exception as exc:
            print(f"[WARN] Skipping {run_dir.name}: {exc}")
            continue

        key = (meta["fleet"], meta["rnd"], meta["mode"])
        peak = meta["peak"]
        grouped_events.setdefault(key, {})[peak] = events
        grouped_meta.setdefault(key, {})[peak] = run_meta
        meta_rows.append({
            **run_meta,
            "snapshot_kind": args.snapshot_kind,
            "num_charging_events": len(events),
            "total_kwh": events["kwh"].dropna().sum() if not events.empty else 0.0,
        })
        if not events.empty:
            all_events.append(events)

    made = []
    for key in sorted(grouped_events):
        out = plot_group(key, grouped_events[key], grouped_meta.get(key, {}), args.output_dir, args.data_dir)
        if out:
            made.append(out)

    if all_events:
        events_df = pd.concat(all_events, ignore_index=True)
    else:
        events_df = pd.DataFrame()
    events_path = args.output_dir / "charging_events_long.csv"
    summary_path = args.output_dir / "charging_solution_summary.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(events_path, index=False)
    pd.DataFrame(meta_rows).to_csv(summary_path, index=False)

    print(f"Wrote {len(made)} Gantt plots to {args.output_dir}")
    print(f"Wrote event table: {events_path}")
    print(f"Wrote summary table: {summary_path}")
    for p in made[:10]:
        print(f"  {p}")
    if len(made) > 10:
        print(f"  ... {len(made) - 10} more")


if __name__ == "__main__":
    main()

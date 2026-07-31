#!/usr/bin/env python3
"""
Analyze observed time gaps between Regular and Recharge events.

Non-Regular/Recharge rows such as Deadhead, Pull-out, Pull-in, Prep-out, and
Prep-in are treated as intervening movement/setup. They are not transition
endpoints, but their elapsed time is naturally included in the gap between the
previous important event and the next important event.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


PRIMARY_GAP_TYPES = ["trip2trip", "trip2charge", "charge2trip"]


def parse_time_to_minutes(value) -> float:
    if pd.isna(value):
        return math.nan
    text = str(value).strip()
    if not text or ":" not in text:
        return math.nan
    hours, minutes = text.split(":", 1)
    return int(hours) * 60 + int(minutes)


def classify_transition(left_ident: str, right_ident: str) -> str:
    if left_ident == "Regular" and right_ident == "Regular":
        return "trip2trip"
    if left_ident == "Regular" and right_ident == "Recharge":
        return "trip2charge"
    if left_ident == "Recharge" and right_ident == "Regular":
        return "charge2trip"
    if left_ident == "Recharge" and right_ident == "Recharge":
        return "charge2charge"
    return "other"


def analyze(input_csv: Path, outdir: Path, threshold_min: float) -> None:
    df = pd.read_csv(input_csv)
    required = {"VehicleTask", "Identifier", "Start1", "End1"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    df = df.copy()
    df["_source_row"] = range(len(df))
    df["_start_min"] = df["Start1"].map(parse_time_to_minutes)
    df["_end_min"] = df["End1"].map(parse_time_to_minutes)
    df["_vehicle"] = df["VehicleTask"].astype(str)
    df["_identifier"] = df["Identifier"].astype(str)

    rows = []

    for vehicle, grp in df.groupby("_vehicle", sort=True):
        ordered = grp.sort_values(["_start_min", "_end_min", "_source_row"]).reset_index(drop=True)
        ordered["_seq"] = range(len(ordered))
        important = ordered[ordered["_identifier"].isin(["Regular", "Recharge"])].copy()

        for pos in range(len(important) - 1):
            left = important.iloc[pos]
            right = important.iloc[pos + 1]

            gap_type = classify_transition(left["_identifier"], right["_identifier"])
            gap_min = float(right["_start_min"] - left["_end_min"])

            between = ordered[
                (ordered["_seq"] > int(left["_seq"]))
                & (ordered["_seq"] < int(right["_seq"]))
            ]
            intervening = "|".join(between["_identifier"].astype(str).tolist())

            rows.append({
                "vehicle_task": vehicle,
                "gap_type": gap_type,
                "gap_min": gap_min,
                "above_threshold": gap_min > threshold_min,
                "from_identifier": left["_identifier"],
                "to_identifier": right["_identifier"],
                "from_start": left.get("Start1"),
                "from_end": left.get("End1"),
                "to_start": right.get("Start1"),
                "to_end": right.get("End1"),
                "from_location": left.get("From1"),
                "to_location": right.get("To1"),
                "from_ordered_trip_id": left.get("Ordered_Trip_ID"),
                "to_ordered_trip_id": right.get("Ordered_Trip_ID"),
                "from_source_row": int(left["_source_row"]),
                "to_source_row": int(right["_source_row"]),
                "intervening_identifiers": intervening,
                "intervening_count": int(len(between)),
            })

    transitions = pd.DataFrame(rows)
    if transitions.empty:
        raise ValueError("No Regular/Recharge transitions found.")

    outdir.mkdir(parents=True, exist_ok=True)

    transitions_path = outdir / "vehicle_gap_transitions.csv"
    transitions.to_csv(transitions_path, index=False)

    primary = transitions[transitions["gap_type"].isin(PRIMARY_GAP_TYPES)].copy()

    summary_rows = []
    for gap_type, grp in primary.groupby("gap_type", sort=False):
        gaps = grp["gap_min"].dropna()
        summary_rows.append({
            "gap_type": gap_type,
            "count": int(len(gaps)),
            "vehicles": int(grp["vehicle_task"].nunique()),
            "mean_min": gaps.mean(),
            "median_min": gaps.quantile(0.50),
            "p75_min": gaps.quantile(0.75),
            "p90_min": gaps.quantile(0.90),
            "p95_min": gaps.quantile(0.95),
            "p99_min": gaps.quantile(0.99),
            "max_min": gaps.max(),
            f"count_gt_{threshold_min:g}": int((gaps > threshold_min).sum()),
            f"share_gt_{threshold_min:g}": float((gaps > threshold_min).mean()),
        })
    summary = pd.DataFrame(summary_rows).sort_values("gap_type")
    summary_path = outdir / "vehicle_gap_summary.csv"
    summary.to_csv(summary_path, index=False)

    vehicle_max = (
        primary
        .groupby(["vehicle_task", "gap_type"], as_index=False)["gap_min"]
        .max()
        .rename(columns={"gap_min": "vehicle_max_gap_min"})
        .sort_values(["gap_type", "vehicle_max_gap_min"], ascending=[True, False])
    )
    vehicle_max_path = outdir / "vehicle_gap_vehicle_maxima.csv"
    vehicle_max.to_csv(vehicle_max_path, index=False)

    bins = list(range(0, int(math.ceil(max(0, primary["gap_min"].max()) / 10.0) * 10) + 20, 10))
    hist = (
        primary.assign(bin_min=pd.cut(primary["gap_min"], bins=bins, right=True, include_lowest=True))
        .groupby(["gap_type", "bin_min"], observed=False)
        .size()
        .reset_index(name="count")
    )
    hist_path = outdir / "vehicle_gap_histogram_10min_bins.csv"
    hist.to_csv(hist_path, index=False)

    top = (
        primary
        .sort_values("gap_min", ascending=False)
        .head(100)
    )
    top_path = outdir / "vehicle_gap_top100.csv"
    top.to_csv(top_path, index=False)

    print("\n=== Observed Gap Summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n=== Top 15 longest primary gaps ===")
    cols = [
        "gap_type", "gap_min", "vehicle_task", "from_end", "to_start",
        "from_identifier", "to_identifier", "intervening_identifiers",
        "from_ordered_trip_id", "to_ordered_trip_id",
    ]
    print(top[cols].head(15).to_string(index=False))

    print("\n=== Files written ===")
    for path in [transitions_path, summary_path, vehicle_max_path, hist_path, top_path]:
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/Par_VehicleDetails_Updated.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("src/results/vehicle_gap_analysis"),
    )
    parser.add_argument("--threshold-min", type=float, default=60.0)
    args = parser.parse_args()
    analyze(args.input, args.outdir, args.threshold_min)


if __name__ == "__main__":
    main()

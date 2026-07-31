from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_STATIONS = ["PARX", "JON_A", "2190L", "4808", "3127L", "7880C"]


def convert_one(input_csv: Path, output_csv: Path, stations: list[str]) -> None:
    df = pd.read_csv(input_csv)
    required = {"time_block", "cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")

    rows = []
    for _, row in df.sort_values("time_block").iterrows():
        for station in stations:
            rows.append({
                "time_block": row["time_block"],
                "station": station,
                "cost": row["cost"],
            })

    out = pd.DataFrame(rows, columns=["time_block", "station", "cost"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"WROTE {output_csv}  rows={len(out)} stations={len(stations)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert temporal hourly price CSVs into spatiotemporal station/hour price CSVs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Directory containing hourly_prices_single_peak_XX.csv files.",
    )
    parser.add_argument(
        "--peaks",
        default="08,12,18",
        help="Comma-separated peak labels to convert, e.g. 08,12,18.",
    )
    parser.add_argument(
        "--stations",
        default=",".join(DEFAULT_STATIONS),
        help="Comma-separated station names to replicate prices across.",
    )
    args = parser.parse_args()

    stations = [x.strip() for x in args.stations.split(",") if x.strip()]
    peaks = [x.strip() for x in args.peaks.split(",") if x.strip()]

    for peak in peaks:
        src = args.data_dir / f"hourly_prices_single_peak_{peak}.csv"
        dst = args.data_dir / f"spatiotemporal_single_peak_{peak}.csv"
        if not src.exists():
            raise FileNotFoundError(src)
        convert_one(src, dst, stations)


if __name__ == "__main__":
    main()

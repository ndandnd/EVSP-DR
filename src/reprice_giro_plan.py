"""Tier-0 decomposition baseline: GIRO's own charging plan repriced per tariff.

Reads the Recharge activities straight out of Par_VehicleDetails_Updated.csv
(station, start/end time, kWh) and evaluates their electricity cost under
each tariff — no optimization, no model. This is the "observed GIRO plan
repriced" row of the decomposition experiment: everything our optimizer
saves is measured against these numbers.

    python reprice_giro_plan.py
    python reprice_giro_plan.py --tariffs hourly_prices_flat.csv,hourly_prices_single_peak_08.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import CHARGE_START_COST, CHARGING_STATIONS
from utils_v2 import load_station_hourly_prices

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_TARIFFS = [
    "hourly_prices_flat.csv",
    "hourly_prices_single_peak_08.csv",
    "hourly_prices_single_peak_12.csv",
    "hourly_prices_single_peak_18.csv",
    "hourly_prices_transdev_sek.csv",
]


def _minutes(hhmm) -> int:
    hh, mm = str(hhmm).split(":")
    return int(hh) * 60 + int(mm)


def segment_cost(start_min, end_min, kwh, curve) -> float:
    """Uniform-in-time energy split across hourly prices."""
    if kwh <= 0 or end_min <= start_min:
        # zero-length recharge rows exist; price at the start hour
        hour = int(start_min // 60)
        return kwh * curve.get(hour, curve[max(curve)])
    span = float(end_min - start_min)
    cost, t = 0.0, float(start_min)
    while t < end_min - 1e-9:
        nxt = min((int(t // 60) + 1) * 60.0, float(end_min))
        hour = int(t // 60)
        price = curve.get(hour, curve[max(curve)])
        cost += price * kwh * (nxt - t) / span
        t = nxt
    return cost


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="Par_VehicleDetails_Updated.csv")
    parser.add_argument("--tariffs", default=",".join(DEFAULT_TARIFFS))
    parser.add_argument("--include-start-cost", action="store_true",
                        help="Add the $%.0f-per-charge-start fee used by the "
                             "optimizer's cost model (for like-for-like "
                             "comparisons)." % CHARGE_START_COST)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    master = pd.read_csv(DATA_DIR / args.master)
    recharges = master[master["Identifier"] == "Recharge"].copy()
    recharges["VehicleTask"] = recharges["VehicleTask"].astype(str)
    recharges["kwh"] = pd.to_numeric(recharges["Recharge kWh"], errors="coerce").fillna(0.0)
    recharges["start_min"] = recharges["Start1"].map(_minutes)
    recharges["end_min"] = recharges["End1"].map(_minutes)

    tariff_files = [t.strip() for t in args.tariffs.split(",") if t.strip()]
    curves = {}
    for tf in tariff_files:
        prices = load_station_hourly_prices(DATA_DIR / tf, CHARGING_STATIONS)
        curves[tf] = prices.get("PARX") or next(iter(prices.values()))

    n_duties = recharges["VehicleTask"].nunique()
    total_kwh = recharges["kwh"].sum()
    n_starts = len(recharges)
    print(f"GIRO plan: {n_duties} duties with recharges, {n_starts} charge "
          f"events, {total_kwh:,.1f} kWh total")
    print(f"{'tariff':<38} {'energy cost':>12} {'with start fees':>16}")

    summary = {"duties": n_duties, "charge_events": n_starts,
               "total_kwh": round(total_kwh, 3), "tariffs": {}}
    per_duty = {}
    for tf, curve in curves.items():
        cost = float(sum(
            segment_cost(r.start_min, r.end_min, r.kwh, curve)
            for r in recharges.itertuples()))
        with_fees = cost + CHARGE_START_COST * n_starts
        print(f"{tf:<38} {cost:>12,.2f} {with_fees:>16,.2f}")
        duty_costs = {}
        for duty, group in recharges.groupby("VehicleTask"):
            duty_costs[duty] = round(float(sum(
                segment_cost(r.start_min, r.end_min, r.kwh, curve)
                for r in group.itertuples())), 4)
        per_duty[tf] = duty_costs
        summary["tariffs"][tf] = {
            "energy_cost": round(cost, 4),
            "with_start_fees": round(with_fees, 4),
        }

    if args.json_out:
        summary["per_duty"] = per_duty
        with open(args.json_out, "w") as fh:
            json.dump(summary, fh, indent=1)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

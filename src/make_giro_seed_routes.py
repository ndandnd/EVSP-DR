"""GIRO seed routes for CHEAT-style exact-CG runs, plus feasibility replay.

Two jobs:

1. ``--validate-only``: replay every GIRO duty's actual activity sequence
   (energy usage and recharges straight from Par_VehicleDetails_Updated.csv)
   under a given battery/charger/reserve, reporting min-SOC and violations.
   This answers "are GIRO's duties feasible at 240 kWh / 220 kW / reserve r"
   with the operator's own telemetry-derived numbers — no model assumptions.

2. Instance seed generation: for a duty-union instance CSV, emit a
   runner-format routes JSON containing one route per constituent duty
   (trips in GIRO's order, charging events from GIRO's Recharge rows).
   Feeding it to the exact pipeline via --extra-routes / --seed-routes
   guarantees the integer master always contains GIRO's plan as an
   incumbent: results are "GIRO or better" by construction.

    python make_giro_seed_routes.py --validate-only --g-kwh 240 --charge-kw 220 --reserve 0.0
    python make_giro_seed_routes.py --instance duty_unions/Practice_Custom_DutyUnion_k08_r3.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MASTER_CSV = DATA_DIR / "Par_VehicleDetails_Updated.csv"
DEPOT_NODE = "PARX_0"
MODEL_RATE_KW = 300.0  # duration-repair reference rate (see Recharge handling)
STATION_BY_BASE = {"2190": "2190L_0", "2190L": "2190L_0", "4808": "4808_0",
                   "PARX": "PARX_1", "3127": "3127L_0", "3127L": "3127L_0",
                   "7880": "7880C_0", "7880C": "7880C_0", "JON": "JON_A_0",
                   "JON_A": "JON_A_0"}


def _minutes(hhmm) -> int:
    hh, mm = str(hhmm).split(":")
    return int(hh) * 60 + int(mm)


def _station_node(raw) -> str:
    s = str(raw).upper()
    for key, node in STATION_BY_BASE.items():
        if key in s:
            return node
    return f"{raw}_0"


def load_master():
    master = pd.read_csv(MASTER_CSV)
    master["VehicleTask"] = master["VehicleTask"].astype(str)
    master["usage"] = pd.to_numeric(master["Usage kWh"], errors="coerce").fillna(0.0)
    master["recharge"] = pd.to_numeric(master["Recharge kWh"], errors="coerce").fillna(0.0)
    master["start_min"] = master["Start1"].map(_minutes)
    master["end_min"] = master["End1"].map(_minutes)
    return master


def replay_duty(rows, g_kwh, charge_kw, reserve_kwh):
    """Replay one duty chronologically; return (min_soc, events, violations)."""
    soc = g_kwh
    min_soc = soc
    violations = []
    for row in rows.itertuples():
        if row.Identifier == "Recharge":
            window_min = max(0.0, row.end_min - row.start_min)
            achievable = window_min * charge_kw / 60.0
            if row.recharge > achievable + 1e-6:
                violations.append(
                    f"recharge {row.recharge:.1f} kWh exceeds "
                    f"{achievable:.1f} kWh achievable at {charge_kw:.0f} kW "
                    f"in {window_min:.0f} min")
            soc = min(g_kwh, soc + row.recharge)
        else:
            soc -= row.usage
            min_soc = min(min_soc, soc)
            if soc < reserve_kwh - 1e-6:
                violations.append(
                    f"SOC {soc:.1f} kWh < reserve {reserve_kwh:.1f} "
                    f"after {row.Identifier} ending {row.End1}")
    return min_soc, violations


def validate(master, g_kwh, charge_kw, reserve_frac):
    reserve = reserve_frac * g_kwh
    print(f"Replay of GIRO duties at G={g_kwh:.0f} kWh, {charge_kw:.0f} kW, "
          f"reserve {reserve_frac:.0%} ({reserve:.0f} kWh):")
    bad = 0
    worst = []
    for duty, rows in master.sort_values("start_min").groupby("VehicleTask"):
        min_soc, violations = replay_duty(rows, g_kwh, charge_kw, reserve)
        worst.append((min_soc, duty))
        if violations:
            bad += 1
            print(f"  INFEASIBLE {duty}: min SOC {min_soc:.1f} kWh "
                  f"({100 * min_soc / g_kwh:.1f}%); first: {violations[0]}")
    worst.sort()
    n = master["VehicleTask"].nunique()
    print(f"{n - bad}/{n} duties feasible; five tightest min-SOC duties:")
    for min_soc, duty in worst[:5]:
        print(f"  {duty}: {min_soc:.1f} kWh ({100 * min_soc / g_kwh:.1f}%)")
    return bad


def build_seeds(master, instance_csv: Path, out_path: Path):
    inst = pd.read_csv(DATA_DIR / instance_csv)
    if "Ordered_Trip_ID" not in inst.columns:
        raise SystemExit("instance CSV lacks Ordered_Trip_ID")
    ordered_to_local = {int(o): i for i, o in enumerate(inst["Ordered_Trip_ID"])}

    regular = master[(master["Identifier"] == "Regular")
                     & master["Ordered_Trip_ID"].notna()].copy()
    regular["Ordered_Trip_ID"] = regular["Ordered_Trip_ID"].astype(int)
    duty_of = dict(zip(regular["Ordered_Trip_ID"], regular["VehicleTask"]))

    duties = sorted({duty_of[o] for o in ordered_to_local if o in duty_of})
    missing = [o for o in ordered_to_local if o not in duty_of]
    if missing:
        raise SystemExit(f"{len(missing)} instance trips not in the GIRO master")

    routes = []
    for duty in duties:
        rows = master[master["VehicleTask"] == duty].sort_values("start_min")
        route_nodes = [DEPOT_NODE]
        charging = {"stations": [], "cst": [], "cet": [], "kwh": []}
        n_local = 0
        for row in rows.itertuples():
            if row.Identifier == "Regular" and not pd.isna(row.Ordered_Trip_ID):
                local = ordered_to_local.get(int(row.Ordered_Trip_ID))
                if local is not None:
                    route_nodes.append(local)
                    n_local += 1
            elif row.Identifier == "Recharge":
                node = _station_node(row.From1)
                route_nodes.append(node)
                charging["stations"].append(node)
                cst = int(row.start_min)
                # Hastus rounds recharge windows to whole minutes, so recorded
                # kWh can imply >nominal rates (e.g. 72.2 kWh in "13 min" =
                # 333 kW). Repair the duration to what the energy requires at
                # the model rate; downstream validation still checks timing.
                needed_min = float(row.recharge) / MODEL_RATE_KW * 60.0
                cet = max(int(row.end_min), int(cst + needed_min + 0.999))
                charging["cst"].append(cst)
                charging["cet"].append(cet)
                charging["kwh"].append(float(row.recharge))
        route_nodes.append(DEPOT_NODE)
        routes.append({
            "route": route_nodes,
            "charging_stops": charging,
            "charging_activities": len(charging["stations"]),
            "type": "truck",
            "deadhead_kwh": 0.0,
            "_rc": 0.0,
            "desc": f"[GIRO seed] duty {duty}: {n_local} trips, "
                    f"{len(charging['stations'])} recharges",
        })

    payload = {"routes": routes, "source": "giro_seed",
               "instance_csv": str(instance_csv), "duties": duties}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1)
    covered = sum(1 for r in routes for n in r["route"] if isinstance(n, int))
    print(f"wrote {out_path}: {len(routes)} GIRO duty routes covering "
          f"{covered}/{len(inst)} instance trips")
    if covered != len(inst):
        raise SystemExit("seed does not partition the instance — investigate")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, default=None,
                        help="Instance CSV relative to data/ (seed generation).")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--g-kwh", type=float, default=300.0)
    parser.add_argument("--charge-kw", type=float, default=300.0)
    parser.add_argument("--reserve", type=float, default=0.0)
    args = parser.parse_args(argv)

    master = load_master()
    if args.validate_only:
        bad = validate(master, args.g_kwh, args.charge_kw, args.reserve)
        return 0 if bad == 0 else 3
    if not args.instance:
        raise SystemExit("need --instance (or --validate-only)")
    name = re.sub(r"\.csv$", "", Path(args.instance).name)
    out = args.out or Path("results/giro_seeds") / f"{name}_giro_seed.json"
    return build_seeds(master, args.instance, out)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Run only the greedy EVSP initializer and inspect the seed routes.

This intentionally avoids Gurobi, the RMP, and the column-generation loop.
It is meant as a quick sanity check for questions like:

  - How many initial greedy vehicle routes are created?
  - Which trips are grouped into each route?
  - Where does greedy insert charging?

Example:

  python3 run_greedy_init_only.py --csv Inst_10B_RND001.csv --G 300
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import pandas as pd

from config import (
    DEPOT_NAME as CONFIG_DEPOT_NAME,
    CHARGING_STATIONS,
    STATION_COPIES,
    TIMEBLOCKS_PER_HOUR,
    bar_t,
    CHARGE_RATE_KW,
)
from greedy_init import build_greedy_routes


ROOT = Path(__file__).resolve().parent
ROOT_DIR = ROOT.parent
DATA_DIR = ROOT_DIR / "data"
OUTDIR = ROOT / "results" / "greedy_init_debug"
OUTDIR.mkdir(parents=True, exist_ok=True)

TB_MIN = int(round(60 / TIMEBLOCKS_PER_HOUR))


def _total_minutes(hhmm: str) -> int:
    hh, mm = str(hhmm).split(":")
    return int(hh) * 60 + int(mm)


def _floor_block(hhmm: str) -> int:
    m = _total_minutes(hhmm)
    blk0 = m // TB_MIN
    return max(1, min(int(bar_t), blk0 + 1))


def _ceil_block(hhmm: str) -> int:
    m = _total_minutes(hhmm)
    blk0 = (m + TB_MIN - 1) // TB_MIN
    return max(1, min(int(bar_t), blk0 + 1))


def ceil_blocks_from_minutes(m: float) -> int:
    return int(math.ceil(float(m) / float(TB_MIN)))


def _norm_token(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _normalize_ref(x):
    s = _norm_token(x)
    if s is None:
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return s


def strip_copy_suffix(name: str) -> str:
    s = str(name).strip()
    if "_" in s:
        left, right = s.rsplit("_", 1)
        if right.isdigit():
            return left
    return s


def expand_station_copies(base_names, station_copies):
    out = []
    for base in base_names:
        for k in range(station_copies.get(base, 1)):
            out.append(f"{base}_{k}")
    return out


def _ordered_ref_pair(a: str, b: str):
    return (a, b) if a <= b else (b, a)


def load_trip_dataframe(routes_csv: Path) -> pd.DataFrame:
    df_trips = pd.read_csv(routes_csv)
    column_renaming = {
        "From1": "SL",
        "Start1": "ST",
        "End1": "ET",
        "To1": "EL",
        "Usage kWh": "Energy used",
        "Energy_used": "Energy used",
    }
    df_trips = df_trips.rename(columns=column_renaming)

    required = {"SL", "ST", "ET", "EL", "Energy used"}
    missing = required - set(df_trips.columns)
    if missing:
        raise ValueError(f"{routes_csv.name} missing required columns: {sorted(missing)}")

    df_trips = df_trips.reset_index(drop=True).copy()
    df_trips["Trip"] = df_trips.index
    df_trips["st_blk"] = df_trips["ST"].astype(str).map(_floor_block)
    df_trips["et_blk"] = df_trips["ET"].astype(str).map(_ceil_block)
    df_trips["st_min"] = df_trips["ST"].astype(str).map(_total_minutes)
    df_trips["et_min"] = df_trips["ET"].astype(str).map(_total_minutes)
    df_trips["eps_kwh"] = df_trips["Energy used"].astype(float)
    return df_trips


def build_arc_data(df_trips: pd.DataFrame, ref_dict_csv: Path, ref_dhd_csv: Path):
    df_ref_dict = pd.read_csv(ref_dict_csv)
    df_ref_dhd = pd.read_csv(ref_dhd_csv)

    loc_col = next((c for c in df_ref_dict.columns if str(c).strip().lower() == "location"), None)
    ref_col = next((c for c in df_ref_dict.columns if str(c).strip().lower() == "ref"), None)
    if loc_col is None or ref_col is None:
        raise ValueError(f"{ref_dict_csv.name} must include columns Location and Ref")

    loc_to_ref = {}
    for _, row in df_ref_dict.iterrows():
        loc = _norm_token(row[loc_col])
        ref = _normalize_ref(row[ref_col])
        if loc is not None and ref is not None:
            loc_to_ref[loc] = ref
    for loc, ref in list(loc_to_ref.items()):
        loc_to_ref.setdefault(strip_copy_suffix(loc), ref)

    start_ref_col, end_ref_col, dur_col, en_col = list(df_ref_dhd.columns[:4])
    ref_pair_minutes_kwh = {}
    for _, row in df_ref_dhd.iterrows():
        ref_a = _normalize_ref(row[start_ref_col])
        ref_b = _normalize_ref(row[end_ref_col])
        dur_min = pd.to_numeric(row[dur_col], errors="coerce")
        eng_kwh = pd.to_numeric(row[en_col], errors="coerce")
        if ref_a is None or ref_b is None or pd.isna(dur_min) or pd.isna(eng_kwh):
            continue
        if ref_a == ref_b:
            continue
        key = _ordered_ref_pair(ref_a, ref_b)
        val = (float(dur_min), float(eng_kwh))
        prev = ref_pair_minutes_kwh.get(key)
        if prev is None or val[0] < prev[0]:
            ref_pair_minutes_kwh[key] = val

    known_refs = set()
    for a_ref, b_ref in ref_pair_minutes_kwh.keys():
        known_refs.add(a_ref)
        known_refs.add(b_ref)

    def resolve_ref(node_name):
        raw = _norm_token(node_name)
        if raw is None:
            return None
        base = strip_copy_suffix(raw)
        candidates = [raw, base, _normalize_ref(raw), _normalize_ref(base)]
        for c in candidates:
            if c is None:
                continue
            if c in loc_to_ref:
                return loc_to_ref[c]
            if c in known_refs:
                return c
        return None

    def arc_from_to(from_node: str, to_node: str):
        a = _norm_token(from_node)
        b = _norm_token(to_node)
        if a is None or b is None:
            return None
        if a == b:
            return (0, 0.0, 0.0)
        ref_a = resolve_ref(a)
        ref_b = resolve_ref(b)
        if ref_a is None or ref_b is None:
            return None
        if ref_a == ref_b:
            return (0, 0.0, 0.0)
        pair = ref_pair_minutes_kwh.get(_ordered_ref_pair(ref_a, ref_b))
        if pair is None:
            return None
        dur_min, eng_kwh = pair
        return (ceil_blocks_from_minutes(dur_min), float(dur_min), float(eng_kwh))

    T = list(df_trips["Trip"].tolist())
    sl = df_trips.set_index("Trip")["SL"].to_dict()
    el = df_trips.set_index("Trip")["EL"].to_dict()
    st = df_trips.set_index("Trip")["st_blk"].to_dict()
    et = df_trips.set_index("Trip")["et_blk"].to_dict()
    st_min = df_trips.set_index("Trip")["st_min"].to_dict()
    et_min = df_trips.set_index("Trip")["et_min"].to_dict()
    epsilon = df_trips.set_index("Trip")["eps_kwh"].to_dict()

    S = expand_station_copies(CHARGING_STATIONS, STATION_COPIES)
    DEPOT = f"{CONFIG_DEPOT_NAME}_0"

    tau = {}
    tau_min = {}
    d = {}

    for i in T:
        pair = arc_from_to(DEPOT, sl[i])
        if pair is not None:
            tau[(DEPOT, i)] = pair[0]
            tau_min[(DEPOT, i)] = pair[1]
            d[(DEPOT, i)] = pair[2]
        pair = arc_from_to(el[i], DEPOT)
        if pair is not None:
            tau[(i, DEPOT)] = pair[0]
            tau_min[(i, DEPOT)] = pair[1]
            d[(i, DEPOT)] = pair[2]

    for i in T:
        for j in T:
            if i == j:
                continue
            pair = arc_from_to(el[i], sl[j])
            if pair is not None:
                tau[(i, j)] = pair[0]
                tau_min[(i, j)] = pair[1]
                d[(i, j)] = pair[2]

    for i in T:
        for h in S:
            pair = arc_from_to(el[i], h)
            if pair is not None:
                tau[(i, h)] = pair[0]
                tau_min[(i, h)] = pair[1]
                d[(i, h)] = pair[2]
            pair = arc_from_to(h, sl[i])
            if pair is not None:
                tau[(h, i)] = pair[0]
                tau_min[(h, i)] = pair[1]
                d[(h, i)] = pair[2]

    for h in S:
        pair = arc_from_to(DEPOT, h)
        if pair is not None:
            tau[(DEPOT, h)] = pair[0]
            tau_min[(DEPOT, h)] = pair[1]
            d[(DEPOT, h)] = pair[2]
        pair = arc_from_to(h, DEPOT)
        if pair is not None:
            tau[(h, DEPOT)] = pair[0]
            tau_min[(h, DEPOT)] = pair[1]
            d[(h, DEPOT)] = pair[2]

    S_use_set = set()
    for i in T:
        for h in S:
            if (i, h) in tau and (et[i] + tau[(i, h)] <= bar_t):
                S_use_set.add(h)
            if (h, i) in tau and (tau[(h, i)] <= st[i]):
                S_use_set.add(h)
    for h in S:
        if (DEPOT, h) in d or (h, DEPOT) in d:
            S_use_set.add(h)

    return {
        "T": T,
        "S_use": sorted(S_use_set),
        "DEPOT": DEPOT,
        "tau": tau,
        "tau_min": tau_min,
        "d": d,
        "st": st,
        "et": et,
        "st_min": st_min,
        "et_min": et_min,
        "sl": sl,
        "el": el,
        "epsilon": epsilon,
    }


def route_trip_ids(route: dict) -> list[int]:
    return [n for n in route["route"] if isinstance(n, int)]


def write_outputs(routes, df_trips, csv_stem: str, G: int):
    out_json = OUTDIR / f"greedy_init_only_{csv_stem}_g{G}.json"
    out_summary = OUTDIR / f"greedy_init_only_{csv_stem}_g{G}_summary.csv"

    payload = {
        "csv": f"{csv_stem}.csv",
        "G": G,
        "num_routes": len(routes),
        "num_unique_trips_covered": len({t for r in routes for t in route_trip_ids(r)}),
        "routes": routes,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    rows = []
    trip_lookup = df_trips.set_index("Trip")
    for ridx, route in enumerate(routes):
        trips = route_trip_ids(route)
        first = trips[0] if trips else None
        last = trips[-1] if trips else None
        rows.append(
            {
                "route_index": ridx,
                "num_trips": len(trips),
                "first_trip": first,
                "first_start": None if first is None else trip_lookup.loc[first, "ST"],
                "last_trip": last,
                "last_end": None if last is None else trip_lookup.loc[last, "ET"],
                "charging_activities": route.get("charging_activities", 0),
                "deadhead_kwh": route.get("deadhead_kwh", 0.0),
                "route_nodes": " -> ".join(map(str, route["route"])),
                "charge_stations": " | ".join(map(str, route["charging_stops"].get("stations", []))),
                "charge_kwh": " | ".join(f"{x:.1f}" for x in route["charging_stops"].get("kwh", [])),
            }
        )
    pd.DataFrame(rows).to_csv(out_summary, index=False)
    return out_json, out_summary


def main():
    parser = argparse.ArgumentParser(description="Run greedy EVSP initialization only")
    parser.add_argument("--csv", default="Inst_10B_RND001.csv", help="Input trip CSV in data/")
    parser.add_argument("--G", type=int, default=300, help="Battery capacity")
    parser.add_argument("--max_trip2trip", type=float, default=57)
    parser.add_argument("--max_trip2charge", type=float, default=61)
    parser.add_argument("--max_charge2trip", type=float, default=220)
    args = parser.parse_args()

    routes_csv = DATA_DIR / args.csv
    ref_dict_csv = DATA_DIR / "Ref_dict.csv"
    ref_dhd_csv = DATA_DIR / "par_ref_dhd.csv"
    if not routes_csv.exists():
        raise FileNotFoundError(routes_csv)

    print(f"[GREEDY-ONLY] CSV: {routes_csv}")
    df_trips = load_trip_dataframe(routes_csv)
    arc_data = build_arc_data(df_trips, ref_dict_csv, ref_dhd_csv)
    print(f"[GREEDY-ONLY] Trips: {len(arc_data['T'])}")
    print(f"[GREEDY-ONLY] S_use: {len(arc_data['S_use'])}")

    t0 = time.time()
    routes = build_greedy_routes(
        T=arc_data["T"],
        S_use=arc_data["S_use"],
        DEPOT=arc_data["DEPOT"],
        tau=arc_data["tau"],
        tau_min=arc_data["tau_min"],
        d=arc_data["d"],
        st=arc_data["st"],
        et=arc_data["et"],
        st_min=arc_data["st_min"],
        et_min=arc_data["et_min"],
        sl=arc_data["sl"],
        el=arc_data["el"],
        epsilon=arc_data["epsilon"],
        G=args.G,
        bar_t=bar_t,
        TB_MIN=TB_MIN,
        CHARGE_RATE_KW=CHARGE_RATE_KW,
        max_trip2trip=args.max_trip2trip,
        max_trip2charge=args.max_trip2charge,
        max_charge2trip=args.max_charge2trip,
        min_soc_fraction=0.0,
        recharge_to_fraction=1.0,
    )
    elapsed = time.time() - t0

    covered = {t for r in routes for t in route_trip_ids(r)}
    print(
        f"[GREEDY-ONLY] Built {len(routes)} routes covering "
        f"{len(covered)}/{len(arc_data['T'])} trips in {elapsed:.3f}s"
    )

    for ridx, route in enumerate(routes):
        trips = route_trip_ids(route)
        stations = route["charging_stops"].get("stations", [])
        kwhs = route["charging_stops"].get("kwh", [])
        print(
            f"  route {ridx:02d}: trips={len(trips):2d}, "
            f"charges={len(stations):2d}, deadhead={route.get('deadhead_kwh', 0.0):7.1f} kWh, "
            f"first={trips[0] if trips else '-'}, last={trips[-1] if trips else '-'}"
        )
        print(f"    nodes: {' -> '.join(map(str, route['route']))}")
        if stations:
            charge_txt = ", ".join(f"{s}:{k:.1f}kWh" for s, k in zip(stations, kwhs))
            print(f"    charge: {charge_txt}")

    out_json, out_summary = write_outputs(routes, df_trips, routes_csv.stem, args.G)
    print(f"[GREEDY-ONLY] Wrote JSON: {out_json}")
    print(f"[GREEDY-ONLY] Wrote CSV : {out_summary}")


if __name__ == "__main__":
    main()

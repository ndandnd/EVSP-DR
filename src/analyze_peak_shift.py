"""Charging-load profiles per tariff from exact-CG pools (Phase-2 evidence).

For each result JSON (with journal), takes the final LP's positive routes,
maps each incidence to its journaled charging events, and accumulates
LP-value-weighted charged kWh per hour of day. Comparing the peak tariffs
against the flat control shows whether the optimizer shifts charging load
away from expensive hours — the demand-response behavior, now expressible
because delayed-start charging is native to the pricing network.

    python analyze_peak_shift.py results/exact_peaks/*.json \
        --csv-out results/exact_peaks/peak_shift_profiles.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path


def hourly_kwh(record: dict) -> dict[int, float]:
    """Distribute each charging stop's kWh across hours, uniform in time."""
    out: dict[int, float] = {}
    stops = record.get("charging_stops") or {}
    for cst, cet, kwh in zip(stops.get("cst", []), stops.get("cet", []),
                             stops.get("kwh", [])):
        if kwh <= 0 or cet <= cst:
            continue
        span = float(cet - cst)
        t = float(cst)
        while t < cet - 1e-9:
            nxt = min((int(t // 60) + 1) * 60.0, float(cet))
            out[int(t // 60)] = out.get(int(t // 60), 0.0) + kwh * (nxt - t) / span
            t = nxt
    return out


def load_profile(result_path: Path):
    with open(result_path) as fh:
        status = json.load(fh)
    final_lp = status.get("final_lp")
    if not final_lp:
        return None
    journal = status.get("columns_journal")
    jpath = Path(journal) if journal else Path(str(result_path) + ".columns.jsonl")
    if not jpath.exists():
        jpath = result_path.parent / Path(str(jpath)).name
        if not jpath.exists():
            return None
    pool = {}
    with open(jpath) as fh:
        for line in fh:
            rec = json.loads(line)
            key = frozenset(rec["trips"])
            if key not in pool or rec["cost"] < pool[key]["cost"] - 1e-9:
                pool[key] = rec

    profile: dict[int, float] = {}
    total = 0.0
    matched = 0
    for entry in final_lp["positive_routes"]:
        rec = pool.get(frozenset(entry["trips"]))
        if rec is None:
            continue
        matched += 1
        for hour, kwh in hourly_kwh(rec).items():
            profile[hour] = profile.get(hour, 0.0) + entry["value"] * kwh
            total += entry["value"] * kwh
    return {
        "file": result_path.name,
        "route_weight": final_lp.get("route_weight"),
        "lp_obj": final_lp.get("objective"),
        "matched_routes": matched,
        "unmatched_routes": len(final_lp["positive_routes"]) - matched,
        "total_kwh": total,
        "profile": profile,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("globs", nargs="+")
    parser.add_argument("--peak-window", type=int, default=2,
                        help="Half-width (hours) of the peak window around "
                             "the tariff's peak hour.")
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = []
    for pattern in args.globs:
        for path in sorted(glob.glob(pattern)):
            p = Path(path)
            if p.name.endswith(("_mip.json",)) or ".snapshot." in p.name:
                continue
            data = load_profile(p)
            if data:
                rows.append(data)

    if not rows:
        raise SystemExit("no analyzable results (need final_lp + journal)")

    def tariff_of(name: str):
        m = re.search(r"_(flat|sek|peak(\d{2}))", name)
        if not m:
            return None, None
        return m.group(1), int(m.group(2)) if m.group(2) else None

    def instance_of(name: str) -> str:
        name = re.sub(r"_(flat|sek|peak\d{2})\.json$", "", name)
        return re.sub(r"_soc\d+(p\d+)?_b\d+\.json$", "", name)

    print(f"{'instance':<38} {'tariff':<8} {'kWh total':>10} "
          f"{'peak-window kWh':>16} {'peak %':>7}  (window = peak±"
          f"{args.peak_window}h; flat rows use each peak hour for reference)")
    by_instance: dict[str, list] = {}
    for data in rows:
        by_instance.setdefault(instance_of(data["file"]), []).append(data)

    csv_rows = []
    for instance, group in sorted(by_instance.items()):
        peaks_present = sorted({tariff_of(d["file"])[1] for d in group
                                if tariff_of(d["file"])[1] is not None})
        for data in sorted(group, key=lambda d: d["file"]):
            tariff, peak_hour = tariff_of(data["file"])
            hours = peaks_present if peak_hour is None else [peak_hour]
            for ph in hours or [None]:
                if ph is None:
                    continue
                window = range(ph - args.peak_window, ph + args.peak_window + 1)
                in_window = sum(data["profile"].get(h, 0.0) for h in window)
                pct = 100.0 * in_window / data["total_kwh"] if data["total_kwh"] else 0.0
                label = tariff if peak_hour is not None else f"flat@{ph:02d}"
                print(f"{instance:<38} {label:<8} {data['total_kwh']:>10.1f} "
                      f"{in_window:>16.1f} {pct:>6.1f}%")
                csv_rows.append({
                    "instance": instance, "tariff": tariff,
                    "reference_peak_hour": ph,
                    "total_kwh": round(data["total_kwh"], 3),
                    "peak_window_kwh": round(in_window, 3),
                    "peak_window_pct": round(pct, 3),
                    "route_weight": data["route_weight"],
                    "lp_obj": data["lp_obj"],
                })
        print()

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

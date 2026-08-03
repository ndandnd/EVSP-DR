"""Build deterministic two-duty instances for the peel-and-price sweep.

Self-sufficient: duties are extracted directly from the tracked
``data/Par_VehicleDetails_Updated.csv`` (Regular rows per VehicleTask), so no
untracked single-duty CSVs are required — the cluster checkout works as-is.
Weekday variants of the same base task (e.g. 13316m / 13316uwt) are never
paired together. Instances plus a manifest land under ``data/duty_pairs/``.

    python make_duty_pair_instances.py --pairs 20 --seed 20260803
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MASTER_CSV = DATA_DIR / "Par_VehicleDetails_Updated.csv"
OUT_DIR = DATA_DIR / "duty_pairs"

# Schema expected by run_ex_unicorn.py (same as the single-duty instances).
INSTANCE_COLUMNS = [
    "Identifier", "From1", "Start1", "End1", "To1",
    "Distance1", "Usage kWh", "count_trip_id", "Ordered_Trip_ID",
]


def _minutes(hhmm: str) -> int:
    hh, mm = str(hhmm).split(":")
    return int(hh) * 60 + int(mm)  # Hastus times may exceed 24:00


def _base_task(duty: str) -> str:
    match = re.match(r"(\d+)", str(duty))
    return match.group(1) if match else str(duty)


def load_duty_frames() -> dict[str, pd.DataFrame]:
    master = pd.read_csv(MASTER_CSV)
    regular = master[
        (master["Identifier"] == "Regular") & master["Ordered_Trip_ID"].notna()
    ].copy()
    regular["VehicleTask"] = regular["VehicleTask"].astype(str)
    regular["Ordered_Trip_ID"] = regular["Ordered_Trip_ID"].astype(int)

    frames: dict[str, pd.DataFrame] = {}
    for duty, group in regular.groupby("VehicleTask"):
        df = group[[c for c in INSTANCE_COLUMNS if c != "count_trip_id"]].copy()
        df = (df.assign(_sort=df["Start1"].map(_minutes))
                .sort_values(["_sort", "Ordered_Trip_ID"])
                .drop(columns="_sort").reset_index(drop=True))
        df["count_trip_id"] = range(len(df))
        frames[duty] = df[INSTANCE_COLUMNS]
    return frames


def merge_duties(frames: dict[str, pd.DataFrame], duties: list[str]) -> pd.DataFrame:
    merged = pd.concat([frames[d] for d in duties], ignore_index=True)
    merged = (merged.assign(_sort=merged["Start1"].map(_minutes))
                    .sort_values(["_sort", "Ordered_Trip_ID"])
                    .drop(columns="_sort").reset_index(drop=True))
    merged["count_trip_id"] = range(len(merged))
    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args(argv)

    frames = load_duty_frames()
    duties = sorted(frames)
    if len(duties) < 2:
        raise SystemExit(f"Fewer than two duties found in {MASTER_CSV}")
    print(f"{len(duties)} duties extracted from {MASTER_CSV.name}")

    candidates = [
        (a, b)
        for i, a in enumerate(duties)
        for b in duties[i + 1:]
        if _base_task(a) != _base_task(b)
    ]
    rng = random.Random(args.seed)
    chosen = rng.sample(candidates, min(args.pairs, len(candidates)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, names = [], []
    for a, b in sorted(chosen):
        merged = merge_duties(frames, [a, b])
        name = f"Practice_Custom_DutyPair_{a}_{b}.csv"
        path = OUT_DIR / name
        merged.to_csv(path, index=False)
        manifest.append({
            "csv": name, "duty_a": a, "duty_b": b, "trips": len(merged),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        names.append(name)
        print(f"wrote {name}: {len(merged)} trips")

    with open(OUT_DIR / "manifest.json", "w") as fh:
        json.dump({"seed": args.seed, "source": MASTER_CSV.name,
                   "pairs": manifest}, fh, indent=1)
    (OUT_DIR / "pairs.txt").write_text("\n".join(names) + "\n")
    print(f"{len(names)} pair instances under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

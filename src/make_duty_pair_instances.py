"""Build deterministic two-duty instances from tracked single-duty CSVs.

Samples duty pairs (never pairing weekday variants of the same base task),
concatenates their trips chronologically, re-indexes ``count_trip_id``, and
writes instances plus a manifest under ``data/duty_pairs/``.

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
OUT_DIR = DATA_DIR / "duty_pairs"
SINGLE_PREFIX = "Practice_Custom_SingleDuty_"


def _minutes(hhmm: str) -> int:
    hh, mm = str(hhmm).split(":")
    return int(hh) * 60 + int(mm)


def _base_task(duty: str) -> str:
    # 13316m / 13316uwt / 13324muw / 13324t are weekday variants of one base.
    match = re.match(r"(\d+)", duty)
    return match.group(1) if match else duty


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args(argv)

    duties = sorted(
        p.stem[len(SINGLE_PREFIX):]
        for p in DATA_DIR.glob(f"{SINGLE_PREFIX}*.csv")
    )
    if len(duties) < 2:
        raise SystemExit("No tracked single-duty CSVs found under data/.")

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
        frames = [pd.read_csv(DATA_DIR / f"{SINGLE_PREFIX}{d}.csv") for d in (a, b)]
        merged = pd.concat(frames, ignore_index=True)
        merged = (merged.assign(_sort=merged["Start1"].map(_minutes))
                        .sort_values(["_sort", "Ordered_Trip_ID"])
                        .drop(columns="_sort").reset_index(drop=True))
        merged["count_trip_id"] = range(len(merged))
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
        json.dump({"seed": args.seed, "pairs": manifest}, fh, indent=1)
    (OUT_DIR / "pairs.txt").write_text("\n".join(names) + "\n")
    print(f"{len(names)} pair instances under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Select audited PREEMPTED rows with no published status artifact."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-scale", type=int, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = root / "medium_event_summary.csv"
    if not summary.is_file():
        raise SystemExit(f"missing completed campaign audit: {summary}")
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        if (
            row.get("slurm_state") == "PREEMPTED"
            and row.get("result_present") == "False"
        ):
            if (
                args.expected_scale is not None
                and int(row["scale"]) != args.expected_scale
            ):
                raise SystemExit(
                    "unexpected preempted scale at index " + row["index"]
                )
            selected.append(int(row["index"]))
    for index in sorted(selected):
        print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

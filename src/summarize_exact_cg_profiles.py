#!/usr/bin/env python3
"""Summarize exact-CG prefix profiles as one read-only TSV or JSON table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = (
    "label", "prefix", "method", "outcome", "successful_repetitions",
    "requested_repetitions", "median_total_s", "min_total_s", "max_total_s",
    "median_backend_s", "objective", "route_weight", "artificials",
    "max_row_violation", "max_bound_violation", "peak_rss_bytes",
    "failure_count", "failures", "source_unchanged",
)


def summarize(campaign_root: Path) -> list[dict]:
    root = campaign_root.expanduser().resolve()
    manifest = json.loads((root / "campaign.json").read_text())
    rows = []
    for job in manifest.get("jobs") or []:
        label = job["label"]
        output = Path(job["output"])
        if not output.is_file():
            rows.append({
                "label": label,
                "outcome": "missing_output",
                "failure_count": 1,
                "failures": "output does not exist",
            })
            continue
        try:
            payload = json.loads(output.read_text())
        except (OSError, ValueError) as exc:
            rows.append({
                "label": label,
                "outcome": "invalid_output",
                "failure_count": 1,
                "failures": repr(exc),
            })
            continue
        for prefix in payload.get("profiles") or []:
            prefix_columns = prefix.get("prefix_columns")
            if not prefix.get("available"):
                rows.append({
                    "label": label,
                    "prefix": prefix_columns,
                    "outcome": "unavailable",
                    "failure_count": 1,
                    "failures": prefix.get("reason"),
                    "source_unchanged": payload.get("source_unchanged"),
                })
                continue
            for method in prefix.get("methods") or []:
                repetitions = method.get("repetitions") or []
                failures = [
                    repetition.get("error", "unknown failure")
                    for repetition in repetitions
                    if repetition.get("outcome") != "ok"
                ]
                rss = [
                    int(repetition["peak_rss_bytes"])
                    for repetition in repetitions
                    if repetition.get("peak_rss_bytes") is not None
                ]
                timing = method.get("timing") or {}
                solution = method.get("solution") or {}
                rows.append({
                    "label": label,
                    "prefix": prefix_columns,
                    "method": method.get("method"),
                    "outcome": method.get("outcome"),
                    "successful_repetitions": method.get(
                        "successful_repetitions"
                    ),
                    "requested_repetitions": method.get(
                        "requested_repetitions"
                    ),
                    "median_total_s": timing.get("total_median_s"),
                    "min_total_s": timing.get("total_min_s"),
                    "max_total_s": timing.get("total_max_s"),
                    "median_backend_s": timing.get("backend_median_s"),
                    "objective": solution.get("objective"),
                    "route_weight": solution.get("route_weight"),
                    "artificials": solution.get("artificial_total"),
                    "max_row_violation": solution.get("max_row_violation"),
                    "max_bound_violation": solution.get(
                        "max_bound_violation"
                    ),
                    "peak_rss_bytes": max(rss) if rss else None,
                    "failure_count": len(failures),
                    "failures": " | ".join(failures),
                    "source_unchanged": payload.get("source_unchanged"),
                })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--format", choices=("tsv", "json"), default="tsv")
    args = parser.parse_args(argv)
    rows = summarize(args.campaign_root)
    if args.format == "json":
        print(json.dumps(rows, indent=2))
        return 0
    print("\t".join(FIELDS))
    for row in rows:
        print("\t".join(
            "" if row.get(field) is None else str(row[field])
            for field in FIELDS
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

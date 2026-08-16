#!/usr/bin/env python3
"""Read-only publication/recovery monitor for k40 factorial MIP campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recover_k40_factorial_mip_campaign import build_recovery_plan


def monitor(campaign_root: Path) -> list[dict]:
    plan, _prepared = build_recovery_plan(campaign_root)
    rows = []
    for job in plan["jobs"]:
        state = job["publication_state"]
        if state == "complete_valid":
            outcome = "complete_valid_output"
        elif job["recoverable"]:
            outcome = "recoverable_validated_raw"
        elif state == "incomplete_publication":
            outcome = "incomplete_publication"
        else:
            outcome = "missing_or_invalid_result"
        rows.append({
            **job,
            "outcome": outcome,
            "recovery_commit": plan["recovery_commit"],
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "tsv"), default="tsv")
    args = parser.parse_args(argv)
    rows = monitor(args.campaign_root)
    if args.format == "json":
        print(json.dumps(rows, indent=2))
        return 0
    fields = (
        "label", "job_id", "outcome", "publication_state",
        "recoverable", "recovery_method", "candidate_path",
        "raw_sha256", "recovered_result_sha256", "errors",
    )
    print("\t".join(fields))
    for row in rows:
        print("\t".join(
            " | ".join(row[field]) if field == "errors"
            else ("" if row.get(field) is None else str(row[field]))
            for field in fields
        ))
    return 0 if all(
        row["outcome"] in {
            "complete_valid_output", "recoverable_validated_raw"
        }
        for row in rows
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())

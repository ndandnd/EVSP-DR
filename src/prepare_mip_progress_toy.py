#!/usr/bin/env python3
"""Prepare a physical pool without a complete singleton MIP start."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from durable_io import atomic_write_json, read_jsonl_records
from run_exact_pool_mip import resolve_pool_journal


def prepare(source, output, witness_result=None):
    source = Path(source).resolve()
    output = Path(output).resolve()
    status = json.loads(source.read_text())
    records = read_jsonl_records(
        resolve_pool_journal(source, status), repair_trailing=False,
    )
    trips = list(status["trip_ids"])
    retained = [record for record in records if len(record["trips"]) > 1]
    witness_indices = []
    if witness_result is not None:
        witness = json.loads(Path(witness_result).read_text())
        witness_indices = [
            int(index) for index in (
                witness.get("selected_route_indices")
                or (witness.get("final") or {}).get(
                    "selected_route_indices", []
                )
            )
        ]
        retained.extend(records[index] for index in witness_indices)
    covered = Counter(
        trip for record in retained for trip in record["trips"]
    )
    missing = [trip for trip in trips if not covered[trip]]
    singleton = {
        record["trips"][0]: record
        for record in records if len(record["trips"]) == 1
    }
    retained.extend(singleton[trip] for trip in missing)
    if all(any(record["trips"] == [trip] for record in retained)
           for trip in trips):
        raise RuntimeError("toy unexpectedly retains a complete singleton start")
    unique = {}
    for record in retained:
        key = frozenset(record["trips"])
        if key not in unique or record["cost"] < unique[key]["cost"]:
            unique[key] = record
    retained = list(unique.values())
    journal = Path(str(output) + ".columns.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("x") as handle:
        for record in retained:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    status.update({
        "columns": len(retained),
        "columns_journal": str(journal),
        "certified": False,
        "certified_rc_optimal": False,
        "stop_reason": "mip_progress_toy",
        "final": None,
        "final_lp": None,
    })
    atomic_write_json(output, status)
    return len(retained), len(missing), len(witness_indices)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--witness-result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    columns, missing, witness = prepare(
        args.source, args.out, args.witness_result,
    )
    print(json.dumps({
        "columns": columns, "singleton_fallbacks": missing,
        "witness_routes": witness,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict physical replay gate for a completed event-pricer journal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from durable_io import atomic_write_json
from run_exact_pool_mip import prepare_strict_partition_pool


SCHEMA = "evsp-dr-event-pricer-physical-witness-audit-v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_jsonl(payload: bytes) -> list[dict]:
    records = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"journal line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"journal line {line_number} is not a JSON object"
            )
        records.append(record)
    return records


def _positive_witnesses(status: dict, accepted: list[dict]) -> list[dict]:
    positive = ((status.get("final_lp") or {}).get("positive_routes") or [])
    witnesses = []
    for selected in positive:
        trips = list(selected.get("trips") or [])
        cost = float(selected["cost"])
        match = next(
            (
                record for record in accepted
                if list(record.get("trips") or []) == trips
                and math.isclose(
                    float(record["cost"]), cost,
                    rel_tol=1e-10, abs_tol=1e-6,
                )
            ),
            None,
        )
        if match is None:
            raise ValueError(
                "final positive LP route is absent from the replayed pool"
            )
        physical = match.get("physical_realization") or {}
        witnesses.append({
            "trips": trips,
            "value": float(selected["value"]),
            "cost": cost,
            "physical_status": physical.get("status"),
            "realization_schema": physical.get("schema"),
            "realization_mapping_sha256":
                physical.get("mapping_sha256"),
            "charging_blocks_sha256": physical.get(
                "continuous_realized_charging_blocks_sha256"
            ),
        })
    return witnesses


def audit(status_path: Path, output_path: Path, *, data_dir=None) -> dict:
    status_path = status_path.expanduser().resolve(strict=True)
    output_path = output_path.expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    status_bytes = status_path.read_bytes()
    status = json.loads(status_bytes)
    if status.get("time_model") != "event":
        raise ValueError("physical witness audit requires an event status")
    journal_path = Path(status["columns_journal"]).expanduser().resolve(
        strict=True
    )
    journal_bytes = journal_path.read_bytes()
    routes = _load_jsonl(journal_bytes)
    accepted, physical_audit = prepare_strict_partition_pool(
        status,
        routes,
        data_dir=data_dir,
    )
    witnesses = _positive_witnesses(status, accepted)
    unchanged = (
        status_path.read_bytes() == status_bytes
        and journal_path.read_bytes() == journal_bytes
    )
    gate_passed = (
        unchanged
        and status.get("certified_rc_optimal") is True
        and physical_audit["total_columns"] == len(routes)
        and physical_audit["accepted_columns"] == len(routes)
        and physical_audit["rejected_columns"] == 0
        and len(witnesses)
        == len(((status.get("final_lp") or {}).get("positive_routes") or []))
    )
    result = {
        "schema": SCHEMA,
        "gate_passed": gate_passed,
        "source_status": str(status_path),
        "source_status_sha256": _sha256(status_bytes),
        "source_journal": str(journal_path),
        "source_journal_sha256": _sha256(journal_bytes),
        "source_inputs_unchanged_during_audit": unchanged,
        "producer_git_commit":
            (status.get("provenance") or {}).get("git_commit"),
        "source_certified_rc_optimal":
            status.get("certified_rc_optimal"),
        "physical_pool_audit": physical_audit,
        "positive_lp_witnesses": witnesses,
    }
    atomic_write_json(output_path, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args(argv)
    result = audit(args.status, args.out, data_dir=args.data_dir)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

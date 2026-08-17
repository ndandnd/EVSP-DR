#!/usr/bin/env python3
"""Read-only physical audit for exact expanded-network column journals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from audit_giro_known_columns import HORIZON_MIN, build_problem
from config import CHARGING_STATIONS
from durable_io import read_jsonl_records
from expanded_path_realization import (
    _arc_map,
    realize_expanded_path,
    realized_costs,
)
from run_exact_pool_mip import (
    resolve_pool_journal,
    validate_injected_route,
)
from utils_v2 import load_station_hourly_prices
import scipy


AUDIT_SCHEMA = "evsp-dr-expanded-pool-physical-audit-v1"
ROUTE_FIELDS = (
    "pool", "ordinal", "incidence_sha256", "trip_count",
    "selected_in_solver_incumbent", "selected_route_ordinal",
    "classification", "recorded_replay_valid", "recorded_replay_reason",
    "recorded_total_kwh", "realized_total_kwh",
    "discarded_grid_residual_kwh", "mapping_sha256",
    "stored_expanded_grid_cost", "recomputed_expanded_grid_cost",
    "continuous_realized_cost", "expanded_minus_realized_cost",
)
POOL_FIELDS = (
    "pool", "status_sha256", "journal_sha256", "journal_records",
    "mip_unique_columns", "selected_incumbent_routes",
    "valid_as_recorded", "deterministically_repairable",
    "infeasible_after_realization", "mapping_set_sha256",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def _selected_indices(campaign_root: Path | None, pool: str):
    if campaign_root is None:
        return {}
    latest = campaign_root / "progress" / pool / "latest.json"
    if not latest.is_file():
        return {}
    payload = json.loads(latest.read_text())
    incumbent = payload.get("incumbent") or {}
    return {
        int(index): ordinal
        for ordinal, index in enumerate(
            incumbent.get("selected_route_indices") or [], start=1
        )
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=ROUTE_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def audit_pools(
    statuses: list[Path],
    *,
    output_dir: Path,
    reference_data_dir: Path,
    campaign_root: Path | None = None,
    archive_sha256: str | None = None,
    route_detail: str = "full",
) -> dict:
    output = output_dir.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"audit output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        dir=output.parent, prefix=f".{output.name}.tmp."
    ))
    route_rows = []
    pool_summaries = []
    try:
        for raw_status in statuses:
            status_path = raw_status.expanduser().resolve()
            status_raw = status_path.read_bytes()
            status = json.loads(status_raw)
            journal_path = resolve_pool_journal(status_path, status)
            journal_before = _sha(journal_path)
            records = read_jsonl_records(
                journal_path, repair_trailing=False
            )
            data_dir = status_path.parent / "data"
            problem = build_problem(
                data_dir,
                status["csv"],
                max_station_to_trip_wait_min=HORIZON_MIN,
                reference_data_dir=reference_data_dir,
            )
            arc_map = _arc_map(problem)
            prices = load_station_hourly_prices(
                data_dir / Path(status["prices_csv"]).name,
                CHARGING_STATIONS,
            )
            g_kwh = float(status["g_kwh"])
            charge_kw = float(status["charge_kw"])
            reserve_kwh = float(status["min_soc_frac"]) * g_kwh
            soc_step = float(status["soc_step"])
            block_min = int(status["block_min"])
            pool = status_path.parent.name
            selected = _selected_indices(campaign_root, pool)
            mip_pool = {}
            for raw_ordinal, candidate in enumerate(records, start=1):
                key = frozenset(candidate.get("trips") or [])
                candidate_cost = float(candidate["cost"])
                if (
                    key not in mip_pool
                    or candidate_cost
                    < float(mip_pool[key][0]["cost"]) - 1e-9
                ):
                    mip_pool[key] = (candidate, raw_ordinal)
            mip_pool_values = list(mip_pool.values())
            selected_raw_ordinals = {
                mip_pool_values[route_index][1]: selected_ordinal
                for route_index, selected_ordinal in selected.items()
            }
            counts = Counter()
            rejection_samples = []
            selected_failures = []
            mapping_hashes = []
            for ordinal, record in enumerate(records, start=1):
                trips = list(record.get("trips") or [])
                incidence_sha = _canonical_sha(sorted(trips))
                recorded_reason = validate_injected_route(
                    problem, record, g_kwh, charge_kw,
                    reserve_kwh, HORIZON_MIN, arc_map=arc_map,
                )
                realized, detail = realize_expanded_path(
                    problem,
                    record,
                    g_kwh=g_kwh,
                    charge_kw=charge_kw,
                    reserve_kwh=reserve_kwh,
                    soc_step=soc_step,
                    block_min=block_min,
                    arc_map=arc_map,
                )
                realized_reason = None
                costs = {}
                mapping = detail.get("mapping")
                if realized is not None:
                    realized_reason = validate_injected_route(
                        problem, realized, g_kwh, charge_kw,
                        reserve_kwh, HORIZON_MIN, arc_map=arc_map,
                    )
                    if realized_reason is None:
                        costs = realized_costs(
                            record, mapping, station_prices=prices
                        )
                if recorded_reason is None:
                    classification = "valid_as_recorded"
                elif realized is not None and realized_reason is None:
                    classification = "deterministically_repairable"
                else:
                    classification = "infeasible_after_realization"
                counts[classification] += 1
                if mapping is not None:
                    mapping_hashes.append(mapping["mapping_sha256"])
                selected_ordinal = selected_raw_ordinals.get(ordinal)
                if selected_ordinal is not None and recorded_reason is not None:
                    selected_failures.append({
                        "selected_route_ordinal": selected_ordinal,
                        "pool_route_index": ordinal - 1,
                        "incidence_sha256": incidence_sha,
                        "recorded_reason": recorded_reason,
                        "classification": classification,
                        "mapping_sha256": (
                            mapping.get("mapping_sha256")
                            if mapping else None
                        ),
                        "soc_trace": (
                            mapping.get("trace") if mapping else None
                        ),
                        "costs": costs or None,
                    })
                if (
                    classification == "infeasible_after_realization"
                    and len(rejection_samples) < 100
                ):
                    rejection_samples.append({
                        "ordinal": ordinal,
                        "trips": trips,
                        "recorded_reason": recorded_reason,
                        "realization_reason": (
                            realized_reason or detail.get("reason")
                        ),
                    })
                recorded_kwh = sum(float(value) for value in (
                    (record.get("charging_stops") or {}).get("kwh") or []
                ))
                route_row = {
                    "pool": pool,
                    "ordinal": ordinal,
                    "incidence_sha256": incidence_sha,
                    "trip_count": len(trips),
                    "selected_in_solver_incumbent":
                        selected_ordinal is not None,
                    "selected_route_ordinal": selected_ordinal,
                    "classification": classification,
                    "recorded_replay_valid": recorded_reason is None,
                    "recorded_replay_reason": recorded_reason,
                    "recorded_total_kwh": recorded_kwh,
                    "realized_total_kwh": (
                        mapping.get("realized_total_kwh")
                        if mapping else None
                    ),
                    "discarded_grid_residual_kwh": (
                        mapping.get("discarded_grid_residual_kwh")
                        if mapping else None
                    ),
                    "mapping_sha256": (
                        mapping.get("mapping_sha256")
                        if mapping else None
                    ),
                    "stored_expanded_grid_cost":
                        costs.get("stored_expanded_grid_cost"),
                    "recomputed_expanded_grid_cost":
                        costs.get("recomputed_expanded_grid_cost"),
                    "continuous_realized_cost":
                        costs.get("continuous_realized_cost"),
                    "expanded_minus_realized_cost":
                        costs.get("expanded_minus_realized_cost"),
                }
                if (
                    route_detail == "full"
                    or route_detail == "selected"
                    and selected_ordinal is not None
                ):
                    route_rows.append(route_row)
            if _sha(journal_path) != journal_before:
                raise ValueError(f"journal changed during audit: {journal_path}")
            pool_summaries.append({
                "pool": pool,
                "status_path": str(status_path),
                "status_sha256": hashlib.sha256(status_raw).hexdigest(),
                "journal_path": str(journal_path),
                "journal_sha256": journal_before,
                "journal_records": len(records),
                "mip_unique_columns": len(mip_pool),
                "selected_incumbent_routes": len(selected),
                "counts": dict(sorted(counts.items())),
                "selected_recorded_failures": selected_failures,
                "rejection_samples": rejection_samples,
                "mapping_set_sha256": _canonical_sha(sorted(mapping_hashes)),
                "physics": {
                    "g_kwh": g_kwh,
                    "charge_kw": charge_kw,
                    "reserve_kwh": reserve_kwh,
                    "soc_step": soc_step,
                    "block_min": block_min,
                },
            })
        report = {
            "schema": AUDIT_SCHEMA,
            "archive_sha256": archive_sha256,
            "read_only": True,
            "pricing_certificate_conclusion": (
                "Existing certified_rc_optimal applies only to the "
                "conservative expanded-grid cost model. Continuous realized "
                "costs are not exact-priced and have no global reduced-cost "
                "certificate."
            ),
            "master_cost_policy": (
                "Keep stored expanded-grid cost unchanged; report continuous "
                "realized cost separately."
            ),
            "pools": pool_summaries,
            "audit_provenance": {
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parents[1],
                    text=True,
                    capture_output=True,
                    check=True,
                ).stdout.strip(),
                "python": platform.python_version(),
                "scipy": scipy.__version__,
                "implementation_sha256": {
                    name: _sha(Path(__file__).resolve().parent / name)
                    for name in (
                        "audit_expanded_pool_physical.py",
                        "expanded_path_realization.py",
                        "run_exact_pool_mip.py",
                    )
                },
            },
        }
        _write_csv(staging / "route_audit.csv", route_rows)
        with (staging / "pool_summary.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=POOL_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for pool in pool_summaries:
                writer.writerow({
                    "pool": pool["pool"],
                    "status_sha256": pool["status_sha256"],
                    "journal_sha256": pool["journal_sha256"],
                    "journal_records": pool["journal_records"],
                    "mip_unique_columns": pool["mip_unique_columns"],
                    "selected_incumbent_routes":
                        pool["selected_incumbent_routes"],
                    "valid_as_recorded":
                        pool["counts"].get("valid_as_recorded", 0),
                    "deterministically_repairable":
                        pool["counts"].get(
                            "deterministically_repairable", 0
                        ),
                    "infeasible_after_realization":
                        pool["counts"].get(
                            "infeasible_after_realization", 0
                        ),
                    "mapping_set_sha256": pool["mapping_set_sha256"],
                })
            handle.flush()
            os.fsync(handle.fileno())
        (staging / "audit.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        markdown = [
            "# Expanded-network charging realization audit",
            "",
            "The audit is read-only. Trip order, station choice, and charging "
            "windows are preserved.",
            "",
            "## Root cause",
            "",
            "The expanded network floors SOC to its lattice at transitions. "
            "The old extractor emitted the full lattice charging gain, while "
            "continuous replay retained discarded residual SOC. Repeated "
            "residuals therefore made recorded schedules overfill the battery.",
            "",
            "## Cost and certificate conclusion",
            "",
            report["pricing_certificate_conclusion"],
            "",
            "| Pool | Journal rows | Unique columns | Valid recorded | "
            "Repairable | Infeasible |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for pool in pool_summaries:
            count = pool["counts"]
            markdown.append(
                f"| {pool['pool']} | {pool['journal_records']} | "
                f"{pool['mip_unique_columns']} | "
                f"{count.get('valid_as_recorded', 0)} | "
                f"{count.get('deterministically_repairable', 0)} | "
                f"{count.get('infeasible_after_realization', 0)} |"
            )
        (staging / "ROOT_CAUSE.md").write_text("\n".join(markdown) + "\n")
        members = {
            path.name: _sha(path)
            for path in sorted(staging.iterdir()) if path.is_file()
        }
        (staging / "completion.json").write_text(json.dumps({
            "schema": "evsp-dr-expanded-pool-audit-completion-v1",
            "members": members,
        }, indent=2, sort_keys=True) + "\n")
        output.mkdir(mode=0o755)
        for path in sorted(staging.iterdir()):
            if path.name == "completion.json":
                continue
            os.link(path, output / path.name)
        output_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(output_fd)
        os.link(
            staging / "completion.json",
            output / "completion.json",
        )
        os.fsync(output_fd)
        os.close(output_fd)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-data-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument(
        "--route-detail",
        choices=("full", "selected", "none"),
        default="full",
    )
    args = parser.parse_args(argv)
    report = audit_pools(
        args.status,
        output_dir=args.out_dir,
        reference_data_dir=args.reference_data_dir,
        campaign_root=args.campaign_root,
        archive_sha256=args.archive_sha256,
        route_detail=args.route_detail,
    )
    print(json.dumps({
        "schema": report["schema"],
        "pools": [
            {"pool": pool["pool"], "counts": pool["counts"]}
            for pool in report["pools"]
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

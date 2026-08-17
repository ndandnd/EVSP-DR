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
    "expanded_grid_cost_matches_stored",
    "continuous_realized_cost", "expanded_minus_realized_cost",
)
POOL_FIELDS = (
    "pool", "status_sha256", "journal_sha256", "reference_sha256",
    "deadhead_sha256", "journal_records",
    "archived_mip_unique_columns", "mip_unique_columns",
    "selected_incumbent_routes",
    "valid_as_recorded", "deterministically_repairable",
    "infeasible_after_realization", "mapping_set_sha256",
    "mip_unique_valid_as_recorded",
    "mip_unique_deterministically_repairable",
    "mip_unique_infeasible_after_realization",
    "expanded_grid_cost_mismatch_count",
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
        return {}, None
    latest = campaign_root / "progress" / pool / "latest.json"
    if not latest.is_file():
        return {}, None
    raw = latest.read_bytes()
    payload = json.loads(raw)
    if (
        payload.get("schema") != "evsp-dr-mip-convergence-v1"
        or payload.get("kind") != "latest"
    ):
        raise ValueError(f"invalid latest checkpoint payload: {latest}")
    incumbent = payload.get("incumbent") or {}
    return {
        int(index): ordinal
        for ordinal, index in enumerate(
            incumbent.get("selected_route_indices") or [], start=1
        )
    }, {
        "path": str(latest),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "metadata": payload.get("metadata") or {},
        "route_vector_sha256": incumbent.get("route_vector_sha256"),
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
    archive_path: Path | None = None,
    route_detail: str = "full",
    expected_pools: set[str] | None = None,
) -> dict:
    output = output_dir.expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"audit output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        dir=output.parent, prefix=f".{output.name}.tmp."
    ))
    route_rows = []
    pool_summaries = []
    observed_pool_names = set()
    if archive_sha256 is not None and archive_path is None:
        raise ValueError(
            "archive_sha256 requires archive_path for verification"
        )
    if archive_path is not None:
        observed_archive_sha = _sha(archive_path.expanduser().resolve())
        if (
            archive_sha256 is not None
            and observed_archive_sha != archive_sha256
        ):
            raise ValueError("archive SHA-256 mismatch")
        archive_sha256 = observed_archive_sha
    try:
        for raw_status in sorted(
            statuses, key=lambda path: str(path.expanduser().resolve())
        ):
            status_path = raw_status.expanduser().resolve()
            status_raw = status_path.read_bytes()
            status = json.loads(status_raw)
            journal_path = resolve_pool_journal(status_path, status)
            journal_before = _sha(journal_path)
            records = read_jsonl_records(
                journal_path, repair_trailing=False
            )
            data_dir = status_path.parent / "data"
            instance_path = data_dir / status["csv"]
            tariff_path = data_dir / Path(status["prices_csv"]).name
            reference_path = (
                reference_data_dir.expanduser().resolve() / "Ref_dict.csv"
            )
            deadhead_path = (
                reference_data_dir.expanduser().resolve()
                / "par_ref_dhd.csv"
            )
            immutable_hashes = {
                "status": hashlib.sha256(status_raw).hexdigest(),
                "journal": journal_before,
                "instance": _sha(instance_path),
                "tariff": _sha(tariff_path),
                "reference": _sha(reference_path),
                "deadhead": _sha(deadhead_path),
            }
            provenance = status.get("provenance") or {}
            if (
                provenance.get("instance_sha256")
                != immutable_hashes["instance"]
                or provenance.get("prices_sha256")
                != immutable_hashes["tariff"]
            ):
                raise ValueError(
                    f"status data provenance mismatch: {status_path}"
                )
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
            if pool in observed_pool_names:
                raise ValueError(f"duplicate pool supplied: {pool}")
            observed_pool_names.add(pool)
            selected, selected_source = _selected_indices(
                campaign_root, pool
            )
            if selected_source is not None:
                metadata = selected_source["metadata"]
                status_sha = hashlib.sha256(status_raw).hexdigest()
                if (
                    metadata.get("source_result_sha256") != status_sha
                    or metadata.get("source_journal_sha256")
                    != journal_before
                    or (
                        selected
                        and selected_source["route_vector_sha256"]
                        != _canonical_sha(sorted(selected))
                    )
                ):
                    raise ValueError(
                        f"latest incumbent source mismatch for {pool}"
                    )
            archived_mip_pool = {}
            for raw_ordinal, candidate in enumerate(records, start=1):
                key = frozenset(candidate.get("trips") or [])
                candidate_cost = float(candidate["cost"])
                if (
                    key not in archived_mip_pool
                    or candidate_cost
                    < float(archived_mip_pool[key][0]["cost"]) - 1e-9
                ):
                    archived_mip_pool[key] = (candidate, raw_ordinal)
            archived_mip_pool_values = list(archived_mip_pool.values())
            selected_raw_ordinals = {
                archived_mip_pool_values[route_index][1]: selected_ordinal
                for route_index, selected_ordinal in selected.items()
            }
            counts = Counter()
            rejection_samples = []
            selected_failures = []
            mapping_hashes = []
            classification_by_ordinal = {}
            grid_cost_mismatches = 0
            admitted_pool = {}
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
                        if not math.isclose(
                            costs["stored_expanded_grid_cost"],
                            costs["recomputed_expanded_grid_cost"],
                            rel_tol=1e-9,
                            abs_tol=1e-6,
                        ):
                            grid_cost_mismatches += 1
                cost_matches = (
                    bool(costs)
                    and math.isclose(
                        costs["stored_expanded_grid_cost"],
                        costs["recomputed_expanded_grid_cost"],
                        rel_tol=1e-9,
                        abs_tol=1e-6,
                    )
                )
                if recorded_reason is None:
                    classification = "valid_as_recorded"
                elif realized is not None and realized_reason is None:
                    classification = "deterministically_repairable"
                else:
                    classification = "infeasible_after_realization"
                counts[classification] += 1
                classification_by_ordinal[ordinal] = classification
                if (
                    realized is not None
                    and realized_reason is None
                    and cost_matches
                ):
                    key = frozenset(trips)
                    if (
                        key not in admitted_pool
                        or float(record["cost"])
                        < float(admitted_pool[key][0]["cost"]) - 1e-9
                    ):
                        admitted_pool[key] = (
                            record, ordinal, classification
                        )
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
                    "expanded_grid_cost_matches_stored": (
                        math.isclose(
                            costs["stored_expanded_grid_cost"],
                            costs["recomputed_expanded_grid_cost"],
                            rel_tol=1e-9,
                            abs_tol=1e-6,
                        )
                        if costs else None
                    ),
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
            after_hashes = {
                "status": _sha(status_path),
                "journal": _sha(journal_path),
                "instance": _sha(instance_path),
                "tariff": _sha(tariff_path),
                "reference": _sha(reference_path),
                "deadhead": _sha(deadhead_path),
            }
            if (
                selected_source is not None
                and _sha(Path(selected_source["path"]))
                != selected_source["sha256"]
            ):
                raise ValueError(
                    f"latest incumbent changed during audit: {pool}"
                )
            if after_hashes != immutable_hashes:
                raise ValueError(
                    f"source changed during audit: {status_path}"
                )
            pool_summaries.append({
                "pool": pool,
                "status_path": str(status_path),
                "status_sha256": hashlib.sha256(status_raw).hexdigest(),
                "journal_path": str(journal_path),
                "journal_sha256": journal_before,
                "instance_sha256": immutable_hashes["instance"],
                "tariff_sha256": immutable_hashes["tariff"],
                "reference_sha256": immutable_hashes["reference"],
                "deadhead_sha256": immutable_hashes["deadhead"],
                "journal_records": len(records),
                "archived_mip_unique_columns":
                    len(archived_mip_pool),
                "mip_unique_columns": len(admitted_pool),
                "selected_incumbent_routes": len(selected),
                "selected_incumbent_source": selected_source,
                "counts": dict(sorted(counts.items())),
                "mip_unique_counts": dict(sorted(Counter(
                    classification
                    for _record, _raw_ordinal, classification
                    in admitted_pool.values()
                ).items())),
                "selected_recorded_failures": selected_failures,
                "rejection_samples": rejection_samples,
                "mapping_set_sha256": _canonical_sha(sorted(mapping_hashes)),
                "expanded_grid_cost_mismatch_count":
                    grid_cost_mismatches,
                "physics": {
                    "g_kwh": g_kwh,
                    "charge_kw": charge_kw,
                    "reserve_kwh": reserve_kwh,
                    "soc_step": soc_step,
                    "block_min": block_min,
                },
            })
            if (
                status.get("columns") is not None
                and int(status["columns"]) != len(archived_mip_pool)
            ):
                raise ValueError(
                    f"status column count mismatch for {pool}: "
                    f"{status['columns']} != {len(archived_mip_pool)}"
                )
        if expected_pools is not None and observed_pool_names != expected_pools:
            raise ValueError(
                "pool set mismatch: "
                f"{sorted(observed_pool_names)} != {sorted(expected_pools)}"
            )
        report = {
            "schema": AUDIT_SCHEMA,
            "archive_sha256": archive_sha256,
            "read_only": True,
            "route_detail": route_detail,
            "expected_pools": (
                sorted(expected_pools)
                if expected_pools is not None else None
            ),
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
                    "reference_sha256": pool["reference_sha256"],
                    "deadhead_sha256": pool["deadhead_sha256"],
                    "journal_records": pool["journal_records"],
                    "mip_unique_columns": pool["mip_unique_columns"],
                    "archived_mip_unique_columns":
                        pool["archived_mip_unique_columns"],
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
                    "mip_unique_valid_as_recorded":
                        pool["mip_unique_counts"].get(
                            "valid_as_recorded", 0
                        ),
                    "mip_unique_deterministically_repairable":
                        pool["mip_unique_counts"].get(
                            "deterministically_repairable", 0
                        ),
                    "mip_unique_infeasible_after_realization":
                        pool["mip_unique_counts"].get(
                            "infeasible_after_realization", 0
                        ),
                    "expanded_grid_cost_mismatch_count":
                        pool["expanded_grid_cost_mismatch_count"],
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
            "| Pool | Journal rows | Admitted unique | Valid recorded | "
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
    parser.add_argument("--expected-pool", action="append", default=[])
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--archive", type=Path)
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
        archive_path=args.archive,
        route_detail=args.route_detail,
        expected_pools=set(args.expected_pool) if args.expected_pool else None,
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

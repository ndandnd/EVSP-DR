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
import tarfile
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
RAW_K40_POOLS = {
    "k40_r1_ca_raw_m1440",
    "k40_r1_cs_raw_m1440",
    "k40_r2_ca_raw_m1440",
    "k40_r2_cs_raw_m1440",
}
ROUTE_FIELDS = (
    "pool", "ordinal", "incidence_sha256", "trip_count",
    "selected_in_solver_incumbent", "selected_route_ordinal",
    "classification", "recorded_replay_valid", "recorded_replay_reason",
    "recorded_total_kwh", "realized_total_kwh",
    "discarded_grid_residual_kwh", "mapping_sha256",
    "continuous_realized_charging_blocks_sha256",
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
    "cost_unavailable_count", "physical_admission_rejected_records",
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


def _archive_member_hashes(path: Path) -> dict[str, str]:
    members = {}
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"archive member unreadable: {member.name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            members[member.name] = digest.hexdigest()
    return members


def _archive_relative(path: Path) -> str:
    parts = path.resolve().parts
    try:
        index = parts.index("src")
    except ValueError as exc:
        raise ValueError(f"audited path is not under src/: {path}") from exc
    return "/".join(parts[index:])


def _selected_indices(campaign_root: Path | None, pool: str):
    if campaign_root is None:
        return {}, None
    latest = campaign_root / "progress" / pool / "latest.json"
    if not latest.is_file():
        raise ValueError(f"missing latest checkpoint payload: {latest}")
    raw = latest.read_bytes()
    payload = json.loads(raw)
    if (
        payload.get("schema") != "evsp-dr-mip-convergence-v1"
        or payload.get("kind") not in {"latest", "final"}
    ):
        raise ValueError(f"invalid latest checkpoint payload: {latest}")
    incumbent = payload.get("incumbent") or {}
    final = payload.get("final") or {}
    selected_indices = (
        final.get("selected_route_indices")
        if payload.get("kind") == "final"
        and final.get("selected_route_indices") is not None
        else incumbent.get("selected_route_indices")
    ) or []
    if (
        any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            for index in selected_indices
        )
        or len(selected_indices) != len(set(selected_indices))
    ):
        raise ValueError(f"invalid selected route indices: {latest}")
    route_vector_sha = (
        final.get("route_vector_sha256")
        if payload.get("kind") == "final"
        and final.get("route_vector_sha256") is not None
        else incumbent.get("route_vector_sha256")
    )
    return {
        int(index): ordinal
        for ordinal, index in enumerate(
            selected_indices, start=1
        )
    }, {
        "path": str(latest),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "metadata": payload.get("metadata") or {},
        "selected_index_scope": (
            "augmented_pool_unresolved"
            if (
                (payload.get("metadata") or {}).get(
                    "extra_route_sources"
                )
                or (payload.get("metadata") or {}).get(
                    "source_initial_partition_sha256"
                ) is not None
            )
            else "physically_admitted_pool"
            if (payload.get("metadata") or {}).get("physical_pool_audit")
            is not None
            else "archived_pre_physical_gate_pool"
        ),
        "route_vector_sha256": route_vector_sha,
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
        archive_path = archive_path.expanduser().resolve()
        observed_archive_sha = _sha(archive_path)
        if (
            archive_sha256 is not None
            and observed_archive_sha != archive_sha256
        ):
            raise ValueError("archive SHA-256 mismatch")
        archive_sha256 = observed_archive_sha
        archive_members = _archive_member_hashes(archive_path)
        repo_root = Path(__file__).resolve().parents[1]
        reference_root = reference_data_dir.expanduser().resolve()
        reference_files = [
            reference_root / "Ref_dict.csv",
            reference_root / "par_ref_dhd.csv",
        ]
        for reference_file in reference_files:
            try:
                relative = reference_file.relative_to(repo_root)
            except ValueError as exc:
                raise ValueError(
                    "archive audit reference data must come from reviewed "
                    "checkout"
                ) from exc
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(relative)],
                cwd=repo_root, check=True, capture_output=True,
            )
        reference_status = subprocess.run(
            ["git", "status", "--porcelain", "--", *[
                str(path.relative_to(repo_root))
                for path in reference_files
            ]],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if reference_status:
            raise ValueError("reviewed reference data has local changes")
        if expected_pools != RAW_K40_POOLS:
            raise ValueError(
                "verified RAW-k40 archive requires the exact four-pool set"
            )
        if campaign_root is None:
            raise ValueError(
                "verified RAW-k40 archive requires campaign_root"
            )
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
            archive_binding = {}
            if archive_path is not None:
                for label, source_path in (
                    ("status", status_path),
                    ("journal", journal_path),
                    ("instance", instance_path),
                    ("tariff", tariff_path),
                ):
                    member = _archive_relative(source_path)
                    if archive_members.get(member) != immutable_hashes[label]:
                        raise ValueError(
                            f"archive member mismatch: {member}"
                        )
                    archive_binding[label] = member
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
            block_min = float(status["block_min"])
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
                if archive_path is not None:
                    latest_member = _archive_relative(
                        Path(selected_source["path"])
                    )
                    if (
                        archive_members.get(latest_member)
                        != selected_source["sha256"]
                    ):
                        raise ValueError(
                            f"archive latest mismatch: {latest_member}"
                        )
                    selected_source["archive_member"] = latest_member
                if (
                    selected
                    and selected_source["selected_index_scope"]
                    == "augmented_pool_unresolved"
                ):
                    raise ValueError(
                        f"selected indices belong to augmented pool for "
                        f"{pool}; selected route identities are required"
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
            if (
                selected_source is not None
                and selected_source["selected_index_scope"]
                == "archived_pre_physical_gate_pool"
                and any(
                    route_index >= len(archived_mip_pool_values)
                    for route_index in selected
                )
            ):
                raise ValueError(
                    f"selected index outside archived pool for {pool}"
                )
            selected_raw_ordinals = (
                {
                    archived_mip_pool_values[route_index][1]:
                        selected_ordinal
                    for route_index, selected_ordinal in selected.items()
                }
                if (
                    selected_source is None
                    or selected_source["selected_index_scope"]
                    == "archived_pre_physical_gate_pool"
                )
                else {}
            )
            counts = Counter()
            rejection_samples = []
            selected_failures = []
            mapping_hashes = []
            classification_by_ordinal = {}
            grid_cost_mismatches = 0
            cost_unavailable = 0
            admitted_pool = {}
            admitted_record_count = 0
            detail_by_ordinal = {}
            gate_valid_hashes = []
            gate_repaired_hashes = []
            gate_rejected_hashes = []
            for ordinal, record in enumerate(records, start=1):
                trips = list(record.get("trips") or [])
                incidence_sha = _canonical_sha(sorted(trips))
                route_nodes = record.get(
                    "route_nodes", record.get("route", [])
                )
                node_trips = [
                    node for node in route_nodes
                    if isinstance(node, int) and not isinstance(node, bool)
                ]
                recorded_reason = (
                    "trip incidence differs from route nodes"
                    if node_trips != trips
                    else validate_injected_route(
                        problem, record, g_kwh, charge_kw,
                        reserve_kwh, HORIZON_MIN, arc_map=arc_map,
                    )
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
                            realized, mapping, station_prices=prices
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
                if not costs:
                    cost_unavailable += 1
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
                    admitted_record_count += 1
                    key = frozenset(trips)
                    if (
                        key not in admitted_pool
                        or float(record["cost"])
                        < float(admitted_pool[key][0]["cost"]) - 1e-9
                    ):
                        admitted_pool[key] = (
                            realized, ordinal, classification
                        )
                    if recorded_reason is None:
                        gate_valid_hashes.append(_canonical_sha({
                            "trips": record.get("trips"),
                            "route_nodes": record.get("route_nodes"),
                            "charging_stops":
                                record.get("charging_stops"),
                            "cost": record.get("cost"),
                        }))
                    else:
                        gate_repaired_hashes.append(
                            mapping["mapping_sha256"]
                        )
                else:
                    gate_rejected_hashes.append(_canonical_sha({
                        "trips": record.get("trips"),
                        "route_nodes": record.get("route_nodes"),
                        "charging_stops": record.get("charging_stops"),
                        "cost": record.get("cost"),
                    }))
                detail_by_ordinal[ordinal] = {
                    "incidence_sha256": incidence_sha,
                    "trips": trips,
                    "recorded_reason": recorded_reason,
                    "classification": classification,
                    "mapping": mapping,
                    "costs": costs or None,
                }
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
                    "continuous_realized_charging_blocks_sha256":
                        costs.get(
                            "continuous_realized_charging_blocks_sha256"
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
                    or (
                        route_detail == "selected"
                        and selected_source is not None
                        and selected_source["selected_index_scope"]
                        == "physically_admitted_pool"
                    )
                ):
                    route_rows.append(route_row)
            unresolved_selected_indices = []
            recomputed_gate = {
                "total_columns": len(records),
                "accepted_columns": admitted_record_count,
                "valid_as_recorded": len(gate_valid_hashes),
                "deterministically_repaired":
                    len(gate_repaired_hashes),
                "rejected_columns": len(gate_rejected_hashes),
                "valid_set_sha256": _canonical_sha(
                    sorted(gate_valid_hashes)
                ),
                "repaired_set_sha256": _canonical_sha(
                    sorted(gate_repaired_hashes)
                ),
                "rejected_set_sha256": _canonical_sha(
                    sorted(gate_rejected_hashes)
                ),
                "mip_unique_accepted_columns": len(admitted_pool),
                "mip_ordered_pool_sha256": _canonical_sha([
                    _canonical_sha({
                        "trips": record.get("trips"),
                        "route_nodes": record.get("route_nodes"),
                        "charging_stops":
                            record.get("charging_stops"),
                        "cost": record.get("cost"),
                    })
                    for record, _ordinal, _classification
                    in admitted_pool.values()
                ]),
            }
            if (
                selected_source is not None
                and selected_source["selected_index_scope"]
                == "physically_admitted_pool"
            ):
                recorded_gate = selected_source["metadata"].get(
                    "physical_pool_audit"
                )
                if not isinstance(recorded_gate, dict) or any(
                    recorded_gate.get(key) != value
                    for key, value in recomputed_gate.items()
                ):
                    raise ValueError(
                        f"physical pool audit identity mismatch for {pool}"
                    )
            if (
                selected_source is not None
                and selected_source["selected_index_scope"]
                == "physically_admitted_pool"
            ):
                admitted_values = list(admitted_pool.values())
                selected_raw_ordinals = {}
                for route_index, selected_ordinal in selected.items():
                    if route_index >= len(admitted_values):
                        unresolved_selected_indices.append(route_index)
                        continue
                    selected_raw_ordinals[
                        admitted_values[route_index][1]
                    ] = selected_ordinal
                selected_failures = []
                for row in route_rows:
                    if row["pool"] != pool:
                        continue
                    selected_ordinal = selected_raw_ordinals.get(
                        int(row["ordinal"])
                    )
                    row["selected_in_solver_incumbent"] = (
                        selected_ordinal is not None
                    )
                    row["selected_route_ordinal"] = selected_ordinal
                if route_detail == "selected":
                    route_rows[:] = [
                        row for row in route_rows
                        if row["pool"] != pool
                        or row["selected_in_solver_incumbent"]
                    ]
                for raw_ordinal, selected_ordinal in (
                    selected_raw_ordinals.items()
                ):
                    detail = detail_by_ordinal[raw_ordinal]
                    if detail["recorded_reason"] is None:
                        continue
                    mapping = detail["mapping"]
                    selected_failures.append({
                        "selected_route_ordinal": selected_ordinal,
                        "pool_route_index": (
                            list(admitted_pool.values()).index(
                                next(
                                    value for value
                                    in admitted_pool.values()
                                    if value[1] == raw_ordinal
                                )
                            )
                        ),
                        "raw_journal_ordinal": raw_ordinal,
                        "incidence_sha256":
                            detail["incidence_sha256"],
                        "recorded_reason": detail["recorded_reason"],
                        "classification": detail["classification"],
                        "mapping_sha256": (
                            mapping.get("mapping_sha256")
                            if mapping else None
                        ),
                        "soc_trace": (
                            mapping.get("trace") if mapping else None
                        ),
                        "costs": detail["costs"],
                    })
                if unresolved_selected_indices:
                    raise ValueError(
                        f"selected indices include augmented/unbound routes "
                        f"for {pool}: {unresolved_selected_indices[:20]}"
                    )
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
                "archive_member_binding": archive_binding,
                "journal_records": len(records),
                "archived_mip_unique_columns":
                    len(archived_mip_pool),
                "mip_unique_columns": len(admitted_pool),
                "selected_incumbent_routes": len(selected),
                "selected_incumbent_source": selected_source,
                "unresolved_selected_indices":
                    unresolved_selected_indices,
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
                "recomputed_physical_pool_audit": recomputed_gate,
                "cost_unavailable_count": cost_unavailable,
                "physical_admission_rejected_records": (
                    len(records) - admitted_record_count
                ),
                "physics": {
                    "g_kwh": g_kwh,
                    "charge_kw": charge_kw,
                    "reserve_kwh": reserve_kwh,
                    "soc_step": soc_step,
                    "block_min": block_min,
                },
            })
            if (
                archive_path is not None
                and status.get("columns") is None
            ):
                raise ValueError(f"status lacks column count for {pool}")
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
        if (
            archive_path is not None
            and _sha(archive_path.expanduser().resolve())
            != archive_sha256
        ):
            raise ValueError("archive changed during audit")
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
                "reference_data_source": "reviewed_git_checkout",
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
                    "cost_unavailable_count":
                        pool["cost_unavailable_count"],
                    "physical_admission_rejected_records":
                        pool["physical_admission_rejected_records"],
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
            "Every admitted route persists compact tariff-bound continuous "
            "charging blocks (station, block interval, realized/grid kWh, "
            "tariff identity, and price) plus a deterministic block-schedule "
            "SHA-256. Full SOC traces remain audit-only.",
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
        markdown.extend([
            "",
            "## Archived CS incumbents",
            "",
        ])
        for pool in pool_summaries:
            failures = pool["selected_recorded_failures"]
            if not failures:
                continue
            first = min(
                failures,
                key=lambda item: item["selected_route_ordinal"],
            )
            markdown.append(
                f"- **{pool['pool']}**: {len(failures)} of "
                f"{pool['selected_incumbent_routes']} selected routes were "
                "invalid as recorded; every one was deterministically "
                "repairable. The first was selected route "
                f"{first['selected_route_ordinal']}: "
                f"`{first['recorded_reason']}`."
            )
        markdown.extend([
            "",
            "No journal route in any of the four pools was infeasible after "
            "continuous realization. This does not make the archived CS "
            "incumbents physical schedules: their stored schedules remain "
            "invalid, and the CA pools remain finite-pool partition-infeasible.",
        ])
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

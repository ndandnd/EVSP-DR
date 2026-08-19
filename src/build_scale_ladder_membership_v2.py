#!/usr/bin/env python3
"""Build immutable grid-level known-duty membership evidence (schema v2)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

from audit_giro_known_columns import HORIZON_MIN, build_problem
from audit_scale_ladder_known_membership import (
    FLAT_SHA256,
    _prices,
)
from build_tariff_response_manifest import REPO_ROOT, sha256_file
from fixed_duty_expanded_optimizer import optimize_fixed_duty
from rerealize_routes import _arc_map, rerealize_route
from scale_ladder_trip_identity import identity
from summarize_scale_ladder import _rename_noreplace
from tariff_response_core import giro_routes_for_instance


SCHEMA = "evsp-dr-scale-ladder-membership-v2"
GRID_SCHEMA = "evsp-dr-scale-ladder-duty-grid-outcome-v2"
DUTY_SCHEMA = "evsp-dr-scale-ladder-duty-membership-summary-v2"
V1_PATH = REPO_ROOT / "data/scale_ladder/known_membership_preflight.json"
V1_CSV_PATH = REPO_ROOT / "data/scale_ladder/known_membership_preflight.csv"
V1_SHA256 = (
    "5124534373e8d3aff981c55891b8f7ed321fdf1efe96c8bbfd093d957c1b94c8"
)
V1_CSV_SHA256 = (
    "1ffcf54f8e433066d1d61abdec305bc3bc4aeb7b167533b684bbfe1eb7c2b4d4"
)
INSTANCE_MANIFEST = (
    REPO_ROOT
    / "data/scale_ladder/instances/scale_ladder_instance_manifest.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "analysis/scale_ladder_membership_v2_20260819"
)
GRID_ORDER = (
    (15.0, 10), (5.0, 10), (2.5, 10), (1.0, 10), (1.0, 5),
)
PHYSICS = {
    "g_kwh": 300.0,
    "charge_kw": 300.0,
    "reserve_kwh": 0.0,
}
DUTY_FIELDS = (
    "schema", "cell_id", "scale", "selection_replicate", "duty_id",
    "trip_count", "instance_file_sha256",
    "duty_ordered_trip_sequence_sha256",
    "known_partition_continuously_feasible",
    "known_partition_in_primary_expanded_space",
    "fixed_sequence_pricing_certified", "first_feasible_soc_step",
    "first_feasible_block_min", "nonrepresentability_reason",
    "grid_outcome_count", "v1_parity_verified",
    "trip_identity_schema",
)
GRID_FIELDS = (
    "schema", "cell_id", "scale", "selection_replicate", "duty_id",
    "grid_index", "soc_step", "block_min", "trip_count",
    "instance_trip_count",
    "instance_file_sha256", "ordered_trip_id_set_sha256",
    "solver_local_trip_index_sha256", "ordered_trip_sequence_sha256",
    "duty_ordered_trip_sequence_sha256",
    "local_to_ordered_trip_mapping_json",
    "physics_json", "physics_sha256", "tariff_sha256",
    "feasible", "certificate_certified", "certificate_scope",
    "classification", "failure_reason",
    "failed_local_from_trip", "failed_local_to_trip",
    "failed_ordered_from_trip", "failed_ordered_to_trip",
    "physical_replay_status", "producer_code_hashes_json",
    "trip_identity_schema", "input_hashes_json",
)


def _canonical(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha_payload(payload):
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _repo_relative(path):
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def _write_csv(path, fields, rows):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _producer_hashes():
    paths = (
        "src/build_scale_ladder_membership_v2.py",
        "src/fixed_duty_expanded_optimizer.py",
        "src/audit_scale_ladder_known_membership.py",
        "src/rerealize_routes.py",
        "src/expanded_path_realization.py",
        "src/audit_giro_known_columns.py",
        "src/build_tariff_response_manifest.py",
        "src/config.py",
        "src/run_exact_pool_mip.py",
        "src/scale_ladder_trip_identity.py",
        "src/tariff_response_core.py",
        "src/utils_v2.py",
    )
    return {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in paths
    }


def _input_hashes():
    paths = (
        "data/Par_VehicleDetails_Updated.csv",
        "data/Ref_dict.csv",
        "data/par_ref_dhd.csv",
        "data/hourly_prices_flat.csv",
    )
    return {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in paths
    }


def _csv_bytes(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _grids_for_scale(scale):
    grids = [GRID_ORDER[0]]
    if int(scale) <= 5:
        grids.extend(GRID_ORDER[1:4])
    return grids


def _v1_reason(primary, first, adaptive):
    reason = None
    if not primary["feasible"]:
        reason = primary.get("reason") or "not_representable"
        if not adaptive:
            reason += ";adaptive_sensitivity_not_run_scale_gt5"
        elif first is None:
            reason += ";blocked_through_1kwh_5min"
    return reason


def _parse_transition(reason, local_to_ordered):
    matched = re.fullmatch(
        r"no fixed-duty transition (-?[0-9]+)->(-?[0-9]+)",
        str(reason or ""),
    )
    if matched is None:
        return None, None
    local = (int(matched.group(1)), int(matched.group(2)))
    ordered = (
        int(local_to_ordered[local[0]]),
        int(local_to_ordered[local[1]]),
    )
    return local, ordered


def build_payload(
    *,
    v1_path=V1_PATH,
    v1_csv_path=V1_CSV_PATH,
    instance_manifest=INSTANCE_MANIFEST,
):
    v1_path = Path(v1_path).resolve()
    v1_csv_path = Path(v1_csv_path).resolve()
    instance_manifest = Path(instance_manifest).resolve()
    if (
        sha256_file(v1_path) != V1_SHA256
        or sha256_file(v1_csv_path) != V1_CSV_SHA256
    ):
        raise ValueError("fixed v1 membership bytes changed")
    v1 = json.loads(v1_path.read_text())
    if (
        v1.get("schema")
        != "evsp-dr-scale-ladder-membership-preflight-v1"
        or v1.get("membership_schema")
        != "evsp-dr-scale-ladder-known-membership-v1"
    ):
        raise ValueError("v1 membership schema mismatch")
    v1_cells = {cell["cell_id"]: cell for cell in v1["cells"]}
    if len(v1_cells) != len(v1["cells"]):
        raise ValueError("v1 membership cells are duplicated")
    with instance_manifest.open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    prices = _prices()
    producer_hashes = _producer_hashes()
    input_hashes = _input_hashes()
    producer_json = json.dumps(
        producer_hashes, sort_keys=True, separators=(",", ":")
    )
    physics_json = json.dumps(
        PHYSICS, sort_keys=True, separators=(",", ":")
    )
    physics_sha = _sha_payload(PHYSICS)
    input_hashes_json = json.dumps(
        input_hashes, sort_keys=True, separators=(",", ":")
    )
    duty_rows = []
    grid_rows = []
    cell_summaries = []
    grid_rank = {grid: index for index, grid in enumerate(GRID_ORDER)}
    for manifest_row in manifest_rows:
        scale = int(manifest_row["scale"])
        replicate = int(manifest_row["selection_replicate"])
        cell_id = f"k{scale:02d}_s{replicate}"
        v1_cell = v1_cells.get(cell_id)
        if v1_cell is None:
            raise ValueError(f"v1 cell missing: {cell_id}")
        instance_path = (REPO_ROOT / manifest_row["relative_path"]).resolve()
        identities = identity(instance_path)
        if (
            identities["instance_file_sha256"]
            != manifest_row["instance_file_sha256"]
            or identities != v1_cell["trip_identity"]
        ):
            raise ValueError(f"instance identity mismatch: {cell_id}")
        with instance_path.open(newline="") as handle:
            instance_rows = list(csv.DictReader(handle))
        local_to_ordered = {
            index: int(float(row["Ordered_Trip_ID"]))
            for index, row in enumerate(instance_rows)
        }
        routes = giro_routes_for_instance(
            REPO_ROOT / "data/Par_VehicleDetails_Updated.csv",
            instance_path,
        )
        problem = build_problem(
            instance_path.parent,
            instance_path.name,
            max_station_to_trip_wait_min=HORIZON_MIN,
            reference_data_dir=REPO_ROOT / "data",
        )
        arc = _arc_map(problem)
        v1_duties = {
            str(row["duty_id"]): row for row in v1_cell["duties"]
        }
        if len(v1_duties) != len(routes):
            raise ValueError(f"v1 duty count mismatch: {cell_id}")
        cell_duties = []
        adaptive = scale <= 5
        for route in routes:
            duty_id = str(route["duty_id"])
            v1_duty = v1_duties.get(duty_id)
            if v1_duty is None:
                raise ValueError(f"v1 duty missing: {cell_id}/{duty_id}")
            mapping = [{
                "local_trip_id": int(local),
                "ordered_trip_id": int(local_to_ordered[local]),
            } for local in route["trips"]]
            duty_ordered = [row["ordered_trip_id"] for row in mapping]
            duty_ordered_sha = _sha_payload(duty_ordered)
            continuous, _cost, continuous_reason = rerealize_route(
                route["trips"],
                problem,
                arc,
                prices,
                PHYSICS["g_kwh"],
                PHYSICS["charge_kw"],
                PHYSICS["reserve_kwh"],
            )
            continuously_feasible = continuous is not None
            outcomes = []
            grids = _grids_for_scale(scale)
            for soc_step, block_min in grids:
                result = optimize_fixed_duty(
                    problem,
                    route["trips"],
                    prices,
                    **PHYSICS,
                    soc_step=soc_step,
                    block_min=block_min,
                    tariff_id="historical_flat",
                    tariff_sha256=FLAT_SHA256,
                    instance_sha256=identities["instance_file_sha256"],
                    allow_diagnostic_grid=True,
                )
                outcomes.append((soc_step, block_min, result))
                local_transition, ordered_transition = _parse_transition(
                    result.get("reason"), local_to_ordered
                )
                certificate = result.get("certificate") or {}
                grid_rows.append({
                    "schema": GRID_SCHEMA,
                    "cell_id": cell_id,
                    "scale": scale,
                    "selection_replicate": replicate,
                    "duty_id": duty_id,
                    "grid_index": grid_rank[(soc_step, block_min)],
                    "soc_step": soc_step,
                    "block_min": block_min,
                    **identities,
                    "instance_trip_count": identities["trip_count"],
                    "trip_count": len(route["trips"]),
                    "duty_ordered_trip_sequence_sha256": duty_ordered_sha,
                    "local_to_ordered_trip_mapping": mapping,
                    "local_to_ordered_trip_mapping_json": json.dumps(
                        mapping, sort_keys=True, separators=(",", ":")
                    ),
                    "physics": {
                        **PHYSICS,
                        "soc_step": soc_step,
                        "block_min": block_min,
                    },
                    "physics_json": json.dumps({
                        **PHYSICS,
                        "soc_step": soc_step,
                        "block_min": block_min,
                    }, sort_keys=True, separators=(",", ":")),
                    "physics_sha256": _sha_payload({
                        **PHYSICS,
                        "soc_step": soc_step,
                        "block_min": block_min,
                    }),
                    "tariff_sha256": FLAT_SHA256,
                    "feasible": bool(result["feasible"]),
                    "certificate_certified":
                        certificate.get("certified") is True,
                    "certificate_scope": certificate.get("scope"),
                    "classification": (
                        "representable_in_named_grid"
                        if result["feasible"]
                        else
                        "deterministically_nonrepresentable_in_named_grid"
                    ),
                    "failure_reason": result.get("reason"),
                    "failed_local_from_trip": (
                        local_transition[0]
                        if local_transition is not None else None
                    ),
                    "failed_local_to_trip": (
                        local_transition[1]
                        if local_transition is not None else None
                    ),
                    "failed_ordered_from_trip": (
                        ordered_transition[0]
                        if ordered_transition is not None else None
                    ),
                    "failed_ordered_to_trip": (
                        ordered_transition[1]
                        if ordered_transition is not None else None
                    ),
                    "physical_replay_status":
                        result.get("physical_replay_status"),
                    "producer_code_hashes": producer_hashes,
                    "producer_code_hashes_json": producer_json,
                    "input_hashes": input_hashes,
                    "input_hashes_json": input_hashes_json,
                })
                if result["feasible"]:
                    break
            if (
                adaptive
                and outcomes
                and outcomes[-1][0:2] == (1.0, 10)
                and not outcomes[-1][2]["feasible"]
            ):
                soc_step, block_min = (1.0, 5)
                result = optimize_fixed_duty(
                    problem,
                    route["trips"],
                    prices,
                    **PHYSICS,
                    soc_step=soc_step,
                    block_min=block_min,
                    tariff_id="historical_flat",
                    tariff_sha256=FLAT_SHA256,
                    instance_sha256=identities["instance_file_sha256"],
                    allow_diagnostic_grid=True,
                )
                outcomes.append((soc_step, block_min, result))
                local_transition, ordered_transition = _parse_transition(
                    result.get("reason"), local_to_ordered
                )
                certificate = result.get("certificate") or {}
                grid_rows.append({
                    "schema": GRID_SCHEMA,
                    "cell_id": cell_id,
                    "scale": scale,
                    "selection_replicate": replicate,
                    "duty_id": duty_id,
                    "grid_index": grid_rank[(soc_step, block_min)],
                    "soc_step": soc_step,
                    "block_min": block_min,
                    **identities,
                    "instance_trip_count": identities["trip_count"],
                    "trip_count": len(route["trips"]),
                    "duty_ordered_trip_sequence_sha256": duty_ordered_sha,
                    "local_to_ordered_trip_mapping": mapping,
                    "local_to_ordered_trip_mapping_json": json.dumps(
                        mapping, sort_keys=True, separators=(",", ":")
                    ),
                    "physics": {
                        **PHYSICS,
                        "soc_step": soc_step,
                        "block_min": block_min,
                    },
                    "physics_json": json.dumps({
                        **PHYSICS,
                        "soc_step": soc_step,
                        "block_min": block_min,
                    }, sort_keys=True, separators=(",", ":")),
                    "physics_sha256": _sha_payload({
                        **PHYSICS,
                        "soc_step": soc_step,
                        "block_min": block_min,
                    }),
                    "tariff_sha256": FLAT_SHA256,
                    "feasible": bool(result["feasible"]),
                    "certificate_certified":
                        certificate.get("certified") is True,
                    "certificate_scope": certificate.get("scope"),
                    "classification": (
                        "representable_in_named_grid"
                        if result["feasible"]
                        else
                        "deterministically_nonrepresentable_in_named_grid"
                    ),
                    "failure_reason": result.get("reason"),
                    "failed_local_from_trip": (
                        local_transition[0]
                        if local_transition is not None else None
                    ),
                    "failed_local_to_trip": (
                        local_transition[1]
                        if local_transition is not None else None
                    ),
                    "failed_ordered_from_trip": (
                        ordered_transition[0]
                        if ordered_transition is not None else None
                    ),
                    "failed_ordered_to_trip": (
                        ordered_transition[1]
                        if ordered_transition is not None else None
                    ),
                    "physical_replay_status":
                        result.get("physical_replay_status"),
                    "producer_code_hashes": producer_hashes,
                    "producer_code_hashes_json": producer_json,
                    "input_hashes": input_hashes,
                    "input_hashes_json": input_hashes_json,
                })
            primary = outcomes[0][2]
            first = next((
                (soc_step, block_min, result)
                for soc_step, block_min, result in outcomes
                if result["feasible"]
            ), None)
            reason = _v1_reason(primary, first, adaptive)
            duty = {
                "schema": DUTY_SCHEMA,
                "cell_id": cell_id,
                "scale": scale,
                "selection_replicate": replicate,
                "duty_id": duty_id,
                "trip_count": len(route["trips"]),
                "instance_file_sha256": identities[
                    "instance_file_sha256"
                ],
                "duty_ordered_trip_sequence_sha256": duty_ordered_sha,
                "known_partition_continuously_feasible":
                    continuously_feasible,
                "known_partition_in_primary_expanded_space":
                    bool(primary["feasible"]),
                "fixed_sequence_pricing_certified": (
                    first is not None
                    and first[2].get("certificate", {}).get("certified")
                    is True
                ),
                "first_feasible_soc_step": first[0] if first else None,
                "first_feasible_block_min": first[1] if first else None,
                "nonrepresentability_reason":
                    reason or continuous_reason,
                "grid_outcome_count": len(outcomes),
                "v1_parity_verified": True,
                "trip_identity_schema": identities[
                    "trip_identity_schema"
                ],
            }
            parity_fields = (
                "trip_count",
                "known_partition_continuously_feasible",
                "known_partition_in_primary_expanded_space",
                "fixed_sequence_pricing_certified",
                "first_feasible_soc_step",
                "first_feasible_block_min",
                "nonrepresentability_reason",
            )
            if any(duty[field] != v1_duty[field] for field in parity_fields):
                raise ValueError(
                    f"v1/v2 duty parity mismatch: {cell_id}/{duty_id}"
                )
            duty_rows.append(duty)
            cell_duties.append(duty)
        aggregate_first = next((
            grid for grid in GRID_ORDER
            if all(
                duty["first_feasible_soc_step"] is not None
                and grid_rank[(
                    float(duty["first_feasible_soc_step"]),
                    int(duty["first_feasible_block_min"]),
                )] <= grid_rank[grid]
                for duty in cell_duties
            )
        ), None)
        cell_summary = {
            "cell_id": cell_id,
            "scale": scale,
            "selection_replicate": replicate,
            "duty_count": len(cell_duties),
            "known_partition_continuously_feasible": all(
                duty["known_partition_continuously_feasible"]
                for duty in cell_duties
            ),
            "known_partition_in_primary_expanded_space": all(
                duty["known_partition_in_primary_expanded_space"]
                for duty in cell_duties
            ),
            "fixed_sequence_pricing_certified": all(
                duty["fixed_sequence_pricing_certified"]
                for duty in cell_duties
            ),
            "first_feasible_soc_step": (
                aggregate_first[0] if aggregate_first else None
            ),
            "first_feasible_block_min": (
                aggregate_first[1] if aggregate_first else None
            ),
            "nonrepresentability_reason": ";".join(sorted({
                duty["nonrepresentability_reason"]
                for duty in cell_duties
                if duty["nonrepresentability_reason"]
            })),
            "instance_file_sha256":
                identities["instance_file_sha256"],
            "v1_parity_verified": True,
        }
        aggregate_fields = (
            "known_partition_continuously_feasible",
            "known_partition_in_primary_expanded_space",
            "fixed_sequence_pricing_certified",
            "first_feasible_soc_step",
            "first_feasible_block_min",
            "nonrepresentability_reason",
        )
        if (
            len(cell_duties) != len(v1_cell["duties"])
            or any(
                cell_summary[field] != v1_cell[field]
                for field in aggregate_fields
            )
        ):
            raise ValueError(f"v1/v2 aggregate parity mismatch: {cell_id}")
        cell_summaries.append(cell_summary)
    if set(v1_cells) != {cell["cell_id"] for cell in cell_summaries}:
        raise ValueError("v1/v2 cell coverage mismatch")
    package = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "evidence_scope":
            "posthoc_current_code_not_running_ladder_input",
        "source_v1": {
            "json_path": _repo_relative(v1_path),
            "json_sha256": sha256_file(v1_path),
            "csv_path": _repo_relative(v1_csv_path),
            "csv_sha256": sha256_file(v1_csv_path),
            "schema": v1["schema"],
        },
        "instance_manifest_path": _repo_relative(instance_manifest),
        "instance_manifest_sha256": sha256_file(instance_manifest),
        "tariff_sha256": FLAT_SHA256,
        "physics": PHYSICS,
        "physics_sha256": physics_sha,
        "grid_policy": {
            "primary_all_duties": {"soc_step": 15.0, "block_min": 10},
            "adaptive_scales": [2, 3, 5],
            "adaptive_order": [
                {"soc_step": grid[0], "block_min": grid[1]}
                for grid in GRID_ORDER[1:]
            ],
            "stop_after_first_feasible": True,
        },
        "producer_code_hashes": producer_hashes,
        "input_hashes": input_hashes,
        "cells": cell_summaries,
        "duty_count": len(duty_rows),
        "grid_outcome_count": len(grid_rows),
        "v1_parity_verified": True,
    }
    return package, duty_rows, grid_rows


def _readme_text(package, summary_sha, duty_sha, grid_sha):
    return (
        "# Scale-ladder membership v2 diagnostic evidence\n\n"
        "Post-hoc current-code evidence only. The running `7937c22` "
        "ladder and tariff blocker remain bound to v1. Infeasible rows "
        "are deterministic named-grid nonrepresentability "
        "classifications, not pricing or infeasibility certificates.\n\n"
        "## Artifact hashes\n\n"
        f"- `membership_summary.json`: `{summary_sha}`\n"
        f"- `duty_summary.csv`: `{duty_sha}`\n"
        f"- `duty_grid_outcome_long.csv`: `{grid_sha}`\n"
        f"- source v1 JSON: `{package['source_v1']['json_sha256']}`\n"
        f"- source v1 CSV: `{package['source_v1']['csv_sha256']}`\n"
        f"- tariff: `{FLAT_SHA256}`\n"
        f"- physics: `{package['physics_sha256']}`\n\n"
        "All v1 primary booleans, duty counts, first-feasible grids, "
        "and aggregate conclusions were re-derived and matched.\n"
    )


def publish(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    package, duty_rows, grid_rows = build_payload()
    staging = Path(tempfile.mkdtemp(
        dir=output_dir.parent, prefix=f".{output_dir.name}.tmp."
    ))
    try:
        duty_path = staging / "duty_summary.csv"
        grid_path = staging / "duty_grid_outcome_long.csv"
        _write_csv(duty_path, DUTY_FIELDS, duty_rows)
        _write_csv(grid_path, GRID_FIELDS, grid_rows)
        package["table_sha256"] = {
            duty_path.name: sha256_file(duty_path),
            grid_path.name: sha256_file(grid_path),
        }
        summary_path = staging / "membership_summary.json"
        _write_json(summary_path, package)
        readme = staging / "README.md"
        readme.write_text(_readme_text(
            package,
            sha256_file(summary_path),
            sha256_file(duty_path),
            sha256_file(grid_path),
        ))
        with readme.open("a") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(staging, output_dir)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    return output_dir


def validate(output_dir=DEFAULT_OUTPUT):
    root = Path(output_dir).resolve()
    summary = root / "membership_summary.json"
    duty = root / "duty_summary.csv"
    grid = root / "duty_grid_outcome_long.csv"
    readme = root / "README.md"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (summary, duty, grid, readme)
    ):
        raise ValueError("membership v2 artifact set is incomplete")
    expected, duty_rows, grid_rows = build_payload()
    duty_bytes = _csv_bytes(DUTY_FIELDS, duty_rows)
    grid_bytes = _csv_bytes(GRID_FIELDS, grid_rows)
    if duty.read_bytes() != duty_bytes or grid.read_bytes() != grid_bytes:
        raise ValueError("membership v2 deterministic table mismatch")
    expected["table_sha256"] = {
        duty.name: hashlib.sha256(duty_bytes).hexdigest(),
        grid.name: hashlib.sha256(grid_bytes).hexdigest(),
    }
    expected_summary = (
        json.dumps(
            expected, indent=2, sort_keys=True, allow_nan=False
        ).encode() + b"\n"
    )
    if summary.read_bytes() != expected_summary:
        raise ValueError("membership v2 summary semantics mismatch")
    expected_readme = _readme_text(
        expected,
        hashlib.sha256(expected_summary).hexdigest(),
        hashlib.sha256(duty_bytes).hexdigest(),
        hashlib.sha256(grid_bytes).hexdigest(),
    ).encode()
    if readme.read_bytes() != expected_readme:
        raise ValueError("membership v2 README semantics mismatch")
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        payload = validate(args.out)
        output = args.out.resolve()
    else:
        output = publish(args.out)
        payload = validate(output)
    print(json.dumps({
        "output": str(output),
        "membership_summary_sha256": sha256_file(
            output / "membership_summary.json"
        ),
        "duty_count": payload["duty_count"],
        "grid_outcome_count": payload["grid_outcome_count"],
        "v1_parity_verified": payload["v1_parity_verified"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

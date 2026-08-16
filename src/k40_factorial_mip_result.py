"""Validation and portable publication for k40 factorial MIP results."""

from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path

from config import BUS_COST_KX
from durable_io import read_jsonl_records
from portable_bundle import publish_bundle
from run_exact_pool_mip import validate_final_selected_routes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_scientific_result(
    result: dict,
    spec: dict,
    source_status: dict,
) -> None:
    if not isinstance(result, dict):
        raise ValueError("MIP result is not an object")
    if result.get("partitioning") is not True:
        raise ValueError("MIP result is not strict partitioning")
    if result.get("route_space_scope") != "finite_augmented_snapshot_pool_only":
        raise ValueError("MIP result has incorrect route-space scope")
    if result.get("source_result_sha256") != spec["staged_result_sha256"]:
        raise ValueError("MIP result status hash mismatch")
    if result.get("source_journal_sha256") != spec["staged_journal_sha256"]:
        raise ValueError("MIP result journal hash mismatch")
    provenance = source_status.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("instance_sha256")
        != spec["staged_instance_sha256"]
        or provenance.get("prices_sha256")
        != spec["staged_prices_sha256"]
    ):
        raise ValueError("source status data provenance mismatch")
    journal_records = read_jsonl_records(
        Path(spec["staged_journal"]),
        repair_trailing=False,
        collect=True,
    )
    trip_ids = source_status.get("trip_ids")
    if not isinstance(trip_ids, list) or not trip_ids:
        raise ValueError("source status trip IDs are invalid")
    known = set(trip_ids)
    pool_costs = {}
    for record in journal_records:
        trips = record.get("trips") if isinstance(record, dict) else None
        if (
            not isinstance(trips, list)
            or not trips
            or any(trip not in known for trip in trips)
        ):
            raise ValueError("source journal contains invalid trip incidence")
        cost = float(record.get("cost"))
        if not math.isfinite(cost):
            raise ValueError("source journal contains non-finite cost")
        pool_costs.setdefault(frozenset(trips), []).append(cost)
    start_payload = json.loads(Path(spec["staged_start"]).read_text())
    start_routes = start_payload.get("routes")
    if (
        not isinstance(start_routes, list)
        or len(start_routes) != 40
        or start_payload.get("infeasible") not in (None, [])
    ):
        raise ValueError("validated start file is missing/partial")
    start_counts = collections.Counter()
    start_keys = set()
    for route in start_routes:
        nodes = route.get("route", route.get("route_nodes", []))
        trips = [
            node for node in nodes
            if isinstance(node, int) and not isinstance(node, bool)
        ]
        if not trips or len(trips) != len(set(trips)):
            raise ValueError("validated start route has invalid trips")
        start_counts.update(trips)
        start_keys.add(frozenset(trips))
    if (
        set(start_counts) != known
        or any(start_counts[trip] != 1 for trip in trip_ids)
    ):
        raise ValueError("validated start is not an exact trip partition")
    expected_cell = {
        key: spec[key] for key in (
            "label", "replicate", "treatment", "snapshot_mark_minutes",
            "time_limit_s", "threads", "mip_gap",
        )
    }
    if result.get("campaign_cell") != expected_cell:
        raise ValueError("MIP result campaign cell mismatch")
    start = result.get("mip_start")
    acceptance = (start or {}).get("solver_acceptance")
    if (
        not isinstance(start, dict)
        or start.get("kind") != "validated_exact_partition"
        or start.get("source_sha256") != spec["staged_start_sha256"]
        or not isinstance(acceptance, dict)
        or acceptance.get("accepted") is not True
    ):
        raise ValueError("validated start evidence is incomplete/rejected")
    columns = start.get("actual_start_columns")
    hashes = start.get("actual_start_column_hashes")
    if isinstance(columns, list):
        if (
            len(columns) != 40
            or any(
                not isinstance(column, dict)
                or type(column.get("index")) is not int
                or not isinstance(column.get("sha256"), str)
                or len(column["sha256"]) != 64
                for column in columns
            )
        ):
            raise ValueError("actual start columns are malformed")
    elif (
        not isinstance(hashes, list)
        or len(hashes) != 40
        or any(not isinstance(value, str) or len(value) != 64 for value in hashes)
    ):
        raise ValueError("actual start column hashes are missing")
    selected = result.get("selected_routes")
    buses = result.get("buses")
    if not isinstance(selected, list) or type(buses) is not int:
        raise ValueError("selected routes/bus count are invalid")
    counts = collections.Counter(
        trip for route in selected for trip in route.get("trips", [])
    )
    if (
        not isinstance(trip_ids, list)
        or buses != len(selected)
        or buses != 40
        or set(counts) != set(trip_ids)
        or any(counts[trip] != 1 for trip in trip_ids)
        or any(
            not isinstance(route.get("charging_stops"), dict)
            for route in selected
        )
    ):
        raise ValueError("result is not a 40-route exact scheduled partition")
    full_objective = 0.0
    for route in selected:
        key = frozenset(route.get("trips") or [])
        cost = float(route.get("cost"))
        if not math.isfinite(cost):
            raise ValueError("selected route has non-finite cost")
        if key not in pool_costs and key not in start_keys:
            raise ValueError("selected route is absent from journal/start pool")
        if key in pool_costs and key not in start_keys and not any(
            math.isclose(cost, candidate, rel_tol=1e-10, abs_tol=1e-6)
            for candidate in pool_costs[key]
        ):
            raise ValueError("selected route cost differs from journal")
        full_objective += cost
    if result.get("fleet_proven") is not True:
        raise ValueError("40-bus fleet is not proven over the finite pool")
    fleet_bound = result.get("fleet_bound")
    if (
        not isinstance(fleet_bound, (int, float))
        or isinstance(fleet_bound, bool)
        or not math.isclose(float(fleet_bound), 40.0, abs_tol=1e-9)
    ):
        raise ValueError("fleet bound does not close at 40")
    if result.get("optimal_scope") != "full_pool_lexicographic":
        raise ValueError("charging-cost stage is not proven optimal")
    detail = result.get("two_stage")
    if not isinstance(detail, dict):
        raise ValueError("two-stage details are missing")
    stage1_buses = detail.get("stage1_buses")
    variable_obj = detail.get("stage2_variable_obj")
    variable_bound = detail.get("stage2_variable_bound")
    full_obj = result.get("mip_obj")
    full_bound = result.get("mip_bound")
    if (
        type(stage1_buses) is not int
        or stage1_buses != buses
        or detail.get("stage2_executed") is not True
        or detail.get("stage2_status") != 2
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in (variable_obj, variable_bound, full_obj, full_bound)
        )
        or not math.isclose(float(variable_obj), float(variable_bound),
                            rel_tol=1e-10, abs_tol=1e-6)
        or not math.isclose(float(full_obj), float(full_bound),
                            rel_tol=1e-10, abs_tol=1e-6)
        or not math.isclose(
            float(full_obj),
            BUS_COST_KX * buses + float(variable_obj),
            rel_tol=1e-10,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("two-stage objective/bound closure is inconsistent")
    if not math.isclose(
        full_objective, float(full_obj), rel_tol=1e-10, abs_tol=1e-6
    ):
        raise ValueError("selected route costs do not reconstruct objective")
    mip_provenance = result.get("mip_provenance")
    arguments = (
        mip_provenance.get("arguments")
        if isinstance(mip_provenance, dict) else None
    )
    if not isinstance(mip_provenance, dict) or (
        mip_provenance.get("expected_git_commit")
        != spec.get("expected_commit")
        or mip_provenance.get("observed_git_commit")
        != spec.get("expected_commit")
        or mip_provenance.get("final_observed_git_commit")
        != spec.get("expected_commit")
        or mip_provenance.get("tracked_clean_at_end") is not True
        or not isinstance(mip_provenance.get("gurobi"), str)
        or not mip_provenance["gurobi"]
        or not isinstance(arguments, dict)
        or arguments.get("two_stage") is not True
        or arguments.get("cover") is not False
        or int(arguments.get("threads", -1)) != int(spec["threads"])
        or int(arguments.get("timelimit", -1)) != int(spec["time_limit_s"])
    ):
        raise ValueError("MIP Git provenance is inconsistent")
    if (
        result.get("status") != 2
        or result.get("status_name") != "OPTIMAL"
        or float(result.get("mip_gap")) != 0.0
    ):
        raise ValueError("MIP result is not solver-optimal")
    relative = Path(spec["csv"])
    instance = Path(spec["staged_instance"]).resolve()
    data_root = instance
    for _part in relative.parts:
        data_root = data_root.parent
    validate_final_selected_routes(
        source_status,
        trip_ids,
        selected,
        data_dir=data_root,
    )


def enrich_result(
    raw_result: dict,
    *,
    spec: dict,
    recovery: dict,
) -> dict:
    result = json.loads(json.dumps(raw_result))
    result["route_space_scope"] = "finite_augmented_snapshot_pool_only"
    result["campaign_cell"] = {
        key: spec[key] for key in (
            "label", "replicate", "treatment", "snapshot_mark_minutes",
            "time_limit_s", "threads", "mip_gap",
        )
    }
    result["completion_attestation"] = {
        "schema": "evsp-dr-k40-factorial-mip-result-v2",
        "job_spec_sha256": recovery["job_spec_sha256"],
        "worker_sha256": recovery["worker_sha256"],
        "runner_sha256": spec["runner_sha256"],
        "validated_start_sha256": spec["staged_start_sha256"],
        "original_job_id": recovery["original_job_id"],
        "raw_sha256": recovery["raw_sha256"],
        "recovery_commit": recovery["recovery_commit"],
        "recovery_method": recovery["recovery_method"],
    }
    return result


def publish_result_bundle(
    destination: Path,
    *,
    result: dict,
    spec: dict,
    source_status: dict,
    recovery: dict,
    allow_existing_incomplete: bool,
) -> dict:
    validate_scientific_result(result, spec, source_status)
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    return publish_bundle(
        destination,
        members={"result.json": encoded},
        metadata={
            "kind": "k40-factorial-mip-result",
            "job_spec_sha256": recovery["job_spec_sha256"],
            "worker_sha256": recovery["worker_sha256"],
            "runner_sha256": spec["runner_sha256"],
            "original_job_id": recovery["original_job_id"],
            "raw_sha256": recovery["raw_sha256"],
            "recovery_commit": recovery["recovery_commit"],
            "recovery_method": recovery["recovery_method"],
        },
        allow_existing_incomplete=allow_existing_incomplete,
    )

"""Fail-closed parsers for EVSP-DR cross-generation evidence schemas."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path


LEGACY_HEURISTIC_HEADER = (
    "Iteration", "Master_Obj", "Master_Improvement", "Master_Time_s",
    "Pricing_Time_s", "Cumulative_Master_Time_s",
    "Cumulative_Pricing_Time_s", "Cols_Added", "Best_RC", "Timed_Out",
    "Pricing_TimeLimit_Used", "Stagnant_Counter", "Total_Runtime_s",
)
CURRENT_HEURISTIC_REQUIRED = {
    "Iteration", "Master_Obj_Before_Add",
    "Master_Improvement_Before_Add", "Master_Time_s",
    "LP_Route_Weight_Before_Add", "Artificial_Trips_Before_Add",
    "Artificial_Total_Before_Add", "Pricing_Time_s",
    "Cumulative_Master_Time_s", "Cumulative_Pricing_Time_s",
    "Cols_Added", "Best_RC", "Timed_Out",
    "Deepest_Tier_Hit_Timelimit", "Pricing_Label_Cap_Evictions",
    "Pricing_Labels_Used", "Pricing_Label_Cap_Configured",
    "Pricing_Completed_Routes", "Pricing_Negative_Completed",
    "Pricing_Eligible_Negative_Incidences",
    "Pricing_Returned_Trip_Count_Min",
    "Pricing_Returned_Trip_Count_Mean",
    "Pricing_Returned_Trip_Count_Max",
    "Pricing_Exhaustive_Deepest_Tier", "Pricing_Queue_Order",
    "Pricing_Output_Selection", "Pricing_Dominance_Mode",
    "Highest_Tier_Reached", "Recent_Window_Sum", "Total_Runtime_s",
}
EXACT_ITER_HEADER = (
    "elapsed_s", "iteration", "lp_obj", "route_weight",
    "artificials", "min_rc", "pool_columns",
)
TELEMETRY_SCHEMA = "evsp-dr-exact-cg-phase-telemetry-v1"
MIP_CHECKPOINT_SCHEMA = "evsp-dr-mip-convergence-v1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _number(value, *, allow_infinity=False):
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value {value!r}") from exc
    if math.isfinite(result):
        return result
    if allow_infinity:
        return None
    raise ValueError(f"non-finite numeric value {value!r}")


def _integer(value):
    number = _number(value)
    if number is None or int(number) != number:
        raise ValueError(f"invalid integer value {value!r}")
    return int(number)


def _boolean(value):
    if type(value) is bool:
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    if text in {"", "none", "null"}:
        return None
    raise ValueError(f"invalid boolean value {value!r}")


def _tail_safe_csv(payload: bytes) -> tuple[tuple[str, ...], list[dict], dict]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV is not UTF-8") from exc
    physical_lines = text.splitlines()
    if not physical_lines:
        raise ValueError("CSV is empty")
    header_rows = list(csv.reader([physical_lines[0]]))
    if len(header_rows) != 1 or len(header_rows[0]) != len(set(header_rows[0])):
        raise ValueError("CSV header is malformed or duplicated")
    header = tuple(header_rows[0])
    records = []
    tail_dropped = False
    for index, line in enumerate(physical_lines[1:], start=2):
        if not line.strip():
            continue
        try:
            parsed = next(csv.reader([line]))
        except (csv.Error, StopIteration) as exc:
            if index == len(physical_lines):
                tail_dropped = True
                break
            raise ValueError(f"CSV corruption before EOF at line {index}") from exc
        if len(parsed) != len(header):
            if index == len(physical_lines):
                tail_dropped = True
                break
            raise ValueError(
                f"CSV row width mismatch before EOF at line {index}"
            )
        records.append(dict(zip(header, parsed)))
    if not records:
        raise ValueError("CSV has no complete data rows")
    return header, records, {
        "tail_dropped": tail_dropped,
        "tail_reason": (
            "interrupted_final_csv_row" if tail_dropped else None
        ),
    }


def _tail_safe_jsonl(payload: bytes) -> tuple[list[dict], dict]:
    records = []
    lines = payload.splitlines()
    tail_dropped = False
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == len(lines):
                tail_dropped = True
                break
            raise ValueError(
                f"JSONL corruption before EOF at line {index}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {index} is not an object")
        records.append(value)
    return records, {
        "tail_dropped": tail_dropped,
        "tail_reason": (
            "interrupted_final_jsonl_record" if tail_dropped else None
        ),
    }


def normalize_termination(value) -> tuple[str | None, str | None]:
    if value in (None, "", "unknown", "running", "initializing"):
        return None, "termination_not_recorded_or_run_not_terminal"
    token = str(value)
    mapping = {
        "certified": "certification",
        "rc_optimal_restricted": "certification_restricted_graph",
        "wall_limit": "wall_limit",
        "active_time_limit_reached": "wall_limit",
        "TIME_LIMIT": "timeout",
        "timeout": "timeout",
        "pricing_truncated_no_new_columns": "timeout_or_label_cap",
        "stagnation_rolling_window": "marginal_return",
        "stalled_marginal_returns": "marginal_return",
        "master_failed": "master_failure",
        "error": "error",
        "max_iters": "iteration_limit",
        "max_iterations_reached": "iteration_limit",
        "no_new_nondominated_columns": "no_new_columns",
        "no_path": "no_path",
        "degenerate_stall": "degenerate_stall",
        "label_cap": "label_cap",
    }
    return mapping.get(token, token), None


def _base_iteration(spec: dict, schema: str, iteration: int) -> dict:
    metadata = spec.get("metadata") or {}
    return {
        "artifact_id": spec["artifact_id"],
        "run_id": spec["run_id"],
        "schema_family": schema,
        "algorithm_family": metadata.get("algorithm_family"),
        "implementation": metadata.get("implementation"),
        "scale_family": metadata.get("scale_family"),
        "scale": metadata.get("scale"),
        "replicate": metadata.get("replicate"),
        "seed": metadata.get("seed"),
        "iteration": iteration,
        "legacy_master_objective": None,
        "legacy_master_improvement": None,
        "master_objective_before_add": None,
        "master_improvement_before_add": None,
        "lp_objective": None,
        "lp_route_weight": None,
        "artificial_trips": None,
        "artificial_total": None,
        "best_reduced_cost": None,
        "best_reduced_cost_reason": None,
        "columns_added": None,
        "pool_columns": None,
        "wall_time_s": None,
        "master_time_s": None,
        "pricing_time_s": None,
        "cumulative_master_time_s": None,
        "cumulative_pricing_time_s": None,
        "master_pricing_split_available": False,
        "timed_out": None,
        "label_cap_evictions": None,
        "pricing_labels_used": None,
        "pricing_label_cap_configured": None,
        "pricing_completed_routes": None,
        "pricing_negative_completed": None,
        "pricing_eligible_negative_incidences": None,
        "pricing_returned_trip_count_min": None,
        "pricing_returned_trip_count_mean": None,
        "pricing_returned_trip_count_max": None,
        "deepest_tier_timed_out": None,
        "pricing_exhaustive": None,
        "pricing_queue_order": None,
        "pricing_output_selection": None,
        "pricing_dominance_mode": None,
        "highest_tier_reached": None,
        "recent_window_sum": None,
        "tier_statistics_json": None,
        "stagnant_counter": None,
        "pricing_time_limit_used_s": None,
        "availability_reason": None,
    }


def parse_legacy_heuristic_csv(payload: bytes, spec: dict) -> dict:
    header, records, tail = _tail_safe_csv(payload)
    if header != LEGACY_HEURISTIC_HEADER:
        raise ValueError("legacy heuristic CSV header mismatch")
    rows = []
    for record in records:
        row = _base_iteration(
            spec, "heuristic_dp_historical", _integer(record["Iteration"])
        )
        row.update({
            "legacy_master_objective": _number(record["Master_Obj"]),
            "legacy_master_improvement": _number(
                record["Master_Improvement"], allow_infinity=True
            ),
            "master_time_s": _number(record["Master_Time_s"]),
            "pricing_time_s": _number(record["Pricing_Time_s"]),
            "cumulative_master_time_s": _number(
                record["Cumulative_Master_Time_s"]
            ),
            "cumulative_pricing_time_s": _number(
                record["Cumulative_Pricing_Time_s"]
            ),
            "master_pricing_split_available": True,
            "columns_added": _integer(record["Cols_Added"]),
            "best_reduced_cost": _number(record["Best_RC"]),
            "timed_out": _boolean(record["Timed_Out"]),
            "pricing_time_limit_used_s": _number(
                record["Pricing_TimeLimit_Used"]
            ),
            "stagnant_counter": _integer(record["Stagnant_Counter"]),
            "wall_time_s": _number(record["Total_Runtime_s"]),
            "availability_reason": (
                "legacy_schema_has_no_route_weight_artificials_or_tiers"
            ),
        })
        rows.append(row)
    return {"cg_rows": rows, "tail": tail, "schema": "heuristic_dp_historical"}


def parse_current_heuristic_csv(payload: bytes, spec: dict) -> dict:
    header, records, tail = _tail_safe_csv(payload)
    if not CURRENT_HEURISTIC_REQUIRED.issubset(header):
        missing = sorted(CURRENT_HEURISTIC_REQUIRED - set(header))
        raise ValueError(f"current heuristic CSV missing fields: {missing}")
    rows = []
    for record in records:
        row = _base_iteration(
            spec, "heuristic_dp_current", _integer(record["Iteration"])
        )
        row.update({
            "master_objective_before_add": _number(
                record["Master_Obj_Before_Add"]
            ),
            "master_improvement_before_add": _number(
                record["Master_Improvement_Before_Add"],
                allow_infinity=True,
            ),
            "lp_route_weight": _number(
                record["LP_Route_Weight_Before_Add"]
            ),
            "artificial_trips": _integer(
                record["Artificial_Trips_Before_Add"]
            ),
            "artificial_total": _number(
                record["Artificial_Total_Before_Add"]
            ),
            "master_time_s": _number(record["Master_Time_s"]),
            "pricing_time_s": _number(record["Pricing_Time_s"]),
            "cumulative_master_time_s": _number(
                record["Cumulative_Master_Time_s"]
            ),
            "cumulative_pricing_time_s": _number(
                record["Cumulative_Pricing_Time_s"]
            ),
            "master_pricing_split_available": True,
            "columns_added": _integer(record["Cols_Added"]),
            "best_reduced_cost": _number(record["Best_RC"]),
            "timed_out": _boolean(record["Timed_Out"]),
            "label_cap_evictions": _integer(
                record["Pricing_Label_Cap_Evictions"]
            ),
            "pricing_labels_used": _integer(
                record["Pricing_Labels_Used"]
            ),
            "pricing_label_cap_configured": _integer(
                record["Pricing_Label_Cap_Configured"]
            ),
            "pricing_completed_routes": _integer(
                record["Pricing_Completed_Routes"]
            ),
            "pricing_negative_completed": _integer(
                record["Pricing_Negative_Completed"]
            ),
            "pricing_eligible_negative_incidences": _integer(
                record["Pricing_Eligible_Negative_Incidences"]
            ),
            "pricing_returned_trip_count_min": _number(
                record["Pricing_Returned_Trip_Count_Min"]
            ),
            "pricing_returned_trip_count_mean": _number(
                record["Pricing_Returned_Trip_Count_Mean"]
            ),
            "pricing_returned_trip_count_max": _number(
                record["Pricing_Returned_Trip_Count_Max"]
            ),
            "deepest_tier_timed_out": _boolean(
                record["Deepest_Tier_Hit_Timelimit"]
            ),
            "pricing_exhaustive": _boolean(
                record["Pricing_Exhaustive_Deepest_Tier"]
            ),
            "pricing_queue_order": record["Pricing_Queue_Order"] or None,
            "pricing_output_selection": (
                record["Pricing_Output_Selection"] or None
            ),
            "pricing_dominance_mode": (
                record["Pricing_Dominance_Mode"] or None
            ),
            "highest_tier_reached": _integer(
                record["Highest_Tier_Reached"]
            ),
            "recent_window_sum": _number(record["Recent_Window_Sum"]),
            "tier_statistics_json": json.dumps(
                {
                    key: (
                        None if value == "" else value
                    )
                    for key, value in record.items()
                    if key.startswith("Tier")
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "wall_time_s": _number(record["Total_Runtime_s"]),
        })
        rows.append(row)
    return {"cg_rows": rows, "tail": tail, "schema": "heuristic_dp_current"}


def parse_exact_iterations(payload: bytes, spec: dict) -> dict:
    header, records, tail = _tail_safe_csv(payload)
    if header != EXACT_ITER_HEADER:
        raise ValueError("exact expanded-network .iters.csv header mismatch")
    rows = []
    previous_pool = None
    for record in records:
        row = _base_iteration(
            spec, "exact_expanded_network", _integer(record["iteration"])
        )
        min_rc = _number(record["min_rc"], allow_infinity=True)
        pool_columns = _integer(record["pool_columns"])
        row.update({
            "lp_objective": _number(record["lp_obj"]),
            "lp_route_weight": _number(record["route_weight"]),
            "artificial_total": _number(record["artificials"]),
            "best_reduced_cost": min_rc,
            "best_reduced_cost_reason": (
                "no_finite_pricing_path"
                if min_rc is None and "inf" in record["min_rc"].lower()
                else None
            ),
            "pool_columns": pool_columns,
            "columns_added": (
                pool_columns - previous_pool
                if previous_pool is not None else None
            ),
            "wall_time_s": _number(record["elapsed_s"]),
            "master_pricing_split_available": False,
            "availability_reason": (
                "exact_iteration_log_does_not_measure_master_pricing_split"
            ),
        })
        rows.append(row)
        previous_pool = pool_columns
    return {"cg_rows": rows, "tail": tail, "schema": "exact_expanded_network"}


def parse_phase_telemetry(payload: bytes, spec: dict) -> dict:
    records, tail = _tail_safe_jsonl(payload)
    if not records:
        raise ValueError("phase telemetry JSONL has no records")
    starts = [
        row for row in records if row.get("record_type") == "session_start"
    ]
    if not starts:
        raise ValueError("phase telemetry lacks session_start")
    identities = {
        row.get("identity_sha256") for row in starts
        if isinstance(row.get("identity_sha256"), str)
    }
    if len(identities) != 1:
        raise ValueError("phase telemetry has mixed/missing identity")
    phases = []
    for row in records:
        if row.get("schema") != TELEMETRY_SCHEMA:
            raise ValueError("phase telemetry schema mismatch")
        if row.get("record_type") != "phase":
            continue
        if row.get("identity_sha256") not in identities:
            raise ValueError("phase telemetry record has mixed identity")
        duration = _number(row.get("duration_s"))
        elapsed = _number(row.get("elapsed_session_s"))
        if duration is None or duration < 0 or elapsed is None or elapsed < 0:
            raise ValueError("phase telemetry has invalid timing")
        phases.append({
            "artifact_id": spec["artifact_id"],
            "run_id": spec["run_id"],
            "session": row.get("session"),
            "phase": row.get("phase"),
            "duration_s": duration,
            "elapsed_session_s": elapsed,
            "iteration": row.get("iteration"),
            "attempt": row.get("attempt"),
            "pool_columns": row.get("pool_columns"),
            "peak_rss_bytes": row.get("peak_rss_bytes"),
            "outcome": row.get("outcome"),
            "identity_sha256": next(iter(identities)),
        })
    return {
        "telemetry_rows": phases,
        "tail": tail,
        "schema": "exact_phase_telemetry",
    }


def parse_mip_checkpoint(payload: bytes, spec: dict) -> dict:
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("schema") != MIP_CHECKPOINT_SCHEMA:
        raise ValueError("MIP checkpoint schema mismatch")
    if value.get("observational_only") is not True:
        raise ValueError("MIP checkpoint is not observational")
    incumbent = value.get("incumbent") or {}
    stats = value.get("latest_statistics") or {}
    metadata = value.get("metadata") or {}
    expected_metadata = spec.get("metadata") or {}
    for expected_key, observed in (
        ("git_commit", metadata.get("git_commit")),
        ("pool_status_sha256", metadata.get("source_result_sha256")),
        ("pool_journal_sha256", metadata.get("source_journal_sha256")),
        ("pool_start_sha256", metadata.get("source_initial_partition_sha256")),
    ):
        expected = expected_metadata.get(expected_key)
        if expected is not None and observed is not None and expected != observed:
            raise ValueError(
                f"MIP checkpoint {expected_key} differs from manifest"
            )
    row = {
        "artifact_id": spec["artifact_id"],
        "run_id": spec["run_id"],
        "algorithm_family": "mip_finite_pool",
        "implementation": (spec.get("metadata") or {}).get("implementation"),
        "scale_family": (spec.get("metadata") or {}).get("scale_family"),
        "scale": (spec.get("metadata") or {}).get("scale"),
        "replicate": (spec.get("metadata") or {}).get("replicate"),
        "treatment": (spec.get("metadata") or {}).get("treatment"),
        "checkpoint_elapsed_s": _number(value.get("checkpoint_elapsed_s")),
        "observed_total_elapsed_s": _number(
            value.get("observed_total_elapsed_s")
        ),
        "statistics_observed_s": _number(
            value.get("latest_statistics_observed_s")
        ),
        "stage": value.get("stage"),
        "incumbent_state": value.get("incumbent_state"),
        "incumbent_fleet": incumbent.get("fleet"),
        "incumbent_objective": incumbent.get("objective"),
        "fleet_bound": stats.get("fleet_bound"),
        "objective_bound": stats.get("objective_bound"),
        "fleet_gap": stats.get("fleet_gap"),
        "node_count": stats.get("node_count"),
        "solution_count": stats.get("solution_count"),
        "route_vector_sha256": incumbent.get("route_vector_sha256"),
        "first_feasible_s": value.get("first_feasible_incumbent_s"),
        "solver_ended_before_checkpoint": value.get(
            "solver_ended_before_checkpoint"
        ),
        "source_result_sha256": metadata.get("source_result_sha256"),
        "source_journal_sha256": metadata.get("source_journal_sha256"),
        "source_start_sha256": metadata.get(
            "source_initial_partition_sha256"
        ),
        "experiment_arm": metadata.get("experiment_arm"),
        "observational_only": True,
    }
    return {"mip_rows": [row], "tail": {}, "schema": "mip_checkpoint"}


def parse_json_artifact(payload: bytes, spec: dict) -> dict:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("JSON artifact is not an object")
    hint = spec["artifact_type"]
    if hint == "mip_checkpoint":
        return parse_mip_checkpoint(payload, spec)
    if hint == "mip_final":
        required = {
            "partitioning", "incumbent_found", "mip_provenance",
            "source_result_sha256", "source_journal_sha256",
        }
        if not required.issubset(value):
            raise ValueError("MIP final is missing required fields")
        metadata = spec.get("metadata") or {}
        mip_provenance = value.get("mip_provenance") or {}
        for expected_key, observed in (
            ("git_commit", mip_provenance.get("observed_git_commit")),
            ("pool_status_sha256", value.get("source_result_sha256")),
            ("pool_journal_sha256", value.get("source_journal_sha256")),
        ):
            expected = metadata.get(expected_key)
            if (
                expected is not None and observed is not None
                and expected != observed
            ):
                raise ValueError(
                    f"MIP final {expected_key} differs from manifest"
                )
        row = {
            "artifact_id": spec["artifact_id"],
            "run_id": spec["run_id"],
            "algorithm_family": "mip_finite_pool",
            "implementation": (spec.get("metadata") or {}).get(
                "implementation"
            ),
            "scale_family": (spec.get("metadata") or {}).get("scale_family"),
            "scale": (spec.get("metadata") or {}).get("scale"),
            "replicate": (spec.get("metadata") or {}).get("replicate"),
            "treatment": (spec.get("metadata") or {}).get("treatment"),
            "incumbent_found": value.get("incumbent_found"),
            "integer_fleet": value.get("buses"),
            "objective": value.get("mip_obj"),
            "objective_bound": value.get("mip_bound"),
            "gap": value.get("mip_gap"),
            "fleet_bound": value.get("fleet_bound"),
            "fleet_proven": value.get("fleet_proven"),
            "status_name": value.get("status_name"),
            "optimal_scope": value.get("optimal_scope"),
            "runtime_s": value.get("runtime_s"),
            "partitioning": value.get("partitioning"),
            "physically_validated_schedule": (
                metadata.get("physical_replay_validated") is True
                and value.get("incumbent_found") is True
                and bool(value.get("selected_routes"))
            ),
            "source_result_sha256": value.get("source_result_sha256"),
            "source_journal_sha256": value.get("source_journal_sha256"),
            "pool_treatment": (
                (spec.get("metadata") or {}).get("treatment")
            ),
            "giro_columns_added": (
                (spec.get("metadata") or {}).get("treatment") == "GIRO"
            ),
        }
        return {"mip_finals": [row], "tail": {}, "schema": "mip_final"}
    # Endpoint/manifests remain inventory/provenance evidence. Preserve fields
    # without inventing iteration trajectories.
    return {
        "endpoint": value,
        "tail": {},
        "schema": hint,
    }


PARSERS = {
    "heuristic_dp_historical_csv": parse_legacy_heuristic_csv,
    "heuristic_dp_current_csv": parse_current_heuristic_csv,
    "exact_cg_iterations_csv": parse_exact_iterations,
    "exact_cg_phase_telemetry_jsonl": parse_phase_telemetry,
    "mip_checkpoint": parse_json_artifact,
    "mip_final": parse_json_artifact,
    "endpoint_json": parse_json_artifact,
    "artifact_manifest_json": parse_json_artifact,
}


def parse_artifact(payload: bytes, spec: dict) -> dict:
    parser = PARSERS.get(spec.get("artifact_type"))
    if parser is None:
        raise ValueError(
            f"unsupported artifact_type {spec.get('artifact_type')!r}"
        )
    return parser(payload, spec)

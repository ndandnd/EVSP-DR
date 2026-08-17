"""Fail-closed parsers for EVSP-DR cross-generation evidence schemas."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import defaultdict
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
    "LP_Route_Weight_Before_Add", "Peak_Trip_Concurrency",
    "Artificial_Trips_Before_Add",
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
TIER_SUFFIX_TYPES = {
    "Time_s": "number",
    "Hit_Timelimit": "boolean",
    "Returned": "integer",
    "Accepted": "integer",
    "Found_Zero": "boolean",
    "Labels_Expanded": "integer",
    "Completed_Routes": "integer",
    "Negative_Completed": "integer",
    "Eligible_Negative_Incidences": "integer",
    "Returned_Trip_Count_Min": "number",
    "Returned_Trip_Count_Mean": "number",
    "Returned_Trip_Count_Max": "number",
    "Label_Cap_Evictions": "integer",
    "Exhaustive": "boolean",
}
CURRENT_HEURISTIC_REQUIRED.update({
    f"Tier{tier}_{suffix}"
    for tier in range(1, 4)
    for suffix in TIER_SUFFIX_TYPES
})
EXACT_ITER_HEADER = (
    "elapsed_s", "iteration", "lp_obj", "route_weight",
    "artificials", "min_rc", "pool_columns",
)
TELEMETRY_SCHEMA = "evsp-dr-exact-cg-phase-telemetry-v1"
MIP_CHECKPOINT_SCHEMA = "evsp-dr-mip-convergence-v1"
HEX = set("0123456789abcdef")


def _hex64(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX for character in value)
    )


def _finite_nonnegative(value, label):
    number = _number(value)
    if number is None or number < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _schedule_fingerprint(routes: list[dict]) -> str:
    projected = []
    for route in routes:
        nodes = route.get("route_nodes", route.get("route"))
        charging = route.get("charging_stops", {})
        if not isinstance(nodes, list) or not isinstance(charging, dict):
            raise ValueError("physical schedule fields are missing")
        projected.append({
            "route_nodes": nodes,
            "charging_stops": charging,
        })
    projected.sort(key=lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ))
    return hashlib.sha256(json.dumps(
        projected, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _number(value, *, allow_positive_infinity=False):
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value {value!r}") from exc
    if math.isfinite(result):
        return result
    if allow_positive_infinity and result == math.inf:
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
    physical_lines = payload.splitlines(keepends=True)
    if not physical_lines:
        raise ValueError("CSV is empty")
    try:
        header_text = physical_lines[0].decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV header is not UTF-8") from exc
    header_rows = list(csv.reader([header_text], strict=True))
    if len(header_rows) != 1 or len(header_rows[0]) != len(set(header_rows[0])):
        raise ValueError("CSV header is malformed or duplicated")
    header = tuple(header_rows[0])
    records = []
    tail_dropped = False
    unterminated = not payload.endswith((b"\n", b"\r"))
    for index, line_bytes in enumerate(physical_lines[1:], start=2):
        final_line = index == len(physical_lines)
        try:
            line = line_bytes.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as exc:
            if final_line and unterminated:
                tail_dropped = True
                break
            raise ValueError(
                f"CSV non-UTF8 corruption at line {index}"
            ) from exc
        if not line.strip():
            continue
        try:
            parsed = next(csv.reader([line], strict=True))
        except (csv.Error, StopIteration) as exc:
            if final_line and unterminated:
                tail_dropped = True
                break
            raise ValueError(f"CSV corruption before EOF at line {index}") from exc
        if len(parsed) != len(header):
            if final_line and unterminated:
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
        "unterminated_final_line": unterminated,
    }


def _tail_safe_jsonl(payload: bytes) -> tuple[list[dict], dict]:
    records = []
    lines = payload.splitlines()
    tail_dropped = False
    unterminated = not payload.endswith((b"\n", b"\r"))
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            incomplete_prefix = (
                isinstance(exc, json.JSONDecodeError)
                and line.lstrip().startswith((b"{", b"["))
                and (
                    exc.pos >= max(0, len(line) - 1)
                    or exc.msg.startswith((
                        "Unterminated string",
                        "Expecting value",
                        "Expecting ',' delimiter",
                        "Expecting property name",
                    ))
                )
                or isinstance(exc, UnicodeDecodeError)
                and exc.end >= len(line)
            )
            if index == len(lines) and unterminated and incomplete_prefix:
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
        "unterminated_final_line": unterminated,
    }


def normalize_termination(value) -> tuple[str | None, str | None]:
    if value in (None, "", "unknown", "running", "initializing"):
        return None, "termination_not_recorded_or_run_not_terminal"
    token = str(value)
    mapping = {
        "certified": "certification",
        "rc_optimal_restricted": "certification_restricted_graph",
        "wall_limit": "wall_limit",
        "active_time_limit_reached": "active_time_limit",
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
        "peak_trip_concurrency": None,
        "best_reduced_cost": None,
        "best_reduced_cost_reason": None,
        "columns_added": None,
        "pool_columns": None,
        "pool_columns_delta": None,
        "wall_time_s": None,
        "process_runtime_s": None,
        "time_clock": None,
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


def _validate_cg_rows(rows: list[dict], schema: str) -> None:
    previous_iteration = 0
    previous_wall = -math.inf
    previous_master = -math.inf
    previous_pricing = -math.inf
    previous_pool = -1
    for row in rows:
        iteration = row["iteration"]
        if iteration <= previous_iteration or iteration <= 0:
            raise ValueError("CG iterations are duplicated/decreasing")
        previous_iteration = iteration
        if schema == "heuristic_dp_historical":
            if (
                row["legacy_master_objective"] is None
                or row["master_time_s"] is None
                or row["pricing_time_s"] is None
                or row["cumulative_master_time_s"] is None
                or row["cumulative_pricing_time_s"] is None
                or row["columns_added"] is None
                or (
                    row["best_reduced_cost"] is None
                    and row["best_reduced_cost_reason"]
                    != "no_finite_pricing_path"
                )
            ):
                raise ValueError("legacy core iteration metrics are required")
            if (
                row["legacy_master_improvement"] is None
                and iteration != 1
            ):
                raise ValueError(
                    "legacy infinite/missing improvement is valid only at "
                    "iteration 1"
                )
        elif schema == "heuristic_dp_current":
            if (
                row["master_objective_before_add"] is None
                or row["master_improvement_before_add"] is None
                or row["lp_route_weight"] is None
                or row["artificial_total"] is None
                or row["peak_trip_concurrency"] is None
                or row["master_time_s"] is None
                or row["pricing_time_s"] is None
                or row["cumulative_master_time_s"] is None
                or row["cumulative_pricing_time_s"] is None
                or row["columns_added"] is None
                or (
                    row["best_reduced_cost"] is None
                    and row["best_reduced_cost_reason"]
                    != "no_finite_pricing_path"
                )
            ):
                raise ValueError("current core iteration metrics are required")
        elif schema == "exact_expanded_network":
            if (
                row["lp_objective"] is None
                or row["lp_route_weight"] is None
                or row["artificial_total"] is None
                or row["pool_columns"] is None
                or (
                    row["best_reduced_cost"] is None
                    and row["best_reduced_cost_reason"]
                    != "no_finite_pricing_path"
                )
            ):
                raise ValueError("exact iteration metrics are required")
        for key in (
            "wall_time_s", "master_time_s", "pricing_time_s",
            "cumulative_master_time_s", "cumulative_pricing_time_s",
            "columns_added", "pool_columns", "artificial_total",
            "artificial_trips", "lp_route_weight",
            "peak_trip_concurrency", "label_cap_evictions",
            "pricing_labels_used", "pricing_label_cap_configured",
            "pricing_completed_routes", "pricing_negative_completed",
            "pricing_eligible_negative_incidences",
            "pricing_returned_trip_count_min",
            "pricing_returned_trip_count_mean",
            "pricing_returned_trip_count_max",
        ):
            value = row.get(key)
            if value is not None and value < 0:
                raise ValueError(f"CG field {key} is negative")
        wall = row.get("wall_time_s")
        if wall is None or wall < previous_wall:
            raise ValueError("CG wall time is missing/decreasing")
        previous_wall = wall
        cumulative_master = row.get("cumulative_master_time_s")
        if cumulative_master is not None:
            if cumulative_master < previous_master:
                raise ValueError("cumulative master time decreased")
            previous_master = cumulative_master
        cumulative_pricing = row.get("cumulative_pricing_time_s")
        if cumulative_pricing is not None:
            if cumulative_pricing < previous_pricing:
                raise ValueError("cumulative pricing time decreased")
            previous_pricing = cumulative_pricing
        pool = row.get("pool_columns")
        if pool is not None:
            if pool < previous_pool:
                raise ValueError("exact pool size decreased")
            previous_pool = pool
        minimum = row.get("pricing_returned_trip_count_min")
        mean = row.get("pricing_returned_trip_count_mean")
        maximum = row.get("pricing_returned_trip_count_max")
        if all(value is not None for value in (minimum, mean, maximum)) and not (
            minimum <= mean <= maximum
        ):
            raise ValueError("returned trip count min/mean/max are inconsistent")
        if schema == "heuristic_dp_current":
            if type(row["timed_out"]) is not bool:
                raise ValueError("current Timed_Out is required")
            highest = row["highest_tier_reached"]
            if highest is None or not 0 <= highest <= 3:
                raise ValueError("highest pricing tier is outside 0..3")
            tiers = json.loads(row["tier_statistics_json"])
            for tier in range(1, 4):
                prefix = f"Tier{tier}_"
                for suffix, kind in TIER_SUFFIX_TYPES.items():
                    value = tiers[prefix + suffix]
                    if (
                        value is not None
                        and kind in {"number", "integer"}
                        and value < 0
                    ):
                        raise ValueError("tier timing/count is negative")
                returned = tiers[prefix + "Returned"]
                accepted = tiers[prefix + "Accepted"]
                if (
                    returned is not None and accepted is not None
                    and accepted > returned
                ):
                    raise ValueError("tier accepted routes exceed returned")
                tmin = tiers[prefix + "Returned_Trip_Count_Min"]
                tmean = tiers[prefix + "Returned_Trip_Count_Mean"]
                tmax = tiers[prefix + "Returned_Trip_Count_Max"]
                if all(v is not None for v in (tmin, tmean, tmax)) and not (
                    tmin <= tmean <= tmax
                ):
                    raise ValueError("tier trip-count statistics are inconsistent")
                values = [
                    tiers[prefix + suffix] for suffix in TIER_SUFFIX_TYPES
                ]
                if tier <= highest:
                    if (
                        tiers[prefix + "Time_s"] is None
                        or tiers[prefix + "Returned"] is None
                        or tiers[prefix + "Accepted"] is None
                        or type(tiers[prefix + "Hit_Timelimit"]) is not bool
                        or type(tiers[prefix + "Exhaustive"]) is not bool
                    ):
                        raise ValueError("reached tier instrumentation is missing")
                elif any(value is not None for value in values):
                    raise ValueError("unreached tier instrumentation is populated")
            if highest:
                prefix = f"Tier{highest}_"
                if (
                    sum(
                        tiers[f"Tier{tier}_Label_Cap_Evictions"] or 0
                        for tier in range(1, highest + 1)
                    ) != row["label_cap_evictions"]
                    or tiers[prefix + "Exhaustive"]
                    != row["pricing_exhaustive"]
                    or tiers[prefix + "Hit_Timelimit"]
                    != row["deepest_tier_timed_out"]
                ):
                    raise ValueError(
                        "deepest-tier aggregate instrumentation mismatch"
                    )


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
                record["Master_Improvement"], allow_positive_infinity=True
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
            "best_reduced_cost": _number(
                record["Best_RC"], allow_positive_infinity=True
            ),
            "best_reduced_cost_reason": (
                "no_finite_pricing_path"
                if record["Best_RC"].strip().lower() in {
                    "inf", "+inf", "infinity", "+infinity"
                } else None
            ),
            "timed_out": _boolean(record["Timed_Out"]),
            "pricing_time_limit_used_s": _number(
                record["Pricing_TimeLimit_Used"]
            ),
            "stagnant_counter": _integer(record["Stagnant_Counter"]),
            "wall_time_s": _number(record["Total_Runtime_s"]),
            "process_runtime_s": _number(record["Total_Runtime_s"]),
            "time_clock": "wall_time",
            "availability_reason": (
                "legacy_schema_has_no_route_weight_artificials_or_tiers"
            ),
        })
        rows.append(row)
    _validate_cg_rows(rows, "heuristic_dp_historical")
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
        tier_statistics = {}
        for tier in range(1, 4):
            for suffix, kind in TIER_SUFFIX_TYPES.items():
                key = f"Tier{tier}_{suffix}"
                raw = record[key]
                if raw == "":
                    value = None
                elif kind == "number":
                    value = _number(raw)
                elif kind == "integer":
                    value = _integer(raw)
                else:
                    value = _boolean(raw)
                tier_statistics[key] = value
        row.update({
            "master_objective_before_add": _number(
                record["Master_Obj_Before_Add"]
            ),
            "master_improvement_before_add": _number(
                record["Master_Improvement_Before_Add"],
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
            "peak_trip_concurrency": _integer(
                record["Peak_Trip_Concurrency"]
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
            "best_reduced_cost": _number(
                record["Best_RC"], allow_positive_infinity=True
            ),
            "best_reduced_cost_reason": (
                "no_finite_pricing_path"
                if record["Best_RC"].strip().lower() in {
                    "inf", "+inf", "infinity", "+infinity"
                } else None
            ),
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
                tier_statistics,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "wall_time_s": (
                _number(record["Cumulative_Master_Time_s"])
                + _number(record["Cumulative_Pricing_Time_s"])
            ),
            "process_runtime_s": _number(record["Total_Runtime_s"]),
            "time_clock": "active_compute_time",
        })
        rows.append(row)
    _validate_cg_rows(rows, "heuristic_dp_current")
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
        min_rc = _number(
            record["min_rc"], allow_positive_infinity=True
        )
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
            "pool_columns_delta": (
                pool_columns - previous_pool
                if previous_pool is not None else None
            ),
            "wall_time_s": _number(record["elapsed_s"]),
            "time_clock": "cumulative_wall_time",
            "master_pricing_split_available": False,
            "availability_reason": (
                "exact_iteration_log_does_not_measure_master_pricing_split"
            ),
        })
        rows.append(row)
        previous_pool = pool_columns
    _validate_cg_rows(rows, "exact_expanded_network")
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
    sessions = {}
    identities = set()
    for start in starts:
        session = start.get("session")
        identity = start.get("identity")
        digest = start.get("identity_sha256")
        if (
            type(session) is not int or session <= 0
            or not isinstance(identity, dict)
            or not _hex64(digest)
            or hashlib.sha256(json.dumps(
                identity, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest() != digest
            or session in sessions
        ):
            raise ValueError("phase telemetry session identity is invalid")
        sessions[session] = digest
        identities.add(digest)
        for key in (
            "output", "csv", "prices_csv", "instance_sha256",
            "prices_sha256", "git_commit", "soc_step", "block_min",
            "g_kwh", "charge_kw", "min_soc_frac",
            "master_sense", "initial_pool",
        ):
            if key not in identity:
                raise ValueError(
                    f"phase telemetry identity lacks {key}"
                )
        if (
            not _hex64(identity["instance_sha256"])
            or not _hex64(identity["prices_sha256"])
            or not isinstance(identity["git_commit"], str)
            or len(identity["git_commit"]) != 40
            or identity["master_sense"] not in {"cover", "partition"}
            or identity["initial_pool"] not in {"artificial", "singletons"}
        ):
            raise ValueError("phase telemetry identity fields are invalid")
        for key in (
            "soc_step", "block_min", "g_kwh",
            "charge_kw", "min_soc_frac",
        ):
            if _number(identity[key]) is None:
                raise ValueError("phase telemetry physics are invalid")
    expected_identity = (spec.get("metadata") or {}).get(
        "telemetry_identity_sha256"
    )
    if len(identities) != 1:
        raise ValueError("phase telemetry contains multiple identities")
    if set(sessions) != set(range(1, len(sessions) + 1)):
        raise ValueError("phase telemetry sessions are not contiguous")
    if expected_identity is not None and identities != {expected_identity}:
        raise ValueError("phase telemetry identity differs from manifest")
    manifest_metadata = spec.get("metadata") or {}
    identity_payload = starts[0]["identity"]
    for metadata_key, identity_key in (
        ("git_commit", "git_commit"),
        ("instance_sha256", "instance_sha256"),
        ("tariff_sha256", "prices_sha256"),
    ):
        expected = manifest_metadata.get(metadata_key)
        if expected is not None and identity_payload.get(identity_key) != expected:
            raise ValueError(
                f"phase telemetry {metadata_key} differs from manifest"
            )
    phases = []
    previous_elapsed = defaultdict(lambda: -math.inf)
    seen_sessions = set()
    next_session = 1
    for row in records:
        if row.get("schema") != TELEMETRY_SCHEMA:
            raise ValueError("phase telemetry schema mismatch")
        record_type = row.get("record_type")
        if record_type == "session_start":
            if row["session"] != next_session:
                raise ValueError("phase telemetry session order is invalid")
            seen_sessions.add(row["session"])
            next_session += 1
            continue
        if record_type != "phase":
            raise ValueError("phase telemetry record_type is unknown")
        session = row.get("session")
        if (
            type(session) is not int
            or session not in seen_sessions
            or row.get("identity_sha256") != sessions[session]
        ):
            raise ValueError("phase telemetry record has mixed identity")
        if not isinstance(row.get("phase"), str) or not row["phase"]:
            raise ValueError("phase telemetry phase is missing")
        duration = _number(row.get("duration_s"))
        elapsed = _number(row.get("elapsed_session_s"))
        if duration is None or duration < 0 or elapsed is None or elapsed < 0:
            raise ValueError("phase telemetry has invalid timing")
        if elapsed < previous_elapsed[session]:
            raise ValueError("phase telemetry elapsed time decreased")
        previous_elapsed[session] = elapsed
        if row.get("outcome") is not None and not isinstance(
                row.get("outcome"), str):
            raise ValueError("phase telemetry outcome is not a string")
        integers = {}
        for key in (
            "iteration", "attempt", "pool_columns", "incidence_nnz",
            "network_nodes", "network_arcs", "peak_rss_bytes",
        ):
            raw = row.get(key)
            if raw is None:
                integers[key] = None
            else:
                value = _integer(raw)
                if value < 0:
                    raise ValueError(f"phase telemetry {key} is negative")
                integers[key] = value
        overhead = row.get("telemetry_overhead_before_s")
        if overhead is not None:
            overhead = _number(overhead)
            if overhead < 0:
                raise ValueError("phase telemetry overhead is negative")
        phases.append({
            "artifact_id": spec["artifact_id"],
            "run_id": spec["run_id"],
            "session": session,
            "phase": row.get("phase"),
            "duration_s": duration,
            "elapsed_session_s": elapsed,
            "iteration": integers["iteration"],
            "attempt": integers["attempt"],
            "pool_columns": integers["pool_columns"],
            "incidence_nnz": integers["incidence_nnz"],
            "network_nodes": integers["network_nodes"],
            "network_arcs": integers["network_arcs"],
            "peak_rss_bytes": integers["peak_rss_bytes"],
            "telemetry_overhead_before_s": overhead,
            "outcome": row.get("outcome"),
            "identity_sha256": next(iter(identities)),
        })
    return {
        "telemetry_rows": phases,
        "tail": tail,
        "schema": "exact_phase_telemetry",
    }


def parse_column_journal(payload: bytes, spec: dict) -> dict:
    records, tail = _tail_safe_jsonl(payload)
    if not records:
        raise ValueError("column journal has no complete records")
    incidences = set()
    incidence_costs = {}
    for index, record in enumerate(records, start=1):
        trips = record.get("trips")
        if (
            not isinstance(trips, list)
            or not trips
            or any(
                not isinstance(trip, int) or isinstance(trip, bool)
                for trip in trips
            )
            or len(trips) != len(set(trips))
        ):
            raise ValueError(f"column journal row {index} has invalid trips")
        cost = _number(record.get("cost"))
        if cost is None:
            raise ValueError(f"column journal row {index} lacks finite cost")
        incidences.add(frozenset(trips))
        incidence_sha = hashlib.sha256(json.dumps(
            sorted(trips), separators=(",", ":")
        ).encode()).hexdigest()
        incidence_costs.setdefault(incidence_sha, set()).add(cost)
    return {
        "tail": tail,
        "schema": "exact_column_journal",
        "journal_summary": {
            "records": len(records),
            "unique_incidences": len(incidences),
            "incidence_sha256": sorted(
                hashlib.sha256(json.dumps(
                    sorted(incidence), separators=(",", ":")
                ).encode()).hexdigest()
                for incidence in incidences
            ),
            "incidence_costs": {
                key: sorted(values)
                for key, values in incidence_costs.items()
            },
        },
    }


def parse_mip_checkpoint(payload: bytes, spec: dict) -> dict:
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("schema") != MIP_CHECKPOINT_SCHEMA:
        raise ValueError("MIP checkpoint schema mismatch")
    if value.get("observational_only") is not True:
        raise ValueError("MIP checkpoint is not observational")
    if (
        value.get("kind") != "checkpoint"
        or value.get("gurobi_tree_restart_supported") is not False
        or value.get("stage") not in {"fleet", "cost", "single"}
        or value.get("incumbent_state") not in {
            "no_incumbent_yet",
            "current_incumbent_at_checkpoint",
            "reused_most_recent_earlier_incumbent",
        }
        or type(value.get("solver_ended_before_checkpoint")) is not bool
    ):
        raise ValueError("MIP checkpoint stage/state/kind is invalid")
    incumbent_raw = value.get("incumbent")
    if incumbent_raw is not None and not isinstance(incumbent_raw, dict):
        raise ValueError("MIP checkpoint incumbent is not an object")
    incumbent = incumbent_raw or {}
    stats = value.get("latest_statistics")
    metadata = value.get("metadata")
    if not isinstance(stats, dict) or not isinstance(metadata, dict):
        raise ValueError("MIP checkpoint metadata/statistics are invalid")
    expected_metadata = spec.get("metadata") or {}
    treatment = expected_metadata.get("treatment")
    if treatment not in {"RAW", "MATCHING", "GIRO"}:
        raise ValueError("MIP checkpoint treatment is missing/invalid")
    expected_arm = "D" if treatment in {"MATCHING", "GIRO"} else "B"
    if metadata.get("experiment_arm") != expected_arm:
        raise ValueError("MIP checkpoint experiment arm mismatch")
    observed_start = metadata.get("source_initial_partition_sha256")
    expected_start = expected_metadata.get("pool_start_sha256")
    if treatment in {"MATCHING", "GIRO"}:
        if not _hex64(expected_start) or observed_start != expected_start:
            raise ValueError(
                "augmented MIP checkpoint start identity mismatch"
            )
    elif observed_start is not None or expected_start is not None:
        raise ValueError("RAW MIP checkpoint contains GIRO start identity")
    for expected_key, observed in (
        ("git_commit", metadata.get("git_commit")),
        ("pool_status_sha256", metadata.get("source_result_sha256")),
        ("pool_journal_sha256", metadata.get("source_journal_sha256")),
        ("pool_start_sha256", metadata.get("source_initial_partition_sha256")),
    ):
        expected = expected_metadata.get(expected_key)
        if expected is not None and expected != observed:
            raise ValueError(
                f"MIP checkpoint {expected_key} differs from manifest"
            )
    for key in ("source_result_sha256", "source_journal_sha256"):
        if not _hex64(metadata.get(key)):
            raise ValueError(f"MIP checkpoint lacks valid {key}")
    if metadata.get("source_initial_partition_sha256") is not None and not (
        _hex64(metadata["source_initial_partition_sha256"])
    ):
        raise ValueError("MIP checkpoint start hash is malformed")
    checkpoint_elapsed = _finite_nonnegative(
        value.get("checkpoint_elapsed_s"), "checkpoint_elapsed_s"
    )
    observed_elapsed = _finite_nonnegative(
        value.get("observed_total_elapsed_s"), "observed_total_elapsed_s"
    )
    statistics_observed = value.get("latest_statistics_observed_s")
    if statistics_observed is not None:
        statistics_observed = _finite_nonnegative(
            statistics_observed, "latest_statistics_observed_s"
        )
        if statistics_observed > observed_elapsed + 1e-9:
            raise ValueError("MIP statistics are observed after publication")
        if statistics_observed > checkpoint_elapsed + 1e-9:
            raise ValueError("MIP statistics are observed after nominal mark")
    if not value["solver_ended_before_checkpoint"] and (
        checkpoint_elapsed > observed_elapsed + 1e-9
    ):
        raise ValueError("live checkpoint precedes its nominal mark")
    if value["solver_ended_before_checkpoint"] != (
        observed_elapsed + 1e-9 < checkpoint_elapsed
    ):
        raise ValueError("MIP ended-before-checkpoint flag is inconsistent")
    normalized_stats = {}
    for key in (
        "statistics_incumbent_fleet",
        "fleet_bound", "objective_bound", "fleet_gap",
        "node_count", "solution_count",
    ):
        raw = stats.get(key)
        if raw is None:
            normalized_stats[key] = None
        elif key == "solution_count":
            normalized_stats[key] = _integer(raw)
            if normalized_stats[key] < 0:
                raise ValueError("MIP solution_count is negative")
        else:
            normalized_stats[key] = _finite_nonnegative(
                raw, f"MIP {key}"
            )
    has_incumbent = value["incumbent_state"] != "no_incumbent_yet"
    if has_incumbent:
        if not isinstance(value.get("incumbent"), dict):
            raise ValueError("MIP incumbent payload is missing")
        fleet = incumbent.get("fleet")
        objective = incumbent.get("objective")
        route_hash = incumbent.get("route_vector_sha256")
        if (
            type(fleet) is not int or fleet <= 0
            or _number(objective) is None
            or not _hex64(route_hash)
        ):
            raise ValueError("MIP incumbent fields are malformed")
        total_elapsed = _finite_nonnegative(
            incumbent.get("total_elapsed_s"), "incumbent total_elapsed_s"
        )
        stage_elapsed = _finite_nonnegative(
            incumbent.get("stage_elapsed_s"), "incumbent stage_elapsed_s"
        )
        if total_elapsed > observed_elapsed + 1e-9:
            raise ValueError("MIP incumbent is observed after checkpoint write")
        if total_elapsed > checkpoint_elapsed + 1e-9:
            raise ValueError("MIP incumbent is observed after nominal mark")
        if stage_elapsed > total_elapsed + 1e-9:
            raise ValueError("MIP stage elapsed exceeds total elapsed")
        expected_state = (
            "current_incumbent_at_checkpoint"
            if math.isclose(
                total_elapsed, checkpoint_elapsed, abs_tol=1e-9
            )
            else "reused_most_recent_earlier_incumbent"
        )
        if value["incumbent_state"] != expected_state:
            raise ValueError("MIP incumbent state/timestamp is inconsistent")
        fleet_bound = normalized_stats["fleet_bound"]
        if fleet_bound is not None:
            gap_fleet = (
                normalized_stats["statistics_incumbent_fleet"]
                if normalized_stats["statistics_incumbent_fleet"] is not None
                else fleet
            )
            if fleet_bound > gap_fleet + 1e-6:
                raise ValueError(
                    "MIP fleet bound exceeds statistics incumbent"
                )
            expected_gap = (
                max(0.0, gap_fleet - fleet_bound)
                / max(1.0, gap_fleet)
            )
            gap = normalized_stats["fleet_gap"]
            if gap is not None and not math.isclose(
                gap, expected_gap, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError("MIP fleet gap is inconsistent")
    elif value.get("incumbent") is not None:
        raise ValueError("no-incumbent checkpoint contains an incumbent")
    else:
        total_elapsed = None
    first_feasible = value.get("first_feasible_incumbent_s")
    if first_feasible is not None:
        first_feasible = _finite_nonnegative(
            first_feasible, "first feasible time"
        )
        if first_feasible > observed_elapsed + 1e-9:
            raise ValueError("first feasible time exceeds observed time")
        if has_incumbent and first_feasible > total_elapsed + 1e-9:
            raise ValueError("first feasible time exceeds incumbent time")
    if not has_incumbent and first_feasible is not None:
        raise ValueError("no-incumbent checkpoint has first-feasible time")
    row = {
        "artifact_id": spec["artifact_id"],
        "run_id": spec["run_id"],
        "algorithm_family": "mip_finite_pool",
        "implementation": (spec.get("metadata") or {}).get("implementation"),
        "scale_family": (spec.get("metadata") or {}).get("scale_family"),
        "scale": (spec.get("metadata") or {}).get("scale"),
        "replicate": (spec.get("metadata") or {}).get("replicate"),
        "treatment": (spec.get("metadata") or {}).get("treatment"),
        "checkpoint_elapsed_s": checkpoint_elapsed,
        "observed_total_elapsed_s": observed_elapsed,
        "statistics_observed_s": statistics_observed,
        "stage": value.get("stage"),
        "incumbent_state": value.get("incumbent_state"),
        "incumbent_fleet": incumbent.get("fleet"),
        "incumbent_objective": incumbent.get("objective"),
        "incumbent_observed_s": total_elapsed,
        "statistics_incumbent_fleet": normalized_stats[
            "statistics_incumbent_fleet"
        ],
        "fleet_bound": normalized_stats["fleet_bound"],
        "objective_bound": normalized_stats["objective_bound"],
        "fleet_gap": normalized_stats["fleet_gap"],
        "node_count": normalized_stats["node_count"],
        "solution_count": normalized_stats["solution_count"],
        "route_vector_sha256": incumbent.get("route_vector_sha256"),
        "first_feasible_s": first_feasible,
        "solver_ended_before_checkpoint": value.get(
            "solver_ended_before_checkpoint"
        ),
        "source_result_sha256": metadata.get("source_result_sha256"),
        "source_journal_sha256": metadata.get("source_journal_sha256"),
        "source_start_sha256": metadata.get(
            "source_initial_partition_sha256"
        ),
        "experiment_arm": metadata.get("experiment_arm"),
        "git_commit": metadata.get("git_commit"),
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
        mip_provenance = value.get("mip_provenance")
        if not isinstance(mip_provenance, dict):
            raise ValueError("MIP final provenance is not an object")
        for expected_key, observed in (
            ("git_commit", mip_provenance.get("observed_git_commit")),
            ("pool_status_sha256", value.get("source_result_sha256")),
            ("pool_journal_sha256", value.get("source_journal_sha256")),
        ):
            expected = metadata.get(expected_key)
            if expected is not None and expected != observed:
                raise ValueError(
                    f"MIP final {expected_key} differs from manifest"
                )
        treatment = metadata.get("treatment")
        if treatment not in {"RAW", "MATCHING", "GIRO"}:
            raise ValueError("MIP final treatment is missing/invalid")
        if value.get("partitioning") is not True:
            raise ValueError("covering MIP result is not a strict schedule")
        if value.get("experiment_arm") != (
            "D" if treatment in {"MATCHING", "GIRO"} else "B"
        ):
            raise ValueError("MIP final experiment arm mismatch")
        start_raw = value.get("mip_start")
        if start_raw is not None and not isinstance(start_raw, dict):
            raise ValueError("MIP start evidence is not an object")
        start = start_raw or {}
        expected_start = metadata.get("pool_start_sha256")
        if treatment in {"MATCHING", "GIRO"}:
            added = start.get("pool_columns_added")
            preserved = start.get("pool_duplicate_incidences_preserved", 0)
            if (
                not _hex64(expected_start)
                or start.get("kind") != "validated_exact_partition"
                or start.get("source_sha256") != expected_start
                or type(added) is not int or added < 0
                or type(preserved) is not int or preserved < 0
                or added + preserved <= 0
                or start.get("assignment_complete") is not True
                or not isinstance(start.get("solver_acceptance"), dict)
                or start["solver_acceptance"].get("accepted") is not True
                or not isinstance(
                    start.get("actual_start_column_hashes"), list
                )
                or not start["actual_start_column_hashes"]
                or any(
                    not _hex64(value)
                    for value in start["actual_start_column_hashes"]
                )
            ):
                raise ValueError(
                    "augmented MIP final start identity mismatch"
                )
        elif (
            expected_start is not None
            or start.get("kind") == "validated_exact_partition"
        ):
            raise ValueError("RAW MIP final contains GIRO augmentation")
        if not _hex64(value.get("source_result_sha256")) or not _hex64(
            value.get("source_journal_sha256")
        ):
            raise ValueError("MIP final source hashes are malformed")
        incumbent_found = value.get("incumbent_found")
        if type(incumbent_found) is not bool:
            raise ValueError("MIP incumbent_found is not boolean")
        buses = value.get("buses")
        selected = value.get("selected_routes")
        selected_incidence_hashes = []
        selected_incidence_costs = {}
        selected_column_hashes = {}
        if incumbent_found:
            if (
                type(buses) is not int or buses <= 0
                or not isinstance(selected, list) or len(selected) != buses
            ):
                raise ValueError("MIP incumbent fleet/routes are malformed")
            counts = {}
            for route in selected:
                trips = route.get("trips") if isinstance(route, dict) else None
                if (
                    not isinstance(trips, list) or not trips
                    or len(trips) != len(set(trips))
                ):
                    raise ValueError("MIP selected route trips are malformed")
                for trip in trips:
                    if not isinstance(trip, int) or isinstance(trip, bool):
                        raise ValueError("MIP selected trip is not an integer")
                    counts[trip] = counts.get(trip, 0) + 1
            if any(count != 1 for count in counts.values()):
                raise ValueError("MIP selected routes are not exact once-only")
            if (
                metadata.get("trip_count") is not None
                and len(counts) != int(metadata["trip_count"])
            ):
                raise ValueError("MIP selected routes have wrong trip count")
            ordered_trips = sorted(counts)
            observed_trip_sha = hashlib.sha256(json.dumps(
                ordered_trips, separators=(",", ":")
            ).encode()).hexdigest()
            if (
                not _hex64(metadata.get("trip_set_sha256"))
                or observed_trip_sha != metadata["trip_set_sha256"]
            ):
                raise ValueError("MIP selected trip set differs from manifest")
            selected_cost = 0.0
            for route in selected:
                cost = _number(route.get("cost"))
                if cost is None:
                    raise ValueError("MIP selected route lacks finite cost")
                selected_cost += cost
                incidence_sha = hashlib.sha256(json.dumps(
                    sorted(route["trips"]), separators=(",", ":")
                ).encode()).hexdigest()
                selected_incidence_hashes.append(incidence_sha)
                selected_incidence_costs[incidence_sha] = cost
                selected_column_hashes[incidence_sha] = hashlib.sha256(
                    json.dumps({
                        "trips": route["trips"],
                        "route_nodes": route.get("route_nodes"),
                        "charging_stops": route.get("charging_stops"),
                        "cost": cost,
                    }, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
        elif buses is not None or selected not in (None, []):
            raise ValueError("MIP no-incumbent result contains a schedule")
        normalized_numbers = {}
        for key in ("mip_obj", "mip_bound", "mip_gap", "runtime_s"):
            raw = value.get(key)
            if raw is None and not incumbent_found and key != "runtime_s":
                normalized_numbers[key] = None
                continue
            number = _number(raw)
            if (
                number is None
                or key in {"mip_gap", "runtime_s"} and number < 0
            ):
                raise ValueError(f"MIP final {key} is invalid")
            normalized_numbers[key] = number
        fleet_proven = value.get("fleet_proven")
        if type(fleet_proven) is not bool:
            raise ValueError("MIP fleet_proven is not boolean")
        optimal_scope = value.get("optimal_scope")
        if optimal_scope not in {
            "none", "fleet_only", "full_pool_lexicographic"
        } or (optimal_scope != "none") != fleet_proven:
            raise ValueError("MIP proof flag/scope are inconsistent")
        if optimal_scope == "full_pool_lexicographic" and (
            value.get("status_name") != "OPTIMAL"
            or normalized_numbers["mip_gap"] != 0.0
            or not math.isclose(
                normalized_numbers["mip_obj"],
                normalized_numbers["mip_bound"],
                rel_tol=1e-10,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("MIP full-pool objective proof is not closed")
        fleet_bound = value.get("fleet_bound")
        if fleet_bound is not None:
            fleet_bound = _number(fleet_bound)
            if fleet_bound is None or fleet_bound < 0:
                raise ValueError("MIP fleet bound is invalid")
        if fleet_proven:
            if (
                not incumbent_found or fleet_bound is None
                or fleet_bound > buses + 1e-6
                or math.ceil(fleet_bound - 1e-6) < buses
            ):
                raise ValueError("MIP fleet proof does not close")
        bound_scope = value.get("mip_bound_scope")
        if bound_scope not in {
            "fleet_count_only_coarse_cost_bound",
            "fixed_proven_fleet_variable_cost",
            "full_pool_objective",
        }:
            raise ValueError("MIP objective bound scope is invalid")
        if (
            optimal_scope == "full_pool_lexicographic"
            and bound_scope != "fixed_proven_fleet_variable_cost"
        ):
            raise ValueError("full MIP proof has the wrong bound scope")
        if (
            optimal_scope == "fleet_only"
            and bound_scope not in {
                "fleet_count_only_coarse_cost_bound",
                "fixed_proven_fleet_variable_cost",
            }
        ):
            raise ValueError("fleet-only proof has the wrong bound scope")
        status_name = value.get("status_name")
        if incumbent_found and status_name in {
            "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"
        }:
            raise ValueError("MIP incumbent contradicts solver status")
        if not incumbent_found and status_name == "OPTIMAL":
            raise ValueError("optimal MIP status lacks an incumbent")
        arguments = mip_provenance.get("arguments")
        if (
            not isinstance(arguments, dict)
            or arguments.get("cover") is not False
            or arguments.get("two_stage") is not True
        ):
            raise ValueError("MIP solver arguments are inconsistent")
        physical = metadata.get("physical_replay_validated")
        if physical not in (True, False, None):
            raise ValueError("MIP physical replay metadata is malformed")
        if (
            physical is True
            and not _hex64(metadata.get(
                "physical_replay_artifact_sha256"
            ))
        ):
            raise ValueError(
                "MIP physical replay claim lacks validation artifact hash"
            )
        if (
            physical is True
            and not _hex64(metadata.get(
                "physical_replay_route_vector_sha256"
            ))
        ):
            raise ValueError(
                "MIP physical replay claim lacks route-vector hash"
            )
        selected_schedule_sha = (
            _schedule_fingerprint(selected)
            if physical is True and incumbent_found else None
        )
        if (
            physical is True
            and selected_schedule_sha
            != metadata["physical_replay_route_vector_sha256"]
        ):
            raise ValueError(
                "MIP selected physical schedule differs from replay hash"
            )
        if incumbent_found and not math.isclose(
            selected_cost,
            normalized_numbers["mip_obj"],
            rel_tol=1e-10,
            abs_tol=1e-6,
        ):
            raise ValueError("MIP selected route costs do not match objective")
        extra_sources = value.get("extra_route_sources")
        if treatment in {"MATCHING", "GIRO"} and not (
            start.get("kind") == "validated_exact_partition"
            and _hex64(start.get("source_sha256"))
        ):
            raise ValueError(
                "augmented MIP final lacks validated augmentation"
            )
        if treatment == "RAW" and extra_sources not in (None, []):
            raise ValueError("RAW MIP final includes extra route sources")
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
            "incumbent_found": incumbent_found,
            "integer_fleet": buses,
            "objective": normalized_numbers["mip_obj"],
            "objective_bound": normalized_numbers["mip_bound"],
            "objective_bound_scope": value.get("mip_bound_scope"),
            "gap": normalized_numbers["mip_gap"],
            "fleet_bound": fleet_bound,
            "fleet_proven": fleet_proven,
            "status_name": value.get("status_name"),
            "optimal_scope": optimal_scope,
            "reported_optimal_scope": optimal_scope,
            "objective_source_bound": treatment == "RAW",
            "objective_validation_reason": (
                None if treatment == "RAW"
                else "pending_verified_augmented_source_cost_binding"
            ),
            "runtime_s": normalized_numbers["runtime_s"],
            "partitioning": True,
            "physically_validated_schedule": (
                physical is True and incumbent_found and bool(selected)
            ),
            "physical_replay_artifact_sha256": metadata.get(
                "physical_replay_artifact_sha256"
            ),
            "physical_replay_route_vector_sha256": metadata.get(
                "physical_replay_route_vector_sha256"
            ),
            "selected_schedule_sha256": selected_schedule_sha,
            "source_result_sha256": value.get("source_result_sha256"),
            "source_journal_sha256": value.get("source_journal_sha256"),
            "source_start_sha256": expected_start,
            "pool_treatment": (
                treatment
            ),
            "giro_columns_added": (
                treatment == "GIRO"
            ),
            "selected_incidence_sha256": sorted(
                selected_incidence_hashes
            ),
            "selected_incidence_costs": selected_incidence_costs,
            "selected_column_hashes": selected_column_hashes,
            "giro_start_column_hashes": (
                start.get("actual_start_column_hashes") or []
            ),
            "git_commit": mip_provenance.get("observed_git_commit"),
        }
        return {"mip_finals": [row], "tail": {}, "schema": "mip_final"}
    if hint == "endpoint_json":
        metadata = spec.get("metadata") or {}
        family = metadata.get("algorithm_family")
        if family == "exact_expanded_network":
            provenance = value.get("provenance")
            stop_reason = value.get("stop_reason")
            certified_flag = value.get("certified_rc_optimal")
            if (
                not isinstance(provenance, dict)
                or stop_reason not in {
                    "certified", "no_path", "wall_limit",
                    "master_failed", "stalled_marginal_returns",
                    "degenerate_stall", "max_iters",
                }
                or _number(value.get("wall_s")) is None
                or _number(value.get("wall_s")) < 0
                or type(certified_flag) is not bool
            ):
                raise ValueError("exact endpoint is malformed/nonterminal")
            rc_eps = _number(provenance.get("rc_eps"))
            if rc_eps is None or rc_eps <= 0:
                raise ValueError("exact endpoint rc_eps is invalid")
            if (stop_reason == "certified") != certified_flag:
                raise ValueError("exact endpoint certification is inconsistent")
            if certified_flag:
                final = value.get("final") or {}
                min_rc = _number(final.get("min_rc"))
                if (
                    stop_reason != "certified"
                    or min_rc is None
                    or min_rc < -rc_eps
                ):
                    raise ValueError(
                        "exact endpoint has false pricing certification"
                    )
        elif family == "heuristic_dp_current":
            termination = (
                value.get("Termination_Reason")
                or value.get("termination_reason")
            )
            if termination not in {
                "target_obj_reached", "stagnation_rolling_window",
                "active_time_limit_reached", "rc_optimal_restricted",
                "pricing_truncated_no_new_columns",
                "no_new_nondominated_columns", "max_iterations_reached",
                "master_failed", "error",
            }:
                raise ValueError("current heuristic endpoint is nonterminal")
            git = value.get("Git")
            if not isinstance(git, dict):
                raise ValueError("current heuristic endpoint lacks Git identity")
            horizons = [
                _number(value.get(key))
                for key in (
                    "Total_Time_s", "Total_Runtime_s",
                    "active_time_s", "Active_Time_s",
                )
                if value.get(key) is not None
            ]
            if not horizons or any(horizon < 0 for horizon in horizons):
                raise ValueError("current heuristic endpoint lacks a horizon")
            if "certified_rc_optimal" in value and (
                type(value["certified_rc_optimal"]) is not bool
                or value["certified_rc_optimal"]
                != (termination == "rc_optimal_restricted")
            ):
                raise ValueError(
                    "current heuristic certification flag is inconsistent"
                )
        return {
            "endpoint": value,
            "tail": {},
            "schema": "endpoint_json",
        }
    if hint == "artifact_manifest_json":
        entries = value.get("files")
        if entries is None:
            entries = value.get("members")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("artifact manifest lacks files/members")
        for name, record in entries.items():
            digest = (
                record if isinstance(record, str)
                else record.get("sha256")
                if isinstance(record, dict) else None
            )
            if (
                not isinstance(name, str) or not name
                or not _hex64(digest)
            ):
                raise ValueError("artifact manifest member hash is invalid")
        return {
            "endpoint": value,
            "tail": {},
            "schema": "artifact_manifest_json",
        }
    if hint == "run_checkpoint_json":
        git = value.get("git")
        if (
            not isinstance(git, dict)
            or not isinstance(git.get("commit"), str)
            or len(git["commit"]) != 40
            or type(git.get("dirty")) is not bool
            or not isinstance(value.get("iteration"), int)
            or not isinstance(value.get("trip_ids"), list)
            or not value["trip_ids"]
            or not _hex64(value.get("instance_sha256"))
            or not _hex64(value.get("price_sha256"))
        ):
            raise ValueError("run checkpoint provenance is incomplete")
        return {
            "checkpoint": value,
            "tail": {},
            "schema": "run_checkpoint_json",
        }
    if hint == "exact_cg_snapshot_json":
        provenance = value.get("provenance")
        mark = value.get("snapshot_mark_minutes")
        if (
            not isinstance(provenance, dict)
            or not isinstance(mark, (int, float))
            or isinstance(mark, bool)
            or mark < 0
            or value.get("stop_reason") != f"snapshot_m{int(mark)}"
            or not isinstance(value.get("trip_ids"), list)
            or not _hex64(provenance.get("instance_sha256"))
            or not _hex64(provenance.get("prices_sha256"))
        ):
            raise ValueError("exact CG snapshot provenance is invalid")
        return {
            "checkpoint": value,
            "tail": {},
            "schema": "exact_cg_snapshot_json",
        }
    if hint == "mip_pool_status_json":
        provenance = value.get("provenance")
        if (
            not isinstance(provenance, dict)
            or not isinstance(value.get("trip_ids"), list)
            or not value["trip_ids"]
            or not _hex64(provenance.get("instance_sha256"))
            or not _hex64(provenance.get("prices_sha256"))
            or not isinstance(value.get("columns"), int)
        ):
            raise ValueError("MIP pool status provenance is invalid")
        return {
            "checkpoint": value,
            "tail": {},
            "schema": "mip_pool_status_json",
        }
    if hint == "route_validation_json":
        routes = value.get("routes")
        if (
            not isinstance(routes, list)
            or not routes
            or value.get("infeasible") not in (None, [])
        ):
            raise ValueError("validated route artifact is missing/partial")
        metadata = spec.get("metadata") or {}
        trip_count = metadata.get("trip_count")
        trip_set_sha = metadata.get("trip_set_sha256")
        instance_sha = metadata.get("instance_sha256")
        physics = value.get("physics")
        if (
            type(trip_count) is not int or trip_count <= 0
            or not _hex64(trip_set_sha)
            or not _hex64(instance_sha)
            or not isinstance(physics, dict)
        ):
            raise ValueError(
                "validated route artifact lacks trip/instance/physics identity"
            )
        counts = {}
        incidence_hashes = []
        for route in routes:
            if not isinstance(route, dict):
                raise ValueError("validated route is not an object")
            nodes = route.get("route", route.get("route_nodes"))
            if not isinstance(nodes, list):
                raise ValueError("validated route nodes are missing")
            trips = [
                node for node in nodes
                if isinstance(node, int) and not isinstance(node, bool)
            ]
            if not trips or len(trips) != len(set(trips)):
                raise ValueError("validated route trip sequence is invalid")
            for trip in trips:
                counts[trip] = counts.get(trip, 0) + 1
            incidence_hashes.append(hashlib.sha256(json.dumps(
                sorted(trips), separators=(",", ":")
            ).encode()).hexdigest())
            if not isinstance(route.get("charging_stops", {}), dict):
                raise ValueError("validated route charging schedule is invalid")
        if (
            len(counts) != trip_count
            or any(count != 1 for count in counts.values())
            or hashlib.sha256(json.dumps(
                sorted(counts), separators=(",", ":")
            ).encode()).hexdigest() != trip_set_sha
        ):
            raise ValueError("validated routes are not the expected exact partition")
        for key in ("g_kwh", "charge_kw", "reserve_frac"):
            if _number(physics.get(key)) is None:
                raise ValueError("validated route physics are incomplete")
        for metadata_key, physics_key in (
            ("battery_kwh", "g_kwh"),
            ("charge_kw", "charge_kw"),
            ("reserve_fraction", "reserve_frac"),
        ):
            if (
                metadata.get(metadata_key) is None
                or not math.isclose(
                    float(metadata[metadata_key]),
                    float(physics[physics_key]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "validated route physics differ from reviewed metadata"
                )
        if not isinstance(value.get("instance_csv"), str):
            raise ValueError("validated route instance source is missing")
        route_vector_sha = _schedule_fingerprint(routes)
        return {
            "checkpoint": {
                **value,
                "route_vector_sha256": route_vector_sha,
                "incidence_sha256": sorted(incidence_hashes),
            },
            "tail": {},
            "schema": "route_validation_json",
        }
    # Preserve other explicitly supported JSON without inventing trajectories.
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
    "exact_cg_column_journal_jsonl": parse_column_journal,
    "mip_checkpoint": parse_json_artifact,
    "mip_final": parse_json_artifact,
    "endpoint_json": parse_json_artifact,
    "artifact_manifest_json": parse_json_artifact,
    "run_checkpoint_json": parse_json_artifact,
    "exact_cg_snapshot_json": parse_json_artifact,
    "mip_pool_status_json": parse_json_artifact,
    "route_validation_json": parse_json_artifact,
}


def parse_artifact(payload: bytes, spec: dict) -> dict:
    parser = PARSERS.get(spec.get("artifact_type"))
    if parser is None:
        raise ValueError(
            f"unsupported artifact_type {spec.get('artifact_type')!r}"
        )
    is_csv = spec.get("artifact_type") in {
        "heuristic_dp_historical_csv",
        "heuristic_dp_current_csv",
        "exact_cg_iterations_csv",
    }
    if (
        is_csv
        and not payload.endswith((b"\n", b"\r"))
        and not (spec.get("metadata") or {}).get(
            "allow_complete_unterminated"
        )
    ):
        if b"\n" not in payload:
            raise ValueError("CSV has no committed data row")
        result = parser(payload.rsplit(b"\n", 1)[0] + b"\n", spec)
        result["tail"] = {
            "tail_dropped": True,
            "tail_reason": "uncommitted_unterminated_final_csv_row",
            "unterminated_final_line": True,
        }
        return result
    try:
        return parser(payload, spec)
    except ValueError:
        last_line = payload.rsplit(b"\n", 1)[-1].decode(
            "utf-8", errors="ignore"
        )
        partial_numeric = any(
            re.fullmatch(
                r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?",
                token.strip(),
            )
            or re.fullmatch(r"[+-]?\d+\.", token.strip())
            for token in last_line.split(",")
        )
        if (
            is_csv and partial_numeric
            and not payload.endswith((b"\n", b"\r"))
            and b"\n" in payload
        ):
            result = parser(payload.rsplit(b"\n", 1)[0] + b"\n", spec)
            result["tail"] = {
                "tail_dropped": True,
                "tail_reason": "interrupted_final_numeric_token",
                "unterminated_final_line": True,
            }
            return result
        raise

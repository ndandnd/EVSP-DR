"""Scientific schemas and metrics for the tariff-response experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from scipy.optimize import linear_sum_assignment

from build_tariff_response_manifest import (
    MANIFEST_FIELDS,
    REPO_ROOT,
    STATIONS,
    read_temporal,
    sha256_file,
    validate_spatial,
)
from config import BUS_COST_KX, CHARGE_START_COST
from make_giro_seed_routes import _minutes, _station_node
from prepare_k40_giro40_partition import (
    EXCLUDED_VARIANTS,
    EXPECTED_MASTER_SHA256,
    INCLUDED_DUTIES,
)


TARIFF_MANIFEST_SCHEMA = "evsp-dr-tariff-manifest-v1"
TIER0_SCHEMA = "evsp-dr-giro-original-v1"
ROUTE_RESPONSE_SCHEMA = "evsp-dr-route-response-v1"
PHYSICS = {
    "g_kwh": 300.0,
    "charge_kw": 300.0,
    "reserve_kwh": 0.0,
    "soc_step": 15.0,
    "block_min": 10,
    "terminal_soc_policy": "depot_arrival_soc_at_least_reserve",
}
GIRO_RECHARGE_SOURCE_SHA256 = (
    "5678f907ac4c3dfc43d4acbd5faf2645b50063a874b9c9252bd5d463214e232b"
)
GIRO40_PARTITION_PATH = (
    REPO_ROOT
    / "analysis/k40_giro40_partition_20260818/giro40_partition.json"
)
GIRO40_PARTITION_SHA256 = (
    "8f9944f93f26cf0121e9ecab2fa412d573e90a0189b7a38008d3b2535f54d428"
)


def canonical_sha(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def load_tariff_manifest(path: Path) -> list[dict]:
    path = path.expanduser().resolve()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != MANIFEST_FIELDS:
        raise ValueError("unexpected tariff manifest schema")
    if len({row["tariff_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate tariff IDs")
    for row in rows:
        artifact = REPO_ROOT / row["relative_path"]
        if not artifact.is_file() or sha256_file(artifact) != row["sha256"]:
            raise ValueError(f"tariff hash mismatch: {row['tariff_id']}")
        if row["format"] == "temporal_hourly":
            read_temporal(artifact)
        elif row["format"] == "station_hourly":
            validate_spatial(artifact)
        else:
            raise ValueError("unknown tariff format")
        if row["availability"] != "available":
            raise ValueError("pilot tariff is unavailable")
    return sorted(rows, key=lambda row: row["tariff_id"])


def tariff_prices(row: dict) -> dict[str, dict[int, float]]:
    path = REPO_ROOT / row["relative_path"]
    if row["format"] == "temporal_hourly":
        curve = read_temporal(path)
        return {station: dict(curve) for station in STATIONS}
    prices = {station: {} for station in STATIONS}
    with path.open(newline="") as handle:
        for item in csv.DictReader(handle):
            prices[item["station"]][int(float(item["time_block"]))] = float(
                item["cost"]
            )
    if any(not curve for curve in prices.values()):
        raise ValueError("spatial tariff omits a modeled station")
    return prices


def _base_duty(value: str) -> str:
    return (
        "13316" if value in {"13316m", "13316uwt"}
        else "13324" if value in {"13324muw", "13324t"}
        else value
    )


def _source_recharge_identity(events: list[dict]) -> str:
    return canonical_sha([{
        "source_row": event["source_row"],
        "duty_id": event["duty_id"],
        "from": event["raw_from"],
        "to": event["raw_to"],
        "start": event["raw_start"],
        "end": event["raw_end"],
        "kwh": event["kwh"],
    } for event in events])


def reconstruct_giro40_original(
    master_path: Path,
    partition_path: Path = GIRO40_PARTITION_PATH,
) -> dict:
    master_path = master_path.expanduser().resolve()
    partition_path = partition_path.expanduser().resolve()
    if sha256_file(master_path) != EXPECTED_MASTER_SHA256:
        raise ValueError("GIRO master hash mismatch")
    if sha256_file(partition_path) != GIRO40_PARTITION_SHA256:
        raise ValueError("GIRO40 partition hash mismatch")
    partition = json.loads(partition_path.read_text())
    duties = list(partition.get("included_duties") or [])
    if duties != sorted(INCLUDED_DUTIES):
        raise ValueError("GIRO40 duty variants differ")
    routes_by_duty = {
        route["giro_duty_id"]: route
        for route in partition["routes"]
    }
    if set(routes_by_duty) != set(INCLUDED_DUTIES):
        raise ValueError("partition duty routes differ")

    frame = pd.read_csv(master_path)
    frame["VehicleTask"] = frame["VehicleTask"].astype(str)
    selected = frame[frame["VehicleTask"].isin(INCLUDED_DUTIES)].copy()
    literal = set(frame["VehicleTask"].unique())
    if literal - set(INCLUDED_DUTIES) != set(EXCLUDED_VARIANTS):
        raise ValueError("unexpected weekday variants")
    events = []
    for index, row in selected.iterrows():
        if row["Identifier"] != "Recharge":
            continue
        duty = str(row["VehicleTask"])
        raw_from = str(row["From1"])
        raw_to = str(row["To1"])
        raw_start = str(row["Start1"])
        raw_end = str(row["End1"])
        try:
            kwh = float(row["Recharge kWh"])
            start = _minutes(raw_start)
            end = _minutes(raw_end)
        except (TypeError, ValueError) as exc:
            raise ValueError("ambiguous GIRO recharge value") from exc
        station = _station_node(raw_from)
        if (
            not math.isfinite(kwh)
            or kwh <= 0.0
            or end <= start
            or raw_from != raw_to
            or station.rsplit("_", 1)[0] not in STATIONS
        ):
            raise ValueError(f"invalid GIRO recharge row {index + 2}")
        events.append({
            "source_row": int(index + 2),
            "duty_id": duty,
            "base_duty_id": _base_duty(duty),
            "station": station,
            "station_base": station.rsplit("_", 1)[0],
            "raw_from": raw_from,
            "raw_to": raw_to,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "start_min": start,
            "end_min": end,
            "kwh": kwh,
            "implied_kw": kwh * 60.0 / (end - start),
        })
    if len(events) != 344:
        raise ValueError("GIRO40 recharge count changed")
    if _source_recharge_identity(events) != GIRO_RECHARGE_SOURCE_SHA256:
        raise ValueError("GIRO40 recharge source identity changed")

    duty_rows = []
    for duty in sorted(INCLUDED_DUTIES):
        route = routes_by_duty[duty]
        duty_frame = selected[selected["VehicleTask"] == duty]
        regular = duty_frame[
            (duty_frame["Identifier"] == "Regular")
            & duty_frame["Ordered_Trip_ID"].notna()
        ]
        source_ids = [int(value) for value in regular["Ordered_Trip_ID"]]
        local_ids = list(route["trips"])
        if not source_ids or not local_ids:
            raise ValueError(f"duty has no trips: {duty}")
        duty_events = [
            event for event in events if event["duty_id"] == duty
        ]
        duty_rows.append({
            "duty_id": duty,
            "base_duty_id": _base_duty(duty),
            "included_variant": True,
            "excluded_variant_id": next((
                variant for variant, reason in EXCLUDED_VARIANTS.items()
                if _base_duty(variant) == _base_duty(duty)
            ), ""),
            "trip_count": len(local_ids),
            "local_trip_ids_json": json.dumps(local_ids, separators=(",", ":")),
            "source_ordered_trip_ids_json": json.dumps(
                source_ids, separators=(",", ":")
            ),
            "route_incidence_sha256": canonical_sha(sorted(local_ids)),
            "recorded_charge_count": len(duty_events),
            "recorded_charge_kwh": sum(
                event["kwh"] for event in duty_events
            ),
        })
    coverage = Counter(
        trip
        for route in routes_by_duty.values()
        for trip in route["trips"]
    )
    if (
        len(duty_rows) != 40
        or set(coverage) != set(range(947))
        or any(count != 1 for count in coverage.values())
    ):
        raise ValueError("GIRO40 duties are not an exact partition")
    return {
        "schema": TIER0_SCHEMA,
        "master_sha256": EXPECTED_MASTER_SHA256,
        "partition_sha256": GIRO40_PARTITION_SHA256,
        "recharge_source_sha256": GIRO_RECHARGE_SOURCE_SHA256,
        "duties": duty_rows,
        "events": events,
        "routes": [
            {
                "duty_id": duty,
                "trips": routes_by_duty[duty]["trips"],
            }
            for duty in sorted(routes_by_duty)
        ],
        "physics": dict(PHYSICS),
        "recorded_terminal_soc_policy":
            "unavailable_no_declared_operator_policy",
    }


def giro_routes_for_instance(
    master_path: Path,
    instance_path: Path,
) -> list[dict]:
    if sha256_file(master_path) != EXPECTED_MASTER_SHA256:
        raise ValueError("GIRO master hash mismatch")
    instance = pd.read_csv(instance_path)
    if "Ordered_Trip_ID" not in instance:
        raise ValueError("instance lacks Ordered_Trip_ID")
    ordered = [int(value) for value in instance["Ordered_Trip_ID"]]
    if len(ordered) != len(set(ordered)):
        raise ValueError("instance repeats Ordered_Trip_ID")
    frame = pd.read_csv(master_path)
    frame["VehicleTask"] = frame["VehicleTask"].astype(str)
    regular = frame[
        (frame["Identifier"] == "Regular")
        & frame["Ordered_Trip_ID"].notna()
    ].copy()
    regular["Ordered_Trip_ID"] = regular["Ordered_Trip_ID"].astype(int)
    duty_of = dict(zip(
        regular["Ordered_Trip_ID"], regular["VehicleTask"]
    ))
    missing = [trip for trip in ordered if trip not in duty_of]
    if missing:
        raise ValueError(f"instance has foreign GIRO trips: {missing[:10]}")
    duties = sorted({duty_of[trip] for trip in ordered})
    if len({_base_duty(duty) for duty in duties}) != len(duties):
        raise ValueError("instance contains weekday-variant siblings")
    ordered_to_local = {
        source_trip: local for local, source_trip in enumerate(ordered)
    }
    routes = []
    for duty in duties:
        source_trips = [
            int(value) for value in regular.loc[
                regular["VehicleTask"] == duty, "Ordered_Trip_ID"
            ]
            if int(value) in ordered_to_local
        ]
        local_trips = [
            ordered_to_local[value] for value in source_trips
        ]
        if not local_trips:
            raise ValueError(f"selected duty has no trips: {duty}")
        routes.append({
            "duty_id": duty,
            "base_duty_id": _base_duty(duty),
            "trips": local_trips,
            "source_ordered_trip_ids": source_trips,
        })
    counts = Counter(
        trip for route in routes for trip in route["trips"]
    )
    if (
        set(counts) != set(range(len(instance)))
        or any(value != 1 for value in counts.values())
    ):
        raise ValueError("GIRO instance duties are not an exact partition")
    return routes


def evaluate_giro_original(
    original: dict,
    tariff_row: dict,
) -> tuple[dict, list[dict]]:
    prices = tariff_prices(tariff_row)
    cost = 0.0
    unavailable = set()
    event_rows = []
    for event in original["events"]:
        curve = prices[event["station_base"]]
        first_hour = event["start_min"] // 60
        last_hour = (event["end_min"] - 1) // 60
        hours = list(range(first_hour, last_hour + 1))
        missing = [hour for hour in hours if hour not in curve]
        unique_prices = {
            curve[hour] for hour in hours if hour in curve
        }
        reason = ""
        exact_event_cost = None
        if missing:
            reason = "tariff_hour_missing"
        elif len(unique_prices) != 1:
            reason = "within_window_energy_allocation_unavailable"
        else:
            exact_event_cost = event["kwh"] * next(iter(unique_prices))
            cost += exact_event_cost
        if reason:
            unavailable.add(reason)
        event_rows.append({
            **event,
            "tier": "TIER0_GIRO_ORIGINAL",
            "tariff_id": tariff_row["tariff_id"],
            "block_kind": "recorded_window_unsplit",
            "price_per_kwh": (
                next(iter(unique_prices))
                if exact_event_cost is not None else None
            ),
            "event_cost": exact_event_cost,
            "availability": "available" if not reason else "unavailable",
            "availability_reason": reason,
        })
    scalar_available = not unavailable
    total_kwh = sum(event["kwh"] for event in original["events"])
    starts = len(original["events"])
    return {
        "tier": "TIER0_GIRO_ORIGINAL",
        "tariff_id": tariff_row["tariff_id"],
        "buses": 40,
        "grid_model_objective": (
            BUS_COST_KX * 40 + CHARGE_START_COST * starts + cost
            if scalar_available else None
        ),
        "continuous_replay_objective": None,
        "charging_cost": (
            CHARGE_START_COST * starts + cost
            if scalar_available else None
        ),
        "total_charged_kwh": total_kwh,
        "charging_starts": starts,
        "discretized_certification_status": "not_applicable_recorded",
        "physical_replay_status":
            "unavailable_recorded_power_profile_ambiguous",
        "terminal_soc_policy": original[
            "recorded_terminal_soc_policy"
        ],
        "scalar_cost_availability": (
            "available" if scalar_available else "unavailable"
        ),
        "availability_reason": ";".join(sorted(unavailable)),
        "continuous_cost_pricing_certified": False,
        "certificate_scope": "none_recorded_schedule",
    }, event_rows


def route_identity(routes: list[dict]) -> str:
    incidences = sorted(
        canonical_sha(sorted(route["trips"])) for route in routes
    )
    return canonical_sha(incidences)


def _route_sets(routes):
    return [set(route["trips"]) for route in routes]


def _neighbor_maps(routes):
    predecessor, successor = {}, {}
    for route in routes:
        trips = list(route["trips"])
        for index, trip in enumerate(trips):
            predecessor[trip] = trips[index - 1] if index else None
            successor[trip] = (
                trips[index + 1] if index + 1 < len(trips) else None
            )
    return predecessor, successor


def _adjacencies(routes):
    return {
        (left, right)
        for route in routes
        for left, right in zip(route["trips"], route["trips"][1:])
    }


def _coassigned_pairs(routes):
    return {
        (left, right)
        for route in routes
        for index, left in enumerate(sorted(route["trips"]))
        for right in sorted(route["trips"])[index + 1:]
    }


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def route_response(baseline: list[dict], candidate: list[dict]) -> dict:
    baseline_sets = _route_sets(baseline)
    candidate_sets = _route_sets(candidate)
    baseline_trips = set().union(*baseline_sets) if baseline_sets else set()
    candidate_trips = set().union(*candidate_sets) if candidate_sets else set()
    if baseline_trips != candidate_trips or not baseline_trips:
        raise ValueError("route response trip sets differ")
    if (
        sum(map(len, baseline_sets)) != len(baseline_trips)
        or sum(map(len, candidate_sets)) != len(candidate_trips)
    ):
        raise ValueError("route response requires exact partitions")
    overlap = [
        [-len(left & right) for right in candidate_sets]
        for left in baseline_sets
    ]
    row_ind, column_ind = linear_sum_assignment(overlap)
    retained = -sum(overlap[row][column] for row, column in zip(
        row_ind, column_ind
    ))
    b_prev, b_next = _neighbor_maps(baseline)
    c_prev, c_next = _neighbor_maps(candidate)
    predecessor_changes = sum(
        b_prev[trip] != c_prev[trip] for trip in baseline_trips
    )
    successor_changes = sum(
        b_next[trip] != c_next[trip] for trip in baseline_trips
    )
    intact = sum(
        tuple(route["trips"]) in {
            tuple(candidate_route["trips"])
            for candidate_route in candidate
        }
        for route in baseline
    )
    return {
        "schema": ROUTE_RESPONSE_SCHEMA,
        "trip_count": len(baseline_trips),
        "baseline_buses": len(baseline),
        "candidate_buses": len(candidate),
        "trips_assigned_to_different_duty": (
            len(baseline_trips) - retained
        ),
        "percent_trips_assigned_to_different_duty":
            100.0 * (len(baseline_trips) - retained) / len(baseline_trips),
        "predecessor_changes": predecessor_changes,
        "successor_changes": successor_changes,
        "trip_adjacency_jaccard": _jaccard(
            _adjacencies(baseline), _adjacencies(candidate)
        ),
        "trip_coassignment_jaccard": _jaccard(
            _coassigned_pairs(baseline), _coassigned_pairs(candidate)
        ),
        "intact_giro_duties_retained": intact,
        "duties_split": sum(
            len({
                index for index, current in enumerate(candidate_sets)
                if baseline_set & current
            }) > 1
            for baseline_set in baseline_sets
        ),
        "duties_merged": sum(
            len({
                index for index, original in enumerate(baseline_sets)
                if candidate_set & original
            }) > 1
            for candidate_set in candidate_sets
        ),
        "selected_giro_columns": sum(
            candidate_set in baseline_sets
            for candidate_set in candidate_sets
        ),
        "newly_generated_columns": sum(
            candidate_set not in baseline_sets
            for candidate_set in candidate_sets
        ),
        "baseline_route_identity_sha256": route_identity(baseline),
        "candidate_route_identity_sha256": route_identity(candidate),
    }


def savings_decomposition(
    tier0_cost, tier1_cost, tier2_cost
) -> dict:
    if any(value is None for value in (tier0_cost, tier1_cost, tier2_cost)):
        return {
            "charging_only_savings": None,
            "rerouting_increment": None,
            "total_price_aware_savings": None,
            "availability": "unavailable_missing_comparable_cost",
        }
    return {
        "charging_only_savings": tier0_cost - tier1_cost,
        "rerouting_increment": tier1_cost - tier2_cost,
        "total_price_aware_savings": tier0_cost - tier2_cost,
        "availability": "available",
    }

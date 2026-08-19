#!/usr/bin/env python3
"""Trace fixed-duty grid transitions and bounded non-certificate alternatives."""

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

from audit_giro_known_columns import DEPOT, HORIZON_MIN, STATIONS, build_problem
from audit_scale_ladder_known_membership import FLAT_SHA256, _prices
from build_tariff_response_manifest import REPO_ROOT, sha256_file
from fixed_duty_expanded_optimizer import (
    _arc_groups,
    _floor,
    optimize_fixed_duty,
)
from rerealize_routes import _arc_map, rerealize_route
from run_exact_pool_mip import validate_injected_route
from scale_ladder_trip_identity import identity
from summarize_scale_ladder import _rename_noreplace
from tariff_response_core import giro_routes_for_instance


SCHEMA = "evsp-dr-fixed-duty-grid-transition-oracle-v1"
COUNTERFACTUAL_SCOPE = "diagnostic_counterfactual_not_a_certificate"
DEFAULT_OUTPUT = (
    REPO_ROOT / "analysis/duty_13411_grid_transition_oracle_20260819"
)
DEFAULT_K5 = (
    REPO_ROOT
    / "data/scale_ladder/instances/Practice_Custom_DutyUnion_k05_r1.csv"
)
DEFAULT_K40 = (
    REPO_ROOT
    / "data/tariff_response/frozen_instances/"
    "Practice_Custom_DutyUnion_k40_r2.csv"
)
DEFAULT_K5_SHA256 = (
    "fc10ac0707becb960364e76b8c1e1c414d5d5639cbc3b7dadaf67a77e03f5322"
)
DEFAULT_K40_SHA256 = (
    "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd"
)
DEFAULT_ORACLE_SPEC = {
    "cell_id": "k05_s1",
    "duty_id": "13411",
    "instance_path":
        "data/scale_ladder/instances/Practice_Custom_DutyUnion_k05_r1.csv",
    "instance_sha256": DEFAULT_K5_SHA256,
    "grids": [
        {"soc_step": 15.0, "block_min": 10},
        {"soc_step": 5.0, "block_min": 10},
        {"soc_step": 2.5, "block_min": 10},
        {"soc_step": 1.0, "block_min": 10},
        {"soc_step": 1.0, "block_min": 5},
    ],
    "comparison_instance_path":
        "data/tariff_response/frozen_instances/"
        "Practice_Custom_DutyUnion_k40_r2.csv",
    "comparison_instance_sha256": DEFAULT_K40_SHA256,
}
PHYSICS = {
    "g_kwh": 300.0,
    "charge_kw": 300.0,
    "reserve_kwh": 0.0,
}
MODES = (
    "production_block_timing_production_soc_flooring",
    "continuous_timing_production_soc_flooring",
    "production_block_timing_without_soc_flooring",
    "continuous_timing_continuous_soc",
)
TRANSITION_FIELDS = (
    "cell_id", "duty_id", "soc_step", "block_min",
    "failed_local_from_trip", "failed_local_to_trip",
    "failed_ordered_from_trip", "failed_ordered_to_trip",
    "position", "option_kind", "station", "predecessor_level",
    "predecessor_soc_kwh", "trip_energy_kwh", "soc_after_trip_kwh",
    "trip_to_station_arc_type", "station_to_successor_arc_type",
    "direct_arc_type", "station_arrival_min",
    "arrival_min",
    "station_arrival_soc_before_floor_kwh", "station_entry_level",
    "station_entry_soc_kwh", "first_charging_block",
    "last_charging_block", "last_possible_charging_block",
    "usable_blocks", "usable_minutes", "delayed_charging",
    "charge_gain_before_floor_kwh", "charge_gain_after_floor_kwh",
    "cumulative_grid_charge_gain_kwh", "battery_cap_kwh",
    "departure_min", "deadline_min", "outgoing_deadhead_min",
    "outgoing_deadhead_kwh", "resulting_soc_before_floor_kwh",
    "resulting_soc_level", "resulting_soc_kwh",
    "successor_energy_kwh", "reserve_kwh", "accepted",
    "predicates_json", "failed_predicates_json",
)
FRONTIER_FIELDS = (
    "cell_id", "duty_id", "soc_step", "block_min", "position",
    "local_trip", "ordered_trip", "local_successor", "ordered_successor",
    "level", "soc_kwh", "cost", "actions_json",
)
COUNTERFACTUAL_FIELDS = (
    "cell_id", "duty_id", "soc_step", "block_min",
    "failed_local_from_trip", "failed_local_to_trip",
    "failed_ordered_from_trip", "failed_ordered_to_trip",
    "mode", "certificate_scope", "feasible", "option_kind", "station",
    "start_soc_after_trip_kwh", "station_arrival_soc_kwh",
    "available_minutes", "usable_blocks", "charge_gain_kwh",
    "outgoing_deadhead_kwh", "resulting_soc_kwh",
    "successor_energy_plus_reserve_kwh", "binding_margin_kwh",
    "binding_inequality", "witness_json",
)


def _canonical(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha_payload(payload):
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _repo_relative(path):
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def _producer_hashes():
    paths = (
        "src/audit_fixed_duty_grid_transitions.py",
        "src/fixed_duty_expanded_optimizer.py",
        "src/rerealize_routes.py",
        "src/run_exact_pool_mip.py",
        "src/expanded_path_realization.py",
        "src/audit_giro_known_columns.py",
        "src/audit_scale_ladder_known_membership.py",
        "src/build_tariff_response_manifest.py",
        "src/config.py",
        "src/scale_ladder_trip_identity.py",
        "src/tariff_response_core.py",
        "src/utils_v2.py",
        "src/pricing_dp_og.py",
        "src/prepare_k40_giro40_partition.py",
        "src/summarize_scale_ladder.py",
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


def _load_duty(instance_path, expected_sha, duty_id):
    instance_path = Path(instance_path).resolve()
    if sha256_file(instance_path) != expected_sha:
        raise ValueError("oracle instance SHA-256 mismatch")
    identities = identity(instance_path)
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
    matches = [
        row for row in routes if str(row["duty_id"]) == str(duty_id)
    ]
    if len(matches) != 1:
        raise ValueError("oracle duty is missing or duplicated")
    route = matches[0]
    problem = build_problem(
        instance_path.parent,
        instance_path.name,
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=REPO_ROOT / "data",
    )
    return {
        "instance_path": instance_path,
        "identities": identities,
        "local_to_ordered": local_to_ordered,
        "route": route,
        "problem": problem,
        "arcs": _arc_groups(problem),
        "arc_map": _arc_map(problem),
    }


def _continuous_replay_trace(problem, record, arc):
    nodes = list(record["route"])
    stops = record["charging_stops"]
    stop_rows = list(zip(
        stops.get("stations") or [],
        stops.get("cst") or [],
        stops.get("cet") or [],
        stops.get("kwh") or [],
    ))
    stop_index = 0
    soc = PHYSICS["g_kwh"]
    trace = []
    previous = nodes[0]
    for position, node in enumerate(nodes[1:], start=1):
        is_last = position == len(nodes) - 1
        key = None if (
            isinstance(previous, str)
            and isinstance(node, str)
            and previous == node
        ) else (previous, node)
        travel, deadhead = arc.get(key, (0.0, 0.0))
        soc -= float(deadhead)
        event = {
            "from_node": previous,
            "node": node,
            "travel_min": float(travel),
            "deadhead_kwh": float(deadhead),
            "soc_after_deadhead_kwh": soc,
        }
        if isinstance(node, int) and not isinstance(node, bool):
            soc -= float(problem.trip_energy[node])
            event.update({
                "event": "trip",
                "trip_energy_kwh": float(problem.trip_energy[node]),
                "soc_after_trip_kwh": soc,
            })
        elif not is_last:
            if (
                stop_index < len(stop_rows)
                and stop_rows[stop_index][0] == node
            ):
                station, cst, cet, kwh = stop_rows[stop_index]
                soc += float(kwh)
                event.update({
                    "event": "charge",
                    "station": station,
                    "cst": float(cst),
                    "cet": float(cet),
                    "charged_kwh": float(kwh),
                    "soc_after_charge_kwh": soc,
                })
                stop_index += 1
            else:
                event["event"] = "station_wait"
        else:
            event["event"] = "depot_return"
        trace.append(event)
        previous = node
    if stop_index != len(stop_rows):
        raise ValueError("continuous witness leaves stops unconsumed")
    return trace, soc


def _continuous_witness(loaded, prices):
    problem = loaded["problem"]
    route = loaded["route"]
    record, objective, reason = rerealize_route(
        route["trips"],
        problem,
        loaded["arc_map"],
        prices,
        PHYSICS["g_kwh"],
        PHYSICS["charge_kw"],
        PHYSICS["reserve_kwh"],
    )
    if reason is not None or record is None:
        raise ValueError(f"continuous whole-duty witness failed: {reason}")
    replay_record = {
        "route_nodes": record["route"],
        "charging_stops": record["charging_stops"],
    }
    verdict = validate_injected_route(
        problem,
        replay_record,
        PHYSICS["g_kwh"],
        PHYSICS["charge_kw"],
        PHYSICS["reserve_kwh"],
        HORIZON_MIN,
        arc_map=loaded["arc_map"],
    )
    if verdict is not None:
        raise ValueError(f"continuous witness physical replay failed: {verdict}")
    trace, terminal_soc = _continuous_replay_trace(
        problem, record, loaded["arc_map"]
    )
    return {
        "solver": "scipy.optimize.milp/HiGHS",
        "solver_status": "success",
        "objective": float(objective),
        "objective_semantics":
            "deadhead_plus_charging_plus_charge_start_no_bus_fixed_cost",
        "route": record["route"],
        "charging_actions": record["charging_stops"],
        "terminal_soc_kwh": terminal_soc,
        "physical_validation": "validated_injected_route",
        "record_sha256": _sha_payload(record),
        "trace_sha256": _sha_payload(trace),
        "trace": trace,
    }


def _floor_soc(value, soc_step, g_kwh):
    levels = int(g_kwh / soc_step) + 1
    grid = [round(index * soc_step, 6) for index in range(levels)]
    level = _floor(grid, soc_step, value)
    return level, (grid[level] if level >= 0 else None)


def evaluate_counterfactual_transition(
    problem,
    arcs,
    *,
    trip,
    successor,
    soc_step,
    block_min,
    production_soc_after_trip,
    no_floor_prefix_soc_after_trip,
    mode,
):
    """Evaluate one bounded transition counterfactual.

    This is diagnostic reachability only and never a feasibility certificate.
    """
    if mode not in MODES:
        raise ValueError(f"unknown counterfactual mode: {mode}")
    continuous_timing = mode.startswith("continuous_timing")
    production_flooring = mode.endswith("production_soc_flooring")
    start_soc = (
        float(production_soc_after_trip)
        if production_flooring
        else float(no_floor_prefix_soc_after_trip)
    )
    deadline = float(problem.start_min[successor])
    required = (
        float(problem.trip_energy[successor])
        + PHYSICS["reserve_kwh"]
    )
    options = []
    direct = arcs["trip_trip"].get(trip, {}).get(successor)
    if direct is not None:
        arrival = float(problem.end_min[trip]) + direct.travel_min
        resulting = start_soc - direct.deadhead_kwh
        if production_flooring:
            _level, resulting = _floor_soc(
                resulting, soc_step, PHYSICS["g_kwh"]
            )
        margin = resulting - required
        options.append({
            "option_kind": "direct",
            "station": None,
            "feasible": (
                arrival <= deadline + 1e-9
                and resulting >= required - 1e-9
            ),
            "start_soc_after_trip_kwh": start_soc,
            "station_arrival_soc_kwh": None,
            "available_minutes": 0.0,
            "usable_blocks": 0,
            "charge_gain_kwh": 0.0,
            "outgoing_deadhead_kwh": direct.deadhead_kwh,
            "resulting_soc_kwh": resulting,
            "successor_energy_plus_reserve_kwh": required,
            "binding_margin_kwh": margin,
            "binding_inequality":
                f"{resulting:.9f} >= {required:.9f}",
        })
    for station in sorted(STATIONS):
        inbound = arcs["trip_station"].get(trip, {}).get(station)
        outbound = arcs["station_trip"].get(station, {}).get(successor)
        if inbound is None or outbound is None:
            continue
        arrival_min = float(problem.end_min[trip]) + inbound.travel_min
        latest_departure = deadline - outbound.travel_min
        arrival_soc = start_soc - inbound.deadhead_kwh
        if production_flooring:
            _entry_level, charge_start_soc = _floor_soc(
                arrival_soc, soc_step, PHYSICS["g_kwh"]
            )
        else:
            charge_start_soc = arrival_soc
        if continuous_timing:
            time_window_valid = latest_departure >= arrival_min - 1e-9
            available_minutes = max(0.0, latest_departure - arrival_min)
            usable_blocks = None
        else:
            first_block = max(
                0,
                int(math.ceil(arrival_min / block_min - 1e-9)),
            )
            last_block = int(math.floor(
                latest_departure / block_min + 1e-9
            )) - 1
            usable_blocks = max(0, last_block - first_block + 1)
            available_minutes = usable_blocks * block_min
            time_window_valid = usable_blocks > 0
        if (
            charge_start_soc is None
            or arrival_soc < PHYSICS["reserve_kwh"] - 1e-9
            or not time_window_valid
        ):
            resulting = (
                arrival_soc - outbound.deadhead_kwh
                if charge_start_soc is None else
                charge_start_soc - outbound.deadhead_kwh
            )
            margin = resulting - required
            options.append({
                "option_kind": "station",
                "station": station,
                "feasible": False,
                "start_soc_after_trip_kwh": start_soc,
                "station_arrival_soc_kwh": arrival_soc,
                "available_minutes": available_minutes,
                "usable_blocks": usable_blocks,
                "charge_gain_kwh": 0.0,
                "outgoing_deadhead_kwh": outbound.deadhead_kwh,
                "resulting_soc_kwh": resulting,
                "successor_energy_plus_reserve_kwh": required,
                "binding_margin_kwh": margin,
                "binding_inequality":
                    f"{resulting:.9f} >= {required:.9f}",
                "arrival_min": arrival_min,
                "latest_departure_min": latest_departure,
                "rejection": (
                    "arrival_soc_below_reserve"
                    if arrival_soc < PHYSICS["reserve_kwh"] - 1e-9
                    else "no_timing_window"
                ),
            })
            continue
        charge_cap = (
            available_minutes * PHYSICS["charge_kw"] / 60.0
        )
        if production_flooring:
            block_kwh = PHYSICS["charge_kw"] * block_min / 60.0
            charged_soc = charge_start_soc
            if continuous_timing:
                charged_soc = min(
                    PHYSICS["g_kwh"], charged_soc + charge_cap
                )
                _charge_level, charged_soc = _floor_soc(
                    charged_soc, soc_step, PHYSICS["g_kwh"]
                )
            else:
                for _index in range(usable_blocks):
                    before = charged_soc
                    target = min(
                        PHYSICS["g_kwh"], before + block_kwh
                    )
                    _charge_level, charged_soc = _floor_soc(
                        target, soc_step, PHYSICS["g_kwh"]
                    )
                    if charged_soc <= before + 1e-9:
                        break
        else:
            charged_soc = min(
                PHYSICS["g_kwh"], charge_start_soc + charge_cap
            )
        gain = charged_soc - charge_start_soc
        resulting = charged_soc - outbound.deadhead_kwh
        if production_flooring:
            _result_level, resulting = _floor_soc(
                resulting, soc_step, PHYSICS["g_kwh"]
            )
        margin = resulting - required
        options.append({
            "option_kind": "station",
            "station": station,
            "feasible": (
                arrival_soc >= PHYSICS["reserve_kwh"] - 1e-9
                and time_window_valid
                and resulting >= required - 1e-9
            ),
            "start_soc_after_trip_kwh": start_soc,
            "station_arrival_soc_kwh": arrival_soc,
            "available_minutes": available_minutes,
            "usable_blocks": usable_blocks,
            "charge_gain_kwh": gain,
            "outgoing_deadhead_kwh": outbound.deadhead_kwh,
            "resulting_soc_kwh": resulting,
            "successor_energy_plus_reserve_kwh": required,
            "binding_margin_kwh": margin,
            "binding_inequality":
                f"{resulting:.9f} >= {required:.9f}",
            "arrival_min": arrival_min,
            "latest_departure_min": latest_departure,
        })
    feasible = [option for option in options if option["feasible"]]
    witness = max(
        feasible or options,
        key=lambda option: (
            option["binding_margin_kwh"],
            option["option_kind"] == "direct",
            str(option.get("station") or ""),
        ),
        default=None,
    )
    return {
        "mode": mode,
        "certificate_scope": COUNTERFACTUAL_SCOPE,
        "counterfactual_prefix_policy": (
            "same_production_actions_timing_stations_and_grid_charge_gains;"
            "retain_soc_residuals_and_clip_at_unchanged_battery_cap"
        ),
        "feasible": bool(feasible),
        "witness": witness,
        "options": options,
    }


def no_floor_prefix_soc_after_trip(
    problem,
    trip_sequence,
    frontier_state,
):
    """Replay one production prefix without SOC flooring.

    Timing, station choices, block counts, and production grid charge gains
    remain fixed. Only discarded SOC residuals are retained; charging is
    clipped at the unchanged battery cap.
    """
    position = int(frontier_state["position"])
    actions = list(frontier_state["actions"])
    if len(actions) != position + 1:
        raise ValueError("frontier action prefix length mismatch")
    source = actions[0]
    if source.get("kind") != "source":
        raise ValueError("frontier prefix lacks source action")
    soc = PHYSICS["g_kwh"] - float(source["deadhead_kwh"])
    arcs = _arc_groups(problem)
    for index in range(position):
        soc -= float(problem.trip_energy[trip_sequence[index]])
        action = actions[index + 1]
        if action.get("kind") == "charge":
            station = action["station"]
            next_trip = trip_sequence[index + 1]
            inbound = arcs["trip_station"].get(
                trip_sequence[index], {}
            ).get(station)
            outbound = arcs["station_trip"].get(
                station, {}
            ).get(next_trip)
            if inbound is None or outbound is None:
                raise ValueError(
                    "frontier charge action lacks production station arcs"
                )
            soc -= float(inbound.deadhead_kwh)
            soc = min(
                PHYSICS["g_kwh"],
                soc + float(action["expanded_grid_kwh"]),
            )
            soc -= float(outbound.deadhead_kwh)
        else:
            soc -= float(action.get("deadhead_kwh", 0.0))
    soc -= float(problem.trip_energy[trip_sequence[position]])
    return soc


def _normalize_production_witness(row):
    if row is None:
        return None
    resulting = row.get("resulting_soc_kwh")
    successor_energy = row.get("successor_energy_kwh")
    reserve = row.get("reserve_kwh")
    required = (
        float(successor_energy) + float(reserve)
        if successor_energy is not None and reserve is not None
        else None
    )
    margin = (
        float(resulting) - required
        if resulting is not None and required is not None else None
    )
    return {
        "option_kind": row.get("option_kind"),
        "station": row.get("station"),
        "feasible": row.get("accepted") is True,
        "start_soc_after_trip_kwh": row.get("soc_after_trip_kwh"),
        "station_arrival_soc_kwh":
            row.get("station_arrival_soc_before_floor_kwh"),
        "available_minutes": row.get("usable_minutes"),
        "usable_blocks": row.get("usable_blocks"),
        "charge_gain_kwh":
            row.get("cumulative_grid_charge_gain_kwh", 0.0),
        "outgoing_deadhead_kwh": row.get("outgoing_deadhead_kwh"),
        "resulting_soc_kwh": resulting,
        "successor_energy_plus_reserve_kwh": required,
        "binding_margin_kwh": margin,
        "binding_inequality": (
            f"{float(resulting):.9f} >= {required:.9f}"
            if resulting is not None and required is not None else None
        ),
        "production_trace_row": row,
    }


def _classify_counterfactuals(by_mode, graph_consistent):
    production = by_mode[
        "production_block_timing_production_soc_flooring"
    ]["feasible"]
    timing = by_mode[
        "continuous_timing_production_soc_flooring"
    ]["feasible"]
    flooring = by_mode[
        "production_block_timing_without_soc_flooring"
    ]["feasible"]
    continuous = by_mode[
        "continuous_timing_continuous_soc"
    ]["feasible"]
    if not graph_consistent:
        return "graph/reference defect"
    if production:
        return "unresolved"
    if timing and not flooring:
        return "block alignment"
    if flooring and not timing:
        return "accumulated SOC flooring"
    if continuous and (not timing and not flooring):
        return "interaction"
    if continuous and timing and flooring:
        return "unresolved"
    if continuous and timing:
        return "block alignment"
    if continuous and flooring:
        return "accumulated SOC flooring"
    return "unresolved"


def oracle_spec(
    *,
    instance_path,
    expected_instance_sha256,
    duty_id,
    grids,
    cell_id,
    comparison_instance=None,
    comparison_instance_sha256=None,
):
    return {
        "cell_id": str(cell_id),
        "duty_id": str(duty_id),
        "instance_path": _repo_relative(instance_path),
        "instance_sha256": str(expected_instance_sha256),
        "grids": [
            {"soc_step": float(soc), "block_min": int(block)}
            for soc, block in grids
        ],
        "comparison_instance_path": (
            _repo_relative(comparison_instance)
            if comparison_instance is not None else None
        ),
        "comparison_instance_sha256": (
            str(comparison_instance_sha256)
            if comparison_instance_sha256 is not None else None
        ),
    }


def audit_duty(
    *,
    instance_path,
    expected_instance_sha256,
    duty_id,
    grids,
    cell_id,
    comparison_instance=None,
    comparison_instance_sha256=None,
):
    trusted_spec = oracle_spec(
        instance_path=instance_path,
        expected_instance_sha256=expected_instance_sha256,
        duty_id=duty_id,
        grids=grids,
        cell_id=cell_id,
        comparison_instance=comparison_instance,
        comparison_instance_sha256=comparison_instance_sha256,
    )
    loaded = _load_duty(
        instance_path, expected_instance_sha256, duty_id
    )
    prices = _prices()
    witness = _continuous_witness(loaded, prices)
    local_to_ordered = loaded["local_to_ordered"]
    route = loaded["route"]
    ordered_sequence = [
        local_to_ordered[trip] for trip in route["trips"]
    ]
    comparison = None
    if comparison_instance is not None:
        compared = _load_duty(
            comparison_instance,
            comparison_instance_sha256,
            duty_id,
        )
        comparison_ordered = [
            compared["local_to_ordered"][trip]
            for trip in compared["route"]["trips"]
        ]
        if comparison_ordered != ordered_sequence:
            raise ValueError(
                "comparison instance encodes a different ordered duty"
            )
        comparison = {
            "instance_path": _repo_relative(compared["instance_path"]),
            "instance_file_sha256": compared["identities"][
                "instance_file_sha256"
            ],
            "local_trip_sequence": compared["route"]["trips"],
            "ordered_trip_sequence": comparison_ordered,
            "ordered_trip_sequence_sha256":
                _sha_payload(comparison_ordered),
            "same_ordered_duty": True,
        }
    grid_results = []
    candidate_rows = []
    frontier_rows = []
    counterfactual_rows = []
    for soc_step, block_min in grids:
        result = optimize_fixed_duty(
            loaded["problem"],
            route["trips"],
            prices,
            **PHYSICS,
            soc_step=soc_step,
            block_min=block_min,
            tariff_id="historical_flat",
            tariff_sha256=FLAT_SHA256,
            instance_sha256=expected_instance_sha256,
            allow_diagnostic_grid=True,
            trace=True,
        )
        diagnostic = result.get("diagnostic_trace") or {}
        failed = diagnostic.get("failed_transition")
        local_transition = (
            (int(failed["trip"]), int(failed["successor"]))
            if failed is not None and failed.get("successor") is not None
            else None
        )
        ordered_transition = (
            (
                int(local_to_ordered[local_transition[0]]),
                int(local_to_ordered[local_transition[1]]),
            )
            if local_transition is not None else None
        )
        if not result["feasible"] and local_transition is None:
            raise ValueError("oracle infeasible result lacks failed transition")
        failed_position = (
            int(failed["position"]) if failed is not None else None
        )
        failed_candidates = [
            row for row in diagnostic.get("transition_candidates") or []
            if row.get("position") == failed_position
        ]
        failed_frontiers = [
            row for row in diagnostic.get("frontier_states") or []
            if row.get("position") == failed_position
        ]
        for row in failed_candidates:
            enriched = {
                **row,
                "cell_id": cell_id,
                "duty_id": str(duty_id),
                "soc_step": soc_step,
                "block_min": block_min,
                "failed_local_from_trip": local_transition[0],
                "failed_local_to_trip": local_transition[1],
                "failed_ordered_from_trip": ordered_transition[0],
                "failed_ordered_to_trip": ordered_transition[1],
                "predicates_json": json.dumps(
                    row.get("predicates") or {},
                    sort_keys=True, separators=(",", ":"),
                ),
                "failed_predicates_json": json.dumps(
                    row.get("failed_predicates") or [],
                    separators=(",", ":"),
                ),
            }
            candidate_rows.append(enriched)
        for row in failed_frontiers:
            frontier_rows.append({
                "cell_id": cell_id,
                "duty_id": str(duty_id),
                "soc_step": soc_step,
                "block_min": block_min,
                "position": row["position"],
                "local_trip": row["trip"],
                "ordered_trip": local_to_ordered[row["trip"]],
                "local_successor": row["successor"],
                "ordered_successor":
                    local_to_ordered[row["successor"]],
                "level": row["level"],
                "soc_kwh": row["soc_kwh"],
                "cost": row["cost"],
                "actions_json": json.dumps(
                    row["actions"], sort_keys=True, separators=(",", ":")
                ),
            })
        by_mode = {}
        if local_transition is not None:
            production_soc_values = {
                float(row["soc_after_trip_kwh"])
                for row in failed_candidates
                if row.get("soc_after_trip_kwh") is not None
            }
            if not production_soc_values:
                raise ValueError("failed transition has no predecessor SOC")
            production_soc = max(production_soc_values)
            no_floor_prefix_soc = max(
                no_floor_prefix_soc_after_trip(
                    loaded["problem"],
                    route["trips"],
                    row,
                )
                for row in failed_frontiers
            )
            for mode in MODES:
                if mode == MODES[0]:
                    production_row = max(
                        failed_candidates,
                        key=lambda row: (
                            float(
                                row.get(
                                    "resulting_soc_kwh",
                                    -math.inf,
                                )
                                if row.get("resulting_soc_kwh")
                                is not None else -math.inf
                            ),
                            str(row.get("station") or ""),
                        ),
                        default=None,
                    )
                    counterfactual = {
                        "mode": mode,
                        "certificate_scope": COUNTERFACTUAL_SCOPE,
                        "feasible": any(
                            row.get("accepted") is True
                            for row in failed_candidates
                        ),
                        "witness":
                            _normalize_production_witness(production_row),
                    }
                else:
                    counterfactual = evaluate_counterfactual_transition(
                        loaded["problem"],
                        loaded["arcs"],
                        trip=local_transition[0],
                        successor=local_transition[1],
                        soc_step=soc_step,
                        block_min=block_min,
                        production_soc_after_trip=production_soc,
                        no_floor_prefix_soc_after_trip=
                            no_floor_prefix_soc,
                        mode=mode,
                    )
                by_mode[mode] = counterfactual
                counterfactual_rows.append({
                    "cell_id": cell_id,
                    "duty_id": str(duty_id),
                    "soc_step": soc_step,
                    "block_min": block_min,
                    "failed_local_from_trip": local_transition[0],
                    "failed_local_to_trip": local_transition[1],
                    "failed_ordered_from_trip": ordered_transition[0],
                    "failed_ordered_to_trip": ordered_transition[1],
                    "mode": mode,
                    "certificate_scope": COUNTERFACTUAL_SCOPE,
                    "feasible": counterfactual["feasible"],
                    **{
                        key: (
                            (counterfactual.get("witness") or {}).get(key)
                        )
                        for key in (
                            "option_kind", "station",
                            "start_soc_after_trip_kwh",
                            "station_arrival_soc_kwh",
                            "available_minutes", "usable_blocks",
                            "charge_gain_kwh", "outgoing_deadhead_kwh",
                            "resulting_soc_kwh",
                            "successor_energy_plus_reserve_kwh",
                            "binding_margin_kwh", "binding_inequality",
                        )
                    },
                    "witness_json": json.dumps(
                        counterfactual.get("witness"),
                        sort_keys=True, separators=(",", ":"),
                    ),
                })
            graph_consistent = any(
                row.get("direct_arc_exists") is True
                or (
                    (row.get("predicates") or {}).get(
                        "trip_to_station_arc_exists"
                    ) is True
                    and (row.get("predicates") or {}).get(
                        "station_to_successor_arc_exists"
                    ) is True
                )
                for row in failed_candidates
            )
            classification = _classify_counterfactuals(
                by_mode, graph_consistent
            )
        else:
            classification = None
        grid_results.append({
            "soc_step": soc_step,
            "block_min": block_min,
            "feasible": result["feasible"],
            "certificate_certified":
                result.get("certificate", {}).get("certified") is True,
            "certificate_scope":
                result.get("certificate", {}).get("scope"),
            "reason": result.get("reason"),
            "failed_local_transition": local_transition,
            "failed_ordered_transition": ordered_transition,
            "frontier_state_count": len(failed_frontiers),
            "transition_candidate_count": len(failed_candidates),
            "cause_classification": classification,
            "counterfactuals": by_mode,
        })
    return {
        "schema": SCHEMA,
        "oracle_spec": trusted_spec,
        "oracle_spec_sha256": _sha_payload(trusted_spec),
        "diagnostic_only": True,
        "certificate_scope": COUNTERFACTUAL_SCOPE,
        "cell_id": cell_id,
        "duty_id": str(duty_id),
        "instance_path": _repo_relative(loaded["instance_path"]),
        "instance_identity": loaded["identities"],
        "local_trip_sequence": route["trips"],
        "ordered_trip_sequence": ordered_sequence,
        "ordered_trip_sequence_sha256": _sha_payload(ordered_sequence),
        "comparison_instance": comparison,
        "physics": PHYSICS,
        "tariff_sha256": FLAT_SHA256,
        "producer_code_hashes": _producer_hashes(),
        "input_hashes": _input_hashes(),
        "continuous_witness": witness,
        "grid_results": grid_results,
    }, candidate_rows, frontier_rows, counterfactual_rows


def _readme_text(payload, oracle_sha, candidates_sha, frontier_sha, counter_sha):
    result_lines = "\n".join(
        (
            "- "
            f"{row['soc_step']:g} kWh/{row['block_min']} min: "
            f"local {row['failed_local_transition'][0]}→"
            f"{row['failed_local_transition'][1]}, ordered "
            f"{row['failed_ordered_transition'][0]}→"
            f"{row['failed_ordered_transition'][1]}, cause "
            f"`{row['cause_classification']}`"
        )
        if row["failed_local_transition"] is not None
        else (
            "- "
            f"{row['soc_step']:g} kWh/{row['block_min']} min: "
            "representable; no failed transition"
        )
        for row in payload["grid_results"]
    )
    return (
        "# Duty 13411 grid-transition oracle\n\n"
        "Read-only post-hoc current-code diagnostic. Counterfactuals are "
        "not feasibility or pricing certificates and do not change "
        "production physics or the running ladder.\n\n"
        "No-floor counterfactuals replay the same production prefix timing, "
        "stations, actions, and grid charge gains while retaining SOC "
        "residuals; they do not substitute the separately optimized "
        "continuous witness state.\n\n"
        "## Grid outcomes\n\n"
        f"{result_lines}\n\n"
        "## Artifact hashes\n\n"
        f"- `oracle.json`: `{oracle_sha}`\n"
        f"- `transition_candidates.csv`: `{candidates_sha}`\n"
        f"- `frontier_states.csv`: `{frontier_sha}`\n"
        f"- `counterfactuals.csv`: `{counter_sha}`\n"
    )


def publish(
    output_dir,
    *,
    instance_path,
    expected_instance_sha256,
    duty_id,
    grids,
    cell_id,
    comparison_instance=None,
    comparison_instance_sha256=None,
):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    payload, candidates, frontiers, counterfactuals = audit_duty(
        instance_path=instance_path,
        expected_instance_sha256=expected_instance_sha256,
        duty_id=duty_id,
        grids=grids,
        cell_id=cell_id,
        comparison_instance=comparison_instance,
        comparison_instance_sha256=comparison_instance_sha256,
    )
    staging = Path(tempfile.mkdtemp(
        dir=output_dir.parent, prefix=f".{output_dir.name}.tmp."
    ))
    try:
        oracle_path = staging / "oracle.json"
        candidates_path = staging / "transition_candidates.csv"
        frontier_path = staging / "frontier_states.csv"
        counterfactual_path = staging / "counterfactuals.csv"
        _write_csv(candidates_path, TRANSITION_FIELDS, candidates)
        _write_csv(frontier_path, FRONTIER_FIELDS, frontiers)
        _write_csv(
            counterfactual_path,
            COUNTERFACTUAL_FIELDS,
            counterfactuals,
        )
        payload["table_sha256"] = {
            candidates_path.name: sha256_file(candidates_path),
            frontier_path.name: sha256_file(frontier_path),
            counterfactual_path.name: sha256_file(counterfactual_path),
        }
        payload["table_row_count"] = {
            candidates_path.name: len(candidates),
            frontier_path.name: len(frontiers),
            counterfactual_path.name: len(counterfactuals),
        }
        _write_json(oracle_path, payload)
        readme = staging / "README.md"
        readme.write_text(_readme_text(
            payload,
            sha256_file(oracle_path),
            sha256_file(candidates_path),
            sha256_file(frontier_path),
            sha256_file(counterfactual_path),
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


def validate(output_dir, trusted_spec=None):
    root = Path(output_dir).resolve()
    oracle = root / "oracle.json"
    tables = {
        "transition_candidates.csv": root / "transition_candidates.csv",
        "frontier_states.csv": root / "frontier_states.csv",
        "counterfactuals.csv": root / "counterfactuals.csv",
    }
    readme = root / "README.md"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (oracle, readme, *tables.values())
    ):
        raise ValueError("oracle artifact set is incomplete")
    payload = json.loads(oracle.read_text())
    if trusted_spec is None:
        if root == DEFAULT_OUTPUT.resolve():
            trusted_spec = DEFAULT_ORACLE_SPEC
        else:
            raise ValueError(
                "oracle validation requires a trusted external specification"
            )
    trusted_spec = json.loads(json.dumps(trusted_spec))
    if (
        payload.get("oracle_spec") != trusted_spec
        or payload.get("oracle_spec_sha256") != _sha_payload(trusted_spec)
    ):
        raise ValueError("oracle trusted specification mismatch")
    grids = [
        (float(row["soc_step"]), int(row["block_min"]))
        for row in trusted_spec["grids"]
    ]
    expected, candidates, frontiers, counterfactuals = audit_duty(
        instance_path=REPO_ROOT / trusted_spec["instance_path"],
        expected_instance_sha256=trusted_spec["instance_sha256"],
        duty_id=trusted_spec["duty_id"],
        grids=grids,
        cell_id=trusted_spec["cell_id"],
        comparison_instance=(
            REPO_ROOT / trusted_spec["comparison_instance_path"]
            if trusted_spec.get("comparison_instance_path") else None
        ),
        comparison_instance_sha256=
            trusted_spec.get("comparison_instance_sha256"),
    )
    expected_table_bytes = {
        "transition_candidates.csv":
            _csv_bytes(TRANSITION_FIELDS, candidates),
        "frontier_states.csv": _csv_bytes(FRONTIER_FIELDS, frontiers),
        "counterfactuals.csv":
            _csv_bytes(COUNTERFACTUAL_FIELDS, counterfactuals),
    }
    for name, expected_bytes in expected_table_bytes.items():
        if tables[name].read_bytes() != expected_bytes:
            raise ValueError(f"oracle deterministic table mismatch: {name}")
    expected["table_sha256"] = {
        name: hashlib.sha256(value).hexdigest()
        for name, value in expected_table_bytes.items()
    }
    expected["table_row_count"] = {
        "transition_candidates.csv": len(candidates),
        "frontier_states.csv": len(frontiers),
        "counterfactuals.csv": len(counterfactuals),
    }
    expected_oracle = (
        json.dumps(
            expected, indent=2, sort_keys=True, allow_nan=False
        ).encode() + b"\n"
    )
    if oracle.read_bytes() != expected_oracle:
        raise ValueError("oracle summary semantics mismatch")
    expected_readme = _readme_text(
        expected,
        hashlib.sha256(expected_oracle).hexdigest(),
        expected["table_sha256"]["transition_candidates.csv"],
        expected["table_sha256"]["frontier_states.csv"],
        expected["table_sha256"]["counterfactuals.csv"],
    ).encode()
    if readme.read_bytes() != expected_readme:
        raise ValueError("oracle README semantics mismatch")
    return expected


def _grid(value):
    matched = re.fullmatch(
        r"(?P<soc>[0-9]+(?:\.[0-9]+)?)/(?P<block>[0-9]+)",
        value,
    )
    if matched is None:
        raise argparse.ArgumentTypeError("grid must be SOC_STEP/BLOCK_MIN")
    return float(matched.group("soc")), int(matched.group("block"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--instance-sha256", required=True)
    parser.add_argument("--duty-id", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--grid", action="append", type=_grid, required=True)
    parser.add_argument("--comparison-instance", type=Path)
    parser.add_argument("--comparison-instance-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    trusted_spec = oracle_spec(
        instance_path=args.instance,
        expected_instance_sha256=args.instance_sha256,
        duty_id=args.duty_id,
        grids=args.grid,
        cell_id=args.cell_id,
        comparison_instance=args.comparison_instance,
        comparison_instance_sha256=args.comparison_instance_sha256,
    )
    if args.validate_only:
        output = args.out.resolve()
        validate(output, trusted_spec)
    else:
        output = publish(
            args.out,
            instance_path=args.instance,
            expected_instance_sha256=args.instance_sha256,
            duty_id=args.duty_id,
            grids=args.grid,
            cell_id=args.cell_id,
            comparison_instance=args.comparison_instance,
            comparison_instance_sha256=args.comparison_instance_sha256,
        )
        validate(output, trusted_spec)
    print(json.dumps({
        "output": str(output),
        "oracle_sha256": sha256_file(output / "oracle.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

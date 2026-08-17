"""Deterministic continuous realization of an expanded-network route path.

The expanded network uses conservative SOC flooring.  Its stored charging kWh
is the grid-level gain, which can overfill a continuous replay that retains the
discarded sub-grid SOC residual.  This module preserves route nodes, station
choices, and charging-block windows while charging only enough in each block
to reach the expanded path's post-block grid SOC target.

The master-column cost is intentionally not changed here.  A realized charging
cost is reported separately because the existing reduced-cost certificate is
for the conservative expanded-grid cost space, not the continuous-cost space.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

from config import (
    BUS_COST_KX,
    CHARGE_START_COST,
    charge_cost_premium,
)
from utils_v2 import base_station_name


REALIZATION_SCHEMA = "evsp-dr-expanded-path-continuous-realization-v1"
BLOCK_SCHEDULE_SCHEMA = (
    "evsp-dr-continuous-realized-charging-blocks-v1"
)
TOLERANCE = 1e-6


def _canonical_sha(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def charging_block_schedule_sha256(blocks: list[dict]) -> str:
    return _canonical_sha(blocks)


def _arc_map(problem):
    return {
        (source, target): (float(travel), float(deadhead))
        for source, arcs in problem.adjacency.items()
        for target, travel, deadhead, _kind in arcs
    }


def _floor_level(soc: float, *, soc_step: float, levels: int) -> int:
    return min(
        max(int(math.floor((soc + 1e-9) / soc_step)), 0),
        levels - 1,
    )


def realize_expanded_path(
    problem,
    record: dict,
    *,
    g_kwh: float,
    charge_kw: float,
    reserve_kwh: float,
    soc_step: float,
    block_min: int,
    arc_map=None,
) -> tuple[dict | None, dict]:
    """Return a replay-valid schedule mapping or a classified rejection.

    No trip, station, route-node, cst, or cet value is changed.  Only per-stop
    kWh is reduced, block by block, to hit the expanded path's grid target.
    """

    try:
        g_kwh = float(g_kwh)
        charge_kw = float(charge_kw)
        reserve_kwh = float(reserve_kwh)
        soc_step = float(soc_step)
        block_min = int(block_min)
    except (TypeError, ValueError) as exc:
        return None, {
            "classification": "infeasible_after_realization",
            "reason": f"invalid physics: {exc}",
        }
    if (
        not all(math.isfinite(value) for value in (
            g_kwh, charge_kw, reserve_kwh, soc_step
        ))
        or g_kwh <= 0
        or charge_kw <= 0
        or soc_step <= 0
        or block_min <= 0
        or not 0 <= reserve_kwh <= g_kwh
    ):
        return None, {
            "classification": "infeasible_after_realization",
            "reason": "invalid/non-finite physics",
        }
    levels = int(g_kwh / soc_step) + 1
    grid = [round(index * soc_step, 6) for index in range(levels)]
    block_kwh = charge_kw * block_min / 60.0
    nodes = list(record.get("route_nodes", record.get("route", [])) or [])
    trips = list(record.get("trips") or [])
    node_trips = [
        node for node in nodes
        if isinstance(node, int) and not isinstance(node, bool)
    ]
    if node_trips != trips:
        return None, {
            "classification": "infeasible_after_realization",
            "reason": "trip sequence differs from route nodes",
        }
    if len(nodes) < 3:
        return None, {
            "classification": "infeasible_after_realization",
            "reason": "route nodes missing",
        }
    emitted_stops = deepcopy(record.get("charging_stops") or {})
    stops = deepcopy(
        record.get("expanded_grid_charging_stops")
        or emitted_stops
    )
    if record.get("expanded_grid_charging_stops") is not None:
        for key in ("stations", "cst", "cet"):
            if list(emitted_stops.get(key, [])) != list(
                stops.get(key, [])
            ):
                return None, {
                    "classification": "infeasible_after_realization",
                    "reason": (
                        f"emitted {key} differs from expanded-grid path"
                    ),
                }
    fields = {
        key: list(stops.get(key, []))
        for key in ("stations", "cst", "cet", "kwh")
    }
    if len({len(values) for values in fields.values()}) != 1:
        return None, {
            "classification": "infeasible_after_realization",
            "reason": "charging stop fields have different lengths",
        }
    arc = arc_map if arc_map is not None else _arc_map(problem)
    continuous_soc = g_kwh
    grid_soc = g_kwh
    stop_index = 0
    trace = []
    realized_kwh = []
    total_discarded = 0.0
    prev = nodes[0]
    for position, node in enumerate(nodes[1:], start=1):
        is_last = position == len(nodes) - 1
        key = None if (
            isinstance(prev, str) and isinstance(node, str) and prev == node
        ) else (prev, node)
        if key is not None and key not in arc:
            return None, {
                "classification": "infeasible_after_realization",
                "reason": f"missing model arc {prev}->{node}",
            }
        _travel, deadhead = arc.get(key, (0.0, 0.0))
        continuous_soc -= deadhead
        grid_soc -= deadhead
        if continuous_soc < reserve_kwh - TOLERANCE:
            return None, {
                "classification": "infeasible_after_realization",
                "reason": f"SOC below reserve before {node}",
            }
        if isinstance(node, int) and not isinstance(node, bool):
            expanded_before_floor = grid_soc
            level = _floor_level(
                grid_soc, soc_step=soc_step, levels=levels
            )
            floored = grid[level]
            discarded = max(0.0, expanded_before_floor - floored)
            total_discarded += discarded
            trace.append({
                "event": "soc_floor",
                "node": node,
                "continuous_soc_before_floor": continuous_soc,
                "expanded_soc_before_floor": expanded_before_floor,
                "expanded_grid_soc": floored,
                "discarded_residual_kwh": discarded,
                "continuous_minus_grid_kwh": (
                    continuous_soc - floored
                ),
            })
            grid_soc = floored
            energy = float(problem.trip_energy[node])
            if grid_soc - energy < reserve_kwh - TOLERANCE:
                return None, {
                    "classification": "infeasible_after_realization",
                    "reason": (
                        f"expanded grid SOC violates reserve after trip {node}"
                    ),
                }
            continuous_soc -= energy
            grid_soc -= energy
            if continuous_soc < reserve_kwh - TOLERANCE:
                return None, {
                    "classification": "infeasible_after_realization",
                    "reason": f"SOC below reserve after trip {node}",
                }
        elif not is_last:
            if (
                stop_index >= len(fields["stations"])
                or fields["stations"][stop_index] != node
            ):
                return None, {
                    "classification": "infeasible_after_realization",
                    "reason": f"charging stop alignment mismatch at {node}",
                }
            cst = float(fields["cst"][stop_index])
            cet = float(fields["cet"][stop_index])
            recorded = float(fields["kwh"][stop_index])
            duration = cet - cst
            block_count = int(round(duration / block_min))
            if (
                duration < -TOLERANCE
                or abs(duration - block_count * block_min) > TOLERANCE
                or abs(cst / block_min - round(cst / block_min)) > TOLERANCE
                or block_count <= 0
            ):
                return None, {
                    "classification": "infeasible_after_realization",
                    "reason": f"non-grid charging window at {node}",
                }
            expanded_before_floor = grid_soc
            entry_level = _floor_level(
                grid_soc, soc_step=soc_step, levels=levels
            )
            grid_soc = grid[entry_level]
            station_discarded = max(
                0.0, expanded_before_floor - grid_soc
            )
            total_discarded += station_discarded
            block_trace = []
            grid_gain = 0.0
            realized_gain = 0.0
            continuous_entry_soc = continuous_soc
            grid_entry_soc = grid_soc
            for block_offset in range(block_count):
                target_level = _floor_level(
                    min(g_kwh, grid_soc + block_kwh),
                    soc_step=soc_step,
                    levels=levels,
                )
                target_soc = grid[target_level]
                expanded_gain = max(0.0, target_soc - grid_soc)
                necessary = max(0.0, target_soc - continuous_soc)
                if necessary > block_kwh + TOLERANCE:
                    return None, {
                        "classification": "infeasible_after_realization",
                        "reason": f"required block energy exceeds charger at {node}",
                    }
                if continuous_soc + necessary > g_kwh + TOLERANCE:
                    return None, {
                        "classification": "infeasible_after_realization",
                        "reason": f"continuous target exceeds capacity at {node}",
                    }
                continuous_before = continuous_soc
                continuous_soc += necessary
                grid_soc = target_soc
                grid_gain += expanded_gain
                realized_gain += necessary
                block_trace.append({
                    "block_start_min": cst + block_offset * block_min,
                    "block_end_min": (
                        cst + (block_offset + 1) * block_min
                    ),
                    "continuous_soc_before": continuous_before,
                    "expanded_grid_soc_before": target_soc - expanded_gain,
                    "expanded_target_soc": target_soc,
                    "expanded_grid_gain_kwh": expanded_gain,
                    "realized_kwh": necessary,
                    "continuous_soc_after": continuous_soc,
                })
            if abs(grid_gain - recorded) > 1e-5:
                return None, {
                    "classification": "infeasible_after_realization",
                    "reason": (
                        f"recorded charge {recorded} differs from expanded "
                        f"block gain {grid_gain} at {node}"
                    ),
                }
            realized_kwh.append(round(realized_gain, 9))
            trace.append({
                "event": "charge_run",
                "station": node,
                "cst": cst,
                "cet": cet,
                "continuous_soc_entry": (
                    block_trace[0]["continuous_soc_before"]
                    if block_trace else continuous_soc
                ),
                "expanded_grid_soc_entry": (
                    block_trace[0]["expanded_grid_soc_before"]
                    if block_trace else grid_soc
                ),
                "recorded_grid_kwh": recorded,
                "station_entry_discarded_residual_kwh":
                    station_discarded,
                "continuous_minus_grid_at_entry_kwh":
                    continuous_entry_soc - grid_entry_soc,
                "realized_kwh": round(realized_gain, 9),
                "blocks": block_trace,
            })
            stop_index += 1
        prev = node
    if stop_index != len(fields["stations"]):
        return None, {
            "classification": "infeasible_after_realization",
            "reason": "unconsumed charging stops",
        }
    if continuous_soc < reserve_kwh - TOLERANCE:
        return None, {
            "classification": "infeasible_after_realization",
            "reason": "final SOC below reserve",
        }
    if grid_soc < reserve_kwh - TOLERANCE:
        return None, {
            "classification": "infeasible_after_realization",
            "reason": "expanded grid final SOC below reserve",
        }
    realized = deepcopy(record)
    realized_stops = deepcopy(emitted_stops)
    realized_stops["kwh"] = realized_kwh
    realized["charging_stops"] = realized_stops
    realized["expanded_grid_charging_stops"] = deepcopy(stops)
    changed = any(
        abs(float(before) - float(after)) > TOLERANCE
        for before, after in zip(fields["kwh"], realized_kwh)
    )
    mapping = {
        "schema": REALIZATION_SCHEMA,
        "trip_sequence_sha256": _canonical_sha(trips),
        "route_nodes_sha256": _canonical_sha(nodes),
        "recorded_charging_sha256": _canonical_sha(stops),
        "realized_charging_sha256": _canonical_sha(realized_stops),
        "station_choice_preserved": True,
        "charging_windows_preserved": True,
        "trip_sequence_preserved": True,
        "recorded_total_kwh": sum(float(value) for value in fields["kwh"]),
        "realized_total_kwh": sum(realized_kwh),
        "discarded_grid_residual_kwh": total_discarded,
        "changed": changed,
        "charge_kw": charge_kw,
        "block_min": block_min,
        "trace": trace,
        "pricing_cost_semantics": "expanded_grid_cost_unchanged",
        "continuous_cost_pricing_certified": False,
        "pricing_certificate_scope": "conservative_expanded_grid_model_only",
    }
    mapping["mapping_sha256"] = _canonical_sha({
        key: value for key, value in mapping.items()
        if key != "trace"
    })
    realized["continuous_realization"] = {
        key: value for key, value in mapping.items() if key != "trace"
    }
    return realized, {
        "classification": (
            "deterministically_repairable" if changed else "valid_as_recorded"
        ),
        "mapping": mapping,
    }


def _tariff_identity(station, minute, station_prices):
    base = base_station_name(station)
    curve = station_prices[base]
    requested_hour = int(float(minute) // 60)
    tariff_hour = (
        requested_hour if requested_hour in curve else max(curve)
    )
    base_price = float(curve[tariff_hour])
    return {
        "tariff_hour": tariff_hour,
        "tariff_key": f"{base}:{tariff_hour}",
        "price_per_kwh": base_price * charge_cost_premium,
    }


def blocks_from_continuous_stops(
    record: dict,
    *,
    station_prices: dict,
    charge_kw: float,
    earliest_start_by_stop: list[float] | None = None,
) -> list[dict]:
    """Split continuous stop kWh into tariff-hour, power-bounded blocks."""

    stops = record.get("charging_stops") or {}
    fields = {
        key: list(stops.get(key, []))
        for key in ("stations", "cst", "cet", "kwh")
    }
    if len({len(value) for value in fields.values()}) != 1:
        raise ValueError("charging stop fields have different lengths")
    blocks = []
    for stop_index, station in enumerate(fields["stations"]):
        start = float(fields["cst"][stop_index])
        end = float(fields["cet"][stop_index])
        remaining = float(fields["kwh"][stop_index])
        block_index = 0
        cursor = max(
            start,
            float(earliest_start_by_stop[stop_index])
            if earliest_start_by_stop is not None else start,
        )
        if (
            earliest_start_by_stop is not None
            and float(earliest_start_by_stop[stop_index])
            > start + TOLERANCE
            and remaining > TOLERANCE
        ):
            raise ValueError(
                "route arrives after priced charging start"
            )
        if cursor > end + TOLERANCE:
            raise ValueError("charging stop begins after its window")
        while remaining > TOLERANCE:
            hour_end = (int(cursor // 60) + 1) * 60.0
            segment_end = min(end, hour_end)
            capacity = max(0.0, segment_end - cursor) * charge_kw / 60.0
            if capacity <= TOLERANCE:
                raise ValueError("charging stop lacks time for recorded kWh")
            energy = min(remaining, capacity)
            actual_end = cursor + energy * 60.0 / charge_kw
            blocks.append({
                "stop_index": stop_index,
                "block_index": block_index,
                "station": station,
                "start_min": cursor,
                "end_min": actual_end,
                "realized_kwh": energy,
                "expanded_grid_kwh": energy,
                **_tariff_identity(station, cursor, station_prices),
            })
            remaining -= energy
            cursor = segment_end
            block_index += 1
        if remaining < -TOLERANCE:
            raise ValueError("charging block allocation exceeded stop kWh")
    validate_continuous_charging_blocks(
        record,
        blocks,
        station_prices=station_prices,
        charge_kw=charge_kw,
    )
    return blocks


def validate_continuous_charging_blocks(
    record: dict,
    blocks: list[dict],
    *,
    station_prices: dict,
    charge_kw: float,
    expected_continuous_cost: float | None = None,
) -> dict:
    """Validate compact block provenance and recompute continuous route cost."""

    stops = record.get("charging_stops") or {}
    fields = {
        key: list(stops.get(key, []))
        for key in ("stations", "cst", "cet", "kwh")
    }
    if len({len(value) for value in fields.values()}) != 1:
        raise ValueError("charging stop fields have different lengths")
    grouped = {index: [] for index in range(len(fields["stations"]))}
    previous_key = None
    previous_end = {}
    global_previous_end = None
    for block in blocks:
        stop_index = block.get("stop_index")
        block_index = block.get("block_index")
        if (
            not isinstance(stop_index, int)
            or not isinstance(block_index, int)
            or stop_index not in grouped
        ):
            raise ValueError("continuous charging block index is invalid")
        if block_index != len(grouped[stop_index]):
            raise ValueError(
                "continuous charging block indices are not contiguous"
            )
        key = (stop_index, block_index)
        if previous_key is not None and key <= previous_key:
            raise ValueError("continuous charging blocks are not ordered")
        previous_key = key
        station = fields["stations"][stop_index]
        start = float(block["start_min"])
        end = float(block["end_min"])
        realized = float(block["realized_kwh"])
        expanded = float(block["expanded_grid_kwh"])
        if block.get("station") != station:
            raise ValueError("continuous block station mismatch")
        if (
            start < float(fields["cst"][stop_index]) - TOLERANCE
            or end > float(fields["cet"][stop_index]) + TOLERANCE
            or end <= start
        ):
            raise ValueError("continuous block lies outside stop window")
        if (
            stop_index in previous_end
            and start < previous_end[stop_index] - TOLERANCE
        ):
            raise ValueError("continuous charging blocks overlap")
        previous_end[stop_index] = end
        if (
            global_previous_end is not None
            and start < global_previous_end - TOLERANCE
        ):
            raise ValueError("continuous charging blocks overlap across stops")
        global_previous_end = end
        capacity = (end - start) * float(charge_kw) / 60.0
        if (
            realized < -TOLERANCE
            or expanded < -TOLERANCE
            or realized > capacity + TOLERANCE
            or expanded > capacity + TOLERANCE
        ):
            raise ValueError("continuous block violates charger power")
        tariff = _tariff_identity(station, start, station_prices)
        if end > (int(start // 60) + 1) * 60.0 + TOLERANCE:
            raise ValueError("continuous block crosses tariff hour")
        for tariff_key, expected in tariff.items():
            observed = block.get(tariff_key)
            if isinstance(expected, float):
                if not math.isclose(
                    float(observed), expected,
                    rel_tol=1e-12, abs_tol=1e-12,
                ):
                    raise ValueError("continuous block tariff mismatch")
            elif observed != expected:
                raise ValueError("continuous block tariff identity mismatch")
        grouped[stop_index].append(block)
    realized_electricity = 0.0
    expanded_electricity = 0.0
    for stop_index, stop_blocks in grouped.items():
        if not stop_blocks and abs(float(fields["kwh"][stop_index])) > TOLERANCE:
            raise ValueError("aggregate stop has no continuous blocks")
        block_sum = sum(float(block["realized_kwh"]) for block in stop_blocks)
        if not math.isclose(
            block_sum, float(fields["kwh"][stop_index]),
            rel_tol=0.0, abs_tol=1e-6,
        ):
            raise ValueError("block kWh does not equal aggregate stop kWh")
        expanded_sum = sum(
            float(block["expanded_grid_kwh"])
            for block in stop_blocks
        )
        expanded_stops = (
            record.get("expanded_grid_charging_stops") or stops
        )
        expected_expanded = float(
            (expanded_stops.get("kwh") or [])[stop_index]
        )
        if not math.isclose(
            expanded_sum, expected_expanded,
            rel_tol=0.0, abs_tol=1e-6,
        ):
            raise ValueError(
                "block grid kWh does not equal expanded stop kWh"
            )
        realized_electricity += sum(
            float(block["realized_kwh"])
            * float(block["price_per_kwh"])
            for block in stop_blocks
        )
        expanded_electricity += sum(
            float(block["expanded_grid_kwh"])
            * float(block["price_per_kwh"])
            for block in stop_blocks
        )
    starts = len(fields["stations"])
    fixed = BUS_COST_KX + starts * CHARGE_START_COST
    continuous_cost = fixed + realized_electricity
    if (
        expected_continuous_cost is not None
        and not math.isclose(
            continuous_cost,
            float(expected_continuous_cost),
            rel_tol=1e-10,
            abs_tol=1e-6,
        )
    ):
        raise ValueError(
            "continuous charging blocks do not reproduce realized cost"
        )
    return {
        "continuous_realized_cost": continuous_cost,
        "recomputed_expanded_grid_cost": fixed + expanded_electricity,
        "realized_electricity_cost": realized_electricity,
        "expanded_grid_electricity_cost": expanded_electricity,
        "block_schedule_sha256":
            charging_block_schedule_sha256(blocks),
    }


def realized_costs(
    record: dict,
    mapping: dict,
    *,
    station_prices: dict,
) -> dict:
    """Report grid and realized charging costs without changing master cost."""

    blocks = []
    stop_index = 0
    for event in mapping.get("trace", []):
        if event.get("event") != "charge_run":
            continue
        station = event["station"]
        for block_index, block in enumerate(event["blocks"]):
            tariff = _tariff_identity(
                station, block["block_start_min"], station_prices
            )
            blocks.append({
                "stop_index": stop_index,
                "block_index": block_index,
                "station": station,
                "start_min": block["block_start_min"],
                "end_min": block["block_end_min"],
                "realized_kwh": block["realized_kwh"],
                "expanded_grid_kwh":
                    block["expanded_grid_gain_kwh"],
                **tariff,
            })
        stop_index += 1
    validation = validate_continuous_charging_blocks(
        record,
        blocks,
        station_prices=station_prices,
        charge_kw=float(mapping["charge_kw"]),
    )
    return {
        "stored_expanded_grid_cost": float(record["cost"]),
        "recomputed_expanded_grid_cost":
            validation["recomputed_expanded_grid_cost"],
        "continuous_realized_cost":
            validation["continuous_realized_cost"],
        "expanded_minus_realized_cost": (
            validation["expanded_grid_electricity_cost"]
            - validation["realized_electricity_cost"]
        ),
        "continuous_realized_charging_blocks": blocks,
        "continuous_realized_charging_blocks_sha256":
            validation["block_schedule_sha256"],
        "master_cost_changed": False,
        "master_cost_semantics": "expanded_grid_cost",
        "continuous_cost_pricing_certified": False,
    }

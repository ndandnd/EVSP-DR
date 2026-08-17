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
TOLERANCE = 1e-6


def _canonical_sha(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


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


def realized_costs(
    record: dict,
    mapping: dict,
    *,
    station_prices: dict,
) -> dict:
    """Report grid and realized charging costs without changing master cost."""

    grid_electricity = 0.0
    realized_electricity = 0.0
    for event in mapping.get("trace", []):
        if event.get("event") != "charge_run":
            continue
        station = event["station"]
        curve = station_prices[base_station_name(station)]
        max_hour = max(curve)
        for block in event["blocks"]:
            hour = min(int(block["block_start_min"] // 60), max_hour)
            price = float(curve[hour]) * charge_cost_premium
            grid_electricity += block["expanded_grid_gain_kwh"] * price
            realized_electricity += block["realized_kwh"] * price
    starts = len((record.get("charging_stops") or {}).get("stations", []))
    fixed = BUS_COST_KX + starts * CHARGE_START_COST
    return {
        "stored_expanded_grid_cost": float(record["cost"]),
        "recomputed_expanded_grid_cost": fixed + grid_electricity,
        "continuous_realized_cost": fixed + realized_electricity,
        "expanded_minus_realized_cost": (
            grid_electricity - realized_electricity
        ),
        "master_cost_changed": False,
        "master_cost_semantics": "expanded_grid_cost",
        "continuous_cost_pricing_certified": False,
    }

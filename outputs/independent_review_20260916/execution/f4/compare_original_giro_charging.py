#!/usr/bin/env python3
"""Audit and price unmodified original GIRO charging for whole-duty subsets.

An original recharge row supplies a station, time window and total energy, not
its power profile. Costs are exact only when every admissible profile has the
same cost. Otherwise we publish attainable per-window cost bounds, plus a
separately labelled uniform-power assumption. No charging window is repaired.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-6
STATION_NODES = {
    "2190": "2190L_0", "2190L": "2190L_0", "4808": "4808_0",
    "PARX": "PARX_1", "3127": "3127L_0", "3127L": "3127L_0",
    "7880": "7880C_0", "7880C": "7880C_0", "JON": "JON_A_0",
    "JON_A": "JON_A_0",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, label, *, blank_zero=False):
    if blank_zero and (value is None or str(value).strip() == ""):
        return 0.0
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {label}")
    return result


def trip_id(value):
    result = number(value, "trip ID")
    if not result.is_integer():
        raise ValueError("nonintegral source trip ID")
    return int(result)


def minutes(value):
    hour, minute = str(value).split(":")
    hour, minute = int(hour), int(minute)
    if hour < 0 or not 0 <= minute < 60:
        raise ValueError(f"invalid time: {value}")
    return hour * 60 + minute


def extract_original(master_rows, instance_rows):
    """Require complete source duties and retain original row order/windows."""
    ordered = [trip_id(row["Ordered_Trip_ID"]) for row in instance_rows]
    if len(set(ordered)) != len(ordered):
        raise ValueError("instance repeats a source trip")
    source_trip = {}
    for row in master_rows:
        if row["Identifier"] == "Regular":
            identity = trip_id(row["Ordered_Trip_ID"])
            if identity in source_trip:
                raise ValueError("master repeats a source trip")
            source_trip[identity] = row
    if any(identity not in source_trip for identity in ordered):
        raise ValueError("instance contains trips absent from GIRO master")
    for row, identity in zip(instance_rows, ordered):
        original = source_trip[identity]
        if any(row[key] != original[key] for key in ("From1", "Start1", "End1", "To1")) or not math.isclose(
                number(row["Usage kWh"], "instance trip energy"),
                number(original["Usage kWh"], "source trip energy"), rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError(f"instance trip {identity} differs from original GIRO service")
    duties = sorted({source_trip[identity]["VehicleTask"] for identity in ordered})
    local = {identity: index for index, identity in enumerate(ordered)}
    result = []
    for duty in duties:
        rows = [(index + 2, row) for index, row in enumerate(master_rows)
                if row["VehicleTask"] == duty]
        full_ids = [trip_id(row["Ordered_Trip_ID"]) for _, row in rows
                    if row["Identifier"] == "Regular"]
        if any(identity not in local for identity in full_ids):
            raise ValueError(f"instance contains only part of duty {duty}")
        nodes, trips, events = ["PARX_0"], [], []
        for line, row in rows:
            if row["Identifier"] == "Regular":
                trip = local[trip_id(row["Ordered_Trip_ID"])]
                nodes.append(trip)
                trips.append(trip)
            elif row["Identifier"] == "Recharge":
                if row["From1"] != row["To1"]:
                    raise ValueError(f"moving recharge at master row {line}")
                station = STATION_NODES.get(row["From1"])
                if station is None:
                    raise ValueError(f"unknown station at master row {line}")
                start, end = minutes(row["Start1"]), minutes(row["End1"])
                kwh = number(row["Recharge kWh"], "recorded recharge")
                if kwh <= 0 or end <= start:
                    raise ValueError(f"nonpositive recharge/window at row {line}")
                nodes.append(station)
                events.append({"duty_id": duty, "source_row": line,
                               "station": station, "start_min": start,
                               "end_min": end, "kwh": kwh})
        nodes.append("PARX_0")
        result.append({
            "duty_id": duty, "trips": trips, "route_nodes": nodes,
            "source_ordered_trip_ids": full_ids, "events": events,
            "source_activities": [{"source_row": line, **row} for line, row in rows],
            "charging_stops": {
                "stations": [event["station"] for event in events],
                "cst": [event["start_min"] for event in events],
                "cet": [event["end_min"] for event in events],
                "kwh": [event["kwh"] for event in events],
            },
        })
    if sorted(trip for route in result for trip in route["trips"]) != list(range(len(ordered))):
        raise ValueError("original duties are not an exact instance partition")
    return result


def recorded_activity_check(rows, *, g_kwh, charge_kw, reserve_kwh):
    """Replay recorded energy without clipping over-capacity recharge away."""
    soc = minimum = g_kwh
    violations = []
    previous_end = None
    for row in rows:
        start, end = minutes(row["Start1"]), minutes(row["End1"])
        label = f"row {row.get('source_row', '?')}"
        if end < start or (previous_end is not None and start < previous_end):
            violations.append(f"{label}: overlapping or reversed activity")
        previous_end = end
        if row["Identifier"] == "Recharge":
            energy = number(row["Recharge kWh"], "recharge")
            if energy < 0 or energy > max(0, end - start) * charge_kw / 60 + TOL:
                violations.append(f"{label}: recharge exceeds power/window")
            soc += energy
            if soc > g_kwh + TOL:
                violations.append(f"{label}: SOC {soc:.9g} exceeds capacity")
        else:
            energy = number(row.get("Usage kWh"), "activity energy", blank_zero=True)
            if energy < 0:
                violations.append(f"{label}: negative activity energy")
            soc -= energy
        minimum = min(minimum, soc)
        if soc < reserve_kwh - TOL:
            violations.append(f"{label}: SOC {soc:.9g} below reserve")
    return {"valid": not violations, "violations": violations,
            "terminal_soc_kwh": soc, "minimum_soc_kwh": minimum}


def window_cost_bounds(start, end, kwh, curve, charge_kw):
    """Exact min/max invoice over all rate-bounded profiles in one window."""
    if not all(math.isfinite(float(x)) for x in (start, end, kwh, charge_kw)):
        raise ValueError("nonfinite charging window")
    if end <= start or kwh < 0 or charge_kw <= 0:
        raise ValueError("invalid charging window")
    segments = []
    cursor = float(start)
    while cursor < end - 1e-9:
        hour = int(cursor // 60)
        stop = min(end, (hour + 1) * 60)
        if hour not in curve or not math.isfinite(float(curve[hour])):
            raise ValueError(f"tariff missing/nonfinite hour {hour}")
        segments.append((float(curve[hour]), (stop - cursor) * charge_kw / 60,
                         stop - cursor))
        cursor = stop
    if kwh > sum(capacity for _, capacity, _ in segments) + TOL:
        raise ValueError("recorded energy exceeds vehicle charging-power/window capacity")

    def extreme(reverse):
        remaining, cost = kwh, 0.0
        for price, capacity, _ in sorted(segments, reverse=reverse):
            energy = min(remaining, capacity)
            cost += price * energy
            remaining -= energy
        return cost

    lower, upper = extreme(False), extreme(True)
    return {"energy_cost_lower": lower, "energy_cost_upper": upper,
            "energy_cost_exact": (lower if abs(upper - lower) <= 1e-8 else None),
            "uniform_power_assumption_energy_cost": sum(
                price * kwh * duration / (end - start)
                for price, _, duration in segments)}


def run(args):
    from audit_giro_known_columns import HORIZON_MIN, build_problem
    from config import CHARGING_STATIONS
    from run_exact_pool_mip import validate_injected_route
    from utils_v2 import base_station_name, load_station_hourly_prices

    if args.out.exists():
        raise FileExistsError(args.out)
    if sha(args.instance) != args.instance_sha256 or sha(args.master) != args.master_sha256:
        raise ValueError("instance/master SHA-256 mismatch")
    if not 0 <= args.reserve_kwh <= args.g_kwh or min(args.g_kwh, args.charge_kw) <= 0:
        raise ValueError("invalid physics")
    if not all(math.isfinite(value) for value in (
            args.g_kwh, args.charge_kw, args.reserve_kwh, args.charge_start_cost)) or args.charge_start_cost < 0:
        raise ValueError("nonfinite physics or invalid start fee")
    references = {name: sha(args.reference_data_dir / name)
                  for name in ("Ref_dict.csv", "par_ref_dhd.csv")}
    tariff_hashes = {str(path): sha(path) for path in args.tariffs}
    routes = extract_original(read_rows(args.master), read_rows(args.instance))
    if len(routes) != args.fleet:
        raise ValueError(f"expected {args.fleet} whole original duties, found {len(routes)}")
    problem = build_problem(args.instance.parent, args.instance.name,
                            max_station_to_trip_wait_min=HORIZON_MIN,
                            reference_data_dir=args.reference_data_dir)
    arc_energy = {(u, v): energy for u, arcs in problem.adjacency.items()
                  for v, _travel, energy, _kind in arcs}
    for route in routes:
        source_check = recorded_activity_check(route["source_activities"],
            g_kwh=args.g_kwh, charge_kw=args.charge_kw, reserve_kwh=args.reserve_kwh)
        reason = validate_injected_route(problem, route, args.g_kwh, args.charge_kw,
                                        args.reserve_kwh, HORIZON_MIN,
                                        arrival_grace_min=0.0, rate_grace_min=0.0)
        terminal = None
        if reason is None:
            nodes = route["route_nodes"]
            deadhead = sum(arc_energy[(u, v)] for u, v in zip(nodes, nodes[1:]) if u != v)
            terminal = args.g_kwh + sum(e["kwh"] for e in route["events"]) - deadhead - sum(
                problem.trip_energy[trip] for trip in route["trips"])
        route["recorded_activity_check"] = source_check
        route["production_model_check"] = {
            "valid": reason is None, "reason": reason,
            "terminal_soc_kwh": terminal,
            "scope": "unmodified_service_and_charge_plan_with_production_deadhead_model",
            "arrival_grace_min": 0.0, "rate_grace_min": 0.0,
        }
        route["matched_physics_comparator_eligible"] = source_check["valid"] and reason is None
        del route["source_activities"]
    all_physical = all(route["matched_physics_comparator_eligible"] for route in routes)
    terminals = [route["production_model_check"]["terminal_soc_kwh"] for route in routes]
    surplus = (sum(value - args.reserve_kwh for value in terminals)
               if all(value is not None for value in terminals) else None)
    summaries, events = [], []
    for tariff in args.tariffs:
        prices = load_station_hourly_prices(tariff, CHARGING_STATIONS)
        tariff_events, cost_errors = [], []
        for route in routes:
            for event in route["events"]:
                try:
                    bounds = window_cost_bounds(event["start_min"], event["end_min"],
                        event["kwh"], prices[base_station_name(event["station"])], args.charge_kw)
                except ValueError as exc:
                    bounds = {"unavailable_reason": str(exc)}
                    cost_errors.append(str(exc))
                tariff_events.append({**event, **bounds, "tariff": str(tariff),
                                      "tariff_sha256": tariff_hashes[str(tariff)]})
        events.extend(tariff_events)
        lower = upper = uniform = exact = None
        fees = args.charge_start_cost * len(tariff_events)
        if not cost_errors:
            lower = sum(event["energy_cost_lower"] for event in tariff_events)
            upper = sum(event["energy_cost_upper"] for event in tariff_events)
            uniform = sum(event["uniform_power_assumption_energy_cost"] for event in tariff_events)
            if abs(upper - lower) <= 1e-8:
                exact = lower
        summaries.append({
            "comparator": "ORIGINAL_GIRO_UNMODIFIED", "tariff": str(tariff),
            "tariff_sha256": tariff_hashes[str(tariff)], "fleet": len(routes),
            "matched_physics_comparator_eligible": all_physical,
            "energy_cost_exact": exact, "energy_cost_lower": lower, "energy_cost_upper": upper,
            "charge_start_fees": fees, "charge_events": len(tariff_events),
            "charging_cost_exact": None if exact is None else exact + fees,
            "charging_cost_lower": None if lower is None else lower + fees,
            "charging_cost_upper": None if upper is None else upper + fees,
            "uniform_power_assumption_energy_cost": uniform,
            "uniform_power_assumption_is_observed": False,
            "total_charged_kwh": sum(event["kwh"] for event in tariff_events),
            "terminal_surplus_total_kwh": surplus,
            "production_energy_consumed_kwh": (args.g_kwh * len(routes) + sum(event["kwh"] for event in tariff_events) - surplus - args.reserve_kwh * len(routes) if surplus is not None else None),
            "charge_start_fee_scope": "modeled_per_start_penalty_not_verified_operator_invoice",
            "terminal_energy_credit_price": args.terminal_energy_price,
            "terminal_energy_normalized_charging_cost_lower": (
                lower + fees - args.terminal_energy_price * surplus
                if lower is not None and surplus is not None and args.terminal_energy_price is not None else None),
            "terminal_energy_normalized_charging_cost_upper": (
                upper + fees - args.terminal_energy_price * surplus
                if upper is not None and surplus is not None and args.terminal_energy_price is not None else None),
            "terminal_energy_credit_scope": "accounting_sensitivity_not_observed_invoice_or_equal_terminal_soc_constraint",
            "cost_errors": sorted(set(cost_errors)),
            "robust_improvement_rule": "eligible AND optimized_same_fleet_cost < original_charging_cost_lower",
        })
    if sha(args.instance) != args.instance_sha256 or sha(args.master) != args.master_sha256:
        raise ValueError("source changed during audit")
    if any(sha(args.reference_data_dir / name) != digest for name, digest in references.items()):
        raise ValueError("reference data changed during audit")
    if any(sha(path) != tariff_hashes[str(path)] for path in args.tariffs):
        raise ValueError("tariff changed during audit")
    payload = {"schema": "evsp-dr-selected-original-giro-charging-v1",
        "instance": str(args.instance), "instance_sha256": args.instance_sha256,
        "master": str(args.master), "master_sha256": args.master_sha256,
        "reference_sha256": references, "implementation_sha256": sha(Path(__file__)),
        "physics": {"g_kwh": args.g_kwh, "charge_kw": args.charge_kw,
            "reserve_kwh": args.reserve_kwh, "initial_soc_kwh": args.g_kwh,
            "imposed_terminal_soc_policy": "depot_arrival_soc_at_least_reserve",
            "recorded_operator_terminal_soc_policy": "unavailable",
            "charge_start_cost": args.charge_start_cost,
            "charge_kw_scope": "per_vehicle_charging_power_limit; shared_station_capacity_not_modeled"},
        "fleet": len(routes), "routes": routes, "summary": summaries, "charge_events": events,
        "recorded_power_profile_available": False,
        "physical_scope": "existence_of_rate_bounded_profile_in_unmodified_recorded_windows; not_observed_power_trace",
        "terminal_energy_comparability": "same_initial_energy_and_terminal_constraint; actual_terminal_soc_must_be_reported",
        "not_event_grid_representability_certificate": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"out": str(args.out), "fleet": len(routes),
                      "matched_physics_comparator_eligible": all_physical}))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--instance-sha256", required=True)
    parser.add_argument("--master", type=Path, default=ROOT / "data/Par_VehicleDetails_Updated.csv")
    parser.add_argument("--master-sha256", required=True)
    parser.add_argument("--reference-data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--tariffs", type=Path, nargs="+", required=True)
    parser.add_argument("--fleet", type=int, required=True)
    parser.add_argument("--g-kwh", type=float, default=240.0)
    parser.add_argument("--charge-kw", type=float, default=240.0)
    parser.add_argument("--reserve-kwh", type=float, default=0.0)
    parser.add_argument("--charge-start-cost", type=float, default=5.0)
    parser.add_argument("--terminal-energy-price", type=float, default=None,
                        help="Optional common salvage value per kWh for a labelled sensitivity.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.terminal_energy_price is not None and not math.isfinite(args.terminal_energy_price):
        parser.error("terminal energy price must be finite")
    for name in ("instance", "master", "reference_data_dir", "out"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.tariffs = [path.expanduser().resolve() for path in args.tariffs]
    run(args)


if __name__ == "__main__":
    main()

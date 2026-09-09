#!/usr/bin/env python3
"""Audit Partille GIRO duties under documented single-vehicle physics.

For every literal duty variant, the audit performs two distinct checks:

1. fixed-duty feasibility: can the documented ordered trip sequence be replayed;
2. unrestricted pricing recovery: on that duty's own trip set, what is the
   maximum-cardinality route found without fixing the GIRO sequence?

The second check is a longest-path resource DP, not column generation.  It is
the intended pricing-oracle gate before spending cluster time on k=2 CG.  A
successful k=1 recovery means the pricing network can express the duty; it does
not certify multi-route charger/platform feasibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from audit_giro_known_columns import ProblemData, build_problem
from giro_partille_physics import (
    VehicleProfile,
    base_site,
    charge_window,
    documented_single_vehicle_scope,
    profile_for_duty,
)
from make_duty_pair_instances import (
    INSTANCE_COLUMNS,
    _base_task,
    merge_duties,
)


SCHEMA = "evsp-dr-giro-duty-recovery-v2"
PAIR_SCHEMA = "evsp-dr-giro-k2-recovery-plan-v2"
DEFAULT_HORIZON_MIN = 1620.0
TOL = 1e-8


@dataclass(frozen=True)
class Arc:
    successor: object
    travel_min: float
    energy_kwh: float
    kind: str


@dataclass(frozen=True)
class Label:
    trip: int
    trip_count: int
    entry_soc_kwh: float
    trips: tuple[int, ...]
    actions: tuple[dict, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arc_groups(problem: ProblemData) -> dict:
    groups = {
        "depot_trip": {}, "trip_depot": {}, "trip_trip": {},
        "trip_station": {}, "station_trip": {}, "station_depot": {},
    }
    for source, entries in problem.adjacency.items():
        for successor, travel, energy, kind in entries:
            if kind not in groups:
                # Depot-to-station arcs are not used: charging begins only
                # after the first service trip in this duty-recovery gate.
                continue
            arc = Arc(successor, float(travel), float(energy), str(kind))
            if kind == "depot_trip":
                groups[kind][successor] = arc
            elif kind in {"trip_depot", "station_depot"}:
                groups[kind][source] = arc
            else:
                groups[kind].setdefault(source, {})[successor] = arc
    return groups


def _idle_energy(profile: VehicleProfile, minutes: float) -> float:
    return max(0.0, float(minutes)) * profile.idle_kw / 60.0


def _transition_options(
    problem: ProblemData,
    arcs: dict,
    profile: VehicleProfile,
    trip: int,
    entry_soc_kwh: float,
    successor: int | None,
    *,
    horizon_min: float,
    charge_policy: str = "early_disconnect",
) -> list[tuple[float, dict]]:
    reserve = profile.reserve_kwh
    after_trip = entry_soc_kwh - float(problem.trip_energy[trip])
    if after_trip < reserve - TOL:
        return []
    deadline = (
        horizon_min if successor is None else float(problem.start_min[successor])
    )
    options: list[tuple[float, dict]] = []
    direct = (
        arcs["trip_depot"].get(trip)
        if successor is None
        else arcs["trip_trip"].get(trip, {}).get(successor)
    )
    if direct is not None:
        arrival = float(problem.end_min[trip]) + direct.travel_min
        waiting = max(0.0, deadline - arrival) if successor is not None else 0.0
        remaining = after_trip - direct.energy_kwh - _idle_energy(profile, waiting)
        if arrival <= deadline + TOL and remaining >= reserve - TOL:
            if successor is None or (
                remaining + TOL
                >= float(problem.trip_energy[successor]) + reserve
            ):
                options.append((remaining, {
                    "kind": "direct",
                    "from_trip": trip,
                    "next_trip": successor,
                    "travel_min": direct.travel_min,
                    "deadhead_kwh": direct.energy_kwh,
                    "waiting_min": waiting,
                    "idle_kwh": _idle_energy(profile, waiting),
                }))

    for station, inbound in sorted(
        arcs["trip_station"].get(trip, {}).items(),
        key=lambda row: str(row[0]),
    ):
        site = base_site(station)
        if site != "PARX" and site not in profile.allowed_opportunity_sites:
            continue
        outbound = (
            arcs["station_depot"].get(station)
            if successor is None
            else arcs["station_trip"].get(station, {}).get(successor)
        )
        if outbound is None:
            continue
        station_arrival = float(problem.end_min[trip]) + inbound.travel_min
        arrival_soc = after_trip - inbound.energy_kwh
        latest_departure = deadline - outbound.travel_min
        available = latest_departure - station_arrival
        if arrival_soc < reserve - TOL or available < -TOL:
            continue
        if charge_policy not in {"early_disconnect", "hold_until_departure"}:
            raise ValueError(f"unknown charge policy {charge_policy!r}")
        charge = charge_window(
            profile,
            station,
            arrival_soc,
            available,
            hold_until_departure=charge_policy == "hold_until_departure",
        )
        if charge is None:
            continue
        remaining = charge["end_soc_kwh"] - outbound.energy_kwh
        if successor is not None:
            # This gate uses a deterministic early-charge policy. Any idle
            # time after disconnect has already been deducted by charge_window.
            required = float(problem.trip_energy[successor]) + reserve
            if remaining + TOL < required:
                continue
        elif remaining < reserve - TOL:
            continue
        options.append((remaining, {
            "kind": "charge",
            "from_trip": trip,
            "next_trip": successor,
            "station": site,
            "arrival_min": station_arrival,
            "latest_departure_min": latest_departure,
            "inbound_min": inbound.travel_min,
            "inbound_kwh": inbound.energy_kwh,
            "outbound_min": outbound.travel_min,
            "outbound_kwh": outbound.energy_kwh,
            "connection_start_min": station_arrival + charge["setup_min"],
            "connection_end_min": (
                station_arrival + charge["setup_min"] + charge["connected_min"]
            ),
            "setup_start_min": station_arrival,
            **charge,
        }))
    return options


def _best_transition(*args, **kwargs) -> tuple[float, dict] | None:
    options = _transition_options(*args, **kwargs)
    if not options:
        return None
    successor = args[5] if len(args) > 5 else kwargs.get("successor")
    if successor is None:
        direct = [row for row in options if row[1]["kind"] == "direct"]
        if direct:
            return max(direct, key=lambda row: row[0])
        return min(
            options,
            key=lambda row: (
                row[1].get("connected_min", math.inf),
                -row[0],
                json.dumps(row[1], sort_keys=True),
            ),
        )
    return max(options, key=lambda row: (row[0], json.dumps(row[1], sort_keys=True)))


def fixed_sequence_recovery(
    problem: ProblemData,
    trip_sequence: tuple[int, ...],
    profile: VehicleProfile,
    *,
    horizon_min: float,
) -> dict:
    started = time.perf_counter()
    arcs = _arc_groups(problem)
    first = arcs["depot_trip"].get(trip_sequence[0])
    if first is None or first.travel_min > problem.start_min[trip_sequence[0]] + TOL:
        return {"feasible": False, "reason": "depot_cannot_reach_first_trip"}
    soc = profile.usable_capacity_kwh - first.energy_kwh
    if soc + TOL < problem.trip_energy[trip_sequence[0]] + profile.reserve_kwh:
        return {"feasible": False, "reason": "first_trip_violates_soc_floor"}
    actions = ({
        "kind": "source", "next_trip": trip_sequence[0],
        "travel_min": first.travel_min, "deadhead_kwh": first.energy_kwh,
    },)
    for position, trip in enumerate(trip_sequence):
        successor = (
            trip_sequence[position + 1]
            if position + 1 < len(trip_sequence) else None
        )
        selected = _best_transition(
            problem, arcs, profile, trip, soc, successor,
            horizon_min=horizon_min,
        )
        if selected is None:
            return {
                "feasible": False,
                "reason": "no_feasible_transition",
                "failed_after_trip": trip,
                "failed_before_trip": successor,
                "runtime_s": time.perf_counter() - started,
            }
        soc, action = selected
        actions += (action,)
    return {
        "feasible": True,
        "reason": None,
        "trip_count": len(trip_sequence),
        "terminal_soc_kwh": soc,
        "terminal_soc_fraction": soc / profile.usable_capacity_kwh,
        "charging_stops": sum(a["kind"] == "charge" for a in actions),
        "actions": actions,
        "runtime_s": time.perf_counter() - started,
    }


def unrestricted_pricing_recovery(
    problem: ProblemData,
    profile: VehicleProfile,
    *,
    horizon_min: float,
) -> dict:
    """Return an exact maximum-cardinality route for the audit transition model."""

    started = time.perf_counter()
    arcs = _arc_groups(problem)
    ordered = sorted(problem.trips, key=lambda t: (problem.start_min[t], t))
    labels: dict[tuple[int, int], Label] = {}
    for trip in ordered:
        first = arcs["depot_trip"].get(trip)
        if first is None or first.travel_min > problem.start_min[trip] + TOL:
            continue
        soc = profile.usable_capacity_kwh - first.energy_kwh
        if soc + TOL < problem.trip_energy[trip] + profile.reserve_kwh:
            continue
        labels[trip, 1] = Label(
            trip=trip, trip_count=1, entry_soc_kwh=soc,
            trips=(trip,), actions=({
                "kind": "source", "next_trip": trip,
                "travel_min": first.travel_min,
                "deadhead_kwh": first.energy_kwh,
            },),
        )
    for trip in ordered:
        current = sorted(
            (label for (last, _count), label in labels.items() if last == trip),
            key=lambda label: label.trip_count,
        )
        for label in current:
            for successor in ordered:
                if problem.start_min[successor] < problem.end_min[trip] - TOL:
                    continue
                if successor in label.trips:
                    continue
                selected = _best_transition(
                    problem, arcs, profile, trip, label.entry_soc_kwh,
                    successor, horizon_min=horizon_min,
                )
                if selected is None:
                    continue
                soc, action = selected
                candidate = Label(
                    trip=successor,
                    trip_count=label.trip_count + 1,
                    entry_soc_kwh=soc,
                    trips=label.trips + (successor,),
                    actions=label.actions + (action,),
                )
                key = (successor, candidate.trip_count)
                incumbent = labels.get(key)
                if incumbent is None or (
                    candidate.entry_soc_kwh,
                    tuple(-t for t in candidate.trips),
                ) > (
                    incumbent.entry_soc_kwh,
                    tuple(-t for t in incumbent.trips),
                ):
                    labels[key] = candidate
    terminal = []
    for label in labels.values():
        selected = _best_transition(
            problem, arcs, profile, label.trip, label.entry_soc_kwh,
            None, horizon_min=horizon_min,
        )
        if selected is not None:
            soc, action = selected
            terminal.append((label.trip_count, soc, label, action))
    if not terminal:
        return {
            "feasible": False, "reason": "no_route_returns_to_depot",
            "runtime_s": time.perf_counter() - started,
        }
    count, soc, label, action = max(
        terminal, key=lambda row: (row[0], row[1], tuple(-t for t in row[2].trips))
    )
    return {
        "feasible": True,
        "reason": None,
        "recovered_trip_count": count,
        "total_trip_count": len(problem.trips),
        "recovered_all_trips": count == len(problem.trips),
        "trips": label.trips,
        "actions": label.actions + (action,),
        "terminal_soc_kwh": soc,
        "runtime_s": time.perf_counter() - started,
        "dominance_scope": "highest SOC retained per (last trip, trip count)",
    }


def _restrict_problem(problem: ProblemData, retained: set[int]) -> ProblemData:
    def retain_node(node):
        return not isinstance(node, int) or node in retained

    adjacency = {}
    for source, entries in problem.adjacency.items():
        if not retain_node(source):
            continue
        filtered = [entry for entry in entries if retain_node(entry[0])]
        if filtered:
            adjacency[source] = filtered
    trips = tuple(trip for trip in problem.trips if trip in retained)
    return ProblemData(
        frame=problem.frame,
        trips=trips,
        adjacency=adjacency,
        start_min={trip: problem.start_min[trip] for trip in trips},
        end_min={trip: problem.end_min[trip] for trip in trips},
        trip_energy={trip: problem.trip_energy[trip] for trip in trips},
    )


def greedy_two_route_recovery(
    problem: ProblemData,
    profile: VehicleProfile,
    *,
    horizon_min: float,
) -> dict:
    """Peel one unrestricted pricing route, then price the exact residual."""

    routes = []
    remaining = set(problem.trips)
    for _iteration in range(2):
        if not remaining:
            break
        residual = _restrict_problem(problem, remaining)
        route = unrestricted_pricing_recovery(
            residual, profile, horizon_min=horizon_min
        )
        if not route.get("feasible") or not route.get("trips"):
            break
        routes.append(route)
        remaining.difference_update(route["trips"])
    return {
        "method": "greedy_max_cardinality_route_then_residual",
        "recovered_exact_partition_with_at_most_two_routes": not remaining,
        "route_count": len(routes),
        "covered_trip_count": len(problem.trips) - len(remaining),
        "total_trip_count": len(problem.trips),
        "remaining_trips": sorted(remaining),
        "routes": routes,
        "scope": "heuristic two-route recovery; not column generation or proof of failure",
    }


PARTILLE_CHARGER_COUNTS = {
    "2190L": 1, "4808": 1, "3127L": 2, "7880C": 1, "JON_A": 1,
}


def charger_overlap_audit(routes: list[dict]) -> dict:
    events = {site: [] for site in PARTILLE_CHARGER_COUNTS}
    for route_index, route in enumerate(routes):
        for action in route.get("actions", ()):
            if action.get("kind") != "charge":
                continue
            site = base_site(action["station"])
            if site not in events:
                continue
            events[site].append((
                float(action["setup_start_min"]), 1, route_index,
            ))
            events[site].append((
                float(action["connection_end_min"]), -1, route_index,
            ))
    rows = {}
    all_fit = True
    for site, site_events in events.items():
        # End before start at equal time implements half-open [start,end).
        current = peak = 0
        for _minute, delta, _route in sorted(
            site_events, key=lambda row: (row[0], row[1])
        ):
            current += delta
            peak = max(peak, current)
        capacity = PARTILLE_CHARGER_COUNTS[site]
        rows[site] = {"documented_chargers": capacity, "peak_connections": peak}
        all_fit = all_fit and peak <= capacity
    return {
        "all_documented_charger_counts_fit": all_fit,
        "locations": rows,
        "scope": (
            "conservative charger-bay occupancy from setup start through disconnect; platform blocking and FIFO are not checked"
        ),
    }


def _write_frame(frame: pd.DataFrame, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    frame.to_csv(path, index=False)
    return path


def _problem_for_frame(
    frame: pd.DataFrame,
    temporary: Path,
    name: str,
    *,
    reference_data_dir: Path,
    horizon_min: float,
) -> tuple[Path, ProblemData]:
    path = _write_frame(frame, temporary, name)
    return path, build_problem(
        temporary, name,
        max_trip2trip_min=horizon_min,
        max_trip_to_station_min=horizon_min,
        max_station_to_trip_wait_min=horizon_min,
        reference_data_dir=reference_data_dir,
        horizon_min=horizon_min,
    )


def load_duty_frames_from(master_path: Path) -> dict[str, pd.DataFrame]:
    """Extract literal duties from the caller-selected master CSV."""

    master = pd.read_csv(master_path)
    regular = master[
        master["Identifier"].eq("Regular")
        & master["Ordered_Trip_ID"].notna()
    ].copy()
    regular["VehicleTask"] = regular["VehicleTask"].astype(str)
    regular["Ordered_Trip_ID"] = regular["Ordered_Trip_ID"].astype(int)

    def minutes(value):
        hour, minute = str(value).split(":")
        return int(hour) * 60 + int(minute)

    frames = {}
    for duty, group in regular.groupby("VehicleTask"):
        frame = group[
            [column for column in INSTANCE_COLUMNS if column != "count_trip_id"]
        ].copy()
        frame = (
            frame.assign(_sort=frame["Start1"].map(minutes))
            .sort_values(["_sort", "Ordered_Trip_ID"])
            .drop(columns="_sort")
            .reset_index(drop=True)
        )
        frame["count_trip_id"] = range(len(frame))
        frames[duty] = frame[INSTANCE_COLUMNS]
    return frames


def select_k2_pairs(frames: dict[str, pd.DataFrame]) -> list[tuple[str, str, str]]:
    by_profile = {
        profile: sorted(
            (duty for duty in frames if profile_for_duty(duty).name == profile),
            key=lambda duty: (len(frames[duty]), duty),
        )
        for profile in ("18E1", "18E2")
    }
    e1, e2 = by_profile["18E1"], by_profile["18E2"]
    candidates = [
        ("e2_short", e2[0], e2[1]),
        ("e2_long", e2[-1], e2[-2]),
        ("e1_short", e1[0], e1[1]),
        ("e1_long", e1[-1], e1[-2]),
    ]
    for _label, left, right in candidates:
        if _base_task(left) == _base_task(right):
            raise ValueError("k2 selection contains weekday variants of one base duty")
    return candidates


def run_audit(
    *,
    data_dir: Path,
    output: Path,
    pair_output: Path,
    input_dir: Path,
    horizon_min: float,
) -> dict:
    source_master = data_dir / "Par_VehicleDetails_Updated.csv"
    frames = load_duty_frames_from(source_master)
    duty_rows = []
    by_duty = {}
    k1_input_dir = input_dir / "k1"
    k2_input_dir = input_dir / "k2"
    for duty, frame in sorted(frames.items()):
        profile = profile_for_duty(duty)
        path, problem = _problem_for_frame(
            frame, k1_input_dir, f"duty_{duty}.csv",
            reference_data_dir=data_dir, horizon_min=horizon_min,
        )
        sequence = tuple(problem.trips)
        fixed = fixed_sequence_recovery(
            problem, sequence, profile, horizon_min=horizon_min
        )
        unrestricted = unrestricted_pricing_recovery(
            problem, profile, horizon_min=horizon_min
        )
        source_ids = tuple(int(value) for value in frame["Ordered_Trip_ID"])
        recovered_ids = tuple(
            source_ids[index] for index in unrestricted.get("trips", ())
        )
        row = {
                "duty_id": duty,
                "base_duty_id": _base_task(duty),
                "vehicle_profile": profile.name,
                "trip_count": len(frame),
                "instance_sha256": sha256(path),
                "fixed_duty_feasible": fixed["feasible"],
                "fixed_duty_failure_reason": fixed.get("reason"),
                "unrestricted_recovered_trip_count": unrestricted.get(
                    "recovered_trip_count", 0
                ),
                "unrestricted_recovered_all_trips": unrestricted.get(
                    "recovered_all_trips", False
                ),
                "unrestricted_exact_giro_order": recovered_ids == source_ids,
                "fixed": fixed,
                "unrestricted": unrestricted,
        }
        duty_rows.append(row)
        by_duty[duty] = row

    pair_rows = []
    for label, left, right in select_k2_pairs(frames):
        merged = merge_duties(frames, [left, right])
        path = _write_frame(merged, k2_input_dir, f"pair_{label}.csv")
        profile = profile_for_duty(left)
        _path, problem = _problem_for_frame(
            merged, k2_input_dir, f"pair_{label}.csv",
            reference_data_dir=data_dir, horizon_min=horizon_min,
        )
        greedy = greedy_two_route_recovery(
            problem, profile, horizon_min=horizon_min
        )
        known_routes = [by_duty[left]["fixed"], by_duty[right]["fixed"]]
        greedy_capacity = charger_overlap_audit(greedy["routes"])
        known_capacity = charger_overlap_audit(known_routes)
        pair_rows.append({
                "cell_id": label,
                "duties": [left, right],
                "base_duties_distinct": _base_task(left) != _base_task(right),
                "profiles": [
                    profile_for_duty(left).name, profile_for_duty(right).name
                ],
                "trip_count": len(merged),
                "instance_sha256": sha256(path),
            "known_two_duty_single_vehicle_feasible": (
                by_duty[left]["fixed_duty_feasible"]
                and by_duty[right]["fixed_duty_feasible"]
            ),
            "known_partition_charger_overlap": known_capacity,
            "greedy_two_route_recovery": greedy,
            "greedy_two_route_charger_overlap": greedy_capacity,
        })

    payload = {
        "schema": SCHEMA,
        "source_master": str(source_master),
        "source_master_sha256": sha256(source_master),
        "input_dir": str(input_dir),
        "horizon_min": horizon_min,
        "charge_connection_policy": "early_disconnect_then_idle_draw",
        "duty_count": len(duty_rows),
        "fixed_duty_feasible_count": sum(
            row["fixed_duty_feasible"] for row in duty_rows
        ),
        "unrestricted_recovered_all_count": sum(
            row["unrestricted_recovered_all_trips"] for row in duty_rows
        ),
        "unrestricted_exact_giro_order_count": sum(
            row["unrestricted_exact_giro_order"] for row in duty_rows
        ),
        "scope": documented_single_vehicle_scope(),
        "duties": duty_rows,
    }
    pair_payload = {
        "schema": PAIR_SCHEMA,
        "selection_rule": (
            "shortest and longest within each vehicle group; no pair mixes "
            "vehicle groups or contains variants of one base duty"
        ),
        "requires_review_before_cluster_launch": True,
        "pair_count": len(pair_rows),
        "pairs": pair_rows,
        "scientific_scope": (
            "Known-duty partition and greedy two-route pricing recovery gate. "
            "The greedy result is not full column generation or an optimality proof."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pair_output.write_text(json.dumps(pair_payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parent.parent
    parser.add_argument("--data-dir", type=Path, default=repo / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair-out", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--horizon-min", type=float, default=DEFAULT_HORIZON_MIN)
    args = parser.parse_args(argv)
    input_dir = (
        args.input_dir.resolve()
        if args.input_dir is not None
        else args.out.resolve().parent / "inputs"
    )
    payload = run_audit(
        data_dir=args.data_dir.resolve(), output=args.out.resolve(),
        pair_output=args.pair_out.resolve(), input_dir=input_dir,
        horizon_min=args.horizon_min,
    )
    print(json.dumps({
        "duty_count": payload["duty_count"],
        "fixed_duty_feasible_count": payload["fixed_duty_feasible_count"],
        "unrestricted_recovered_all_count": payload[
            "unrestricted_recovered_all_count"
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Trip-dual weighted route pricing for the bounded GIRO small-CG study."""

from __future__ import annotations

import time
from dataclasses import dataclass

from audit_giro_duty_recovery import _arc_groups, _best_transition


TOL = 1e-9


@dataclass(frozen=True)
class WeightedLabel:
    trip: int
    reward: float
    entry_soc_kwh: float
    trips: tuple[int, ...]
    actions: tuple[dict, ...]


def _dominates(left: WeightedLabel, right: WeightedLabel) -> bool:
    return (
        left.reward >= right.reward - TOL
        and left.entry_soc_kwh >= right.entry_soc_kwh - TOL
        and (
            left.reward > right.reward + TOL
            or left.entry_soc_kwh > right.entry_soc_kwh + TOL
            or left.trips <= right.trips
        )
    )


def _accept(frontier: list[WeightedLabel], candidate: WeightedLabel) -> bool:
    if any(_dominates(row, candidate) for row in frontier):
        return False
    frontier[:] = [row for row in frontier if not _dominates(candidate, row)]
    frontier.append(candidate)
    frontier.sort(key=lambda row: (-row.reward, -row.entry_soc_kwh, row.trips))
    return True


def weighted_price_route(
    problem,
    profile,
    trip_duals,
    *,
    horizon_min,
    wall_limit_s,
    label_limit,
) -> dict:
    """Minimize ``1 - sum(trip duals)`` over one feasible route.

    The DP uses hold-until-departure charging while labeling. Under that
    policy, higher entry SOC and higher collected reward are valid dominance
    resources. The driver separately replays the selected trip order with its
    early-disconnect policy to add a less capacity-intensive schedule variant.
    """

    started = time.perf_counter()
    deadline = started + float(wall_limit_s)
    limit = int(label_limit)
    arcs = _arc_groups(problem)
    ordered = sorted(problem.trips, key=lambda trip: (problem.start_min[trip], trip))
    frontiers = {trip: [] for trip in ordered}
    labels_created = 0
    transitions_tested = 0
    guard = None

    for trip in ordered:
        first = arcs["depot_trip"].get(trip)
        if first is None or first.travel_min > problem.start_min[trip] + TOL:
            continue
        soc = profile.usable_capacity_kwh - first.energy_kwh
        if soc + TOL < problem.trip_energy[trip] + profile.reserve_kwh:
            continue
        label = WeightedLabel(
            trip=trip,
            reward=float(trip_duals.get(trip, 0.0)),
            entry_soc_kwh=soc,
            trips=(trip,),
            actions=({
                "kind": "source", "next_trip": trip,
                "travel_min": first.travel_min,
                "deadhead_kwh": first.energy_kwh,
            },),
        )
        _accept(frontiers[trip], label)
        labels_created += 1

    for position, trip in enumerate(ordered):
        for label in tuple(frontiers[trip]):
            if time.perf_counter() >= deadline:
                guard = "pricing_wall_limit"
                break
            if labels_created >= limit:
                guard = "pricing_label_limit"
                break
            for successor in ordered[position + 1:]:
                if transitions_tested % 128 == 0 and time.perf_counter() >= deadline:
                    guard = "pricing_wall_limit"
                    break
                if problem.start_min[successor] < problem.end_min[trip] - TOL:
                    continue
                transitions_tested += 1
                selected = _best_transition(
                    problem, arcs, profile, trip, label.entry_soc_kwh,
                    successor, horizon_min=horizon_min,
                    charge_policy="hold_until_departure",
                )
                if selected is None:
                    continue
                soc, action = selected
                candidate = WeightedLabel(
                    trip=successor,
                    reward=label.reward + float(trip_duals.get(successor, 0.0)),
                    entry_soc_kwh=soc,
                    trips=label.trips + (successor,),
                    actions=label.actions + (action,),
                )
                if _accept(frontiers[successor], candidate):
                    labels_created += 1
                    if labels_created >= limit:
                        guard = "pricing_label_limit"
                        break
            if guard:
                break
            if guard:
                break
        if guard:
            break

    terminal = []
    terminal_time_exhausted = False
    for frontier in frontiers.values():
        for label in frontier:
            if time.perf_counter() >= deadline:
                guard = guard or "pricing_wall_limit"
                terminal_time_exhausted = True
                break
            selected = _best_transition(
                problem, arcs, profile, label.trip, label.entry_soc_kwh,
                None, horizon_min=horizon_min,
                charge_policy="hold_until_departure",
            )
            if selected is None:
                continue
            soc, action = selected
            terminal.append((1.0 - label.reward, -soc, label, action))
        if terminal_time_exhausted:
            break
    if not terminal:
        return {
            "route": None,
            "guard": guard,
            "labels_created": labels_created,
            "transitions_tested": transitions_tested,
            "runtime_s": time.perf_counter() - started,
        }
    reduced_cost, negative_soc, label, action = min(
        terminal,
        key=lambda row: (row[0], row[1], row[2].trips),
    )
    return {
        "route": {
            "trips": list(label.trips),
            "actions": list(label.actions + (action,)),
            "profile": profile.name,
            "cost": 1.0,
            "pricing_charge_policy": "hold_until_departure",
        },
        "reduced_cost_without_capacity_duals": float(reduced_cost),
        "reward": float(label.reward),
        "guard": guard,
        "labels_created": labels_created,
        "transitions_tested": transitions_tested,
        "runtime_s": time.perf_counter() - started,
        "single_vehicle_weighted_pricing_complete": guard is None,
    }

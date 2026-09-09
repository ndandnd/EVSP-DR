"""Trip-dual weighted route pricing for the bounded GIRO small-CG study."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from audit_giro_duty_recovery import _arc_groups, _transition_options


TOL = 1e-9


@dataclass(frozen=True)
class WeightedLabel:
    trip: int
    reward: float
    trip_reward: float
    capacity_dual_reward: float
    entry_soc_kwh: float
    trips: tuple[int, ...]
    actions: tuple[dict, ...]
    capacity_rows: frozenset[tuple[str, int]]


def action_capacity_rows(action: dict) -> frozenset[tuple[str, int]]:
    """Return the same conservative one-minute plug rows used by the master."""

    if action.get("kind") != "charge" or action.get("station") == "PARX":
        return frozenset()
    start = float(action["setup_start_min"])
    end = float(action["connection_end_min"])
    rows = {
        (str(action["station"]), minute)
        for minute in range(int(math.floor(start)), int(math.ceil(end)))
        if minute + 1e-9 < end and minute + 1.0 > start + 1e-9
    }
    return frozenset(rows)


def _extend_reward(label, action, capacity_duals):
    action_rows = action_capacity_rows(action)
    overlap = action_rows & label.capacity_rows
    if overlap:
        raise AssertionError(
            "chronological pricing transition reused charger rows: "
            f"{sorted(overlap)[:5]}"
        )
    new_rows = action_rows - label.capacity_rows
    capacity_delta = sum(float(capacity_duals.get(row, 0.0)) for row in new_rows)
    return capacity_delta, label.capacity_rows | action_rows


def _dominates(left: WeightedLabel, right: WeightedLabel) -> bool:
    # Prior charger rows are not a separate future resource. Transition times
    # are chronological: every charge ends before its successor trip starts,
    # and the next charge cannot start before that trip ends. _extend_reward
    # asserts the resulting row sets are disjoint for every generated label.
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
    capacity_duals=None,
    *,
    horizon_min,
    wall_limit_s,
    label_limit,
) -> dict:
    """Price one route with trip and conservative charger-row duals.

    The DP uses hold-until-departure charging while labeling. Under that
    policy, higher entry SOC and higher collected dual reward are valid
    dominance resources. Every direct and charging transition is retained
    through the reward/SOC Pareto test. The driver separately replays the
    selected trip order with its early-disconnect policy to add a less
    capacity-intensive schedule variant outside this pricing space.
    """

    started = time.perf_counter()
    capacity_duals = capacity_duals or {}
    hard_deadline = started + float(wall_limit_s)
    expansion_deadline = started + 0.80 * float(wall_limit_s)
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
            trip_reward=float(trip_duals.get(trip, 0.0)),
            capacity_dual_reward=0.0,
            entry_soc_kwh=soc,
            trips=(trip,),
            actions=({
                "kind": "source", "next_trip": trip,
                "travel_min": first.travel_min,
                "deadhead_kwh": first.energy_kwh,
            },),
            capacity_rows=frozenset(),
        )
        _accept(frontiers[trip], label)
        labels_created += 1

    for position, trip in enumerate(ordered):
        for label in tuple(frontiers[trip]):
            if time.perf_counter() >= expansion_deadline:
                guard = "pricing_expansion_wall_reserve"
                break
            if labels_created >= limit:
                guard = "pricing_label_limit"
                break
            for successor in ordered[position + 1:]:
                if (
                    transitions_tested % 128 == 0
                    and time.perf_counter() >= expansion_deadline
                ):
                    guard = "pricing_expansion_wall_reserve"
                    break
                if problem.start_min[successor] < problem.end_min[trip] - TOL:
                    continue
                transitions_tested += 1
                options = _transition_options(
                    problem, arcs, profile, trip, label.entry_soc_kwh,
                    successor, horizon_min=horizon_min,
                    charge_policy="hold_until_departure",
                )
                for soc, action in options:
                    capacity_delta, capacity_rows = _extend_reward(
                        label, action, capacity_duals,
                    )
                    trip_delta = float(trip_duals.get(successor, 0.0))
                    candidate = WeightedLabel(
                        trip=successor,
                        reward=label.reward + trip_delta + capacity_delta,
                        trip_reward=label.trip_reward + trip_delta,
                        capacity_dual_reward=(
                            label.capacity_dual_reward + capacity_delta
                        ),
                        entry_soc_kwh=soc,
                        trips=label.trips + (successor,),
                        actions=label.actions + (action,),
                        capacity_rows=capacity_rows,
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
        if guard:
            break

    terminal = []
    terminal_time_exhausted = False
    terminal_candidates = sorted(
        (label for frontier in frontiers.values() for label in frontier),
        key=lambda label: (-label.reward, -label.entry_soc_kwh, label.trips),
    )
    for label in terminal_candidates:
        if time.perf_counter() >= hard_deadline:
            guard = (
                f"{guard}+pricing_terminal_wall_limit"
                if guard else "pricing_terminal_wall_limit"
            )
            terminal_time_exhausted = True
            break
        options = _transition_options(
            problem, arcs, profile, label.trip, label.entry_soc_kwh,
            None, horizon_min=horizon_min,
            charge_policy="hold_until_departure",
        )
        for soc, action in options:
            capacity_delta, capacity_rows = _extend_reward(
                label, action, capacity_duals,
            )
            total_reward = label.reward + capacity_delta
            terminal.append((
                1.0 - total_reward,
                -soc,
                label,
                action,
                label.capacity_dual_reward + capacity_delta,
                capacity_rows,
            ))
    if not terminal:
        return {
            "route": None,
            "guard": guard,
            "labels_created": labels_created,
            "transitions_tested": transitions_tested,
            "runtime_s": time.perf_counter() - started,
        }
    reduced_cost, negative_soc, label, action, capacity_reward, capacity_rows = min(
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
            "capacity_rows": sorted([list(row) for row in capacity_rows]),
        },
        "reduced_cost_with_capacity_duals": float(reduced_cost),
        "trip_dual_reward": float(label.trip_reward),
        "capacity_dual_reward": float(capacity_reward),
        "total_dual_reward": float(label.trip_reward + capacity_reward),
        "guard": guard,
        "labels_created": labels_created,
        "transitions_tested": transitions_tested,
        "runtime_s": time.perf_counter() - started,
        "hold_policy_label_search_complete": guard is None,
        "full_pricing_space_certified": False,
    }

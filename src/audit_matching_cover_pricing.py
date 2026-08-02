#!/usr/bin/env python3
"""Audit a finite, model-derived matching cover against a saved route pool.

This current-checkout diagnostic rebuilds the active trip graph and a
deterministic matching cover from the locally resolved EVSP inputs.  It does
not read an incumbent bus assignment.  Every route cost used below is recomputed with the same
hour-split, station-specific charging-cost function as the restricted master.

For both the saved seed pool and the saved final pool, the script repeatedly
adds only matching-cover columns having negative reduced cost at the selected
SciPy/HiGHS dual solution.  Finding no such column exhausts only this finite
matching-cover candidate set.  It is not an exact pricing result and must not
be reported as column-generation optimality.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from audit_giro_known_columns import (
    DEFAULT_DATA_DIR,
    DEPOT,
    HORIZON_MIN,
    MAX_DAILY_RECHARGES,
    STATION_NODE_BY_BASE,
    STATIONS,
    ProblemData,
    _route_cost,
    _route_trips,
    build_problem,
)
from config import (
    BIG_M_PENALTY,
    CHARGE_RATE_KW,
    CHARGE_START_COST,
    TRAVEL_COST_FACTOR,
    charge_cost_premium,
)
from master_lp_scipy import (
    RestrictedMasterLPResult,
    build_route_incidence,
    solve_restricted_master_lp,
)
from matching_init import build_matching_initial_routes
from utils_v2 import (
    _compute_charging_cost_accurate,
    base_station_name,
    load_station_hourly_prices,
)


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SCHEMA_VERSION = 1
REDUCED_COST_EPSILON = 1e-6
COLUMN_COST_EPSILON = 1e-6
MASTER_METHOD = "highs-ds"


class MatchingCoverAuditInputError(ValueError):
    """Raised when saved inputs cannot support a provenance-safe audit."""


@dataclass(frozen=True)
class MatchingAuditConfig:
    requested_battery_kwh: float
    effective_battery_kwh: float
    horizon_min: float
    max_charge2trip_min: float
    successor_charge_targets: bool
    max_successor_charge_targets: int
    direct_only: bool
    max_matching_attempts: int
    matching_order_seed: int


@dataclass(frozen=True)
class CostedColumn:
    route: dict[str, Any]
    trips: tuple[int, ...]
    incidence: frozenset[int]
    master_cost: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = path.expanduser().resolve()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def resolve_saved_data_path(
    data_dir: Path,
    saved_path: str | os.PathLike[str],
    *,
    expected_sha256: str | None,
    label: str,
) -> tuple[Path, str]:
    """Resolve a possibly machine-specific saved path under ``data_dir``.

    Cluster pools commonly save absolute paths that do not exist on the audit
    machine.  We retain any suffix after a ``data`` path component, plus a
    basename fallback for top-level files, then validate the selected bytes by
    the hash saved with the pool.  Ambiguity is rejected rather than guessed.
    """

    root = data_dir.expanduser().resolve()
    raw = Path(saved_path).expanduser()
    candidates: list[Path] = []
    if not raw.is_absolute():
        candidates.append(root / raw)
        if raw.parts and raw.parts[0] == "data":
            candidates.append(root.joinpath(*raw.parts[1:]))
    else:
        candidates.append(raw)

    data_positions = [
        index for index, part in enumerate(raw.parts) if part == "data"
    ]
    for index in reversed(data_positions):
        suffix = raw.parts[index + 1 :]
        if suffix:
            candidates.append(root.joinpath(*suffix))
    candidates.append(root / raw.name)

    existing = [path for path in _unique_paths(candidates) if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"Could not resolve saved {label} path {saved_path!s} under {root}"
        )

    hashes = {path: _sha256(path) for path in existing}
    if expected_sha256:
        matches = [
            path for path in existing if hashes[path] == expected_sha256
        ]
        if not matches:
            found = ", ".join(
                f"{path}={digest}" for path, digest in hashes.items()
            )
            raise MatchingCoverAuditInputError(
                f"{label} SHA-256 mismatch: expected {expected_sha256}; "
                f"resolved candidates were {found}"
            )
        # Identical bytes at multiple candidate locations are harmless.  Prefer
        # the explicitly selected data root over a still-mounted cluster path.
        def selection_key(path: Path) -> tuple[int, int, str]:
            try:
                relative = path.relative_to(root)
            except ValueError:
                return (1, len(path.parts), str(path))
            return (0, len(relative.parts), str(relative))

        selected = min(matches, key=selection_key)
        return selected, hashes[selected]

    if len(existing) != 1:
        raise MatchingCoverAuditInputError(
            f"Saved pool has no {label} SHA-256 and resolves ambiguously: "
            + ", ".join(str(path) for path in existing)
        )
    selected = existing[0]
    return selected, hashes[selected]


def validate_saved_pool(pool: Mapping[str, Any]) -> tuple[tuple[int, ...], int]:
    """Validate the saved final-pool fields used by this audit."""

    required = {
        "csv_name",
        "prices_csv",
        "trip_ids",
        "routes",
        "seed_route_count",
    }
    missing = sorted(required - set(pool))
    if missing:
        raise MatchingCoverAuditInputError(
            f"Saved final pool is missing required fields: {missing}"
        )

    trip_ids = tuple(pool["trip_ids"])
    if not trip_ids or any(not isinstance(trip, int) for trip in trip_ids):
        raise MatchingCoverAuditInputError(
            "trip_ids must be a nonempty list of integer local trip IDs"
        )
    if len(set(trip_ids)) != len(trip_ids):
        raise MatchingCoverAuditInputError("trip_ids contains duplicates")

    routes = pool["routes"]
    if not isinstance(routes, list):
        raise MatchingCoverAuditInputError("routes must be a list")
    seed_count = pool["seed_route_count"]
    if isinstance(seed_count, bool) or not isinstance(seed_count, int):
        raise MatchingCoverAuditInputError("seed_route_count must be an integer")
    if not 0 <= seed_count <= len(routes):
        raise MatchingCoverAuditInputError(
            "seed_route_count must be between zero and len(routes)"
        )
    return trip_ids, seed_count


def _required_run_argument(pool: Mapping[str, Any], name: str) -> Any:
    run_arguments = pool.get("run_arguments")
    if not isinstance(run_arguments, Mapping) or name not in run_arguments:
        raise MatchingCoverAuditInputError(
            f"Saved final pool lacks run_arguments.{name}; refusing to guess "
            "the pricing configuration"
        )
    return run_arguments[name]


def matching_audit_config(pool: Mapping[str, Any]) -> MatchingAuditConfig:
    """Extract and cross-check model settings recorded with a final pool."""

    requested = pool.get("battery_kwh")
    if requested is None:
        requested = _required_run_argument(pool, "G")
    requested = float(requested)
    run_g = float(_required_run_argument(pool, "G"))
    if not math.isclose(requested, run_g, rel_tol=0.0, abs_tol=1e-9):
        raise MatchingCoverAuditInputError(
            f"battery_kwh={requested} disagrees with run_arguments.G={run_g}"
        )
    if requested <= 0:
        raise MatchingCoverAuditInputError("battery capacity must be positive")

    # This mirrors run_ex_unicorn.py's current VSP sentinel handling.
    effective = 300.0 if requested >= 9000.0 else requested
    max_wait = float(_required_run_argument(pool, "max_charge2trip"))
    successor_targets = _required_run_argument(
        pool, "successor_charge_targets"
    )
    direct_only = _required_run_argument(pool, "matching_direct_only")
    if not isinstance(successor_targets, bool):
        raise MatchingCoverAuditInputError(
            "run_arguments.successor_charge_targets must be boolean"
        )
    if not isinstance(direct_only, bool):
        raise MatchingCoverAuditInputError(
            "run_arguments.matching_direct_only must be boolean"
        )
    max_targets = int(_required_run_argument(pool, "max_successor_charge_targets"))
    attempts = int(_required_run_argument(pool, "matching_attempts"))
    order_seed = int(_required_run_argument(pool, "matching_order_seed"))

    top_level_checks = {
        "max_charge2trip": max_wait,
        "successor_charge_targets": successor_targets,
        "max_successor_charge_targets": max_targets,
    }
    disagreements = {
        key: {"top_level": pool[key], "run_arguments": expected}
        for key, expected in top_level_checks.items()
        if key in pool and pool[key] is not None and pool[key] != expected
    }
    if disagreements:
        raise MatchingCoverAuditInputError(
            "Saved top-level configuration disagrees with run_arguments: "
            + json.dumps(disagreements, sort_keys=True)
        )
    if max_wait < 0:
        raise MatchingCoverAuditInputError("max_charge2trip cannot be negative")
    if max_targets <= 0 or attempts <= 0:
        raise MatchingCoverAuditInputError(
            "matching target cap and attempt count must be positive"
        )
    return MatchingAuditConfig(
        requested_battery_kwh=requested,
        effective_battery_kwh=effective,
        horizon_min=float(HORIZON_MIN),
        max_charge2trip_min=max_wait,
        successor_charge_targets=successor_targets,
        max_successor_charge_targets=max_targets,
        direct_only=direct_only,
        max_matching_attempts=attempts,
        matching_order_seed=order_seed,
    )


def validate_matching_cover(
    routes: Sequence[Mapping[str, Any]],
    trip_ids: Sequence[int],
    *,
    direct_only: bool,
) -> dict[str, Any]:
    """Validate exact partitioning and matching resource provenance."""

    if not routes:
        raise MatchingCoverAuditInputError("Matching initializer returned no routes")
    active = set(trip_ids)
    occurrences: list[int] = []
    provenances: list[Mapping[str, Any]] = []
    for route_index, route in enumerate(routes):
        integers = [node for node in route.get("route", ()) if isinstance(node, int)]
        unknown = [trip for trip in integers if trip not in active]
        if unknown:
            raise MatchingCoverAuditInputError(
                f"Matching route {route_index} contains unknown trips: {unknown}"
            )
        if not integers:
            raise MatchingCoverAuditInputError(
                f"Matching route {route_index} contains no active trips"
            )
        if len(integers) != len(set(integers)):
            raise MatchingCoverAuditInputError(
                f"Matching route {route_index} repeats an active trip"
            )
        occurrences.extend(integers)
        provenance = route.get("_matching_init")
        if not isinstance(provenance, Mapping):
            raise MatchingCoverAuditInputError(
                f"Matching route {route_index} lacks _matching_init provenance"
            )
        provenances.append(provenance)

    missing = sorted(active - set(occurrences))
    duplicates = sorted(
        trip for trip in active if occurrences.count(trip) != 1
    )
    if missing or duplicates or len(occurrences) != len(trip_ids):
        raise MatchingCoverAuditInputError(
            "Matching cover is not an exact trip partition: "
            f"missing={missing}, non_singleton_occurrences={duplicates}"
        )

    first = dict(provenances[0])
    if any(dict(provenance) != first for provenance in provenances[1:]):
        raise MatchingCoverAuditInputError(
            "Matching routes do not share identical resource provenance"
        )
    expected_mode = "direct_only" if direct_only else "full"
    expected_fields = {
        "compatibility_mode": expected_mode,
        "path_count": len(routes),
        "resource_feasible_path_count": len(routes),
    }
    mismatches = {
        key: {"expected": expected, "actual": first.get(key)}
        for key, expected in expected_fields.items()
        if first.get(key) != expected
    }
    repair_mode = first.get("resource_repair_mode")
    if repair_mode not in {"none", "contiguous_split"}:
        mismatches["resource_repair_mode"] = {
            "expected": "none or contiguous_split",
            "actual": repair_mode,
        }
    if first.get("is_exact_minimum_path_cover") != (repair_mode == "none"):
        mismatches["is_exact_minimum_path_cover"] = {
            "expected": repair_mode == "none",
            "actual": first.get("is_exact_minimum_path_cover"),
        }
    if mismatches:
        raise MatchingCoverAuditInputError(
            "Invalid matching resource provenance: "
            + json.dumps(mismatches, sort_keys=True)
        )

    return {
        "exact_trip_partition": True,
        "partition_trip_count": len(occurrences),
        "unique_incidence_count": len(
            {frozenset(_route_trips(dict(route))) for route in routes}
        ),
        "resource_provenance_validated": True,
        "provenance": first,
    }


def cost_columns(
    routes: Sequence[dict[str, Any]],
    trip_ids: Sequence[int],
    route_cost: Callable[[dict[str, Any]], float],
) -> list[CostedColumn]:
    """Validate incidences and recompute every exact restricted-master cost."""

    active = set(trip_ids)
    trip_lists = [_route_trips(route) for route in routes]
    # This validates nonempty columns, repeated trips, and unknown trip IDs.
    build_route_incidence(trip_ids, trip_lists)
    columns: list[CostedColumn] = []
    for route, trips in zip(routes, trip_lists):
        # Reject integer route nodes outside the active set even if a malformed
        # route happened to retain at least one valid active trip.
        unknown = [
            node
            for node in route.get("route", ())
            if isinstance(node, int) and node not in active
        ]
        if unknown:
            raise MatchingCoverAuditInputError(
                f"Route contains integer nodes outside active trips: {unknown}"
            )
        cost = float(route_cost(route))
        if not math.isfinite(cost):
            raise MatchingCoverAuditInputError(
                "Recomputed route master cost is not finite"
            )
        columns.append(
            CostedColumn(
                route=route,
                trips=tuple(trips),
                incidence=frozenset(trips),
                master_cost=cost,
            )
        )
    return columns


def _solve_master(
    trip_ids: Sequence[int],
    columns: Sequence[CostedColumn],
) -> RestrictedMasterLPResult:
    return solve_restricted_master_lp(
        trip_ids=trip_ids,
        route_incidence=build_route_incidence(
            trip_ids, [column.trips for column in columns]
        ),
        route_costs=[column.master_cost for column in columns],
        artificial_penalty=BIG_M_PENALTY,
        method=MASTER_METHOD,
    )


def _master_summary(result: RestrictedMasterLPResult) -> dict[str, Any]:
    return {
        "objective": result.objective,
        "route_weight": result.route_weight,
        "artificial_total": result.artificial_total,
        "artificial_trip_count": sum(
            value > 1e-8 for value in result.artificial_values.values()
        ),
    }


def _best_incidence_costs(
    columns: Sequence[CostedColumn],
) -> dict[frozenset[int], float]:
    best: dict[frozenset[int], float] = {}
    for column in columns:
        best[column.incidence] = min(
            column.master_cost,
            best.get(column.incidence, math.inf),
        )
    return best


def run_negative_matching_waves(
    trip_ids: Sequence[int],
    base_columns: Sequence[CostedColumn],
    matching_columns: Sequence[CostedColumn],
    *,
    reduced_cost_epsilon: float = REDUCED_COST_EPSILON,
    column_cost_epsilon: float = COLUMN_COST_EPSILON,
) -> dict[str, Any]:
    """Add all improving members of one finite matching cover by dual waves."""

    columns = list(base_columns)
    best_cost = _best_incidence_costs(columns)
    waves: list[dict[str, Any]] = []
    termination = "finite_matching_cover_wave_limit_reached"

    for wave_number in range(1, len(matching_columns) + 2):
        master = _solve_master(trip_ids, columns)
        addable: list[tuple[int, CostedColumn, float]] = []
        candidate_reduced_costs: list[float] = []
        cost_improving_candidates = 0
        for candidate_index, candidate in enumerate(matching_columns):
            incumbent_cost = best_cost.get(candidate.incidence, math.inf)
            if candidate.master_cost >= incumbent_cost - column_cost_epsilon:
                continue
            cost_improving_candidates += 1
            reduced_cost = candidate.master_cost - sum(
                master.trip_duals[trip] for trip in candidate.incidence
            )
            candidate_reduced_costs.append(reduced_cost)
            if reduced_cost < -reduced_cost_epsilon:
                addable.append((candidate_index, candidate, reduced_cost))

        wave = {
            "wave": wave_number,
            "master_before_add": _master_summary(master),
            "pool_size_before_add": len(columns),
            "unseen_or_cheaper_matching_routes": cost_improving_candidates,
            "negative_matching_routes": len(addable),
            "best_candidate_reduced_cost": (
                min(candidate_reduced_costs)
                if candidate_reduced_costs
                else None
            ),
            "added": [
                {
                    "matching_route_index": candidate_index,
                    "trip_count": len(candidate.trips),
                    "trips": list(candidate.trips),
                    "master_cost": candidate.master_cost,
                    "reduced_cost_at_selected_dual": reduced_cost,
                }
                for candidate_index, candidate, reduced_cost in addable
            ],
        }
        waves.append(wave)
        if not addable:
            termination = (
                "no_negative_route_in_finite_matching_cover_at_selected_dual"
            )
            break
        for _candidate_index, candidate, _reduced_cost in addable:
            columns.append(candidate)
            best_cost[candidate.incidence] = candidate.master_cost

    final_master = _solve_master(trip_ids, columns)
    return {
        "initial_pool_size": len(base_columns),
        "matching_candidate_count": len(matching_columns),
        "waves": waves,
        "termination_reason": termination,
        "final_pool_size": len(columns),
        "final_master": _master_summary(final_master),
        "pricing_optimality_certified": False,
        "scope_warning": (
            "Heuristic exhaustion means only that no negative member of this "
            "one finite matching cover was found at the selected dual. It does "
            "not prove that the DP pricing graph has no improving column."
        ),
    }


def _bundle_matching_cover(
    base_columns: Sequence[CostedColumn],
    matching_columns: Sequence[CostedColumn],
) -> list[CostedColumn]:
    """Append matching routes absent from, or cheaper than, the base pool."""

    bundled = list(base_columns)
    best_cost = _best_incidence_costs(bundled)
    for candidate in matching_columns:
        if candidate.master_cost < (
            best_cost.get(candidate.incidence, math.inf) - COLUMN_COST_EPSILON
        ):
            bundled.append(candidate)
            best_cost[candidate.incidence] = candidate.master_cost
    return bundled


def _cost_summary(columns: Sequence[CostedColumn]) -> dict[str, Any]:
    costs = [column.master_cost for column in columns]
    return {
        "route_count": len(columns),
        "minimum": min(costs) if costs else None,
        "maximum": max(costs) if costs else None,
        "sum": sum(costs),
    }


def _matching_route_records(
    columns: Sequence[CostedColumn],
) -> list[dict[str, Any]]:
    return [
        {
            "matching_route_index": index,
            "trip_count": len(column.trips),
            "trips": list(column.trips),
            "master_cost": column.master_cost,
            "charging_activities": int(
                column.route.get("charging_activities", 0)
            ),
            "deadhead_kwh": float(column.route.get("deadhead_kwh", 0.0)),
        }
        for index, column in enumerate(columns)
    ]


def audit_matching_cover_pool(
    pool_path: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """Build and price a deterministic current-model cover for one final pool."""

    resolved_pool = pool_path.expanduser().resolve()
    with resolved_pool.open(encoding="utf-8") as handle:
        pool = json.load(handle)
    trip_ids, seed_count = validate_saved_pool(pool)
    config = matching_audit_config(pool)

    instance_path, instance_hash = resolve_saved_data_path(
        data_dir,
        pool["csv_name"],
        expected_sha256=pool.get("instance_sha256"),
        label="instance",
    )
    price_path, price_hash = resolve_saved_data_path(
        data_dir,
        pool["prices_csv"],
        expected_sha256=pool.get("price_sha256"),
        label="price",
    )
    data_root = data_dir.expanduser().resolve()
    try:
        instance_name = str(instance_path.relative_to(data_root))
    except ValueError as exc:
        raise MatchingCoverAuditInputError(
            f"Resolved instance must be inside data directory {data_root}: "
            f"{instance_path}"
        ) from exc

    problem = build_problem(
        data_root,
        instance_name,
        max_station_to_trip_wait_min=config.max_charge2trip_min,
    )
    if tuple(problem.trips) != trip_ids:
        raise MatchingCoverAuditInputError(
            "Saved trip_ids do not exactly match local trip IDs rebuilt from "
            f"{instance_path}"
        )

    station_prices = load_station_hourly_prices(
        price_path,
        tuple(STATION_NODE_BY_BASE),
    )
    hourly_prices = station_prices[base_station_name(DEPOT)]

    def charging_cost(station: Any, start_min: float, energy_kwh: float) -> float:
        curve = station_prices[base_station_name(str(station))]
        return _compute_charging_cost_accurate(
            start_min=float(start_min),
            energy_kwh=float(energy_kwh),
            charge_rate_kw=CHARGE_RATE_KW,
            hourly_prices=curve,
            charge_cost_premium=charge_cost_premium,
        )

    matching_routes = build_matching_initial_routes(
        trips=problem.trips,
        adjacency=problem.adjacency,
        depot=DEPOT,
        stations=STATIONS,
        trip_start_min=problem.start_min,
        trip_end_min=problem.end_min,
        trip_energy_kwh=problem.trip_energy,
        battery_capacity_kwh=config.effective_battery_kwh,
        charge_rate_kw=CHARGE_RATE_KW,
        soc_charge_levels=[
            config.effective_battery_kwh * index / 10.0
            for index in range(1, 11)
        ],
        horizon_min=config.horizon_min,
        max_daily_recharges=MAX_DAILY_RECHARGES,
        max_station_to_trip_wait_min=config.max_charge2trip_min,
        successor_boundary_soc_target=config.successor_charge_targets,
        max_successor_charge_targets=config.max_successor_charge_targets,
        station_waiting_unrestricted=(
            config.max_charge2trip_min >= config.horizon_min - 1e-6
        ),
        charge_start_cost=CHARGE_START_COST,
        charging_cost=charging_cost,
        # Match the production initializer exactly. The restricted master cost
        # is recomputed separately with TRAVEL_COST_FACTOR below.
        deadhead_cost_per_kwh=0.0,
        direct_only=config.direct_only,
        max_matching_attempts=config.max_matching_attempts,
        matching_order_seed=config.matching_order_seed,
    )
    cover_validation = validate_matching_cover(
        matching_routes,
        trip_ids,
        direct_only=config.direct_only,
    )

    def exact_master_cost(route: dict[str, Any]) -> float:
        return _route_cost(route, hourly_prices, station_prices)

    saved_columns = cost_columns(pool["routes"], trip_ids, exact_master_cost)
    matching_columns = cost_columns(
        matching_routes, trip_ids, exact_master_cost
    )
    seed_columns = saved_columns[:seed_count]

    seed_bundle = _bundle_matching_cover(seed_columns, matching_columns)
    full_bundle = _bundle_matching_cover(saved_columns, matching_columns)
    seed_bundle_master = _solve_master(trip_ids, seed_bundle)
    full_bundle_master = _solve_master(trip_ids, full_bundle)
    seed_waves = run_negative_matching_waves(
        trip_ids, seed_columns, matching_columns
    )
    full_waves = run_negative_matching_waves(
        trip_ids, saved_columns, matching_columns
    )
    stated_final_objective = pool.get("final_lp_obj")
    stated_final_route_weight = pool.get("final_lp_route_weight")
    recomputed_full = full_waves["waves"][0]["master_before_add"]

    return {
        "schema_version": SCHEMA_VERSION,
        "audit": "current_model_matching_cover_negative_waves",
        "pool": {
            "path": str(resolved_pool),
            "mode": pool.get("mode"),
            "saved_git": pool.get("git"),
            "seed_route_count": seed_count,
            "saved_route_count": len(saved_columns),
        },
        "inputs": {
            "instance_path": str(instance_path),
            "instance_sha256": instance_hash,
            "saved_instance_hash_present": bool(pool.get("instance_sha256")),
            "instance_hash_matches_saved": (
                instance_hash == pool["instance_sha256"]
                if pool.get("instance_sha256")
                else None
            ),
            "price_path": str(price_path),
            "price_sha256": price_hash,
            "saved_price_hash_present": bool(pool.get("price_sha256")),
            "price_hash_matches_saved": (
                price_hash == pool["price_sha256"]
                if pool.get("price_sha256")
                else None
            ),
            "trip_count": len(trip_ids),
            "current_model_only": True,
            "incumbent_assignment_dependency": False,
            "provenance_limit": (
                "The pool records hashes for the instance and price CSV only. "
                "Auxiliary deadhead/reference files and imported model constants "
                "come from the current checkout and are not proven identical to "
                "an older originating commit."
            ),
        },
        "configuration": {
            "requested_battery_kwh": config.requested_battery_kwh,
            "effective_battery_kwh": config.effective_battery_kwh,
            "charge_rate_kw": float(CHARGE_RATE_KW),
            "matching_initializer_deadhead_cost_per_kwh": 0.0,
            "master_travel_cost_factor": float(TRAVEL_COST_FACTOR),
            "horizon_min": config.horizon_min,
            "max_charge2trip_min": config.max_charge2trip_min,
            "successor_charge_targets": config.successor_charge_targets,
            "max_successor_charge_targets": config.max_successor_charge_targets,
            "direct_only": config.direct_only,
            "max_matching_attempts": config.max_matching_attempts,
            "matching_order_seed": config.matching_order_seed,
            "master_backend": "scipy_highs",
            "master_method": MASTER_METHOD,
            "artificial_penalty": float(BIG_M_PENALTY),
            "reduced_cost_epsilon": REDUCED_COST_EPSILON,
            "column_cost_epsilon": COLUMN_COST_EPSILON,
        },
        "cost_recomputation": {
            "formula": (
                "fixed bus cost plus configured deadhead-energy cost plus "
                "hour-split station-specific charging energy and one charge "
                "start cost per recorded charging activity"
            ),
            "saved_seed_routes": _cost_summary(seed_columns),
            "saved_full_pool": _cost_summary(saved_columns),
            "matching_cover": _cost_summary(matching_columns),
            "saved_master_rebuild": {
                "from_saved_seeds": seed_waves["waves"][0][
                    "master_before_add"
                ],
                "from_saved_full_pool": recomputed_full,
                "saved_final_lp_objective": stated_final_objective,
                "saved_final_lp_route_weight": stated_final_route_weight,
                "objective_difference_recomputed_minus_saved": (
                    recomputed_full["objective"] - float(stated_final_objective)
                    if stated_final_objective is not None
                    else None
                ),
                "route_weight_difference_recomputed_minus_saved": (
                    recomputed_full["route_weight"]
                    - float(stated_final_route_weight)
                    if stated_final_route_weight is not None
                    else None
                ),
            },
        },
        "matching_cover": {
            "route_count": len(matching_columns),
            "validation": cover_validation,
            "routes": _matching_route_records(matching_columns),
        },
        "negative_only_waves": {
            "from_saved_seeds": seed_waves,
            "from_saved_full_pool": full_waves,
        },
        "all_matching_routes_bundle_counterfactual": {
            "warning": (
                "This bundles every absent-or-cheaper matching route without "
                "requiring negative reduced cost; it is a diagnostic, not a "
                "column-generation step."
            ),
            "from_saved_seeds": _master_summary(seed_bundle_master),
            "from_saved_full_pool": _master_summary(full_bundle_master),
        },
        "interpretation": {
            "pricing_optimality_certified": False,
            "heuristic_scope": (
                "Only one deterministic, finite matching cover is tested. A "
                "no-negative termination does not exhaust the current DP "
                "pricing graph."
            ),
            "checkout_scope": (
                "The matching graph, auxiliary deadhead/reference inputs, and "
                "model constants are those in the checkout running this audit."
            ),
            "dual_degeneracy_warning": (
                "Reduced costs are evaluated at one optimal SciPy/HiGHS dual "
                "solution. Individual matching-route reduced costs may change "
                "across equally optimal dual vectors."
            ),
        },
    }


def write_json_report(report: Mapping[str, Any], path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        required=True,
        help="One saved routes_colgen_final_*.json file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Tracked data directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file; the identical JSON is always printed to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit_matching_cover_pool(args.pool, args.data_dir)
    if args.output:
        write_json_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

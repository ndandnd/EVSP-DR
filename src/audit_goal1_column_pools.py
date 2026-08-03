"""Merge saved Goal-1 column pools and re-solve their union master LP.

This audit is deliberately separate from the column-generation runner.  It
does not generate, repair, or otherwise change routes.  It verifies the saved
instance/trip/battery/price identity and every model-action field recorded by
the pools, reports any missing historical fields, deduplicates columns by their
trip-cover incidence, retains the cheapest realization under the current
checkout's master-cost function, and solves the resulting restricted master
with SciPy/HiGHS.

Example
-------
python src/audit_goal1_column_pools.py \
    run_a/routes_colgen_final_*.json \
    run_b/routes_colgen_final_*.json \
    --output union_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence

from config import (
    BIG_M_PENALTY,
    BUS_COST_KX,
    CHARGE_RATE_KW,
    CHARGE_START_COST,
    DEPOT_NAME,
    TRAVEL_COST_FACTOR,
)
from master_lp_scipy import build_route_incidence, solve_restricted_master_lp
from utils_v2 import (
    base_station_name,
    calculate_truck_route_cost_accurate,
    load_station_hourly_prices,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# These affect route feasibility or its master objective.  Search policy
# settings such as queue_order, dominance_mode, and kbest are intentionally
# *not* compared: combining pools from different search policies is the point
# of this audit.
MODEL_CRITICAL_LOCATIONS: dict[str, tuple[tuple[str, ...], ...]] = {
    "max_trip2trip": (("max_trip2trip",), ("run_arguments", "max_trip2trip")),
    "max_charge2trip": (("max_charge2trip",), ("run_arguments", "max_charge2trip")),
    "successor_charge_targets": (
        ("successor_charge_targets",),
        ("run_arguments", "successor_charge_targets"),
    ),
    "max_successor_charge_targets": (
        ("max_successor_charge_targets",),
        ("run_arguments", "max_successor_charge_targets"),
    ),
    "horizon_min": (("horizon_min",), ("run_arguments", "horizon_min")),
    "charge_rate_kw": (("charge_rate_kw",), ("run_arguments", "charge_rate_kw")),
    "charge_start_cost": (
        ("charge_start_cost",),
        ("run_arguments", "charge_start_cost"),
    ),
    "bus_cost_kx": (("bus_cost_kx",), ("run_arguments", "bus_cost_kx")),
    "travel_cost_factor": (
        ("travel_cost_factor",),
        ("run_arguments", "travel_cost_factor"),
    ),
    "artificial_penalty": (
        ("artificial_penalty",),
        ("run_arguments", "artificial_penalty"),
    ),
}

# Pools created before the trip-gap option was exposed all used the runner's
# hard-coded 57-minute value.  Treat an omitted historical field as that known
# legacy value so a relaxed-gap pool can never be silently merged with it.
LEGACY_MODEL_CRITICAL_DEFAULTS = {
    "max_trip2trip": 57.0,
}

CURRENT_COST_CONFIG = {
    "bus_cost_kx": float(BUS_COST_KX),
    "charge_rate_kw": float(CHARGE_RATE_KW),
    "charge_start_cost": float(CHARGE_START_COST),
    "travel_cost_factor": float(TRAVEL_COST_FACTOR),
    "artificial_penalty": float(BIG_M_PENALTY),
}


class ColumnPoolAuditError(ValueError):
    """Raised when saved pools cannot safely be combined."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested_value(mapping: dict[str, Any], location: Sequence[str]) -> Any:
    value: Any = mapping
    for key in location:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _one_consistent_value(
    pool: dict[str, Any],
    *,
    pool_id: str,
    field: str,
    locations: Sequence[Sequence[str]],
) -> Any:
    observed = [
        (".".join(location), _nested_value(pool, location))
        for location in locations
        if _nested_value(pool, location) is not None
    ]
    if not observed:
        return None
    first = observed[0][1]
    disagreements = [(location, value) for location, value in observed if value != first]
    if disagreements:
        raise ColumnPoolAuditError(
            f"{pool_id} records inconsistent {field!r} values: {observed}"
        )
    return first


def _load_pool(path: Path, index: int) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ColumnPoolAuditError(f"{resolved} must contain one JSON object")

    pool_id = f"pool_{index + 1:03d}"
    required = ("instance_sha256", "trip_ids", "battery_kwh", "price_sha256", "routes")
    missing = [field for field in required if value.get(field) is None]
    if missing:
        raise ColumnPoolAuditError(
            f"{pool_id} ({resolved}) is missing required fields: {missing}"
        )
    if not isinstance(value["routes"], list):
        raise ColumnPoolAuditError(f"{pool_id} routes must be a JSON list")
    if not isinstance(value["trip_ids"], list) or not value["trip_ids"]:
        raise ColumnPoolAuditError(f"{pool_id} trip_ids must be a nonempty JSON list")
    try:
        unique_trip_count = len(set(value["trip_ids"]))
    except TypeError as exc:
        raise ColumnPoolAuditError(f"{pool_id} trip_ids must be hashable") from exc
    if unique_trip_count != len(value["trip_ids"]):
        raise ColumnPoolAuditError(f"{pool_id} trip_ids contains duplicates")

    declared_count = value.get("num_routes")
    if declared_count is not None and declared_count != len(value["routes"]):
        raise ColumnPoolAuditError(
            f"{pool_id} declares num_routes={declared_count}, "
            f"but contains {len(value['routes'])} routes"
        )

    seed_count = value.get("seed_route_count")
    if seed_count is not None:
        if isinstance(seed_count, bool) or not isinstance(seed_count, int):
            raise ColumnPoolAuditError(f"{pool_id} seed_route_count must be an integer")
        if not 0 <= seed_count <= len(value["routes"]):
            raise ColumnPoolAuditError(
                f"{pool_id} seed_route_count={seed_count} lies outside "
                f"[0, {len(value['routes'])}]"
            )
        declared_dp = value.get("dp_columns_generated")
        if declared_dp is not None and declared_dp != len(value["routes"]) - seed_count:
            raise ColumnPoolAuditError(
                f"{pool_id} declares dp_columns_generated={declared_dp}, but "
                f"route_count - seed_route_count={len(value['routes']) - seed_count}"
            )

    # battery_kwh is duplicated as run_arguments.G in current saved pools.
    # Reject an internally inconsistent pool before comparing pools.
    saved_g = _nested_value(value, ("run_arguments", "G"))
    if saved_g is not None and saved_g != value["battery_kwh"]:
        raise ColumnPoolAuditError(
            f"{pool_id} battery_kwh={value['battery_kwh']!r} disagrees with "
            f"run_arguments.G={saved_g!r}"
        )

    value["_audit_pool_id"] = pool_id
    value["_audit_path"] = resolved
    value["_audit_sha256"] = _sha256(resolved)
    return value


def _validate_identity(pools: Sequence[dict[str, Any]]) -> dict[str, Any]:
    identity_fields = ("instance_sha256", "trip_ids", "battery_kwh", "price_sha256")
    reference = pools[0]
    for pool in pools[1:]:
        mismatches = {
            field: {
                reference["_audit_pool_id"]: reference[field],
                pool["_audit_pool_id"]: pool[field],
            }
            for field in identity_fields
            if pool[field] != reference[field]
        }
        if mismatches:
            raise ColumnPoolAuditError(
                "saved pools describe different mathematical instances: "
                f"{mismatches}"
            )

    critical_report: dict[str, Any] = {}
    for field, locations in MODEL_CRITICAL_LOCATIONS.items():
        values: list[tuple[str, Any]] = []
        missing: list[str] = []
        assumed_legacy_default: list[str] = []
        for pool in pools:
            pool_id = pool["_audit_pool_id"]
            value = _one_consistent_value(
                pool,
                pool_id=pool_id,
                field=field,
                locations=locations,
            )
            if value is None:
                if field in LEGACY_MODEL_CRITICAL_DEFAULTS:
                    value = LEGACY_MODEL_CRITICAL_DEFAULTS[field]
                    values.append((pool_id, value))
                    assumed_legacy_default.append(pool_id)
                else:
                    missing.append(pool_id)
            else:
                values.append((pool_id, value))
        if values:
            first = values[0][1]
            if any(value != first for _, value in values[1:]):
                raise ColumnPoolAuditError(
                    f"saved pools disagree on model-critical {field!r}: {values}"
                )
            if field in CURRENT_COST_CONFIG and first != CURRENT_COST_CONFIG[field]:
                raise ColumnPoolAuditError(
                    f"saved {field}={first!r} does not match the current master-cost "
                    f"configuration {CURRENT_COST_CONFIG[field]!r}"
                )
            critical_report[field] = {
                "value": first,
                "present_in_pool_ids": [
                    pool_id
                    for pool_id, _ in values
                    if pool_id not in assumed_legacy_default
                ],
                "assumed_legacy_default_in_pool_ids": assumed_legacy_default,
                "missing_from_pool_ids": missing,
            }

    return {
        "instance_sha256": reference["instance_sha256"],
        "trip_ids": list(reference["trip_ids"]),
        "battery_kwh": reference["battery_kwh"],
        "price_sha256": reference["price_sha256"],
        "model_critical_config_where_recorded": critical_report,
        "validation_limit": (
            "Historical pools did not necessarily save every model/cost constant. "
            "Recorded fields are compared; omitted constants are supplied by and "
            "reported from the current checkout when route costs are recomputed."
        ),
        "allowed_to_differ": [
            "git",
            "mode",
            "master_backend",
            "queue_order",
            "pricing_output_selection",
            "dominance_mode",
            "kbest",
            "max_labels",
            "pricing time limits",
        ],
    }


def _path_candidates(raw_name: Any, data_dir: Path) -> list[Path]:
    if raw_name is None:
        return []
    raw = Path(str(raw_name)).expanduser()
    candidates: list[Path] = [raw] if raw.is_absolute() else [Path.cwd() / raw]
    if not raw.is_absolute():
        candidates.append(data_dir / raw)

    parts = raw.parts
    if "data" in parts:
        data_index = len(parts) - 1 - tuple(reversed(parts)).index("data")
        suffix = parts[data_index + 1 :]
        if suffix:
            candidates.append(data_dir.joinpath(*suffix))
    candidates.append(data_dir / raw.name)
    if data_dir.exists():
        candidates.extend(data_dir.rglob(raw.name))

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _resolve_hashed_file(
    raw_names: Iterable[Any],
    *,
    data_dir: Path,
    expected_sha256: str,
    label: str,
    required: bool,
) -> tuple[Path | None, bool]:
    existing: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for raw_name in raw_names:
        for candidate in _path_candidates(raw_name, data_dir):
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            actual = _sha256(candidate)
            existing.append((candidate, actual))
            if actual == expected_sha256:
                return candidate, True
    if existing:
        rendered = {str(path): digest for path, digest in existing}
        raise ColumnPoolAuditError(
            f"no resolved {label} file matches saved SHA-256 {expected_sha256}: "
            f"{rendered}"
        )
    if required:
        raise FileNotFoundError(
            f"could not resolve {label} from saved paths under {data_dir}"
        )
    return None, False


def _route_trip_ids(
    route: dict[str, Any],
    *,
    trip_ids: Sequence[Hashable],
    pool_id: str,
    route_index: int,
) -> tuple[Hashable, ...]:
    route_nodes = route.get("route")
    if not isinstance(route_nodes, list):
        raise ColumnPoolAuditError(
            f"{pool_id} route {route_index} has no JSON-list route field"
        )
    trip_set = set(trip_ids)
    observed = [node for node in route_nodes if node in trip_set]
    if not observed:
        raise ColumnPoolAuditError(
            f"{pool_id} route {route_index} contains no active trip"
        )
    if len(set(observed)) != len(observed):
        raise ColumnPoolAuditError(
            f"{pool_id} route {route_index} repeats an active trip"
        )
    observed_set = set(observed)
    # Incidence identity follows trip_ids order, not route traversal order, so
    # equal set-cover columns have one deterministic key.
    return tuple(trip for trip in trip_ids if trip in observed_set)


def _route_cost(
    route: dict[str, Any],
    *,
    hourly_prices: dict[int, float],
    station_prices: dict[str, dict[int, float]],
) -> float:
    if route.get("dummy", False):
        cost = float(route.get("dummy_cost", 1e7))
    else:
        cost = calculate_truck_route_cost_accurate(
            route,
            BUS_COST_KX,
            hourly_prices,
            charge_rate_kw=CHARGE_RATE_KW,
            travel_cost_factor=TRAVEL_COST_FACTOR,
            station_hourly_prices=station_prices,
            charge_start_cost=CHARGE_START_COST,
        )
    if not math.isfinite(cost) or cost < 0:
        raise ColumnPoolAuditError(
            f"master route cost must be finite and nonnegative: {cost}"
        )
    return float(cost)


def _origin(route_index: int, seed_route_count: int | None) -> str:
    if seed_route_count is None:
        return "unknown"
    return "seed" if route_index < seed_route_count else "dp"


def _source_pool_groups(occurrences: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for occurrence in occurrences:
        pool_id = occurrence["pool_id"]
        group = grouped.setdefault(
            pool_id,
            {"pool_id": pool_id, "origins": set(), "route_indices": []},
        )
        group["origins"].add(occurrence["origin"])
        group["route_indices"].append(occurrence["route_index"])
    return [
        {
            "pool_id": pool_id,
            "origins": sorted(group["origins"]),
            "route_indices": sorted(group["route_indices"]),
        }
        for pool_id, group in grouped.items()
    ]


def audit_column_pools(
    pool_paths: Sequence[Path],
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    method: str = "highs-ds",
    activity_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Validate, merge, deduplicate, and solve saved Goal-1 route pools."""

    if not pool_paths:
        raise ColumnPoolAuditError("at least one saved pool path is required")
    tolerance = float(activity_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ColumnPoolAuditError("activity_tolerance must be positive and finite")

    pools = [_load_pool(path, index) for index, path in enumerate(pool_paths)]
    identity = _validate_identity(pools)
    trip_ids = tuple(identity["trip_ids"])
    data_dir = data_dir.expanduser().resolve()

    price_path, price_verified = _resolve_hashed_file(
        (pool.get("prices_csv") for pool in pools),
        data_dir=data_dir,
        expected_sha256=identity["price_sha256"],
        label="price CSV",
        required=True,
    )
    assert price_path is not None
    instance_path, instance_verified = _resolve_hashed_file(
        (pool.get("csv_name") for pool in pools),
        data_dir=data_dir,
        expected_sha256=identity["instance_sha256"],
        label="instance CSV",
        required=False,
    )

    station_bases = {base_station_name(DEPOT_NAME)}
    for pool in pools:
        for route in pool["routes"]:
            stops = route.get("charging_stops") or {}
            station_bases.update(
                base_station_name(station) for station in stops.get("stations", [])
            )
    station_prices = load_station_hourly_prices(price_path, sorted(station_bases))
    depot_base = base_station_name(DEPOT_NAME)
    if depot_base not in station_prices:
        raise ColumnPoolAuditError(
            f"price CSV does not provide the depot price curve {depot_base!r}"
        )
    hourly_prices = station_prices[depot_base]

    union_by_incidence: dict[tuple[Hashable, ...], dict[str, Any]] = {}
    dp_covered: set[Hashable] = set()
    occurrence_counts = {"seed": 0, "dp": 0, "unknown": 0}
    input_reports: list[dict[str, Any]] = []

    for pool in pools:
        pool_id = pool["_audit_pool_id"]
        seed_count = pool.get("seed_route_count")
        for route_index, route in enumerate(pool["routes"]):
            trip_incidence = _route_trip_ids(
                route,
                trip_ids=trip_ids,
                pool_id=pool_id,
                route_index=route_index,
            )
            cost = _route_cost(
                route,
                hourly_prices=hourly_prices,
                station_prices=station_prices,
            )
            origin = _origin(route_index, seed_count)
            occurrence_counts[origin] += 1
            if origin == "dp":
                dp_covered.update(trip_incidence)
            occurrence = {
                "pool_id": pool_id,
                "route_index": route_index,
                "origin": origin,
                "cost": cost,
            }
            record = union_by_incidence.get(trip_incidence)
            if record is None:
                union_by_incidence[trip_incidence] = {
                    "trip_ids": trip_incidence,
                    "route": route,
                    "cost": cost,
                    "retained_source": occurrence,
                    "source_occurrences": [occurrence],
                }
            else:
                record["source_occurrences"].append(occurrence)
                if cost < record["cost"]:
                    record["route"] = route
                    record["cost"] = cost
                    record["retained_source"] = occurrence

        run_arguments = pool.get("run_arguments") or {}
        input_reports.append({
            "pool_id": pool_id,
            "path": str(pool["_audit_path"]),
            "file_sha256": pool["_audit_sha256"],
            "route_count": len(pool["routes"]),
            "seed_route_count": seed_count,
            "dp_route_count": (
                len(pool["routes"]) - seed_count if seed_count is not None else None
            ),
            "classification_complete": seed_count is not None,
            "mode": pool.get("mode"),
            "queue_order": pool.get("queue_order", run_arguments.get("queue_order")),
            "pricing_output_selection": pool.get(
                "pricing_output_selection", run_arguments.get("pricing_output_selection")
            ),
            "dominance_mode": pool.get(
                "dominance_mode", run_arguments.get("dominance_mode")
            ),
            "termination_reason": pool.get("termination_reason"),
            "git": pool.get("git"),
            "saved_final_lp": {
                "objective": pool.get("final_lp_obj"),
                "route_weight": pool.get("final_lp_route_weight"),
                "artificial_total": pool.get("final_lp_artificial_total"),
                "artificial_trip_count": pool.get("final_lp_artificial_trips"),
            },
        })

    union_columns = list(union_by_incidence.values())
    route_trip_ids = [column["trip_ids"] for column in union_columns]
    route_costs = [column["cost"] for column in union_columns]
    incidence = build_route_incidence(trip_ids, route_trip_ids)
    master = solve_restricted_master_lp(
        trip_ids=trip_ids,
        route_incidence=incidence,
        route_costs=route_costs,
        artificial_penalty=BIG_M_PENALTY,
        method=method,
    )

    active_columns = []
    for index, (column, value) in enumerate(zip(union_columns, master.route_values)):
        if value <= tolerance:
            continue
        retained = column["retained_source"]
        occurrences = column["source_occurrences"]
        active_columns.append({
            "union_column_index": index,
            "lp_value": value,
            "trip_ids": list(column["trip_ids"]),
            "master_cost": column["cost"],
            "retained_from": {
                "pool_id": retained["pool_id"],
                "route_index": retained["route_index"],
                "origin": retained["origin"],
            },
            "source_pool_ids": list(dict.fromkeys(
                occurrence["pool_id"] for occurrence in occurrences
            )),
            "source_origins": sorted({
                occurrence["origin"] for occurrence in occurrences
            }),
            "source_pools": _source_pool_groups(occurrences),
        })

    artificial_trip_ids = [
        trip for trip in trip_ids if master.artificial_values[trip] > tolerance
    ]
    dp_covered_ordered = [trip for trip in trip_ids if trip in dp_covered]
    dp_missing = [trip for trip in trip_ids if trip not in dp_covered]
    total_columns = sum(len(pool["routes"]) for pool in pools)
    unique_with_origin = {
        origin: sum(
            any(source["origin"] == origin for source in column["source_occurrences"])
            for column in union_columns
        )
        for origin in ("seed", "dp", "unknown")
    }

    identity["price_csv"] = {
        "path": str(price_path),
        "sha256_verified": price_verified,
    }
    identity["instance_csv"] = {
        "path": str(instance_path) if instance_path else None,
        "sha256_verified": instance_verified,
        "note": (
            None
            if instance_path
            else "Instance file was not needed for the union LP and could not be resolved locally."
        ),
    }

    return {
        "audit": "goal1_column_pool_union",
        "audit_version": 1,
        "scope": {
            "does": (
                "Recompute current master costs from saved route metadata, merge trip-cover "
                "incidences, and solve the union restricted-master LP."
            ),
            "does_not": (
                "Re-run full path time/SOC/charging resource feasibility for every saved "
                "route; this report trusts the saved route construction after provenance "
                "and pricing-action compatibility checks."
            ),
            "provenance_limit": (
                "A saved pool may omit historical cost constants. Such constants are not "
                "claimed verified against the original run; the union is an explicit "
                "current-checkout cost recomputation."
            ),
        },
        "input_pools": input_reports,
        "validated_identity": identity,
        "master_cost_config": CURRENT_COST_CONFIG,
        "columns": {
            "total_input_columns": total_columns,
            "unique_trip_incidences": len(union_columns),
            "duplicate_incidences_removed": total_columns - len(union_columns),
            "input_occurrences_by_origin": occurrence_counts,
            "unique_incidences_with_origin": unique_with_origin,
        },
        "dp_trip_coverage": {
            "basis": "all input route occurrences classified as DP before incidence deduplication",
            "classification_complete": occurrence_counts["unknown"] == 0,
            "covered_count": len(dp_covered_ordered),
            "missing_count": len(dp_missing),
            "covered_trip_ids": dp_covered_ordered,
            "missing_trip_ids": dp_missing,
        },
        "restricted_master": {
            "objective": master.objective,
            "route_weight": master.route_weight,
            "artificial_total": master.artificial_total,
            "artificial_count": len(artificial_trip_ids),
            "artificial_trip_ids": artificial_trip_ids,
            "activity_tolerance": tolerance,
            "status": master.status,
            "runtime_s": master.runtime_s,
            "solver": master.backend.solver,
            "method": master.backend.method,
            "scipy_version": master.backend.scipy_version,
        },
        "active_columns": active_columns,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pools",
        type=Path,
        nargs="+",
        help="Two or more (or one) saved routes_colgen_final_*.json files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Data directory used to resolve saved CSV paths (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--method",
        choices=("highs", "highs-ds", "highs-ipm"),
        default="highs-ds",
        help="SciPy/HiGHS LP method (default: highs-ds).",
    )
    parser.add_argument(
        "--activity-tolerance",
        type=float,
        default=1e-8,
        help="Threshold for reporting active route/artificial variables (default: 1e-8).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path receiving the same JSON report printed to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_column_pools(
        args.pools,
        data_dir=args.data_dir,
        method=args.method,
        activity_tolerance=args.activity_tolerance,
    )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().resolve().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

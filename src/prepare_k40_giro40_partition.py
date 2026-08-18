#!/usr/bin/env python3
"""Build the verified 40-duty GIRO partition for the frozen k40 instance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

from audit_giro_known_columns import HORIZON_MIN, build_problem
from config import (
    BUS_COST_KX,
    CHARGE_START_COST,
    CHARGING_STATIONS,
)
from expanded_path_realization import (
    BLOCK_SCHEDULE_SCHEMA,
    blocks_from_continuous_stops,
    validate_continuous_charging_blocks,
)
from make_giro_seed_routes import load_master
from rerealize_routes import _arc_map, rerealize_route
from run_exact_pool_mip import (
    charging_stop_arrivals,
    validate_injected_route,
)
from utils_v2 import (
    calculate_truck_route_cost_accurate,
    load_station_hourly_prices,
)


SCHEMA = "evsp-dr-k40-giro40-partition-v1"
EXPECTED_MASTER_SHA256 = (
    "6b46acce8b0870aff967c73aac372b90873ed32a6e424e55b851e4b8676ab57f"
)
EXPECTED_INSTANCE_SHA256 = (
    "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd"
)
EXPECTED_TARIFF_SHA256 = (
    "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200"
)
EXPECTED_TRIP_SET_SHA256 = (
    "35604b22facf1646963e85eb98a858906f0dd7dbebd86ea0d3ac7b797de62ed0"
)
INCLUDED_DUTIES = tuple(
    [str(value) for value in range(13301, 13316)]
    + ["13316uwt"]
    + [str(value) for value in range(13317, 13324)]
    + ["13324t", "13325", "13326"]
    + [str(value) for value in range(13401, 13415)]
)
EXCLUDED_VARIANTS = {
    "13316m": "weekday duplicate of included 13316uwt",
    "13324muw": "weekday duplicate of included 13324t",
}


def validate_duty_selection(
    duties: list[str], literal_duties: set[str]
) -> dict[str, str]:
    if len(duties) != 40 or len(set(duties)) != 40:
        raise ValueError("GIRO partition must contain exactly 40 unique duties")
    if set(duties) != set(INCLUDED_DUTIES):
        raise ValueError(f"frozen instance duties differ: {sorted(duties)}")
    excluded = literal_duties - set(duties)
    if excluded != set(EXCLUDED_VARIANTS):
        raise ValueError(f"unexpected excluded GIRO variants: {excluded}")
    return dict(EXCLUDED_VARIANTS)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def prepare(
    *,
    status_paths: list[Path],
    master_path: Path,
    reference_data_dir: Path,
    output_path: Path,
    expected_status_sha256: list[str],
) -> dict:
    if os.path.lexists(output_path):
        raise FileExistsError(f"partition output exists: {output_path}")
    if (
        not status_paths
        or len(status_paths) != len(expected_status_sha256)
        or len(set(expected_status_sha256)) != len(expected_status_sha256)
    ):
        raise ValueError("distinct status paths/hashes are required")
    status_paths = [
        path.expanduser().resolve() for path in status_paths
    ]
    master_path = master_path.expanduser().resolve()
    reference_data_dir = reference_data_dir.expanduser().resolve()
    statuses = []
    for path, expected in zip(status_paths, expected_status_sha256):
        if _sha(path) != expected:
            raise ValueError("status SHA-256 mismatch")
        statuses.append(json.loads(path.read_text()))
    if _sha(master_path) != EXPECTED_MASTER_SHA256:
        raise ValueError("GIRO master SHA-256 mismatch")
    status = statuses[0]
    status_path = status_paths[0]
    status_identity_fields = (
        "csv", "prices_csv", "g_kwh", "charge_kw", "min_soc_frac",
        "soc_step", "block_min",
    )
    if any(
        any(other.get(key) != status.get(key)
            for key in status_identity_fields)
        for other in statuses[1:]
    ):
        raise ValueError("R1/R2 status model identities differ")
    data_dir = status_path.parent / "data"
    instance_path = data_dir / status["csv"]
    tariff_path = data_dir / Path(status["prices_csv"]).name
    provenance = status.get("provenance") or {}
    if (
        _sha(instance_path) != EXPECTED_INSTANCE_SHA256
        or provenance.get("instance_sha256") != EXPECTED_INSTANCE_SHA256
        or _sha(tariff_path) != EXPECTED_TARIFF_SHA256
        or provenance.get("prices_sha256") != EXPECTED_TARIFF_SHA256
    ):
        raise ValueError("frozen instance/tariff identity mismatch")
    for other_path, other in zip(status_paths[1:], statuses[1:]):
        other_data = other_path.parent / "data"
        other_instance = other_data / other["csv"]
        other_tariff = other_data / Path(other["prices_csv"]).name
        other_provenance = other.get("provenance") or {}
        if (
            _sha(other_instance) != EXPECTED_INSTANCE_SHA256
            or _sha(other_tariff) != EXPECTED_TARIFF_SHA256
            or other_provenance.get("instance_sha256")
            != EXPECTED_INSTANCE_SHA256
            or other_provenance.get("prices_sha256")
            != EXPECTED_TARIFF_SHA256
        ):
            raise ValueError("cross-check status data identity mismatch")
    instance = pd.read_csv(instance_path)
    if len(instance) != 947:
        raise ValueError("frozen instance does not contain 947 trips")
    trip_ids = list(range(len(instance)))
    if hashlib.sha256(_canonical(trip_ids)).hexdigest() != (
        EXPECTED_TRIP_SET_SHA256
    ):
        raise ValueError("frozen trip-set SHA-256 mismatch")

    # load_master uses the tracked authoritative path; independently prove it
    # is byte-identical to the explicit source before relying on its parser.
    master = load_master()
    if _sha(Path(__file__).resolve().parents[1] / "data"
            / "Par_VehicleDetails_Updated.csv") != EXPECTED_MASTER_SHA256:
        raise ValueError("tracked GIRO master differs from explicit source")
    regular = master[
        (master["Identifier"] == "Regular")
        & master["Ordered_Trip_ID"].notna()
    ].copy()
    regular["Ordered_Trip_ID"] = regular["Ordered_Trip_ID"].astype(int)
    duty_of = dict(zip(
        regular["Ordered_Trip_ID"],
        regular["VehicleTask"].astype(str),
    ))
    ordered_to_local = {
        int(ordered): local
        for local, ordered in enumerate(instance["Ordered_Trip_ID"])
    }
    missing = [
        ordered for ordered in ordered_to_local if ordered not in duty_of
    ]
    if missing:
        raise ValueError(f"frozen instance has foreign trips: {missing[:20]}")
    duties = sorted({duty_of[ordered] for ordered in ordered_to_local})
    literal_duties = set(master["VehicleTask"].astype(str).unique())
    validate_duty_selection(duties, literal_duties)

    g_kwh = float(status["g_kwh"])
    charge_kw = float(status["charge_kw"])
    reserve_kwh = float(status["min_soc_frac"]) * g_kwh
    if (g_kwh, charge_kw, reserve_kwh) != (300.0, 300.0, 0.0):
        raise ValueError("partition physics differ from frozen model")
    problem = build_problem(
        data_dir,
        status["csv"],
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=reference_data_dir,
    )
    arc = _arc_map(problem)
    prices = load_station_hourly_prices(
        tariff_path, CHARGING_STATIONS
    )
    depot_curve = prices.get("PARX") or next(iter(prices.values()))
    routes = []
    coverage = Counter()
    for duty in duties:
        ordered = [
            value for value, owner in duty_of.items()
            if owner == duty and value in ordered_to_local
        ]
        ordered.sort(key=lambda value: ordered_to_local[value])
        trip_sequence = [ordered_to_local[value] for value in ordered]
        if not trip_sequence:
            raise ValueError(f"GIRO duty has no frozen trips: {duty}")
        route, _planned_cost, reason = rerealize_route(
            trip_sequence,
            problem,
            arc,
            prices,
            g_kwh,
            charge_kw,
            reserve_kwh,
        )
        if route is None:
            raise ValueError(f"GIRO duty {duty} infeasible: {reason}")
        candidate = {
            **route,
            "trips": trip_sequence,
            "route_nodes": route["route"],
        }
        replay_reason = validate_injected_route(
            problem,
            candidate,
            g_kwh,
            charge_kw,
            reserve_kwh,
            HORIZON_MIN,
            arc_map=arc,
        )
        if replay_reason is not None:
            raise ValueError(
                f"GIRO duty {duty} replay failed: {replay_reason}"
            )
        cost = calculate_truck_route_cost_accurate(
            candidate,
            BUS_COST_KX,
            depot_curve,
            charge_rate_kw=charge_kw,
            station_hourly_prices=prices,
            charge_start_cost=CHARGE_START_COST,
        )
        blocks = blocks_from_continuous_stops(
            candidate,
            station_prices=prices,
            charge_kw=charge_kw,
            earliest_start_by_stop=charging_stop_arrivals(
                problem, candidate
            ),
        )
        validation = validate_continuous_charging_blocks(
            candidate,
            blocks,
            station_prices=prices,
            charge_kw=charge_kw,
            expected_continuous_cost=cost,
        )
        candidate.update({
            "cost": float(cost),
            "continuous_realized_cost": float(cost),
            "master_cost_semantics": "continuous_realized_cost",
            "continuous_realized_charging_blocks": blocks,
            "physical_realization": {
                "status": "validated_continuous_injection",
                "continuous_realized_charging_blocks_schema":
                    BLOCK_SCHEDULE_SCHEMA,
                "continuous_realized_charging_blocks_sha256":
                    validation["block_schedule_sha256"],
                "continuous_cost_pricing_certified": False,
            },
            "giro_duty_id": duty,
        })
        coverage.update(trip_sequence)
        routes.append(candidate)
    if (
        len(routes) != 40
        or set(coverage) != set(trip_ids)
        or any(coverage[trip] != 1 for trip in trip_ids)
    ):
        raise ValueError("GIRO routes are not an exact 40-duty partition")
    route_hashes = [
        hashlib.sha256(_canonical({
            "duty": route["giro_duty_id"],
            "trips": route["trips"],
            "route": route["route"],
            "charging_stops": route["charging_stops"],
            "blocks": route["continuous_realized_charging_blocks"],
            "cost": route["cost"],
        })).hexdigest()
        for route in routes
    ]
    payload = {
        "schema": SCHEMA,
        "routes": routes,
        "source": "GIRO40-AUGMENTED",
        "instance_csv": status["csv"],
        "instance_sha256": EXPECTED_INSTANCE_SHA256,
        "trip_ids": trip_ids,
        "trip_set_sha256": EXPECTED_TRIP_SET_SHA256,
        "prices_csv": status["prices_csv"],
        "tariff_sha256": EXPECTED_TARIFF_SHA256,
        "physics": {
            "g_kwh": g_kwh,
            "charge_kw": charge_kw,
            "reserve_frac": 0.0,
            "soc_step": status["soc_step"],
            "block_min": status["block_min"],
        },
        "included_duties": duties,
        "excluded_variant_ids": EXCLUDED_VARIANTS,
        "literal_giro_duty_count": len(literal_duties),
        "base_duty_count": 40,
        "route_count": len(routes),
        "route_hashes": route_hashes,
        "route_set_sha256": hashlib.sha256(
            _canonical(route_hashes)
        ).hexdigest(),
        "continuous_cost_pricing_certified": False,
        "pricing_certificate_scope": "none_for_augmented_routes",
        "provenance": {
            "source_status_names": sorted(
                path.name for path in status_paths
            ),
            "source_status_sha256": sorted(expected_status_sha256),
            "giro_master_name": master_path.name,
            "giro_master_sha256": EXPECTED_MASTER_SHA256,
            "reference_sha256": _sha(
                reference_data_dir / "Ref_dict.csv"
            ),
            "deadhead_sha256": _sha(
                reference_data_dir / "par_ref_dhd.csv"
            ),
        },
    }
    payload["partition_sha256"] = hashlib.sha256(
        _canonical({
            key: value for key, value in payload.items()
            if key != "partition_sha256"
        })
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, action="append", required=True)
    parser.add_argument(
        "--status-sha256", action="append", required=True
    )
    parser.add_argument("--giro-master", type=Path, required=True)
    parser.add_argument(
        "--reference-data-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = prepare(
        status_paths=args.status,
        master_path=args.giro_master,
        reference_data_dir=args.reference_data_dir,
        output_path=args.out,
        expected_status_sha256=args.status_sha256,
    )
    print(json.dumps({
        "partition_sha256": payload["partition_sha256"],
        "route_set_sha256": payload["route_set_sha256"],
        "route_count": payload["route_count"],
        "trip_count": len(payload["trip_ids"]),
        "included_duties": payload["included_duties"],
        "excluded_variant_ids": payload["excluded_variant_ids"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

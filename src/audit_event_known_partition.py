#!/usr/bin/env python3
"""Construct a physical industrial-partition witness in an event route space."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

from audit_giro_known_columns import DEPOT, HORIZON_MIN, STATIONS, build_problem
from config import CHARGING_STATIONS
from durable_io import atomic_write_json
from event_pricer_network import EventExpandedNetwork
from tariff_response_core import giro_routes_for_instance
from utils_v2 import load_station_hourly_prices


SCHEMA = "evsp-dr-event-known-partition-model-witness-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _restricted_problem(problem, trips):
    trip_set = set(trips)
    station_set = set(STATIONS)
    adjacency = {}
    for source, arcs in problem.adjacency.items():
        if (
            source not in trip_set
            and source not in station_set
            and source != DEPOT
        ):
            continue
        retained = [
            arc for arc in arcs
            if (
                arc[0] == DEPOT
                or arc[0] in trip_set
                or arc[0] in station_set
            )
        ]
        if retained:
            adjacency[source] = retained
    return replace(
        problem,
        trips=tuple(trips),
        adjacency=adjacency,
        start_min={trip: problem.start_min[trip] for trip in trips},
        end_min={trip: problem.end_min[trip] for trip in trips},
        trip_energy={trip: problem.trip_energy[trip] for trip in trips},
    )


def audit(args) -> dict:
    status_path = Path(args.status).expanduser().resolve(strict=True)
    status_bytes = status_path.read_bytes()
    status = json.loads(status_bytes)
    if status.get("time_model") != "event":
        raise ValueError("known-partition witness requires event time model")
    data_dir = Path(args.data_dir).expanduser().resolve()
    instance = (data_dir / status["csv"]).resolve(strict=True)
    tariff = (data_dir / status["prices_csv"]).resolve(strict=True)
    provenance = status.get("provenance") or {}
    if (
        _sha256(instance) != provenance.get("instance_sha256")
        or _sha256(tariff) != provenance.get("prices_sha256")
    ):
        raise ValueError("status input hashes do not match known witness inputs")
    problem = build_problem(
        data_dir,
        status["csv"],
        max_station_to_trip_wait_min=HORIZON_MIN,
    )
    prices = load_station_hourly_prices(tariff, CHARGING_STATIONS)
    known = giro_routes_for_instance(
        data_dir / "Par_VehicleDetails_Updated.csv",
        instance,
    )
    records = []
    counts = Counter()
    for route in known:
        trips = list(route["trips"])
        restricted = _restricted_problem(problem, trips)
        network = EventExpandedNetwork(
            restricted,
            prices,
            soc_step=float(status["soc_step"]),
            block_min=int(status["block_min"]),
            g_kwh=float(status["g_kwh"]),
            charge_kw=float(status["charge_kw"]),
            reserve_kwh=(
                float(status["min_soc_frac"]) * float(status["g_kwh"])
            ),
            strict_tariff_coverage=bool(
                status.get("strict_tariff_coverage", False)
            ),
            arc_mode="lazy",
        )
        record = network.fixed_sequence_record(trips)
        if record is None:
            raise ValueError(
                f"industrial duty {route['duty_id']} is absent from event model"
            )
        if record.get("trips") != trips:
            raise ValueError("event witness changed industrial trip order")
        physical = record.get("physical_realization") or {}
        if physical.get("status") != "valid_event_time_realized":
            raise ValueError("event witness lacks physical realization")
        counts.update(trips)
        records.append({
            "duty_id": route["duty_id"],
            "record": record,
        })
    expected = set(problem.trips)
    if set(counts) != expected or any(counts[trip] != 1 for trip in expected):
        raise ValueError("industrial event witnesses are not an exact partition")
    fleet = len(records)
    lower_bound = float(args.fleet_lower_bound)
    model_optimum_proven = (
        math.ceil(lower_bound - 1e-6) == fleet
    )
    if not model_optimum_proven:
        raise ValueError(
            "known event witness does not close certified fleet lower bound"
        )
    payload = {
        "schema": SCHEMA,
        "scope": "named_discrete_event_model_only",
        "status": str(status_path),
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "instance": str(instance),
        "instance_sha256": _sha256(instance),
        "fleet_lp_lower_bound": lower_bound,
        "integer_fleet_witness": fleet,
        "model_integer_optimum": fleet,
        "model_optimum_proven_by_sandwich": True,
        "physical_witness_valid": True,
        "trip_partition_count": len(counts),
        "records": records,
    }
    output = Path(args.out).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    atomic_write_json(output, payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--fleet-lower-bound", type=float, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit(args)
    print(json.dumps({
        "fleet_lp_lower_bound": payload["fleet_lp_lower_bound"],
        "model_integer_optimum": payload["model_integer_optimum"],
        "physical_witness_valid": payload["physical_witness_valid"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

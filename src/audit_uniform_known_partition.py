#!/usr/bin/env python3
"""Audit whether the industrial partition witnesses a uniform model optimum."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from audit_giro_known_columns import HORIZON_MIN, build_problem
from config import CHARGING_STATIONS
from durable_io import atomic_write_json
from expanded_path_realization import normalize_event_station_prices
from tariff_response_core import giro_routes_for_instance
from utils_v2 import load_station_hourly_prices
from warm_pool_fixed_duty_optimizer import optimize_fixed_duty


SCHEMA = "evsp-dr-uniform-known-partition-model-witness-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(args) -> dict:
    status_path = Path(args.status).expanduser().resolve(strict=True)
    status_bytes = status_path.read_bytes()
    status = json.loads(status_bytes)
    if status.get("time_model", "uniform") != "uniform":
        raise ValueError("uniform witness audit requires uniform time model")
    data_dir = Path(args.data_dir).expanduser().resolve()
    instance = (data_dir / status["csv"]).resolve(strict=True)
    tariff = (data_dir / status["prices_csv"]).resolve(strict=True)
    provenance = status.get("provenance") or {}
    if (
        _sha256(instance) != provenance.get("instance_sha256")
        or _sha256(tariff) != provenance.get("prices_sha256")
    ):
        raise ValueError("status input hashes do not match witness inputs")
    problem = build_problem(
        data_dir,
        status["csv"],
        max_station_to_trip_wait_min=HORIZON_MIN,
    )
    prices = normalize_event_station_prices(
        load_station_hourly_prices(tariff, CHARGING_STATIONS),
        horizon_min=HORIZON_MIN,
        strict_tariff_coverage=False,
    )
    known = giro_routes_for_instance(
        data_dir / "Par_VehicleDetails_Updated.csv",
        instance,
    )
    records = []
    counts = Counter()
    failures = []
    for route in known:
        result = optimize_fixed_duty(
            problem,
            route["trips"],
            prices,
            g_kwh=float(status["g_kwh"]),
            charge_kw=float(status["charge_kw"]),
            reserve_kwh=(
                float(status["min_soc_frac"]) * float(status["g_kwh"])
            ),
            soc_step=float(status["soc_step"]),
            block_min=int(status["block_min"]),
            tariff_id="historical_flat",
            tariff_sha256=provenance.get("prices_sha256"),
            instance_sha256=provenance.get("instance_sha256"),
            allow_declared_physics=True,
        )
        if not result["feasible"]:
            failures.append({
                "duty_id": route["duty_id"],
                "reason": result.get("reason"),
            })
            continue
        record = result["record"]
        counts.update(record["trips"])
        records.append({
            "duty_id": route["duty_id"],
            "record": record,
            "certificate": result["certificate"],
        })
    all_representable = not failures
    exact_partition = (
        all_representable
        and set(counts) == set(problem.trips)
        and all(counts[trip] == 1 for trip in problem.trips)
    )
    fleet = len(records) if exact_partition else None
    lower_bound = float(args.fleet_lower_bound)
    model_optimum_proven = bool(
        fleet is not None
        and math.ceil(lower_bound - 1e-6) == fleet
    )
    payload = {
        "schema": SCHEMA,
        "scope": "named_discrete_uniform_model_only",
        "status": str(status_path),
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "instance": str(instance),
        "instance_sha256": _sha256(instance),
        "grid": {
            "soc_step": status["soc_step"],
            "block_min": status["block_min"],
        },
        "fleet_lp_lower_bound": lower_bound,
        "all_industrial_duties_representable": all_representable,
        "exact_partition_witness": exact_partition,
        "integer_fleet_witness": fleet,
        "model_integer_optimum":
            fleet if model_optimum_proven else None,
        "model_optimum_proven_by_sandwich": model_optimum_proven,
        "physical_witness_valid": exact_partition,
        "failures": failures,
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
        "all_industrial_duties_representable":
            payload["all_industrial_duties_representable"],
        "model_integer_optimum": payload["model_integer_optimum"],
        "model_optimum_proven_by_sandwich":
            payload["model_optimum_proven_by_sandwich"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

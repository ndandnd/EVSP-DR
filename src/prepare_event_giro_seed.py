#!/usr/bin/env python3
"""Prepare tariff-specific GIRO trip sequences on the production event graph."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "evsp-dr-event-fixed-duty-partition-v1"
ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def route_signature(route):
    return canonical({key: route[key] for key in (
        "trips", "route_nodes", "charging_stops", "expanded_grid_charging_stops",
        "expanded_grid_cost", "continuous_realized_cost", "continuous_realized_charging_blocks")})


def validate_event_seed(payload, network, *, problem, tariff_sha256, instance_sha256):
    """Recompute every ordered witness on the actual CG graph before injection."""
    if network is None or not instance_sha256:
        raise ValueError("event seed requires the current event network")
    physics = payload.get("physics") or {}
    if (payload.get("schema") != SCHEMA
            or payload.get("instance_sha256") != instance_sha256
            or (payload.get("tariff") or {}).get("sha256") != tariff_sha256
            or payload.get("event_lattice_sha256") != network.metrics().get("event_lattice_sha256")
            or payload.get("continuous_cost_pricing_certified") is not False
            or any(not math.isclose(float(physics.get(key, math.nan)), value, abs_tol=1e-9, rel_tol=0)
                   for key, value in (("g_kwh", network.g), ("charge_kw", network.charge_kw),
                       ("reserve_kwh", network.reserve), ("soc_step", network.soc_step),
                       ("block_min", network.block_min)))):
        raise ValueError("event seed identity/physics/tariff mismatch")
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("event seed has no routes")
    coverage, records = Counter(), []
    for route in routes:
        trips = route.get("trips") or []
        if any(not isinstance(trip, int) or isinstance(trip, bool) for trip in trips):
            raise ValueError("event seed contains invalid trip IDs")
        if not trips or len(set(trips)) != len(trips):
            raise ValueError("event seed contains empty/repeated trips")
        recomputed = network.fixed_sequence_record(trips)
        if (recomputed is None or route_signature(route) != route_signature(recomputed)
                or route.get("master_cost_semantics") != "expanded_grid_cost"
                or route.get("cost_tariff_sha256") != tariff_sha256
                or float(route.get("cost", math.nan)) != float(recomputed["cost"])):
            raise ValueError("event seed route differs from current event witness")
        record = deepcopy(route)
        record.update(found_iter=0, origin="validated_event_giro_seed")
        records.append(record)
        coverage.update(trips)
    if set(coverage) != set(problem.trips) or any(value != 1 for value in coverage.values()):
        raise ValueError("event seed is not an exact trip partition")
    return records


def main():
    from audit_giro_known_columns import HORIZON_MIN, build_problem
    from compare_original_giro_charging import sha, extract_original, read_rows
    from config import CHARGING_STATIONS
    from event_pricer_network import EventExpandedNetwork
    from exact_pricer_expanded import (_event_network_cache_identity,
        _load_event_network_cache, _write_event_network_cache)
    from run_exact_pool_mip import verified_mip_code_identity
    from utils_v2 import load_station_hourly_prices

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--instance-sha256", required=True)
    parser.add_argument("--master", type=Path, default=ROOT / "data/Par_VehicleDetails_Updated.csv")
    parser.add_argument("--master-sha256", required=True)
    parser.add_argument("--tariff", type=Path, required=True)
    parser.add_argument("--tariff-sha256", required=True)
    parser.add_argument("--reference-data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--fleet", type=int, required=True)
    parser.add_argument("--g-kwh", type=float, default=240.)
    parser.add_argument("--charge-kw", type=float, default=240.)
    parser.add_argument("--soc-step", type=float, default=2.5)
    parser.add_argument("--block-min", type=int, default=5)
    parser.add_argument("--network-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for key in ("instance", "master", "tariff", "reference_data_dir", "network_cache", "out"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.out.exists():
        raise FileExistsError(args.out)
    expected = {args.instance: args.instance_sha256, args.master: args.master_sha256,
                args.tariff: args.tariff_sha256}
    if any(sha(path) != digest for path, digest in expected.items()):
        raise ValueError("instance/master/tariff hash mismatch")
    if any(not math.isfinite(v) or v <= 0 for v in (args.g_kwh,args.charge_kw,args.soc_step,args.block_min)):
        raise ValueError("positive finite event physics required")
    identity = verified_mip_code_identity()
    source_routes = extract_original(read_rows(args.master), read_rows(args.instance))
    if len(source_routes) != args.fleet:
        raise ValueError("selected whole-duty fleet mismatch")
    problem = build_problem(args.instance.parent, args.instance.name,
        max_station_to_trip_wait_min=HORIZON_MIN, reference_data_dir=args.reference_data_dir)
    provenance = {"git_commit": identity["observed_commit"], "instance_sha256": args.instance_sha256,
        "prices_sha256": args.tariff_sha256,
        "reference_sha256": sha(args.reference_data_dir / "Ref_dict.csv"),
        "deadhead_sha256": sha(args.reference_data_dir / "par_ref_dhd.csv")}
    expected.update({args.reference_data_dir / "Ref_dict.csv": provenance["reference_sha256"],
                     args.reference_data_dir / "par_ref_dhd.csv": provenance["deadhead_sha256"]})
    config = SimpleNamespace(g_kwh=args.g_kwh, charge_kw=args.charge_kw,
        soc_step=args.soc_step, block_min=args.block_min, min_soc_frac=0.,
        strict_tariff_coverage=True, event_arc_mode="lazy")
    cache_identity = _event_network_cache_identity(config, provenance)
    prices = load_station_hourly_prices(args.tariff, CHARGING_STATIONS)
    if Path(str(args.network_cache) + ".manifest.json").exists():
        network, _manifest = _load_event_network_cache(args.network_cache, cache_identity)
    else:
        import time
        started = time.perf_counter()
        network = EventExpandedNetwork(problem, prices, soc_step=args.soc_step,
            block_min=args.block_min, g_kwh=args.g_kwh, charge_kw=args.charge_kw,
            reserve_kwh=0., strict_tariff_coverage=True, arc_mode="lazy")
        _manifest = _write_event_network_cache(args.network_cache, network, cache_identity,
                                             time.perf_counter() - started)
    routes = []
    for source in source_routes:
        record = network.fixed_sequence_record(source["trips"])
        if record is None:
            raise ValueError(f"GIRO duty {source['duty_id']} is not representable on this event graph")
        record.update(duty_id=source["duty_id"], source_ordered_trip_ids=source["source_ordered_trip_ids"],
                      cost_tariff_sha256=args.tariff_sha256, found_iter=0, origin="validated_event_giro_seed")
        routes.append(record)
    payload = {"schema": SCHEMA, "source": "GIRO_EVENT_AUGMENTED",
        "instance_sha256": args.instance_sha256, "master_sha256": args.master_sha256,
        "tariff": {"sha256": args.tariff_sha256, "path": str(args.tariff)},
        "physics": {"g_kwh": args.g_kwh, "charge_kw": args.charge_kw,
            "reserve_kwh": 0., "soc_step": args.soc_step, "block_min": args.block_min,
            "initial_soc_kwh": args.g_kwh, "terminal_soc_policy": "depot_arrival_soc_at_least_reserve"},
        "provenance": provenance, "network_cache": str(args.network_cache),
        "event_lattice_sha256": network.metrics()["event_lattice_sha256"],
        "route_count": len(routes), "routes": routes,
        "continuous_cost_pricing_certified": False,
        "certificate_scope": "cheapest_event_graph_route_for_each_fixed_giro_trip_sequence",
        "original_giro_charging_preserved": False,
        "baseline_label": "GIRO_FIXED_DUTIES_REOPTIMIZED_EVENT_CHARGING"}
    validate_event_seed(payload, network, problem=problem,
        tariff_sha256=args.tariff_sha256, instance_sha256=args.instance_sha256)
    if any(sha(path) != digest for path, digest in expected.items()):
        raise ValueError("source changed while constructing seed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"out": str(args.out), "fleet": len(routes), "event_seed_validated": True}))


if __name__ == "__main__":
    main()

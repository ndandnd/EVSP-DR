#!/usr/bin/env python3
"""Build a deterministic, physically replayed GREEDY event seed partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path

from audit_giro_known_columns import HORIZON_MIN, build_problem
from durable_io import atomic_write_json
from exact_pricer_expanded import (
    _event_network_cache_manifest_path,
    _file_sha256,
    _load_event_network_cache,
)
from greedy_init import build_greedy_routes
from run_greedy_init_only import (
    build_arc_data,
    load_trip_dataframe,
    route_trip_ids,
)


SCHEMA = "evsp-dr-event-greedy-partition-v1"


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    )
    return result.stdout.strip()


def event_partition(sequences, network):
    """Map greedy sequences to event records, splitting only when required."""

    records = []
    split_routes = 0
    for sequence in sequences:
        start = 0
        pieces = []
        while start < len(sequence):
            selected = None
            for end in range(len(sequence), start, -1):
                record = network.fixed_sequence_record(sequence[start:end])
                if record is not None:
                    selected = (end, record)
                    break
            if selected is None:
                raise ValueError(
                    f"event graph has no singleton route for trip {sequence[start]}"
                )
            end, record = selected
            pieces.append(record)
            start = end
        split_routes += max(0, len(pieces) - 1)
        records.extend(pieces)
    return records, split_routes


def prepare(args) -> dict:
    started = time.perf_counter()
    repo = Path(args.repo).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    instance = (data_dir / args.csv).resolve(strict=True)
    tariff = (data_dir / args.prices_csv).resolve(strict=True)
    reference = data_dir / "Ref_dict.csv"
    deadhead = data_dir / "par_ref_dhd.csv"
    cache = Path(args.event_network_cache).expanduser().resolve(strict=True)
    manifest_path = _event_network_cache_manifest_path(cache)
    manifest = json.loads(manifest_path.read_text())
    identity = manifest.get("identity") or {}
    commit = _git_commit(repo)
    expected_inputs = {
        "instance_sha256": _file_sha256(instance),
        "prices_sha256": _file_sha256(tariff),
        "reference_sha256": _file_sha256(reference),
        "deadhead_sha256": _file_sha256(deadhead),
    }
    expected_identity = {
        "schema": "evsp-dr-event-network-cache-v1",
        "git_commit": commit,
        **expected_inputs,
        "soc_step": float(args.soc_step),
        "block_min": int(args.block_min),
        "g_kwh": float(args.g_kwh),
        "charge_kw": float(args.charge_kw),
        "reserve_kwh": float(args.reserve_kwh),
        "strict_tariff_coverage": bool(args.strict_tariff_coverage),
        "event_arc_mode": "lazy",
    }
    if identity != expected_identity:
        raise ValueError("event cache identity does not match greedy inputs")
    network, manifest = _load_event_network_cache(cache, expected_identity)
    problem = build_problem(
        data_dir, args.csv, max_station_to_trip_wait_min=HORIZON_MIN,
    )
    if tuple(problem.trips) != tuple(network.problem.trips):
        raise ValueError("event cache problem trip order mismatch")

    frame = load_trip_dataframe(instance)
    arc_data = build_arc_data(frame, reference, deadhead)
    proposals = build_greedy_routes(
        T=arc_data["T"], S_use=arc_data["S_use"],
        DEPOT=arc_data["DEPOT"], tau=arc_data["tau"],
        tau_min=arc_data["tau_min"], d=arc_data["d"],
        st=arc_data["st"], et=arc_data["et"],
        st_min=arc_data["st_min"], et_min=arc_data["et_min"],
        sl=arc_data["sl"], el=arc_data["el"],
        epsilon=arc_data["epsilon"], G=float(args.g_kwh),
        bar_t=int(HORIZON_MIN), TB_MIN=1.0,
        CHARGE_RATE_KW=float(args.charge_kw),
        min_soc_fraction=float(args.reserve_kwh) / float(args.g_kwh),
    )
    sequences = [route_trip_ids(route) for route in proposals]
    records, split_routes = event_partition(sequences, network)
    counts = Counter()
    for record in records:
        record["cost_tariff_sha256"] = expected_inputs["prices_sha256"]
        counts.update(record["trips"])
    trip_set = set(problem.trips)
    if set(counts) != trip_set or any(counts[trip] != 1 for trip in trip_set):
        raise ValueError("GREEDY event records are not an exact partition")
    route_hashes = [canonical_sha256(record) for record in records]
    physics = {
        "g_kwh": float(args.g_kwh),
        "charge_kw": float(args.charge_kw),
        "reserve_kwh": float(args.reserve_kwh),
        "soc_step": float(args.soc_step),
        "block_min": int(args.block_min),
    }
    if any(not math.isfinite(value) for value in physics.values()):
        raise ValueError("non-finite physics")
    payload = {
        "schema": SCHEMA,
        "source": "GREEDY",
        "column_pool_treatment": "GREEDY",
        "exact_trip_partition": True,
        "continuous_cost_pricing_certified": False,
        "certificate_scope": (
            "deterministic_greedy_trip_partition_reoptimized_and_replayed_"
            "on_named_event_graph"
        ),
        "instance": args.csv,
        "input_hashes": expected_inputs,
        "physics": physics,
        "event_network_cache": {
            "identity": manifest["identity"],
            "manifest_sha256": _file_sha256(manifest_path),
            "pickle_sha256": manifest["pickle_sha256"],
            "network_metrics": manifest["network_metrics"],
        },
        "algorithm": {
            "name": "legacy_greedy_trip_sequence_current_physics_event_gate",
            "trip_order": "start_min_then_local_trip_id",
            "candidate_order": "earliest_feasible_then_local_trip_id",
            "event_failure_policy": "longest_feasible_prefix_split",
            "legacy_charge_plan_retained": False,
            "note": (
                "The legacy heuristic proposes only trip sequences under the "
                "requested physics. Charging is discarded and recomputed by "
                "EventExpandedNetwork.fixed_sequence_record."
            ),
        },
        "proposed_route_count": len(sequences),
        "event_split_count": split_routes,
        "route_count": len(records),
        "trip_count": len(problem.trips),
        "route_sequence_sha256": canonical_sha256(
            [list(record["trips"]) for record in records]
        ),
        "route_record_sha256": route_hashes,
        "routes": records,
        "runtime_s": time.perf_counter() - started,
        "provenance": {"git_commit": commit},
    }
    output = Path(args.out).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parents[1] / "data")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--prices-csv", default="hourly_prices_flat.csv")
    parser.add_argument("--event-network-cache", type=Path, required=True)
    parser.add_argument("--g-kwh", type=float, default=240.0)
    parser.add_argument("--charge-kw", type=float, default=240.0)
    parser.add_argument("--reserve-kwh", type=float, default=0.0)
    parser.add_argument("--soc-step", type=float, default=2.5)
    parser.add_argument("--block-min", type=int, default=5)
    parser.add_argument("--strict-tariff-coverage", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = prepare(args)
    print(json.dumps({
        "route_count": payload["route_count"],
        "trip_count": payload["trip_count"],
        "event_split_count": payload["event_split_count"],
        "route_sequence_sha256": payload["route_sequence_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Merge physically audited CG pools across grids and test target feasibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from run_exact_pool_mip import (
    file_sha256,
    load_pool,
    prepare_strict_partition_pool,
    validate_injected_route,
    resolve_pool_journal,
    verified_mip_code_identity,
    write_new_json,
)
from target_pool_feasibility import solve_target_feasibility


SCHEMA = "evsp-dr-resolution-pool-union-v1"
MIP_SCHEMA = "evsp-dr-resolution-pool-union-target-feasibility-v1"


def _canonical(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def route_sha256(route):
    """Hash one complete route realization for exact deduplication."""

    return hashlib.sha256(_canonical(route)).hexdigest()


def _identity(status):
    provenance = status.get("provenance") or {}
    return {
        "instance_sha256": provenance.get("instance_sha256"),
        "prices_sha256": provenance.get("prices_sha256"),
        "reference_sha256": provenance.get("reference_sha256"),
        "deadhead_sha256": provenance.get("deadhead_sha256"),
        "g_kwh": status.get("g_kwh"),
        "charge_kw": status.get("charge_kw"),
        "min_soc_frac": status.get("min_soc_frac"),
        "csv": status.get("csv"),
        "prices_csv": status.get("prices_csv"),
        "trip_ids": status.get("trip_ids"),
    }


def merge_route_sets(sources):
    """Return a deterministic route-hash union and superset proof."""

    if len(sources) < 2:
        raise ValueError("resolution union requires at least two sources")
    expected = sources[0]["identity"]
    for source in sources:
        if source["identity"] != expected:
            differences = [
                key for key in expected
                if source["identity"].get(key) != expected.get(key)
            ]
            raise ValueError(
                "pool union source identity mismatch: "
                + ",".join(differences)
            )
    union = {}
    source_hash_sets = {}
    for source_index, source in enumerate(sources):
        hashes = set()
        for route in source["routes"]:
            digest = route_sha256(route)
            hashes.add(digest)
            union.setdefault(digest, route)
        source_hash_sets[
            source.get("source_id", f"source_{source_index}")
        ] = hashes
    ordered_hashes = sorted(union)
    union_hashes = set(ordered_hashes)
    if any(not hashes <= union_hashes for hashes in source_hash_sets.values()):
        raise RuntimeError("union is not a superset of every input pool")
    return (
        [union[digest] for digest in ordered_hashes],
        {
            "verified": True,
            "basis": "every_physically_admitted_source_route_hash_is_in_union",
            "source_route_hash_counts": {
                digest: len(hashes)
                for digest, hashes in sorted(source_hash_sets.items())
            },
            "union_route_hashes_sha256": hashlib.sha256(
                _canonical(ordered_hashes)
            ).hexdigest(),
        },
    )


def _write_new_jsonl(path, records):
    if os.path.lexists(path):
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(parent_fd)
    os.close(parent_fd)


def validate_union_witness(union, routes, *, data_dir=None,
                           reference_data_dir=None):
    from audit_giro_known_columns import HORIZON_MIN, build_problem
    from expanded_path_realization import validate_continuous_charging_blocks
    from config import CHARGING_STATIONS
    from utils_v2 import load_station_hourly_prices

    data_root = Path(
        data_dir if data_dir is not None
        else Path(__file__).resolve().parent.parent/"data"
    ).resolve()
    reference_root = Path(
        reference_data_dir if reference_data_dir is not None else data_root
    ).resolve()
    problem = build_problem(
        data_root,
        union["csv"],
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=reference_root,
    )
    if list(problem.trips) != list(union["trip_ids"]):
        raise RuntimeError("union witness instance trip identity changed")
    prices = load_station_hourly_prices(
        data_root/union["prices_csv"], CHARGING_STATIONS,
    )
    counts = {trip: 0 for trip in problem.trips}
    hashes = []
    reserve = float(union["min_soc_frac"])*float(union["g_kwh"])
    for route in routes:
        reason = validate_injected_route(
            problem, route, float(union["g_kwh"]),
            float(union["charge_kw"]), reserve, HORIZON_MIN,
        )
        if reason is not None:
            raise RuntimeError(f"union witness route failed replay: {reason}")
        validate_continuous_charging_blocks(
            route,
            route.get("continuous_realized_charging_blocks"),
            station_prices=prices,
            charge_kw=float(union["charge_kw"]),
            expected_continuous_cost=route.get("continuous_realized_cost"),
        )
        for trip in route["trips"]:
            counts[trip] += 1
        hashes.append(route_sha256(route))
    if any(value != 1 for value in counts.values()):
        raise RuntimeError("union witness is not an immutable exact partition")
    return {
        "validated": True,
        "route_count": len(routes),
        "route_hashes_sha256": hashlib.sha256(
            _canonical(sorted(hashes))
        ).hexdigest(),
    }


def build_union(result_paths, *, output_path, data_dir=None,
                reference_data_dir=None):
    requested_output = Path(output_path)
    if os.path.lexists(requested_output):
        raise FileExistsError("union output already exists")
    output_path = requested_output.resolve()
    journal_out = Path(str(output_path) + ".columns.jsonl")
    if os.path.lexists(journal_out):
        raise FileExistsError("union output or journal already exists")
    code_identity = verified_mip_code_identity()
    from audit_giro_known_columns import HORIZON_MIN, build_problem
    data_root = Path(
        data_dir if data_dir is not None
        else Path(__file__).resolve().parent.parent/"data"
    ).resolve()
    reference_root = Path(
        reference_data_dir if reference_data_dir is not None else data_root
    ).resolve()
    loaded = []
    for result in result_paths:
        result = Path(result).resolve()
        status_raw = result.read_bytes()
        status = json.loads(status_raw)
        journal = resolve_pool_journal(result, status)
        result_sha = hashlib.sha256(status_raw).hexdigest()
        journal_sha = file_sha256(journal)
        loaded_status, routes, trips = load_pool(
            result, deduplicate=False,
        )
        if (
            loaded_status != status
            or file_sha256(result) != result_sha
            or file_sha256(journal) != journal_sha
        ):
            raise RuntimeError("pool union source changed while loading")
        admitted, audit = prepare_strict_partition_pool(
            status,
            routes,
            data_dir=data_dir,
            reference_data_dir=reference_data_dir,
        )
        if (
            file_sha256(result) != result_sha
            or file_sha256(journal) != journal_sha
        ):
            raise RuntimeError("pool union source changed during physical audit")
        provenance = status.get("provenance") or {}
        audited_hashes = audit.get("input_hashes") or {}
        for field in (
            "instance_sha256", "prices_sha256",
            "reference_sha256", "deadhead_sha256",
        ):
            if not provenance.get(field) or provenance[field] != audited_hashes.get(field):
                raise ValueError(
                    f"pool union source lacks matching {field}"
                )
        problem = build_problem(
            data_root,
            status["csv"],
            max_station_to_trip_wait_min=HORIZON_MIN,
            reference_data_dir=reference_root,
        )
        if list(problem.trips) != trips:
            raise ValueError(
                "pool union status trip_ids differ from immutable instance"
            )
        loaded.append({
            "source_id": hashlib.sha256(_canonical(
                [str(result), result_sha, journal_sha]
            )).hexdigest(),
            "result": str(result),
            "result_sha256": result_sha,
            "journal": str(journal.resolve()),
            "journal_sha256": journal_sha,
            "identity": _identity(status),
            "grid": {
                "soc_step": status.get("soc_step"),
                "block_min": status.get("block_min"),
            },
            "routes": admitted,
            "physical_pool_audit": audit,
            "trip_ids": trips,
        })
    loaded.sort(key=lambda row: row["journal_sha256"])
    routes, superset = merge_route_sets(loaded)
    identity = loaded[0]["identity"]
    if not identity["instance_sha256"] or not identity["trip_ids"]:
        raise ValueError("union source lacks immutable instance identity")
    for source in loaded:
        if (
            file_sha256(Path(source["result"])) != source["result_sha256"]
            or file_sha256(Path(source["journal"])) != source["journal_sha256"]
        ):
            raise RuntimeError("pool union source changed before publication")
    if verified_mip_code_identity() != code_identity:
        raise RuntimeError("pool union code identity changed during build")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_new_jsonl(journal_out, routes)
    payload = {
        "schema": SCHEMA,
        "csv": identity["csv"],
        "prices_csv": identity["prices_csv"],
        "trip_ids": identity["trip_ids"],
        "g_kwh": identity["g_kwh"],
        "charge_kw": identity["charge_kw"],
        "min_soc_frac": identity["min_soc_frac"],
        "source_grids": [source["grid"] for source in loaded],
        "source_count": len(loaded),
        "sources": [
            {
                key: source[key] for key in (
                    "result", "result_sha256", "journal", "journal_sha256",
                    "grid", "physical_pool_audit",
                )
            }
            for source in loaded
        ],
        "columns": len(routes),
        "columns_journal": str(journal_out),
        "columns_journal_sha256": file_sha256(journal_out),
        "route_hash_deduplication": superset,
        "provenance": {
            key: identity[key] for key in (
                "instance_sha256", "prices_sha256",
                "reference_sha256", "deadhead_sha256",
            )
        },
        "physics_identity": {
            key: identity[key] for key in (
                "g_kwh", "charge_kw", "min_soc_frac",
            )
        },
        "cross_resolution_cost_comparability": False,
        "intended_use": "fleet_partition_or_target_feasibility",
        "code_identity": code_identity,
    }
    write_new_json(output_path, payload)
    return payload, routes


def evaluate(args):
    union_path = Path(args.out)
    requested_mip = Path(args.mip_out)
    if os.path.lexists(requested_mip):
        raise FileExistsError(requested_mip)
    mip_path = requested_mip.resolve()
    union, routes = build_union(
        args.result,
        output_path=union_path,
        data_dir=args.data_dir,
        reference_data_dir=args.reference_data_dir,
    )
    union_path = union_path.resolve()
    union_status_sha256 = file_sha256(union_path)
    solved = solve_target_feasibility(
        routes,
        union["trip_ids"],
        args.target,
        timelimit=args.timelimit,
        threads=args.threads,
        seed=args.seed,
    )
    selected = solved.pop("selected_indices")
    selected_routes = [routes[index] for index in selected]
    witness_audit = (
        validate_union_witness(
            union,
            selected_routes,
            data_dir=args.data_dir,
            reference_data_dir=args.reference_data_dir,
        )
        if solved["outcome"] == "FEASIBLE" else None
    )
    if (
        file_sha256(union_path) != union_status_sha256
        or file_sha256(Path(union["columns_journal"]))
        != union["columns_journal_sha256"]
    ):
        raise RuntimeError("published union changed during target solve")
    final_code_identity = verified_mip_code_identity()
    if final_code_identity != union["code_identity"]:
        raise RuntimeError("pool union code identity changed during target solve")
    payload = {
        "schema": MIP_SCHEMA,
        "outcome": solved["outcome"],
        "conclusion": (
            "target_partition_exists"
            if solved["outcome"] == "FEASIBLE"
            else "target_partition_absent_from_union_pool"
            if solved["outcome"] == "INFEASIBLE"
            else None
        ),
        "censored": solved["outcome"] == "TIME_LIMIT",
        "target_fleet": args.target,
        "witness_route_count": (
            len(selected_routes) if solved["outcome"] == "FEASIBLE" else None
        ),
        "witness_routes": (
            selected_routes if solved["outcome"] == "FEASIBLE" else []
        ),
        "witness_audit": witness_audit,
        "union": {
            "status": str(union_path),
            "status_sha256": union_status_sha256,
            "journal": union["columns_journal"],
            "journal_sha256": union["columns_journal_sha256"],
            "columns": union["columns"],
            "superset_verification": union["route_hash_deduplication"],
        },
        "finite_union_pool_scope_only": True,
        "solver": solved,
        "code_identity": union["code_identity"],
    }
    write_new_json(mip_path, payload)
    return union, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", type=Path, action="append", required=True,
        help="Repeat for each exact-CG source status.",
    )
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--timelimit", type=float, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--reference-data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mip-out", type=Path, required=True)
    args = parser.parse_args(argv)
    _union, mip = evaluate(args)
    print(json.dumps({
        "outcome": mip["outcome"],
        "target_fleet": mip["target_fleet"],
        "witness_route_count": mip["witness_route_count"],
        "censored": mip["censored"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

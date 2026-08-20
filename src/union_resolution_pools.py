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

    return hashlib.sha256(_canonical({
        "trips": route.get("trips"),
        "route_nodes": route.get("route_nodes"),
        "charging_stops": route.get("charging_stops"),
        "expanded_grid_charging_stops":
            route.get("expanded_grid_charging_stops"),
        "cost": route.get("cost"),
        "continuous_realized_charging_blocks":
            route.get("continuous_realized_charging_blocks"),
    })).hexdigest()


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
    journal_ids = [source["journal_sha256"] for source in sources]
    if len(set(journal_ids)) != len(journal_ids):
        raise ValueError("resolution union repeats a source journal")
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
    for source in sources:
        hashes = set()
        for route in source["routes"]:
            digest = route_sha256(route)
            hashes.add(digest)
            union.setdefault(digest, route)
        source_hash_sets[source["journal_sha256"]] = hashes
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


def build_union(result_paths, *, output_path, data_dir=None,
                reference_data_dir=None):
    output_path = Path(output_path).resolve()
    journal_out = Path(str(output_path) + ".columns.jsonl")
    if output_path.exists() or journal_out.exists():
        raise FileExistsError("union output or journal already exists")
    loaded = []
    for result in result_paths:
        result = Path(result).resolve()
        status_raw = result.read_bytes()
        status = json.loads(status_raw)
        journal = resolve_pool_journal(result, status)
        _loaded_status, routes, trips = load_pool(
            result, deduplicate=False,
        )
        admitted, audit = prepare_strict_partition_pool(
            status,
            routes,
            data_dir=data_dir,
            reference_data_dir=reference_data_dir,
        )
        journal_sha = file_sha256(journal)
        loaded.append({
            "result": str(result),
            "result_sha256": hashlib.sha256(status_raw).hexdigest(),
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
    }
    write_new_json(output_path, payload)
    return payload, routes


def evaluate(args):
    union_path = Path(args.out).resolve()
    mip_path = Path(args.mip_out).resolve()
    if mip_path.exists():
        raise FileExistsError(mip_path)
    union, routes = build_union(
        args.result,
        output_path=union_path,
        data_dir=args.data_dir,
        reference_data_dir=args.reference_data_dir,
    )
    solved = solve_target_feasibility(
        routes,
        union["trip_ids"],
        args.target,
        timelimit=args.timelimit,
        threads=args.threads,
        seed=args.seed,
    )
    selected = solved.pop("selected_indices")
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
            len(selected) if solved["outcome"] == "FEASIBLE" else None
        ),
        "witness_routes": (
            [routes[index] for index in selected]
            if solved["outcome"] == "FEASIBLE" else []
        ),
        "union": {
            "status": str(union_path),
            "status_sha256": file_sha256(union_path),
            "journal": union["columns_journal"],
            "journal_sha256": union["columns_journal_sha256"],
            "columns": union["columns"],
            "superset_verification": union["route_hash_deduplication"],
        },
        "finite_union_pool_scope_only": True,
        "solver": solved,
        "code_identity": verified_mip_code_identity(),
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

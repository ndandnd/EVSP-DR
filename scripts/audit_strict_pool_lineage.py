#!/usr/bin/env python3
"""Authenticate and replay the k17 -> saved zero-pricing k19 pool; no graph/solver.

The audit program's commit is distinct from the hash-verified model-source pin.
It never writes a route pool or modifies an original checkpoint identifier.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

SCHEMA = "evsp-dr-strict-k17-k19-lineage-gate-v1"
PARENT_COMMIT = "35770aae2c08e7d5a356cc3b673e67608e5b1036"
MODEL_COMMIT = "fedf421461f94727e6b1292a0e7789ab76ed8587"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def journal(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_new_json(path, payload):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite audit output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compare_inherited(regenerated, saved, *, old_checkpoint, new_checkpoint, route_key):
    """Exact ordered records; only the expected new checkpoint is normalized."""
    require(len(regenerated) == len(saved), "inherited route count mismatch")
    require(len({route_key(route) for route in regenerated}) == len(regenerated),
            "duplicate inherited route keys")
    ordered_hashes = []
    for index, (new, old) in enumerate(zip(regenerated, saved)):
        require(old.get("cg_checkpoint_id") == old_checkpoint,
                f"old child checkpoint mismatch at inherited route {index}")
        require(new.get("cg_checkpoint_id") == new_checkpoint,
                f"new checkpoint mismatch at inherited route {index}")
        # The original dictionary/journal is unchanged. Every other field,
        # including parent-route hashes, costs and realization metadata, stays.
        normalized = {**old, "cg_checkpoint_id": new_checkpoint}
        require(canonical(new) == canonical(normalized),
                f"inherited full-record mismatch at route {index}")
        ordered_hashes.append(canonical(new))
    return canonical(ordered_hashes)


def singleton_trip_ids(routes, *, missing_trips, checkpoint):
    observed = []
    for index, route in enumerate(routes):
        trips = route.get("trips")
        node_trips = [node for node in route.get("route_nodes", []) if type(node) is int]
        require(isinstance(trips, list) and len(trips) == 1 and trips == node_trips,
                f"singleton trip/node mismatch at suffix route {index}")
        require(route.get("origin") == "exact_event_singleton"
                and route.get("found_iter") == 0 and "inheritance" not in route,
                f"singleton provenance mismatch at suffix route {index}")
        require(route.get("cg_checkpoint_id") == checkpoint,
                f"singleton checkpoint mismatch at suffix route {index}")
        observed.extend(trips)
    require(len(observed) == len(set(observed)), "duplicate singleton trip")
    require(set(observed) == set(missing_trips), "singleton trips differ from missing child trips")
    return observed


def load_model(root, expected_hashes):
    root = Path(root).resolve()
    for relative, expected in expected_hashes.items():
        require(sha(root / relative) == expected, f"model source hash mismatch: {relative}")
    expected_python = {name for name in expected_hashes if name.startswith("src/") and name.endswith(".py")}
    actual_python = {str(path.relative_to(root)) for path in (root / "src").rglob("*.py")}
    require(actual_python == expected_python, "model Python source inventory mismatch")
    sys.path.insert(0, str(root / "src"))
    modules = {name: importlib.import_module(name) for name in (
        "run_capacity_speed_event_cg", "inherit_capacity_pool", "expanded_path_realization",
        "run_exact_pool_mip", "audit_giro_known_columns", "utils_v2")}
    for name, module in modules.items():
        require(Path(module.__file__).resolve().parent == root / "src",
                f"model module imported outside verified source: {name}")
    return SimpleNamespace(runner=modules["run_capacity_speed_event_cg"],
        inheritance=modules["inherit_capacity_pool"], realization=modules["expanded_path_realization"],
        physical=modules["run_exact_pool_mip"], problem=modules["audit_giro_known_columns"],
        utils=modules["utils_v2"])


def reconstructed_singleton(model, problem, route, args, prices, arc_map):
    """Re-realize the saved path/grid schedule, not a shortest-path search."""
    expanded = copy.deepcopy(route["expanded_grid_charging_stops"])
    base = {"trips": route["trips"], "route_nodes": route["route_nodes"],
            "charging_stops": expanded, "expanded_grid_charging_stops": copy.deepcopy(expanded),
            "cost": 0.0}
    record, detail = model.realization.realize_expanded_path(problem, base,
        g_kwh=args.battery_kwh, charge_kw=args.non_parx_kw, reserve_kwh=args.reserve_kwh,
        soc_step=args.soc_step, block_min=args.block_min, arc_map=arc_map,
        time_model="event", station_charge_kw=model.runner.station_power(args.arm))
    require(record is not None, f"singleton grid replay failed: {detail.get('reason')}")
    costs = model.realization.realized_costs(record, detail["mapping"], station_prices=prices)
    blocks = costs["continuous_realized_charging_blocks"]
    cost = float(costs["recomputed_expanded_grid_cost"])
    record.update({"cost": cost, "expanded_grid_cost": cost,
        "continuous_realized_cost": costs["continuous_realized_cost"],
        "continuous_realized_charging_blocks": blocks,
        "continuous_realized_charging_blocks_json_bytes": len(json.dumps(blocks, sort_keys=True, separators=(",", ":"))),
        "cost_semantics": "expanded_grid_cost", "master_cost_semantics": "expanded_grid_cost",
        "continuous_cost_pricing_certified": False,
        "physical_realization": {"status": "valid_event_time_realized", "time_model": "event",
            "realization_schema": detail["mapping"]["schema"],
            "realization_mapping_sha256": detail["mapping"]["mapping_sha256"],
            "continuous_realized_charging_blocks_schema": model.realization.BLOCK_SCHEDULE_SCHEMA,
            "continuous_realized_charging_blocks_sha256": model.realization.charging_block_schedule_sha256(blocks),
            "continuous_cost_pricing_certified": False}})
    record.update({key: route[key] for key in ("origin", "found_iter", "cg_checkpoint_id")})
    require(canonical(record) == canonical(route), "singleton stored schedule/cost/metadata differs from independent replay")
    return canonical(record)


def run_gate(manifest_path, input_root, model_root, out):
    started = time.monotonic()
    manifest = read_json(manifest_path)
    require(manifest["schema"] == SCHEMA, "gate schema mismatch")
    require(manifest["model_execution_commit"] == MODEL_COMMIT, "wrong model pin")
    require(manifest["compatible_parent_commit"] == PARENT_COMMIT, "wrong audited parent pin")
    require(not Path(out).exists(), "audit output exists")
    files = {}
    for key, item in manifest["input_files"].items():
        path = Path(input_root) / item["local_name"]
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                f"input size/hash mismatch: {key}")
        files[key] = path
    model = load_model(model_root, manifest["model_source_sha256"])
    runner = model.runner
    statuses = {key: read_json(files[key + "_status"]) for key in ("k17", "k19")}
    arguments = {}
    problems = {}
    for key in ("k17", "k19"):
        command = read_json(files[key + "_command"])
        args = runner.parser().parse_args(command[2:])
        require(args.mode == "cg" and args.arm == "parx60" and args.arc_mode == "lazy",
                f"unexpected {key} original mode/arm/arcs")
        require(args.expected_commit == PARENT_COMMIT and args.require_clean,
                f"unverified {key} original execution pin")
        require(not args.resume, f"unexpected original {key} resume")
        arguments[key] = args
        status = statuses[key]
        require(status["schema"] == runner.SCHEMA and status["mode"] == "cg" and status["arm"] == args.arm,
                f"unexpected {key} status model")
        require(status["physics"] == manifest["physics"], f"{key} physics mismatch")
        require(status["provenance"]["git_commit"] == PARENT_COMMIT
                and status["provenance"]["git_tracked_dirty"] is False, f"{key} source pin/dirty mismatch")
        require(status["pool_sha256"] == manifest["input_files"][key + "_pool"]["sha256"],
                f"{key} status/journal hash mismatch")
        for name, input_key in (("instance_sha256", key + "_instance"), ("prices_sha256", "prices"),
                                ("reference_sha256", "reference"), ("deadhead_sha256", "deadhead")):
            require(status["provenance"][name] == manifest["input_files"][input_key]["sha256"],
                    f"{key} provenance mismatch: {name}")
        problem = model.problem.build_problem(files[key + "_instance"].parent, files[key + "_instance"].name,
            reference_data_dir=files["reference"].parent, max_station_to_trip_wait_min=args.max_station_wait_min)
        problems[key] = problem
        require(len(problem.trips) == manifest["expected"][key + "_trips"], f"{key} trip count mismatch")
        require(runner.checkpoint_id(args, problem, status["provenance"]) == status["checkpoint"]["id"],
                f"{key} checkpoint reconstruction mismatch")
    child_status = statuses["k19"]
    require(child_status["iterations"] == [] and child_status["stop_reason"] == "cg_wall_limit"
            and child_status["certified_rc_optimal"] is False
            and child_status["terminal_exact_min_reduced_cost"] is None,
            "saved k19 is not the expected zero-pricing uncertified attempt")
    parent_count = manifest["expected"]["inherited_routes"]
    child_count = manifest["expected"]["child_routes"]
    require(child_status["checkpoint"]["initial_pool_columns"] == child_count
            and child_status["final"]["pool_columns"] == child_count,
            "k19 initial/final pool count mismatch")
    lineage = child_status["inheritance"]
    for field, input_key in (("parent_status_sha256", "k17_status"), ("parent_pool_sha256", "k17_pool"),
                              ("parent_instance_sha256", "k17_instance")):
        require(lineage[field] == manifest["input_files"][input_key]["sha256"], f"old k19 lineage hash mismatch: {field}")
    require(lineage["inherited_columns"] == parent_count and lineage["every_inherited_route_replayed"] is True,
            "old inherited count/replay status mismatch")
    parent_routes = journal(files["k17_pool"])
    saved = journal(files["k19_pool"])
    require(len(parent_routes) == parent_count and len(saved) == child_count, "journal route count mismatch")
    require(statuses["k17"]["final"]["pool_columns"] == parent_count, "k17 status route count mismatch")
    require(len({runner.route_key(route) for route in parent_routes}) == parent_count, "duplicate authentic parent route keys")
    require(len({runner.route_key(route) for route in saved}) == child_count, "duplicate saved child route keys")
    problem = problems["k19"]
    args = arguments["k19"]
    arcs = {(source, target): (travel, energy) for source, rows in problem.adjacency.items()
            for target, travel, energy, _kind in rows}
    replay_count = 0
    def replay(route):
        nonlocal replay_count
        trips = route.get("trips")
        if trips != [node for node in route.get("route_nodes", []) if type(node) is int]:
            return "trip sequence differs from route nodes"
        if not trips or not set(trips).issubset(problem.trips) or len(set(trips)) != len(trips):
            return "invalid/repeated trip identity"
        reason = model.physical.validate_injected_route(problem, route, args.battery_kwh,
            args.non_parx_kw, args.reserve_kwh, model.problem.HORIZON_MIN,
            arrival_grace_min=0.0, arc_map=arcs, station_charge_kw=runner.station_power(args.arm))
        if reason is None:
            replay_count += 1
        return reason
    # This provenance identifies verified model bytes, not the separate audit
    # wrapper's commit. No saved status or journal is changed.
    target_prov = {**child_status["provenance"], "git_commit": MODEL_COMMIT}
    new_checkpoint = runner.checkpoint_id(args, problem, target_prov)
    inherited, metadata = model.inheritance.inherit_pool(files["k17_status"], files["k17_pool"],
        files["k17_instance"], files["k19_instance"], expected_physics=manifest["physics"],
        child_provenance=target_prov, new_checkpoint_id=new_checkpoint,
        route_validator=replay, compatible_parent_commit=PARENT_COMMIT)
    require(replay_count == parent_count, "not every inherited route replayed")
    inherited_hash = compare_inherited(inherited, saved[:parent_count],
        old_checkpoint=child_status["checkpoint"]["id"], new_checkpoint=new_checkpoint,
        route_key=runner.route_key)
    mapping = model.inheritance.trip_mapping(files["k17_instance"], files["k19_instance"])
    require(metadata["trip_mapping_sha256"] == lineage["trip_mapping_sha256"], "trip mapping identity differs from old k19")
    missing = set(problem.trips) - set(mapping.values())
    singletons = saved[parent_count:]
    require(len(singletons) == manifest["expected"]["singleton_routes"], "singleton suffix count mismatch")
    singleton_ids = singleton_trip_ids(singletons, missing_trips=missing, checkpoint=child_status["checkpoint"]["id"])
    prices = model.utils.load_station_hourly_prices(files["prices"],
        sorted({model.utils.base_station_name(station) for station in model.problem.STATIONS}))
    prices = model.realization.normalize_event_station_prices(prices, horizon_min=model.problem.HORIZON_MIN,
                                                            strict_tariff_coverage=False)
    singleton_hashes = []
    for index, route in enumerate(singletons):
        reason = replay(route)
        require(reason is None, f"singleton {index} physical replay failed: {reason}")
        singleton_hashes.append(reconstructed_singleton(model, problem, route, args, prices, arcs))
    # Rehash input journals/statuses after use: mutable input files are not
    # silently accepted even if changed while this audit was running.
    for key, path in files.items():
        require(sha(path) == manifest["input_files"][key]["sha256"], f"input changed during audit: {key}")
    result = {"schema": SCHEMA, "status": "passed", "audit_code_commit": manifest["audit_code_commit"],
        "model_execution_commit": MODEL_COMMIT, "compatible_parent_commit": PARENT_COMMIT,
        "manifest_sha256": sha(manifest_path), "source_files_verified": len(manifest["model_source_sha256"]),
        "input_files": manifest["input_files"], "physics": manifest["physics"],
        "old_child_checkpoint_id": child_status["checkpoint"]["id"], "new_model_checkpoint_id": new_checkpoint,
        "inherited_routes": parent_count, "inherited_routes_replayed": parent_count,
        "inherited_full_record_equality": True, "comparison_normalized_fields": ["cg_checkpoint_id"],
        "inherited_ordered_record_sha256": inherited_hash, "inheritance": metadata,
        "saved_child_routes": child_count, "singleton_routes": len(singletons),
        "singleton_trip_ids": singleton_ids, "missing_trip_ids": sorted(missing),
        "singleton_schedule_cost_metadata_replay_equal": True,
        "singleton_ordered_record_sha256": canonical(singleton_hashes),
        "all_routes_physically_replayed": replay_count, "solver_started": False, "graph_built": False,
        "fresh_singleton_optimum_equality_proved": False, "cg_pricing_certificate": None,
        "scope": "authenticated inherited-column equality and saved singleton replay; no fresh graph/pricing/optimality claim",
        "runtime_s": time.monotonic() - started,
        "maxrss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "maxrss_native_unit": "bytes" if sys.platform == "darwin" else "KiB",
        "input_problem_adjacency_arcs": sum(len(rows) for rows in problem.adjacency.values())}
    write_new_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_gate(args.manifest, args.input_root, args.model_root, args.out)
    print(json.dumps({key: result[key] for key in ("status", "inherited_routes", "singleton_routes", "runtime_s")}))


if __name__ == "__main__":
    main()

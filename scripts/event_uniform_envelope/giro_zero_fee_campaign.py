#!/usr/bin/env python3
"""Fee-only terminal-energy comparison around the corrected fair pilot.

This module is an experiment wrapper.  It does not change the solver modules
or the immutable source pools.  Fee-zero source copies retain the fee-five
route costs as explicit provenance and contain only destination-fee costs.
Both fee arms then use the union of the two fee-specific fixed-duty frontiers
at the destination fee, so the joint MIPs compare a common candidate set.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = ROOT / "scripts/event_uniform_envelope/terminal_energy_fair_pilot.py"
TARIFFS = ("peak08", "peak12", "peak18")
FEES = (0.0, 5.0)
SOURCE_FEE = 5.0
TARGET_KWH = 280.7833253
INSTANCE_REL = (
    "data/scale_ladder/instances/original_replay_eligible_20260908/"
    "Practice_Custom_DutyUnion_original_eligible_k05_20260908.csv"
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def finite(value, label: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite")
    return value


def fee_label(fee: float) -> str:
    if fee not in FEES:
        raise ValueError(f"unsupported fee {fee}; expected 0 or 5")
    return f"fee{int(fee)}"


def route_starts(route: dict) -> int:
    """Return starts from the expanded schedule, failing on inconsistent rows."""
    expanded = route.get("expanded_grid_charging_stops") or {}
    charging = route.get("charging_stops") or {}
    stations = expanded.get("stations")
    if stations is None:
        stations = charging.get("stations")
    if stations is None:
        raise ValueError("route has no charging-stop station list")
    count = len(stations)
    for stops in (expanded, charging):
        values = stops.get("stations")
        if values is not None and len(values) != count:
            raise ValueError("route charging-stop lists have inconsistent lengths")
    return count


def reprice_route(route: dict, source_fee: float, destination_fee: float) -> dict:
    """Reprice a saved route while preserving its complete source cost audit."""
    source_fee = finite(source_fee, "source fee")
    destination_fee = finite(destination_fee, "destination fee")
    if source_fee < 0 or destination_fee < 0:
        raise ValueError("charge-start fees must be nonnegative")
    result = copy.deepcopy(route)
    starts = route_starts(route)
    delta = (destination_fee - source_fee) * starts
    source_cost = finite(route.get("cost"), "route cost")
    result["source_charge_start_fee"] = source_fee
    result["source_charge_start_count"] = starts
    result["source_cost"] = source_cost
    result["destination_charge_start_fee"] = destination_fee
    result["charge_start_count"] = starts
    result["cost"] = source_cost + delta
    if route.get("expanded_grid_cost") is not None:
        result["source_expanded_grid_cost"] = finite(
            route["expanded_grid_cost"], "expanded-grid route cost"
        )
        result["expanded_grid_cost"] = result["source_expanded_grid_cost"] + delta
    if route.get("continuous_realized_cost") is not None:
        result["source_continuous_realized_cost"] = finite(
            route["continuous_realized_cost"], "continuous route cost"
        )
        result["continuous_realized_cost"] = (
            result["source_continuous_realized_cost"] + delta
        )
    return result


def _adjust_if_present(summary: dict, field: str, delta: float) -> None:
    if summary.get(field) is not None:
        summary[field] = finite(summary[field], field) + delta


def reprice_original(payload: dict, source_fee: float, destination_fee: float) -> dict:
    """Reprice original GIRO accounting fields; recorded events stay unchanged."""
    result = copy.deepcopy(payload)
    source_fee = finite(source_fee, "source fee")
    destination_fee = finite(destination_fee, "destination fee")
    if source_fee < 0 or destination_fee < 0:
        raise ValueError("charge-start fees must be nonnegative")
    summaries = result.get("summary")
    if not isinstance(summaries, list) or not summaries:
        raise ValueError("original GIRO record has no summary")
    total_starts = None
    for summary in summaries:
        starts = int(summary.get("charge_events", -1))
        if starts < 0:
            raise ValueError("original GIRO summary has no charge count")
        total_starts = starts if total_starts is None else total_starts
        if starts != total_starts:
            raise ValueError("tariff summaries disagree on original charge count")
        delta = (destination_fee - source_fee) * starts
        summary["source_charge_start_fee"] = source_fee
        summary["destination_charge_start_fee"] = destination_fee
        summary["charge_start_fees"] = destination_fee * starts
        for field in (
            "charging_cost_exact", "charging_cost_lower", "charging_cost_upper",
            "terminal_energy_normalized_charging_cost_lower",
            "terminal_energy_normalized_charging_cost_upper",
        ):
            _adjust_if_present(summary, field, delta)
    physics = result.setdefault("physics", {})
    physics["source_charge_start_cost"] = source_fee
    physics["charge_start_cost"] = destination_fee
    result["fee_repricing"] = {
        "source_charge_start_fee": source_fee,
        "destination_charge_start_fee": destination_fee,
        "recorded_charge_events_unchanged": True,
    }
    return result


def reprice_source_root(source_root: Path, destination_root: Path,
                        source_fee: float, destination_fee: float) -> dict:
    """Create an authenticated fee-specific copy of a saved source pool."""
    source_root = Path(source_root).expanduser().resolve()
    destination_root = Path(destination_root).expanduser().resolve()
    if destination_root.exists():
        raise FileExistsError(destination_root)
    required = [source_root / name for name in (
        "original.json", "snapshot.json", "snapshot.json.columns.jsonl",
    )]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError(f"incomplete saved pool at {source_root}")
    destination_root.mkdir(parents=True)
    original = json.loads(required[0].read_text())
    snapshot = json.loads(required[1].read_text())
    source_hashes = {path.name: digest(path) for path in required}
    record_count = 0
    with required[2].open() as source_stream, (destination_root / required[2].name).open("x") as dest_stream:
        for line_number, line in enumerate(source_stream, 1):
            if not line.endswith("\n"):
                raise ValueError(f"source journal line {line_number} is incomplete")
            route = json.loads(line)
            repriced = reprice_route(route, source_fee, destination_fee)
            dest_stream.write(json.dumps(repriced, sort_keys=True, allow_nan=False) + "\n")
            record_count += 1
    repriced_original = reprice_original(original, source_fee, destination_fee)
    atomic_json(destination_root / "original.json", repriced_original)
    snapshot["columns_journal"] = str(destination_root / required[2].name)
    snapshot["fee_repricing"] = {
        "source_charge_start_fee": finite(source_fee, "source fee"),
        "destination_charge_start_fee": finite(destination_fee, "destination fee"),
        "records_repriced": record_count,
        "source_hashes": source_hashes,
        "source_root": str(source_root),
    }
    atomic_json(destination_root / "snapshot.json", snapshot)
    manifest = {
        "schema": "evsp-dr-terminal-fee-repriced-source-v1",
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "source_charge_start_fee": finite(source_fee, "source fee"),
        "destination_charge_start_fee": finite(destination_fee, "destination fee"),
        "source_hashes": source_hashes,
        "destination_hashes": {
            name: digest(destination_root / name)
            for name in ("original.json", "snapshot.json", "snapshot.json.columns.jsonl")
        },
        "records_repriced": record_count,
        "source_route_costs_preserved_as_fields": True,
    }
    atomic_json(destination_root / "fee_repricing.json", manifest)
    return manifest


def source_files(root: Path) -> dict[str, str]:
    return {
        name: digest(root / name)
        for name in ("original.json", "snapshot.json", "snapshot.json.columns.jsonl")
    }


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    source_map: dict[str, Path] = {}
    for value in args.source:
        tariff, separator, source = value.partition("=")
        if not separator or tariff in source_map or tariff not in TARIFFS:
            raise ValueError("source must contain one unique peak08/12/18=TARIFF_ROOT pair")
        source_map[tariff] = Path(source).expanduser().resolve()
    if set(source_map) != set(TARIFFS):
        raise ValueError("all three tariff source roots are required")
    instance = ROOT / INSTANCE_REL
    cells = []
    for tariff in TARIFFS:
        source = source_map[tariff]
        original_path = source / "original.json"
        original = json.loads(original_path.read_text())
        observed = finite(original["summary"][0]["terminal_surplus_total_kwh"], "GIRO terminal energy")
        if not math.isclose(observed, TARGET_KWH, abs_tol=1e-7, rel_tol=0):
            raise ValueError(f"{tariff} source terminal total differs from {TARGET_KWH}")
        source_fee = finite(original.get("physics", {}).get("charge_start_cost", SOURCE_FEE), "source fee")
        if not math.isclose(source_fee, SOURCE_FEE, abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"{tariff} source fee is {source_fee}, expected {SOURCE_FEE}")
        fee0_root = root / "repriced_sources" / tariff / "fee0"
        reprice_manifest = reprice_source_root(source, fee0_root, SOURCE_FEE, 0.0)
        tariff_path = ROOT / f"data/tariff_response/{tariff}_h26.csv"
        for fee in FEES:
            source_root = fee0_root if fee == 0.0 else source
            pair_id = f"{tariff}_fee{int(fee)}"
            cells.append({
                "id": pair_id, "tariff": tariff, "fee": fee,
                "source_charge_start_fee": SOURCE_FEE,
                "source_root": str(source_root),
                "source_hashes": source_files(source_root),
                "instance": str(instance), "instance_sha256": digest(instance),
                "tariff_path": str(tariff_path), "tariff_sha256": digest(tariff_path),
                "source_repricing_manifest": (
                    str(fee0_root / "fee_repricing.json") if fee == 0.0 else None
                ),
                "source_repricing_manifest_sha256": (
                    digest(fee0_root / "fee_repricing.json") if fee == 0.0 else None
                ),
                "frontier_dir": str(root / "results" / tariff / fee_label(fee)),
            })
    plan = {
        "schema": "evsp-dr-terminal-energy-fee-comparison-v1",
        "root": str(root), "commit": args.commit,
        "python": "/home/nc437/evsp_env/bin/python",
        "cells": cells,
        "target_physical_terminal_energy_kwh": TARGET_KWH,
        "terminal_row_coefficient": "expanded_grid_terminal_soc_kwh",
        "aggregate_terminal_energy_required_in_both_models": True,
        "fleet_cap": 5, "initial_energy_per_bus_kwh": 240.0,
        "charge_power_kw": 350.0, "soc_step_kwh": 2.5, "block_minutes": 5,
        "source_pool_fee": SOURCE_FEE,
        "candidate_set": "common_union_of_fee0_and_fee5_frontiers_plus_repriced_saved_pool",
        "proof_scope": "finite_saved_pool_plus_common_frontier_union; no full-model pricing certificate",
        "resources": {
            "frontier": {"partition": "default_partition", "cpus": 1, "memory": "24G", "time": "02:00:00", "exclude": ["scaglione-compute-01"]},
            "mip": {"partition": "scaglione", "cpus": 8, "memory": "48G", "time": "02:00:00", "exclude": ["scaglione-compute-01", "scaglione-cpu-04"]},
        },
        "frontier_tasks": [cell["id"] for cell in cells],
        "mip_tasks": [cell["id"] for cell in cells],
        "repricing": {
            "source_fee": SOURCE_FEE,
            "fee0_source_copy": "repriced_sources/<tariff>/fee0",
            "fee5_source": "original immutable source root",
            "fee5_cost_metadata_preserved": True,
        },
    }
    atomic_json(root / "plan.json", plan)
    print(json.dumps({"root": str(root), "plan_sha256": digest(root / "plan.json"), "cells": len(cells)}))


def load_pilot():
    spec = importlib.util.spec_from_file_location("terminal_energy_fair_pilot", PILOT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(PILOT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_runtime_fee(fee: float):
    """Set the imported solver configuration before the pilot imports cost users."""
    sys.path.insert(0, str(ROOT / "src"))
    import config
    config.CHARGE_START_COST = finite(fee, "destination fee")
    return config


def read_plan(root: Path) -> dict:
    plan = json.loads((root / "plan.json").read_text())
    if plan.get("schema") != "evsp-dr-terminal-energy-fee-comparison-v1":
        raise ValueError("unexpected fee campaign plan schema")
    return plan


def find_cell(plan: dict, pair_id: str) -> dict:
    for cell in plan["cells"]:
        if cell["id"] == pair_id:
            return cell
    raise KeyError(pair_id)


def authenticate_cell(plan: dict, cell: dict) -> None:
    for key, hash_key in (("instance", "instance_sha256"),
                          ("tariff_path", "tariff_sha256")):
        path = Path(cell[key])
        if digest(path) != cell[hash_key]:
            raise ValueError(f"{key} hash mismatch: {path}")
    for name, expected in cell["source_hashes"].items():
        path = Path(cell["source_root"]) / name
        if digest(path) != expected:
            raise ValueError(f"source hash mismatch: {path}")
    manifest = cell.get("source_repricing_manifest")
    expected_manifest = cell.get("source_repricing_manifest_sha256")
    if manifest is not None:
        if expected_manifest is None or digest(Path(manifest)) != expected_manifest:
            raise ValueError(f"source repricing manifest hash mismatch: {manifest}")


def allocation(path: Path, plan: dict, cell: dict, stage: str) -> None:
    atomic_json(path, {
        "schema": "evsp-dr-terminal-energy-fee-allocation-v1",
        "pair_id": cell["id"], "tariff": cell["tariff"],
        "destination_charge_start_fee": cell["fee"],
        "source_charge_start_fee": SOURCE_FEE,
        "stage": stage, "commit": plan["commit"],
    })


def augment_frontier(path: Path, cell: dict, plan: dict) -> None:
    frontier = json.loads(path.read_text())
    frontier["fee_provenance"] = {
        "destination_charge_start_fee": cell["fee"],
        "source_pool_charge_start_fee": SOURCE_FEE,
        "fresh_frontier_generated_at_destination_fee": True,
        "aggregate_terminal_target_kwh": TARGET_KWH,
    }
    atomic_json(path, frontier)
    atomic_json(path.parent / "FRONTIER_COMPLETE.json", {"frontier_sha256": digest(path)})


def route_union(target: dict, sibling: dict, target_fee: float) -> tuple[dict, dict]:
    """Build target-fee frontiers from both fee-specific frontier artifacts."""
    if len(target["frontiers"]) != len(sibling["frontiers"]):
        raise ValueError("fee frontiers have different duty counts")
    union = []
    repriced = 0
    raw_counts = {}
    for target_duty, sibling_duty in zip(target["frontiers"], sibling["frontiers"]):
        if target_duty["duty_id"] != sibling_duty["duty_id"]:
            raise ValueError("fee frontiers have different duty IDs")
        routes = []
        for source_fee, duty in ((target_fee, target_duty), (5.0 if target_fee == 0.0 else 0.0, sibling_duty)):
            raw_counts[fee_label(source_fee)] = raw_counts.get(fee_label(source_fee), 0) + len(duty["routes"])
            for route in duty["routes"]:
                candidate = reprice_route(route, source_fee, target_fee)
                candidate["frontier_source_fee"] = source_fee
                candidate["frontier_repriced_for_fee"] = target_fee
                routes.append(candidate)
                repriced += int(not math.isclose(source_fee, target_fee, abs_tol=1e-12, rel_tol=0))
        seen = {}
        for route in routes:
            key = canonical({key: route.get(key) for key in (
                "trips", "route_nodes", "charging_stops",
                "expanded_grid_charging_stops", "cost",
            )})
            seen[key] = route
        union.append({"duty_id": target_duty["duty_id"], "routes": list(seen.values())})
    return (
        {"frontiers": union},
        {"repriced_routes": repriced, "raw_counts": raw_counts},
    )


def write_common_frontier(root: Path, cell: dict, plan: dict,
                          target_path: Path, sibling_path: Path,
                          joint_dir: Path) -> dict:
    target = json.loads(target_path.read_text())
    sibling = json.loads(sibling_path.read_text())
    union, audit = route_union(target, sibling, cell["fee"])
    combined = dict(target)
    combined["frontiers"] = union["frontiers"]
    combined["candidate_union"] = {
        "schema": "evsp-dr-terminal-common-frontier-union-v1",
        "target_fee": cell["fee"],
        "frontier_fee0_sha256": digest(target_path if cell["fee"] == 0.0 else sibling_path),
        "frontier_fee5_sha256": digest(sibling_path if cell["fee"] == 0.0 else target_path),
        "raw_route_counts_by_source_fee": audit["raw_counts"],
        "repriced_route_count": audit["repriced_routes"],
        "aggregate_terminal_target_kwh": TARGET_KWH,
        "saved_pool_repriced_to_target_fee": True,
    }
    joint_dir.mkdir(parents=True)
    atomic_json(joint_dir / "frontier.json", combined)
    atomic_json(joint_dir / "FRONTIER_COMPLETE.json", {"frontier_sha256": digest(joint_dir / "frontier.json")})
    return combined["candidate_union"]


def patch_mip_provenance(joint_dir: Path, cell: dict, union_audit: dict) -> None:
    mip_path = joint_dir / "mip.json"
    comparison_path = joint_dir / "comparison.json"
    mip = json.loads(mip_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    fee_audit = {
        "destination_charge_start_fee": cell["fee"],
        "source_pool_charge_start_fee": SOURCE_FEE,
        "saved_pool_repriced_to_destination_fee": True,
        "common_frontier_union": union_audit,
        "proof_scope": "finite_saved_pool_plus_common_fee_frontier_union",
    }
    mip["fee_provenance"] = fee_audit
    comparison["fee_provenance"] = fee_audit
    comparison.setdefault("unchanged_conditions", {})["charge_start_fee"] = cell["fee"]
    comparison["unchanged_conditions"]["source_pool_charge_start_fee"] = SOURCE_FEE
    for field in ("fixed_duties_optimized", "joint_pool_optimized"):
        selected = comparison.get(field, {}).get("selected_routes", [])
        comparison[field]["charge_start_count"] = sum(
            route_starts(route) for route in selected
        )
        comparison[field]["charge_start_counts_by_route"] = [
            {"duty_id": route.get("duty_id"), "count": route_starts(route)}
            for route in selected
        ]
    atomic_json(mip_path, mip)
    atomic_json(comparison_path, comparison)
    atomic_json(joint_dir / "COMPLETE.json", {
        "mip_sha256": digest(mip_path), "comparison_sha256": digest(comparison_path),
    })


def worker(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    plan = read_plan(root)
    cell = find_cell(plan, args.pair_id)
    authenticate_cell(plan, cell)
    fee = finite(cell["fee"], "destination fee")
    if not math.isclose(fee, float(args.fee), abs_tol=1e-12, rel_tol=0):
        raise ValueError("requested fee disagrees with frozen plan")
    set_runtime_fee(fee)
    pilot = load_pilot()
    pilot.check_identity(plan["commit"])
    base = Path(cell["frontier_dir"])
    if args.stage == "frontier":
        if base.exists():
            raise FileExistsError(base)
        base.mkdir(parents=True)
        allocation(base / "frontier.allocation.json", plan, cell, "frontier")
        pilot.frontier_worker(args, plan, cell, base)
        augment_frontier(base / "frontier.json", cell, plan)
        return
    if args.stage != "mip":
        raise ValueError(args.stage)
    sibling_fee = 5.0 if fee == 0.0 else 0.0
    sibling_id = f"{cell['tariff']}_fee{int(sibling_fee)}"
    sibling = find_cell(plan, sibling_id)
    target_frontier = base / "frontier.json"
    sibling_frontier = Path(sibling["frontier_dir"]) / "frontier.json"
    for path in (target_frontier, sibling_frontier):
        if not path.is_file():
            raise FileNotFoundError(f"required frontier is missing: {path}")
    joint_dir = base / "joint"
    union_audit = write_common_frontier(
        root, cell, plan, target_frontier, sibling_frontier, joint_dir,
    )
    allocation(joint_dir / "mip.allocation.json", plan, cell, "mip")
    pilot.mip_worker(args, plan, cell, joint_dir)
    patch_mip_provenance(joint_dir, cell, union_audit)


def frontier_command(plan: dict, cell: dict) -> list[str]:
    return [
        plan["python"], "-u", str(ROOT / "scripts/event_uniform_envelope/giro_zero_fee_campaign.py"),
        "worker", "--root", plan["root"], "--pair-id", cell["id"],
        "--fee", str(int(cell["fee"])), "--stage", "frontier",
    ]


def mip_command(plan: dict, cell: dict, dependencies: list[str] | None = None) -> list[str]:
    command = [
        plan["python"], "-u", str(ROOT / "scripts/event_uniform_envelope/giro_zero_fee_campaign.py"),
        "worker", "--root", plan["root"], "--pair-id", cell["id"],
        "--fee", str(int(cell["fee"])), "--stage", "mip",
    ]
    if dependencies:
        command.extend(["--requires-frontiers", *dependencies])
    return command


def command_plan(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    plan = read_plan(root)
    pairs = {cell["id"]: cell for cell in plan["cells"]}
    output = {"frontier_commands": [], "mip_commands": []}
    for tariff in TARIFFS:
        for fee in FEES:
            cell = pairs[f"{tariff}_fee{int(fee)}"]
            output["frontier_commands"].append(frontier_command(plan, cell))
        for fee in FEES:
            cell = pairs[f"{tariff}_fee{int(fee)}"]
            output["mip_commands"].append(mip_command(
                plan, cell,
                [f"{tariff}_fee0/frontier.json", f"{tariff}_fee5/frontier.json"],
            ))
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", required=True, type=Path)
    prep.add_argument("--commit", required=True)
    prep.add_argument("--source", action="append", required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--root", required=True, type=Path)
    worker_parser.add_argument("--pair-id", required=True)
    worker_parser.add_argument("--fee", required=True, type=float)
    worker_parser.add_argument("--stage", choices=("frontier", "mip"), required=True)
    worker_parser.add_argument("--requires-frontiers", nargs="*")
    commands = sub.add_parser("commands")
    commands.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "worker":
        worker(args)
    else:
        command_plan(args)


if __name__ == "__main__":
    main()

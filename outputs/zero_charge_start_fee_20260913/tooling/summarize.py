#!/usr/bin/env python3
"""Create paired fee-0 versus fee-5 CSV/Markdown summaries from collection JSON."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


ARMS = ("fee0", "fee5")


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def scalar(value):
    return value if value is None or isinstance(value, (str, int, float, bool)) else None


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def difference(left, right):
    return float(left) - float(right) if finite_number(left) and finite_number(right) else None


def index_unique(records, label):
    indexed = {}
    for record in records:
        key = (record.get("source_case_id"), record.get("arm"))
        if key[0] is None or key[1] not in ARMS:
            continue
        if key in indexed:
            raise ValueError(
                f"multiple {label} records for {key}; resolve attempts explicitly before pairing"
            )
        indexed[key] = record
    return indexed


def validate_fee_identity(record, arm, label):
    if record is None:
        return
    expected = 0.0 if arm == "fee0" else 5.0
    declared = record.get("charge_start_cost")
    actual = (
        nested(record, "physics", "charge_start_cost")
        if label == "MIP" else nested(record, "args", "charge_start_cost")
    )
    for name, value in (("declared", declared), ("actual", actual)):
        try:
            matches = math.isclose(
                float(value), expected, rel_tol=0.0, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(
                f"{label} {arm} {name} charge_start_cost is {value!r}, "
                f"expected {expected:g}"
            )


def arm_fields(cg, mip):
    metrics = (mip or {}).get("charging_comparison_metrics") or {}
    two_stage = (mip or {}).get("two_stage") or {}
    return {
        "cg_present": cg is not None,
        "cg_process_success": nested(cg, "process", "process_success"),
        "cg_certified_rc_optimal": (cg or {}).get("certified_rc_optimal"),
        "cg_pricing_scope": nested(cg, "proof_scope", "pricing"),
        "cg_stop_reason": (cg or {}).get("stop_reason"),
        "cg_iterations": (cg or {}).get("iterations"),
        "cg_columns": (cg or {}).get("columns"),
        "cg_runtime_s": (cg or {}).get("runtime_s"),
        "mip_present": mip is not None,
        "mip_process_success": nested(mip, "process", "process_success"),
        "mip_status": (mip or {}).get("status"),
        "mip_status_name": (mip or {}).get("status_name"),
        "mip_incumbent_found": (mip or {}).get("incumbent_found"),
        "buses": (mip or {}).get("buses"),
        "mip_obj": (mip or {}).get("mip_obj"),
        "mip_bound": (mip or {}).get("mip_bound"),
        "mip_gap": (mip or {}).get("mip_gap"),
        "finite_pool_optimal_scope": nested(mip, "proof_scope", "finite_pool_mip"),
        "mip_bound_scope": nested(mip, "proof_scope", "mip_bound_scope"),
        "fleet_proven": (mip or {}).get("fleet_proven"),
        "physical_replay_validated": (mip or {}).get("physical_replay_validated"),
        "duplicate_trip_removal_validated": (mip or {}).get(
            "duplicate_trip_removal_validated"
        ),
        "cross_route_charger_capacity_validated": (mip or {}).get(
            "cross_route_charger_capacity_validated"
        ),
        "mip_runtime_s": (mip or {}).get("runtime_s"),
        "stage2_executed": two_stage.get("stage2_executed"),
        "stage2_fleet_cap": two_stage.get("stage2_fleet_cap"),
        "stage2_fleet_cap_proven": two_stage.get("stage2_fleet_cap_proven"),
        "stage2_fleet_constraint": two_stage.get("stage2_fleet_constraint"),
        "charging_metrics_available": metrics.get("available"),
        "cost_components_reconcile": metrics.get("cost_components_reconcile"),
        "expanded_grid_charging_total": metrics.get("expanded_grid_charging_total"),
        "expanded_grid_electricity_cost": metrics.get("expanded_grid_electricity_cost"),
        "expanded_grid_start_fees": metrics.get("expanded_grid_start_fees"),
        "expanded_grid_charging_starts": metrics.get("expanded_grid_charging_starts"),
        "expanded_grid_charging_kwh": metrics.get("expanded_grid_charging_kwh"),
        "expanded_grid_terminal_kwh": metrics.get("expanded_grid_terminal_kwh"),
        "continuous_charging_total": metrics.get("continuous_charging_total"),
        "continuous_electricity_cost": metrics.get("continuous_electricity_cost"),
        "continuous_start_fees": metrics.get("continuous_start_fees"),
        "continuous_charging_starts": metrics.get("continuous_charging_starts"),
        "continuous_charging_kwh": metrics.get("continuous_charging_kwh"),
        "continuous_terminal_kwh": metrics.get("continuous_terminal_kwh"),
    }


def build_rows(collection):
    if collection.get("schema") != "evsp-zero-charge-start-fee-v1":
        raise ValueError("input is not an evsp-zero-charge-start-fee-v1 collection")
    cg = index_unique(collection.get("cg", []), "CG")
    mip = index_unique(collection.get("mip", []), "MIP")
    case_ids = sorted(
        {key[0] for key in cg}
        | {key[0] for key in mip}
        | {
            pair.get("case_id") for pair in collection.get("pairs", [])
            if pair.get("case_id") is not None
        }
    )
    rows = []
    delta_fields = (
        "buses", "mip_obj", "mip_runtime_s", "cg_runtime_s",
        "expanded_grid_charging_total", "expanded_grid_electricity_cost",
        "expanded_grid_start_fees", "expanded_grid_charging_starts",
        "expanded_grid_charging_kwh", "expanded_grid_terminal_kwh",
        "continuous_charging_total", "continuous_electricity_cost",
        "continuous_start_fees", "continuous_charging_starts",
        "continuous_charging_kwh", "continuous_terminal_kwh",
    )
    for case_id in case_ids:
        for arm in ARMS:
            validate_fee_identity(cg.get((case_id, arm)), arm, "CG")
            validate_fee_identity(mip.get((case_id, arm)), arm, "MIP")
        arm_values = {
            arm: arm_fields(cg.get((case_id, arm)), mip.get((case_id, arm)))
            for arm in ARMS
        }
        target_k = next((
            nested(cg.get((case_id, arm)), "input", "target_k")
            for arm in ARMS
            if nested(cg.get((case_id, arm)), "input", "target_k") is not None
        ), None)
        row = {"source_case_id": case_id, "target_k": target_k}
        for arm in ARMS:
            for key, value in arm_values[arm].items():
                row[f"{arm}_{key}"] = scalar(value)
        for key in delta_fields:
            row[f"delta_fee0_minus_fee5_{key}"] = difference(
                arm_values["fee0"].get(key), arm_values["fee5"].get(key)
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows):
    fields = list(rows[0]) if rows else ["source_case_id", "target_k"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def display(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def write_markdown(path: Path, rows, collection_path: Path):
    columns = (
        ("Case", "source_case_id"), ("k", "target_k"),
        ("Buses 0", "fee0_buses"), ("Buses 5", "fee5_buses"),
        ("Δ buses", "delta_fee0_minus_fee5_buses"),
        ("Starts 0", "fee0_continuous_charging_starts"),
        ("Starts 5", "fee5_continuous_charging_starts"),
        ("Δ starts", "delta_fee0_minus_fee5_continuous_charging_starts"),
        ("Elec. 0", "fee0_continuous_electricity_cost"),
        ("Elec. 5", "fee5_continuous_electricity_cost"),
        ("Δ elec.", "delta_fee0_minus_fee5_continuous_electricity_cost"),
        ("CG cert. 0/5", None), ("Fleet proof 0/5", None),
        ("Physical 0/5", None),
    )
    lines = [
        "# Paired charge-start-fee comparison",
        "",
        f"Source collection: `{collection_path}`",
        "",
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for label, key in columns:
            if key is not None:
                values.append(display(row.get(key)))
            elif label == "CG cert. 0/5":
                values.append(
                    f"{display(row.get('fee0_cg_certified_rc_optimal'))}/"
                    f"{display(row.get('fee5_cg_certified_rc_optimal'))}"
                )
            elif label == "Fleet proof 0/5":
                values.append(
                    f"{display(row.get('fee0_fleet_proven'))}/"
                    f"{display(row.get('fee5_fleet_proven'))}"
                )
            else:
                values.append(
                    f"{display(row.get('fee0_physical_replay_validated'))}/"
                    f"{display(row.get('fee5_physical_replay_validated'))}"
                )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        "All deltas are fee 0 minus fee 5. Process success, CG pricing certification, finite-pool MIP scope, fleet proof, and physical replay remain separate CSV fields.",
        "",
        "Charging totals are reported for selected routes before duplicate-trip removal. The baseline model does not validate cross-route charger capacity. A missing value is shown as an em dash and is never treated as zero.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.csv.resolve() == args.markdown.resolve():
        parser.error("--csv and --markdown must be different paths")
    collection = read_json(args.collection)
    rows = build_rows(collection)
    write_csv(args.csv, rows)
    write_markdown(args.markdown, rows, args.collection)
    print(json.dumps({
        "collection": str(args.collection), "rows": len(rows),
        "csv": str(args.csv), "markdown": str(args.markdown),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

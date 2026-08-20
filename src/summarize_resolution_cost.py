#!/usr/bin/env python3
"""Summarize resolution-cost cells and extrapolate exact-CG scaling."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from resolution_cost_study import SCHEMA, _sha256_json


LONG_FIELDS = (
    "job_key", "cell_id", "scale", "selection_replicate",
    "physics_profile", "g_kwh", "charge_kw", "soc_step", "block_min",
    "grid_id", "grid_role", "commensurate", "charge_kwh_per_block",
    "credited_charge_kwh_per_block", "charge_grid_loss_kwh",
    "trip_count", "target_fleet", "instance_sha256",
    "status_path", "status_present", "stop_reason", "certified",
    "dag_nodes", "dag_arcs", "dag_build_wall_s", "estimated_dag_nodes_upper",
    "peak_rss_mb", "iterations_to_certificate", "wall_to_certificate",
    "iterations_observed", "master_wall_s_total", "pricing_wall_s_total",
    "columns_added_total", "pool_columns_final", "fleet_lp_bound",
    "lp_gap_buses", "integer_fleet", "integer_fleet_proven",
    "integer_gap_buses", "mip_gap", "lp_gap_closed_vs_anchor",
    "integer_gap_closed_vs_anchor", "cpu_hours_to_certificate",
    "buses_gap_closed_per_cpu_hour",
)


def _number(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _key(instance_sha, physics):
    return (
        instance_sha,
        _number(physics.get("g_kwh")),
        _number(physics.get("charge_kw")),
        _number(physics.get("min_soc_frac")),
        _number(physics.get("soc_step")),
        _number(physics.get("block_min")),
    )


def load_mip_index(roots):
    index = {}
    for root in roots or []:
        root = Path(root)
        paths = root.rglob("*.json") if root.is_dir() else [root]
        for path in paths:
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if "buses" not in payload or not isinstance(
                payload.get("physics"), dict
            ):
                continue
            provenance = payload.get("pricer_provenance") or {}
            identity = _key(
                provenance.get("instance_sha256"), payload["physics"],
            )
            if identity[0] is None:
                continue
            candidate = {
                "integer_fleet": payload.get("buses"),
                "integer_fleet_proven": bool(payload.get("fleet_proven")),
                "mip_gap": payload.get("mip_gap"),
                "path": str(path),
            }
            current = index.get(identity)
            if current is None or (
                candidate["integer_fleet_proven"]
                and not current["integer_fleet_proven"]
            ):
                index[identity] = candidate
    return index


def _fleet_bound(status):
    if not status.get("certified", status.get("certified_rc_optimal", False)):
        return None
    direct = _number(status.get("phase_2_fleet_lp_bound"))
    if direct is not None:
        return direct
    final_lp = status.get("final_lp") or {}
    return _number(final_lp.get("route_weight"))


def _status_row(job, mip_index):
    path = Path(job["output"])
    status = {}
    if path.is_file():
        try:
            status = json.loads(path.read_text())
        except (OSError, ValueError):
            status = {"stop_reason": "invalid_status_json"}
    metrics = status.get("iteration_metrics")
    metrics = metrics if isinstance(metrics, list) else []
    certified = bool(status.get(
        "certified", status.get("certified_rc_optimal", False),
    ))
    bound = _fleet_bound(status)
    mip = mip_index.get(_key(job["instance_sha256"], job))
    integer = _number(mip.get("integer_fleet")) if mip else None
    wall = _number(status.get("wall_to_certificate"))
    row = {
        **{key: job.get(key) for key in LONG_FIELDS if key in job},
        "status_path": str(path),
        "status_present": path.is_file(),
        "stop_reason": status.get(
            "stop_reason", "missing" if not path.is_file() else None,
        ),
        "certified": certified,
        "dag_nodes": status.get("dag_nodes"),
        "dag_arcs": status.get("dag_arcs"),
        "dag_build_wall_s": status.get("dag_build_wall_s"),
        "peak_rss_mb": status.get("peak_rss_mb"),
        "iterations_to_certificate": status.get("iterations_to_certificate"),
        "wall_to_certificate": wall,
        "iterations_observed": len(metrics),
        "master_wall_s_total": sum(
            _number(item.get("master_wall_s")) or 0.0 for item in metrics
            if isinstance(item, dict)
        ),
        "pricing_wall_s_total": sum(
            _number(item.get("pricing_wall_s")) or 0.0 for item in metrics
            if isinstance(item, dict)
        ),
        "columns_added_total": sum(
            int(item.get("columns_added") or 0) for item in metrics
            if isinstance(item, dict)
        ),
        "pool_columns_final": status.get(
            "pool_columns_final", status.get("columns"),
        ),
        "fleet_lp_bound": bound,
        "lp_gap_buses": (
            max(0.0, bound - job["target_fleet"])
            if bound is not None else None
        ),
        "integer_fleet": integer,
        "integer_fleet_proven": (
            mip["integer_fleet_proven"] if mip else None
        ),
        "integer_gap_buses": (
            integer - job["target_fleet"] if integer is not None else None
        ),
        "mip_gap": mip["mip_gap"] if mip else None,
        "cpu_hours_to_certificate": wall / 3600.0 if wall else None,
    }
    return row


def build_rows(plan, mip_roots=None):
    mip_index = load_mip_index(mip_roots)
    rows = [_status_row(job, mip_index) for job in plan["jobs"]]
    anchors = {}
    for row in rows:
        if row["soc_step"] == 15.0 and row["block_min"] == 10.0:
            anchors[(row["cell_id"], row["physics_profile"])] = row
    for row in rows:
        anchor = anchors.get((row["cell_id"], row["physics_profile"]))
        anchor_bound = anchor.get("fleet_lp_bound") if anchor else None
        anchor_integer = anchor.get("integer_fleet") if anchor else None
        row["lp_gap_closed_vs_anchor"] = (
            anchor_bound - row["fleet_lp_bound"]
            if anchor_bound is not None and row["fleet_lp_bound"] is not None
            else None
        )
        row["integer_gap_closed_vs_anchor"] = (
            anchor_integer - row["integer_fleet"]
            if anchor_integer is not None and row["integer_fleet"] is not None
            else None
        )
        hours = row["cpu_hours_to_certificate"]
        closed = row["lp_gap_closed_vs_anchor"]
        row["buses_gap_closed_per_cpu_hour"] = (
            closed / hours
            if closed is not None and hours is not None and hours > 0 else None
        )
    return rows


def fit_log_model(rows, response, physics_profile="p240"):
    usable = [
        row for row in rows
        if row["physics_profile"] == physics_profile
        and _number(row.get(response)) not in {None, 0.0}
        and row["trip_count"] > 0
        and row["soc_step"] > 0
        and row["block_min"] > 0
    ]
    if len(usable) < 4:
        return {"status": "insufficient_data", "n": len(usable)}
    matrix = np.asarray([
        [1.0, math.log(row["trip_count"]),
         math.log(1.0 / row["soc_step"]),
         math.log(1.0 / row["block_min"])]
        for row in usable
    ])
    values = np.log(np.asarray([float(row[response]) for row in usable]))
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        matrix, values, rcond=None,
    )
    fitted = matrix @ coefficients
    total = float(np.sum((values - np.mean(values)) ** 2))
    residual = float(np.sum((values - fitted) ** 2))
    return {
        "status": "fit" if rank == 4 else "rank_deficient",
        "n": len(usable), "rank": int(rank),
        "intercept": float(coefficients[0]),
        "trips_exponent": float(coefficients[1]),
        "inverse_soc_step_exponent": float(coefficients[2]),
        "inverse_block_min_exponent": float(coefficients[3]),
        "r_squared": 1.0 - residual / total if total > 0 else 1.0,
        "training_ranges": {
            "trips": [
                min(row["trip_count"] for row in usable),
                max(row["trip_count"] for row in usable),
            ],
            "soc_step": [
                min(row["soc_step"] for row in usable),
                max(row["soc_step"] for row in usable),
            ],
            "block_min": [
                min(row["block_min"] for row in usable),
                max(row["block_min"] for row in usable),
            ],
        },
    }


def predict(model, trips, soc_step, block_min):
    if model.get("status") != "fit":
        return None
    log_value = (
        model["intercept"]
        + model["trips_exponent"] * math.log(trips)
        + model["inverse_soc_step_exponent"] * math.log(1.0 / soc_step)
        + model["inverse_block_min_exponent"] * math.log(1.0 / block_min)
    )
    return math.exp(log_value)


def summarize(plan, output_dir, mip_roots=None, affordable_hours=None):
    rows = build_rows(plan, mip_roots)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "resolution_cost_long.csv"
    with long_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LONG_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in LONG_FIELDS} for row in rows)
    node_model = fit_log_model(rows, "dag_nodes")
    wall_model = fit_log_model([
        row for row in rows if row["certified"]
    ], "wall_to_certificate")
    bridge_node_model = fit_log_model(
        rows, "dag_nodes", physics_profile="p300_bridge",
    )
    bridge_wall_model = fit_log_model(
        [row for row in rows if row["certified"]],
        "wall_to_certificate", physics_profile="p300_bridge",
    )
    target = plan["prediction_target"]
    predicted_nodes = predict(
        node_model, target["trip_count"], target["soc_step"], target["block_min"],
    )
    predicted_wall = predict(
        wall_model, target["trip_count"], target["soc_step"], target["block_min"],
    )
    threshold = float(
        affordable_hours
        if affordable_hours is not None
        else target["affordable_wall_hours"]
    )
    extrapolation = {
        "schema": "evsp-dr-resolution-cost-extrapolation-v1",
        "node_model": node_model,
        "wall_model": wall_model,
        "bridge_models": {
            "node_model": bridge_node_model,
            "wall_model": bridge_wall_model,
        },
        "prediction_target": target,
        "predicted_dag_nodes": predicted_nodes,
        "predicted_wall_to_certificate_s": predicted_wall,
        "predicted_wall_to_certificate_h": (
            predicted_wall / 3600.0 if predicted_wall is not None else None
        ),
        "affordable_threshold_h": threshold,
        "affordable": (
            predicted_wall / 3600.0 <= threshold
            if predicted_wall is not None else None
        ),
        "extrapolation_warning":
            "Target 947 trips and 1 kWh are outside the local fitted ranges "
            "(k2/k3: 29-71 trips; 2.5-15 kWh); wall fit uses certified, "
            "hence selected, cells only.",
        "rows": len(rows),
        "certified_rows": sum(bool(row["certified"]) for row in rows),
    }
    prediction_path = output_dir / "resolution_cost_extrapolation.json"
    prediction_path.write_text(
        json.dumps(extrapolation, indent=2, sort_keys=True) + "\n"
    )
    return long_path, prediction_path, extrapolation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mip-root", type=Path, action="append")
    parser.add_argument("--affordable-hours", type=float)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != SCHEMA:
        raise ValueError("unexpected study plan schema")
    if _sha256_json({
        key: value for key, value in plan.items() if key != "plan_sha256"
    }) != plan.get("plan_sha256"):
        raise ValueError("study plan hash mismatch")
    long_path, prediction_path, extrapolation = summarize(
        plan, args.output_dir, args.mip_root, args.affordable_hours,
    )
    print(json.dumps({
        "long_csv": str(long_path),
        "extrapolation": str(prediction_path),
        "predicted_wall_h": extrapolation["predicted_wall_to_certificate_h"],
        "affordable": extrapolation["affordable"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

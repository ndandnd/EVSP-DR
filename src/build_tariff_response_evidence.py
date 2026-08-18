#!/usr/bin/env python3
"""Build deterministic normalized tariff-response evidence from verified cells."""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from build_tariff_response_manifest import REPO_ROOT, sha256_file
from expanded_path_realization import charging_block_schedule_sha256
from tariff_response_core import (
    PHYSICS,
    load_tariff_manifest,
    route_response,
    savings_decomposition,
    tariff_prices,
)


SCHEMA = "evsp-dr-tariff-response-experiment-v1"
TIERS = {
    "TIER0_GIRO_ORIGINAL",
    "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
    "TIER2_RAW_ROUTE_CHARGING",
    "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING",
}
STATION_COLORS = {
    "PARX": "#1f77b4",
    "JON_A": "#ff7f0e",
    "2190L": "#2ca02c",
    "4808": "#d62728",
    "3127L": "#9467bd",
    "7880C": "#8c564b",
}
SUMMARY_FIELDS = (
    "cell_id", "instance_id", "scale", "tariff_id", "tariff_sha256",
    "tier", "treatment", "buses", "grid_model_objective",
    "continuous_replay_objective", "charging_cost",
    "continuous_charging_cost", "total_charged_kwh",
    "peak_window_kwh", "charging_kwh_by_hour_json",
    "charging_kwh_by_station_json", "charging_starts_by_hour_json",
    "waiting_min", "deadhead_min", "deadhead_kwh",
    "charging_stops", "discretized_certification_status",
    "physical_replay_status", "terminal_soc_policy",
    "runtime_preprocessing_s", "runtime_master_s", "runtime_pricing_s",
    "runtime_postprocessing_s", "certificate_scope",
    "continuous_cost_pricing_certified", "charging_only_savings_grid",
    "rerouting_increment_grid", "total_price_aware_savings_grid",
    "charging_only_savings_continuous",
    "rerouting_increment_continuous",
    "total_price_aware_savings_continuous",
    "availability", "availability_reason",
)
BLOCK_FIELDS = (
    "cell_id", "instance_id", "tariff_id", "tier", "treatment",
    "route_order", "route_id", "stop_index", "block_index", "station",
    "start_min", "end_min", "realized_kwh", "expanded_grid_kwh",
    "price_per_kwh", "tariff_hour", "tariff_key",
)
ROUTE_FIELDS = (
    "instance_id", "tariff_id", "treatment", "trip_count",
    "baseline_buses", "candidate_buses",
    "trips_assigned_to_different_duty",
    "percent_trips_assigned_to_different_duty",
    "predecessor_changes", "successor_changes",
    "trip_adjacency_jaccard", "trip_coassignment_jaccard",
    "intact_giro_duties_retained", "duties_split", "duties_merged",
    "selected_giro_columns", "newly_generated_columns",
    "baseline_route_identity_sha256",
    "candidate_route_identity_sha256", "deadhead_kwh_change",
    "deadhead_min_change",
)
CERT_FIELDS = (
    "cell_id", "instance_id", "tariff_id", "duty_id",
    "certified", "scope", "certificate_sha256",
    "continuous_cost_optimality_certified",
)
CG_FIELDS = (
    "cell_id", "instance_id", "scale", "tariff_id", "treatment",
    "iteration", "elapsed_s", "lp_obj", "route_weight",
    "artificials", "min_rc", "pool_columns",
)
MIP_FIELDS = (
    "cell_id", "instance_id", "scale", "tariff_id", "treatment",
    "checkpoint_elapsed_s", "incumbent_fleet", "fleet_bound",
    "fleet_gap", "node_count", "solution_count",
    "route_vector_sha256", "solver_ended_before_checkpoint",
)
INVENTORY_FIELDS = (
    "role", "cell_id", "path", "sha256", "size_bytes",
)


def _canonical(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _write_csv(path, fields, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _rename_noreplace(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise OSError("renameat2 is unavailable")
    function.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    if function(
        -100, os.fsencode(source), -100, os.fsencode(target), 1
    ) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(target)
        raise OSError(error, os.strerror(error), target)


def _finite(value, field):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} is non-finite")
    return number


def _source_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _validate_routes(cell, tariff):
    routes = cell.get("routes")
    trip_ids = cell.get("trip_ids")
    if not isinstance(routes, list) or not isinstance(trip_ids, list):
        raise ValueError(f"{cell['cell_id']} lacks route/trip arrays")
    counts = Counter(
        trip for route in routes for trip in route.get("trips", [])
    )
    if (
        set(counts) != set(trip_ids)
        or any(count != 1 for count in counts.values())
        or sum(counts.values()) != len(trip_ids)
    ):
        raise ValueError(f"{cell['cell_id']} is not an exact partition")
    if len(routes) != int(cell["metrics"]["buses"]):
        raise ValueError(f"{cell['cell_id']} bus count differs")
    for route in routes:
        blocks = route.get("continuous_realized_charging_blocks")
        stations = (
            route.get("charging_stops") or {}
        ).get("stations") or []
        if cell["tier"] != "TIER0_GIRO_ORIGINAL":
            if cell.get("physical_replay_status") != "validated_all_routes":
                raise ValueError(f"{cell['cell_id']} lacks physical replay")
            if not isinstance(blocks, list):
                raise ValueError(
                    f"{cell['cell_id']} selected route lacks blocks"
                )
            if stations and not blocks:
                positive = sum(
                    float(value) for value in (
                        (route.get("charging_stops") or {}).get("kwh")
                        or []
                    )
                )
                if positive > 1e-9:
                    raise ValueError(
                        f"{cell['cell_id']} charging route lacks blocks"
                    )
            expected = route.get(
                "continuous_realized_charging_blocks_sha256"
            ) or (route.get("physical_realization") or {}).get(
                "continuous_realized_charging_blocks_sha256"
            )
            if charging_block_schedule_sha256(blocks) != expected:
                raise ValueError(f"{cell['cell_id']} block hash mismatch")
            for block in blocks:
                if (
                    block.get("tariff_key") is None
                    or block.get("price_per_kwh") is None
                    or block.get("station") is None
                    or float(block["end_min"]) <= float(block["start_min"])
                ):
                    raise ValueError(
                        f"{cell['cell_id']} block provenance invalid"
                    )
        if (
            route.get("cost_tariff_sha256") not in {None, tariff["sha256"]}
            or (
                cell["tier"] != "TIER0_GIRO_ORIGINAL"
                and route.get("cost_tariff_sha256") != tariff["sha256"]
            )
        ):
            raise ValueError(
                f"{cell['cell_id']} reused a different tariff cost"
            )


def _validate_cell(cell, tariffs):
    required = {
        "cell_id", "instance_id", "scale", "tariff_id", "tier",
        "treatment", "trip_ids", "routes", "metrics", "tariff_sha256",
        "physical_replay_status", "terminal_soc_policy",
        "continuous_cost_pricing_certified", "certificate_scope",
    }
    if required - set(cell):
        raise ValueError(f"cell fields missing: {required - set(cell)}")
    if cell["tier"] not in TIERS:
        raise ValueError("unknown tier")
    tariff = tariffs.get(cell["tariff_id"])
    if tariff is None or cell["tariff_sha256"] != tariff["sha256"]:
        raise ValueError(f"{cell['cell_id']} tariff identity mismatch")
    if cell["continuous_cost_pricing_certified"] is not False:
        raise ValueError("continuous-cost certificate is prohibited")
    if cell["terminal_soc_policy"] != PHYSICS["terminal_soc_policy"]:
        if cell["tier"] != "TIER0_GIRO_ORIGINAL":
            raise ValueError("terminal SOC policy differs")
    if (
        cell["tier"].startswith("TIER2_")
        and cell["treatment"] not in {"RAW", "GIRO40-AUGMENTED"}
    ):
        raise ValueError("route-flexible result has a fixed-duty label")
    if (
        cell["tier"] == "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING"
        and cell["treatment"] != "GIRO40-AUGMENTED"
    ):
        raise ValueError("augmented Tier 2 treatment mislabeled")
    _validate_routes(cell, tariff)
    metrics = cell["metrics"]
    for field in (
        "buses", "total_charged_kwh", "waiting_min",
        "deadhead_min", "deadhead_kwh", "charging_stops",
        "runtime_preprocessing_s", "runtime_master_s",
        "runtime_pricing_s", "runtime_postprocessing_s",
    ):
        _finite(metrics[field], field)
    for field in (
        "charging_kwh_by_hour_json",
        "charging_kwh_by_station_json",
        "charging_starts_by_hour_json",
    ):
        value = metrics.get(field)
        if cell["tier"] == "TIER0_GIRO_ORIGINAL" and value is None:
            continue
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{field} is unavailable/invalid") from exc
        if not isinstance(decoded, dict):
            raise ValueError(f"{field} is not a mapping")
    return tariff


def _stable_orders(baseline, routes):
    baseline_sets = [set(route["trips"]) for route in baseline]
    order = {}
    for route in routes:
        route_set = set(route["trips"])
        score = min(
            (
                -len(route_set & baseline_set),
                index,
            )
            for index, baseline_set in enumerate(baseline_sets)
        )
        order[route["route_id"]] = (
            score[1], min(route_set), len(route_set), route["route_id"]
        )
    return {
        route_id: index for index, (route_id, _key) in enumerate(
            sorted(order.items(), key=lambda item: item[1])
        )
    }


def _figures(staging, cells, tariffs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(dict)
    for cell in cells:
        grouped[(cell["instance_id"], cell["tariff_id"])][
            cell["tier"]
        ] = cell
    complete = [
        (key, tiers)
        for key, tiers in sorted(grouped.items())
        if {
            "TIER0_GIRO_ORIGINAL",
            "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
            "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING",
        } <= set(tiers)
        and all(
            any(route.get("trip_blocks") for route in cell["routes"])
            for cell in tiers.values()
        )
    ]
    if not complete:
        raise ValueError("no complete cell with Gantt trip blocks")
    (instance_id, tariff_id), tiers = complete[0]
    panel_tiers = (
        "TIER0_GIRO_ORIGINAL",
        "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
        "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING",
    )
    baseline = tiers[panel_tiers[0]]["routes"]
    gantt_rows = []
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    tariff = tariffs[tariff_id]
    curves = tariff_prices(tariff)
    average_curve = {
        hour: sum(curve[hour] for curve in curves.values()) / len(curves)
        for hour in sorted(set.intersection(*(
            set(curve) for curve in curves.values()
        )))
    }
    low, high = min(average_curve.values()), max(average_curve.values())
    for panel, tier in enumerate(panel_tiers):
        cell = tiers[tier]
        order = _stable_orders(baseline, cell["routes"])
        ax = axes[panel]
        for route in cell["routes"]:
            y = order[route["route_id"]]
            for trip_block in route["trip_blocks"]:
                start = float(trip_block["start_min"])
                end = float(trip_block["end_min"])
                ax.broken_barh(
                    [(start, end - start)], (y - 0.35, 0.7),
                    facecolors="#b7b7b7", edgecolors="#555555",
                    linewidth=0.25,
                )
                gantt_rows.append({
                    "instance_id": instance_id,
                    "tariff_id": tariff_id,
                    "tier": tier,
                    "route_order": y,
                    "event_kind": "trip",
                    "trip_id": trip_block["trip_id"],
                    "station": "",
                    "start_min": start,
                    "end_min": end,
                    "kwh": "",
                    "price_per_kwh": "",
                })
            all_blocks = route.get(
                "continuous_realized_charging_blocks"
            ) or route.get("recorded_charging_blocks") or []
            for block in all_blocks:
                station = base_station(block["station"])
                start = float(block["start_min"])
                end = float(block["end_min"])
                price = block.get("price_per_kwh")
                intensity = (
                    0.35 if price is None or high == low
                    else 0.25 + 0.75 * (float(price) - low) / (high - low)
                )
                ax.broken_barh(
                    [(start, end - start)], (y - 0.45, 0.9),
                    facecolors=STATION_COLORS[station],
                    alpha=intensity, edgecolors="black", linewidth=0.35,
                )
                gantt_rows.append({
                    "instance_id": instance_id,
                    "tariff_id": tariff_id,
                    "tier": tier,
                    "route_order": y,
                    "event_kind": "charge",
                    "trip_id": "",
                    "station": station,
                    "start_min": start,
                    "end_min": end,
                    "kwh": block.get(
                        "realized_kwh", block.get("kwh")
                    ),
                    "price_per_kwh": price,
                })
        ax.set_ylabel(tier.split("_", 1)[0])
        ax.set_title(tier.replace("_", " "))
        price_ax = ax.twinx()
        hours = sorted(average_curve)
        price_ax.step(
            [hour * 60 for hour in hours],
            [average_curve[hour] for hour in hours],
            where="post", color="#222222", linewidth=0.8, alpha=0.55,
        )
        price_ax.set_ylabel("price", fontsize=7)
        price_ax.tick_params(axis="y", labelsize=6)
    axes[-1].set_xlabel("minute of service day")
    fig.suptitle(f"{instance_id}: {tariff_id}")
    fig.tight_layout()
    png = staging / "gantt_three_tiers.png"
    pdf = staging / "gantt_three_tiers.pdf"
    fig.savefig(png, dpi=180, metadata={"Software": "EVSP-DR"})
    fig.savefig(pdf, metadata={
        "Creator": "EVSP-DR", "CreationDate": None, "ModDate": None,
    })
    plt.close(fig)
    _write_csv(
        staging / "gantt_plot.csv",
        (
            "instance_id", "tariff_id", "tier", "route_order",
            "event_kind", "trip_id", "station", "start_min", "end_min",
            "kwh", "price_per_kwh",
        ),
        gantt_rows,
    )

    alpha_rows = []
    for cell in cells:
        tariff = tariffs[cell["tariff_id"]]
        if tariff["alpha_family"] != "peak12":
            continue
        metrics = cell["metrics"]
        alpha_rows.append({
            "instance_id": cell["instance_id"],
            "tariff_id": cell["tariff_id"],
            "tier": cell["tier"],
            "treatment": cell["treatment"],
            "alpha": float(tariff["alpha"]),
            "peak_window_kwh": metrics.get("peak_window_kwh"),
            "charging_cost": metrics.get("charging_cost"),
            "route_similarity": cell.get("route_similarity"),
            "buses": metrics.get("buses"),
            "deadhead_kwh": metrics.get("deadhead_kwh"),
        })
    required_alpha = {0.0, 0.25, 0.5, 1.0, 2.0}
    if not required_alpha <= {row["alpha"] for row in alpha_rows}:
        raise ValueError("alpha response cells are incomplete")
    response_fields = (
        "instance_id", "tariff_id", "tier", "treatment", "alpha",
        "peak_window_kwh", "charging_cost", "route_similarity",
        "buses", "deadhead_kwh",
    )
    _write_csv(
        staging / "tariff_response_plot.csv",
        response_fields,
        sorted(alpha_rows, key=lambda row: (
            row["tier"], row["treatment"], row["alpha"]
        )),
    )
    metrics = (
        ("peak_window_kwh", "Peak-window kWh"),
        ("charging_cost", "Charging cost"),
        ("route_similarity", "Trip co-assignment similarity"),
        ("buses", "Buses"),
        ("deadhead_kwh", "Deadhead kWh"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    grouped_rows = defaultdict(list)
    for row in alpha_rows:
        grouped_rows[(row["tier"], row["treatment"])].append(row)
    for axis, (field, label) in zip(axes.flat, metrics):
        for key, rows in sorted(grouped_rows.items()):
            rows.sort(key=lambda row: row["alpha"])
            if any(row[field] is None for row in rows):
                continue
            axis.plot(
                [row["alpha"] for row in rows],
                [float(row[field]) for row in rows],
                marker="o", label=" / ".join(key),
            )
        axis.set(xlabel="peak amplitude α", ylabel=label)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(
        staging / "price_amplitude_response.png",
        dpi=180, metadata={"Software": "EVSP-DR"},
    )
    fig.savefig(
        staging / "price_amplitude_response.pdf",
        metadata={
            "Creator": "EVSP-DR", "CreationDate": None, "ModDate": None,
        },
    )
    plt.close(fig)


def base_station(value):
    for station in STATION_COLORS:
        if value == station or str(value).startswith(station + "_"):
            return station
    raise ValueError(f"unknown station: {value}")


def build(manifest_path: Path, output_dir: Path) -> dict:
    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output exists: {output_dir}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected experiment manifest schema")
    if manifest.get("physics") != PHYSICS:
        raise ValueError("experiment physics/terminal policy changed")
    tariff_manifest = _source_path(manifest["tariff_manifest"])
    if sha256_file(tariff_manifest) != manifest[
        "tariff_manifest_sha256"
    ]:
        raise ValueError("tariff manifest hash mismatch")
    tariff_rows = load_tariff_manifest(tariff_manifest)
    tariffs = {row["tariff_id"]: row for row in tariff_rows}
    cells = sorted(
        manifest.get("cells") or [], key=lambda cell: cell["cell_id"]
    )
    if not cells or len({cell["cell_id"] for cell in cells}) != len(cells):
        raise ValueError("experiment cells are empty/duplicated")
    for cell in cells:
        _validate_cell(cell, tariffs)

    summaries = []
    block_rows = []
    certificates = []
    cg_rows = []
    mip_rows = []
    route_rows = []
    by_key = defaultdict(dict)
    for cell in cells:
        by_key[(cell["instance_id"], cell["tariff_id"])][
            cell["tier"]
        ] = cell
        row = {
            key: cell.get(key) for key in SUMMARY_FIELDS
        }
        row.update(cell["metrics"])
        row.update({
            "tariff_sha256": cell["tariff_sha256"],
            "physical_replay_status": cell[
                "physical_replay_status"
            ],
            "terminal_soc_policy": cell["terminal_soc_policy"],
            "certificate_scope": cell["certificate_scope"],
            "continuous_cost_pricing_certified":
                cell["continuous_cost_pricing_certified"],
            "availability": cell.get("availability", "available"),
            "availability_reason": cell.get(
                "availability_reason", ""
            ),
        })
        summaries.append(row)
        for route_order, route in enumerate(cell["routes"]):
            blocks = route.get(
                "continuous_realized_charging_blocks"
            ) or route.get("recorded_charging_blocks") or []
            for block in blocks:
                block_rows.append({
                    **block,
                    "cell_id": cell["cell_id"],
                    "instance_id": cell["instance_id"],
                    "tariff_id": cell["tariff_id"],
                    "tier": cell["tier"],
                    "treatment": cell["treatment"],
                    "route_order": route_order,
                    "route_id": route["route_id"],
                })
        for certificate in cell.get("fixed_duty_certificates") or []:
            certificates.append({
                **certificate,
                "cell_id": cell["cell_id"],
                "instance_id": cell["instance_id"],
                "tariff_id": cell["tariff_id"],
            })
        for iteration in cell.get("cg_iterations") or []:
            cg_rows.append({
                **iteration,
                "cell_id": cell["cell_id"],
                "instance_id": cell["instance_id"],
                "scale": cell["scale"],
                "tariff_id": cell["tariff_id"],
                "treatment": cell["treatment"],
            })
        for checkpoint in cell.get("mip_checkpoints") or []:
            mip_rows.append({
                **checkpoint,
                "cell_id": cell["cell_id"],
                "instance_id": cell["instance_id"],
                "scale": cell["scale"],
                "tariff_id": cell["tariff_id"],
                "treatment": cell["treatment"],
            })
    summary_by = {
        (row["instance_id"], row["tariff_id"], row["tier"]): row
        for row in summaries
    }
    for key, tiers in sorted(by_key.items()):
        tier1 = tiers.get("TIER1_FIXED_GIRO_OPTIMIZED_CHARGING")
        if tier1 is None:
            continue
        for tier_name in (
            "TIER2_RAW_ROUTE_CHARGING",
            "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING",
        ):
            tier2 = tiers.get(tier_name)
            if tier2 is None:
                continue
            response = route_response(tier1["routes"], tier2["routes"])
            response.update({
                "instance_id": key[0],
                "tariff_id": key[1],
                "treatment": tier2["treatment"],
                "deadhead_kwh_change": (
                    tier2["metrics"]["deadhead_kwh"]
                    - tier1["metrics"]["deadhead_kwh"]
                ),
                "deadhead_min_change": (
                    tier2["metrics"]["deadhead_min"]
                    - tier1["metrics"]["deadhead_min"]
                ),
            })
            route_rows.append(response)
            tier2["route_similarity"] = response[
                "trip_coassignment_jaccard"
            ]
        tier0 = tiers.get("TIER0_GIRO_ORIGINAL")
        tier2 = tiers.get(
            "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING"
        )
        if tier0 is None or tier2 is None:
            continue
        for suffix, field in (
            ("grid", "grid_model_objective"),
            ("continuous", "continuous_replay_objective"),
        ):
            decomposition = savings_decomposition(
                tier0["metrics"].get(field),
                tier1["metrics"].get(field),
                tier2["metrics"].get(field),
            )
            for name in (
                "charging_only_savings", "rerouting_increment",
                "total_price_aware_savings",
            ):
                summary_by[(
                    tier2["instance_id"], tier2["tariff_id"],
                    tier2["tier"],
                )][f"{name}_{suffix}"] = decomposition[name]

    staging = Path(tempfile.mkdtemp(
        dir=output_dir.parent, prefix=f".{output_dir.name}.tmp."
    ))
    try:
        shutil.copyfile(
            tariff_manifest, staging / "tariff_manifest.csv"
        )
        duty_manifest = _source_path(manifest["giro40_duty_manifest"])
        if sha256_file(duty_manifest) != manifest[
            "giro40_duty_manifest_sha256"
        ]:
            raise ValueError("GIRO40 duty manifest hash mismatch")
        shutil.copyfile(
            duty_manifest, staging / "giro40_duty_manifest.csv"
        )
        _write_csv(staging / "charging_blocks_long.csv",
                   BLOCK_FIELDS, block_rows)
        _write_csv(staging / "tariff_response_summary.csv",
                   SUMMARY_FIELDS, summaries)
        _write_csv(staging / "route_change_summary.csv",
                   ROUTE_FIELDS, route_rows)
        _write_csv(staging / "fixed_duty_certificate_summary.csv",
                   CERT_FIELDS, certificates)
        _write_csv(staging / "cg_iteration_long.csv",
                   CG_FIELDS, cg_rows)
        _write_csv(staging / "mip_checkpoint_long.csv",
                   MIP_FIELDS, mip_rows)
        _figures(staging, cells, tariffs)
        dictionary = [
            {"field": "grid_model_objective",
             "definition": "Expanded-grid objective; certificate scope is tier-specific."},
            {"field": "continuous_replay_objective",
             "definition": "Cost of validated continuous blocks; never pricing-certified."},
            {"field": "charging_only_savings",
             "definition": "Tier 0 cost minus Tier 1 fixed-route optimized cost."},
            {"field": "rerouting_increment",
             "definition": "Tier 1 cost minus Tier 2 augmented route-flexible cost."},
            {"field": "route response",
             "definition": "Bus-label-invariant partition change; not called elasticity."},
            {"field": "terminal_soc_policy",
             "definition": PHYSICS["terminal_soc_policy"]},
        ]
        _write_csv(
            staging / "data_dictionary.csv",
            ("field", "definition"),
            dictionary,
        )
        inventory = [{
            "role": "experiment_manifest",
            "cell_id": "",
            "path": str(manifest_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }]
        for cell in cells:
            for artifact in cell.get("source_artifacts") or []:
                path = _source_path(artifact["path"])
                if (
                    not path.is_file()
                    or sha256_file(path) != artifact["sha256"]
                ):
                    raise ValueError("source artifact hash mismatch")
                inventory.append({
                    "role": artifact["role"],
                    "cell_id": cell["cell_id"],
                    "path": str(path),
                    "sha256": artifact["sha256"],
                    "size_bytes": path.stat().st_size,
                })
        _write_csv(
            staging / "artifact_inventory.csv",
            INVENTORY_FIELDS,
            sorted(inventory, key=lambda row: (
                row["cell_id"], row["role"], row["path"]
            )),
        )
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "provenance.json"
        }
        provenance = {
            "schema": "evsp-dr-tariff-response-provenance-v1",
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "python": platform.python_version(),
            "experiment_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "tariff_manifest_sha256": sha256_file(tariff_manifest),
            "physics": PHYSICS,
            "continuous_cost_pricing_certified": False,
            "output_sha256": output_hashes,
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        _rename_noreplace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "output_dir": str(output_dir),
        "cells": len(cells),
        "tariffs": len(tariffs),
        "route_comparisons": len(route_rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(
        args.experiment_manifest, args.out_dir
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

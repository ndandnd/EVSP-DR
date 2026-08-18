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
from config import BUS_COST_KX, CHARGE_START_COST
from utils_v2 import base_station_name


SCHEMA = "evsp-dr-tariff-response-experiment-v1"
TIERS = {
    "TIER0_GIRO_ORIGINAL",
    "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
    "TIER2_RAW_ROUTE_CHARGING",
    "TIER2_GIRO_AUGMENTED_ROUTE_CHARGING",
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
    "terminal_soc_min_kwh", "terminal_soc_max_kwh",
    "terminal_surplus_total_kwh",
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
    "fleet_change_from_tier1",
    "analysis_role", "primary_response_eligible",
    "terminal_energy_treatment",
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


def _schedule_fingerprint(routes):
    def physical_value(route, key):
        return route.get(key, (route.get("physical_realization") or {}).get(key))

    return hashlib.sha256(_canonical(sorted([
        {
            "trips": route.get("trips"),
            "route_nodes": route.get("route_nodes", route.get("route")),
            "charging_stops": route.get("charging_stops"),
            "expanded_grid_cost": route.get("expanded_grid_cost"),
            "continuous_realized_cost":
                route.get("continuous_realized_cost"),
            "continuous_realized_charging_blocks":
                route.get("continuous_realized_charging_blocks"),
            "continuous_terminal_soc_kwh":
                physical_value(route, "continuous_terminal_soc_kwh"),
            "expanded_grid_terminal_soc_kwh":
                physical_value(route, "expanded_grid_terminal_soc_kwh"),
            "waiting_min": route.get("waiting_min"),
            "deadhead_min": route.get("deadhead_min"),
            "deadhead_kwh": route.get("deadhead_kwh"),
        }
        for route in routes
    ], key=lambda item: (
        item["trips"], item["expanded_grid_cost"] or 0
    )))).hexdigest()


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


def _validate_metrics(cell, tariff):
    if cell["tier"] == "TIER0_GIRO_ORIGINAL":
        return
    prices = tariff_prices(tariff)
    metrics = cell["metrics"]
    grid_objective = 0.0
    continuous_objective = 0.0
    total_kwh = 0.0
    stops = 0
    waiting = 0.0
    deadhead_min = 0.0
    deadhead_kwh = 0.0
    by_hour = defaultdict(float)
    by_station = defaultdict(float)
    starts_by_hour = defaultdict(int)
    peak_start = (
        int(tariff["peak_window_start_hour"]) * 60
        if tariff["peak_window_start_hour"] != "" else None
    )
    peak_end = (
        int(tariff["peak_window_end_hour"]) * 60
        if tariff["peak_window_end_hour"] != "" else None
    )
    peak_kwh = 0.0 if peak_start is not None else None
    terminal_soc = []
    for route in cell["routes"]:
        blocks = route["continuous_realized_charging_blocks"]
        route_stops = route.get("charging_stops") or {}
        starts = list(route_stops.get("cst") or [])
        stations = list(route_stops.get("stations") or [])
        if len(starts) != len(stations):
            raise ValueError("charging start/station arrays differ")
        stops += len(stations)
        for start in starts:
            starts_by_hour[str(int(float(start) // 60))] += 1
        expanded_energy_cost = 0.0
        continuous_energy_cost = 0.0
        for block in blocks:
            station = base_station_name(block["station"])
            start = float(block["start_min"])
            end = float(block["end_min"])
            hour = int(start // 60)
            realized = float(block["realized_kwh"])
            expanded = float(block["expanded_grid_kwh"])
            capacity = (end - start) * PHYSICS["charge_kw"] / 60.0
            expected_price = prices[station].get(hour)
            if (
                expected_price is None
                or int(block["tariff_hour"]) != hour
                or not math.isclose(
                    float(block["price_per_kwh"]), expected_price,
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                or end > (hour + 1) * 60 + 1e-9
                or min(realized, expanded) < -1e-9
                or max(realized, expanded) > capacity + 1e-6
            ):
                raise ValueError("charging block tariff/power mismatch")
            expanded_energy_cost += expanded * expected_price
            continuous_energy_cost += realized * expected_price
            total_kwh += realized
            by_hour[str(hour)] += realized
            by_station[block["station"]] += realized
            if (
                peak_kwh is not None
                and start >= peak_start and end <= peak_end
            ):
                peak_kwh += realized
        route_grid = BUS_COST_KX + CHARGE_START_COST * len(stations) \
            + expanded_energy_cost
        route_continuous = BUS_COST_KX + CHARGE_START_COST * len(stations) \
            + continuous_energy_cost
        if (
            not math.isclose(
                float(route.get("expanded_grid_cost", math.nan)),
                route_grid, rel_tol=1e-10, abs_tol=1e-6,
            )
            or not math.isclose(
                float(route.get("continuous_realized_cost", math.nan)),
                route_continuous, rel_tol=1e-10, abs_tol=1e-6,
            )
        ):
            raise ValueError("route objective does not match charging blocks")
        grid_objective += route_grid
        continuous_objective += route_continuous
        waiting += _finite(route.get("waiting_min"), "route waiting_min")
        deadhead_min += _finite(
            route.get("deadhead_min"), "route deadhead_min"
        )
        deadhead_kwh += _finite(
            route.get("deadhead_kwh"), "route deadhead_kwh"
        )
        terminal_soc.append(_finite(
            route.get("continuous_terminal_soc_kwh"),
            "continuous_terminal_soc_kwh",
        ))
    expected_values = {
        "grid_model_objective": grid_objective,
        "continuous_replay_objective": continuous_objective,
        "charging_cost": grid_objective - BUS_COST_KX * len(cell["routes"]),
        "continuous_charging_cost":
            continuous_objective - BUS_COST_KX * len(cell["routes"]),
        "total_charged_kwh": total_kwh,
        "waiting_min": waiting,
        "deadhead_min": deadhead_min,
        "deadhead_kwh": deadhead_kwh,
        "charging_stops": stops,
        "terminal_soc_min_kwh": min(terminal_soc),
        "terminal_soc_max_kwh": max(terminal_soc),
        "terminal_surplus_total_kwh": sum(
            max(0.0, value - PHYSICS["reserve_kwh"])
            for value in terminal_soc
        ),
    }
    if peak_kwh is not None:
        expected_values["peak_window_kwh"] = peak_kwh
    for field, expected in expected_values.items():
        if not math.isclose(
            _finite(metrics.get(field), field), float(expected),
            rel_tol=1e-10, abs_tol=1e-6,
        ):
            raise ValueError(f"summary metric mismatch: {field}")
    mappings = {
        "charging_kwh_by_hour_json": dict(by_hour),
        "charging_kwh_by_station_json": dict(by_station),
        "charging_starts_by_hour_json": dict(starts_by_hour),
    }
    for field, expected in mappings.items():
        observed = json.loads(metrics[field])
        if set(observed) != set(expected) or any(
            not math.isclose(
                float(observed[key]), float(expected[key]),
                rel_tol=1e-10, abs_tol=1e-6,
            )
            for key in expected
        ):
            raise ValueError(f"summary mapping mismatch: {field}")


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
        and cell["treatment"] not in {
            "RAW", "GIRO-AUGMENTED", "GIRO40-AUGMENTED",
        }
    ):
        raise ValueError("route-flexible result has a fixed-duty label")
    if (
        cell["tier"] == "TIER2_RAW_ROUTE_CHARGING"
        and cell["treatment"] != "RAW"
    ):
        raise ValueError("RAW Tier 2 treatment mislabeled")
    if (
        cell["tier"] == "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING"
        and cell["treatment"] != "GIRO40-AUGMENTED"
    ):
        raise ValueError("augmented Tier 2 treatment mislabeled")
    if (
        cell["tier"] == "TIER2_GIRO_AUGMENTED_ROUTE_CHARGING"
        and cell["treatment"] != "GIRO-AUGMENTED"
    ):
        raise ValueError("augmented Tier 2 treatment mislabeled")
    _validate_routes(cell, tariff)
    if cell["tier"] == "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING":
        certificates = cell.get("fixed_duty_certificates")
        if not isinstance(certificates, list) or len(certificates) != len(
            cell["routes"]
        ):
            raise ValueError("Tier 1 certificate set is incomplete")
        certificate_by_key = {
            certificate.get("duty_id", certificate.get("route_id")):
                certificate
            for certificate in certificates
        }
        for route in cell["routes"]:
            key = route.get("duty_id", route.get("route_id"))
            certificate = certificate_by_key.get(key)
            if not isinstance(certificate, dict):
                raise ValueError("Tier 1 route certificate is missing")
            payload = {
                name: value for name, value in certificate.items()
                if name not in {
                    "certificate_sha256", "duty_id", "route_id"
                }
            }
            expected = hashlib.sha256(_canonical(payload)).hexdigest()
            if (
                certificate.get("certified") is not True
                or certificate.get(
                    "continuous_cost_optimality_certified"
                ) is not False
                or certificate.get("certificate_sha256") != expected
                or route.get("fixed_duty_certificate_sha256") != expected
            ):
                raise ValueError("Tier 1 certificate is invalid")
    metrics = cell["metrics"]
    numeric_fields = [
        "buses", "total_charged_kwh", "charging_stops",
        "runtime_preprocessing_s", "runtime_master_s",
        "runtime_pricing_s", "runtime_postprocessing_s",
    ]
    if cell["tier"] != "TIER0_GIRO_ORIGINAL":
        numeric_fields.extend([
            "waiting_min", "deadhead_min", "deadhead_kwh",
            "terminal_soc_min_kwh", "terminal_soc_max_kwh",
            "terminal_surplus_total_kwh",
        ])
    for field in numeric_fields:
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
    _validate_metrics(cell, tariff)
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


def _slug(value):
    return "".join(
        character.lower() if character.isalnum() else "-"
        for character in str(value)
    ).strip("-")


def _render_gantt_group(
    staging, key, tiers, tariffs, *, synthetic, stem
):
    import matplotlib.pyplot as plt

    instance_id, tariff_id = key
    augmented_tier = (
        "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING"
        if "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING" in tiers
        else "TIER2_GIRO_AUGMENTED_ROUTE_CHARGING"
    )
    panel_tiers = (
        "TIER0_GIRO_ORIGINAL",
        "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
        augmented_tier,
    )
    baseline = tiers[panel_tiers[0]]["routes"]
    tariff = tariffs[tariff_id]
    curves = tariff_prices(tariff)
    common_hours = sorted(set.intersection(*(
        set(curve) for curve in curves.values()
    )))
    average = {
        hour: sum(curve[hour] for curve in curves.values()) / len(curves)
        for hour in common_hours
    }
    all_prices = [
        price for curve in curves.values() for price in curve.values()
    ]
    low, high = min(all_prices), max(all_prices)
    rows = []
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    for panel, tier in enumerate(panel_tiers):
        cell = tiers[tier]
        order = _stable_orders(baseline, cell["routes"])
        axis = axes[panel]
        for route in cell["routes"]:
            y = order[route["route_id"]]
            for trip in route["trip_blocks"]:
                start, end = float(trip["start_min"]), float(trip["end_min"])
                axis.broken_barh(
                    [(start, end - start)], (y - 0.35, 0.7),
                    facecolors="#b7b7b7", edgecolors="#555555",
                    linewidth=0.25,
                )
                rows.append({
                    "instance_id": instance_id, "tariff_id": tariff_id,
                    "tier": tier, "route_order": y,
                    "event_kind": "trip", "trip_id": trip["trip_id"],
                    "station": "", "start_min": start, "end_min": end,
                    "kwh": "", "price_per_kwh": "",
                })
            for block in (
                route.get("continuous_realized_charging_blocks")
                or route.get("recorded_charging_blocks") or []
            ):
                station = base_station(block["station"])
                start = float(block["start_min"])
                end = float(block["end_min"])
                price = block.get("price_per_kwh")
                intensity = (
                    0.35 if price is None or high == low
                    else min(1.0, max(
                        0.0,
                        0.25 + 0.75 * (float(price) - low) / (high - low),
                    ))
                )
                axis.broken_barh(
                    [(start, end - start)], (y - 0.45, 0.9),
                    facecolors=STATION_COLORS[station],
                    alpha=intensity, edgecolors="black", linewidth=0.35,
                )
                rows.append({
                    "instance_id": instance_id, "tariff_id": tariff_id,
                    "tier": tier, "route_order": y,
                    "event_kind": "charge", "trip_id": "",
                    "station": station, "start_min": start, "end_min": end,
                    "kwh": block.get(
                        "realized_kwh", block.get("kwh")
                    ),
                    "price_per_kwh": price,
                })
        axis.set_ylabel(tier.split("_", 1)[0])
        price_axis = axis.twinx()
        price_axis.step(
            [hour * 60 for hour in common_hours],
            [average[hour] for hour in common_hours],
            where="post", color="#222222", linewidth=0.8, alpha=0.55,
        )
    axes[-1].set_xlabel("minute of service day")
    fig.suptitle(
        (
            f"NEGATIVE-PRICE STRESS — EXCLUDED FROM PRIMARY: "
            f"{instance_id}: {tariff_id}"
        )
        if tariff["analysis_role"] == "negative_price_stress"
        else f"{instance_id}: {tariff_id}"
    )
    if synthetic:
        fig.text(
            0.5, 0.5, "SYNTHETIC — NOT EVIDENCE",
            ha="center", va="center", fontsize=28,
            color="red", alpha=0.16, rotation=25,
        )
    fig.tight_layout()
    fig.savefig(
        staging / f"{stem}.png", dpi=180,
        metadata={"Software": "EVSP-DR"},
    )
    fig.savefig(
        staging / f"{stem}.pdf",
        metadata={
            "Creator": "EVSP-DR", "CreationDate": None, "ModDate": None,
        },
    )
    plt.close(fig)
    return rows


def _render_response_instance(staging, instance_id, rows, *, synthetic):
    import matplotlib.pyplot as plt

    metrics = (
        ("peak_window_kwh", "Peak-window kWh"),
        ("charging_cost", "Charging cost"),
        ("route_similarity", "Trip co-assignment similarity"),
        ("buses", "Buses"),
        ("deadhead_kwh", "Deadhead kWh"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["tier"], row["treatment"])].append(row)
    for axis, (field, label) in zip(axes.flat, metrics):
        for key, values in sorted(groups.items()):
            values.sort(key=lambda row: row["alpha"])
            if any(row[field] is None for row in values):
                continue
            axis.plot(
                [row["alpha"] for row in values],
                [float(row[field]) for row in values],
                marker="o", label=" / ".join(key),
            )
        axis.set(xlabel="peak amplitude α", ylabel=label)
    axes.flat[-1].axis("off")
    fig.suptitle(str(instance_id))
    if synthetic:
        fig.text(
            0.5, 0.5, "SYNTHETIC — NOT EVIDENCE",
            ha="center", va="center", fontsize=28,
            color="red", alpha=0.16, rotation=25,
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower right", fontsize=7)
    fig.tight_layout()
    stem = f"price_amplitude_response_{_slug(instance_id)}"
    fig.savefig(
        staging / f"{stem}.png", dpi=180,
        metadata={"Software": "EVSP-DR"},
    )
    fig.savefig(
        staging / f"{stem}.pdf",
        metadata={
            "Creator": "EVSP-DR", "CreationDate": None, "ModDate": None,
        },
    )
    plt.close(fig)


def _figures(staging, cells, tariffs, *, synthetic=False):
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
        } <= set(tiers)
        and bool({
            "TIER2_GIRO_AUGMENTED_ROUTE_CHARGING",
            "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING",
        } & set(tiers))
        and all(
            any(route.get("trip_blocks") for route in cell["routes"])
            for cell in tiers.values()
        )
    ]
    if not complete:
        raise ValueError("no complete cell with Gantt trip blocks")
    (instance_id, tariff_id), tiers = complete[0]
    augmented_tier = (
        "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING"
        if "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING" in tiers
        else "TIER2_GIRO_AUGMENTED_ROUTE_CHARGING"
    )
    panel_tiers = (
        "TIER0_GIRO_ORIGINAL",
        "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
        augmented_tier,
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
    all_prices = [
        price for curve in curves.values() for price in curve.values()
    ]
    low, high = min(all_prices), max(all_prices)
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
                    else min(1.0, max(
                        0.0,
                        0.25 + 0.75 * (float(price) - low) / (high - low),
                    ))
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
    fig.suptitle(
        (
            f"NEGATIVE-PRICE STRESS — EXCLUDED FROM PRIMARY: "
            f"{instance_id}: {tariff_id}"
        )
        if tariff["analysis_role"] == "negative_price_stress"
        else f"{instance_id}: {tariff_id}"
    )
    if synthetic:
        fig.text(
            0.5, 0.5, "SYNTHETIC — NOT EVIDENCE",
            ha="center", va="center", fontsize=28,
            color="red", alpha=0.16, rotation=25,
        )
    fig.tight_layout()
    png = staging / "gantt_three_tiers.png"
    pdf = staging / "gantt_three_tiers.pdf"
    fig.savefig(png, dpi=180, metadata={"Software": "EVSP-DR"})
    fig.savefig(pdf, metadata={
        "Creator": "EVSP-DR", "CreationDate": None, "ModDate": None,
    })
    plt.close(fig)
    for extra_key, extra_tiers in complete:
        extra_rows = _render_gantt_group(
            staging,
            extra_key,
            extra_tiers,
            tariffs,
            synthetic=synthetic,
            stem=(
                f"gantt_three_tiers_{_slug(extra_key[0])}_"
                f"{_slug(extra_key[1])}"
            ),
        )
        if extra_key != (instance_id, tariff_id):
            gantt_rows.extend(extra_rows)
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
            "analysis_role": tariff["analysis_role"],
            "peak_window_kwh": metrics.get("peak_window_kwh"),
            "charging_cost": metrics.get("charging_cost"),
            "route_similarity": cell.get("route_similarity"),
            "buses": metrics.get("buses"),
            "deadhead_kwh": metrics.get("deadhead_kwh"),
            "terminal_surplus_total_kwh":
                metrics.get("terminal_surplus_total_kwh"),
        })
    primary_rows = [
        row for row in alpha_rows if row["analysis_role"] == "primary"
    ]
    stress_rows = [
        row for row in alpha_rows
        if row["analysis_role"] == "negative_price_stress"
    ]
    required_alpha = {0.0, 0.25, 0.5, 1.0}
    alpha_groups = defaultdict(set)
    for row in primary_rows:
        alpha_groups[(
            row["instance_id"], row["tier"], row["treatment"]
        )].add(row["alpha"])
    if not any(
        required_alpha <= values for values in alpha_groups.values()
    ):
        raise ValueError("alpha response cells are incomplete")
    response_fields = (
        "instance_id", "tariff_id", "tier", "treatment", "alpha",
        "peak_window_kwh", "charging_cost", "route_similarity",
        "buses", "deadhead_kwh", "terminal_surplus_total_kwh",
        "analysis_role",
    )
    _write_csv(
        staging / "tariff_response_plot.csv",
        response_fields,
        sorted(primary_rows, key=lambda row: (
            row["tier"], row["treatment"], row["alpha"]
        )),
    )
    _write_csv(
        staging / "negative_price_stress_plot.csv",
        response_fields,
        sorted(stress_rows, key=lambda row: (
            row["instance_id"], row["tier"], row["treatment"]
        )),
    )
    elasticity_rows = []
    elasticity_groups = defaultdict(list)
    for row in primary_rows:
        elasticity_groups[(
            row["instance_id"], row["tier"], row["treatment"]
        )].append(row)
    for key, rows in sorted(elasticity_groups.items()):
        positive = sorted(
            (row for row in rows if row["alpha"] > 0.0),
            key=lambda row: row["alpha"],
        )
        for left, right in zip(positive, positive[1:]):
            for field in ("peak_window_kwh", "charging_cost"):
                left_value = left[field]
                right_value = right[field]
                elasticity = None
                if (
                    left_value is not None
                    and right_value is not None
                    and float(left_value) != 0.0
                ):
                    elasticity = (
                        (float(right_value) - float(left_value))
                        / float(left_value)
                    ) / (
                        (right["alpha"] - left["alpha"])
                        / left["alpha"]
                    )
                elasticity_rows.append({
                    "instance_id": key[0],
                    "tier": key[1],
                    "treatment": key[2],
                    "metric": field,
                    "alpha_left": left["alpha"],
                    "alpha_right": right["alpha"],
                    "arc_elasticity": elasticity,
                    "analysis_role": "primary",
                })
    _write_csv(
        staging / "price_response_elasticity.csv",
        (
            "instance_id", "tier", "treatment", "metric",
            "alpha_left", "alpha_right", "arc_elasticity",
            "analysis_role",
        ),
        elasticity_rows,
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
    for row in primary_rows:
        grouped_rows[(
            row["instance_id"], row["tier"], row["treatment"]
        )].append(row)
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
    if synthetic:
        fig.text(
            0.5, 0.5, "SYNTHETIC — NOT EVIDENCE",
            ha="center", va="center", fontsize=28,
            color="red", alpha=0.16, rotation=25,
        )
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
    stress_fig, stress_axes = plt.subplots(1, 2, figsize=(12, 4.8))
    stress_groups = defaultdict(list)
    for row in stress_rows:
        stress_groups[row["instance_id"]].append(row)
    for axis, field, label in (
        (stress_axes[0], "charging_cost", "Charging cost"),
        (
            stress_axes[1], "terminal_surplus_total_kwh",
            "Terminal surplus kWh",
        ),
    ):
        offset = 0
        for instance, rows in sorted(stress_groups.items()):
            labels_ = [
                f"{row['tier']} / {row['treatment']}" for row in rows
            ]
            positions = list(range(offset, offset + len(rows)))
            axis.bar(
                positions,
                [float(row[field]) for row in rows],
                label=instance,
            )
            axis.set_xticks(positions, labels_, rotation=75, fontsize=6)
            offset += len(rows) + 1
        axis.set_ylabel(label)
    stress_fig.suptitle(
        "NEGATIVE-PRICE STRESS (α=2) — EXCLUDED FROM PRIMARY RESULTS",
        color="darkred", fontweight="bold",
    )
    if synthetic:
        stress_fig.text(
            0.5, 0.5, "SYNTHETIC — NOT EVIDENCE",
            ha="center", va="center", fontsize=24,
            color="red", alpha=0.16, rotation=25,
        )
    stress_fig.tight_layout()
    stress_fig.savefig(
        staging / "negative_price_stress.png",
        dpi=180, metadata={"Software": "EVSP-DR"},
    )
    stress_fig.savefig(
        staging / "negative_price_stress.pdf",
        metadata={
            "Creator": "EVSP-DR", "CreationDate": None, "ModDate": None,
        },
    )
    plt.close(stress_fig)
    for response_instance in sorted({
        row["instance_id"] for row in primary_rows
    }):
        _render_response_instance(
            staging,
            response_instance,
            [
                row for row in primary_rows
                if row["instance_id"] == response_instance
            ],
            synthetic=synthetic,
        )


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
        tariff = tariffs[cell["tariff_id"]]
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
            "analysis_role": tariff["analysis_role"],
            "primary_response_eligible":
                tariff["primary_response_eligible"],
            "terminal_energy_treatment":
                tariff["terminal_energy_treatment"],
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
            "TIER2_GIRO_AUGMENTED_ROUTE_CHARGING",
            "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING",
        ):
            tier2 = tiers.get(tier_name)
            if tier2 is None:
                continue
            if tier1["terminal_soc_policy"] != tier2[
                "terminal_soc_policy"
            ]:
                raise ValueError(
                    "Tier 1/Tier 2 terminal policies differ"
                )
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
        tier2 = (
            tiers.get("TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING")
            or tiers.get("TIER2_GIRO_AUGMENTED_ROUTE_CHARGING")
        )
        if tier0 is None or tier2 is None:
            continue
        for suffix, field in (
            ("grid", "grid_model_objective"),
            ("continuous", "continuous_replay_objective"),
        ):
            same_fleet_and_terminal = (
                tier0["metrics"]["buses"]
                == tier1["metrics"]["buses"]
                == tier2["metrics"]["buses"]
                and tier0["terminal_soc_policy"]
                == tier1["terminal_soc_policy"]
                == tier2["terminal_soc_policy"]
            )
            primary_eligible = str(
                tariffs[key[1]]["primary_response_eligible"]
            ).lower() == "true"
            primary_comparison = (
                same_fleet_and_terminal and primary_eligible
            )
            decomposition = savings_decomposition(
                tier0["metrics"].get(field)
                if primary_comparison else None,
                tier1["metrics"].get(field)
                if primary_comparison else None,
                tier2["metrics"].get(field)
                if primary_comparison else None,
            )
            if not primary_comparison:
                target_row = summary_by[(
                    tier2["instance_id"], tier2["tariff_id"],
                    tier2["tier"],
                )]
                target_row["availability"] = "partial"
                reasons = {
                    value for value in str(
                        target_row.get("availability_reason") or ""
                    ).split(";") if value
                }
                reasons.add(
                    "negative_price_stress_excluded_from_primary"
                    if not primary_eligible
                    else
                    "decomposition_unavailable_fleet_or_terminal_policy_changed"
                )
                target_row["availability_reason"] = ";".join(
                    sorted(reasons)
                )
            for name in (
                "charging_only_savings", "rerouting_increment",
                "total_price_aware_savings",
            ):
                summary_by[(
                    tier2["instance_id"], tier2["tariff_id"],
                    tier2["tier"],
                )][f"{name}_{suffix}"] = decomposition[name]
        summary_by[(
            tier2["instance_id"], tier2["tariff_id"], tier2["tier"],
        )]["fleet_change_from_tier1"] = (
            tier2["metrics"]["buses"] - tier1["metrics"]["buses"]
        )

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
        _write_csv(
            staging / "primary_savings_summary.csv",
            SUMMARY_FIELDS,
            [
                row for row in summaries
                if str(row.get("primary_response_eligible")).lower()
                == "true"
            ],
        )
        _write_csv(
            staging / "negative_price_stress_summary.csv",
            SUMMARY_FIELDS,
            [
                row for row in summaries
                if row.get("analysis_role") == "negative_price_stress"
            ],
        )
        _write_csv(staging / "route_change_summary.csv",
                   ROUTE_FIELDS, route_rows)
        _write_csv(staging / "fixed_duty_certificate_summary.csv",
                   CERT_FIELDS, certificates)
        _write_csv(staging / "cg_iteration_long.csv",
                   CG_FIELDS, cg_rows)
        _write_csv(staging / "mip_checkpoint_long.csv",
                   MIP_FIELDS, mip_rows)
        synthetic = manifest.get("synthetic") is True
        _figures(staging, cells, tariffs, synthetic=synthetic)
        if synthetic:
            (staging / "SYNTHETIC_ONLY.txt").write_text(
                "SYNTHETIC FIXTURE — NOT EXPERIMENTAL EVIDENCE.\n"
            )
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
            {"field": "terminal_soc_min_kwh / terminal_soc_max_kwh",
             "definition": "Selected-route continuous depot-arrival SOC range; exposes negative-price energy accumulation."},
            {"field": "negative_price_policy",
             "definition": "Manifest policy allows feasible consumption but no energy export."},
            {"field": "analysis_role",
             "definition": "primary or negative_price_stress; stress rows are excluded from primary savings and elasticity."},
            {"field": "terminal_surplus_total_kwh",
             "definition": "Sum over selected routes of max(terminal SOC - reserve, 0)."},
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
            schedule_artifact_bound = False
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
                if artifact["role"] in {
                    "tier1_seed", "normalized_schedule",
                }:
                    source_payload = json.loads(path.read_text())
                    source_routes = source_payload.get("routes")
                    if (
                        not isinstance(source_routes, list)
                        or _schedule_fingerprint(source_routes)
                        != _schedule_fingerprint(cell["routes"])
                    ):
                        raise ValueError(
                            "cell routes differ from validated source artifact"
                        )
                    schedule_artifact_bound = True
            if (
                not synthetic
                and cell["tier"] != "TIER0_GIRO_ORIGINAL"
                and not schedule_artifact_bound
            ):
                raise ValueError(
                    "optimized cell lacks a schedule-bound source artifact"
                )
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
            "campaign_provenance": manifest.get("campaign_provenance"),
            "continuous_cost_pricing_certified": False,
            "synthetic": synthetic,
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

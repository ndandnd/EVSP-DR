#!/usr/bin/env python3
"""Build duty-union instance features and descriptive RAW recovery summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from audit_giro_known_columns import DEPOT, HORIZON_MIN, STATIONS, build_problem
from audit_scale_ladder_known_membership import _prices
from build_tariff_response_manifest import REPO_ROOT, sha256_file
from fixed_duty_expanded_optimizer import optimize_fixed_duty
from rerealize_routes import _arc_map
from tariff_response_core import giro_routes_for_instance


MANIFEST = (
    REPO_ROOT / "data/scale_ladder/instances/"
    "scale_ladder_instance_manifest_6sel_seed20260803.csv"
)
PREFLIGHT = (
    REPO_ROOT
    / "data/scale_ladder/known_membership_preflight_6sel_seed20260803.json"
)
MASTER = REPO_ROOT / "data/Par_VehicleDetails_Updated.csv"
GRIDS = (
    ("soc15_b10", 15.0, 10),
    ("soc5_b10", 5.0, 10),
    ("soc2p5_b10", 2.5, 10),
    ("soc1_b10", 1.0, 10),
    ("soc1_b5", 1.0, 5),
)
FEATURES = (
    "trip_count", "duty_count", "deadhead_density",
    "deadhead_energy_fraction", "service_kwh_per_trip",
    "service_kwh_per_duty", "layover_slack_min",
    "layover_slack_median", "layover_slack_max",
    "station_reachability_fraction",
    *(f"representable_fraction_{key}" for key, _soc, _block in GRIDS),
)
CELL_ID = re.compile(r"^k(?P<scale>\d+)_s(?P<selection>\d+)(?:_c\d+)?$")


def _bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _percentile(values, which):
    return float(np.percentile(values, which)) if values else math.nan


def _instance_metrics(problem, routes):
    trips = tuple(problem.trips)
    trip_set = set(trips)
    arcs = _arc_map(problem)
    temporal_pairs = sum(
        problem.end_min[left] <= problem.start_min[right] + 1e-9
        for left in trips for right in trips if left != right
    )
    direct_pairs = {
        (left, right)
        for left, entries in problem.adjacency.items()
        if left in trip_set
        for right, _minutes, _kwh, kind in entries
        if kind == "trip_trip" and right in trip_set
    }
    station_out = {
        trip for trip in trips
        if any(
            kind == "trip_station" and target in STATIONS
            for target, _minutes, _kwh, kind
            in problem.adjacency.get(trip, ())
        )
    }
    station_in = {
        target
        for station in STATIONS
        for target, _minutes, _kwh, kind
        in problem.adjacency.get(station, ())
        if kind == "station_trip" and target in trip_set
    }

    slack = []
    missing_direct = 0
    deadhead_kwh = 0.0
    deadhead_legs = 0
    duty_service = []
    for route in routes:
        sequence = list(route["trips"])
        duty_service.append(sum(problem.trip_energy[trip] for trip in sequence))
        legs = (
            [(DEPOT, sequence[0])]
            + list(zip(sequence, sequence[1:]))
            + [(sequence[-1], DEPOT)]
        )
        for left, right in legs:
            if (left, right) in arcs:
                deadhead_kwh += float(arcs[left, right][1])
                deadhead_legs += 1
            else:
                missing_direct += 1
        for left, right in zip(sequence, sequence[1:]):
            if (left, right) not in arcs:
                continue
            slack.append(
                float(problem.start_min[right])
                - float(problem.end_min[left])
                - float(arcs[left, right][0])
            )

    service_kwh = float(sum(problem.trip_energy.values()))
    return {
        "total_service_kwh": service_kwh,
        "service_kwh_per_trip": service_kwh / len(trips),
        "service_kwh_per_duty": service_kwh / len(routes),
        "duty_service_kwh_min": min(duty_service),
        "duty_service_kwh_median": float(np.median(duty_service)),
        "duty_service_kwh_max": max(duty_service),
        "deadhead_density": (
            len(direct_pairs) / temporal_pairs if temporal_pairs else 0.0
        ),
        "direct_deadhead_edge_count": len(direct_pairs),
        "temporal_trip_pair_count": temporal_pairs,
        "known_duty_direct_deadhead_kwh": deadhead_kwh,
        "deadhead_energy_fraction": (
            deadhead_kwh / (service_kwh + deadhead_kwh)
            if service_kwh + deadhead_kwh else 0.0
        ),
        "known_duty_direct_deadhead_leg_count": deadhead_legs,
        "known_duty_missing_direct_leg_count": missing_direct,
        "layover_slack_min": min(slack) if slack else math.nan,
        "layover_slack_median": (
            float(np.median(slack)) if slack else math.nan
        ),
        "layover_slack_max": max(slack) if slack else math.nan,
        "layover_slack_p10": _percentile(slack, 10),
        "layover_slack_p90": _percentile(slack, 90),
        "layover_direct_pair_count": len(slack),
        "station_reachability_fraction": len(station_out) / len(trips),
        "station_return_reachability_fraction": len(station_in) / len(trips),
    }


def build_instance_features(manifest=MANIFEST, preflight=PREFLIGHT):
    manifest = Path(manifest).resolve()
    preflight = Path(preflight).resolve()
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(
        "SyntheticRandom" in row["relative_path"]
        or row["generator_family"].startswith("generate_random_goal1")
        for row in rows
    ):
        raise ValueError("SyntheticRandom rows cannot enter duty-union features")
    membership = json.loads(preflight.read_text())
    by_cell = {cell["cell_id"]: cell for cell in membership["cells"]}
    prices = _prices()
    duty_grid = {}
    output = []
    for row in rows:
        scale = int(row["scale"])
        selection = int(row["selection_replicate"])
        cell_id = f"k{scale:02d}_s{selection}"
        instance = REPO_ROOT / row["relative_path"]
        problem = build_problem(
            instance.parent, instance.name,
            max_station_to_trip_wait_min=HORIZON_MIN,
            reference_data_dir=REPO_ROOT / "data",
        )
        routes = giro_routes_for_instance(MASTER, instance)
        if sorted(route["duty_id"] for route in routes) != sorted(
            json.loads(row["duties_json"])
        ):
            raise ValueError(f"duty identity mismatch: {cell_id}")
        for route in routes:
            duty = route["duty_id"]
            if duty in duty_grid:
                continue
            duty_grid[duty] = {}
            for key, soc_step, block_min in GRIDS:
                result = optimize_fixed_duty(
                    problem, route["trips"], prices,
                    g_kwh=300.0, charge_kw=300.0, reserve_kwh=0.0,
                    soc_step=soc_step, block_min=block_min,
                    tariff_id="historical_flat_instance_features",
                    instance_sha256=row["instance_file_sha256"],
                    allow_diagnostic_grid=True,
                )
                duty_grid[duty][key] = result.get("feasible") is True
        feature = {
            "cell_id": cell_id,
            "scale": scale,
            "selection_replicate": selection,
            "target_fleet": int(row["target_fleet"]),
            "instance_file_sha256": row["instance_file_sha256"],
            "duty_set_sha256": row["duty_set_sha256"],
            "generator_seed": int(row["generator_seed"]),
            "generator_family": row["generator_family"],
            "trip_count": int(row["trip_count"]),
            "duty_count": int(row["duty_count"]),
            "duties_json": row["duties_json"],
            **_instance_metrics(problem, routes),
        }
        duties = [route["duty_id"] for route in routes]
        for key, _soc_step, _block_min in GRIDS:
            feature[f"representable_fraction_{key}"] = (
                sum(duty_grid[duty][key] for duty in duties) / len(duties)
            )
        expected_primary = sum(
            duty["known_partition_in_primary_expanded_space"]
            for duty in by_cell[cell_id]["duties"]
        ) / len(by_cell[cell_id]["duties"])
        if not math.isclose(
            feature["representable_fraction_soc15_b10"],
            expected_primary, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(f"primary representability mismatch: {cell_id}")
        output.append(feature)
    return pd.DataFrame(output).sort_values(
        ["scale", "selection_replicate"]
    ).reset_index(drop=True)


def _normalize_results(paths, instances):
    if not paths:
        return pd.DataFrame(columns=[
            "cell_id", "instance_file_sha256", "target_fleet", "buses",
            "fleet_proven", "recovery_status",
        ])
    frames = [pd.read_csv(path) for path in paths]
    source = pd.concat(frames, ignore_index=True)
    if "arm" in source and any(
        source["arm"].fillna("RAW").astype(str) != "RAW"
    ):
        source = source[
            source["arm"].fillna("RAW").astype(str) == "RAW"
        ].copy()
    if "buses" not in source and "mip_incumbent_fleet" in source:
        source["buses"] = source["mip_incumbent_fleet"]
    required = {"cell_id", "buses", "fleet_proven"}
    if not required <= set(source):
        raise ValueError(
            "RAW results lack required columns: "
            + ",".join(sorted(required - set(source)))
        )
    instance_lookup = instances.set_index("cell_id")
    parsed = []
    for row in source.to_dict("records"):
        raw_cell = str(row["cell_id"])
        match = CELL_ID.match(raw_cell)
        base_cell = (
            f"k{int(match.group('scale')):02d}_s"
            f"{int(match.group('selection'))}"
            if match else raw_cell
        )
        if base_cell not in instance_lookup.index:
            raise ValueError(f"RAW result has unknown duty-union cell: {raw_cell}")
        instance = instance_lookup.loc[base_cell]
        buses = _finite(row.get("buses"))
        proven = _bool(row.get("fleet_proven"))
        supplied_target = _finite(row.get("target_fleet"))
        target = int(
            supplied_target
            if supplied_target is not None else instance["target_fleet"]
        )
        status = (
            "recovered" if proven and buses is not None and buses <= target
            else "missed" if proven and buses is not None
            else "unknown"
        )
        parsed.append({
            **row,
            "cell_id": raw_cell,
            "base_cell_id": base_cell,
            "instance_file_sha256": instance["instance_file_sha256"],
            "target_fleet": target,
            "buses": buses,
            "fleet_proven": proven,
            "recovery_status": status,
        })
    return pd.DataFrame(parsed)


def descriptive_associations(instances, cells):
    if cells.empty:
        return pd.DataFrame(columns=[
            "feature", "n_cells", "n_instances", "recovered_cells",
            "missed_cells", "recovered_mean", "missed_mean",
            "recovered_median", "missed_median",
            "spearman_rho_cell_weighted",
            "spearman_rho_instance_balanced",
        ])
    joined = cells.merge(
        instances, on="instance_file_sha256", how="left",
        validate="many_to_one", suffixes=("", "_instance"),
    )
    observable = joined[joined["recovery_status"] != "unknown"].copy()
    observable["recovered"] = (
        observable["recovery_status"] == "recovered"
    ).astype(float)
    rows = []
    for feature in FEATURES:
        if feature not in observable:
            rows.append({
                "feature": feature, "n_cells": 0, "n_instances": 0,
                "recovered_cells": 0, "missed_cells": 0,
                "recovered_mean": math.nan, "missed_mean": math.nan,
                "recovered_median": math.nan, "missed_median": math.nan,
                "spearman_rho_cell_weighted": math.nan,
                "spearman_rho_instance_balanced": math.nan,
            })
            continue
        valid = observable.dropna(subset=[feature])
        recovered = valid[valid["recovered"] == 1.0][feature]
        missed = valid[valid["recovered"] == 0.0][feature]
        rho = (
            float(spearmanr(valid[feature], valid["recovered"]).statistic)
            if len(valid) > 1
            and valid[feature].nunique() > 1
            and valid["recovered"].nunique() > 1 else math.nan
        )
        balanced = valid.groupby(
            "instance_file_sha256", as_index=False
        ).agg({feature: "first", "recovered": "mean"})
        rho_balanced = (
            float(spearmanr(
                balanced[feature], balanced["recovered"]
            ).statistic)
            if len(balanced) > 1
            and balanced[feature].nunique() > 1
            and balanced["recovered"].nunique() > 1 else math.nan
        )
        rows.append({
            "feature": feature,
            "n_cells": len(valid),
            "n_instances": valid["instance_file_sha256"].nunique(),
            "recovered_cells": len(recovered),
            "missed_cells": len(missed),
            "recovered_mean": recovered.mean(),
            "missed_mean": missed.mean(),
            "recovered_median": recovered.median(),
            "missed_median": missed.median(),
            "spearman_rho_cell_weighted": rho,
            "spearman_rho_instance_balanced": rho_balanced,
        })
    return pd.DataFrame(rows)


def feature_distribution(instances):
    rows = []
    for feature in FEATURES:
        values = pd.to_numeric(
            instances.get(feature, pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        rows.append({
            "feature": feature,
            "n_instances": len(values),
            "min": values.min(),
            "p25": values.quantile(0.25),
            "median": values.median(),
            "p75": values.quantile(0.75),
            "max": values.max(),
        })
    return pd.DataFrame(rows)


def _write_csv(frame, path):
    frame.to_csv(path, index=False, lineterminator="\n")


def publish(output_dir, *, manifest=MANIFEST, preflight=PREFLIGHT,
            raw_results=()):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    instances = build_instance_features(manifest, preflight)
    cells = _normalize_results(tuple(raw_results), instances)
    associations = descriptive_associations(instances, cells)
    distribution = feature_distribution(instances)
    _write_csv(instances, output / "instance_features.csv")
    _write_csv(cells, output / "raw_cell_outcomes.csv")
    _write_csv(associations, output / "feature_associations.csv")
    _write_csv(distribution, output / "feature_distribution.csv")
    status = (
        "complete_descriptive_noncausal"
        if not cells.empty else "not_estimable_no_auditable_raw_results"
    )
    summary = {
        "schema": "evsp-dr-raw-recovery-feature-analysis-v1",
        "analysis_status": status,
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": sha256_file(Path(manifest)),
        "preflight": str(Path(preflight).resolve()),
        "preflight_sha256": sha256_file(Path(preflight)),
        "raw_result_inputs": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}
            for path in raw_results
        ],
        "instance_rows": len(instances),
        "raw_result_rows": len(cells),
        "observable_raw_rows": int(
            (cells.get("recovery_status", pd.Series(dtype=str)) != "unknown").sum()
        ),
        "non_independence_warning": (
            "Instances overlap in duties and cells repeat instances across "
            "grids/physics/pools; associations are descriptive, not causal."
        ),
        "output_sha256": {
            name: sha256_file(output / name)
            for name in (
                "instance_features.csv", "feature_distribution.csv",
                "raw_cell_outcomes.csv", "feature_associations.csv",
            )
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    report = [
        "# RAW recovery and instance features",
        "",
        f"Analysis status: `{status}`.",
        "",
        "This audit is descriptive. Duty-union instances overlap and repeated "
        "cells share instances, so rows are not independent observations. "
        "No association is interpreted causally.",
        "",
        "## Feature definitions",
        "",
        "- `deadhead_density`: direct trip-to-trip deadhead edges divided by "
        "all temporally ordered trip pairs.",
        "- `deadhead_energy_fraction`: direct known-duty deadhead kWh divided "
        "by service plus direct deadhead kWh.",
        "- layover slack: calendar gap minus direct deadhead minutes for "
        "consecutive trips in the GIRO duty; missing direct legs are counted "
        "and excluded from the distribution.",
        "- `station_reachability_fraction`: fraction of trips with at least "
        "one outgoing trip-to-station model arc.",
        "- grid fractions: fraction of GIRO duties feasible in the frozen "
        "fixed-duty expanded optimizer at each named 300/300 grid.",
        "",
        "Distribution summaries across the 40 duty-union instances are in "
        "`feature_distribution.csv`.",
        "",
    ]
    if cells.empty:
        report += [
            "## Recovery association unavailable",
            "",
            "No auditable normalized RAW integer-result rows are tracked in "
            "this checkout. `records/RESULTS_LOG.csv` is header-only and no "
            "`mip_run_summary.csv` exists on any available git ref. Therefore "
            "no feature/recovery correlation or threshold claim is reported. "
            "Supply normalized RAW rows via `--raw-results`; prose and LP "
            "route weights are intentionally not recoded as integer recovery.",
            "",
        ]
    else:
        report += [
            "## Recovery association",
            "",
            "See `feature_associations.csv`. It reports cell-weighted and "
            "instance-balanced Spearman coefficients without p-values. These "
            "are exploratory descriptions, not independent-sample inference.",
            "",
        ]
    (output / "README.md").write_text("\n".join(report))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument(
        "--raw-results", type=Path, action="append", default=[],
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(publish(
        args.out, manifest=args.manifest, preflight=args.preflight,
        raw_results=args.raw_results,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

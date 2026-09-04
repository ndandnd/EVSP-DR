#!/usr/bin/env python3
"""Summarize an audited event-CG campaign by scale and row.

This consumes ``medium_event_summary.csv`` written by the fail-closed auditor.
It never upgrades a Slurm completion to an optimization certificate.  The two
output CSVs preserve both aggregate frontier statistics and row-level phase
telemetry for later research tables.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path


TRUE = {"1", "true", "yes"}


def truth(value: object) -> bool:
    return str(value).strip().lower() in TRUE


def number(value: object) -> float | None:
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def integer(value: object) -> int | None:
    parsed = number(value)
    return int(parsed) if parsed is not None else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def family_role(cell: str) -> tuple[str, str]:
    if "_p" in cell:
        return "probability", "fixed_seed_probability"
    roles = {
        "_xlight": "trip_light",
        "_xtrip": "trip_heavy",
        "_xenergy": "energy_heavy",
        "_xgap": "tight_gap",
    }
    for token, role in roles.items():
        if token in cell:
            return "structural", role
    return "legacy", "legacy_selection"


def classify(row: dict[str, str]) -> str:
    state = str(row.get("slurm_state") or "")
    # A task killed by Slurm can have no result-side configuration fields.
    # Classify the observed scheduler event before inspecting those fields so
    # operational censoring is not mislabeled as a scientific configuration
    # failure.
    if state == "PREEMPTED":
        return "preempted"
    if state in {"FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED"}:
        return "execution_failure"
    if not truth(row.get("configuration_match")):
        return "invalid_configuration"
    if truth(row.get("certified_rc_optimal")):
        return "certified"
    stop = str(row.get("stop_reason") or "")
    if stop == "wall_limit":
        return "wall_limit"
    if stop == "external_signal":
        return "external_signal"
    return stop or "missing_result"


def enrich(row: dict[str, str]) -> dict[str, object]:
    result: dict[str, object] = dict(row)
    cell = str(row.get("cell_id") or "")
    family, role = family_role(cell)
    result["sample_family"] = family
    result["selection_role"] = role
    result["outcome"] = classify(row)
    scale = integer(row.get("scale"))
    lower = number(row.get("L_model"))
    artificials = number(row.get("artificials"))
    result["ceil_L_model"] = (
        math.ceil(lower - 1e-7) if lower is not None else ""
    )
    result["fleet_target_proved"] = bool(
        result["outcome"] == "certified"
        and scale is not None
        and lower is not None
        and math.ceil(lower - 1e-7) == scale
        and (artificials is None or artificials <= 1e-7)
    )
    shortest = number(row.get("pricing_shortest_path_s")) or 0.0
    batch = number(row.get("pricing_batch_s")) or 0.0
    # In the pinned pricer, pricing_extra_columns is measured from the start
    # of the pass and therefore already includes the exact shortest path.
    # Do not add shortest again.  Older artifacts without the batch phase use
    # the exact-shortest-path time as the conservative pricing total.
    result["pricing_total_s"] = batch if batch > 0.0 else shortest
    result["pricing_exact_best_s"] = shortest
    iterations = integer(row.get("iterations"))
    columns = integer(row.get("pool_columns"))
    trips = integer(row.get("trip_count"))
    result["retained_pool_growth_per_iteration"] = (
        max(0, columns - trips) / iterations
        if iterations and columns is not None and trips is not None else ""
    )
    wall = number(row.get("cg_wall_s"))
    for name, seconds in (
        ("pricing_share_wall", float(result["pricing_total_s"])),
        ("master_share_wall", number(row.get("master_lp_s")) or 0.0),
        ("network_build_share_wall", number(row.get("network_build_s")) or 0.0),
        (
            "incidence_share_wall",
            number(row.get("incidence_construction_s")) or 0.0,
        ),
    ):
        result[name] = seconds / wall if wall and wall > 0 else ""
    result["pricing_exact_best_share_wall"] = (
        shortest / wall if wall and wall > 0 else ""
    )
    return result


def aggregate(scale: int, rows: list[dict[str, object]]) -> dict[str, object]:
    outcomes = Counter(str(row["outcome"]) for row in rows)
    certified = [row for row in rows if row["outcome"] == "certified"]
    probability = [row for row in rows if row["sample_family"] == "probability"]
    probability_certified = [
        row for row in probability if row["outcome"] == "certified"
    ]
    wall_values = [
        value for row in rows if (value := number(row.get("cg_wall_s"))) is not None
    ]
    certified_wall = [
        value
        for row in certified
        if (value := number(row.get("cg_wall_s"))) is not None
    ]
    iterations = [
        float(value)
        for row in rows
        if (value := integer(row.get("iterations"))) is not None
    ]
    columns = [
        float(value)
        for row in rows
        if (value := integer(row.get("pool_columns"))) is not None
    ]
    pool_growth = [
        value
        for row in rows
        if (
            value := number(row.get("retained_pool_growth_per_iteration"))
        ) is not None
    ]
    trips = [
        value
        for row in rows
        if (value := integer(row.get("trip_count"))) is not None
    ]
    total_wall = sum(wall_values)

    def phase_total(field: str) -> float:
        return sum(number(row.get(field)) or 0.0 for row in rows)

    return {
        "scale": scale,
        "rows": len(rows),
        "certified": outcomes["certified"],
        "wall_limit": outcomes["wall_limit"],
        "preempted": outcomes["preempted"],
        "execution_failure": outcomes["execution_failure"],
        "invalid_configuration": outcomes["invalid_configuration"],
        "fleet_target_proved": sum(truth(row["fleet_target_proved"]) for row in rows),
        "probability_rows": len(probability),
        "probability_certified": len(probability_certified),
        "certification_rate": outcomes["certified"] / len(rows),
        "probability_certification_rate": (
            len(probability_certified) / len(probability) if probability else ""
        ),
        "trip_count_min": min(trips) if trips else "",
        "trip_count_max": max(trips) if trips else "",
        "certified_wall_h_median": (
            median(certified_wall) / 3600 if certified_wall else ""
        ),
        "observed_wall_h_max": max(wall_values) / 3600 if wall_values else "",
        "iterations_median": median(iterations),
        "iterations_max": max(iterations) if iterations else "",
        "pool_columns_median": median(columns),
        "pool_columns_max": max(columns) if columns else "",
        "retained_pool_growth_per_iteration_median": median(pool_growth),
        "pricing_share_aggregate": (
            phase_total("pricing_total_s") / total_wall if total_wall else ""
        ),
        "pricing_exact_best_share_aggregate": (
            phase_total("pricing_exact_best_s") / total_wall
            if total_wall else ""
        ),
        "master_share_aggregate": (
            phase_total("master_lp_s") / total_wall if total_wall else ""
        ),
        "network_build_share_aggregate": (
            phase_total("network_build_s") / total_wall if total_wall else ""
        ),
        "incidence_share_aggregate": (
            phase_total("incidence_construction_s") / total_wall
            if total_wall else ""
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, object]]) -> None:
    columns = (
        "scale", "rows", "certified", "wall_limit", "preempted",
        "fleet_target_proved", "certification_rate", "trip_count_min",
        "trip_count_max", "certified_wall_h_median", "iterations_median",
        "pool_columns_median", "retained_pool_growth_per_iteration_median",
        "pricing_share_aggregate", "pricing_exact_best_share_aggregate",
        "master_share_aggregate", "network_build_share_aggregate",
    )
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(fmt(row[column]) for column in columns))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    source = root / "medium_event_summary.csv"
    if not source.is_file():
        raise SystemExit(f"missing audited CSV: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        original = list(csv.DictReader(handle))
    if not original:
        raise SystemExit("audited CSV is empty")
    rows = [enrich(row) for row in original]
    parsed_scales = {integer(row.get("scale")) for row in rows}
    if None in parsed_scales:
        raise SystemExit("audited row lacks scale")
    scales = sorted(scale for scale in parsed_scales if scale is not None)
    summary = [
        aggregate(scale, [row for row in rows if integer(row.get("scale")) == scale])
        for scale in scales
    ]
    row_output = root / "cg_frontier_rows.csv"
    scale_output = root / "cg_frontier_by_scale.csv"
    write_csv(row_output, rows)
    write_csv(scale_output, summary)
    print_table(summary)
    print("outcomes:", dict(Counter(str(row["outcome"]) for row in rows)))
    print(f"By-scale CSV: {scale_output}")
    print(f"Row-level CSV: {row_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

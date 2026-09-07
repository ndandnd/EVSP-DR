#!/usr/bin/env python3
"""Collect one analysis-ready snapshot of the k9--k15 threshold experiment.

The baseline contains all 70 pre-outcome-selected cells.  The continuation
contains only the 49 cells that reached the 12-hour cap.  This collector joins
them without treating the continuation as the full cohort, sums phase
telemetry across both stages, and emits a downsampled convergence table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PHASES = (
    "master_attempt",
    "incidence_construction",
    "pricing_shortest_path",
    "pricing_extra_columns",
    "route_insertion",
    "journal_fsync",
    "status_checkpoint",
)


def load_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"JSON artifact is not an object: {path}")
    return value


def finite(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_source_matrix(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if len(fields) != 11:
                raise SystemExit(f"unexpected baseline matrix width: {len(fields)}")
            rows.append({
                "source_index": int(fields[0]),
                "cell_id": fields[1],
                "scale": int(fields[2]),
                "selection_replicate": int(fields[3]),
                "trip_count": int(fields[4]),
                "representation_id": fields[7],
            })
    return rows


def read_resume_matrix(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {row["cell"]: row for row in rows}


def read_manifest(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["cell_id"]: row for row in csv.DictReader(handle)}


def telemetry(paths: list[Path]) -> tuple[dict[str, float], int, int, list[dict]]:
    totals: dict[str, float] = defaultdict(float)
    phase_rows = 0
    malformed = 0
    master_errors: list[dict] = []
    for path in paths:
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                malformed += 1
                if index != len(lines) - 1:
                    raise SystemExit(f"malformed telemetry before EOF: {path}")
                continue
            if record.get("record_type") != "phase":
                continue
            phase_rows += 1
            phase = str(record.get("phase"))
            totals[phase] += float(record.get("duration_s") or 0.0)
            if phase == "master_attempt" and record.get("outcome") == "error":
                master_errors.append(record)
    return totals, phase_rows, malformed, master_errors


def iteration_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            parsed = {key: finite(value) for key, value in row.items()}
            if parsed.get("iteration") is not None:
                rows.append(parsed)
    return rows


def sample_iterations(rows: list[dict], maximum: int = 160) -> list[dict]:
    if len(rows) <= maximum:
        return rows
    chosen = {0, len(rows) - 1}
    for index in range(1, maximum - 1):
        chosen.add(round(index * (len(rows) - 1) / (maximum - 1)))
    return [rows[index] for index in sorted(chosen)]


def tail_improvement(rows: list[dict], count: int) -> float | str:
    if len(rows) < 2:
        return ""
    start = rows[max(0, len(rows) - 1 - count)].get("lp_obj")
    end = rows[-1].get("lp_obj")
    if start is None or end is None:
        return ""
    return start - end


def classify(status: dict, cap_s: float, source_certified: bool) -> str:
    if status.get("certified_rc_optimal") is True:
        return "certified_le_12h" if source_certified else "certified_12_to_48h"
    stop = str(status.get("stop_reason") or "missing")
    wall_s = float(status.get("wall_s") or 0.0)
    if stop == "wall_limit" and wall_s >= cap_s - 120.0:
        return "wall_cap_48h"
    return stop


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    resume_root = args.resume_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source_plan = load_object(source_root / "execution_plan.json")
    resume_plan = load_object(resume_root / "execution_plan.json")
    if source_plan.get("schema") != "evsp-dr-threshold-9-15-event-cg-v1":
        raise SystemExit("unexpected baseline campaign schema")
    if resume_plan.get("schema") != "evsp-dr-threshold-9-15-resume48h-v1":
        raise SystemExit("unexpected continuation campaign schema")
    if int(source_plan.get("cells", -1)) != 70:
        raise SystemExit("baseline must contain 70 cells")
    if int(resume_plan.get("cells", -1)) != 49:
        raise SystemExit("continuation must contain 49 selected cells")

    source_rows = read_source_matrix(source_root / "matrix.tsv")
    resume_rows = read_resume_matrix(resume_root / "matrix.tsv")
    descriptors = read_manifest(source_root / "input_selection_manifest.csv")
    if len(source_rows) != 70 or len(descriptors) != 70:
        raise SystemExit("baseline matrix/manifest does not contain 70 cells")

    cap_s = float(resume_plan["cumulative_scientific_wall_limit_s"])
    detail_rows: list[dict] = []
    convergence_rows: list[dict] = []
    failure_rows: list[dict] = []

    for source in source_rows:
        cell = source["cell_id"]
        descriptor = descriptors.get(cell)
        if descriptor is None:
            raise SystemExit(f"missing descriptor for {cell}")
        source_status_path = source_root / "cg" / "b030_reduced" / (
            f"M__{cell}__{source['representation_id']}.json"
        )
        source_status = load_object(source_status_path)
        source_certified = source_status.get("certified_rc_optimal") is True
        resume_item = resume_rows.get(cell)
        if source_certified:
            status_path = source_status_path
            status = source_status
            telemetry_paths = [Path(str(source_status_path) + ".phase-telemetry.jsonl")]
            iteration_path = Path(str(source_status_path) + ".iters.csv")
            stage = "baseline"
        else:
            if resume_item is None:
                raise SystemExit(f"uncertified baseline cell absent from continuation: {cell}")
            status_path = Path(resume_item["resume_status"])
            status = load_object(status_path)
            telemetry_paths = [
                Path(str(source_status_path) + ".phase-telemetry.jsonl"),
                Path(str(status_path) + ".phase-telemetry.jsonl"),
            ]
            iteration_path = Path(str(status_path) + ".iters.csv")
            stage = "continuation"

        phases, phase_rows, malformed, master_errors = telemetry(telemetry_paths)
        iterations = iteration_rows(iteration_path)
        final = status.get("final") or {}
        metrics = status.get("network_metrics") or source_status.get("network_metrics") or {}
        wall_s = finite(status.get("wall_s")) or 0.0
        pricing_total_s = phases.get("pricing_extra_columns", 0.0)
        if pricing_total_s <= 0.0:
            pricing_total_s = phases.get("pricing_shortest_path", 0.0)
        measured_s = (
            phases.get("master_attempt", 0.0)
            + phases.get("incidence_construction", 0.0)
            + pricing_total_s
            + phases.get("route_insertion", 0.0)
            + phases.get("journal_fsync", 0.0)
            + phases.get("status_checkpoint", 0.0)
        )
        network_manifest = source_root / "network_cache" / (
            f"M__{cell}__{source['representation_id']}.pkl.manifest.json"
        )
        network = load_object(network_manifest)
        outcome = classify(status, cap_s, source_certified)
        detail = {
            "source_index": source["source_index"],
            "cell_id": cell,
            "scale": source["scale"],
            "sample_family": descriptor["sample_family"],
            "selection_role": descriptor["selection_role"],
            "family_replicate": descriptor["family_replicate"],
            "nested_chain_id": descriptor["nested_chain_id"],
            "nested_parent_cell_id": descriptor["nested_parent_cell_id"],
            "duties_json": descriptor["duties_json"],
            "trip_count": descriptor["trip_count"],
            "peak_concurrency_lb": descriptor["peak_concurrency_lb"],
            "direct_compatibility_density": descriptor["direct_compatibility_density"],
            "service_kwh_total": descriptor["service_kwh_total"],
            "service_kwh_per_duty": descriptor["service_kwh_per_duty"],
            "duty_trip_count_max": descriptor["duty_trip_count_max"],
            "duty_trip_count_cv": descriptor["duty_trip_count_cv"],
            "scheduled_intertrip_gap_median_min": descriptor["scheduled_intertrip_gap_median_min"],
            "service_span_min": descriptor["service_span_min"],
            "outcome": outcome,
            "result_stage": stage,
            "certified_rc_optimal": status.get("certified_rc_optimal", ""),
            "stop_reason": status.get("stop_reason", ""),
            "cumulative_wall_h": wall_s / 3600.0,
            "iterations": status.get("iterations", ""),
            "pool_columns": status.get("columns", ""),
            "retained_pool_growth_per_iteration": (
                (int(status.get("columns", 0)) - int(source["trip_count"]))
                / int(status.get("iterations", 1))
                if int(status.get("iterations", 0)) > 0 else ""
            ),
            "lp_objective": final.get("lp_obj", ""),
            "route_weight_endpoint": final.get("route_weight", ""),
            "artificials": final.get("artificials", ""),
            "minimum_reduced_cost": final.get("min_rc", ""),
            "objective_improvement_last_100_iterations": tail_improvement(iterations, 100),
            "objective_improvement_last_500_iterations": tail_improvement(iterations, 500),
            "network_original_build_s": network.get("original_build_s", ""),
            "network_dag_nodes": metrics.get("dag_nodes", ""),
            "network_dag_arcs": metrics.get("dag_arcs", ""),
            "phase_rows": phase_rows,
            "telemetry_malformed_rows": malformed,
            "master_lp_s": phases.get("master_attempt", 0.0),
            "master_share": phases.get("master_attempt", 0.0) / wall_s if wall_s else "",
            "incidence_construction_s": phases.get("incidence_construction", 0.0),
            "incidence_share": phases.get("incidence_construction", 0.0) / wall_s if wall_s else "",
            "pricing_total_s": pricing_total_s,
            "pricing_share": pricing_total_s / wall_s if wall_s else "",
            "pricing_exact_best_s": phases.get("pricing_shortest_path", 0.0),
            "pricing_exact_best_share": phases.get("pricing_shortest_path", 0.0) / wall_s if wall_s else "",
            "route_insertion_s": phases.get("route_insertion", 0.0),
            "route_insertion_share": phases.get("route_insertion", 0.0) / wall_s if wall_s else "",
            "checkpoint_io_s": phases.get("journal_fsync", 0.0) + phases.get("status_checkpoint", 0.0),
            "checkpoint_io_share": (phases.get("journal_fsync", 0.0) + phases.get("status_checkpoint", 0.0)) / wall_s if wall_s else "",
            "unattributed_s": max(0.0, wall_s - measured_s),
            "unattributed_share": max(0.0, wall_s - measured_s) / wall_s if wall_s else "",
            "master_error_count": len(master_errors),
            "master_backend": "scipy_highs",
            "status_path": str(status_path),
        }
        detail_rows.append(detail)

        for point in sample_iterations(iterations):
            convergence_rows.append({
                "cell_id": cell,
                "scale": source["scale"],
                "sample_family": descriptor["sample_family"],
                "selection_role": descriptor["selection_role"],
                "family_replicate": descriptor["family_replicate"],
                "outcome": outcome,
                "elapsed_h": (point.get("elapsed_s") or 0.0) / 3600.0,
                "iteration": point.get("iteration", ""),
                "lp_objective": point.get("lp_obj", ""),
                "route_weight_endpoint": point.get("route_weight", ""),
                "artificials": point.get("artificials", ""),
                "minimum_reduced_cost": point.get("min_rc", ""),
                "pool_columns": point.get("pool_columns", ""),
            })

        for record in master_errors:
            info = record.get("details") or {}
            failure_rows.append({
                "cell_id": cell,
                "scale": source["scale"],
                "iteration": record.get("iteration", ""),
                "attempt": record.get("attempt", ""),
                "purpose": info.get("purpose", "main"),
                "method": info.get("method", ""),
                "duration_s": record.get("duration_s", ""),
                "error": info.get("error", ""),
            })

    output_root.mkdir(parents=True, exist_ok=False)
    write_csv(output_root / "threshold_deep_dive_rows.csv", detail_rows)
    write_csv(output_root / "threshold_deep_dive_convergence.csv", convergence_rows)
    write_csv(output_root / "threshold_master_failures.csv", failure_rows)
    metadata = {
        "schema": "evsp-dr-threshold-deep-dive-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "resume_root": str(resume_root),
        "rows": len(detail_rows),
        "outcomes": dict(sorted(Counter(row["outcome"] for row in detail_rows).items())),
        "convergence_rows": len(convergence_rows),
        "master_error_rows": len(failure_rows),
        "notes": {
            "pricing_total": "pricing_extra_columns includes the shortest-path work; do not add pricing_exact_best again",
            "network_build": "separate cached preprocessing, excluded from the 12h/48h CG wall budget",
            "master_backend": "these historical threshold runs use SciPy/HiGHS, not Gurobi",
        },
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Deep-dive outcomes:", metadata["outcomes"])
    print(f"Rows CSV: {output_root / 'threshold_deep_dive_rows.csv'}")
    print(f"Convergence CSV: {output_root / 'threshold_deep_dive_convergence.csv'}")
    print(f"Master failures CSV: {output_root / 'threshold_master_failures.csv'}")
    print(f"Metadata: {output_root / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

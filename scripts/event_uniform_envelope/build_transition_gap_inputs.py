#!/usr/bin/env python3
"""Build pre-outcome k3/k4/k6/k7 probability and structural-stress inputs.

The selected instances are unions of original GIRO duty trip sequences.  Each
individual sequence is first reoptimized and physically replayed under the
current 240 kWh / 240 kW continuous fixed-duty model.  This establishes a
physical k-route upper bound without injecting those routes into column
generation.  It does not establish event-lattice representability or
optimality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import tempfile
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from audit_giro_known_columns import HORIZON_MIN, build_problem  # noqa: E402
from config import CHARGING_STATIONS  # noqa: E402
from fixed_duty_continuous_optimizer import (  # noqa: E402
    optimize_fixed_duty_continuous,
)
from make_duty_pair_instances import (  # noqa: E402
    _base_task,
    _peak_concurrency,
    load_duty_frames,
    merge_duties,
)
from utils_v2 import load_station_hourly_prices  # noqa: E402


SEED = 20260902
SCALES = (3, 4, 6, 7)
CANDIDATES_PER_SCALE = 512
PROBABILITY_PER_SCALE = 6
STRESS_RULES = (
    ("trip_heavy", "trip_count", "max"),
    ("energy_heavy", "service_kwh_per_duty", "max"),
    ("tight_gap", "scheduled_intertrip_gap_median_min", "min"),
)
DEFAULT_OUTPUT = (
    REPO / "data" / "scale_ladder" / "instances"
    / "transition_gap_20260902"
)
TARIFF = REPO / "data" / "tariff_response" / "flat_h26.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def minutes(value: object) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)


def duty_gaps(frame: pd.DataFrame) -> list[float]:
    ordered = frame.assign(_start=frame["Start1"].map(minutes)).sort_values(
        ["_start", "Ordered_Trip_ID"]
    )
    starts = [minutes(value) for value in ordered["Start1"]]
    ends = [minutes(value) for value in ordered["End1"]]
    return [float(starts[i] - ends[i - 1]) for i in range(1, len(starts))]


def candidate_features(
    frames: dict[str, pd.DataFrame], duties: list[str]
) -> dict[str, float | int]:
    merged = merge_duties(frames, duties)
    energy = float(pd.to_numeric(
        merged["Usage kWh"], errors="coerce"
    ).fillna(0.0).sum())
    gaps = [gap for duty in duties for gap in duty_gaps(frames[duty])]
    counts = [len(frames[duty]) for duty in duties]
    starts = [minutes(value) for value in merged["Start1"]]
    ends = [minutes(value) for value in merged["End1"]]
    mean_count = sum(counts) / len(counts)
    variance = sum((count - mean_count) ** 2 for count in counts) / len(counts)
    return {
        "trip_count": len(merged),
        "peak_concurrency_lb": _peak_concurrency(merged),
        "service_kwh_total": round(energy, 9),
        "service_kwh_per_duty": round(energy / len(duties), 9),
        "duty_trip_count_max": max(counts),
        "duty_trip_count_cv": round(
            math.sqrt(variance) / mean_count if mean_count else 0.0, 9
        ),
        "scheduled_intertrip_gap_median_min": round(
            float(statistics.median(gaps)) if gaps else math.inf, 9
        ),
        "service_span_min": max(ends) - min(starts),
    }


def existing_duty_sets(manifest: Path) -> set[tuple[str, ...]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        return {
            tuple(sorted(json.loads(row["duties_json"])))
            for row in csv.DictReader(handle)
        }


def physical_duty_certificates(
    frames: dict[str, pd.DataFrame], output: Path
) -> dict[str, dict[str, object]]:
    prices = load_station_hourly_prices(TARIFF, CHARGING_STATIONS)
    records: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="evsp-transition-duties-") as tmp:
        temporary = Path(tmp)
        for duty, frame in sorted(frames.items()):
            instance = temporary / f"duty_{duty}.csv"
            frame.to_csv(instance, index=False, lineterminator="\n")
            instance_sha = sha256(instance)
            problem = build_problem(
                temporary,
                instance.name,
                max_station_to_trip_wait_min=HORIZON_MIN,
                reference_data_dir=REPO / "data",
            )
            result = optimize_fixed_duty_continuous(
                problem,
                list(problem.trips),
                prices,
                g_kwh=240.0,
                charge_kw=240.0,
                reserve_kwh=0.0,
                charge_start_cost=5.0,
                terminal_soc_policy="free",
                timing_mode="optimized",
                tariff_id="flat_h26",
                tariff_sha256=sha256(TARIFF),
                instance_sha256=instance_sha,
                time_limit_s=60.0,
            )
            certificate = result.get("certificate") or {}
            if (
                result.get("feasible") is not True
                or result.get("physical_replay_status") != "validated"
                or certificate.get("certified") is not True
                or certificate.get("scope")
                != "optimal_continuous_charging_for_fixed_trip_sequence"
            ):
                raise ValueError(f"current-physics duty validation failed: {duty}")
            records[duty] = {
                "duty_id": duty,
                "base_duty_id": _base_task(duty),
                "trip_count": len(frame),
                "duty_instance_sha256": instance_sha,
                "continuous_physical_feasible_240_240": True,
                "physical_replay_status": result["physical_replay_status"],
                "certificate_scope": certificate["scope"],
                "certificate_sha256": certificate["certificate_sha256"],
                "solver": certificate["solver"],
                "scipy_version": certificate["scipy_version"],
                "implementation_sha256": certificate["implementation_sha256"],
                "problem_identity_sha256": certificate[
                    "problem_identity_sha256"
                ],
                "replay_sha256": certificate["replay_sha256"],
                "objective": round(float(result["objective"]), 9),
                "terminal_soc_kwh": round(float(result["terminal_soc_kwh"]), 9),
                "charge_events": int(result["charge_events"]),
            }
    fields = list(next(iter(records.values())))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records[duty] for duty in sorted(records))
    return records


def generate_candidates(
    frames: dict[str, pd.DataFrame],
    physical: dict[str, dict[str, object]],
    exclusion_manifest: Path,
) -> list[dict[str, object]]:
    duties = sorted(physical)
    excluded = existing_duty_sets(exclusion_manifest)
    candidates: list[dict[str, object]] = []
    for scale in SCALES:
        rng = random.Random(SEED * 100 + scale)
        seen: set[tuple[str, ...]] = set()
        attempts = 0
        while len(seen) < CANDIDATES_PER_SCALE:
            attempts += 1
            if attempts > CANDIDATES_PER_SCALE * 1000:
                raise RuntimeError(f"could not build k{scale} candidate universe")
            selected = tuple(sorted(rng.sample(duties, scale)))
            if len({_base_task(duty) for duty in selected}) != scale:
                continue
            if selected in excluded or selected in seen:
                continue
            seen.add(selected)
            record: dict[str, object] = {
                "scale": scale,
                "candidate_rank": len(seen),
                "duties_json": json.dumps(selected, separators=(",", ":")),
                "duty_set_sha256": canonical_sha(selected),
                "candidate_seed": SEED * 100 + scale,
                "selected_family": "",
                "selected_role": "",
            }
            record.update(candidate_features(frames, list(selected)))
            candidates.append(record)
    return candidates


def choose(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for scale in SCALES:
        group = [row for row in candidates if row["scale"] == scale]
        probability = group[:PROBABILITY_PER_SCALE]
        used = {str(row["duty_set_sha256"]) for row in probability}
        for replicate, row in enumerate(probability, start=1):
            row["selected_family"] = "probability"
            row["selected_role"] = f"fixed_seed_probability_{replicate}"
            selected.append(row)
        for role, field, direction in STRESS_RULES:
            ordered = sorted(
                group[PROBABILITY_PER_SCALE:],
                key=lambda row: (
                    -float(row[field]) if direction == "max" else float(row[field]),
                    str(row["duties_json"]),
                ),
            )
            winner = next(
                row for row in ordered
                if str(row["duty_set_sha256"]) not in used
            )
            used.add(str(winner["duty_set_sha256"]))
            winner["selected_family"] = "stress"
            winner["selected_role"] = role
            selected.append(winner)
    return selected


def direct_compatibility_density(instance: Path) -> float:
    problem = build_problem(
        instance.parent,
        instance.name,
        reference_data_dir=REPO / "data",
    )
    direct = sum(
        1
        for node, arcs in problem.adjacency.items()
        if isinstance(node, int)
        for _successor, _travel, _energy, kind in arcs
        if kind == "trip_trip"
    )
    denominator = len(problem.trips) * (len(problem.trips) - 1) / 2
    return direct / denominator if denominator else 0.0


def write_candidate_universe(
    candidates: list[dict[str, object]], output: Path
) -> None:
    fields = list(candidates[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(candidates)


def write_selected(
    frames: dict[str, pd.DataFrame],
    physical: dict[str, dict[str, object]],
    selected: list[dict[str, object]],
    output_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_scale_count = {scale: 0 for scale in SCALES}
    stress_codes = {
        "trip_heavy": "xtrip",
        "energy_heavy": "xenergy",
        "tight_gap": "xgap",
    }
    for candidate in sorted(selected, key=lambda row: (
        int(row["scale"]),
        0 if row["selected_family"] == "probability" else 1,
        int(row["candidate_rank"]),
    )):
        scale = int(candidate["scale"])
        by_scale_count[scale] += 1
        selection_replicate = by_scale_count[scale]
        family = str(candidate["selected_family"])
        role = str(candidate["selected_role"])
        if family == "probability":
            family_replicate = int(role.rsplit("_", 1)[1])
            cell_id = f"k{scale:02d}_p{family_replicate}"
            file_token = f"p{family_replicate:02d}"
        else:
            family_replicate = 1 + [rule[0] for rule in STRESS_RULES].index(role)
            cell_id = f"k{scale:02d}_{stress_codes[role]}"
            file_token = stress_codes[role]
        duties = list(json.loads(str(candidate["duties_json"])))
        instance = output_dir / (
            f"Practice_Custom_DutyUnion_k{scale:02d}_{file_token}_20260902.csv"
        )
        frame = merge_duties(frames, duties)
        frame.to_csv(instance, index=False, lineterminator="\n")
        certificate_set = sorted(
            str(physical[duty]["certificate_sha256"]) for duty in duties
        )
        rows.append({
            "cell_id": cell_id,
            "scale": scale,
            "selection_replicate": selection_replicate,
            "sample_family": family,
            "family_replicate": family_replicate,
            "selection_role": role,
            "candidate_rank": candidate["candidate_rank"],
            "candidate_seed": candidate["candidate_seed"],
            "relative_path": str(instance.relative_to(REPO)),
            "instance_file_sha256": sha256(instance),
            "duties_json": candidate["duties_json"],
            "duty_set_sha256": candidate["duty_set_sha256"],
            "target_fleet": scale,
            "known_partition_continuous_physical_upper_bound": True,
            "known_duty_certificate_set_sha256": canonical_sha(certificate_set),
            "trip_count": candidate["trip_count"],
            "peak_concurrency_lb": candidate["peak_concurrency_lb"],
            "direct_compatibility_density": round(
                direct_compatibility_density(instance), 12
            ),
            "service_kwh_total": candidate["service_kwh_total"],
            "service_kwh_per_duty": candidate["service_kwh_per_duty"],
            "duty_trip_count_max": candidate["duty_trip_count_max"],
            "duty_trip_count_cv": candidate["duty_trip_count_cv"],
            "scheduled_intertrip_gap_median_min":
                candidate["scheduled_intertrip_gap_median_min"],
            "service_span_min": candidate["service_span_min"],
        })
    return rows


def write_manifest(rows: list[dict[str, object]], output: Path) -> None:
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--existing-manifest",
        type=Path,
        required=True,
        help=(
            "The validated six-selection scale-ladder manifest whose duty "
            "sets must be excluded (expected source commit ff7fb2b)."
        ),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    exclusion_manifest = args.existing_manifest.resolve()
    if not exclusion_manifest.is_file():
        raise SystemExit(f"missing exclusion manifest: {exclusion_manifest}")
    if not output_dir.is_relative_to(REPO):
        raise SystemExit("output directory must be inside the repository")
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    frames = load_duty_frames()
    duty_path = output_dir / "known_duty_continuous_240_240.csv"
    physical = physical_duty_certificates(frames, duty_path)
    candidates = generate_candidates(frames, physical, exclusion_manifest)
    selected = choose(candidates)
    write_candidate_universe(
        candidates, output_dir / "candidate_universe.csv"
    )
    rows = write_selected(frames, physical, selected, output_dir)
    manifest = output_dir / "selection_manifest.csv"
    write_manifest(rows, manifest)

    plan = {
        "schema": "evsp-dr-transition-gap-inputs-v1",
        "created_date": "2026-09-02",
        "generator_script_sha256": sha256(Path(__file__).resolve()),
        "generator_seed": SEED,
        "scales": list(SCALES),
        "candidate_universe_per_scale": CANDIDATES_PER_SCALE,
        "probability_per_scale": PROBABILITY_PER_SCALE,
        "stress_rules": [
            {"role": role, "field": field, "direction": direction}
            for role, field, direction in STRESS_RULES
        ],
        "existing_set_exclusion_manifest": (
            "excluded_existing_scale_ladder_manifest.csv"
        ),
        "existing_set_exclusion_source_commit": (
            "ff7fb2ba93cf13a31171e1e4aeb2d28dc8aeee20"
        ),
        "existing_set_exclusion_manifest_sha256": sha256(
            exclusion_manifest
        ),
        "source_master": "data/Par_VehicleDetails_Updated.csv",
        "source_master_sha256": sha256(
            REPO / "data" / "Par_VehicleDetails_Updated.csv"
        ),
        "reference_sha256": sha256(REPO / "data" / "Ref_dict.csv"),
        "deadhead_sha256": sha256(REPO / "data" / "par_ref_dhd.csv"),
        "tariff": str(TARIFF.relative_to(REPO)),
        "tariff_sha256": sha256(TARIFF),
        "known_partition_scope": (
            "each original GIRO duty trip order has an independently "
            "optimized and replay-validated continuous schedule"
        ),
        "known_partition_caveat": (
            "physical k-route upper bound only; not an event-lattice "
            "representability certificate and not injected into raw CG"
        ),
        "physics": {
            "battery_kwh": 240.0,
            "charge_kw": 240.0,
            "reserve_kwh": 0.0,
            "charge_start_cost": 5.0,
            "terminal_soc_policy": "free",
        },
        "files": {
            "selection_manifest.csv": sha256(manifest),
            "candidate_universe.csv": sha256(
                output_dir / "candidate_universe.csv"
            ),
            "known_duty_continuous_240_240.csv": sha256(duty_path),
            "excluded_existing_scale_ladder_manifest.csv": sha256(
                exclusion_manifest
            ),
        },
        "selected_rows": len(rows),
    }
    (output_dir / "input_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "excluded_existing_scale_ladder_manifest.csv").write_bytes(
        exclusion_manifest.read_bytes()
    )
    print(f"wrote {len(rows)} selected instances under {output_dir}")
    print("selection counts: 6 probability + 3 stress at each k=3,4,6,7")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

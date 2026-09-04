#!/usr/bin/env python3
"""Fail-closed validation for the frozen small-threshold input cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_SCALES = (2, 5, 8, 9, 10)
STRUCTURAL_RULES = (
    ("trip_light", "trip_count", "min"),
    ("trip_heavy", "trip_count", "max"),
    ("energy_heavy", "service_kwh_per_duty", "max"),
    ("tight_gap", "scheduled_intertrip_gap_median_min", "min"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def base_duty(value: str) -> str:
    match = re.match(r"(\d+)", value)
    return match.group(1) if match else value


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--campaign", default="small_threshold_20260903",
        help="directory below data/scale_ladder/instances",
    )
    parser.add_argument(
        "--schema", default="evsp-dr-small-threshold-inputs-v1"
    )
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--scales", type=int, nargs="+", default=list(DEFAULT_SCALES)
    )
    parser.add_argument(
        "--generator-script", default=(
            "scripts/event_uniform_envelope/build_small_threshold_inputs.py"
        ),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    scales = tuple(args.scales)
    root = (
        repo / "data" / "scale_ladder" / "instances"
        / args.campaign
    )
    plan = json.loads((root / "input_plan.json").read_text(encoding="utf-8"))
    expected_plan = {
        "schema": args.schema,
        "generator_seed": args.seed,
        "scales": list(scales),
        "candidate_universe_per_scale": 512,
        "probability_per_scale": 6,
        "selected_rows": 10 * len(scales),
    }
    observed_plan = {key: plan.get(key) for key in expected_plan}
    if observed_plan != expected_plan:
        raise SystemExit(f"input-plan identity mismatch: {observed_plan}")
    source_files = {
        "generator_script_sha256": (
            repo / args.generator_script
        ),
        "source_master_sha256": repo / plan["source_master"],
        "reference_sha256": repo / "data" / "Ref_dict.csv",
        "deadhead_sha256": repo / "data" / "par_ref_dhd.csv",
        "tariff_sha256": repo / plan["tariff"],
    }
    for field, path in source_files.items():
        if sha256(path) != plan[field]:
            raise SystemExit(f"source identity mismatch: {field}")
    if "shared_builder_sha256" in plan:
        shared_builder = (
            repo / "scripts" / "event_uniform_envelope"
            / "build_small_threshold_inputs.py"
        )
        if sha256(shared_builder) != plan["shared_builder_sha256"]:
            raise SystemExit("source identity mismatch: shared_builder_sha256")
    for name, expected in plan["files"].items():
        if sha256(root / name) != expected:
            raise SystemExit(f"input-plan file hash mismatch: {name}")

    candidates = read_csv(root / "candidate_universe.csv")
    selected = read_csv(root / "selection_manifest.csv")
    duties = {
        row["duty_id"]: row
        for row in read_csv(root / "known_duty_continuous_240_240.csv")
    }
    exclusion_names = plan.get("existing_set_exclusion_manifests") or [
        plan["existing_set_exclusion_manifest"]
    ]
    excluded = {
        tuple(sorted(json.loads(row["duties_json"])))
        for name in exclusion_names
        for row in read_csv(root / name)
    }
    if len(duties) != 42 or not all(
        truth(row["continuous_physical_feasible_240_240"])
        and row["physical_replay_status"] == "validated"
        for row in duties.values()
    ):
        raise SystemExit("known-duty current-physics validation mismatch")

    counts = Counter(int(row["scale"]) for row in candidates)
    if counts != Counter({scale: 512 for scale in scales}):
        raise SystemExit(f"candidate universe count mismatch: {counts}")
    for scale in scales:
        group = [row for row in candidates if int(row["scale"]) == scale]
        if [int(row["candidate_rank"]) for row in group] != list(range(1, 513)):
            raise SystemExit(f"candidate ranks differ at k{scale}")
        for row in group:
            duty_set = tuple(json.loads(row["duties_json"]))
            if (
                len(duty_set) != scale
                or len(set(duty_set)) != scale
                or len({base_duty(duty) for duty in duty_set}) != scale
                or duty_set in excluded
                or canonical_sha(duty_set) != row["duty_set_sha256"]
            ):
                raise SystemExit(f"invalid candidate duty set: {duty_set}")

    expected_selected: dict[str, tuple[str, str]] = {}
    for scale in scales:
        group = [row for row in candidates if int(row["scale"]) == scale]
        used: set[str] = set()
        for replicate, row in enumerate(group[:6], start=1):
            digest = row["duty_set_sha256"]
            used.add(digest)
            expected_selected[digest] = (
                "probability", f"fixed_seed_probability_{replicate}"
            )
        for role, field, direction in STRUCTURAL_RULES:
            ordered = sorted(
                group[6:],
                key=lambda row: (
                    -float(row[field]) if direction == "max" else float(row[field]),
                    row["duties_json"],
                ),
            )
            winner = next(
                row for row in ordered if row["duty_set_sha256"] not in used
            )
            used.add(winner["duty_set_sha256"])
            expected_selected[winner["duty_set_sha256"]] = ("structural", role)

    expected_rows = 10 * len(scales)
    if len(selected) != expected_rows or len(expected_selected) != expected_rows:
        raise SystemExit("selected-row count mismatch")
    if len({row["cell_id"] for row in selected}) != expected_rows:
        raise SystemExit("duplicate selected cell id")
    for row in selected:
        digest = row["duty_set_sha256"]
        if expected_selected.get(digest) != (
            row["sample_family"], row["selection_role"]
        ):
            raise SystemExit(f"selection-rule mismatch: {row['cell_id']}")
        duty_set = list(json.loads(row["duties_json"]))
        if not all(duty in duties for duty in duty_set):
            raise SystemExit(f"unknown duty in {row['cell_id']}")
        certificates = sorted(duties[duty]["certificate_sha256"] for duty in duty_set)
        if (
            not truth(row["known_partition_continuous_physical_upper_bound"])
            or canonical_sha(certificates)
            != row["known_duty_certificate_set_sha256"]
        ):
            raise SystemExit(f"physical upper-bound mismatch: {row['cell_id']}")
        instance = repo / row["relative_path"]
        if sha256(instance) != row["instance_file_sha256"]:
            raise SystemExit(f"selected instance hash mismatch: {row['cell_id']}")
        instance_rows = read_csv(instance)
        if (
            len(instance_rows) != int(row["trip_count"])
            or len({item["Ordered_Trip_ID"] for item in instance_rows})
            != len(instance_rows)
            or not 0 <= float(row["direct_compatibility_density"]) <= 1
        ):
            raise SystemExit(f"selected instance content mismatch: {row['cell_id']}")

    if str(plan.get("probability_design", "")).startswith(
        "six_nested_chains_"
    ):
        for replicate in range(1, 7):
            chain = sorted(
                (
                    row for row in selected
                    if row["sample_family"] == "probability"
                    and int(row["family_replicate"]) == replicate
                ),
                key=lambda row: int(row["scale"]),
            )
            if [int(row["scale"]) for row in chain] != list(scales):
                raise SystemExit(f"nested-chain scale mismatch: {replicate}")
            previous: set[str] | None = None
            for row in chain:
                current = set(json.loads(row["duties_json"]))
                scale = int(row["scale"])
                expected_parent = (
                    f"k{scale - 1:02d}_p{replicate}"
                    if scale > min(scales) else ""
                )
                if (
                    row.get("nested_chain_id")
                    != f"nested_probability_{replicate}"
                    or row.get("nested_parent_cell_id", "") != expected_parent
                    or (previous is not None and not (
                        previous < current and len(current - previous) == 1
                    ))
                ):
                    raise SystemExit(
                        f"nested-chain relation mismatch: {row['cell_id']}"
                    )
                previous = current

    selected_counts = Counter(
        (int(row["scale"]), row["sample_family"]) for row in selected
    )
    print(f"validated {args.campaign} inputs: rows={len(selected)}")
    print(f"scale/family counts: {dict(sorted(selected_counts.items()))}")
    print("known GIRO duty orders physically feasible at 240/240: 42/42")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

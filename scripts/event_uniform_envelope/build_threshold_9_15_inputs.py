#!/usr/bin/env python3
"""Build pre-outcome k9--k15 threshold-study inputs.

This is a new frozen cohort, separate from the 2026-09-03 k2/k5/k8/k9/k10
cohort.  It deliberately uses the same sampling design: six fixed-seed
probability samples and four structural stress samples at every scale.
Selection uses input descriptors only, never solver outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.event_uniform_envelope import build_small_threshold_inputs as base  # noqa: E402


SEED = 20260904
SCALES = (9, 10, 11, 12, 13, 14, 15)
CAMPAIGN_DATE = "20260904"
DEFAULT_OUTPUT = (
    REPO / "data" / "scale_ladder" / "instances"
    / "threshold_9_15_20260904"
)


def candidate_record(
    frames,
    duties: tuple[str, ...],
    scale: int,
    rank: int,
    seed: int,
    nested_chain_id: str = "",
) -> dict[str, object]:
    record: dict[str, object] = {
        "scale": scale,
        "candidate_rank": rank,
        "duties_json": json.dumps(duties, separators=(",", ":")),
        "duty_set_sha256": base.canonical_sha(duties),
        "candidate_seed": seed,
        "nested_chain_id": nested_chain_id,
        "selected_family": "",
        "selected_role": "",
    }
    record.update(base.candidate_features(frames, list(duties)))
    return record


def generate_candidates(frames, physical, exclusion_manifests: list[Path]):
    """Generate six nested probability chains plus independent stress pools."""
    available = sorted(physical)
    excluded = set().union(*(
        base.existing_duty_sets(path) for path in exclusion_manifests
    ))
    candidates: list[dict[str, object]] = []
    seen = {scale: set() for scale in SCALES}

    # Each chain samples an ordered k15 set once.  Its k9--k14 prefixes make
    # every adjacent pair differ by exactly one duty.
    for replicate in range(1, base.PROBABILITY_PER_SCALE + 1):
        chain_seed = SEED * 1000 + replicate
        rng = random.Random(chain_seed)
        for _attempt in range(100000):
            ordered = rng.sample(available, max(SCALES))
            if len({base._base_task(duty) for duty in ordered}) != max(SCALES):
                continue
            prefixes = {
                scale: tuple(sorted(ordered[:scale])) for scale in SCALES
            }
            if any(
                duties in excluded or duties in seen[scale]
                for scale, duties in prefixes.items()
            ):
                continue
            break
        else:
            raise RuntimeError(f"could not construct nested chain {replicate}")
        chain_id = f"nested_probability_{replicate}"
        for scale, duties in prefixes.items():
            seen[scale].add(duties)
            candidates.append(candidate_record(
                frames, duties, scale, replicate, chain_seed, chain_id
            ))

    # The remaining 506 candidates per scale form the independent universe
    # from which the four pre-outcome structural extremes are selected.
    for scale in SCALES:
        rng = random.Random(SEED * 100 + scale)
        attempts = 0
        while len(seen[scale]) < base.CANDIDATES_PER_SCALE:
            attempts += 1
            if attempts > base.CANDIDATES_PER_SCALE * 1000:
                raise RuntimeError(f"could not build k{scale} candidate universe")
            duties = tuple(sorted(rng.sample(available, scale)))
            if len({base._base_task(duty) for duty in duties}) != scale:
                continue
            if duties in excluded or duties in seen[scale]:
                continue
            seen[scale].add(duties)
            candidates.append(candidate_record(
                frames, duties, scale, len(seen[scale]),
                SEED * 100 + scale,
            ))
    candidates.sort(key=lambda row: (
        int(row["scale"]), int(row["candidate_rank"])
    ))
    return candidates


def write_selected(
    frames,
    physical,
    selected: list[dict[str, object]],
    output_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_scale_count = {scale: 0 for scale in SCALES}
    stress_codes = {
        "trip_light": "xlight",
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
            family_replicate = 1 + [
                rule[0] for rule in base.STRUCTURAL_RULES
            ].index(role)
            cell_id = f"k{scale:02d}_{stress_codes[role]}"
            file_token = stress_codes[role]
        duties = list(json.loads(str(candidate["duties_json"])))
        instance = output_dir / (
            f"Practice_Custom_DutyUnion_k{scale:02d}_{file_token}_"
            f"{CAMPAIGN_DATE}.csv"
        )
        frame = base.merge_duties(frames, duties)
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
            "nested_chain_id": candidate.get("nested_chain_id", ""),
            "nested_parent_cell_id": (
                f"k{scale - 1:02d}_p{family_replicate}"
                if family == "probability" and scale > min(SCALES) else ""
            ),
            "candidate_rank": candidate["candidate_rank"],
            "candidate_seed": candidate["candidate_seed"],
            "relative_path": str(instance.relative_to(REPO)),
            "instance_file_sha256": base.sha256(instance),
            "duties_json": candidate["duties_json"],
            "duty_set_sha256": candidate["duty_set_sha256"],
            "target_fleet": scale,
            "known_partition_continuous_physical_upper_bound": True,
            "known_duty_certificate_set_sha256": base.canonical_sha(
                certificate_set
            ),
            "trip_count": candidate["trip_count"],
            "peak_concurrency_lb": candidate["peak_concurrency_lb"],
            "direct_compatibility_density": round(
                base.direct_compatibility_density(instance), 12
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scale-ladder-manifest", type=Path, required=True)
    parser.add_argument("--small-threshold-manifest", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    exclusion_manifests = [
        args.scale_ladder_manifest.resolve(),
        args.small_threshold_manifest.resolve(),
    ]
    for path in exclusion_manifests:
        if not path.is_file():
            raise SystemExit(f"missing exclusion manifest: {path}")
    if not output_dir.is_relative_to(REPO):
        raise SystemExit("output directory must be inside the repository")
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    # Reuse the reviewed physical validation and pre-outcome sampling code while
    # changing only the immutable seed and scale grid for this new cohort.
    base.SEED = SEED
    base.SCALES = SCALES
    frames = base.load_duty_frames()
    duty_path = output_dir / "known_duty_continuous_240_240.csv"
    physical = base.physical_duty_certificates(frames, duty_path)
    candidates = generate_candidates(frames, physical, exclusion_manifests)
    selected = base.choose(candidates)
    base.write_candidate_universe(
        candidates, output_dir / "candidate_universe.csv"
    )
    rows = write_selected(frames, physical, selected, output_dir)
    manifest = output_dir / "selection_manifest.csv"
    base.write_manifest(rows, manifest)

    copied_exclusions = {
        "excluded_existing_scale_ladder_manifest.csv": exclusion_manifests[0],
        "excluded_small_threshold_manifest.csv": exclusion_manifests[1],
    }
    for name, source in copied_exclusions.items():
        (output_dir / name).write_bytes(source.read_bytes())
    plan = {
        "schema": "evsp-dr-threshold-9-15-inputs-v1",
        "created_date": "2026-09-04",
        "generator_script_sha256": base.sha256(Path(__file__).resolve()),
        "shared_builder_sha256": base.sha256(Path(base.__file__).resolve()),
        "generator_seed": SEED,
        "scales": list(SCALES),
        "candidate_universe_per_scale": base.CANDIDATES_PER_SCALE,
        "probability_per_scale": base.PROBABILITY_PER_SCALE,
        "probability_design": "six_nested_chains_k9_through_k15",
        "structural_rules": [
            {"role": role, "field": field, "direction": direction}
            for role, field, direction in base.STRUCTURAL_RULES
        ],
        "selection_uses_solver_outcomes": False,
        "existing_set_exclusion_manifests": list(copied_exclusions),
        "existing_set_exclusion_sources": [
            {
                "name": "reviewed_six_selection_scale_ladder",
                "source_commit":
                    "ff7fb2ba93cf13a31171e1e4aeb2d28dc8aeee20",
                "sha256": base.sha256(exclusion_manifests[0]),
            },
            {
                "name": "small_threshold_20260903",
                "source_commit":
                    "91e05d7590e4fe310137bf93ad7f6b4438dc3d72",
                "sha256": base.sha256(exclusion_manifests[1]),
            },
        ],
        "source_master": "data/Par_VehicleDetails_Updated.csv",
        "source_master_sha256": base.sha256(
            REPO / "data" / "Par_VehicleDetails_Updated.csv"
        ),
        "reference_sha256": base.sha256(REPO / "data" / "Ref_dict.csv"),
        "deadhead_sha256": base.sha256(REPO / "data" / "par_ref_dhd.csv"),
        "tariff": str(base.TARIFF.relative_to(REPO)),
        "tariff_sha256": base.sha256(base.TARIFF),
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
            "selection_manifest.csv": base.sha256(manifest),
            "candidate_universe.csv": base.sha256(
                output_dir / "candidate_universe.csv"
            ),
            "known_duty_continuous_240_240.csv": base.sha256(duty_path),
            **{
                name: base.sha256(output_dir / name)
                for name in copied_exclusions
            },
        },
        "selected_rows": len(rows),
    }
    (output_dir / "input_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} selected instances under {output_dir}")
    print("selection counts: 6 nested probability + 4 structural at k=9--15")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

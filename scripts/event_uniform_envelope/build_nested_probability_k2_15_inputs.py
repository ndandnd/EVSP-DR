#!/usr/bin/env python3
"""Extend the six published k9--k15 probability chains down to k2."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.event_uniform_envelope import build_small_threshold_inputs as base  # noqa: E402


SEED = 20260904
SCALES = tuple(range(2, 16))
SOURCE = REPO / "data/scale_ladder/instances/threshold_9_15_20260904"
DEFAULT_OUTPUT = REPO / "data/scale_ladder/instances/nested_probability_k2_15_20260908"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def recover_orders(available: list[str], excluded: set[tuple[str, ...]]):
    seen = {scale: set() for scale in range(9, 16)}
    orders = []
    for replicate in range(1, 7):
        rng = random.Random(SEED * 1000 + replicate)
        for _attempt in range(100000):
            ordered = rng.sample(available, 15)
            if len({base._base_task(duty) for duty in ordered}) != 15:
                continue
            prefixes = {
                scale: tuple(sorted(ordered[:scale])) for scale in range(9, 16)
            }
            if any(value in excluded or value in seen[scale]
                   for scale, value in prefixes.items()):
                continue
            break
        else:
            raise RuntimeError(f"could not replay chain {replicate}")
        for scale, value in prefixes.items():
            seen[scale].add(value)
        orders.append(ordered)
    return orders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    if not output.is_relative_to(REPO):
        raise SystemExit("output must be inside repository")

    source_plan_path = SOURCE / "input_plan.json"
    source_manifest_path = SOURCE / "selection_manifest.csv"
    source_plan = json.loads(source_plan_path.read_text())
    if (source_plan.get("schema") != "evsp-dr-threshold-9-15-inputs-v1"
            or base.sha256(source_manifest_path)
            != source_plan["files"]["selection_manifest.csv"]):
        raise SystemExit("published source identity mismatch")
    source_rows = read_csv(source_manifest_path)
    upper = {
        (int(row["scale"]), int(row["family_replicate"])): row
        for row in source_rows if row["sample_family"] == "probability"
    }
    if len(upper) != 42:
        raise SystemExit("expected 42 published probability rows")
    certificate_path = SOURCE / "known_duty_continuous_240_240.csv"
    certificates = {row["duty_id"]: row for row in read_csv(certificate_path)}
    exclusion_paths = [
        SOURCE / "excluded_existing_scale_ladder_manifest.csv",
        SOURCE / "excluded_small_threshold_manifest.csv",
    ]
    excluded = set().union(*(base.existing_duty_sets(path) for path in exclusion_paths))
    orders = recover_orders(sorted(certificates), excluded)
    for replicate, ordered in enumerate(orders, 1):
        for scale in range(9, 16):
            expected = upper[(scale, replicate)]
            if tuple(sorted(ordered[:scale])) != tuple(json.loads(expected["duties_json"])):
                raise SystemExit(f"RNG replay differs at k{scale} p{replicate}")

    frames = base.load_duty_frames()
    output.mkdir(parents=True)
    shutil.copyfile(certificate_path, output / certificate_path.name)
    rows = []
    chain_rows = []
    for replicate, ordered in enumerate(orders, 1):
        for rank, duty in enumerate(ordered, 1):
            chain_rows.append({
                "chain_id": f"nested_probability_{replicate}",
                "family_replicate": replicate, "addition_rank": rank,
                "duty_id": duty, "removed_when_descending_from_k": rank,
            })
        for scale in SCALES:
            duties = tuple(sorted(ordered[:scale]))
            filename = f"Practice_Custom_DutyUnion_k{scale:02d}_p{replicate:02d}_20260908.csv"
            destination = output / filename
            if scale >= 9:
                old = upper[(scale, replicate)]
                source_instance = REPO / old["relative_path"]
                if base.sha256(source_instance) != old["instance_file_sha256"]:
                    raise SystemExit(f"published instance hash mismatch: {old['cell_id']}")
                shutil.copyfile(source_instance, destination)
                if base.sha256(destination) != old["instance_file_sha256"]:
                    raise SystemExit(f"upper copy changed bytes: {old['cell_id']}")
            else:
                base.merge_duties(frames, list(duties)).to_csv(
                    destination, index=False, lineterminator="\n"
                )
            features = base.candidate_features(frames, list(duties))
            certificate_set = sorted(
                certificates[duty]["certificate_sha256"] for duty in duties
            )
            rows.append({
                "cell_id": f"k{scale:02d}_p{replicate}", "scale": scale,
                "selection_replicate": replicate, "sample_family": "probability",
                "family_replicate": replicate,
                "selection_role": f"fixed_seed_probability_{replicate}",
                "nested_chain_id": f"nested_probability_{replicate}",
                "nested_parent_cell_id": f"k{scale-1:02d}_p{replicate}" if scale > 2 else "",
                "addition_rank_duty": ordered[scale - 1],
                "relative_path": str(destination.relative_to(REPO)),
                "instance_file_sha256": base.sha256(destination),
                "published_upper_instance_sha256": (
                    upper[(scale, replicate)]["instance_file_sha256"] if scale >= 9 else ""
                ),
                "duties_json": json.dumps(duties, separators=(",", ":")),
                "duty_set_sha256": base.canonical_sha(duties),
                "target_fleet": scale,
                "known_partition_continuous_physical_upper_bound": True,
                "known_duty_certificate_set_sha256": base.canonical_sha(certificate_set),
                **features,
                "direct_compatibility_density": round(
                    base.direct_compatibility_density(destination), 12
                ),
            })
    rows.sort(key=lambda row: (row["scale"], row["family_replicate"]))
    with (output / "selection_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    with (output / "chain_order.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chain_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(chain_rows)
    plan = {
        "schema": "evsp-dr-nested-probability-k2-15-inputs-v1",
        "created_date": "2026-09-08", "generator_seed": SEED,
        "scales": list(SCALES), "probability_per_scale": 6,
        "selected_rows": len(rows),
        "probability_design": "six_original_rng_prefix_chains_k2_through_k15",
        "selection_uses_solver_outcomes": False,
        "removal_policy": "descending k removes the duty with addition_rank k",
        "published_upper_rows": 42,
        "published_upper_bytes_preserved": True,
        "source_input_plan_sha256": base.sha256(source_plan_path),
        "source_selection_manifest_sha256": base.sha256(source_manifest_path),
        "source_generator_sha256": base.sha256(
            REPO / "scripts/event_uniform_envelope/build_threshold_9_15_inputs.py"
        ),
        "generator_script_sha256": base.sha256(Path(__file__)),
        "shared_builder_sha256": base.sha256(Path(base.__file__)),
        "known_partition_scope": source_plan["known_partition_scope"],
        "known_partition_caveat": source_plan["known_partition_caveat"],
        "physics": source_plan["physics"],
        "files": {
            name: base.sha256(output / name) for name in (
                "selection_manifest.csv", "chain_order.csv",
                "known_duty_continuous_240_240.csv",
            )
        },
    }
    (output / "input_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(rows)} nested rows under {output}")
    print("verified byte-identical published upper instances: 42/42")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

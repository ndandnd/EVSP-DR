#!/usr/bin/env python3
"""Build deterministic, identity-domain-explicit scale-ladder inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
from pathlib import Path

from build_tariff_response_manifest import REPO_ROOT, sha256_file
from make_duty_pair_instances import (
    _base_task,
    load_duty_frames,
    merge_duties,
)


SEED = 20260803
SOURCE_FROZEN_MANIFEST = (
    REPO_ROOT
    / "data/tariff_response/frozen_instances/frozen_input_manifest.csv"
)
SOURCE_FROZEN_MANIFEST_SHA256 = (
    "5473e8d83c8e7e1f0b6e872125419466bb5044bbbb014df3184254f6a2b601c6"
)
HISTORICAL_FLAT = REPO_ROOT / "data/hourly_prices_flat.csv"
HISTORICAL_FLAT_SHA256 = (
    "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200"
)
EXTENDED_FLAT = REPO_ROOT / "data/tariff_response/flat_h26.csv"
EXTENDED_FLAT_SHA256 = (
    "59a9d3b7c1516b19b56267de7309cfe014a13f84b8b390738754a5f652f451f3"
)
TRIP_IDENTITY_SCHEMA = "evsp-dr-trip-identity-v1"
OUTPUT_DIR = REPO_ROOT / "data/scale_ladder/instances"
LEGACY_INSTANCE_MANIFEST_SHA256 = (
    "a7ef8b77351440a8d7873b949891663ca7b28f135d366d4c6b003d09ca84839a"
)
EXTENSION_SEED = SEED
EXTENSION_SCALES = (2, 3, 5, 8, 13, 20)
EXTENSION_REPLICATES = (4, 5, 6)
EXTENDED_MANIFEST_NAME = (
    "scale_ladder_instance_manifest_6sel_seed20260803.csv"
)
EXTENDED_CAMPAIGN_NAME = (
    "campaign_input_manifest_6sel_seed20260803.json"
)
EXTENSION_RECORD_NAME = "duty_union_extension_seed20260803.json"
FIELDS = (
    "scale", "selection_replicate", "cg_replicates", "target_fleet",
    "relative_path", "instance_file_sha256", "trip_count",
    "ordered_trip_id_set_sha256", "solver_local_trip_index_sha256",
    "ordered_trip_sequence_sha256", "trip_identity_schema",
    "duty_count", "duties_json", "duty_set_sha256",
    "generator_seed", "generator_family", "weekday_variant_policy",
    "reused_frozen_input",
)


def canonical_sha(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def replicate_selections(duties, sizes, *, per_size=6):
    rng = random.Random(SEED)
    selections = {}
    for scale in sizes:
        seen = set()
        made = attempts = 0
        while made < per_size and attempts < 20000:
            attempts += 1
            sample = rng.sample(duties, scale)
            if len({_base_task(duty) for duty in sample}) != scale:
                continue
            key = frozenset(sample)
            if key in seen:
                continue
            seen.add(key)
            made += 1
            selections[(scale, made)] = sorted(sample)
    return selections


def _curve(path):
    with path.open(newline="") as handle:
        return {
            int(float(row["time_block"])): float(row["cost"])
            for row in csv.DictReader(handle)
        }


def _tariff_equivalence():
    historical = _curve(HISTORICAL_FLAT)
    extended = _curve(EXTENDED_FLAT)
    common_equal = all(
        historical[hour] == extended[hour] for hour in historical
    )
    extension_constant = all(
        extended[hour] == historical[max(historical)]
        for hour in set(extended) - set(historical)
    )
    if (
        sha256_file(HISTORICAL_FLAT) != HISTORICAL_FLAT_SHA256
        or sha256_file(EXTENDED_FLAT) != EXTENDED_FLAT_SHA256
        or not common_equal
        or not extension_constant
        or len(set(historical.values())) != 1
    ):
        raise ValueError("historical flat-tariff equivalence proof failed")
    return {
        "primary_tariff_relative_path": str(
            HISTORICAL_FLAT.relative_to(REPO_ROOT)
        ),
        "primary_tariff_sha256": HISTORICAL_FLAT_SHA256,
        "extended_comparator_relative_path": str(
            EXTENDED_FLAT.relative_to(REPO_ROOT)
        ),
        "extended_comparator_sha256": EXTENDED_FLAT_SHA256,
        "equivalence_scope": "hours_0_through_24_exact",
        "hours_25_26_policy":
            "historical_last_hour_extension_verified_constant",
        "equivalence_verified": True,
    }


def build(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256_file(SOURCE_FROZEN_MANIFEST) != (
        SOURCE_FROZEN_MANIFEST_SHA256
    ):
        raise ValueError("source frozen manifest identity changed")
    with SOURCE_FROZEN_MANIFEST.open(newline="") as handle:
        frozen = {
            (int(row["scale"]), int(row["replicate"])): row
            for row in csv.DictReader(handle)
        }
    frames = load_duty_frames()
    duties = sorted(frames)
    k2 = replicate_selections(duties, (2,), per_size=3)
    small = replicate_selections(duties, (3, 5, 8, 13))
    large = replicate_selections(duties, (15, 20, 30, 40))
    output_dir.mkdir(parents=True)
    rows = []
    for scale in (2, 3, 5, 8, 13, 20, 30):
        family = (
            "pair_union_k2_seed20260803"
            if scale == 2
            else "small_3_5_8_13_per6_seed20260803"
            if scale in {3, 5, 8, 13}
            else "large_15_20_30_40_per6_seed20260803"
        )
        source = k2 if scale == 2 else small if scale <= 13 else large
        for replicate in (1, 2, 3):
            selected = source[(scale, replicate)]
            frame = merge_duties(frames, selected)
            reused = scale in {5, 8} and replicate == 2
            if reused:
                frozen_row = frozen[(scale, 2)]
                path = REPO_ROOT / frozen_row["relative_path"]
                if sha256_file(path) != frozen_row["file_sha256"]:
                    raise ValueError("reused frozen input hash changed")
                regenerated = output_dir / (
                    f".verify_k{scale:02d}_r{replicate}.csv"
                )
                frame.to_csv(regenerated, index=False, lineterminator="\n")
                if sha256_file(regenerated) != frozen_row["file_sha256"]:
                    raise ValueError("frozen input regeneration differs")
                regenerated.unlink()
            else:
                path = output_dir / (
                    f"Practice_Custom_DutyUnion_k{scale:02d}_r{replicate}.csv"
                )
                frame.to_csv(path, index=False, lineterminator="\n")
            rows.append(_row(
                path, frame, selected, scale, replicate, family, reused
            ))
    selected = large[(40, 2)]
    frame = merge_duties(frames, selected)
    frozen_row = frozen[(40, 2)]
    path = REPO_ROOT / frozen_row["relative_path"]
    if sha256_file(path) != frozen_row["file_sha256"]:
        raise ValueError("reused frozen k40 input hash changed")
    regenerated = output_dir / ".verify_k40_r2.csv"
    frame.to_csv(regenerated, index=False, lineterminator="\n")
    if sha256_file(regenerated) != frozen_row["file_sha256"]:
        raise ValueError("frozen k40 regeneration differs")
    regenerated.unlink()
    rows.append(_row(
        path, frame, selected, 40, 2,
        "large_15_20_30_40_per6_seed20260803", True,
        cg_replicates=2,
    ))
    rows.sort(key=lambda row: (
        int(row["scale"]), int(row["selection_replicate"])
    ))
    manifest = output_dir / "scale_ladder_instance_manifest.csv"
    with manifest.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    campaign = {
        "schema": "evsp-dr-scale-ladder-input-manifest-v1",
        "instance_manifest": str(manifest.relative_to(REPO_ROOT)),
        "instance_manifest_sha256": sha256_file(manifest),
        "source_frozen_manifest": str(
            SOURCE_FROZEN_MANIFEST.relative_to(REPO_ROOT)
        ),
        "source_frozen_manifest_sha256":
            SOURCE_FROZEN_MANIFEST_SHA256,
        "trip_identity_schema": TRIP_IDENTITY_SCHEMA,
        "legacy_trip_hash_policy":
            "classify_before_comparison_never_compare_cross_domain",
        "tariff": _tariff_equivalence(),
        "physics": {
            "g_kwh": 300.0,
            "charge_kw": 300.0,
            "reserve_kwh": 0.0,
            "soc_step_kwh": 15.0,
            "block_min": 10,
        },
    }
    campaign_path = output_dir / "campaign_input_manifest.json"
    campaign_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n"
    )
    return manifest, campaign_path, rows


def _row(
    path, frame, selected, scale, replicate, family, reused,
    *, cg_replicates=1,
):
    source_ids = [int(value) for value in frame["Ordered_Trip_ID"]]
    local_ids = list(range(len(frame)))
    return {
        "scale": scale,
        "selection_replicate": replicate,
        "cg_replicates": cg_replicates,
        "target_fleet": scale,
        "relative_path": str(path.relative_to(REPO_ROOT)),
        "instance_file_sha256": sha256_file(path),
        "trip_count": len(frame),
        "ordered_trip_id_set_sha256": canonical_sha(sorted(source_ids)),
        "solver_local_trip_index_sha256": canonical_sha(local_ids),
        "ordered_trip_sequence_sha256": canonical_sha(source_ids),
        "trip_identity_schema": TRIP_IDENTITY_SCHEMA,
        "duty_count": len(selected),
        "duties_json": json.dumps(selected, separators=(",", ":")),
        "duty_set_sha256": canonical_sha(selected),
        "generator_seed": SEED,
        "generator_family": family,
        "weekday_variant_policy":
            "one_literal_per_numeric_base_no_siblings",
        "reused_frozen_input": reused,
    }


def extension_family(scale):
    if scale == 2:
        return "pair_union_k2_seed20260803"
    if scale in {3, 5, 8, 13}:
        return "small_3_5_8_13_per6_seed20260803"
    if scale == 20:
        return "large_15_20_30_40_per6_seed20260803"
    raise ValueError(f"unsupported extension scale: {scale}")


def _write_generated(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"generated file differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def build_six_selection_extension(
    output_dir=OUTPUT_DIR,
    *,
    legacy_manifest=OUTPUT_DIR / "scale_ladder_instance_manifest.csv",
    legacy_campaign=OUTPUT_DIR / "campaign_input_manifest.json",
):
    """Publish a versioned additive 4--6 selection extension."""

    output_dir = Path(output_dir).resolve()
    legacy_manifest = Path(legacy_manifest).resolve()
    legacy_campaign = Path(legacy_campaign).resolve()
    legacy_raw = legacy_manifest.read_bytes()
    if hashlib.sha256(legacy_raw).hexdigest() != (
        LEGACY_INSTANCE_MANIFEST_SHA256
    ):
        raise ValueError("legacy scale-ladder manifest identity changed")
    with legacy_manifest.open(newline="") as handle:
        legacy_rows = list(csv.DictReader(handle))
    if len(legacy_rows) != 22:
        raise ValueError("legacy scale-ladder manifest row count changed")

    frames = load_duty_frames()
    duties = sorted(frames)
    selections = {
        2: replicate_selections(duties, (2,), per_size=6),
        "small": replicate_selections(duties, (3, 5, 8, 13)),
        "large": replicate_selections(duties, (15, 20, 30, 40)),
    }
    new_rows = []
    for scale in EXTENSION_SCALES:
        source = (
            selections[2] if scale == 2
            else selections["small"] if scale <= 13
            else selections["large"]
        )
        for replicate in EXTENSION_REPLICATES:
            selected = source[(scale, replicate)]
            frame = merge_duties(frames, selected)
            path = output_dir / (
                f"Practice_Custom_DutyUnion_k{scale:02d}_r{replicate}.csv"
            )
            _write_generated(
                path,
                frame.to_csv(
                    index=False, lineterminator="\n"
                ).encode(),
            )
            new_rows.append(_row(
                path, frame, selected, scale, replicate,
                extension_family(scale), False,
            ))
    new_rows.sort(key=lambda row: (
        int(row["scale"]), int(row["selection_replicate"])
    ))
    old_keys = {
        (int(row["scale"]), int(row["selection_replicate"]))
        for row in legacy_rows
    }
    new_keys = {
        (int(row["scale"]), int(row["selection_replicate"]))
        for row in new_rows
    }
    if len(new_rows) != 18 or old_keys & new_keys:
        raise ValueError("selection extension cardinality/identity differs")

    suffix = io.StringIO(newline="")
    writer = csv.DictWriter(
        suffix, fieldnames=FIELDS, lineterminator="\n"
    )
    writer.writerows(new_rows)
    manifest_payload = legacy_raw + suffix.getvalue().encode()
    manifest = output_dir / EXTENDED_MANIFEST_NAME
    _write_generated(manifest, manifest_payload)

    extension_record = {
        "schema": "evsp-dr-duty-union-selection-extension-v1",
        "validation_status":
            "producer_only_pending_independent_validation",
        "generator": "src/build_scale_ladder_inputs.py",
        "seed": EXTENSION_SEED,
        "selection_replicates": list(EXTENSION_REPLICATES),
        "weekday_variant_policy":
            "one_literal_per_numeric_base_no_siblings",
        "generator_families": {
            str(scale): extension_family(scale)
            for scale in EXTENSION_SCALES
        },
        "instances": [
            {
                key: row[key] for key in (
                    "scale", "selection_replicate", "relative_path",
                    "instance_file_sha256", "duty_count", "duties_json",
                    "duty_set_sha256", "target_fleet",
                    "generator_seed", "generator_family",
                    "weekday_variant_policy",
                )
            }
            for row in new_rows
        ],
    }
    extension_record_path = output_dir / EXTENSION_RECORD_NAME
    _write_generated(
        extension_record_path,
        (json.dumps(
            extension_record, indent=2, sort_keys=True
        ) + "\n").encode(),
    )

    campaign = json.loads(legacy_campaign.read_text())
    campaign.update({
        "instance_manifest": str(manifest.relative_to(REPO_ROOT)),
        "instance_manifest_sha256":
            hashlib.sha256(manifest_payload).hexdigest(),
        "legacy_instance_manifest": str(
            legacy_manifest.relative_to(REPO_ROOT)
        ),
        "legacy_instance_manifest_sha256":
            LEGACY_INSTANCE_MANIFEST_SHA256,
        "selection_extensions": [{
            "generator": "src/build_scale_ladder_inputs.py",
            "validation_status":
                "producer_only_pending_independent_validation",
            "generator_families":
                extension_record["generator_families"],
            "seed": EXTENSION_SEED,
            "scales": list(EXTENSION_SCALES),
            "selection_replicates": list(EXTENSION_REPLICATES),
            "extension_manifest": str(
                extension_record_path.relative_to(REPO_ROOT)
            ),
            "extension_manifest_sha256": sha256_file(
                extension_record_path
            ),
        }],
    })
    campaign_path = output_dir / EXTENDED_CAMPAIGN_NAME
    _write_generated(
        campaign_path,
        (json.dumps(campaign, indent=2, sort_keys=True) + "\n").encode(),
    )
    return manifest, campaign_path, new_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--extend-six-selections", action="store_true")
    args = parser.parse_args(argv)
    builder = (
        build_six_selection_extension
        if args.extend_six_selections else build
    )
    manifest, campaign, rows = builder(args.out_dir)
    print(json.dumps({
        "instance_manifest": str(manifest),
        "instance_manifest_sha256": sha256_file(manifest),
        "campaign_manifest": str(campaign),
        "campaign_manifest_sha256": sha256_file(campaign),
        "rows": len(rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

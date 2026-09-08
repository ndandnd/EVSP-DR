#!/usr/bin/env python3
"""Build six nested GIRO40 prefixes ordered by decreasing trip count."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.event_uniform_envelope import build_small_threshold_inputs as base  # noqa: E402


SCALES = (2, 3, 5, 8, 10, 15)
DEFAULT_OUTPUT = (
    REPO / "data/scale_ladder/instances/heavy_trip_nested_20260908"
)
GIRO40_MANIFEST = REPO / "data/tariff_response/giro40_duty_manifest.csv"
CERTIFICATE_SOURCE = (
    REPO / "data/scale_ladder/instances/nested_probability_k2_15_20260908"
    / "known_duty_continuous_240_240.csv"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    if not output.is_relative_to(REPO):
        raise SystemExit("output must be inside repository")

    giro = read_rows(GIRO40_MANIFEST)
    if len(giro) != 40 or any(row["included_variant"] != "True" for row in giro):
        raise SystemExit("published GIRO40 manifest identity is invalid")
    if len({row["base_duty_id"] for row in giro}) != 40:
        raise SystemExit("GIRO40 manifest has duplicate base duties")
    ranked = sorted(
        giro, key=lambda row: (-int(row["trip_count"]), row["duty_id"])
    )
    certificates = {
        row["duty_id"]: row for row in read_rows(CERTIFICATE_SOURCE)
    }
    if any(row["duty_id"] not in certificates for row in ranked):
        raise SystemExit("a canonical GIRO40 duty lacks a 240/240 certificate")
    frames = base.load_duty_frames()
    if any(
        len(frames[row["duty_id"]]) != int(row["trip_count"])
        for row in ranked
    ):
        raise SystemExit("master data trip counts differ from GIRO40 manifest")

    output.mkdir(parents=True)
    shutil.copyfile(CERTIFICATE_SOURCE, output / CERTIFICATE_SOURCE.name)
    order_rows = []
    for rank, row in enumerate(ranked, 1):
        order_rows.append({
            "addition_rank": rank,
            "duty_id": row["duty_id"],
            "base_duty_id": row["base_duty_id"],
            "trip_count": row["trip_count"],
            "included_variant": row["included_variant"],
            "excluded_variant_id": row["excluded_variant_id"],
        })
    with (output / "duty_order.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(order_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(order_rows)

    selected = []
    for scale in SCALES:
        duties = [row["duty_id"] for row in ranked[:scale]]
        destination = output / (
            f"Practice_Custom_DutyUnion_heavy_k{scale:02d}_20260908.csv"
        )
        merged = base.merge_duties(frames, duties)
        merged.to_csv(destination, index=False, lineterminator="\n")
        features = base.candidate_features(frames, duties)
        certificate_set = sorted(
            certificates[duty]["certificate_sha256"] for duty in duties
        )
        selected.append({
            "cell_id": f"heavy_k{scale:02d}",
            "scale": scale,
            "selection_replicate": 1,
            "sample_family": "deterministic_trip_descending_prefix",
            "family_replicate": 1,
            "selection_role": "canonical_giro40_most_trip_prefix",
            "nested_chain_id": "canonical_giro40_trip_descending",
            "nested_parent_cell_id": (
                f"heavy_k{SCALES[SCALES.index(scale) - 1]:02d}"
                if scale != SCALES[0] else ""
            ),
            "relative_path": str(destination.relative_to(REPO)),
            "instance_file_sha256": base.sha256(destination),
            "duties_json": json.dumps(duties, separators=(",", ":")),
            "duty_set_sha256": base.canonical_sha(sorted(duties)),
            "target_fleet": scale,
            "known_partition_continuous_physical_upper_bound": True,
            "known_duty_certificate_set_sha256": base.canonical_sha(
                certificate_set
            ),
            **features,
            "direct_compatibility_density": round(
                base.direct_compatibility_density(destination), 12
            ),
        })
    with (output / "selection_manifest.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(selected[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(selected)

    plan = {
        "schema": "evsp-dr-heavy-trip-nested-inputs-v1",
        "created_date": "2026-09-08",
        "scales": list(SCALES),
        "selected_rows": len(selected),
        "selection_rule": (
            "published canonical GIRO40 duties sorted by "
            "(regular trip count descending, literal duty_id descending); prefix k"
        ),
        "selection_uses_solver_outcomes": False,
        "source_giro40_manifest_sha256": base.sha256(GIRO40_MANIFEST),
        "source_master_sha256": base.sha256(
            REPO / "data/Par_VehicleDetails_Updated.csv"
        ),
        "source_certificate_sha256": base.sha256(CERTIFICATE_SOURCE),
        "generator_script_sha256": base.sha256(Path(__file__)),
        "shared_builder_sha256": base.sha256(Path(base.__file__)),
        "known_partition_scope": (
            "the k source-duty sequences are independently feasible under the "
            "continuous 240 kWh / 240 kW fixed-duty model"
        ),
        "known_partition_caveat": (
            "continuous fixed-duty feasibility is not event-grid representability; "
            "RAW column generation does not receive these routes"
        ),
        "physics": {
            "battery_kwh": 240.0,
            "charge_kw": 240.0,
            "reserve_kwh": 0.0,
            "terminal_soc_policy": "free",
            "charge_start_cost": 5.0,
        },
        "files": {
            name: base.sha256(output / name)
            for name in (
                "selection_manifest.csv",
                "duty_order.csv",
                "known_duty_continuous_240_240.csv",
            )
        },
    }
    (output / "input_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(selected)} nested heavy cells under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

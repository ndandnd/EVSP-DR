#!/usr/bin/env python3
"""Publish the exact April 2026 Selected-10 input with current metadata."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.event_uniform_envelope import build_small_threshold_inputs as base  # noqa: E402


SOURCE_COMMIT = "7c564da"
SOURCE_PATH = "data/Practice_Selected_10bus.csv"
EXPECTED_SHA = "75673708acf2b2099b2240829e2603920bc5608d34fef2f7d28cd7d2931868c7"
DEFAULT_OUTPUT = (
    REPO / "data/scale_ladder/instances/legacy_selected10_20260426"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    if not output.is_relative_to(REPO):
        raise SystemExit("output must be inside repository")
    payload = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{SOURCE_COMMIT}:{SOURCE_PATH}"],
        check=True,
        capture_output=True,
    ).stdout
    output.mkdir(parents=True)
    instance = output / "Practice_Selected_10bus_20260426.csv"
    instance.write_bytes(payload)
    if base.sha256(instance) != EXPECTED_SHA:
        raise SystemExit("historical input bytes changed")

    frames = base.load_duty_frames()
    duties = [
        "13307", "13311", "13310", "13309", "13314",
        "13316uwt", "13320", "13321", "13323", "13324muw",
    ]
    features = base.candidate_features(frames, duties)
    if features["trip_count"] != 175 or features["peak_concurrency_lb"] != 10:
        raise SystemExit("historical descriptor replay changed")
    row = {
        "cell_id": "legacy_selected10_240_raw",
        "scale": 10,
        "selection_replicate": 1,
        "sample_family": "historical_exact_replay",
        "family_replicate": 1,
        "selection_role": "april26_selected10",
        "nested_chain_id": "",
        "nested_parent_cell_id": "",
        "relative_path": str(instance.relative_to(REPO)),
        "instance_file_sha256": EXPECTED_SHA,
        "duties_json": json.dumps(duties, separators=(",", ":")),
        "duty_set_sha256": base.canonical_sha(sorted(duties)),
        "target_fleet": 10,
        "known_partition_continuous_physical_upper_bound": True,
        **features,
        "direct_compatibility_density": round(
            base.direct_compatibility_density(instance), 12
        ),
    }
    manifest = output / "selection_manifest.csv"
    with manifest.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(row), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(row)
    plan = {
        "schema": "evsp-dr-legacy-selected10-input-v1",
        "created_date": "2026-09-08",
        "selected_rows": 1,
        "source_commit": SOURCE_COMMIT,
        "source_path": SOURCE_PATH,
        "source_instance_sha256": EXPECTED_SHA,
        "selection_uses_solver_outcomes": False,
        "historical_result_scope": (
            "April 26 one-hour MIP incumbent 12, not proven optimal"
        ),
        "current_replay_physics": {
            "battery_kwh": 240.0,
            "charge_kw": 240.0,
            "reserve_kwh": 0.0,
            "soc_step": 2.5,
            "block_min": 5,
        },
        "files": {
            instance.name: base.sha256(instance),
            manifest.name: base.sha256(manifest),
        },
    }
    (output / "input_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    print(f"published {instance} sha256={EXPECTED_SHA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

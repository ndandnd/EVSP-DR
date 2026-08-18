#!/usr/bin/env python3
"""Publish the verified, nonduplicate GIRO40 duty manifest."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from build_tariff_response_manifest import REPO_ROOT, sha256_file
from tariff_response_core import reconstruct_giro40_original


FIELDS = (
    "duty_id", "base_duty_id", "included_variant",
    "excluded_variant_id", "trip_count", "local_trip_ids_json",
    "source_ordered_trip_ids_json", "route_incidence_sha256",
    "recorded_charge_count", "recorded_charge_kwh",
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "data/tariff_response/giro40_duty_manifest.csv"
)


def build(output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    payload = reconstruct_giro40_original(
        REPO_ROOT / "data/Par_VehicleDetails_Updated.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(payload["duties"])
        handle.flush()
        os.fsync(handle.fileno())
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    path = build(args.out)
    print(f"{path} sha256={sha256_file(path)} rows=40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

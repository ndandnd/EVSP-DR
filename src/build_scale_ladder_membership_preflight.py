#!/usr/bin/env python3
"""Freeze pre-launch known-route membership classifications for the ladder."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from audit_scale_ladder_known_membership import (
    FIELDS,
    SCHEMA,
    audit,
)
from build_tariff_response_manifest import REPO_ROOT, sha256_file


INSTANCE_MANIFEST = (
    REPO_ROOT
    / "data/scale_ladder/instances/scale_ladder_instance_manifest.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "data/scale_ladder/known_membership_preflight.json"
)


def build(output_path=DEFAULT_OUTPUT, *, instance_manifest=INSTANCE_MANIFEST):
    output_path = Path(output_path).resolve()
    instance_manifest = Path(instance_manifest).resolve()
    csv_path = output_path.with_suffix(".csv")
    if output_path.exists() or csv_path.exists():
        raise FileExistsError(output_path)
    with instance_manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    cells = []
    duty_rows = []
    for row in rows:
        payload = audit(
            REPO_ROOT / row["relative_path"],
            row["instance_file_sha256"],
            int(row["scale"]),
            int(row["selection_replicate"]),
        )
        cells.append(payload)
        duty_rows.extend(payload["duties"])
    package = {
        "schema": "evsp-dr-scale-ladder-membership-preflight-v1",
        "membership_schema": SCHEMA,
        "instance_manifest_sha256": sha256_file(instance_manifest),
        "cells": cells,
    }
    temporary = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )
    with temporary.open("x") as handle:
        json.dump(package, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(duty_rows)
        handle.flush()
        os.fsync(handle.fileno())
    return output_path, csv_path, package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--instance-manifest", type=Path, default=INSTANCE_MANIFEST,
    )
    args = parser.parse_args(argv)
    path, csv_path, payload = build(
        args.out, instance_manifest=args.instance_manifest,
    )
    print(json.dumps({
        "json": str(path), "json_sha256": sha256_file(path),
        "csv": str(csv_path), "csv_sha256": sha256_file(csv_path),
        "cells": len(payload["cells"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

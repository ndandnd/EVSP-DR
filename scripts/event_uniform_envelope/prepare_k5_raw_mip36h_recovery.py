#!/usr/bin/env python3
"""Stage a retry of the four k5 MIPs after the reviewed path-policy failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_CELLS = {"k05_p5", "k05_xenergy", "k05_xgap", "k05_xtrip"}
EXPECTED_FAILURE = "[MIP] final replay instance escapes data/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty {label}: {path}")
    return path.resolve()


def validate_failed_run(root: Path) -> tuple[list[dict], list[dict]]:
    manifest_path = required_file(
        root / "snapshot_manifest.csv", "snapshot manifest"
    )
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4 or {row["cell"] for row in rows} != EXPECTED_CELLS:
        raise SystemExit("failed run does not contain the four expected k5 cells")

    provenance = []
    for row in rows:
        snapshot = required_file(Path(row["snapshot"]), "frozen snapshot")
        journal = required_file(Path(row["journal"]), "frozen journal")
        if sha256(snapshot) != row["snapshot_sha256"]:
            raise SystemExit(f"snapshot hash mismatch for {row['cell']}")
        if sha256(journal) != row["journal_sha256"]:
            raise SystemExit(f"journal hash mismatch for {row['cell']}")
        result = (
            root / "mip"
            / f"M__{row['cell']}__{row['representation_id']}.raw_pool_mip36h.json"
        )
        if os.path.lexists(result):
            raise SystemExit(f"final MIP result already exists for {row['cell']}")
        diagnostic = required_file(
            result.with_name(f"{result.name}.rejected_physical_replay.json"),
            "rejected-replay diagnostic",
        )
        payload = json.loads(diagnostic.read_text())
        failure = str((payload.get("failure") or {}).get("reason") or "")
        physical = payload.get("physical_pool_audit") or {}
        input_hashes = physical.get("input_hashes") or {}
        if (
            payload.get("schema")
            != "evsp-dr-mip-rejected-physical-replay-v1"
            or payload.get("physical_replay_validated") is not False
            or failure != EXPECTED_FAILURE
            or payload.get("source_result_sha256") != row["snapshot_sha256"]
            or payload.get("source_journal_sha256") != row["journal_sha256"]
            or input_hashes.get("instance_sha256") != row["instance_sha256"]
            or not isinstance(payload.get("solver_incumbent"), dict)
        ):
            raise SystemExit(
                f"diagnostic is not the reviewed path-policy failure for {row['cell']}"
            )
        provenance.append({
            "cell": row["cell"],
            "diagnostic": str(diagnostic),
            "diagnostic_sha256": sha256(diagnostic),
            "failure": failure,
            "prior_solver_incumbent": payload["solver_incumbent"],
        })
    return rows, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    failed = args.failed_root.resolve(strict=True)
    output = args.output_root.resolve()
    rows, provenance = validate_failed_run(failed)
    if os.path.lexists(output):
        raise SystemExit(f"recovery output root already exists: {output}")
    output.mkdir(parents=True)
    (output / "mip").mkdir()
    (output / "logs").mkdir()
    for name in (
        "snapshot_manifest.csv", "snapshot_manifest.tsv",
        "snapshot_manifest.sha256",
    ):
        shutil.copy2(required_file(failed / name, name), output / name)
    record = {
        "schema": "evsp-dr-k5-raw-mip36h-path-policy-recovery-v1",
        "failed_root": str(failed),
        "failed_snapshot_manifest_sha256": sha256(
            failed / "snapshot_manifest.csv"
        ),
        "cells": provenance,
    }
    with (output / "recovery_provenance.json").open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"staged {len(rows)} hash-identical k5 recovery inputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unambiguous trip identity domains for scale-ladder artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


SCHEMA = "evsp-dr-trip-identity-v1"
FIELDS = (
    "ordered_trip_id_set_sha256",
    "solver_local_trip_index_sha256",
    "ordered_trip_sequence_sha256",
    "instance_file_sha256",
)


def canonical_sha(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = Path(path).resolve()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "Ordered_Trip_ID" not in rows[0]:
        raise ValueError("instance lacks Ordered_Trip_ID")
    ordered = [int(float(row["Ordered_Trip_ID"])) for row in rows]
    if len(ordered) != len(set(ordered)):
        raise ValueError("instance repeats Ordered_Trip_ID")
    local = list(range(len(rows)))
    return {
        "trip_identity_schema": SCHEMA,
        "trip_count": len(rows),
        "ordered_trip_id_set_sha256": canonical_sha(sorted(ordered)),
        "solver_local_trip_index_sha256": canonical_sha(local),
        "ordered_trip_sequence_sha256": canonical_sha(ordered),
        "instance_file_sha256": file_sha(path),
    }


def classify_legacy_trip_hash(value: str | None, identities: dict) -> dict:
    if not value:
        return {
            "legacy_trip_hash": value,
            "legacy_trip_hash_schema": "missing",
            "legacy_trip_hash_field": None,
        }
    matches = [
        field for field in FIELDS[:3]
        if identities.get(field) == value
    ]
    if len(matches) == 1:
        return {
            "legacy_trip_hash": value,
            "legacy_trip_hash_schema": SCHEMA,
            "legacy_trip_hash_field": matches[0],
        }
    if not matches:
        return {
            "legacy_trip_hash": value,
            "legacy_trip_hash_schema": "unknown",
            "legacy_trip_hash_field": None,
        }
    raise ValueError("legacy trip hash collides across identity domains")


def require_compatible(left: dict, right: dict, field: str) -> None:
    if field not in FIELDS:
        raise ValueError(f"unknown identity domain: {field}")
    if (
        left.get("trip_identity_schema") != SCHEMA
        or right.get("trip_identity_schema") != SCHEMA
        or left.get(field) != right.get(field)
    ):
        raise ValueError(f"trip identity mismatch in domain {field}")

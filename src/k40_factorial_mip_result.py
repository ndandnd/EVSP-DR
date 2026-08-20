"""Shared schema validation and atomic publication for k40 MIP results."""

from __future__ import annotations

import collections
import ctypes
import errno
import math
import os
from pathlib import Path


def validate_scientific_result(
    result: dict,
    spec: dict,
    source_status: dict,
) -> None:
    if not isinstance(result, dict):
        raise ValueError("MIP result is not an object")
    if result.get("partitioning") is not True:
        raise ValueError("MIP result is not strict partitioning")
    if result.get("route_space_scope") != "finite_augmented_snapshot_pool_only":
        raise ValueError("MIP result has incorrect route-space scope")
    if not isinstance(result.get("status_name"), str) \
            or not result["status_name"]:
        raise ValueError("MIP result has no solver status name")
    if result.get("optimal_scope") not in {
        "none", "fleet_only", "full_pool_lexicographic"
    }:
        raise ValueError("MIP result has invalid optimality scope")
    if (
        result.get("source_result_sha256")
        != spec["staged_result_sha256"]
        or result.get("source_journal_sha256")
        != spec["staged_journal_sha256"]
    ):
        raise ValueError("MIP result source hashes mismatch")
    expected_cell = {
        key: spec[key] for key in (
            "label", "replicate", "treatment", "snapshot_mark_minutes",
            "time_limit_s", "threads", "mip_gap",
        )
    }
    if result.get("campaign_cell") != expected_cell:
        raise ValueError("MIP result cell identity mismatch")
    start = result.get("mip_start")
    acceptance = (start or {}).get("solver_acceptance")
    columns = (start or {}).get("actual_start_columns")
    if (
        not isinstance(start, dict)
        or start.get("kind") != "validated_exact_partition"
        or start.get("source_sha256") != spec["staged_start_sha256"]
        or not isinstance(acceptance, dict)
        or acceptance.get("accepted") is not True
        or not isinstance(columns, list)
        or len(columns) != 40
        or any(
            not isinstance(column, dict)
            or not isinstance(column.get("index"), int)
            or len(str(column.get("sha256") or "")) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(column.get("sha256") or "")
            )
            for column in columns
        )
        or len({column["index"] for column in columns}) != 40
    ):
        raise ValueError("MIP result start evidence is incomplete")
    selected = result.get("selected_routes")
    if not isinstance(selected, list) or not selected:
        raise ValueError("MIP result selected routes are missing")
    counts = collections.Counter(
        trip for route in selected for trip in route.get("trips", [])
    )
    trip_ids = source_status.get("trip_ids")
    if (
        not isinstance(trip_ids, list)
        or set(counts) != set(trip_ids)
        or any(counts[trip] != 1 for trip in trip_ids)
        or any(
            not isinstance(route.get("charging_stops"), dict)
            for route in selected
        )
        or result.get("buses") != len(selected)
    ):
        raise ValueError("MIP result is not an exact scheduled partition")
    if not isinstance(result.get("fleet_proven"), bool):
        raise ValueError("MIP result fleet proof flag is invalid")
    for key in ("mip_obj", "runtime_s"):
        value = result.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"MIP result {key} is invalid")
    for key in ("fleet_bound", "mip_bound", "mip_gap"):
        value = result.get(key)
        if value is not None and (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"MIP result {key} is invalid")
    if not isinstance(result.get("two_stage"), dict):
        raise ValueError("MIP result lacks two-stage details")


def rename_directory_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication unavailable")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source),
        -100, os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(f"refusing to overwrite bundle: {destination}")
    raise OSError(error, os.strerror(error), str(destination))

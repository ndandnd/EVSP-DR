#!/usr/bin/env python3
"""Build deterministic, provenance-bound convergence evidence for coauthors."""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

from durable_io import read_jsonl_records
from run_exact_pool_mip import resolve_pool_journal


FACTORIAL_COMMIT = "eb85ca0cc439956939ba6bf9c42958808d89aadd"
HISTORICAL_COMMIT = "f43475b732c3fbc8447a30845834a7d9e8822ef3"
INSTANCE_SHA256 = "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd"
TARIFF_SHA256 = "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200"
HISTORICAL_ROUTE_WEIGHT = 39.252026205592166
EXPECTED_K40_TRIPS = 947
CHECKPOINTS = {
    "h1": 60,
    "h3": 180,
    "h6": 360,
    "h12": 720,
    "h22": 1320,
    "h24": 1440,
}
ARMS = {
    "CA": ("cover", "artificial"),
    "CS": ("cover", "singletons"),
    "PA": ("partition", "artificial"),
    "PS": ("partition", "singletons"),
}
TRAJECTORY_FIELDS = (
    "replicate", "campaign", "arm", "master_sense", "initial_pool",
    "checkpoint", "nominal_hours", "actual_wall_s", "actual_hours",
    "route_weight", "artificials", "objective", "min_reduced_cost",
    "target_reached_in_lp", "zero_artificials", "pricing_certified",
    "exact_integer_partition_found", "finite_pool_fleet_proven",
    "lp_feasible", "scientific_label", "status_sha256",
    "journal_sha256", "instance_sha256", "tariff_sha256",
    "source_commit",
)
SMALL_FIELDS = (
    "scale_family", "scale", "availability", "artifact_count",
    "target_reached_in_lp", "zero_artificials", "pricing_certified",
    "exact_integer_partition_found", "finite_pool_fleet_proven",
    "source_artifacts", "notes",
)
HISTORICAL_FIELDS = (
    "replicate", "arm", "checkpoint", "actual_hours",
    "route_weight", "artificials", "lp_feasible",
    "historical_route_weight", "delta_from_historical",
    "historical_actual_hours", "historical_status_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(
            candidate for candidate in path.rglob("*")
            if candidate.is_file() and not candidate.is_symlink()):
        relative = str(member.relative_to(path))
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(member).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication unavailable")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(f"output exists: {destination}")
    raise OSError(error, os.strerror(error), str(destination))


def _json_object(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _finite(status: dict, key: str):
    final_lp = status.get("final_lp")
    final = status.get("final")
    final_lp = final_lp if isinstance(final_lp, dict) else {}
    final = final if isinstance(final, dict) else {}
    if key == "route_weight":
        value = final_lp.get("route_weight", final.get("route_weight"))
    elif key == "artificials":
        value = final_lp.get(
            "artificial_total", final.get("artificials")
        )
    elif key == "objective":
        value = final_lp.get("objective", final.get("lp_obj"))
    elif key == "min_rc":
        value = final.get("min_rc")
    else:
        raise KeyError(key)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {key}")
    return number


def _find_data(status_path: Path, relative: str) -> Path | None:
    relative_path = Path(str(relative))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None
    for parent in status_path.parents:
        candidate = parent / "data" / relative_path
        if candidate.is_file():
            return candidate.resolve()
    return None


def _validate_status(
    path: Path,
    *,
    expected_commit: str,
    arm: str | None,
    mark_minutes: int | None,
    require_data=True,
) -> dict:
    status, status_sha = _json_object(path)
    provenance = status.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"missing provenance: {path}")
    if provenance.get("git_commit") != expected_commit:
        raise ValueError(f"source commit mismatch: {path}")
    if provenance.get("instance_sha256") != INSTANCE_SHA256:
        raise ValueError(f"instance hash mismatch: {path}")
    if provenance.get("prices_sha256") != TARIFF_SHA256:
        raise ValueError(f"tariff hash mismatch: {path}")
    for key, expected in (
        ("soc_step", 15.0), ("block_min", 10.0),
        ("g_kwh", 300.0), ("charge_kw", 300.0),
        ("min_soc_frac", 0.0),
    ):
        if not math.isclose(
            float(status.get(key)), expected, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"physics mismatch {key}: {path}")
    if arm is not None:
        sense, initial = ARMS[arm]
        if (
            status.get("master_sense") != sense
            or status.get("initial_pool") != initial
        ):
            raise ValueError(f"treatment mismatch for {arm}: {path}")
        args = provenance.get("args")
        if not isinstance(args, dict) or (
            args.get("master_sense") != sense
            or args.get("initial_pool") != initial
            or int(args.get("columns_per_iter", -1)) != 30
            or not math.isclose(
                float(args.get("rc_eps", math.nan)),
                1e-4, rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise ValueError(f"provenance treatment/CG controls mismatch: {path}")
    elif expected_commit == HISTORICAL_COMMIT:
        if status.get("master_sense") not in (None, "cover") or (
            status.get("initial_pool") not in (None, "artificial")
        ):
            raise ValueError("historical comparator treatment is contradictory")
    trip_ids = status.get("trip_ids")
    if (
        not isinstance(trip_ids, list)
        or not trip_ids
        or len(trip_ids) != len(set(trip_ids))
    ):
        raise ValueError(f"invalid trip IDs: {path}")
    if trip_ids != list(range(EXPECTED_K40_TRIPS)):
        raise ValueError(f"unexpected k40 trip identity: {path}")
    wall_s = float(status.get("wall_s"))
    if not math.isfinite(wall_s) or wall_s < 0:
        raise ValueError(f"invalid actual wall_s: {path}")
    if mark_minutes is not None:
        if (
            float(status.get("snapshot_mark_minutes")) != mark_minutes
            or status.get("stop_reason") != f"snapshot_m{mark_minutes}"
            or wall_s + 1e-6 < mark_minutes * 60
        ):
            raise ValueError(f"checkpoint identity mismatch: {path}")
    journal = resolve_pool_journal(path, status).resolve()
    expected_journal = Path(str(path) + ".columns.jsonl").resolve()
    if not journal.is_file() or (
        path.name.endswith(".snapshot.json")
        and journal != expected_journal
    ):
        raise ValueError(f"snapshot/journal pairing mismatch: {path}")
    records = read_jsonl_records(
        journal, repair_trailing=False, collect=True
    )
    effective = {}
    known = set(trip_ids)
    for record in records:
        trips = record.get("trips")
        try:
            cost = float(record["cost"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid journal cost: {journal}") from exc
        if (
            not isinstance(trips, list)
            or not trips
            or any(
                not isinstance(trip, int) or isinstance(trip, bool)
                for trip in trips
            )
            or len(trips) != len(set(trips))
            or any(trip not in known for trip in trips)
            or not math.isfinite(cost)
        ):
            raise ValueError(f"invalid journal route: {journal}")
        key = frozenset(trips)
        if key not in effective or cost < effective[key]:
            effective[key] = cost
    if len(effective) != int(status.get("columns", -1)):
        raise ValueError(f"journal/status column count mismatch: {path}")
    if require_data:
        instance = _find_data(path, status["csv"])
        tariff = _find_data(path, status["prices_csv"])
        if instance is None or sha256_file(instance) != INSTANCE_SHA256:
            raise ValueError(f"instance bytes unavailable/mismatched: {path}")
        if tariff is None or sha256_file(tariff) != TARIFF_SHA256:
            raise ValueError(f"tariff bytes unavailable/mismatched: {path}")
    route_weight = _finite(status, "route_weight")
    artificials = _finite(status, "artificials")
    objective = _finite(status, "objective")
    min_rc = _finite(status, "min_rc")
    if artificials is None or artificials < 0:
        raise ValueError(f"invalid artificial total: {path}")
    final_lp = status.get("final_lp")
    if isinstance(final_lp, dict):
        positive = final_lp.get("positive_routes")
        if not isinstance(positive, list):
            raise ValueError(f"invalid final LP routes: {path}")
        reconstructed_weight = 0.0
        reconstructed_objective = 0.0
        for route in positive:
            key = frozenset(route.get("trips") or [])
            if key not in effective:
                raise ValueError(f"final LP route absent from journal: {path}")
            value = float(route.get("value"))
            cost = float(route.get("cost"))
            if (
                not math.isfinite(value)
                or value <= 0
                or not any(
                    math.isclose(
                        cost, recorded["cost"],
                        rel_tol=1e-10, abs_tol=1e-6,
                    )
                    for recorded in records
                    if frozenset(recorded.get("trips") or []) == key
                )
            ):
                raise ValueError(f"invalid final LP route metric: {path}")
            reconstructed_weight += value
            reconstructed_objective += value * cost
        if not math.isclose(
                reconstructed_weight, route_weight,
                rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(f"final LP route weight mismatch: {path}")
        if not math.isclose(
                reconstructed_objective + 500000.0 * artificials,
                objective, rel_tol=1e-9, abs_tol=1e-3):
            raise ValueError(f"final LP objective mismatch: {path}")
    certified = status.get("certified_rc_optimal") is True
    if certified and (
        min_rc is None or min_rc < -1e-4
        or (
            mark_minutes is None
            and status.get("stop_reason") != "certified"
        )
    ):
        raise ValueError(f"false pricing certification: {path}")
    lp_feasible = artificials == 0.0
    scale = 40
    return {
        "status": status,
        "status_path": str(path.resolve()),
        "status_sha256": status_sha,
        "journal_path": str(journal),
        "journal_sha256": sha256_file(journal),
        "actual_wall_s": wall_s,
        "route_weight": route_weight,
        "artificials": artificials,
        "objective": objective,
        "min_reduced_cost": min_rc,
        "lp_feasible": lp_feasible,
        "target_reached_in_lp": (
            lp_feasible
            and route_weight is not None
            and route_weight <= scale + 1e-9
        ),
        "zero_artificials": lp_feasible,
        "pricing_certified": certified,
        "exact_integer_partition_found": False,
        "finite_pool_fleet_proven": False,
        "source_commit": provenance["git_commit"],
        "instance_sha256": provenance["instance_sha256"],
        "tariff_sha256": provenance["prices_sha256"],
        "trip_ids": trip_ids,
    }


def _find_arm_path(
    campaign: Path,
    arm: str,
    suffix: str,
) -> Path | None:
    matches = [
        campaign / f"{prefix}_flat_{arm}{suffix}"
        for prefix in ("k40r1", "k40r2")
        if (campaign / f"{prefix}_flat_{arm}{suffix}").is_file()
    ]
    if len(matches) > 1:
        raise ValueError(f"ambiguous k40r1/k40r2 artifact for {arm}{suffix}")
    return matches[0] if matches else None


def _validate_factorial_campaign(campaign: Path) -> str:
    launch_path = campaign / "launch.tsv"
    prep_path = campaign / "prep_attestation.tsv"
    if not launch_path.is_file() or not prep_path.is_file():
        raise ValueError(f"factorial launch/prep manifest missing: {campaign}")
    with launch_path.open(newline="") as handle:
        launch = list(csv.DictReader(handle, delimiter="\t"))
    if len(launch) != 5:
        raise ValueError(f"factorial launch row count mismatch: {campaign}")
    prep_rows = [row for row in launch if row.get("role") == "prep"]
    arm_rows = [row for row in launch if row.get("role") == "arm"]
    if (
        len(prep_rows) != 1
        or prep_rows[0].get("job_name") != "K40-PREP"
    ):
        raise ValueError(f"factorial prep launch mismatch: {campaign}")
    for arm, (sense, initial) in ARMS.items():
        matches = [
            row for row in arm_rows
            if row.get("job_name") == f"K40-{arm}24"
            and row.get("master_sense") == sense
            and row.get("initial_pool") == initial
        ]
        if len(matches) != 1:
            raise ValueError(f"factorial launch treatment mismatch: {arm}")
    attestation = {}
    with prep_path.open(newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 2 or row[0] in attestation:
                raise ValueError("factorial prep attestation malformed")
            attestation[row[0]] = row[1]
    if (
        attestation.get("git_commit") != FACTORIAL_COMMIT
        or attestation.get("instance_sha256") != INSTANCE_SHA256
        or attestation.get("prices_sha256") != TARIFF_SHA256
    ):
        raise ValueError("factorial prep attestation mismatch")
    prefixes = {
        path.name.split("_flat_", 1)[0]
        for path in campaign.glob("k40r*_flat_*.json")
    }
    if len(prefixes) != 1 or prefixes.pop() not in {"k40r1", "k40r2"}:
        raise ValueError("factorial campaign mixes/omits k40r1/k40r2 stems")
    return next(iter({
        path.name.split("_flat_", 1)[0]
        for path in campaign.glob("k40r*_flat_*.json")
    }))


def _factorial_rows(
    campaigns: list[Path],
    *,
    allow_missing: bool,
) -> tuple[list[dict], list[dict]]:
    rows, missing = [], []
    resolved_campaigns = [
        campaign.expanduser().resolve() for campaign in campaigns
    ]
    if len(set(resolved_campaigns)) != len(resolved_campaigns):
        raise ValueError("factorial replicate campaign paths must be distinct")
    expected_trip_set = None
    for replicate_index, raw_campaign in enumerate(campaigns, start=1):
        campaign = raw_campaign.expanduser().resolve()
        replicate = f"R{replicate_index}"
        if not campaign.is_dir():
            missing.append({
                "input": str(campaign),
                "reason": "factorial_campaign_missing",
            })
            continue
        expected_prefix = _validate_factorial_campaign(campaign)
        for arm, (sense, initial) in ARMS.items():
            previous_wall = -math.inf
            checkpoints = [
                (label, mark, f".m{mark}.snapshot.json")
                for label, mark in CHECKPOINTS.items()
            ] + [("terminal", None, ".json")]
            for label, mark, suffix in checkpoints:
                path = _find_arm_path(campaign, arm, suffix)
                if path is None:
                    missing.append({
                        "input": str(campaign),
                        "arm": arm,
                        "checkpoint": label,
                        "reason": "checkpoint_missing",
                    })
                    continue
                if not path.name.startswith(expected_prefix + "_flat_"):
                    raise ValueError("factorial replicate stem changed")
                record = _validate_status(
                    path,
                    expected_commit=FACTORIAL_COMMIT,
                    arm=arm,
                    mark_minutes=mark,
                )
                if record["actual_wall_s"] < previous_wall:
                    raise ValueError(
                        f"factorial checkpoint chronology regressed: {path}"
                    )
                previous_wall = record["actual_wall_s"]
                if expected_trip_set is None:
                    expected_trip_set = record["trip_ids"]
                elif record["trip_ids"] != expected_trip_set:
                    raise ValueError("mixed factorial trip sets")
                rows.append({
                    "replicate": replicate,
                    "campaign": campaign.name,
                    "arm": arm,
                    "master_sense": sense,
                    "initial_pool": initial,
                    "checkpoint": label,
                    "nominal_hours": (
                        mark / 60.0 if mark is not None else None
                    ),
                    "actual_wall_s": record["actual_wall_s"],
                    "actual_hours": record["actual_wall_s"] / 3600.0,
                    "route_weight": record["route_weight"],
                    "artificials": record["artificials"],
                    "objective": record["objective"],
                    "min_reduced_cost": record["min_reduced_cost"],
                    "target_reached_in_lp": record[
                        "target_reached_in_lp"
                    ],
                    "zero_artificials": record["zero_artificials"],
                    "pricing_certified": record["pricing_certified"],
                    "exact_integer_partition_found": False,
                    "finite_pool_fleet_proven": False,
                    "lp_feasible": record["lp_feasible"],
                    "scientific_label": (
                        "covering_restricted_master_lp_not_integer_schedule"
                        if sense == "cover"
                        else (
                            "partition_lp_infeasible_artificials_remain"
                            if not record["lp_feasible"]
                            else "partition_restricted_master_lp"
                        )
                    ),
                    "status_sha256": record["status_sha256"],
                    "journal_sha256": record["journal_sha256"],
                    "instance_sha256": record["instance_sha256"],
                    "tariff_sha256": record["tariff_sha256"],
                    "source_commit": record["source_commit"],
                    "_trip_set_sha256": hashlib.sha256(json.dumps(
                        record["trip_ids"], separators=(",", ":")
                    ).encode()).hexdigest(),
                })
    if missing and not allow_missing:
        raise ValueError(
            "required factorial checkpoints are missing: "
            + json.dumps(missing[:10])
        )
    return rows, missing


def _historical(path: Path, *, allow_missing: bool) -> tuple[dict | None, list]:
    source = path.expanduser().resolve()
    if not source.is_file():
        if allow_missing:
            return None, [{
                "input": str(source),
                "reason": "historical_comparator_missing",
            }]
        raise ValueError(f"historical comparator missing: {source}")
    record = _validate_status(
        source,
        expected_commit=HISTORICAL_COMMIT,
        arm=None,
        mark_minutes=None,
    )
    if not math.isclose(
        record["route_weight"], HISTORICAL_ROUTE_WEIGHT,
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise ValueError("historical endpoint route weight mismatch")
    if not record["lp_feasible"]:
        raise ValueError("historical comparator retains artificials")
    return record, []


def _artifact_scale(path: Path, payload: dict) -> tuple[str, int | str] | None:
    text = f"{path.name} {payload.get('instance', '')} {payload.get('csv', '')}"
    pair = re.search(r"(?:pair|DutyPair)", text, re.IGNORECASE)
    if pair:
        return "pair", "pair"
    match = re.search(r"(?:^|[_-])k(3|5|8|13|15|20|30|40)", text)
    if match:
        return "union", int(match.group(1))
    match = re.search(r"Practice_(\d+)bus", text)
    if match:
        return "single", int(match.group(1))
    return None


def _validated_scale_flags(path: Path, payload: dict) -> dict | None:
    if not path.is_file():
        return None
    if "partitioning" in payload:
        if payload.get("partitioning") is not True:
            return None
        provenance = payload.get("mip_provenance")
        if not isinstance(provenance, dict):
            return None
        commit = provenance.get("git_commit")
        if not isinstance(commit, str) or not re.fullmatch(
                r"[0-9a-f]{40}", commit):
            return None
        source_result = Path(str(payload.get("source_result") or ""))
        source_journal = Path(str(payload.get("source_journal") or ""))
        if not source_result.is_file() or not source_journal.is_file():
            return None
        if (
            sha256_file(source_result)
            != payload.get("source_result_sha256")
            or sha256_file(source_journal)
            != payload.get("source_journal_sha256")
        ):
            return None
        source_status = json.loads(source_result.read_text())
        trip_ids = source_status.get("trip_ids")
        if not isinstance(trip_ids, list) or not trip_ids:
            return None
        counts = defaultdict(int)
        for route in payload.get("selected_routes") or []:
            for trip in route.get("trips") or []:
                counts[trip] += 1
        integer = (
            payload.get("incumbent_found") is True
            and set(counts) == set(trip_ids)
            and all(counts[trip] == 1 for trip in trip_ids)
        )
        proof = (
            integer
            and payload.get("fleet_proven") is True
            and payload.get("optimal_scope") in {
                "fleet_only", "full_pool_lexicographic"
            }
        )
        return {
            "target_reached_in_lp": None,
            "zero_artificials": None,
            "pricing_certified": None,
            "exact_integer_partition_found": integer,
            "finite_pool_fleet_proven": proof,
        }
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return None
    if (
        not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("git_commit") or ""))
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(provenance.get("instance_sha256") or "")
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(provenance.get("prices_sha256") or "")
        )
    ):
        return None
    trip_ids = payload.get("trip_ids")
    if (
        not isinstance(trip_ids, list)
        or not trip_ids
        or any(
            not isinstance(trip, int) or isinstance(trip, bool)
            for trip in trip_ids
        )
        or len(trip_ids) != len(set(trip_ids))
    ):
        return None
    for key in (
        "soc_step", "block_min", "g_kwh", "charge_kw", "min_soc_frac",
    ):
        try:
            if not math.isfinite(float(payload[key])):
                return None
        except (KeyError, TypeError, ValueError):
            return None
    try:
        journal = resolve_pool_journal(path, payload).resolve()
    except SystemExit:
        return None
    if not journal.is_file():
        return None
    try:
        read_jsonl_records(journal, repair_trailing=False, collect=False)
    except Exception:
        return None
    instance = _find_data(path, payload.get("csv"))
    tariff = _find_data(path, payload.get("prices_csv"))
    if (
        instance is None
        or tariff is None
        or sha256_file(instance) != provenance["instance_sha256"]
        or sha256_file(tariff) != provenance["prices_sha256"]
    ):
        return None
    route_weight = _finite(payload, "route_weight")
    artificials = _finite(payload, "artificials")
    if artificials is None or artificials < 0:
        return None
    certified = payload.get("certified_rc_optimal") is True
    min_rc = _finite(payload, "min_rc")
    if certified and (min_rc is None or min_rc < -1e-4):
        return None
    match = _artifact_scale(path, payload)
    numeric_scale = match[1] if match and isinstance(match[1], int) else None
    feasible = artificials == 0.0
    return {
        "target_reached_in_lp": (
            feasible
            and numeric_scale is not None
            and route_weight is not None
            and route_weight <= numeric_scale + 1e-9
        ),
        "zero_artificials": feasible,
        "pricing_certified": certified,
        "exact_integer_partition_found": False,
        "finite_pool_fleet_proven": False,
    }


def _verified_scale_evidence(paths: list[Path]) -> list[dict]:
    found = defaultdict(list)
    temporary_dirs = []
    scan_roots = []
    for raw in paths:
        path = raw.expanduser().resolve()
        if not path.exists():
            continue
        if path.is_file() and tarfile.is_tarfile(path):
            temporary = tempfile.TemporaryDirectory()
            temporary_dirs.append(temporary)
            extracted_root = Path(temporary.name)
            with tarfile.open(path, "r:*") as archive:
                for member in sorted(
                        archive.getmembers(), key=lambda item: item.name):
                    member_path = Path(member.name)
                    if (
                        not member.isfile()
                        or member_path.is_absolute()
                        or ".." in member_path.parts
                    ):
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    destination = extracted_root / member_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as handle:
                        shutil.copyfileobj(source, handle)
            scan_roots.append((extracted_root, path))
        else:
            scan_roots.append((path, None))
    for path, archive_source in scan_roots:
        payloads = []
        files = [path] if path.is_file() else sorted(path.rglob("*.json"))
        for candidate in files:
            try:
                payloads.append((
                    candidate, json.loads(candidate.read_text())
                ))
            except (OSError, ValueError):
                continue
        for candidate, payload in payloads:
            if not isinstance(payload, dict):
                continue
            scale = _artifact_scale(candidate, payload)
            if scale is None:
                continue
            flags = _validated_scale_flags(candidate, payload)
            if flags is None:
                continue
            display = (
                Path(
                    f"{archive_source}!"
                    f"{candidate.relative_to(path)}"
                )
                if archive_source is not None else candidate
            )
            found[scale].append((display, payload, flags))
    required = [
        ("single", "single duties"),
        ("pair", "pairs"),
        *[("union", value) for value in (3, 5, 8, 13, 15, 20, 30, 40)],
    ]
    rows = []
    for family, value in required:
        records = (
            [
                item for key, values in found.items()
                if key[0] == family
                and (family in {"single", "pair"} or key[1] == value)
                for item in values
            ]
        )
        target = any(
            flags["target_reached_in_lp"] is True
            for _path, _payload, flags in records
        )
        zero_art = any(
            flags["zero_artificials"] is True
            for _path, _payload, flags in records
        )
        pricing = any(
            flags["pricing_certified"] is True
            for _path, _payload, flags in records
        )
        integer = any(
            flags["exact_integer_partition_found"] is True
            for _path, _payload, flags in records
        )
        proof = any(
            flags["finite_pool_fleet_proven"] is True
            for _path, _payload, flags in records
        )
        rows.append({
            "scale_family": family,
            "scale": value,
            "availability": "verified" if records else "not available",
            "artifact_count": len(records),
            "target_reached_in_lp": target if records else None,
            "zero_artificials": zero_art if records else None,
            "pricing_certified": pricing if records else None,
            "exact_integer_partition_found": integer if records else None,
            "finite_pool_fleet_proven": proof if records else None,
            "source_artifacts": " | ".join(
                str(path) for path, _payload, _flags in records
            ),
            "notes": (
                "Proof, when present, is finite-pool only."
                if records else "No verified supplied artifact."
            ),
        })
    for temporary in temporary_dirs:
        temporary.cleanup()
    return rows


def _write_csv(path: Path, fields, rows) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _figures(staging: Path, rows: list[dict], historical: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdf_metadata = {
        "Creator": "EVSP-DR deterministic evidence builder",
        "CreationDate": None,
        "ModDate": None,
    }

    def save(fig, stem):
        fig.tight_layout()
        fig.savefig(
            staging / f"{stem}.png", dpi=180,
            metadata={"Software": "EVSP-DR"},
        )
        fig.savefig(staging / f"{stem}.pdf", metadata=pdf_metadata)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = defaultdict(list)
    for row in rows:
        if row["arm"] in {"CA", "CS"} and row["route_weight"] is not None:
            grouped[(row["arm"], row["replicate"])].append(row)
    colors = {"CA": "tab:blue", "CS": "tab:orange"}
    for (arm, replicate), values in sorted(grouped.items()):
        values.sort(key=lambda item: item["actual_hours"])
        ax.plot(
            [item["actual_hours"] for item in values],
            [item["route_weight"] for item in values],
            marker="o", alpha=0.45, color=colors[arm],
            label=f"{arm} {replicate}",
        )
    for arm in ("CA", "CS"):
        by_checkpoint = defaultdict(list)
        for row in rows:
            if row["arm"] == arm and row["route_weight"] is not None:
                by_checkpoint[row["checkpoint"]].append(row)
        means = []
        for values in by_checkpoint.values():
            if len(values) == 2:
                means.append((
                    sum(value["actual_hours"] for value in values) / 2,
                    sum(value["route_weight"] for value in values) / 2,
                ))
        means.sort()
        if means:
            ax.plot(
                [item[0] for item in means],
                [item[1] for item in means],
                linewidth=2.5, color=colors[arm],
                label=f"{arm} replicate mean",
            )
    ax.axhline(40.0, color="black", linestyle="--",
               label="40-duty LP reference")
    if historical is not None:
        ax.axhline(HISTORICAL_ROUTE_WEIGHT, color="purple", linestyle=":",
                   label="Historical 39.2520 LP endpoint")
    if not grouped:
        ax.text(0.5, 0.5, "Validated k40 inputs not available",
                transform=ax.transAxes, ha="center")
    ax.set(
        xlabel="Recorded exact-CG wall hours",
        ylabel="Restricted-master route weight (not integer buses)",
        title="k40 cover-master convergence (CA/CS kept separate)",
    )
    ax.legend(fontsize=7)
    save(fig, "k40_convergence_cover")

    fig, (ax_weight, ax_art) = plt.subplots(2, 1, figsize=(8, 7),
                                            sharex=True)
    for arm, color in (("PA", "tab:red"), ("PS", "tab:green")):
        values = [
            row for row in rows
            if row["arm"] == arm and row["route_weight"] is not None
        ]
        values.sort(key=lambda item: (
            item["replicate"], item["actual_hours"]
        ))
        for replicate in sorted({value["replicate"] for value in values}):
            replicate_rows = [
                value for value in values
                if value["replicate"] == replicate
            ]
            ax_weight.plot(
                [value["actual_hours"] for value in replicate_rows],
                [value["route_weight"] for value in replicate_rows],
                marker="o", color=color, alpha=0.55,
                label=f"{arm} {replicate} route weight",
            )
            ax_art.plot(
                [value["actual_hours"] for value in replicate_rows],
                [value["artificials"] for value in replicate_rows],
                marker="x", color=color, alpha=0.7,
                label=f"{arm} {replicate} artificials",
            )
    ax_weight.set_ylabel("Route weight (separate from artificials)")
    ax_art.set_ylabel("Artificial total")
    ax_art.set_xlabel("Recorded exact-CG wall hours")
    ax_weight.set_title(
        "Partition-master diagnostic: PA is infeasible while artificials remain"
    )
    if ax_weight.get_legend_handles_labels()[0]:
        ax_weight.legend(fontsize=7)
    if ax_art.get_legend_handles_labels()[0]:
        ax_art.legend(fontsize=7)
    if not rows:
        ax_weight.text(0.5, 0.5, "Validated k40 inputs not available",
                       transform=ax_weight.transAxes, ha="center")
    save(fig, "partition_failure_diagnostic")


def _input_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_file():
            records.append({
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "available": True,
            })
        elif resolved.is_dir():
            records.append({
                "path": str(resolved),
                "sha256": sha256_directory(resolved),
                "available": True,
                "kind": "directory",
            })
        else:
            records.append({
                "path": str(resolved),
                "sha256": None,
                "available": False,
            })
    return records


def build(
    *,
    factorial_campaigns: list[Path],
    historical_path: Path,
    legacy_analysis: Path,
    release_archives: list[Path],
    verified_artifacts: list[Path],
    output_dir: Path,
    generation_command: str,
    replace_output=False,
    allow_missing=False,
) -> dict:
    if len(factorial_campaigns) != 2:
        raise ValueError("exactly two factorial campaign paths are required")
    output = output_dir.expanduser().resolve()
    if output.exists():
        if not replace_output:
            raise FileExistsError(f"output exists: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, missing = _factorial_rows(
        factorial_campaigns, allow_missing=allow_missing
    )
    historical, historical_missing = _historical(
        historical_path, allow_missing=allow_missing
    )
    missing.extend(historical_missing)
    if historical is not None and rows:
        historical_trip_sha = hashlib.sha256(json.dumps(
            historical["trip_ids"], separators=(",", ":")
        ).encode()).hexdigest()
        if any(
            row["_trip_set_sha256"] != historical_trip_sha
            for row in rows
        ):
            raise ValueError("historical/factorial trip sets differ")
    comparison = []
    if historical is not None:
        for row in rows:
            if row["checkpoint"] != "h22":
                continue
            comparison.append({
                "replicate": row["replicate"],
                "arm": row["arm"],
                "checkpoint": row["checkpoint"],
                "actual_hours": row["actual_hours"],
                "route_weight": row["route_weight"],
                "artificials": row["artificials"],
                "lp_feasible": row["lp_feasible"],
                "historical_route_weight": historical["route_weight"],
                "delta_from_historical": (
                    row["route_weight"] - historical["route_weight"]
                    if row["lp_feasible"]
                    and row["route_weight"] is not None else None
                ),
                "historical_actual_hours": (
                    historical["actual_wall_s"] / 3600.0
                ),
                "historical_status_sha256": historical["status_sha256"],
            })
    legacy = legacy_analysis.expanduser().resolve()
    if not legacy.exists():
        missing.append({
            "input": str(legacy),
            "reason": "legacy_analysis_missing",
        })
    for release in release_archives:
        resolved = release.expanduser().resolve()
        if not resolved.exists():
            missing.append({
                "input": str(resolved),
                "reason": "release_archive_missing",
            })
    for artifact in verified_artifacts:
        resolved = artifact.expanduser().resolve()
        if not resolved.exists():
            missing.append({
                "input": str(resolved),
                "reason": "verified_artifact_missing",
            })
    if missing and not allow_missing:
        raise ValueError(
            "required evidence inputs are missing: "
            + json.dumps(missing[:10])
        )
    scale_inputs = [
        path for path in [legacy, *release_archives, *verified_artifacts]
        if path.exists()
    ]
    small_rows = _verified_scale_evidence(scale_inputs)
    staging = Path(tempfile.mkdtemp(
        dir=output.parent, prefix=f".{output.name}.tmp."
    ))
    try:
        _write_csv(
            staging / "k40_factorial_trajectory.csv",
            TRAJECTORY_FIELDS,
            sorted(rows, key=lambda row: (
                row["replicate"], row["arm"],
                row["actual_hours"], row["checkpoint"],
            )),
        )
        _write_csv(
            staging / "small_instance_certification.csv",
            SMALL_FIELDS,
            small_rows,
        )
        _write_csv(
            staging / "historical_endpoint_comparison.csv",
            HISTORICAL_FIELDS,
            sorted(comparison, key=lambda row: (
                row["replicate"], row["arm"]
            )),
        )
        _figures(staging, rows, historical)
        source_paths = [
            *factorial_campaigns,
            historical_path,
            legacy_analysis,
            *release_archives,
            *verified_artifacts,
        ]
        provenance = {
            "schema": "evsp-dr-convergence-evidence-v1",
            "source_artifacts": _input_records(source_paths),
            "factorial_commit": FACTORIAL_COMMIT,
            "historical_commit": HISTORICAL_COMMIT,
            "instance_sha256": INSTANCE_SHA256,
            "tariff_sha256": TARIFF_SHA256,
            "generation_command": generation_command,
            "missing_inputs": missing,
            "output_files": {},
            "scientific_guards": [
                "Route weight and artificials are never combined.",
                "Covering LP values are not integer schedules.",
                "Pricing certification requires explicit true provenance.",
                "Integer fleet proofs are finite-pool statements only.",
                "Missing scale evidence remains not available.",
            ],
        }
        readme = """# Convergence evidence (2026-08-15)

This directory is generated by `src/build_convergence_evidence.py`.

Evidence levels are deliberately separate:

1. target reached in a restricted-master LP;
2. zero artificial mass;
3. exact pricing certification;
4. an exact integer partition incumbent;
5. an integer fleet proof over one finite supplied pool.

Cover-master route weights are LP diagnostics, never integer schedules. RAW and
GIRO-augmented MIP pools must remain separate because augmentation changes the
feasible column set. PA route weight is explicitly infeasible whenever
artificials remain. No finite-pool proof is global pricing optimality.

Missing archives or scales are shown as **not available** and are never inferred
from filenames or neighboring instances.
"""
        if missing:
            readme += (
                "\n## Inputs unavailable in this checkout\n\n"
                "This tracked package is a deterministic placeholder. Rebuild "
                "from the explicit verified sources before using the figures "
                "as scientific evidence:\n\n"
                + "\n".join(
                    f"- `{item['input']}`: {item['reason']}"
                    for item in missing
                )
                + "\n"
            )
        (staging / "README.md").write_text(readme)
        for path in sorted(staging.iterdir()):
            if path.name == "provenance.json":
                continue
            provenance["output_files"][path.name] = sha256_file(path)
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        _rename_noreplace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "output_dir": str(output),
        "trajectory_rows": len(rows),
        "comparison_rows": len(comparison),
        "missing_inputs": missing,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factorial-campaign", type=Path, action="append", required=True
    )
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--legacy-analysis", type=Path, required=True)
    parser.add_argument(
        "--release-archive", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--verified-artifact", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("analysis/convergence_evidence_20260815"),
    )
    parser.add_argument("--generation-command")
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--allow-missing-inputs", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    command_argv = (
        sys.argv if argv is None
        else ["build_convergence_evidence.py", *argv]
    )
    result = build(
        factorial_campaigns=args.factorial_campaign,
        historical_path=args.historical,
        legacy_analysis=args.legacy_analysis,
        release_archives=args.release_archive,
        verified_artifacts=args.verified_artifact,
        output_dir=args.out_dir,
        generation_command=(
            args.generation_command or shlex.join(command_argv)
        ),
        replace_output=args.replace_output,
        allow_missing=args.allow_missing_inputs,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

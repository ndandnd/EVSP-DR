"""Read-only inventory and deterministic selection for MIP-statistics pools."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from durable_io import read_jsonl_records
from run_exact_pool_mip import resolve_pool_journal


PILOT_BUDGET_HOURS = {
    8: 4,
    10: 6,
    13: 6,
    15: 8,
    20: 12,
    30: 18,
    40: 23,
}
SECONDARY_SCALES = (20, 30, 40)
SECONDARY_AGES = (1, 3, 6, (10, 12), (15, 24))
EXPECTED_SOURCE_SCALES = {
    "repool_small": (8, 13, 15),
    "exact_big": (30, 40),
    "k40_factorial": (40,),
    "bigtar_snapshots": (30, 40),
    "fresh_preparation": (10, 20),
}
PILOT_ALLOWED_FAMILIES = {
    8: {"repool_small"},
    13: {"repool_small"},
    15: {"repool_small"},
    30: {"exact_big", "bigtar_snapshots"},
    40: {"exact_big", "k40_factorial", "bigtar_snapshots"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trip_set_sha256(trip_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(
        trip_ids, separators=(",", ":")
    ).encode()).hexdigest()


def _scale_from_status(status: dict) -> int | None:
    text = str(status.get("csv") or "")
    patterns = (
        r"(?:^|[_-])k(\d+)(?:[_-]|\.|$)",
        r"Practice_(\d+)bus",
        r"(?:^|[_-])(\d+)bus(?:[_-]|\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _replicate(status: dict, path: Path) -> str:
    text = f"{status.get('csv', '')} {path.name}"
    for pattern in (
        r"[_-]r(\d+)", r"RND(\d+)", r"rep(?:licate)?[_-]?(\d+)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"r{int(match.group(1))}"
    return hashlib.sha256(text.encode()).hexdigest()[:4]


def _age_hours(status: dict, path: Path) -> float | None:
    mark = status.get("snapshot_mark_minutes")
    if mark is not None:
        try:
            value = float(mark) / 60.0
            return value if math.isfinite(value) and value >= 0 else None
        except (TypeError, ValueError):
            return None
    match = re.search(r"\.m(\d+)\.snapshot\.json$", path.name)
    if match:
        return int(match.group(1)) / 60.0
    try:
        value = float(status.get("wall_s")) / 3600.0
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _find_data_file(
    status_path: Path,
    relative: str,
    data_roots: list[Path],
) -> Path | None:
    relative_path = Path(str(relative))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None
    candidates = [
        root.expanduser().resolve() / relative_path for root in data_roots
    ]
    candidates.extend(
        parent / "data" / relative_path
        for parent in status_path.parents
    )
    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        if candidate.is_file():
            return candidate
    return None


def validate_candidate(
    status_path: Path,
    *,
    source_family: str,
    data_roots: list[Path],
) -> dict:
    path = status_path.expanduser().resolve()
    raw = path.read_bytes()
    status = json.loads(raw)
    if not isinstance(status, dict):
        raise ValueError("status JSON is not an object")
    trip_ids = status.get("trip_ids")
    if (
        not isinstance(trip_ids, list)
        or not trip_ids
        or any(
            not isinstance(trip, int) or isinstance(trip, bool)
            for trip in trip_ids
        )
        or len(trip_ids) != len(set(trip_ids))
    ):
        raise ValueError("status has invalid trip IDs")
    provenance = status.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("status lacks provenance")
    for key in ("instance_sha256", "prices_sha256", "git_commit"):
        if not provenance.get(key):
            raise ValueError(f"status lacks provenance {key}")
    for key in (
        "soc_step", "block_min", "g_kwh", "charge_kw", "min_soc_frac",
    ):
        try:
            value = float(status[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"status has invalid {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"status has non-finite {key}")
    wall_s = float(status.get("wall_s"))
    if not math.isfinite(wall_s) or wall_s < 0:
        raise ValueError("status has invalid actual wall_s")
    journal = resolve_pool_journal(path, status).resolve()
    expected_sibling = Path(str(path) + ".columns.jsonl").resolve()
    if not journal.is_file():
        raise ValueError("paired journal is missing")
    if path.name.endswith(".snapshot.json") and journal != expected_sibling:
        raise ValueError("snapshot does not use its immutable sibling journal")
    read_jsonl_records(journal, repair_trailing=False, collect=False)
    known_trips = set(trip_ids)
    incidences = {}
    with journal.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            trips = record.get("trips") if isinstance(record, dict) else None
            if (
                not isinstance(trips, list)
                or not trips
                or any(
                    not isinstance(trip, int) or isinstance(trip, bool)
                    for trip in trips
                )
                or len(trips) != len(set(trips))
                or any(trip not in known_trips for trip in trips)
            ):
                raise ValueError(
                    f"journal line {line_number} has invalid trips"
                )
            try:
                cost = float(record["cost"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"journal line {line_number} has invalid cost"
                ) from exc
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(
                    f"journal line {line_number} has non-finite cost"
                )
            key = frozenset(trips)
            if key not in incidences or cost < incidences[key]:
                incidences[key] = cost
    if len(incidences) != int(status.get("columns", -1)):
        raise ValueError("journal unique incidence count differs from status")
    covered = set().union(*(set(key) for key in incidences)) \
        if incidences else set()
    if covered != known_trips:
        raise ValueError("journal does not cover every status trip")
    instance = _find_data_file(path, status.get("csv"), data_roots)
    tariff = _find_data_file(path, status.get("prices_csv"), data_roots)
    if instance is None or tariff is None:
        raise ValueError("instance or tariff bytes are unavailable")
    if sha256_file(instance) != provenance["instance_sha256"]:
        raise ValueError("instance bytes differ from provenance")
    if sha256_file(tariff) != provenance["prices_sha256"]:
        raise ValueError("tariff bytes differ from provenance")
    scale = _scale_from_status(status)
    if scale not in PILOT_BUDGET_HOURS:
        raise ValueError(f"unrecognized campaign scale: {scale}")
    treatment = {
        "master_sense": status.get("master_sense"),
        "initial_pool": status.get("initial_pool"),
    }
    if path.name.endswith(".snapshot.json"):
        match = re.search(r"\.m(\d+)\.snapshot\.json$", path.name)
        if (
            match is None
            or int(match.group(1))
            != int(round(float(status.get("snapshot_mark_minutes"))))
            or status.get("stop_reason")
            != f"snapshot_m{int(match.group(1))}"
        ):
            raise ValueError("snapshot filename/mark/stop reason mismatch")
    return {
        "candidate_id": hashlib.sha256(
            (str(path) + hashlib.sha256(raw).hexdigest()).encode()
        ).hexdigest()[:20],
        "available": True,
        "source_family": source_family,
        "status_path": str(path),
        "status_sha256": hashlib.sha256(raw).hexdigest(),
        "journal_path": str(journal),
        "journal_sha256": sha256_file(journal),
        "instance_path": str(instance),
        "instance_sha256": provenance["instance_sha256"],
        "tariff_path": str(tariff),
        "tariff_sha256": provenance["prices_sha256"],
        "source_commit": provenance["git_commit"],
        "scale": scale,
        "replicate": _replicate(status, path),
        "trip_count": len(trip_ids),
        "trip_set_sha256": trip_set_sha256(trip_ids),
        "age_hours": _age_hours(status, path),
        "actual_wall_s": wall_s,
        "snapshot_mark_minutes": status.get("snapshot_mark_minutes"),
        "stop_reason": status.get("stop_reason"),
        "physics": {
            key: status[key] for key in (
                "soc_step", "block_min", "g_kwh",
                "charge_kw", "min_soc_frac",
            )
        },
        "treatment": treatment,
        "csv": status.get("csv"),
        "prices_csv": status.get("prices_csv"),
        "certified_rc_optimal": (
            status.get("certified_rc_optimal") is True
        ),
    }


def _is_pool_status(path: Path) -> bool:
    name = path.name
    if not name.endswith(".json"):
        return False
    excluded = (
        "_mip", "campaign", "manifest", "routes_", "giro_",
        "profile", "attempt", "provenance",
    )
    return not any(token in name.lower() for token in excluded)


def inventory(
    roots: dict[str, Path],
    *,
    data_roots: list[Path],
) -> dict:
    candidates = []
    missing_roots = []
    rejected = []
    unknown = set(roots) - set(EXPECTED_SOURCE_SCALES)
    if unknown:
        raise ValueError(f"unknown inventory source families: {sorted(unknown)}")
    for family, root_value in sorted(roots.items()):
        root = root_value.expanduser().resolve()
        if not root.exists():
            missing_roots.append({
                "source_family": family,
                "path": str(root),
                "reason": "root_missing",
            })
            continue
        if root.is_file():
            missing_roots.append({
                "source_family": family,
                "path": str(root),
                "reason": "archive_requires_verified_extraction",
            })
            continue
        for path in sorted(root.rglob("*.json")):
            if not _is_pool_status(path):
                continue
            try:
                candidates.append(validate_candidate(
                    path,
                    source_family=family,
                    data_roots=data_roots,
                ))
            except (OSError, TypeError, ValueError, SystemExit) as exc:
                rejected.append({
                    "source_family": family,
                    "path": str(path.resolve()),
                    "reason": str(exc),
                })
    trajectory_groups = defaultdict(list)
    for candidate in candidates:
        trajectory_groups[(
            candidate["source_family"],
            candidate["scale"],
            candidate["replicate"],
            candidate["instance_sha256"],
        )].append(candidate)
    mixed_ids = set()
    for key, values in trajectory_groups.items():
        if len({value["trip_set_sha256"] for value in values}) > 1:
            mixed_ids.update(value["candidate_id"] for value in values)
            rejected.append({
                "source_family": key[0],
                "path": None,
                "reason": (
                    f"trajectory trip set changed for scale={key[1]} "
                    f"replicate={key[2]}"
                ),
            })
    candidates = [
        candidate for candidate in candidates
        if candidate["candidate_id"] not in mixed_ids
    ]
    missing_slots = []
    for family, scales in EXPECTED_SOURCE_SCALES.items():
        for scale in scales:
            if not any(
                candidate["source_family"] == family
                and candidate["scale"] == scale
                for candidate in candidates
            ):
                missing_slots.append({
                    "source_family": family,
                    "scale": scale,
                    "reason": "no_verified_candidate",
                })
    for scale in SECONDARY_SCALES:
        required_family = (
            "fresh_preparation" if scale == 20 else "bigtar_snapshots"
        )
        for target in SECONDARY_AGES:
            allowed = target if isinstance(target, tuple) else (target,)
            if not any(
                candidate["source_family"] == required_family
                and candidate["scale"] == scale
                and candidate.get("age_hours") in allowed
                for candidate in candidates
            ):
                missing_slots.append({
                    "source_family": required_family,
                    "scale": scale,
                    "age_hours": list(allowed),
                    "reason": "no_verified_age_candidate",
                })
    return {
        "schema": "evsp-dr-mip-statistics-inventory-v1",
        "selection_rule": (
            "Collapse each instance/replicate to its latest immutable age, "
            "then select the lower deterministic median by "
            "(trip_count, instance_sha256, replicate, candidate_id); never "
            "use observed LP/MIP quality."
        ),
        "roots": {
            family: str(path.expanduser().resolve())
            for family, path in sorted(roots.items())
        },
        "data_roots": [
            str(path.expanduser().resolve()) for path in data_roots
        ],
        "candidates": sorted(
            candidates,
            key=lambda item: (
                item["scale"], item["trip_count"],
                item["instance_sha256"], item["replicate"],
                item["age_hours"] if item["age_hours"] is not None else -1,
                item["candidate_id"],
            ),
        ),
        "missing_roots": missing_roots,
        "missing_slots": missing_slots,
        "rejected": rejected,
    }


def representative_candidates(inventory_payload: dict) -> dict[int, dict]:
    latest = {}
    for candidate in inventory_payload.get("candidates") or []:
        key = (
            candidate["scale"],
            candidate["source_family"],
            candidate["instance_sha256"],
            candidate["trip_set_sha256"],
            candidate["replicate"],
        )
        previous = latest.get(key)
        candidate_age = candidate.get("age_hours")
        previous_age = previous.get("age_hours") if previous else None
        if (
            previous is None
            or (candidate_age if candidate_age is not None else -1)
            > (previous_age if previous_age is not None else -1)
        ):
            latest[key] = candidate
    selected = {}
    for scale in PILOT_BUDGET_HOURS:
        if scale not in PILOT_ALLOWED_FAMILIES:
            continue
        choices = [
            candidate for key, candidate in latest.items()
            if key[0] == scale
            and candidate["source_family"] in PILOT_ALLOWED_FAMILIES[scale]
        ]
        choices.sort(key=lambda item: (
            item["trip_count"], item["instance_sha256"],
            item["replicate"], item["candidate_id"],
        ))
        if choices:
            selected[scale] = choices[(len(choices) - 1) // 2]
    return selected


def select_age_candidate(
    candidates: list[dict],
    *,
    scale: int,
    target,
) -> dict | None:
    allowed = target if isinstance(target, tuple) else (target,)
    matching = [
        candidate for candidate in candidates
        if candidate["scale"] == scale
        and candidate["source_family"] == (
            "fresh_preparation" if scale == 20 else "bigtar_snapshots"
        )
        and candidate.get("age_hours") is not None
        and str(candidate.get("status_path", "")).endswith(
            ".snapshot.json"
        )
        and any(
            abs(float(candidate["age_hours"]) - age) <= 1e-9
            for age in allowed
        )
    ]
    if not matching:
        return None
    matching.sort(key=lambda item: (
        allowed.index(min(
            allowed,
            key=lambda age: abs(float(item["age_hours"]) - age),
        )),
        item["trip_count"], item["instance_sha256"],
        item["replicate"], item["candidate_id"],
    ))
    return matching[0]

"""Shared read-only validation for completed k40 factorial artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from run_exact_pool_mip import resolve_pool_journal


INSTANCE_SHA256 = "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd"
PRICES_SHA256 = "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200"
FACTORIAL_COMMIT = "eb85ca0c"
HISTORICAL_COMMIT = "f43475b732c3fbc8447a30845834a7d9e8822ef3"
HISTORICAL_WEIGHT = 39.252026205592166
MARKS = (60, 180, 360, 720, 1320, 1440)
ARMS = {
    "CA": ("cover", "artificial"),
    "CS": ("cover", "singletons"),
    "PA": ("partition", "artificial"),
    "PS": ("partition", "singletons"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def _validate_journal(path: Path, status: dict) -> None:
    trip_ids = status.get("trip_ids")
    if not isinstance(trip_ids, list) or not trip_ids:
        raise ValueError(f"status trip_ids are missing: {path}")
    known = set(trip_ids)
    incidences = set()
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"malformed journal record {line_number}: {path}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"non-object journal record {line_number}: {path}"
                )
            trips = record.get("trips")
            if (
                not isinstance(trips, list)
                or not trips
                or len(trips) != len(set(trips))
                or any(trip not in known for trip in trips)
            ):
                raise ValueError(
                    f"invalid journal trips at line {line_number}: {path}"
                )
            try:
                cost = float(record["cost"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid journal cost at line {line_number}: {path}"
                ) from exc
            if not math.isfinite(cost):
                raise ValueError(
                    f"non-finite journal cost at line {line_number}: {path}"
                )
            key = frozenset(trips)
            if key in incidences:
                raise ValueError(
                    f"duplicate journal incidence at line {line_number}: {path}"
                )
            incidences.add(key)
    try:
        expected_columns = int(status["columns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"status columns are invalid: {path}") from exc
    if len(incidences) != expected_columns:
        raise ValueError(
            f"journal/status column mismatch ({len(incidences)} != "
            f"{expected_columns}): {path}"
        )
    final_lp = status.get("final_lp")
    final_lp = final_lp if isinstance(final_lp, dict) else {}
    for route in final_lp.get("positive_routes") or []:
        if (
            not isinstance(route, dict)
            or frozenset(route.get("trips") or []) not in incidences
        ):
            raise ValueError(
                f"final LP route is absent from paired journal: {path}"
            )


def _read_tsv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _metric(status: dict, key: str):
    final_lp = status.get("final_lp")
    final = status.get("final")
    final_lp = final_lp if isinstance(final_lp, dict) else {}
    final = final if isinstance(final, dict) else {}
    if key == "objective":
        value = final_lp.get("objective", final.get("lp_obj"))
    elif key == "route_weight":
        value = final_lp.get("route_weight", final.get("route_weight"))
    elif key == "artificials":
        value = final_lp.get(
            "artificial_total", final.get("artificials")
        )
    elif key == "min_rc":
        value = final.get("min_rc")
    else:
        raise KeyError(key)
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _validate_common_status(
    path: Path,
    status: dict,
    *,
    expected_commit_prefix: str,
    factorial_controls: bool = True,
) -> tuple[Path, str]:
    provenance = status.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"missing provenance: {path}")
    if provenance.get("instance_sha256") != INSTANCE_SHA256:
        raise ValueError(f"wrong k40-r2 instance hash: {path}")
    if provenance.get("prices_sha256") != PRICES_SHA256:
        raise ValueError(f"wrong flat tariff hash: {path}")
    commit = str(provenance.get("git_commit") or "")
    if not commit.startswith(expected_commit_prefix):
        raise ValueError(f"unexpected generation commit {commit}: {path}")
    for field, expected in (
        ("soc_step", 15.0),
        ("block_min", 10),
        ("g_kwh", 300.0),
        ("charge_kw", 300.0),
        ("min_soc_frac", 0.0),
    ):
        if not math.isclose(
                float(status.get(field)), float(expected),
                rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{field} mismatch in {path}")
    args = provenance.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"missing provenance args: {path}")
    if int(args.get("columns_per_iter", -1)) != 30:
        raise ValueError(f"columns_per_iter mismatch: {path}")
    if not math.isclose(
            float(args.get("rc_eps", math.nan)), 1e-4,
            rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"rc_eps mismatch: {path}")
    if not math.isclose(
            float(provenance.get("rc_eps", math.nan)), 1e-4,
            rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"top-level provenance rc_eps mismatch: {path}")
    if factorial_controls:
        for field, expected in (
            ("max_iters", 200000),
            ("wall_limit_s", 90000.0),
            ("checkpoint_every", 25),
        ):
            if not math.isclose(
                    float(args.get(field, math.nan)), float(expected),
                    rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{field} mismatch: {path}")
        snapshots = str(
            args.get("snapshot_at_minutes") or ""
        ).replace(" ", "")
        if snapshots != ",".join(str(mark) for mark in MARKS):
            raise ValueError(f"snapshot controls mismatch: {path}")
        if args.get("resume") is not True:
            raise ValueError(f"resume control mismatch: {path}")
    try:
        journal = resolve_pool_journal(path, status).resolve()
    except SystemExit as exc:
        raise ValueError(f"journal is missing for {path}") from exc
    if not journal.is_file():
        raise ValueError(f"journal is missing: {journal}")
    expected_journal = Path(str(path) + ".columns.jsonl").resolve()
    if journal != expected_journal:
        raise ValueError(
            f"journal/status pairing mismatch: {path} -> {journal}"
        )
    _validate_journal(journal, status)
    return journal, commit


def _row(
    *,
    replicate: str,
    campaign: str,
    arm: str,
    checkpoint: str,
    nominal_minutes: int | None,
    status_path: Path,
    status: dict,
    journal: Path,
    naming_error: bool,
) -> dict:
    wall_s = float(status.get("wall_s"))
    if not math.isfinite(wall_s) or wall_s < 0.0:
        raise ValueError(f"invalid wall_s: {status_path}")
    artificials = _metric(status, "artificials")
    real_lp_feasible = (
        artificials is not None and artificials <= 1e-6
    )
    route_weight = _metric(status, "route_weight")
    return {
        "row_type": "replicate",
        "replicate": replicate,
        "campaign": campaign,
        "arm": arm,
        "master_sense": status.get("master_sense"),
        "initial_pool": status.get("initial_pool"),
        "checkpoint": checkpoint,
        "nominal_minutes": nominal_minutes,
        "actual_wall_s": wall_s,
        "actual_hours": wall_s / 3600.0,
        "iterations": int(status.get("iterations", 0)),
        "columns": int(status.get("columns", 0)),
        "objective": _metric(status, "objective"),
        "route_weight": route_weight,
        "artificials": artificials,
        "min_reduced_cost": _metric(status, "min_rc"),
        "real_lp_feasible": real_lp_feasible,
        "feasible_route_weight": (
            route_weight if real_lp_feasible else None
        ),
        "certified": status.get("certified_rc_optimal") is True,
        "stop_reason": status.get("stop_reason"),
        "final_lp_source": status.get("final_lp_source"),
        "status_path": str(status_path.resolve()),
        "status_sha256": sha256_file(status_path),
        "journal_path": str(journal),
        "journal_sha256": sha256_file(journal),
        "visible_k40r1_naming_error": naming_error,
    }


def _find_arm_status(
    campaign_dir: Path,
    arm: str,
    suffix: str,
) -> tuple[Path, bool]:
    candidates = [
        campaign_dir / f"k40r2_flat_{arm}{suffix}",
        campaign_dir / f"k40r1_flat_{arm}{suffix}",
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(
            f"expected one k40r1/k40r2 {arm}{suffix}, found {existing}"
        )
    return existing[0], "k40r1" in existing[0].name


def validate_campaign(
    campaign_dir: Path,
    *,
    replicate: str,
) -> dict:
    root = campaign_dir.expanduser().resolve()
    launch_path = root / "launch.tsv"
    prep_path = root / "prep_attestation.tsv"
    input_manifest = root / "input_manifest.json"
    for path in (launch_path, prep_path, input_manifest):
        if not path.is_file():
            raise ValueError(f"campaign manifest is missing: {path}")
    launch = _read_tsv(launch_path)
    arm_rows = {
        row["job_name"].split("-")[1][:2]: row
        for row in launch if row.get("role") == "arm"
    }
    if set(arm_rows) != set(ARMS):
        raise ValueError(f"campaign does not contain CA/CS/PA/PS: {root}")
    if len({row.get("job_id") for row in arm_rows.values()}) != len(ARMS):
        raise ValueError(f"campaign arm job IDs are not unique: {root}")
    prep = {
        row[0]: row[1]
        for row in csv.reader(prep_path.open(), delimiter="\t")
        if len(row) >= 2
    }
    if not prep.get("git_commit", "").startswith(FACTORIAL_COMMIT):
        raise ValueError(f"unexpected factorial commit: {root}")
    if prep.get("instance_sha256") != INSTANCE_SHA256:
        raise ValueError(f"prep instance hash mismatch: {root}")
    if prep.get("prices_sha256") != PRICES_SHA256:
        raise ValueError(f"prep tariff hash mismatch: {root}")
    _load_object(input_manifest)

    rows = []
    files = {launch_path, prep_path, input_manifest}
    expected_trip_ids = None
    for arm, (sense, initial_pool) in ARMS.items():
        manifest_row = arm_rows[arm]
        if (manifest_row.get("master_sense") != sense
                or manifest_row.get("initial_pool") != initial_pool):
            raise ValueError(f"launch treatment mismatch for {arm}: {root}")
        for mark in MARKS:
            status_path, naming_error = _find_arm_status(
                root, arm, f".m{mark}.snapshot.json"
            )
            status = _load_object(status_path)
            journal, status_commit = _validate_common_status(
                status_path, status,
                expected_commit_prefix=FACTORIAL_COMMIT,
            )
            if status_commit != prep["git_commit"]:
                raise ValueError(f"status/prep commit mismatch: {status_path}")
            if status.get("master_sense") != sense:
                raise ValueError(f"status master treatment mismatch: {status_path}")
            if status.get("initial_pool") != initial_pool:
                raise ValueError(f"status initializer mismatch: {status_path}")
            if not math.isclose(
                    float(status.get("snapshot_mark_minutes")),
                    float(mark), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"snapshot mark mismatch: {status_path}")
            if status.get("stop_reason") != f"snapshot_m{mark}":
                raise ValueError(f"snapshot stop reason mismatch: {status_path}")
            if expected_trip_ids is None:
                expected_trip_ids = status.get("trip_ids")
            if status.get("trip_ids") != expected_trip_ids:
                raise ValueError(f"campaign trip set mismatch: {status_path}")
            rows.append(_row(
                replicate=replicate,
                campaign=root.name,
                arm=arm,
                checkpoint=f"m{mark}",
                nominal_minutes=mark,
                status_path=status_path,
                status=status,
                journal=journal,
                naming_error=naming_error,
            ))
            files.update((status_path, journal))
        terminal_path, naming_error = _find_arm_status(root, arm, ".json")
        terminal = _load_object(terminal_path)
        journal, status_commit = _validate_common_status(
            terminal_path, terminal,
            expected_commit_prefix=FACTORIAL_COMMIT,
        )
        if status_commit != prep["git_commit"]:
            raise ValueError(f"terminal/prep commit mismatch: {terminal_path}")
        if (terminal.get("master_sense") != sense
                or terminal.get("initial_pool") != initial_pool):
            raise ValueError(f"terminal treatment mismatch: {terminal_path}")
        if terminal.get("trip_ids") != expected_trip_ids:
            raise ValueError(f"terminal trip set mismatch: {terminal_path}")
        if terminal.get("stop_reason") in {
                None, "initializing", "running"
        } or str(terminal.get("stop_reason")).startswith("snapshot_"):
            raise ValueError(f"terminal status is not terminal: {terminal_path}")
        rows.append(_row(
            replicate=replicate,
            campaign=root.name,
            arm=arm,
            checkpoint="terminal",
            nominal_minutes=None,
            status_path=terminal_path,
            status=terminal,
            journal=journal,
            naming_error=naming_error,
        ))
        files.update((terminal_path, journal))
        iters = Path(str(terminal_path) + ".iters.csv")
        if not iters.is_file():
            raise ValueError(f"terminal trajectory is missing: {iters}")
        files.add(iters)
        allocations = root / f"{terminal_path.stem}.allocations.tsv"
        if allocations.is_file():
            files.add(allocations)
    return {
        "campaign_dir": str(root),
        "campaign": root.name,
        "replicate": replicate,
        "rows": rows,
        "files": sorted(str(path.resolve()) for path in files),
        "prep": prep,
    }


def validate_historical(path: Path) -> dict:
    status_path = path.expanduser().resolve()
    status = _load_object(status_path)
    journal, commit = _validate_common_status(
        status_path,
        status,
        expected_commit_prefix=HISTORICAL_COMMIT,
        factorial_controls=False,
    )
    if commit != HISTORICAL_COMMIT:
        raise ValueError("historical comparator commit mismatch")
    if status.get("stop_reason") != "wall_limit":
        raise ValueError("historical comparator is not terminal")
    wall_s = float(status.get("wall_s"))
    if not 20.0 <= wall_s / 3600.0 <= 24.0:
        raise ValueError("historical comparator is not approximately 22 hours")
    weight = _metric(status, "route_weight")
    if weight is None or not math.isclose(
            weight, HISTORICAL_WEIGHT, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"historical route weight {weight} differs from comparator "
            f"{HISTORICAL_WEIGHT}"
        )
    return {
        "status_path": str(status_path),
        "status_sha256": sha256_file(status_path),
        "journal_path": str(journal),
        "journal_sha256": sha256_file(journal),
        "actual_wall_s": wall_s,
        "actual_hours": wall_s / 3600.0,
        "route_weight": weight,
        "objective": _metric(status, "objective"),
        "artificials": _metric(status, "artificials"),
        "min_reduced_cost": _metric(status, "min_rc"),
        "certified": status.get("certified_rc_optimal") is True,
        "files": [str(status_path), str(journal)],
    }

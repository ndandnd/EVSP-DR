"""Shared read-only validation for completed k40 factorial artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

from run_exact_pool_mip import resolve_pool_journal


INSTANCE_SHA256 = "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd"
PRICES_SHA256 = "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200"
FACTORIAL_COMMIT = "eb85ca0cc439956939ba6bf9c42958808d89aadd"
HISTORICAL_COMMIT = "f43475b732c3fbc8447a30845834a7d9e8822ef3"
HISTORICAL_WEIGHT = 39.252026205592166
EXPECTED_TRIPS = 947
BIG_M_PENALTY = 500000.0
KNOWN_CAMPAIGNS = {
    "R1": "k40fx_20260814T140232Z_eb85ca0c",
    "R2": "k40fx_20260814T191933Z_eb85ca0c",
}
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


def _load_object_with_sha(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _validate_journal(
    path: Path,
    status: dict,
    *,
    require_lp_provenance: bool,
) -> tuple[dict[frozenset, dict], str]:
    trip_ids = status.get("trip_ids")
    if not isinstance(trip_ids, list) or not trip_ids:
        raise ValueError(f"status trip_ids are missing: {path}")
    if (
        trip_ids != list(range(EXPECTED_TRIPS))
        or len(set(trip_ids)) != EXPECTED_TRIPS
        or any(
            not isinstance(trip, int) or isinstance(trip, bool)
            for trip in trip_ids
        )
    ):
        raise ValueError(
            f"status trip IDs are not the exact deterministic 0..946 set: {path}"
        )
    known = set(trip_ids)
    effective = {}
    recorded_costs = {}
    journal_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            journal_digest.update(line)
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
                or any(
                    not isinstance(trip, int) or isinstance(trip, bool)
                    for trip in trips
                )
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
            if not math.isfinite(cost) or cost < 0.0:
                raise ValueError(
                    f"non-finite journal cost at line {line_number}: {path}"
                )
            key = frozenset(trips)
            previous = effective.get(key)
            if previous is not None and cost >= float(previous["cost"]) - 1e-9:
                raise ValueError(
                    f"journal replacement is not cheaper at line "
                    f"{line_number}: {path}"
                )
            effective[key] = record
            recorded_costs.setdefault(key, []).append(cost)
    try:
        expected_columns = int(status["columns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"status columns are invalid: {path}") from exc
    if len(effective) != expected_columns:
        raise ValueError(
            f"journal/status column mismatch ({len(effective)} != "
            f"{expected_columns}): {path}"
        )
    final_lp = status.get("final_lp")
    if not isinstance(final_lp, dict):
        raise ValueError(f"status final_lp is missing: {path}")
    positive = final_lp.get("positive_routes")
    if not isinstance(positive, list):
        raise ValueError(f"status positive_routes is invalid: {path}")
    route_weight = 0.0
    route_objective = 0.0
    coverage = {trip: 0.0 for trip in trip_ids}
    detail_source = final_lp.get("source")
    if require_lp_provenance:
        if (
            not isinstance(detail_source, str)
            or status.get("final_lp_source") != detail_source
        ):
            raise ValueError(f"final LP source provenance mismatch: {path}")
        try:
            lp_pool_columns = int(final_lp["pool_columns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"final LP pool column provenance is invalid: {path}"
            ) from exc
        if (
            lp_pool_columns < 0
            or lp_pool_columns > expected_columns
            or (
                detail_source == "final_pool_resolve"
                and lp_pool_columns != expected_columns
            )
        ):
            raise ValueError(
                f"final LP pool column provenance mismatch: {path}"
            )
    for route in positive:
        if not isinstance(route, dict):
            raise ValueError(f"final LP route is invalid: {path}")
        key = frozenset(route.get("trips") or [])
        if key not in effective:
            raise ValueError(
                f"final LP route is absent from paired journal: {path}"
            )
        try:
            value = float(route["value"])
            route_cost = float(route["cost"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"final LP route metric is invalid: {path}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"final LP route value is not positive: {path}")
        if not any(
            math.isclose(
                route_cost, recorded_cost,
                rel_tol=1e-10, abs_tol=1e-6,
            )
            for recorded_cost in recorded_costs[key]
        ):
            raise ValueError(f"final LP route cost disagrees with journal: {path}")
        if (
            (not require_lp_provenance or detail_source == "final_pool_resolve")
            and not math.isclose(
                route_cost, float(effective[key]["cost"]),
                rel_tol=1e-10, abs_tol=1e-6,
            )
        ):
            raise ValueError(
                f"final-pool LP route does not use effective journal cost: {path}"
            )
        route_weight += value
        route_objective += value * route_cost
        for trip in key:
            coverage[trip] += value
    try:
        recorded_weight = float(final_lp["route_weight"])
        artificials = float(final_lp["artificial_total"])
        recorded_objective = float(final_lp["objective"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"final LP summary metric is invalid: {path}") from exc
    if not all(math.isfinite(value) for value in (
            recorded_weight, artificials, recorded_objective)):
        raise ValueError(f"final LP summary metric is non-finite: {path}")
    if artificials < 0.0:
        raise ValueError(f"final LP artificial total is negative: {path}")
    if not math.isclose(
            route_weight, recorded_weight, rel_tol=1e-9, abs_tol=1e-7):
        raise ValueError(f"final LP route weight is inconsistent: {path}")
    recomputed_objective = route_objective + BIG_M_PENALTY * artificials
    if not math.isclose(
            recomputed_objective, recorded_objective,
            rel_tol=1e-9, abs_tol=1e-3):
        raise ValueError(f"final LP objective is inconsistent: {path}")
    master_sense = status.get("master_sense", "cover")
    if master_sense not in {"cover", "partition"}:
        raise ValueError(f"unknown master sense for LP validation: {path}")
    if master_sense == "partition" and any(
            value > 1.0 + 1e-6 for value in coverage.values()):
        raise ValueError(f"partition LP overcovers a trip: {path}")
    total_deficit = sum(max(0.0, 1.0 - value) for value in coverage.values())
    if not math.isclose(
            total_deficit, artificials, rel_tol=1e-8, abs_tol=1e-5):
        raise ValueError(
            f"final LP trip coverage disagrees with artificials: {path}"
        )
    final = status.get("final")
    if not isinstance(final, dict):
        raise ValueError(f"status final iteration is missing: {path}")
    min_rc = final.get("min_rc")
    try:
        min_rc = float(min_rc)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"minimum reduced cost is invalid: {path}") from exc
    if not math.isfinite(min_rc):
        raise ValueError(f"minimum reduced cost is non-finite: {path}")
    return effective, journal_digest.hexdigest()


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
    expected_commit: str,
    factorial_controls: bool = True,
) -> tuple[Path, str, str]:
    provenance = status.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"missing provenance: {path}")
    if provenance.get("instance_sha256") != INSTANCE_SHA256:
        raise ValueError(f"wrong k40-r2 instance hash: {path}")
    if provenance.get("prices_sha256") != PRICES_SHA256:
        raise ValueError(f"wrong flat tariff hash: {path}")
    commit = str(provenance.get("git_commit") or "")
    if commit != expected_commit:
        raise ValueError(f"unexpected generation commit {commit}: {path}")
    if provenance.get("git_dirty") is not False:
        raise ValueError(f"generation checkout was dirty/unknown: {path}")
    csv_name = Path(str(status.get("csv") or "")).name
    if csv_name not in {
        "Practice_Custom_DutyUnion_k40_r1.csv",
        "Practice_Custom_DutyUnion_k40_r2.csv",
    }:
        raise ValueError(f"unexpected instance path in {path}")
    if Path(str(status.get("prices_csv") or "")).name != "hourly_prices_flat.csv":
        raise ValueError(f"unexpected tariff path in {path}")
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
    for arg_name, status_name in (
        ("csv", "csv"),
        ("prices_csv", "prices_csv"),
        ("soc_step", "soc_step"),
        ("block_min", "block_min"),
        ("g_kwh", "g_kwh"),
        ("charge_kw", "charge_kw"),
        ("min_soc_frac", "min_soc_frac"),
    ):
        if str(args.get(arg_name)) != str(status.get(status_name)):
            raise ValueError(
                f"provenance {arg_name} disagrees with status: {path}"
            )
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
        if provenance.get("git_branch") not in (None, ""):
            raise ValueError(f"generation checkout was not detached: {path}")
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
        if args.get("master_sense") != status.get("master_sense"):
            raise ValueError(f"provenance master treatment mismatch: {path}")
        if args.get("initial_pool") != status.get("initial_pool"):
            raise ValueError(f"provenance initializer mismatch: {path}")
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
    _pool, journal_sha = _validate_journal(
        journal,
        status,
        require_lp_provenance=factorial_controls,
    )
    return journal, commit, journal_sha


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
    status_sha256: str,
    journal_sha256: str,
    naming_error: bool,
) -> dict:
    wall_s = float(status.get("wall_s"))
    if not math.isfinite(wall_s) or wall_s < 0.0:
        raise ValueError(f"invalid wall_s: {status_path}")
    if (
        nominal_minutes is not None
        and wall_s + 1e-6 < nominal_minutes * 60.0
    ):
        raise ValueError(
            f"snapshot wall_s predates its checkpoint: {status_path}"
        )
    artificials = _metric(status, "artificials")
    real_lp_feasible = artificials == 0.0
    route_weight = _metric(status, "route_weight")
    certified = status.get("certified_rc_optimal") is True
    min_rc = _metric(status, "min_rc")
    if certified and (
        status.get("stop_reason") != "certified"
        or min_rc is None
        or min_rc < -1e-4
    ):
        raise ValueError(
            f"certification flag is inconsistent with status/min_rc: "
            f"{status_path}"
        )
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
        "min_reduced_cost": min_rc,
        "real_lp_feasible": real_lp_feasible,
        "feasible_route_weight": (
            route_weight if real_lp_feasible else None
        ),
        "certified": certified,
        "stop_reason": status.get("stop_reason"),
        "final_lp_source": status.get("final_lp_source"),
        "status_path": str(status_path.resolve()),
        "status_sha256": status_sha256,
        "journal_path": str(journal),
        "journal_sha256": journal_sha256,
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


def _validate_input_manifest(path: Path) -> dict:
    manifest = _load_object(path)
    if manifest.get("seed") != 20260803:
        raise ValueError(f"input manifest seed mismatch: {path}")
    if manifest.get("source") != "Par_VehicleDetails_Updated.csv":
        raise ValueError(f"input manifest source mismatch: {path}")
    unions = manifest.get("unions")
    if not isinstance(unions, list):
        raise ValueError(f"input manifest unions are missing: {path}")
    matches = [
        entry for entry in unions
        if isinstance(entry, dict)
        and entry.get("sha256") == INSTANCE_SHA256
    ]
    if len(matches) != 1:
        raise ValueError(
            f"input manifest must identify exactly one intended k40-r2 entry: "
            f"{path}"
        )
    entry = matches[0]
    if (
        int(entry.get("k", -1)) != 40
        or int(entry.get("trips", -1)) != EXPECTED_TRIPS
        or entry.get("csv") not in {
            "Practice_Custom_DutyUnion_k40_r1.csv",
            "Practice_Custom_DutyUnion_k40_r2.csv",
        }
    ):
        raise ValueError(f"input manifest k40 entry mismatch: {path}")
    duties = entry.get("duties")
    if (
        not isinstance(duties, list)
        or len(duties) != 40
        or len(set(duties)) != 40
    ):
        raise ValueError(f"input manifest duties mismatch: {path}")
    return entry


def _validate_trajectory(path: Path, terminal: dict) -> str:
    required = (
        "elapsed_s", "iteration", "lp_obj", "route_weight",
        "artificials", "min_rc", "pool_columns",
    )
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode()))
    if tuple(reader.fieldnames or ()) != required:
        raise ValueError(f"trajectory header mismatch: {path}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"trajectory has no iterations: {path}")
    previous_elapsed = -math.inf
    previous_iteration = -1
    previous_pool_columns = -1
    for row in rows:
        try:
            elapsed = float(row["elapsed_s"])
            iteration = int(row["iteration"])
            values = [
                float(row[name]) for name in (
                    "lp_obj", "route_weight", "artificials", "min_rc"
                )
            ]
            pool_columns = int(row["pool_columns"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"trajectory row is malformed: {path}") from exc
        if (
            not all(math.isfinite(value) for value in [elapsed, *values])
            or elapsed < previous_elapsed
            or iteration <= previous_iteration
            or pool_columns < previous_pool_columns
        ):
            raise ValueError(f"trajectory row is inconsistent: {path}")
        previous_elapsed = elapsed
        previous_iteration = iteration
        previous_pool_columns = pool_columns
    final = terminal["final"]
    last = rows[-1]
    for status_key, csv_key, tolerance in (
        ("lp_obj", "lp_obj", 1e-3),
        ("route_weight", "route_weight", 1e-7),
        ("artificials", "artificials", 1e-6),
        ("min_rc", "min_rc", 1e-5),
    ):
        if not math.isclose(
            float(final[status_key]), float(last[csv_key]),
            rel_tol=1e-8, abs_tol=tolerance,
        ):
            raise ValueError(f"trajectory final {csv_key} mismatch: {path}")
    if previous_iteration != int(terminal["iterations"]):
        raise ValueError(f"trajectory iteration count mismatch: {path}")
    if int(last["pool_columns"]) > int(terminal["columns"]):
        raise ValueError(f"trajectory exceeds terminal pool columns: {path}")
    if previous_elapsed > float(terminal["wall_s"]) + 1.0:
        raise ValueError(f"trajectory elapsed time exceeds terminal wall_s: {path}")
    return hashlib.sha256(raw).hexdigest()


def _validate_allocations(
    path: Path,
    *,
    expected_job_id: str,
) -> str:
    raw = path.read_bytes()
    rows = list(csv.DictReader(
        io.StringIO(raw.decode()), delimiter="\t"
    ))
    if not rows:
        raise ValueError(f"allocation log has no rows: {path}")
    for row in rows:
        if (
            row.get("job_id") != expected_job_id
            or row.get("instance_sha256") != INSTANCE_SHA256
            or row.get("prices_sha256") != PRICES_SHA256
            or not row.get("utc")
            or not row.get("host")
        ):
            raise ValueError(f"allocation attestation mismatch: {path}")
        try:
            int(row["restart"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"allocation restart is invalid: {path}") from exc
    return hashlib.sha256(raw).hexdigest()


def trip_set_sha256(trip_ids: list[int]) -> str:
    encoded = json.dumps(
        sorted(trip_ids), separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_campaign(
    campaign_dir: Path,
    *,
    replicate: str,
) -> dict:
    root = campaign_dir.expanduser().resolve()
    if replicate not in KNOWN_CAMPAIGNS:
        raise ValueError(f"replicate must be R1 or R2, found {replicate}")
    if root.name != KNOWN_CAMPAIGNS[replicate]:
        raise ValueError(
            f"{replicate} must be approved campaign "
            f"{KNOWN_CAMPAIGNS[replicate]}, found {root.name}"
        )
    launch_path = root / "launch.tsv"
    prep_path = root / "prep_attestation.tsv"
    input_manifest = root / "input_manifest.json"
    for path in (launch_path, prep_path, input_manifest):
        if not path.is_file():
            raise ValueError(f"campaign manifest is missing: {path}")
    launch = _read_tsv(launch_path)
    if len(launch) != 5:
        raise ValueError(f"campaign launch must contain prep plus four arms: {root}")
    prep_launch = [row for row in launch if row.get("role") == "prep"]
    arm_launch = [row for row in launch if row.get("role") == "arm"]
    if (
        len(prep_launch) != 1
        or prep_launch[0].get("job_name") != "K40-PREP"
        or prep_launch[0].get("master_sense") != "-"
        or prep_launch[0].get("initial_pool") != "-"
    ):
        raise ValueError(f"campaign prep launch row mismatch: {root}")
    arm_rows = {}
    for arm, (sense, initial_pool) in ARMS.items():
        matches = [
            row for row in arm_launch
            if row.get("job_name") == f"K40-{arm}24"
            and row.get("master_sense") == sense
            and row.get("initial_pool") == initial_pool
        ]
        if len(matches) != 1:
            raise ValueError(f"campaign launch row mismatch for {arm}: {root}")
        arm_rows[arm] = matches[0]
    job_ids = [row.get("job_id") for row in launch]
    if (
        len(set(job_ids)) != 5
        or any(not str(job_id).isdigit() for job_id in job_ids)
    ):
        raise ValueError(f"campaign job IDs are invalid or duplicated: {root}")
    prep = {}
    with prep_path.open(newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 2 or row[0] in prep:
                raise ValueError(f"prep attestation is malformed: {prep_path}")
            prep[row[0]] = row[1]
    if prep.get("git_commit") != FACTORIAL_COMMIT:
        raise ValueError(f"unexpected factorial commit: {root}")
    if prep.get("instance_sha256") != INSTANCE_SHA256:
        raise ValueError(f"prep instance hash mismatch: {root}")
    if prep.get("prices_sha256") != PRICES_SHA256:
        raise ValueError(f"prep tariff hash mismatch: {root}")
    manifest_entry = _validate_input_manifest(input_manifest)
    repo_root = root.parents[3]
    instance_path = (
        repo_root / "data/duty_unions_big" / manifest_entry["csv"]
    ).resolve()
    prices_path = (repo_root / "data/hourly_prices_flat.csv").resolve()
    if (
        not instance_path.is_file()
        or sha256_file(instance_path) != INSTANCE_SHA256
    ):
        raise ValueError(f"campaign instance bytes are unavailable/mismatched: {root}")
    if (
        not prices_path.is_file()
        or sha256_file(prices_path) != PRICES_SHA256
    ):
        raise ValueError(f"campaign tariff bytes are unavailable/mismatched: {root}")

    rows = []
    files = {
        launch_path, prep_path, input_manifest, instance_path, prices_path
    }
    validated_file_hashes = {
        str(launch_path.resolve()): sha256_file(launch_path),
        str(prep_path.resolve()): sha256_file(prep_path),
        str(input_manifest.resolve()): sha256_file(input_manifest),
        str(instance_path): INSTANCE_SHA256,
        str(prices_path): PRICES_SHA256,
    }
    expected_trip_ids = None
    for arm, (sense, initial_pool) in ARMS.items():
        manifest_row = arm_rows[arm]
        if (manifest_row.get("master_sense") != sense
                or manifest_row.get("initial_pool") != initial_pool):
            raise ValueError(f"launch treatment mismatch for {arm}: {root}")
        previous_wall = -math.inf
        for mark in MARKS:
            status_path, naming_error = _find_arm_status(
                root, arm, f".m{mark}.snapshot.json"
            )
            status, status_sha = _load_object_with_sha(status_path)
            journal, status_commit, journal_sha = _validate_common_status(
                status_path, status,
                expected_commit=FACTORIAL_COMMIT,
            )
            if status_commit != prep["git_commit"]:
                raise ValueError(f"status/prep commit mismatch: {status_path}")
            if status.get("master_sense") != sense:
                raise ValueError(f"status master treatment mismatch: {status_path}")
            if status.get("initial_pool") != initial_pool:
                raise ValueError(f"status initializer mismatch: {status_path}")
            if Path(str(status.get("csv"))).name != manifest_entry["csv"]:
                raise ValueError(f"status/input-manifest CSV mismatch: {status_path}")
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
            report_row = _row(
                replicate=replicate,
                campaign=root.name,
                arm=arm,
                checkpoint=f"m{mark}",
                nominal_minutes=mark,
                status_path=status_path,
                status=status,
                journal=journal,
                status_sha256=status_sha,
                journal_sha256=journal_sha,
                naming_error=naming_error,
            )
            if report_row["actual_wall_s"] < previous_wall:
                raise ValueError(
                    f"snapshot wall chronology regressed: {status_path}"
                )
            previous_wall = report_row["actual_wall_s"]
            rows.append(report_row)
            validated_file_hashes[report_row["status_path"]] = status_sha
            validated_file_hashes[report_row["journal_path"]] = journal_sha
            files.update((status_path, journal))
        terminal_path, naming_error = _find_arm_status(root, arm, ".json")
        terminal, terminal_sha = _load_object_with_sha(terminal_path)
        journal, status_commit, journal_sha = _validate_common_status(
            terminal_path, terminal,
            expected_commit=FACTORIAL_COMMIT,
        )
        if status_commit != prep["git_commit"]:
            raise ValueError(f"terminal/prep commit mismatch: {terminal_path}")
        if (terminal.get("master_sense") != sense
                or terminal.get("initial_pool") != initial_pool):
            raise ValueError(f"terminal treatment mismatch: {terminal_path}")
        if terminal.get("trip_ids") != expected_trip_ids:
            raise ValueError(f"terminal trip set mismatch: {terminal_path}")
        if Path(str(terminal.get("csv"))).name != manifest_entry["csv"]:
            raise ValueError(f"terminal/input-manifest CSV mismatch: {terminal_path}")
        if terminal.get("stop_reason") in {
                None, "initializing", "running"
        } or str(terminal.get("stop_reason")).startswith("snapshot_"):
            raise ValueError(f"terminal status is not terminal: {terminal_path}")
        terminal_row = _row(
            replicate=replicate,
            campaign=root.name,
            arm=arm,
            checkpoint="terminal",
            nominal_minutes=None,
            status_path=terminal_path,
            status=terminal,
            journal=journal,
            status_sha256=terminal_sha,
            journal_sha256=journal_sha,
            naming_error=naming_error,
        )
        if terminal_row["actual_wall_s"] < previous_wall:
            raise ValueError(
                f"terminal wall chronology regressed: {terminal_path}"
            )
        rows.append(terminal_row)
        validated_file_hashes[terminal_row["status_path"]] = terminal_sha
        validated_file_hashes[terminal_row["journal_path"]] = journal_sha
        files.update((terminal_path, journal))
        iters = Path(str(terminal_path) + ".iters.csv")
        if not iters.is_file():
            raise ValueError(f"terminal trajectory is missing: {iters}")
        trajectory_sha = _validate_trajectory(iters, terminal)
        files.add(iters)
        validated_file_hashes[str(iters.resolve())] = trajectory_sha
        allocations = root / f"{terminal_path.stem}.allocations.tsv"
        if not allocations.is_file():
            raise ValueError(f"allocation log is missing: {allocations}")
        allocation_sha = _validate_allocations(
            allocations, expected_job_id=str(manifest_row["job_id"])
        )
        files.add(allocations)
        validated_file_hashes[str(allocations.resolve())] = allocation_sha
    if expected_trip_ids is None:
        raise ValueError(f"campaign contains no trip identity: {root}")
    return {
        "campaign_dir": str(root),
        "campaign": root.name,
        "replicate": replicate,
        "rows": rows,
        "files": sorted(str(path.resolve()) for path in files),
        "prep": prep,
        "trip_ids": expected_trip_ids,
        "trip_set_sha256": trip_set_sha256(expected_trip_ids),
        "job_ids": job_ids,
        "launch": launch,
        "manifest_entry": manifest_entry,
        "instance_path": str(instance_path),
        "prices_path": str(prices_path),
        "validated_file_hashes": validated_file_hashes,
    }


def validate_historical(path: Path) -> dict:
    status_path = path.expanduser().resolve()
    status, status_sha = _load_object_with_sha(status_path)
    journal, commit, journal_sha = _validate_common_status(
        status_path,
        status,
        expected_commit=HISTORICAL_COMMIT,
        factorial_controls=False,
    )
    if commit != HISTORICAL_COMMIT:
        raise ValueError("historical comparator commit mismatch")
    if status.get("stop_reason") != "wall_limit":
        raise ValueError("historical comparator is not terminal")
    wall_s = float(status.get("wall_s"))
    if not 21.5 <= wall_s / 3600.0 <= 22.5:
        raise ValueError("historical comparator is not approximately 22 hours")
    weight = _metric(status, "route_weight")
    if weight is None or not math.isclose(
            weight, HISTORICAL_WEIGHT, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"historical route weight {weight} differs from comparator "
            f"{HISTORICAL_WEIGHT}"
        )
    artificials = _metric(status, "artificials")
    if artificials != 0.0:
        raise ValueError("historical comparator is not artificial-free")
    return {
        "status_path": str(status_path),
        "status_sha256": status_sha,
        "journal_path": str(journal),
        "journal_sha256": journal_sha,
        "actual_wall_s": wall_s,
        "actual_hours": wall_s / 3600.0,
        "route_weight": weight,
        "objective": _metric(status, "objective"),
        "artificials": artificials,
        "min_reduced_cost": _metric(status, "min_rc"),
        "certified": status.get("certified_rc_optimal") is True,
        "master_sense": "cover",
        "initial_pool": "artificial",
        "trip_ids": status["trip_ids"],
        "trip_set_sha256": trip_set_sha256(status["trip_ids"]),
        "files": [str(status_path), str(journal)],
        "validated_file_hashes": {
            str(status_path): status_sha,
            str(journal): journal_sha,
        },
    }

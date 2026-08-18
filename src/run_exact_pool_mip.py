"""Strict integer master over an exact-pricer column journal.

Differences from the historical final MIP (master.py / run_final_mip.py):
  * binary route variables from the start;
  * NO artificial variables — if some trip has no covering column the script
    says so and refuses, instead of silently paying BIG-M;
  * trip coverage is exact partitioning (== 1) by default (--cover relaxes
    to >= 1, reporting overcovered trips);
  * costs are the exact-pricer's stored route costs (bus 100k + charging),
    so "minimize buses first, charging second" holds lexicographically
    because charging never reaches 1% of one bus.

Usage (Gurobi required, e.g. on Unicorn):

    python run_exact_pool_mip.py --result results/exact_big/<...>.json \
        --timelimit 3600 --out results/exact_big/<...>_mip.json

`--validate-only` loads the pool and reports row coverage plus whether a known
singleton partition is present, without importing Gurobi. Row coverage alone
does not prove that a binary exact partition exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

from durable_io import read_jsonl_records


class PhysicalReplayError(SystemExit):
    """Structured final replay rejection after a solver incumbent exists."""

    def __init__(self, message, *, route_ordinal=None, reason=None, route=None):
        super().__init__(message)
        self.route_ordinal = route_ordinal
        self.reason = reason or message
        self.route = route


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_pool_sha256(routes) -> str:
    route_hashes = [
        hashlib.sha256(json.dumps({
            "trips": route.get("trips"),
            "route_nodes": route.get("route_nodes"),
            "charging_stops": route.get("charging_stops"),
            "cost": route.get("cost"),
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for route in routes
    ]
    return hashlib.sha256(json.dumps(
        route_hashes, separators=(",", ":")
    ).encode()).hexdigest()


def write_new_json(path: Path, payload: dict) -> None:
    """Durably publish one JSON artifact without replacing any path."""

    if os.path.lexists(path):
        raise FileExistsError(f"output already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, sort_keys=True, indent=1)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(parent_fd)
    os.close(parent_fd)


def git_value(*args) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parent,
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_result(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )


def verified_mip_code_identity() -> dict:
    """Bind a submitted solve to one clean reviewed Git commit."""

    expected = os.environ.get("EVSP_EXPECTED_COMMIT")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    require_detached = os.environ.get("EVSP_REQUIRE_DETACHED") == "1"
    enforce = bool(expected or slurm_job_id or require_detached)
    head_result = git_result("rev-parse", "--verify", "HEAD")
    observed = head_result.stdout.strip()
    tracked_result = git_result(
        "status", "--porcelain", "--untracked-files=no"
    )
    symbolic_result = git_result("symbolic-ref", "-q", "HEAD")
    if (head_result.returncode != 0 or len(observed) != 40
            or any(character not in "0123456789abcdef"
                   for character in observed)):
        raise SystemExit("[MIP] solver has no verifiable Git HEAD")
    if slurm_job_id and not expected:
        raise SystemExit("[MIP] submitted solve lacks EVSP_EXPECTED_COMMIT")
    if expected and observed != expected:
        raise SystemExit(
            f"[MIP] solver commit mismatch: expected {expected}, "
            f"found {observed}"
        )
    if tracked_result.returncode != 0:
        raise SystemExit("[MIP] could not verify solver worktree state")
    tracked_status = tracked_result.stdout.strip()
    if enforce and tracked_status:
        raise SystemExit("[MIP] solver checkout has tracked modifications")
    if symbolic_result.returncode == 0:
        branch = symbolic_result.stdout.strip()
        detached = False
    elif symbolic_result.returncode == 1:
        branch = ""
        detached = True
    else:
        raise SystemExit("[MIP] could not verify detached solver HEAD")
    if enforce and (require_detached or slurm_job_id) and not detached:
        raise SystemExit(
            f"[MIP] solver must run detached; found branch {branch}"
        )
    return {
        "expected_commit": expected,
        "observed_commit": observed,
        "branch": branch,
        "detached": detached,
        "tracked_clean": not bool(tracked_status),
        "enforced": enforce,
    }


def resolve_pool_journal(result_path: Path, status: dict) -> Path:
    """Resolve the journal exactly as :func:`load_pool` will read it."""

    # A live status file normally records the journal path directly.  Frozen
    # snapshots, however, are often copied to a timestamped directory (or a
    # release archive) together with the journal.  In that case the recorded
    # absolute Unicorn path may no longer exist, so also look beside the
    # snapshot using the recorded basename.  The final candidate preserves the
    # exact-pricer's historical ``RESULT.json.columns.jsonl`` convention.
    recorded_journal = status.get("columns_journal")
    candidates = []
    is_snapshot = result_path.name.endswith(".snapshot.json")
    recorded_path = Path(recorded_journal) if recorded_journal else None
    if is_snapshot and recorded_path is not None:
        # Prefer the frozen sibling over a recorded live/cluster path that may
        # still exist but no longer represent this snapshot.
        candidates.append(result_path.parent / recorded_path.name)
    if is_snapshot:
        snapshot_stem = result_path.name[: -len(".snapshot.json")]
        candidates.append(result_path.with_name(f"{snapshot_stem}.columns.jsonl"))
    if recorded_journal:
        candidates.append(recorded_path)
        if not is_snapshot:
            candidates.append(result_path.parent / recorded_path.name)
    candidates.append(Path(str(result_path) + ".columns.jsonl"))

    unique_candidates = list(dict.fromkeys(candidates))
    journal_path = next((path for path in unique_candidates if path.exists()), None)
    if journal_path is None:
        tried = "\n  ".join(str(path) for path in unique_candidates)
        raise SystemExit(
            f"{result_path} has no readable column journal. Tried:\n  {tried}\n"
            "The run may predate pool persistence, or its journal was not "
            "copied with the status file.")
    return journal_path


def load_pool(result_path: Path, *, deduplicate=True):
    with open(result_path) as fh:
        status = json.load(fh)

    trips = status.get("trip_ids")
    if trips is None:
        raise SystemExit(f"{result_path} lacks trip_ids; rerun with current code.")
    trips = [int(trip) for trip in trips]
    if len(trips) != len(set(trips)):
        raise SystemExit(f"{result_path} repeats a trip in trip_ids")
    allowed = set(trips)
    journal_path = resolve_pool_journal(result_path, status)
    pool = {}
    validated_records = []
    records = read_jsonl_records(journal_path, repair_trailing=False)
    for ordinal, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            raise SystemExit(
                f"{journal_path} record {ordinal} is not a JSON object"
            )
        route_trips = rec.get("trips")
        if not isinstance(route_trips, list) or not route_trips:
            raise SystemExit(
                f"{journal_path} record {ordinal} has no nonempty trips list"
            )
        if any(
                not isinstance(trip, int) or isinstance(trip, bool)
                for trip in route_trips):
            raise SystemExit(
                f"{journal_path} record {ordinal} has non-integer trips"
            )
        try:
            unique_trips = set(route_trips)
        except TypeError as exc:
            raise SystemExit(
                f"{journal_path} record {ordinal} has unhashable trips"
            ) from exc
        if len(route_trips) != len(unique_trips):
            raise SystemExit(
                f"{journal_path} record {ordinal} repeats a trip"
            )
        unknown = [trip for trip in route_trips if trip not in allowed]
        if unknown:
            raise SystemExit(
                f"{journal_path} record {ordinal} contains trips outside "
                f"the snapshot: {unknown[:15]}"
            )
        try:
            cost = float(rec["cost"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"{journal_path} record {ordinal} has invalid cost"
            ) from exc
        if not math.isfinite(cost):
            raise SystemExit(
                f"{journal_path} record {ordinal} has non-finite cost"
            )
        key = frozenset(route_trips)
        validated_records.append(rec)
        if key not in pool or cost < float(pool[key]["cost"]) - 1e-9:
            pool[key] = rec
    return (
        status,
        list(pool.values()) if deduplicate else validated_records,
        trips,
    )


def deduplicate_pool(routes: list[dict]) -> list[dict]:
    """Keep the cheapest physically admitted record per trip incidence."""

    pool = {}
    for route in routes:
        key = frozenset(route["trips"])
        if (
            key not in pool
            or float(route["cost"]) < float(pool[key]["cost"]) - 1e-9
        ):
            pool[key] = route
    return list(pool.values())


def singleton_partition_indices(routes: list[dict], trips: list[int]) -> list[int]:
    """Return one singleton-column index per trip, or an empty list."""

    by_trip = {}
    trip_set = set(trips)
    for index, route in enumerate(routes):
        route_trips = route.get("trips", [])
        if len(route_trips) == 1 and route_trips[0] in trip_set:
            by_trip[route_trips[0]] = index
    if len(by_trip) != len(trips):
        return []
    return [by_trip[trip] for trip in trips]


def greedy_partition_start_indices(
    routes: list[dict],
    trips: list[int],
    singleton_partition: list[int],
) -> list[int]:
    """Build a deterministic feasible start from disjoint priced routes.

    Start with the guaranteed singleton partition, greedily replace groups of
    singletons by saved multi-trip routes with the largest cost savings, and
    fill every remaining trip with its singleton.  This is only an incumbent
    heuristic; Gurobi remains responsible for optimizing the full pool.
    """

    if not singleton_partition:
        return []
    singleton_by_trip = {
        routes[index]["trips"][0]: index for index in singleton_partition
    }
    candidates = []
    for index, route in enumerate(routes):
        route_trips = route.get("trips", [])
        if len(route_trips) <= 1:
            continue
        singleton_cost = sum(
            routes[singleton_by_trip[trip]]["cost"] for trip in route_trips
        )
        savings = singleton_cost - route["cost"]
        if savings > 1e-9:
            candidates.append((-savings, -len(route_trips), route["cost"], index))
    candidates.sort()

    selected = []
    covered = set()
    for _negative_savings, _negative_length, _cost, index in candidates:
        route_trips = set(routes[index]["trips"])
        if covered.isdisjoint(route_trips):
            selected.append(index)
            covered.update(route_trips)
    selected.extend(singleton_by_trip[trip] for trip in trips if trip not in covered)
    return selected


def validate_injected_route(problem, record, g_kwh, charge_kw, reserve_kwh,
                            horizon_min, arrival_grace_min=1.0,
                            rate_grace_min=0.0, arc_map=None):
    """Replay an injected route against the model graph and pool physics.

    Checks every consecutive arc exists in the restricted adjacency, times
    chain (fixed trip times; charging inside its [cst, cet] window at <= the
    pool's charger power.  A small arrival-time grace accommodates Hastus
    minute rounding, but it does not create extra charging energy),
    and continuous SOC never drops below the reserve. Returns None if valid,
    else a short reason.
    """
    from audit_giro_known_columns import DEPOT

    arc = arc_map
    if arc is None:
        arc = {}
        for u, arcs in problem.adjacency.items():
            for v, travel, dh, _kind in arcs:
                arc[(u, v)] = (travel, dh)

    nodes = record.get("route_nodes") or []
    if len(nodes) < 3:
        return "route_nodes missing"
    if nodes[0] != DEPOT or nodes[-1] != DEPOT:
        return "route must start and end at the depot"
    stops = record.get("charging_stops") or {}
    stop_fields = [list(stops.get(name, []))
                   for name in ("stations", "cst", "cet", "kwh")]
    if len({len(values) for values in stop_fields}) != 1:
        return "charging stop fields have different lengths"
    st_list = list(zip(*stop_fields))
    normalized_stops = []
    for station, cst, cet, kwh in st_list:
        try:
            cst_value = float(cst)
            cet_value = float(cet)
            kwh_value = float(kwh)
        except (TypeError, ValueError):
            return f"non-numeric charging stop at {station}"
        if not all(math.isfinite(value)
                   for value in (cst_value, cet_value, kwh_value)):
            return f"non-finite charging stop at {station}"
        if cet_value < cst_value - 1e-6:
            return f"charging stop at {station} ends before it starts"
        if cst_value < -1e-6:
            return f"charging stop at {station} starts before the horizon"
        if cet_value > float(horizon_min) + 1e-6:
            return f"charging stop at {station} ends past the horizon"
        if kwh_value < -1e-6:
            return f"negative charge {kwh_value:.1f} kWh at {station}"
        normalized_stops.append(
            (station, cst_value, cet_value, kwh_value)
        )
    st_list = normalized_stops
    si = 0
    soc = float(g_kwh)
    time_now = None  # depot departure is flexible
    prev = nodes[0]
    for position, v in enumerate(nodes[1:], start=1):
        is_last = position == len(nodes) - 1
        key = (prev, v) if not (isinstance(prev, str) and isinstance(v, str)
                                and prev == v) else None
        if key is not None and key not in arc:
            return f"missing model arc {prev}->{v}"
        travel, dh = arc.get(key, (0.0, 0.0))
        soc -= dh
        if soc < reserve_kwh - 1e-6:
            return f"SOC {soc:.1f} < reserve before {v}"
        if isinstance(v, int):
            arrive_earliest = (time_now + travel) if time_now is not None else None
            if arrive_earliest is not None and                     arrive_earliest > problem.start_min[v] + 1e-6:
                return f"arrives {arrive_earliest:.0f} after trip {v} start"
            soc -= problem.trip_energy[v]
            if soc < reserve_kwh - 1e-6:
                return f"SOC {soc:.1f} < reserve after trip {v}"
            time_now = problem.end_min[v]
        elif is_last:
            if time_now is not None:
                time_now += travel
        else:
            if si < len(st_list) and st_list[si][0] == v:
                _stn, cst, cet, kwh = st_list[si]
                si += 1
                if (time_now is not None
                        and time_now + travel
                        > float(cst) + arrival_grace_min + 1e-6):
                    return f"reaches {v} at {time_now + travel:.0f} after cst {cst}"
                effective_start = (
                    max(float(cst), time_now + travel)
                    if time_now is not None else float(cst)
                )
                window = max(
                    0.0, float(cet) - effective_start
                ) + rate_grace_min
                if kwh > window * charge_kw / 60.0 + 1e-6:
                    return (f"charge {kwh:.1f} kWh exceeds {charge_kw:.0f} kW "
                            f"in {window:.0f} min at {v}")
                if soc + float(kwh) > float(g_kwh) + 1e-6:
                    return (f"charge at {v} raises SOC to "
                            f"{soc + float(kwh):.1f} > capacity {g_kwh:.1f}")
                soc += float(kwh)
                time_now = float(cet)
            elif time_now is not None:
                # A station visit without a recorded stop is a pure-wait
                # waypoint.  At minimum its inbound travel must advance time.
                time_now += travel
        prev = v
    if si != len(st_list):
        return f"{len(st_list) - si} charging stop record(s) were not consumed"
    if time_now is not None and time_now > horizon_min + 1e-6:
        return f"ends at {time_now:.0f} past horizon"
    return None


def charging_stop_arrivals(problem, record) -> list[float]:
    """Return actual earliest arrival for each persisted charging stop."""

    arc = {
        (source, target): (float(travel), float(deadhead))
        for source, arcs in problem.adjacency.items()
        for target, travel, deadhead, _kind in arcs
    }
    nodes = list(record.get("route_nodes", record.get("route", [])) or [])
    stops = record.get("charging_stops") or {}
    stations = list(stops.get("stations", []))
    cets = list(stops.get("cet", []))
    arrivals = []
    stop_index = 0
    time_now = None
    previous = nodes[0]
    for position, node in enumerate(nodes[1:], start=1):
        is_last = position == len(nodes) - 1
        key = None if (
            isinstance(previous, str)
            and isinstance(node, str)
            and previous == node
        ) else (previous, node)
        travel = arc.get(key, (0.0, 0.0))[0]
        if isinstance(node, int) and not isinstance(node, bool):
            time_now = float(problem.end_min[node])
        elif not is_last and stop_index < len(stations):
            if stations[stop_index] == node:
                arrival = (
                    float(time_now + travel)
                    if time_now is not None else float(travel)
                )
                arrivals.append(arrival)
                time_now = float(cets[stop_index])
                stop_index += 1
            elif time_now is not None:
                time_now += travel
        previous = node
    if stop_index != len(stations):
        raise ValueError("charging stop arrival mapping is incomplete")
    return arrivals


def prepare_strict_partition_pool(
    status,
    routes,
    *,
    data_dir=None,
    reference_data_dir=None,
):
    """Replay or deterministically map every raw pool column before MIP use.

    Stored master costs remain the expanded-grid costs.  Repaired continuous
    schedules and their separately computed costs are provenance only; they do
    not expand the scope of the existing reduced-cost certificate.
    """

    from audit_giro_known_columns import HORIZON_MIN, build_problem
    from config import CHARGING_STATIONS
    from expanded_path_realization import (
        BLOCK_SCHEDULE_SCHEMA,
        _arc_map,
        realize_expanded_path,
        realized_costs,
        validate_continuous_charging_blocks,
    )
    from utils_v2 import load_station_hourly_prices

    data_dir = Path(
        data_dir
        if data_dir is not None
        else Path(__file__).resolve().parent.parent / "data"
    ).resolve()
    reference_data_dir = Path(
        reference_data_dir if reference_data_dir is not None else data_dir
    ).resolve()
    instance_path = (data_dir / str(status["csv"])).resolve()
    tariff_path = (data_dir / str(status["prices_csv"])).resolve()
    try:
        tariff_path.relative_to(data_dir)
    except ValueError as exc:
        raise SystemExit(
            "[MIP] physical pool tariff escapes data/"
        ) from exc
    reference_path = reference_data_dir / "Ref_dict.csv"
    deadhead_path = reference_data_dir / "par_ref_dhd.csv"
    provenance = status.get("provenance") or {}
    if (
        not instance_path.is_file()
        or not tariff_path.is_file()
        or not reference_path.is_file()
        or not deadhead_path.is_file()
    ):
        raise SystemExit(
            "[MIP] physical pool preparation reference data missing"
        )
    observed_reference_sha = file_sha256(reference_path)
    observed_deadhead_sha = file_sha256(deadhead_path)
    input_hashes_before = {
        "instance_sha256": file_sha256(instance_path),
        "prices_sha256": file_sha256(tariff_path),
        "reference_sha256": observed_reference_sha,
        "deadhead_sha256": observed_deadhead_sha,
    }
    source_reference_hashes_bound = (
        isinstance(provenance.get("reference_sha256"), str)
        and isinstance(provenance.get("deadhead_sha256"), str)
    )
    if (
        input_hashes_before["instance_sha256"]
        != provenance.get("instance_sha256")
        or input_hashes_before["prices_sha256"]
        != provenance.get("prices_sha256")
        or (
            source_reference_hashes_bound
            and (
                observed_reference_sha
                != provenance["reference_sha256"]
                or observed_deadhead_sha
                != provenance["deadhead_sha256"]
            )
        )
    ):
        raise SystemExit(
            "[MIP] physical pool preparation input hash mismatch"
        )
    problem = build_problem(
        data_dir,
        status["csv"],
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=reference_data_dir,
    )
    arc_map = _arc_map(problem)
    prices = load_station_hourly_prices(
        tariff_path,
        CHARGING_STATIONS,
    )
    g_kwh = float(status["g_kwh"])
    charge_kw = float(status["charge_kw"])
    reserve_kwh = float(status["min_soc_frac"]) * g_kwh
    soc_step = float(status["soc_step"])
    block_min = int(status["block_min"])
    accepted = []
    valid_hashes = []
    repaired_hashes = []
    rejected_hashes = []
    rejected = []
    persisted_block_count = 0
    persisted_block_json_bytes = 0
    for index, route in enumerate(routes):
        route_identity = {
            "trips": route.get("trips"),
            "route_nodes": route.get("route_nodes"),
            "charging_stops": route.get("charging_stops"),
            "cost": route.get("cost"),
        }
        route_sha = hashlib.sha256(json.dumps(
            route_identity, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        route_nodes = route.get(
            "route_nodes", route.get("route", [])
        )
        node_trips = [
            node for node in route_nodes
            if isinstance(node, int) and not isinstance(node, bool)
        ]
        reason = (
            "trip incidence differs from route nodes"
            if node_trips != list(route.get("trips") or [])
            else validate_injected_route(
                problem, route, g_kwh, charge_kw, reserve_kwh,
                HORIZON_MIN, arc_map=arc_map,
            )
        )
        persisted_blocks = route.get(
            "continuous_realized_charging_blocks"
        )
        persisted_physical = route.get("physical_realization") or {}
        if (
            persisted_blocks is not None
            or persisted_physical.get(
                "continuous_realized_charging_blocks_sha256"
            ) is not None
        ):
            try:
                persisted_validation = validate_continuous_charging_blocks(
                    route,
                    persisted_blocks,
                    station_prices=prices,
                    charge_kw=charge_kw,
                    expected_continuous_cost=route.get(
                        "continuous_realized_cost"
                    ),
                )
                if (
                    persisted_physical.get(
                        "continuous_realized_charging_blocks_schema"
                    ) != BLOCK_SCHEDULE_SCHEMA
                    or persisted_validation["block_schedule_sha256"]
                    != persisted_physical.get(
                        "continuous_realized_charging_blocks_sha256"
                    )
                ):
                    raise ValueError("persisted block hash/schema mismatch")
            except (KeyError, TypeError, ValueError) as exc:
                rejected_hashes.append(route_sha)
                if len(rejected) < 100:
                    rejected.append({
                        "route_index": index,
                        "route_sha256": route_sha,
                        "trips": route.get("trips"),
                        "recorded_reason": reason,
                        "realization_reason":
                            f"persisted block validation failed: {exc}",
                    })
                continue
        realized, detail = realize_expanded_path(
            problem,
            route,
            g_kwh=g_kwh,
            charge_kw=charge_kw,
            reserve_kwh=reserve_kwh,
            soc_step=soc_step,
            block_min=block_min,
            arc_map=arc_map,
        )
        realized_reason = (
            validate_injected_route(
                problem, realized, g_kwh, charge_kw, reserve_kwh,
                HORIZON_MIN, arc_map=arc_map,
            )
            if realized is not None else detail.get("reason")
        )
        if realized is None or realized_reason is not None:
            rejected_hashes.append(route_sha)
            if len(rejected) < 100:
                rejected.append({
                    "route_index": index,
                    "route_sha256": route_sha,
                    "trips": route.get("trips"),
                    "recorded_reason": reason,
                    "realization_reason": realized_reason,
                })
            continue
        costs = realized_costs(
            realized, detail["mapping"], station_prices=prices
        )
        if persisted_blocks is not None and (
            costs["continuous_realized_charging_blocks"]
            != persisted_blocks
            or costs["continuous_realized_charging_blocks_sha256"]
            != persisted_physical.get(
                "continuous_realized_charging_blocks_sha256"
            )
        ):
            rejected_hashes.append(route_sha)
            if len(rejected) < 100:
                rejected.append({
                    "route_index": index,
                    "route_sha256": route_sha,
                    "trips": route.get("trips"),
                    "recorded_reason": reason,
                    "realization_reason":
                        "persisted blocks differ from deterministic mapping",
                })
            continue
        if not math.isclose(
            costs["stored_expanded_grid_cost"],
            costs["recomputed_expanded_grid_cost"],
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            rejected_hashes.append(route_sha)
            if len(rejected) < 100:
                rejected.append({
                    "route_index": index,
                    "route_sha256": route_sha,
                    "trips": route.get("trips"),
                    "recorded_reason": reason,
                    "realization_reason":
                        "stored expanded-grid cost mismatch",
                })
            continue
        realized["cost"] = float(route["cost"])
        realized["master_cost_semantics"] = "expanded_grid_cost"
        realized["expanded_grid_cost"] = float(route["cost"])
        realized["continuous_realized_cost"] = costs[
            "continuous_realized_cost"
        ]
        realized["continuous_realized_charging_blocks"] = costs[
            "continuous_realized_charging_blocks"
        ]
        realized["continuous_realized_charging_blocks_json_bytes"] = len(
            json.dumps(
                realized["continuous_realized_charging_blocks"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        persisted_block_count += len(
            realized["continuous_realized_charging_blocks"]
        )
        persisted_block_json_bytes += len(json.dumps(
            realized["continuous_realized_charging_blocks"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode())
        realized["physical_realization"] = realized.pop(
            "continuous_realization"
        )
        realized["physical_realization"].update({
            "status": (
                "valid_as_recorded_mapped"
                if reason is None else "deterministically_repaired"
            ),
            "recorded_route_sha256": route_sha,
            "recorded_replay_reason": reason,
            "continuous_realized_charging_blocks_sha256": costs[
                "continuous_realized_charging_blocks_sha256"
            ],
            "continuous_realized_charging_blocks_schema":
                BLOCK_SCHEDULE_SCHEMA,
            "costs": {
                key: value for key, value in costs.items()
                if key not in {
                    "continuous_realized_charging_blocks",
                    "continuous_realized_charging_blocks_sha256",
                }
            },
        })
        accepted.append(realized)
        if reason is None:
            valid_hashes.append(route_sha)
        else:
            repaired_hashes.append(
                realized["physical_realization"]["mapping_sha256"]
            )
    deduplicated = deduplicate_pool(accepted)
    ordered_pool_hash = ordered_pool_sha256(deduplicated)
    input_hashes_after = {
        "instance_sha256": file_sha256(instance_path),
        "prices_sha256": file_sha256(tariff_path),
        "reference_sha256": file_sha256(reference_path),
        "deadhead_sha256": file_sha256(deadhead_path),
    }
    if input_hashes_after != input_hashes_before:
        raise SystemExit(
            "[MIP] physical pool preparation inputs changed during replay"
        )
    audit = {
        "schema": "evsp-dr-strict-pool-physical-gate-v1",
        "total_columns": len(routes),
        "accepted_columns": len(accepted),
        "mip_unique_accepted_columns": len(deduplicated),
        "mip_ordered_pool_sha256": ordered_pool_hash,
        "valid_as_recorded": len(valid_hashes),
        "deterministically_repaired": len(repaired_hashes),
        "rejected_columns": len(rejected_hashes),
        "valid_set_sha256": hashlib.sha256(json.dumps(
            sorted(valid_hashes), separators=(",", ":")
        ).encode()).hexdigest(),
        "repaired_set_sha256": hashlib.sha256(json.dumps(
            sorted(repaired_hashes), separators=(",", ":")
        ).encode()).hexdigest(),
        "rejected_set_sha256": hashlib.sha256(json.dumps(
            sorted(rejected_hashes), separators=(",", ":")
        ).encode()).hexdigest(),
        "rejected_samples": rejected,
        "master_cost_semantics": "expanded_grid_cost_unchanged",
        "pricing_certificate_scope":
            (
                "conservative_expanded_grid_model_only"
                if status.get("certified_rc_optimal") is True
                else "not_certified"
            ),
        "source_pricing_certified":
            status.get("certified_rc_optimal") is True,
        "continuous_cost_pricing_certified": False,
        "persisted_charging_block_count": persisted_block_count,
        "persisted_charging_block_payload_bytes":
            persisted_block_json_bytes,
        "input_hashes": input_hashes_before,
        "source_reference_hashes_bound":
            source_reference_hashes_bound,
    }
    return accepted, audit


def merge_extra_routes(
    routes, trips, extra_paths, prices_csv, status=None, *,
    data_dir=None, reference_data_dir=None,
):
    """Merge runner-format route dicts (e.g. MATCHING covers) into the pool.

    Pool MIPs at k>=8 stall near singleton incumbents because CG pools lack an
    integral backbone; constructive covers supply one. Costs are recomputed
    with the exact master cost function, and EVERY candidate is replayed
    against the model graph under the POOL'S physics (g_kwh, charge_kw,
    reserve) — routes that fail are refused, not warned about.
    """

    from audit_giro_known_columns import HORIZON_MIN, build_problem
    from config import (BUS_COST_KX, CHARGE_RATE_KW, CHARGE_START_COST,
                        CHARGING_STATIONS)
    from expanded_path_realization import (
        BLOCK_SCHEDULE_SCHEMA,
        blocks_from_continuous_stops,
        validate_continuous_charging_blocks,
    )
    from utils_v2 import calculate_truck_route_cost_accurate, load_station_hourly_prices

    status = status or {}
    g_kwh = float(status.get("g_kwh", 300.0))
    charge_kw = float(status.get("charge_kw", CHARGE_RATE_KW))
    reserve_kwh = float(status.get("min_soc_frac", 0.0)) * g_kwh
    data_dir = Path(
        data_dir
        if data_dir is not None
        else Path(__file__).resolve().parent.parent / "data"
    ).resolve()
    reference_data_dir = Path(
        reference_data_dir if reference_data_dir is not None else data_dir
    ).resolve()
    problem = build_problem(
        data_dir, status["csv"],
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=reference_data_dir,
    )
    price_name = (
        str(prices_csv) if prices_csv else "hourly_prices_flat.csv"
    )
    prices = load_station_hourly_prices(data_dir / price_name, CHARGING_STATIONS)
    depot_curve = prices.get("PARX") or next(iter(prices.values()))

    pool = {frozenset(r["trips"]): r for r in routes}
    trip_set = set(trips)
    merged = 0
    rejected = 0
    for path in extra_paths:
        with open(path) as fh:
            payload = json.load(fh)
        for route in payload.get("routes", []):
            route_trips = [n for n in route.get("route", []) if isinstance(n, int)]
            if not route_trips or not set(route_trips) <= trip_set:
                continue
            cost = calculate_truck_route_cost_accurate(
                route, BUS_COST_KX, depot_curve,
                charge_rate_kw=charge_kw,
                station_hourly_prices=prices,
                charge_start_cost=CHARGE_START_COST,
            )
            key = frozenset(route_trips)
            candidate = {
                "trips": route_trips,
                "route_nodes": route.get("route", []),
                "charging_stops": route.get("charging_stops", {}),
            }
            reason = validate_injected_route(
                problem, candidate, g_kwh, charge_kw, reserve_kwh, HORIZON_MIN)
            if reason is not None:
                rejected += 1
                if rejected <= 5:
                    print(f"[MIP] REJECTED injected route ({len(route_trips)} "
                          f"trips): {reason}")
                continue
            if key not in pool or cost < pool[key]["cost"] - 1e-9:
                candidate_record = {
                    "trips": route_trips,
                    "cost": cost,
                    "route_nodes": route.get("route", []),
                    "charging_stops": route.get("charging_stops", {}),
                    "deadhead_kwh": route.get("deadhead_kwh", 0.0),
                    "charges_started": len(
                        (route.get("charging_stops") or {}).get("stations", [])),
                    "found_iter": 0,
                    "origin": f"extra:{path.name[:40]}",
                }
                blocks = blocks_from_continuous_stops(
                    candidate_record,
                    station_prices=prices,
                    charge_kw=charge_kw,
                    earliest_start_by_stop=charging_stop_arrivals(
                        problem, candidate_record
                    ),
                )
                block_validation = validate_continuous_charging_blocks(
                    candidate_record,
                    blocks,
                    station_prices=prices,
                    charge_kw=charge_kw,
                    expected_continuous_cost=cost,
                )
                candidate_record.update({
                    "master_cost_semantics":
                        "continuous_realized_cost",
                    "continuous_realized_cost": cost,
                    "continuous_realized_charging_blocks": blocks,
                    "continuous_realized_charging_blocks_json_bytes":
                        len(json.dumps(
                            blocks, sort_keys=True,
                            separators=(",", ":"),
                        ).encode()),
                    "physical_realization": {
                        "status": "validated_continuous_injection",
                        "continuous_realized_charging_blocks_sha256":
                            block_validation["block_schedule_sha256"],
                        "continuous_realized_charging_blocks_schema":
                            BLOCK_SCHEDULE_SCHEMA,
                        "continuous_cost_pricing_certified": False,
                    },
                })
                pool[key] = candidate_record
                merged += 1
    print(f"[MIP] merged {merged} extra route(s), REJECTED {rejected} as "
          f"infeasible under pool physics (G={g_kwh:.0f} kWh, "
          f"{charge_kw:.0f} kW, reserve {reserve_kwh:.0f}) from "
          f"{len(extra_paths)} file(s)")
    return list(pool.values())


def merge_validated_partition_start(
    routes,
    trips,
    partition_path,
    prices_csv,
    status=None,
    *,
    data_dir=None,
    reference_data_dir=None,
    preserve_expanded_grid_cost=False,
):
    """Merge and select one explicitly supplied exact-partition start.

    Unlike ``--extra-routes``, this path is fail-closed: every supplied route
    must be a real route under the pool's physics, and the supplied routes
    together must cover every pool trip exactly once. Supplied routes are
    appended as auditable columns and assigned directly as the MIP start;
    duplicate incidences are retained and counted rather than silently
    replacing either cost realization.
    """

    from audit_giro_known_columns import HORIZON_MIN, build_problem
    from config import (BUS_COST_KX, CHARGE_START_COST,
                        CHARGING_STATIONS)
    from expanded_path_realization import (
        BLOCK_SCHEDULE_SCHEMA,
        blocks_from_continuous_stops,
        validate_continuous_charging_blocks,
    )
    from utils_v2 import (
        calculate_truck_route_cost_accurate,
        load_station_hourly_prices,
    )

    path = Path(partition_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"[MIP] initial partition source is missing: {path}")
    raw = path.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"[MIP] initial partition is not valid JSON: {path}"
        ) from exc
    supplied = payload.get("routes") if isinstance(payload, dict) else None
    if not isinstance(supplied, list) or not supplied:
        raise SystemExit(
            f"[MIP] initial partition must contain a nonempty routes list: {path}"
        )

    status = status or {}
    required = (
        "csv", "prices_csv", "soc_step", "block_min",
        "g_kwh", "charge_kw", "min_soc_frac",
    )
    missing = [field for field in required if status.get(field) is None]
    if missing:
        raise SystemExit(
            f"[MIP] pool snapshot lacks required start-validation fields: "
            f"{missing}"
        )
    if prices_csv is not None and str(prices_csv) != str(status["prices_csv"]):
        raise SystemExit(
            "[MIP] requested start-validation prices differ from the pool "
            "snapshot"
        )
    try:
        g_kwh = float(status["g_kwh"])
        charge_kw = float(status["charge_kw"])
        reserve_frac = float(status["min_soc_frac"])
        soc_step = float(status["soc_step"])
        block_min = float(status["block_min"])
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "[MIP] pool snapshot has non-numeric physics"
        ) from exc
    if (not all(math.isfinite(value) for value in (
            g_kwh, charge_kw, reserve_frac, soc_step, block_min))
            or g_kwh <= 0.0 or charge_kw <= 0.0
            or soc_step <= 0.0 or block_min <= 0.0
            or not 0.0 <= reserve_frac <= 1.0):
        raise SystemExit("[MIP] pool snapshot has invalid or non-finite physics")
    reserve_kwh = reserve_frac * g_kwh

    provenance = status.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit(
            "[MIP] pool snapshot lacks hashed input provenance"
        )
    data_dir = Path(
        data_dir
        if data_dir is not None
        else Path(__file__).resolve().parent.parent / "data"
    ).resolve()
    reference_data_dir = Path(
        reference_data_dir if reference_data_dir is not None else data_dir
    ).resolve()

    def verified_data_file(relative, hash_field, label):
        expected_hash = provenance.get(hash_field)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise SystemExit(
                f"[MIP] pool snapshot lacks {hash_field} provenance"
            )
        candidate = (data_dir / str(relative)).resolve()
        try:
            candidate.relative_to(data_dir)
        except ValueError as exc:
            raise SystemExit(
                f"[MIP] pool {label} escapes the data directory: {relative}"
            ) from exc
        if not candidate.is_file():
            raise SystemExit(
                f"[MIP] pool {label} is missing: {candidate}"
            )
        actual_hash = file_sha256(candidate)
        if actual_hash != expected_hash:
            raise SystemExit(
                f"[MIP] pool {label} hash mismatch: expected "
                f"{expected_hash}, found {actual_hash}"
            )
        return candidate

    instance_path = verified_data_file(
        status["csv"], "instance_sha256", "instance"
    )
    prices_path = verified_data_file(
        status["prices_csv"], "prices_sha256", "prices"
    )
    problem = build_problem(
        data_dir,
        str(instance_path.relative_to(data_dir)),
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=reference_data_dir,
    )
    trip_set = set(trips)
    if list(problem.trips) != list(trips):
        raise SystemExit(
            "[MIP] initial partition validation reconstructed a different "
            "ordered trip set from the pool snapshot"
        )
    for trip in problem.trips:
        values = (
            float(problem.start_min[trip]),
            float(problem.end_min[trip]),
            float(problem.trip_energy[trip]),
        )
        if (not all(math.isfinite(value) for value in values)
                or values[2] < 0.0 or values[1] < values[0]):
            raise SystemExit(
                f"[MIP] reconstructed instance has invalid trip data for {trip}"
            )
    prices = load_station_hourly_prices(
        prices_path, CHARGING_STATIONS
    )
    depot_curve = prices.get("PARX") or next(iter(prices.values()))
    arc_deadhead = {}
    for node, arcs in problem.adjacency.items():
        for successor, travel, deadhead_kwh, _kind in arcs:
            travel = float(travel)
            deadhead_kwh = float(deadhead_kwh)
            if (not math.isfinite(travel)
                    or not math.isfinite(deadhead_kwh)
                    or travel < 0.0 or deadhead_kwh < 0.0):
                raise SystemExit(
                    f"[MIP] reconstructed instance has invalid arc data "
                    f"{node}->{successor}"
                )
            arc_deadhead[(node, successor)] = deadhead_kwh

    validated = []
    counts = Counter()
    for ordinal, route in enumerate(supplied, start=1):
        if not isinstance(route, dict):
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} is not a JSON object"
            )
        nodes = route.get("route")
        if nodes is None:
            nodes = route.get("route_nodes")
        if not isinstance(nodes, list):
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} has no route nodes"
            )
        route_trips = [
            node for node in nodes
            if isinstance(node, int) and not isinstance(node, bool)
        ]
        if not route_trips:
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} contains no trips"
            )
        if len(route_trips) != len(set(route_trips)):
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} repeats a trip"
            )
        unknown = sorted(set(route_trips) - trip_set)
        if unknown:
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} contains trips "
                f"outside the pool instance: {unknown[:15]}"
            )

        candidate = {
            "trips": route_trips,
            "route_nodes": nodes,
            "charging_stops": route.get("charging_stops", {}),
        }
        reason = validate_injected_route(
            problem,
            candidate,
            g_kwh,
            charge_kw,
            reserve_kwh,
            HORIZON_MIN,
        )
        if reason is not None:
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} failed physical "
                f"validation: {reason}"
            )
        recomputed_deadhead_kwh = 0.0
        for left, right in zip(nodes, nodes[1:]):
            if isinstance(left, str) and isinstance(right, str) and left == right:
                continue
            try:
                recomputed_deadhead_kwh += arc_deadhead[(left, right)]
            except KeyError as exc:
                # ``validate_injected_route`` should already have refused it;
                # keep pricing independently fail-closed.
                raise SystemExit(
                    f"[MIP] initial partition route {ordinal} has no "
                    f"deadhead-energy arc {left}->{right}"
                ) from exc
        pricing_route = dict(route)
        pricing_route["route"] = nodes
        pricing_route["route_nodes"] = nodes
        pricing_route["deadhead_kwh"] = recomputed_deadhead_kwh
        cost = calculate_truck_route_cost_accurate(
            pricing_route,
            BUS_COST_KX,
            depot_curve,
            charge_rate_kw=charge_kw,
            station_hourly_prices=prices,
            charge_start_cost=CHARGE_START_COST,
        )
        if not math.isfinite(float(cost)):
            raise SystemExit(
                f"[MIP] initial partition route {ordinal} has non-finite cost"
            )
        counts.update(route_trips)
        validated_record = {
            **candidate,
            "cost": float(cost),
            "deadhead_kwh": recomputed_deadhead_kwh,
            "charges_started": len(
                (route.get("charging_stops") or {}).get("stations", [])
            ),
            "found_iter": 0,
            "origin": f"initial_partition:{path.name[:40]}",
        }
        blocks = blocks_from_continuous_stops(
            validated_record,
            station_prices=prices,
            charge_kw=charge_kw,
            earliest_start_by_stop=charging_stop_arrivals(
                problem, validated_record
            ),
        )
        block_validation = validate_continuous_charging_blocks(
            validated_record,
            blocks,
            station_prices=prices,
            charge_kw=charge_kw,
            expected_continuous_cost=cost,
        )
        master_cost = float(cost)
        master_cost_semantics = "continuous_realized_cost"
        if preserve_expanded_grid_cost:
            expanded_cost = route.get("expanded_grid_cost")
            if (
                route.get("master_cost_semantics") != "expanded_grid_cost"
                or expanded_cost is None
                or not math.isclose(
                    float(expanded_cost),
                    float(block_validation["recomputed_expanded_grid_cost"]),
                    rel_tol=1e-10,
                    abs_tol=1e-6,
                )
            ):
                raise SystemExit(
                    "[MIP] verified expanded-grid initial route cost mismatch"
                )
            master_cost = float(expanded_cost)
            master_cost_semantics = "expanded_grid_cost"
        validated_record["cost"] = master_cost
        validated_record.update({
            "master_cost_semantics": master_cost_semantics,
            "continuous_realized_cost": float(cost),
            "expanded_grid_cost": (
                master_cost if preserve_expanded_grid_cost else None
            ),
            "continuous_realized_charging_blocks": blocks,
            "continuous_realized_charging_blocks_json_bytes":
                len(json.dumps(
                    blocks, sort_keys=True, separators=(",", ":"),
                ).encode()),
            "physical_realization": {
                "status": "validated_continuous_injection",
                "continuous_realized_charging_blocks_sha256":
                    block_validation["block_schedule_sha256"],
                "continuous_realized_charging_blocks_schema":
                    BLOCK_SCHEDULE_SCHEMA,
                "continuous_cost_pricing_certified": False,
            },
        })
        validated.append(validated_record)

    missing = [trip for trip in trips if counts[trip] == 0]
    repeated = {trip: counts[trip] for trip in trips if counts[trip] > 1}
    if missing or repeated:
        raise SystemExit(
            "[MIP] supplied initial routes are not an exact partition: "
            f"missing={missing[:15]}, repeated={list(repeated.items())[:15]}"
        )

    existing_keys = {frozenset(route["trips"]) for route in routes}
    merged = list(routes)
    start_indices = []
    added = preserved_duplicates = 0
    for record in validated:
        key = frozenset(record["trips"])
        if key not in existing_keys:
            added += 1
        else:
            preserved_duplicates += 1
        start_indices.append(len(merged))
        merged.append(record)
        existing_keys.add(key)
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_sha256:
        raise SystemExit(
            f"[MIP] initial partition source changed while loading: {path}"
        )

    detail = {
        "kind": "validated_exact_partition",
        "source": str(path),
        "source_sha256": source_sha256,
        "validated": True,
        "validated_bus_count": len(start_indices),
        "assigned_mip_start_route_count": len(start_indices),
        "expected_full_objective": float(
            sum(merged[index]["cost"] for index in start_indices)
        ),
        "pool_columns_added": added,
        "pool_columns_replaced": 0,
        "pool_columns_reused": 0,
        "pool_duplicate_incidences_preserved": preserved_duplicates,
        "actual_start_column_hashes": [
            hashlib.sha256(json.dumps(
                {
                    "trips": merged[index]["trips"],
                    "route_nodes": merged[index]["route_nodes"],
                    "charging_stops": merged[index]["charging_stops"],
                    "cost": merged[index]["cost"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()).hexdigest()
            for index in start_indices
        ],
    }
    detail["added_giro_route_count"] = len(start_indices)
    detail["added_giro_route_set_sha256"] = hashlib.sha256(
        json.dumps(
            detail["actual_start_column_hashes"],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    print(
        f"[MIP] validated exact-partition start: {len(start_indices)} buses "
        f"from {path} (added {added}, preserved duplicate incidences "
        f"{preserved_duplicates})"
    )
    return merged, start_indices, detail


def optimize_with_start_audit(
    model,
    GRB,
    *,
    start_supplied: bool,
    progress_observer=None,
) -> dict:
    """Optimize with one composed start-audit/progress callback."""

    audit = {
        "status": "not_observed" if start_supplied else "not_supplied",
        "accepted": None,
        "rejection_observed": False,
        "messages": [],
    }
    callback_api = getattr(GRB, "Callback", None)
    can_capture = (
        callback_api is not None
        and hasattr(callback_api, "MESSAGE")
        and hasattr(callback_api, "MSG_STRING")
        and callable(getattr(model, "cbGet", None))
    )
    if not can_capture and progress_observer is None:
        model.optimize()
        return audit

    def callback(callback_model, where):
        if progress_observer is not None:
            progress_observer(callback_model, where)
        if (
            not start_supplied
            or not can_capture
            or where != callback_api.MESSAGE
        ):
            return
        message = str(
            callback_model.cbGet(callback_api.MSG_STRING)
        ).strip()
        if message and "MIP start" in message:
            audit["messages"].append(message)
            if (
                "Loaded user MIP start with objective" in message
                or "User MIP start produced solution with objective" in message
            ):
                audit["status"] = "accepted"
                audit["accepted"] = True
            elif "User MIP start violates constraint" in message:
                audit["rejection_observed"] = True
                if audit["accepted"] is not True:
                    audit["status"] = "rejected_infeasible"
                    audit["accepted"] = False
            elif "User MIP start did not produce a new incumbent" in message:
                audit["rejection_observed"] = True
                if audit["accepted"] is not True:
                    audit["status"] = "not_loaded_as_incumbent"
                    audit["accepted"] = False

    model.optimize(callback)
    return audit


def finite_solver_value(value):
    """Map Gurobi infinity/sentinel values to JSON null."""

    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or abs(number) >= 1e100:
        return None
    return number


def fleet_bound_proves_incumbent(
    incumbent_buses: int,
    lower_bound: float | None,
    _status_code: int,
) -> bool:
    """Whether a minimization bound certifies the integer fleet incumbent."""

    return (
        lower_bound is not None
        and math.isfinite(lower_bound)
        and math.ceil(lower_bound - 1e-6) >= int(incumbent_buses)
    )


def optimal_scope(*, two_stage: bool, fleet_proven: bool,
                  cost_stage_executed: bool, final_status: int) -> str:
    """Name exactly what, if anything, the final status proves."""

    if not two_stage:
        return "full_pool_objective" if final_status == 2 else "none"
    if not fleet_proven:
        return "none"
    if cost_stage_executed and final_status == 2:
        return "full_pool_lexicographic"
    return "fleet_only"


def validate_final_selected_routes(
    status, trips, selected_routes, *, data_dir=None,
    reference_data_dir=None, physical_pool_audit=None,
) -> None:
    """Rebuild the instance and physically replay every final selected route."""

    from audit_giro_known_columns import HORIZON_MIN, build_problem
    from config import CHARGING_STATIONS
    from expanded_path_realization import (
        BLOCK_SCHEDULE_SCHEMA,
        validate_continuous_charging_blocks,
    )
    from utils_v2 import load_station_hourly_prices

    provenance = status.get("provenance") or {}
    data_dir = Path(
        data_dir
        if data_dir is not None
        else Path(__file__).resolve().parent.parent / "data"
    ).resolve()
    instance_path = (data_dir / str(status.get("csv"))).resolve()
    try:
        instance_path.relative_to(data_dir)
    except ValueError as exc:
        raise SystemExit("[MIP] final replay instance escapes data/") from exc
    expected_hash = provenance.get("instance_sha256")
    if (
        not instance_path.is_file()
        or not isinstance(expected_hash, str)
        or file_sha256(instance_path) != expected_hash
    ):
        raise SystemExit("[MIP] final replay instance hash mismatch")
    tariff_path = (
        data_dir / str(status.get("prices_csv"))
    ).resolve()
    try:
        tariff_path.relative_to(data_dir)
    except ValueError as exc:
        raise SystemExit("[MIP] final replay tariff escapes data/") from exc
    expected_tariff_hash = provenance.get("prices_sha256")
    if (
        not tariff_path.is_file()
        or not isinstance(expected_tariff_hash, str)
        or file_sha256(tariff_path) != expected_tariff_hash
    ):
        raise SystemExit("[MIP] final replay tariff hash mismatch")
    if physical_pool_audit is not None:
        expected_inputs = physical_pool_audit.get("input_hashes") or {}
        reference_root = Path(
            reference_data_dir
            if reference_data_dir is not None else data_dir
        ).resolve()
        observed_inputs = {
            "instance_sha256": file_sha256(instance_path),
            "prices_sha256": file_sha256(tariff_path),
            "reference_sha256": file_sha256(
                reference_root / "Ref_dict.csv"
            ),
            "deadhead_sha256": file_sha256(
                reference_root / "par_ref_dhd.csv"
            ),
        }
        if observed_inputs != expected_inputs:
            raise SystemExit(
                "[MIP] final replay model inputs changed after preparation"
            )
    problem = build_problem(
        data_dir,
        str(instance_path.relative_to(data_dir)),
        max_station_to_trip_wait_min=HORIZON_MIN,
        reference_data_dir=reference_data_dir,
    )
    if list(problem.trips) != list(trips):
        raise SystemExit("[MIP] final replay reconstructed a different trip set")
    prices = None
    try:
        g_kwh = float(status["g_kwh"])
        charge_kw = float(status["charge_kw"])
        reserve_frac = float(status["min_soc_frac"])
        reserve_kwh = reserve_frac * g_kwh
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("[MIP] final replay physics are invalid") from exc
    if (
        not all(math.isfinite(value) for value in (
            g_kwh, charge_kw, reserve_frac, reserve_kwh
        ))
        or g_kwh <= 0.0
        or charge_kw <= 0.0
        or not 0.0 <= reserve_frac <= 1.0
    ):
        raise SystemExit("[MIP] final replay physics are invalid/non-finite")
    counts = Counter()
    for ordinal, route in enumerate(selected_routes, start=1):
        route_trips = list(route.get("trips") or [])
        counts.update(route_trips)
        route_nodes = route.get(
            "route_nodes", route.get("route", [])
        )
        node_trips = [
            node for node in route_nodes
            if isinstance(node, int) and not isinstance(node, bool)
        ]
        if node_trips != route_trips:
            raise SystemExit(
                f"[MIP] final selected route {ordinal} trip incidence "
                "differs from its replayed route nodes"
            )
        candidate = {
            "trips": route_trips,
            "route_nodes": route_nodes,
            "charging_stops": route.get("charging_stops", {}),
        }
        reason = validate_injected_route(
            problem,
            candidate,
            g_kwh,
            charge_kw,
            reserve_kwh,
            HORIZON_MIN,
        )
        if reason is not None:
            message = (
                f"[MIP] final selected route {ordinal} failed physical "
                f"replay: {reason}"
            )
            raise PhysicalReplayError(
                message,
                route_ordinal=ordinal,
                reason=reason,
                route=route,
            )
        blocks = route.get("continuous_realized_charging_blocks")
        physical = route.get("physical_realization") or {}
        expected_block_sha = physical.get(
            "continuous_realized_charging_blocks_sha256"
        )
        if not isinstance(blocks, list) or not isinstance(
            expected_block_sha, str
        ) or physical.get(
            "continuous_realized_charging_blocks_schema"
        ) != BLOCK_SCHEDULE_SCHEMA:
            raise PhysicalReplayError(
                f"[MIP] final selected route {ordinal} lacks persisted "
                "continuous charging blocks",
                route_ordinal=ordinal,
                reason="persisted continuous charging blocks missing",
                route=route,
            )
        continuous_realized_cost = route.get(
            "continuous_realized_cost"
        )
        if (
            not isinstance(continuous_realized_cost, (int, float))
            or isinstance(continuous_realized_cost, bool)
            or not math.isfinite(float(continuous_realized_cost))
        ):
            raise PhysicalReplayError(
                f"[MIP] final selected route {ordinal} lacks a finite "
                "continuous realized cost",
                route_ordinal=ordinal,
                reason="continuous realized cost missing/non-finite",
                route=route,
            )
        if prices is None:
            try:
                prices = load_station_hourly_prices(
                    tariff_path,
                    CHARGING_STATIONS,
                )
            except (KeyError, OSError, ValueError) as exc:
                raise PhysicalReplayError(
                    "[MIP] final replay tariff provenance is invalid",
                    route_ordinal=ordinal,
                    reason=str(exc),
                    route=route,
                ) from exc
        try:
            arrivals = charging_stop_arrivals(problem, route)
            if any(
                float(block["start_min"])
                < arrivals[int(block["stop_index"])] - 1e-6
                for block in blocks
            ):
                raise ValueError(
                    "continuous charging block begins before route arrival"
                )
            block_validation = validate_continuous_charging_blocks(
                route,
                blocks,
                station_prices=prices,
                charge_kw=charge_kw,
                expected_continuous_cost=continuous_realized_cost,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PhysicalReplayError(
                f"[MIP] final selected route {ordinal} has invalid "
                f"continuous charging blocks: {exc}",
                route_ordinal=ordinal,
                reason=str(exc),
                route=route,
            ) from exc
        if block_validation["block_schedule_sha256"] != expected_block_sha:
            raise PhysicalReplayError(
                f"[MIP] final selected route {ordinal} block schedule hash "
                "mismatch",
                route_ordinal=ordinal,
                reason="continuous charging block schedule hash mismatch",
                route=route,
            )
        master_semantics = route.get("master_cost_semantics")
        expected_master_cost = (
            block_validation["recomputed_expanded_grid_cost"]
            if master_semantics == "expanded_grid_cost"
            else block_validation["continuous_realized_cost"]
            if master_semantics == "continuous_realized_cost"
            else None
        )
        if (
            expected_master_cost is None
            or not math.isclose(
                float(route["cost"]),
                float(expected_master_cost),
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
        ):
            raise PhysicalReplayError(
                f"[MIP] final selected route {ordinal} master cost does "
                "not match persisted charging blocks",
                route_ordinal=ordinal,
                reason="master cost/block schedule mismatch",
                route=route,
            )
    if any(counts[trip] != 1 for trip in trips):
        raise SystemExit(
            "[MIP] final selected routes do not cover every trip exactly once"
        )


def publish_rejected_physical_replay(
    out: Path,
    *,
    error: PhysicalReplayError,
    chosen: list[int],
    routes: list[dict],
    status_name: str,
    solver_bound,
    mip_gap,
    source_result_sha256: str,
    source_journal_sha256: str,
    progress_path: Path | None,
    physical_pool_audit: dict | None,
    bound_scope: str,
    code_identity: dict,
    augmentation_sources: list[dict],
    master_cost_semantics: str | None,
    source_pricing_certified: bool,
    source_reference_hashes_bound: bool,
) -> Path:
    """Publish a non-feasible solver-incumbent diagnostic, never a final."""

    diagnostic_path = out.with_name(
        f"{out.name}.rejected_physical_replay.json"
    )
    checkpoint_refs = []
    selected_vector_sha = hashlib.sha256(json.dumps(
        sorted(chosen), separators=(",", ":")
    ).encode()).hexdigest()
    if progress_path is not None and progress_path.is_dir():
        paths = list(progress_path.glob("checkpoint_*.json"))
        paths.extend(
            path for path in (
                progress_path / "latest.json",
                progress_path / "final.json",
            ) if path.is_file()
        )
        for path in sorted(set(paths)):
            if path.is_file():
                checkpoint_payload = json.loads(path.read_text())
                checkpoint_incumbent = (
                    checkpoint_payload.get("incumbent") or {}
                )
                checkpoint_final = (
                    checkpoint_payload.get("final") or {}
                )
                observed_vector = (
                    checkpoint_final.get("route_vector_sha256")
                    if checkpoint_payload.get("kind") == "final"
                    else checkpoint_incumbent.get("route_vector_sha256")
                ) or checkpoint_incumbent.get(
                    "route_vector_sha256"
                ) or checkpoint_final.get("route_vector_sha256")
                checkpoint_refs.append({
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "kind": checkpoint_payload.get("kind"),
                    "route_vector_sha256": observed_vector,
                    "matches_rejected_incumbent": (
                        observed_vector == selected_vector_sha
                        if observed_vector is not None else None
                    ),
                })
    failing_route = error.route or {}
    selected_cost = sum(float(routes[index]["cost"]) for index in chosen)
    selected_identities = [
        hashlib.sha256(json.dumps({
            "trips": routes[index].get("trips"),
            "route_nodes": routes[index].get("route_nodes"),
            "charging_stops": routes[index].get("charging_stops"),
            "expanded_grid_charging_stops":
                routes[index].get("expanded_grid_charging_stops"),
            "expanded_grid_cost": routes[index].get("cost"),
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for index in chosen
    ]
    payload = {
        "schema": "evsp-dr-mip-rejected-physical-replay-v1",
        "kind": "rejected_physical_replay",
        "physical_replay_validated": False,
        "incumbent_usable_as_physical_schedule": False,
        "observational_only": True,
        "source_result_sha256": source_result_sha256,
        "source_journal_sha256": source_journal_sha256,
        "solver_incumbent": {
            "status_name": status_name,
            "selected_route_indices": list(chosen),
            "route_vector_sha256": selected_vector_sha,
            "buses": len(chosen),
            "expanded_grid_objective": selected_cost,
            "bound": solver_bound,
            "bound_scope": bound_scope,
            "gap": mip_gap,
            "selected_route_identity_sha256": selected_identities,
        },
        "failure": {
            "route_ordinal": error.route_ordinal,
            "reason": error.reason,
            "trips": failing_route.get("trips"),
            "route_nodes": failing_route.get(
                "route_nodes", failing_route.get("route")
            ),
            "charging_stops": failing_route.get("charging_stops"),
        },
        "checkpoint_references": checkpoint_refs,
        "augmentation_sources": augmentation_sources,
        "code_identity": code_identity,
        "physical_pool_audit": physical_pool_audit,
        "physical_pool_preparation_wall_s": (
            physical_pool_audit.get("preparation_wall_s")
            if physical_pool_audit else None
        ),
        "master_cost_semantics": master_cost_semantics,
        "pricing_certificate_scope":
            (
                "conservative_expanded_grid_model_only"
                if source_pricing_certified
                and source_reference_hashes_bound
                and master_cost_semantics == "expanded_grid_cost"
                else "source_grid_certificate_missing_reference_hash_binding"
                if source_pricing_certified
                and not source_reference_hashes_bound
                else "source_grid_certificate_does_not_cover_augmented_pool"
                if source_pricing_certified
                else "not_certified"
            ),
        "continuous_cost_pricing_certified": False,
    }
    write_new_json(diagnostic_path, payload)
    return diagnostic_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True,
                        help="Exact-pricer status JSON (with columns_journal).")
    parser.add_argument("--timelimit", type=int, default=3600)
    parser.add_argument("--mipgap", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cover", action="store_true",
                        help="Relax partitioning (==1) to covering (>=1).")
    parser.add_argument(
        "--require-singleton-partition",
        action="store_true",
        help="Refuse a pool that lacks one singleton column per trip.",
    )
    parser.add_argument(
        "--extra-routes",
        type=Path,
        action="append",
        default=None,
        help="Runner-format routes JSON (e.g. a --matching run's "
             "routes_colgen_final_*.json) whose real columns are merged into "
             "the pool before solving. Costs are recomputed with the exact "
             "master cost function. Repeatable.",
    )
    parser.add_argument(
        "--initial-partition-routes",
        type=Path,
        default=None,
        help="Runner-format routes JSON that must validate as one exact "
             "partition under the pool physics. Its routes are merged into "
             "the pool and used explicitly as the complete MIP start.",
    )
    parser.add_argument(
        "--verified-expanded-initial-partition",
        action="store_true",
        help=(
            "Preserve independently verified expanded-grid route costs for "
            "the initial partition instead of converting its master costs "
            "to continuous replay costs."
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Data root containing the hash-bound instance and tariff.",
    )
    parser.add_argument(
        "--reference-data-dir", type=Path,
        default=None,
        help="Optional separate root containing Ref_dict.csv.",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Lexicographic solve: stage 1 minimizes the bus count alone; "
             "stage 1 may use the full --timelimit. Stage 2 minimizes route "
             "cost only if stage 1 proves the fleet early and time remains.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--progress-dir",
        type=Path,
        default=None,
        help="Opt-in directory for atomic observational convergence "
             "checkpoints (not a Gurobi tree restart).",
    )
    args = parser.parse_args(argv)
    if (
        args.verified_expanded_initial_partition
        and args.initial_partition_routes is None
    ):
        parser.error(
            "--verified-expanded-initial-partition requires "
            "--initial-partition-routes"
        )
    end_to_end_started = time.perf_counter()

    if args.out is not None and args.out.resolve() == args.result.resolve():
        parser.error("--out must not overwrite --result")
    code_identity = verified_mip_code_identity()

    # Bind the solve to immutable bytes.  If a caller mistakenly gives a live
    # journal and it changes while being loaded, refuse the ambiguous result.
    source_hashing_started = time.perf_counter()
    with open(args.result) as fh:
        source_status = json.load(fh)
    source_journal = resolve_pool_journal(args.result, source_status)
    source_result_sha256 = file_sha256(args.result)
    source_journal_sha256 = file_sha256(source_journal)
    source_hashing_wall_s = (
        time.perf_counter() - source_hashing_started
    )
    expected_result_sha256 = os.environ.get(
        "EVSP_MIP_EXPECTED_RESULT_SHA256"
    )
    expected_journal_sha256 = os.environ.get(
        "EVSP_MIP_EXPECTED_JOURNAL_SHA256"
    )
    if os.environ.get("SLURM_JOB_ID"):
        if not expected_result_sha256 or not expected_journal_sha256:
            raise SystemExit(
                "[MIP] submitted solve lacks required input hashes"
            )
    if (expected_result_sha256
            and source_result_sha256 != expected_result_sha256):
        raise SystemExit(
            "[MIP] staged result does not match its submission-manifest hash"
        )
    if (expected_journal_sha256
            and source_journal_sha256 != expected_journal_sha256):
        raise SystemExit(
            "[MIP] staged journal does not match its submission-manifest hash"
        )
    expected_initial_partition_sha256 = os.environ.get(
        "EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256"
    )
    if (os.environ.get("SLURM_JOB_ID")
            and args.initial_partition_routes is not None
            and not expected_initial_partition_sha256):
        raise SystemExit(
            "[MIP] submitted initial partition lacks required hash"
        )
    if expected_initial_partition_sha256:
        if args.initial_partition_routes is None:
            raise SystemExit(
                "[MIP] submission expects an initial partition but none was "
                "supplied"
            )
        if (file_sha256(args.initial_partition_routes)
                != expected_initial_partition_sha256):
            raise SystemExit(
                "[MIP] staged initial partition does not match its "
                "submission-manifest hash"
            )
    extra_route_sources = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in (args.extra_routes or [])
    ]
    initial_partition_start = None

    physical_preparation_started = (
        time.perf_counter() if not args.cover else None
    )
    status, routes, trips = load_pool(
        args.result, deduplicate=args.cover
    )
    physical_pool_audit = None
    if not args.cover:
        routes, physical_pool_audit = prepare_strict_partition_pool(
            status,
            routes,
            data_dir=args.data_dir,
            reference_data_dir=args.reference_data_dir,
        )
        routes = deduplicate_pool(routes)
        physical_pool_audit["mip_unique_accepted_columns"] = len(
            routes
        )
        recomputed_pool_hash = ordered_pool_sha256(routes)
        recorded_pool_hash = physical_pool_audit.get(
            "mip_ordered_pool_sha256"
        )
        if (
            recorded_pool_hash is not None
            and recorded_pool_hash != recomputed_pool_hash
        ):
            raise SystemExit(
                "[MIP] physical pool identity changed after preparation"
            )
        physical_pool_audit["mip_ordered_pool_sha256"] = (
            recomputed_pool_hash
        )
        physical_pool_audit.update({
            "base_pool_column_count": len(routes),
            "base_pool_ordered_sha256":
                physical_pool_audit["mip_ordered_pool_sha256"],
            "added_giro_route_count": 0,
            "added_giro_route_set_sha256": hashlib.sha256(
                b"[]"
            ).hexdigest(),
        })
    if (file_sha256(args.result) != source_result_sha256
            or file_sha256(source_journal) != source_journal_sha256):
        raise SystemExit(
            "[MIP] source status or column journal changed while the pool was "
            "being loaded; use an immutable snapshot and retry"
        )
    if args.extra_routes:
        routes = merge_extra_routes(routes, trips, args.extra_routes,
                                    status.get("prices_csv"), status,
                                    data_dir=args.data_dir,
                                    reference_data_dir=args.reference_data_dir)
        for recorded, path in zip(extra_route_sources, args.extra_routes):
            if file_sha256(path) != recorded["sha256"]:
                raise SystemExit(
                    f"[MIP] extra route source changed while loading: {path}"
                )
    if args.initial_partition_routes is not None:
        routes, mip_start, initial_partition_start = (
            merge_validated_partition_start(
                routes,
                trips,
                args.initial_partition_routes,
                status.get("prices_csv"),
                status,
                data_dir=args.data_dir,
                reference_data_dir=args.reference_data_dir,
                preserve_expanded_grid_cost=(
                    args.verified_expanded_initial_partition
                ),
            )
        )
        if physical_pool_audit is not None:
            added_route_count = initial_partition_start.get(
                "added_giro_route_count", len(mip_start)
            )
            added_route_hash = initial_partition_start.get(
                "added_giro_route_set_sha256"
            ) or hashlib.sha256(json.dumps(
                initial_partition_start.get(
                    "actual_start_column_hashes", []
                ),
                separators=(",", ":"),
            ).encode()).hexdigest()
            assigned_route_count = initial_partition_start.get(
                "assigned_mip_start_route_count", len(mip_start)
            )
            physical_pool_audit.update({
                "base_pool_column_count":
                    physical_pool_audit["mip_unique_accepted_columns"],
                "base_pool_ordered_sha256":
                    physical_pool_audit["mip_ordered_pool_sha256"],
                "added_giro_route_count":
                    added_route_count,
                "added_giro_route_set_sha256":
                    added_route_hash,
                "assigned_mip_start_route_count":
                    assigned_route_count,
            })
    coverage = Counter(t for r in routes for t in r["trips"])
    uncovered = [t for t in trips if coverage[t] == 0]
    seed_partition = singleton_partition_indices(routes, trips)
    if initial_partition_start is None:
        mip_start = greedy_partition_start_indices(
            routes, trips, seed_partition
        )
        initial_partition_start = {
            "kind": (
                "greedy_pool_partition" if mip_start else "none"
            ),
            "source": None,
            "source_sha256": None,
            "validated": bool(mip_start),
            "validated_bus_count": len(mip_start) if mip_start else None,
            "expected_full_objective": (
                float(sum(routes[index]["cost"] for index in mip_start))
                if mip_start else None
            ),
        }
    if physical_pool_audit is not None:
        physical_pool_audit["assigned_mip_start_route_count"] = (
            len(mip_start) if mip_start else 0
        )
    if physical_pool_audit is not None:
        physical_pool_audit["post_augmentation_columns"] = len(routes)
        physical_pool_audit["augmented_pool_column_count"] = len(routes)
        physical_pool_audit["augmented_pool_ordered_sha256"] = (
            ordered_pool_sha256(routes)
        )
        pool_semantics = {
            route.get("master_cost_semantics") for route in routes
        }
        physical_pool_audit["post_augmentation_master_cost_semantics"] = (
            next(iter(pool_semantics))
            if len(pool_semantics) == 1
            else "mixed_expanded_grid_and_continuous_augmented_cost"
        )
        physical_pool_audit["preparation_wall_s"] = (
            time.perf_counter() - physical_preparation_started
        )
        print(
            "[MIP] physical pool preparation: "
            f"{physical_pool_audit['valid_as_recorded']} valid recorded, "
            f"{physical_pool_audit['deterministically_repaired']} repaired, "
            f"{physical_pool_audit['rejected_columns']} rejected, "
            f"{physical_pool_audit['preparation_wall_s']:.3f}s"
        )
    pool_master_cost_semantics = (
        (physical_pool_audit or {}).get(
            "post_augmentation_master_cost_semantics"
        )
        or (
            next(iter({
                route.get("master_cost_semantics") for route in routes
            }))
            if len({
                route.get("master_cost_semantics") for route in routes
            }) == 1
            else "mixed_expanded_grid_and_continuous_augmented_cost"
        )
    )
    if (
        not args.two_stage
        and pool_master_cost_semantics
        == "mixed_expanded_grid_and_continuous_augmented_cost"
    ):
        raise SystemExit(
            "[MIP] single-stage objective cannot mix expanded-grid and "
            "continuous augmented route costs"
        )
    print(f"[MIP] pool: {len(routes)} columns over {len(trips)} trips "
          f"(instance {status['csv']}, soc_step={status['soc_step']}, "
          f"certified={status.get('certified_rc_optimal')})")
    print(f"[MIP] coverage: min {min(coverage[t] for t in trips) if not uncovered else 0}"
          f" / median {sorted(coverage[t] for t in trips)[len(trips)//2]}"
          f" / max {max(coverage[t] for t in trips)} columns per trip")
    if uncovered:
        raise SystemExit(f"[MIP] {len(uncovered)} trips have NO covering column "
                         f"({uncovered[:15]}...): pool is MIP-infeasible without "
                         "artificials — extend the CG run first.")
    if seed_partition:
        print(f"[MIP] strict-feasibility seed: {len(seed_partition)} exact "
              "singleton columns (one per trip)")
        if initial_partition_start["kind"] == "greedy_pool_partition":
            print(f"[MIP] greedy feasible MIP start: {len(mip_start)} buses")
    else:
        print("[MIP] WARNING: row coverage is complete, but the pool has no "
              "known integer partition seed")
        if args.require_singleton_partition:
            raise SystemExit(
                "[MIP] singleton partition required; prepare this legacy pool "
                "with prepare_exact_pool_mip.py before submission"
            )
    if args.validate_only:
        if initial_partition_start["kind"] == "validated_exact_partition":
            print("[MIP] validate-only: supplied start is a physically valid "
                  "exact partition. OK.")
        elif seed_partition:
            print("[MIP] validate-only: strict partition feasibility is "
                  "guaranteed by the singleton seed. OK.")
        else:
            print("[MIP] validate-only: row coverage only; strict partition "
                  "feasibility has NOT been established.")
        return 0

    out = args.out or Path(str(args.result).replace(".json", "_mip.json"))
    rejected_out = out.with_name(
        f"{out.name}.rejected_physical_replay.json"
    )
    if os.path.lexists(out) or os.path.lexists(rejected_out):
        raise SystemExit(f"[MIP] refusing to overwrite existing output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    progress_path = (
        args.progress_dir.expanduser().resolve()
        if args.progress_dir is not None else None
    )
    if progress_path is not None:
        protected = {
            args.result.resolve(),
            source_journal.resolve(),
            out.resolve(),
            *(
                path.resolve()
                for path in (args.extra_routes or [])
            ),
        }
        if args.initial_partition_routes is not None:
            protected.add(args.initial_partition_routes.resolve())
        if progress_path in protected:
            parser.error("--progress-dir aliases a protected input/output")
        if progress_path.exists():
            parser.error("--progress-dir already exists; choose a new path")

    import gurobipy as gp
    from gurobipy import GRB
    from config import BUS_COST_KX
    from mip_convergence import (
        GurobiProgressObserver,
        MIPProgressRecorder,
        TerminationRequest,
    )

    # Gurobi's documented optimization status codes are stable integers.  Use
    # the numbers here so reporting also works across installations where a
    # newer symbolic constant is absent from an older gurobipy build.
    status_names = {
        1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
        5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT",
        8: "NODE_LIMIT", 9: "TIME_LIMIT", 10: "SOLUTION_LIMIT",
        11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL",
        14: "INPROGRESS", 15: "USER_OBJ_LIMIT",
    }

    t0 = time.time()
    m = gp.Model("exact_pool_mip")
    m.Params.TimeLimit = args.timelimit
    m.Params.MIPGap = args.mipgap
    m.Params.Threads = args.threads
    experiment_arm = {
        (False, False): "A",
        (True, False): "B",
        (False, True): "C",
        (True, True): "D",
    }[(args.two_stage, args.initial_partition_routes is not None)]
    progress = None
    termination = None
    if progress_path is not None:
        progress = MIPProgressRecorder(
            progress_path,
            time_limit_s=args.timelimit,
            metadata={
                "source_result_sha256": source_result_sha256,
                "source_journal_sha256": source_journal_sha256,
                "source_initial_partition_sha256": (
                    expected_initial_partition_sha256
                    or initial_partition_start.get("source_sha256")
                ),
                "extra_route_sources": extra_route_sources,
                "gurobi_version": ".".join(
                    str(value) for value in gp.gurobi.version()
                ),
                "parameters": {
                    "time_limit_s": args.timelimit,
                    "mip_gap": args.mipgap,
                    "threads": args.threads,
                    "two_stage": args.two_stage,
                    "cover": args.cover,
                },
                "physical_pool_audit": physical_pool_audit,
                "git_commit": code_identity["observed_commit"],
                "expected_git_commit": code_identity["expected_commit"],
                "experiment_arm": experiment_arm,
            },
        )
        termination = TerminationRequest()
        termination.install()
    a = m.addVars(len(routes), vtype=GRB.BINARY, name="a")
    if mip_start:
        start_set = set(mip_start)
        for index in range(len(routes)):
            a[index].Start = 1.0 if index in start_set else 0.0
    initial_partition_start["assignment_complete"] = bool(mip_start)
    initial_partition_start["assigned_variable_count"] = (
        len(routes) if mip_start else 0
    )
    initial_partition_start["selected_variable_count"] = (
        len(mip_start) if mip_start else 0
    )
    if progress is not None:
        progress.transition_stage(
            "fleet" if args.two_stage else "single",
            elapsed_s=0.0,
        )
        if mip_start:
            progress.record_initial_incumbent(
                list(mip_start),
                objective=float(sum(
                    routes[index]["cost"] for index in mip_start
                )),
                fleet=len(mip_start),
                kind=(
                    "validated_partition_at_t0"
                    if initial_partition_start["kind"]
                    == "validated_exact_partition"
                    else "initial_mip_start_at_t0"
                ),
            )
        progress.emit_zero()
    sense = ">" if args.cover else "="
    trip_rows = {t: [] for t in trips}
    for i, r in enumerate(routes):
        for t in r["trips"]:
            trip_rows[t].append(i)
    for t in trips:
        expr = gp.quicksum(a[i] for i in trip_rows[t])
        if args.cover:
            m.addConstr(expr >= 1, name=f"cov_{t}")
        else:
            m.addConstr(expr == 1, name=f"part_{t}")

    def progress_observer(stage, fixed_fleet=None):
        if progress is None:
            return None
        return GurobiProgressObserver(
            progress,
            GRB=GRB,
            variables=a,
            routes=routes,
            bus_cost=BUS_COST_KX,
            stage=stage,
            fixed_fleet=fixed_fleet,
            termination=termination,
        )

    two_stage_detail = None
    cost_stage_executed = False
    cost_stage_has_solution = False
    gurobi_optimize_wall_s = []
    validated_start_available = (
        initial_partition_start["kind"] == "validated_exact_partition"
        and bool(mip_start)
    )
    if args.two_stage:
        # Fleet recovery is the primary experiment.  Stage 1 may consume the
        # complete budget.  Cost optimization is allowed only after the
        # integer fleet count has been proved, using whatever time remains.
        m.Params.TimeLimit = args.timelimit
        m.setObjective(
            gp.quicksum(a[i] for i in range(len(routes))), GRB.MINIMIZE
        )
        optimize_started = time.perf_counter()
        initial_partition_start["solver_acceptance"] = (
            optimize_with_start_audit(
                m,
                GRB,
                start_supplied=bool(mip_start),
                progress_observer=progress_observer("fleet"),
            )
        )
        gurobi_optimize_wall_s.append(
            time.perf_counter() - optimize_started
        )
        stage1_has_solution = m.SolCount > 0
        validated_start_fallback = (
            not stage1_has_solution
            and validated_start_available
        )
        stage1_buses = (
            int(round(m.ObjVal))
            if stage1_has_solution
            else (len(mip_start) if validated_start_fallback else None)
        )
        stage1_bound = finite_solver_value(m.ObjBound)
        stage1_status = int(m.Status)
        stage1_gap = (
            finite_solver_value(m.MIPGap)
            if stage1_has_solution else None
        )
        fleet_proven = (
            fleet_bound_proves_incumbent(
                stage1_buses, stage1_bound, stage1_status
            )
            if (stage1_has_solution or validated_start_fallback) else False
        )
        stage1_solution = (
            [i for i in range(len(routes)) if a[i].X > 0.5]
            if stage1_has_solution
            else (list(mip_start) if validated_start_fallback else [])
        )
        if (
            validated_start_fallback
            and stage1_bound is not None
            and stage1_buses is not None
        ):
            stage1_gap = (
                max(0.0, stage1_buses - stage1_bound)
                / max(1.0, stage1_buses)
            )
        stage1_runtime_s = (
            progress.elapsed_s() if progress is not None
            else time.time() - t0
        )
        remaining_s = max(0.0, float(args.timelimit) - stage1_runtime_s)
        print(
            f"[MIP] stage 1: fleet={stage1_buses} "
            f"(bound {stage1_bound}, gap {stage1_gap}, "
            f"proven={fleet_proven}, remaining={remaining_s:.1f}s)"
        )
        two_stage_detail = {
            "stage1_buses": stage1_buses,
            "stage1_solver_has_solution": stage1_has_solution,
            "stage1_incumbent_source": (
                "solver" if stage1_has_solution
                else (
                    "validated_start_fallback"
                    if validated_start_fallback else None
                )
            ),
            "stage1_bound": stage1_bound,
            "stage1_status": stage1_status,
            "stage1_status_name": status_names.get(
                stage1_status, f"UNKNOWN_{stage1_status}"
            ),
            "stage1_gap": stage1_gap,
            "fleet_proven": fleet_proven,
            "stage1_runtime_s": stage1_runtime_s,
            "stage1_node_count": finite_solver_value(
                getattr(m, "NodeCount", None)
            ),
            "stage1_solution_count": int(m.SolCount),
            "stage2_executed": False,
            "stage2_has_solution": False,
            "stage2_skip_reason": None,
        }
        if (
            stage1_has_solution
            and fleet_proven
            and remaining_s >= 1.0
            and not (termination and termination.requested)
            and pool_master_cost_semantics
            != "mixed_expanded_grid_and_continuous_augmented_cost"
        ):
            m.addConstr(
                gp.quicksum(a[i] for i in range(len(routes))) == stage1_buses,
                name="fleet_budget",
            )
            stage1_set = set(stage1_solution)
            for index in range(len(routes)):
                a[index].Start = 1.0 if index in stage1_set else 0.0
            variable_costs = [
                route["cost"] - BUS_COST_KX for route in routes
            ]
            m.setObjective(
                gp.quicksum(variable_costs[i] * a[i]
                            for i in range(len(routes))),
                GRB.MINIMIZE,
            )
            m.Params.TimeLimit = remaining_s
            if progress is not None:
                progress.transition_stage(
                    "cost", elapsed_s=progress.elapsed_s()
                )
            optimize_started = time.perf_counter()
            stage2_start_acceptance = optimize_with_start_audit(
                m,
                GRB,
                start_supplied=True,
                progress_observer=progress_observer(
                    "cost", fixed_fleet=stage1_buses
                ),
            )
            gurobi_optimize_wall_s.append(
                time.perf_counter() - optimize_started
            )
            cost_stage_executed = True
            cost_stage_has_solution = m.SolCount > 0
            two_stage_detail["stage2_executed"] = True
            two_stage_detail["stage2_has_solution"] = cost_stage_has_solution
            two_stage_detail["stage2_start_acceptance"] = (
                stage2_start_acceptance
            )
            two_stage_detail["stage2_node_count"] = finite_solver_value(
                getattr(m, "NodeCount", None)
            )
            two_stage_detail["stage2_solution_count"] = int(m.SolCount)
        elif termination and termination.requested:
            two_stage_detail["stage2_skip_reason"] = (
                "termination_signal_requested"
            )
        elif not stage1_has_solution:
            two_stage_detail["stage2_skip_reason"] = "no_fleet_incumbent"
        elif not fleet_proven:
            two_stage_detail["stage2_skip_reason"] = "fleet_not_proven"
        elif (
            pool_master_cost_semantics
            == "mixed_expanded_grid_and_continuous_augmented_cost"
        ):
            two_stage_detail["stage2_skip_reason"] = (
                "mixed_route_cost_semantics"
            )
        else:
            two_stage_detail["stage2_skip_reason"] = "no_time_remaining"
    else:
        m.setObjective(
            gp.quicksum(routes[i]["cost"] * a[i]
                        for i in range(len(routes))),
            GRB.MINIMIZE,
        )
        optimize_started = time.perf_counter()
        initial_partition_start["solver_acceptance"] = (
            optimize_with_start_audit(
                m,
                GRB,
                start_supplied=bool(mip_start),
                progress_observer=progress_observer("single"),
            )
        )
        gurobi_optimize_wall_s.append(
            time.perf_counter() - optimize_started
        )
        fleet_proven = int(m.Status) == 2

    if args.two_stage and not stage1_has_solution and validated_start_fallback:
        chosen = list(stage1_solution)
        status_code = stage1_status
        solver_obj = float(stage1_buses)
        solver_bound = stage1_bound
        mip_gap = stage1_gap
    elif args.two_stage and not stage1_has_solution:
        chosen = []
        status_code = stage1_status
        solver_obj = None
        solver_bound = stage1_bound
        mip_gap = None
    elif args.two_stage and not cost_stage_executed:
        chosen = list(stage1_solution)
        status_code = stage1_status
        solver_obj = float(stage1_buses)
        solver_bound = stage1_bound
        mip_gap = stage1_gap
    elif args.two_stage and not cost_stage_has_solution:
        # The proved fleet incumbent is still a valid deliverable even if the
        # second optimizer fails to accept its warm start before interruption.
        chosen = list(stage1_solution)
        status_code = int(m.Status)
        solver_obj = float(sum(
            routes[i]["cost"] - BUS_COST_KX for i in chosen
        ))
        solver_bound = finite_solver_value(m.ObjBound)
        mip_gap = (
            max(0.0, solver_obj - solver_bound) / max(1.0, abs(solver_obj))
            if solver_bound is not None else None
        )
    elif (
        not args.two_stage
        and m.SolCount == 0
        and validated_start_available
    ):
        chosen = list(mip_start)
        status_code = int(m.Status)
        solver_obj = float(sum(routes[index]["cost"] for index in chosen))
        solver_bound = finite_solver_value(m.ObjBound)
        mip_gap = (
            max(0.0, solver_obj - solver_bound)
            / max(1.0, abs(solver_obj))
            if solver_bound is not None else None
        )
    else:
        status_code = int(m.Status)
        chosen = [i for i in range(len(routes)) if a[i].X > 0.5] \
            if m.SolCount > 0 else []
        solver_obj = finite_solver_value(m.ObjVal) if m.SolCount > 0 else None
        solver_bound = finite_solver_value(m.ObjBound)
        mip_gap = finite_solver_value(m.MIPGap) if m.SolCount > 0 else None

    status_name = status_names.get(status_code, f"UNKNOWN_{status_code}")
    selected_routes = [routes[i] for i in chosen]
    if selected_routes and not args.cover:
        try:
            validate_final_selected_routes(
                status,
                trips,
                selected_routes,
                data_dir=args.data_dir,
                reference_data_dir=args.reference_data_dir,
                physical_pool_audit=physical_pool_audit,
            )
        except (PhysicalReplayError, SystemExit, Exception) as caught:
            exc = (
                caught if isinstance(caught, PhysicalReplayError)
                else PhysicalReplayError(str(caught))
            )
            diagnostic_path = out.with_name(
                f"{out.name}.rejected_physical_replay.json"
            )
            final_code_identity = verified_mip_code_identity()
            if (
                final_code_identity["observed_commit"]
                != code_identity["observed_commit"]
                or not final_code_identity["tracked_clean"]
            ):
                raise SystemExit(
                    "[MIP] solver code identity changed before rejection "
                    "publication"
                )
            code_identity["final_observed_commit"] = (
                final_code_identity["observed_commit"]
            )
            code_identity["final_tracked_clean"] = (
                final_code_identity["tracked_clean"]
            )
            if progress is not None:
                progress.finalize(
                    elapsed_s=progress.elapsed_s(),
                    final={
                        "status": status_code,
                        "status_name": "REJECTED_PHYSICAL_REPLAY",
                        "incumbent_found": True,
                        "physically_validated": False,
                        "buses": len(chosen),
                        "fleet_proven": fleet_proven,
                        "selected_route_indices": list(chosen),
                        "route_vector_sha256": hashlib.sha256(json.dumps(
                            sorted(chosen), separators=(",", ":")
                        ).encode()).hexdigest(),
                        "rejected_diagnostic": str(diagnostic_path),
                    },
                )
            diagnostic_path = publish_rejected_physical_replay(
                out,
                error=exc,
                chosen=chosen,
                routes=routes,
                status_name=status_name,
                solver_bound=solver_bound,
                mip_gap=mip_gap,
                source_result_sha256=source_result_sha256,
                source_journal_sha256=source_journal_sha256,
                progress_path=progress_path,
                physical_pool_audit=physical_pool_audit,
                bound_scope=(
                    "fixed_fleet_variable_cost"
                    if cost_stage_executed
                    else "fleet_count"
                    if args.two_stage
                    else "full_pool_objective"
                ),
                code_identity=code_identity,
                augmentation_sources=[
                    *extra_route_sources,
                    *(
                        [{
                            "path": str(args.initial_partition_routes),
                            "sha256":
                                expected_initial_partition_sha256
                                or (
                                    initial_partition_start or {}
                                ).get("source_sha256"),
                        }]
                        if args.initial_partition_routes is not None
                        else []
                    ),
                ],
                master_cost_semantics=(
                    (physical_pool_audit or {}).get(
                        "post_augmentation_master_cost_semantics"
                    )
                    or (
                        next(iter({
                            routes[index].get("master_cost_semantics")
                            for index in chosen
                        }))
                        if len({
                            routes[index].get("master_cost_semantics")
                            for index in chosen
                        }) == 1
                        else "mixed_expanded_grid_and_continuous_augmented_cost"
                    )
                ),
                source_pricing_certified=(
                    status.get("certified_rc_optimal") is True
                ),
                source_reference_hashes_bound=(
                    (physical_pool_audit or {}).get(
                        "source_reference_hashes_bound"
                    ) is True
                ),
            )
            raise SystemExit(
                f"{exc}; diagnostic={diagnostic_path}"
            ) from exc
    over = {t: c for t, c in Counter(
        t for i in chosen for t in routes[i]["trips"]).items() if c > 1}
    has_incumbent = bool(chosen)
    solver_incumbent_found = (
        bool(stage1_has_solution)
        if args.two_stage and not cost_stage_executed
        else bool(m.SolCount > 0)
    )
    incumbent_source = (
        "validated_start_fallback"
        if validated_start_available
        and not solver_incumbent_found
        and has_incumbent
        else ("solver" if has_incumbent else None)
    )
    mip_obj = (float(sum(routes[i]["cost"] for i in chosen))
               if chosen else None)
    continuous_selected_costs = [
        routes[index].get("continuous_realized_cost")
        for index in chosen
    ]
    continuous_realized_mip_obj = (
        float(sum(continuous_selected_costs))
        if chosen and all(
            value is not None for value in continuous_selected_costs
        )
        else None
    )
    conservative_grid_selected_costs = [
        routes[index].get("cost")
        if routes[index].get("master_cost_semantics")
        in {"expanded_grid_cost", "expanded_grid_cost_unchanged"}
        else None
        for index in chosen
    ]
    conservative_grid_mip_obj = (
        float(sum(conservative_grid_selected_costs))
        if chosen and all(
            value is not None
            for value in conservative_grid_selected_costs
        )
        else None
    )
    selected_route_hashes = [
        hashlib.sha256(json.dumps({
            "trips": routes[index].get("trips"),
            "route_nodes": routes[index].get("route_nodes"),
            "charging_stops": routes[index].get("charging_stops"),
            "cost": routes[index].get("cost"),
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for index in chosen
    ]
    selected_block_hashes = [
        (routes[index].get("physical_realization") or {}).get(
            "continuous_realized_charging_blocks_sha256"
        )
        for index in chosen
    ]
    master_cost_semantics = pool_master_cost_semantics
    source_reference_hashes_bound = (
        (physical_pool_audit or {}).get(
            "source_reference_hashes_bound"
        ) is True
    )
    pricing_certificate_scope = (
        "conservative_expanded_grid_model_only"
        if status.get("certified_rc_optimal") is True
        and source_reference_hashes_bound
        and master_cost_semantics == "expanded_grid_cost"
        else "source_grid_certificate_missing_reference_hash_binding"
        if status.get("certified_rc_optimal") is True
        and not source_reference_hashes_bound
        else "source_grid_certificate_does_not_cover_augmented_pool"
        if status.get("certified_rc_optimal") is True
        and master_cost_semantics is not None
        else "not_certified"
        if status.get("certified_rc_optimal") is not True
        else "none_for_mixed_or_continuous_augmented_pool"
    )
    if cost_stage_executed and solver_bound is not None and two_stage_detail:
        mip_bound = (BUS_COST_KX * two_stage_detail["stage1_buses"]
                     + solver_bound)
        mip_bound_scope = "fixed_proven_fleet_variable_cost"
    elif args.two_stage and solver_bound is not None:
        # A negative-price tariff can make route-variable costs negative.
        # At most one nonempty selected route per trip is needed in a strict
        # partition, so this remains conservative without assuming
        # nonnegative charging cost.
        minimum_variable_cost = min(
            float(route["cost"]) - BUS_COST_KX for route in routes
        )
        negative_variable_floor = (
            min(0.0, minimum_variable_cost) * len(trips)
        )
        mip_bound = (
            BUS_COST_KX * solver_bound + negative_variable_floor
        )
        mip_bound_scope = (
            "fleet_bound_plus_negative_route_cost_floor"
        )
    else:
        mip_bound = solver_bound
        mip_bound_scope = "full_pool_objective"
    stage2_absolute_gap = (
        max(0.0, solver_obj - solver_bound)
        if (cost_stage_executed and solver_obj is not None
            and solver_bound is not None) else None
    )
    if two_stage_detail is not None:
        two_stage_detail.update({
            "stage2_status": status_code if cost_stage_executed else None,
            "stage2_status_name": status_name if cost_stage_executed else None,
            "stage2_variable_obj": solver_obj if cost_stage_executed else None,
            "stage2_variable_bound": (solver_bound
                                      if cost_stage_executed else None),
            "stage2_absolute_gap": stage2_absolute_gap,
            "stage2_reported_incumbent_source": (
                "stage2_solver" if cost_stage_has_solution
                else ("stage1_fallback" if cost_stage_executed else None)
            ),
        })
    final_code_identity = verified_mip_code_identity()
    if (final_code_identity["observed_commit"]
            != code_identity["observed_commit"]):
        raise SystemExit("[MIP] solver commit changed during optimization")
    code_identity["final_observed_commit"] = final_code_identity[
        "observed_commit"
    ]
    code_identity["final_tracked_clean"] = final_code_identity[
        "tracked_clean"
    ]
    solver_runtime_s = (
        progress.elapsed_s() if progress is not None
        else time.time() - t0
    )
    summary = {
        "source_result": str(args.result),
        "instance": status["csv"],
        "partitioning": not args.cover,
        "experiment_arm": experiment_arm,
        "status": status_code,
        "status_name": status_name,
        "optimal_scope": optimal_scope(
            two_stage=args.two_stage,
            fleet_proven=fleet_proven,
            cost_stage_executed=cost_stage_executed,
            final_status=status_code,
        ),
        "mip_obj": mip_obj,
        "mip_bound": mip_bound,
        "mip_bound_scope": mip_bound_scope,
        "requested_mip_gap": args.mipgap,
        "mip_gap": mip_gap,
        "absolute_cost_gap": stage2_absolute_gap,
        "buses": len(chosen) if has_incumbent else None,
        "incumbent_found": has_incumbent,
        "solver_incumbent_found": solver_incumbent_found,
        "incumbent_source": incumbent_source,
        "fleet_proven": fleet_proven,
        "fleet_bound": (two_stage_detail.get("stage1_bound")
                        if two_stage_detail else None),
        "charging_cost": (mip_obj - BUS_COST_KX * len(chosen))
                         if chosen and mip_obj is not None else None,
        "charging_cost_semantics": master_cost_semantics,
        "conservative_grid_mip_obj": conservative_grid_mip_obj,
        "conservative_grid_charging_cost": (
            conservative_grid_mip_obj - BUS_COST_KX * len(chosen)
            if conservative_grid_mip_obj is not None else None
        ),
        "conservative_grid_cost_availability": (
            "available_for_all_selected_routes"
            if conservative_grid_mip_obj is not None
            else "unavailable_for_continuous_augmented_route"
            if chosen else "no_incumbent"
        ),
        "continuous_realized_mip_obj": continuous_realized_mip_obj,
        "continuous_realized_charging_cost": (
            continuous_realized_mip_obj - BUS_COST_KX * len(chosen)
            if continuous_realized_mip_obj is not None else None
        ),
        "selected_route_hashes": selected_route_hashes,
        "selected_route_set_sha256": hashlib.sha256(json.dumps(
            selected_route_hashes, separators=(",", ":")
        ).encode()).hexdigest(),
        "selected_charging_block_hashes": selected_block_hashes,
        "selected_charging_block_set_sha256": hashlib.sha256(json.dumps(
            selected_block_hashes, separators=(",", ":")
        ).encode()).hexdigest(),
        "variable_route_cost": (mip_obj - BUS_COST_KX * len(chosen))
                               if chosen and mip_obj is not None else None,
        "overcovered_trips": len(over),
        "runtime_s": solver_runtime_s,
        "node_count": (
            sum(
                float(value) for value in (
                    (two_stage_detail or {}).get("stage1_node_count"),
                    (two_stage_detail or {}).get("stage2_node_count"),
                ) if value is not None
            )
            if args.two_stage else finite_solver_value(
                getattr(m, "NodeCount", None)
            )
        ),
        "solution_count": (
            sum(
                int(value) for value in (
                    (two_stage_detail or {}).get("stage1_solution_count"),
                    (two_stage_detail or {}).get("stage2_solution_count"),
                ) if value is not None
            )
            if args.two_stage else int(m.SolCount)
        ),
        "gurobi_optimize_wall_s": sum(gurobi_optimize_wall_s),
        "gurobi_optimize_stage_wall_s": gurobi_optimize_wall_s,
        "source_hashing_wall_s": source_hashing_wall_s,
        "end_to_end_before_publication_s": (
            time.perf_counter() - end_to_end_started
        ),
        "pool_columns": len(routes),
        "singleton_partition_columns": len(seed_partition),
        "mip_start_assigned": bool(mip_start),
        "mip_start_used": (
            (initial_partition_start.get("solver_acceptance") or {})
            .get("accepted") is True
        ),
        "mip_start_buses": len(mip_start) if mip_start else None,
        "mip_start": initial_partition_start,
        "pool_preparation": status.get("pool_preparation"),
        "physical_pool_audit": physical_pool_audit,
        "physical_pool_preparation_wall_s": (
            physical_pool_audit.get("preparation_wall_s")
            if physical_pool_audit else None
        ),
        "master_cost_semantics": master_cost_semantics,
        "pricing_certificate_scope":
            pricing_certificate_scope,
        "continuous_cost_pricing_certified": False,
        "source_cg_wall_s": status.get("wall_s"),
        "source_cg_iterations": status.get("iterations"),
        "source_snapshot_mark_minutes": status.get(
            "snapshot_mark_minutes"
        ),
        "two_stage": two_stage_detail,
        "pricer_provenance": status.get("provenance"),
        "physics": {
            "soc_step": status.get("soc_step"),
            "block_min": status.get("block_min"),
            "g_kwh": status.get("g_kwh"),
            "charge_kw": status.get("charge_kw"),
            "min_soc_frac": status.get("min_soc_frac"),
            "prices_csv": status.get("prices_csv"),
        },
        "source_result_sha256": source_result_sha256,
        "source_journal": str(source_journal),
        "source_journal_sha256": source_journal_sha256,
        "extra_route_sources": extra_route_sources,
        "mip_provenance": {
            "git_commit": code_identity["observed_commit"],
            "expected_git_commit": code_identity["expected_commit"],
            "observed_git_commit": code_identity["observed_commit"],
            "final_observed_git_commit": code_identity[
                "final_observed_commit"
            ],
            "git_branch": code_identity["branch"],
            "git_detached": code_identity["detached"],
            "git_dirty": not code_identity["tracked_clean"],
            "tracked_clean_at_end": code_identity["final_tracked_clean"],
            "python": platform.python_version(),
            "gurobi": ".".join(str(value) for value in gp.gurobi.version()),
            "host": platform.node(),
            "gurobi_parameters": {
                "TimeLimit_s": args.timelimit,
                "MIPGap": args.mipgap,
                "Threads": args.threads,
                "Seed": int(getattr(m.Params, "Seed", 0)),
                "seed_source": "gurobi_default",
                "seed_explicitly_set": False,
            },
            "arguments": {
                "timelimit": args.timelimit,
                "mipgap": args.mipgap,
                "threads": args.threads,
                "two_stage": args.two_stage,
                "cover": args.cover,
                "initial_partition_routes": (
                    str(args.initial_partition_routes)
                    if args.initial_partition_routes is not None else None
                ),
                "verified_expanded_initial_partition":
                    args.verified_expanded_initial_partition,
            },
        },
        "selected_routes": selected_routes,
        "progress": (
            {
                "directory": str(progress_path),
                "checkpoint_schedule_s": progress.schedule,
                "observational_only": True,
                "gurobi_tree_restart_supported": False,
                "disabled_reason": progress.disabled_reason,
                "termination_signal": (
                    termination.signal_name if termination else None
                ),
            }
            if progress is not None else None
        ),
    }
    if progress is not None:
        progress.finalize(
            elapsed_s=summary["runtime_s"],
            final={
                "status": status_code,
                "status_name": status_name,
                "incumbent_found": has_incumbent,
                "buses": summary["buses"],
                "mip_obj": mip_obj,
                "mip_bound": mip_bound,
                "mip_gap": mip_gap,
                "fleet_proven": fleet_proven,
                "optimal_scope": summary["optimal_scope"],
                "selected_route_indices": list(chosen),
                "route_vector_sha256": (
                    hashlib.sha256(json.dumps(
                        sorted(chosen), separators=(",", ":")
                    ).encode()).hexdigest()
                    if chosen else None
                ),
                "termination_signal": (
                    termination.signal_name if termination else None
                ),
            },
        )
    write_new_json(out, summary)
    if termination is not None:
        termination.restore()
    print(f"[MIP] status={status_name}({status_code}) buses={summary['buses']} "
          f"obj={summary['mip_obj']} gap={summary['mip_gap']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

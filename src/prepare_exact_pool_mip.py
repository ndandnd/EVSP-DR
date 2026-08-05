"""Create a strict-MIP-ready copy of a persisted exact-pricer pool.

Legacy exact-CG campaigns started artificial-only and journaled only priced
routes.  Such a pool can be row-coverage-complete (and have a feasible LP)
without containing any binary exact partition.  This tool preserves every
priced route and adds one same-model depot-trip-depot singleton per trip.

The input snapshot and journal are never modified.  The output snapshot points
to a new adjacent journal and records the augmentation provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_giro_known_columns import HORIZON_MIN, build_problem
from exact_pricer_expanded import DATA_DIR, direct_singleton_seed_records
from run_exact_pool_mip import load_pool


def default_output_path(result_path: Path) -> Path:
    if result_path.name.endswith(".snapshot.json"):
        stem = result_path.name[: -len(".snapshot.json")]
    elif result_path.name.endswith(".json"):
        stem = result_path.name[:-len(".json")]
    else:
        stem = result_path.name
    return result_path.with_name(f"{stem}.partition_ready.snapshot.json")


def journal_for_output(output_path: Path) -> Path:
    if not output_path.name.endswith(".snapshot.json"):
        raise ValueError("output path must end in .snapshot.json")
    stem = output_path.name[: -len(".snapshot.json")]
    return output_path.with_name(f"{stem}.columns.jsonl")


def merge_pool(routes: list[dict], seeds: list[dict]) -> tuple[list[dict], int]:
    pool = {frozenset(route["trips"]): route for route in routes}
    added = 0
    for seed in seeds:
        key = frozenset(seed["trips"])
        if key not in pool or seed["cost"] < pool[key]["cost"] - 1e-9:
            pool[key] = seed
            added += 1
    return list(pool.values()), added


def prepare_pool(
    result_path: Path,
    output_path: Path,
    *,
    data_dir: Path = DATA_DIR,
    force: bool = False,
) -> dict:
    result_path = result_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_journal = journal_for_output(output_path)
    if output_path == result_path:
        raise ValueError("output snapshot must not overwrite the input")
    if not force:
        existing = [path for path in (output_path, output_journal) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite prepared pool: "
                + ", ".join(str(path) for path in existing)
            )

    status, routes, trip_ids = load_pool(result_path)
    required = ("csv", "soc_step", "g_kwh", "min_soc_frac")
    missing_fields = [field for field in required if status.get(field) is None]
    if missing_fields:
        raise ValueError(
            f"{result_path} lacks required model fields: {missing_fields}"
        )
    problem = build_problem(
        data_dir,
        status["csv"],
        max_station_to_trip_wait_min=HORIZON_MIN,
    )
    if set(problem.trips) != set(trip_ids):
        raise ValueError(
            "snapshot trip_ids do not match the reconstructed instance: "
            f"snapshot={len(trip_ids)}, instance={len(problem.trips)}"
        )

    seeds, missing_singletons = direct_singleton_seed_records(
        problem,
        g_kwh=float(status["g_kwh"]),
        soc_step=float(status["soc_step"]),
        reserve_kwh=float(status["min_soc_frac"]) * float(status["g_kwh"]),
    )
    if missing_singletons:
        raise ValueError(
            "direct singleton seeds do not form a complete partition; "
            f"missing {len(missing_singletons)} trips "
            f"({missing_singletons[:15]})"
        )
    seed_counts = {trip: 0 for trip in trip_ids}
    for seed in seeds:
        for trip in seed["trips"]:
            seed_counts[trip] += 1
    bad_counts = {trip: count for trip, count in seed_counts.items() if count != 1}
    if bad_counts:
        raise ValueError(
            "singleton seed is not an exact partition: "
            f"{list(bad_counts.items())[:15]}"
        )

    merged, added = merge_pool(routes, seeds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    journal_tmp = Path(str(output_journal) + ".tmp")
    status_tmp = Path(str(output_path) + ".tmp")
    try:
        with journal_tmp.open("w") as handle:
            for route in merged:
                handle.write(json.dumps(route) + "\n")
        prepared = dict(status)
        prepared["columns"] = len(merged)
        prepared["columns_journal"] = str(output_journal)
        prepared["master_sense"] = status.get("master_sense", "cover")
        prepared["pool_preparation"] = {
            "kind": "exact_direct_singleton_partition",
            "source_result": str(result_path),
            "source_result_sha256": hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest(),
            "source_unique_columns": len(routes),
            "singleton_partition_columns": len(seeds),
            "singleton_columns_added": added,
            "prepared_unique_columns": len(merged),
            "input_master_sense": status.get("master_sense", "cover_legacy"),
        }
        with status_tmp.open("w") as handle:
            json.dump(prepared, handle, indent=1)
        journal_tmp.replace(output_journal)
        status_tmp.replace(output_path)
    finally:
        for tmp in (journal_tmp, status_tmp):
            if tmp.exists():
                tmp.unlink()
    return prepared


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = args.out or default_output_path(args.result)
    prepared = prepare_pool(
        args.result,
        output,
        data_dir=args.data_dir,
        force=args.force,
    )
    info = prepared["pool_preparation"]
    print(
        "[PREP] strict partition seed ready: "
        f"{info['singleton_partition_columns']} singleton columns, "
        f"{info['singleton_columns_added']} added, "
        f"{info['prepared_unique_columns']} total -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate reproducible synthetic Goal-1 instances from historical bus blocks.

The tracked combined GIRO-derived table contains weekday variants for two
VehicleTask blocks.  This generator samples *base* blocks uniformly without
replacement and, when one of those two bases is selected, chooses exactly one
literal weekday variant.  The resulting files are useful scaling/rediscovery
benchmarks, but they are not evidence of a verified single service day.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "Par_VehicleDetails_Updated.csv"
DEFAULT_SEED = 20260802
GENERATOR_VERSION = 1

COMPACT_COLUMNS = (
    "Identifier",
    "From1",
    "Start1",
    "End1",
    "To1",
    "Distance1",
    "Usage kWh",
    "count_trip_id",
    "Ordered_Trip_ID",
)

# These are explicit aliases from the combined source export.  Do not infer
# arbitrary suffixes: a future alphanumeric task ID may be a real base block.
LITERAL_TO_BASE = {
    "13316m": "13316",
    "13316uwt": "13316",
    "13324muw": "13324",
    "13324t": "13324",
}

SYNTHETIC_NOTE = (
    "Synthetic base-VehicleTask sample from the combined export; weekday "
    "compatibility has not been verified, so this is not a single-day GIRO instance."
)


@dataclass(frozen=True)
class SourceRows:
    source_path: Path
    source_sha256: str
    rows_by_literal_task: Mapping[str, tuple[dict[str, str], ...]]
    literals_by_base_task: Mapping[str, tuple[str, ...]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_vehicle_task(literal_task: str) -> str:
    """Return the explicit base group for a literal VehicleTask value."""

    literal = str(literal_task).strip()
    return LITERAL_TO_BASE.get(literal, literal)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _require_columns(fieldnames: Sequence[str] | None, required: Iterable[str]) -> None:
    present = set(fieldnames or ())
    missing = sorted(set(required) - present)
    if missing:
        raise ValueError(f"Source CSV is missing required columns: {missing}")


def load_regular_source(source_path: Path) -> SourceRows:
    """Load exact string values for Regular rows and construct base groups."""

    source_path = source_path.resolve()
    rows_by_literal: dict[str, list[dict[str, str]]] = {}
    with source_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, ("VehicleTask", *COMPACT_COLUMNS))
        for source_index, row in enumerate(reader):
            if row["Identifier"].strip() != "Regular":
                continue
            literal = row["VehicleTask"].strip()
            if not literal:
                raise ValueError(f"Regular source row {source_index + 2} has no VehicleTask")
            copied = dict(row)
            copied["_source_index"] = str(source_index)
            rows_by_literal.setdefault(literal, []).append(copied)

    if not rows_by_literal:
        raise ValueError(f"No Identifier=Regular rows found in {source_path}")

    literals_by_base: dict[str, list[str]] = {}
    for literal in sorted(rows_by_literal):
        literals_by_base.setdefault(base_vehicle_task(literal), []).append(literal)

    return SourceRows(
        source_path=source_path,
        source_sha256=sha256_file(source_path),
        rows_by_literal_task={
            key: tuple(value) for key, value in sorted(rows_by_literal.items())
        },
        literals_by_base_task={
            key: tuple(sorted(value)) for key, value in sorted(literals_by_base.items())
        },
    )


def _derived_rng(seed: int, size: int, replicate: int, namespace: str) -> random.Random:
    material = f"evsp-goal1-v{GENERATOR_VERSION}|{seed}|{size}|{replicate}|{namespace}"
    derived = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest(), "big")
    return random.Random(derived)


def select_vehicle_tasks(
    source: SourceRows,
    *,
    size: int,
    seed: int,
    replicate: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Sample base tasks uniformly and choose one literal per selected base."""

    if size <= 0:
        raise ValueError("size must be positive")
    if replicate <= 0:
        raise ValueError("replicate must be positive")

    base_universe = sorted(source.literals_by_base_task)
    if size > len(base_universe):
        raise ValueError(
            f"Cannot sample {size} base tasks from a universe of {len(base_universe)}"
        )

    sample_rng = _derived_rng(seed, size, replicate, "base-sample")
    selected_bases = tuple(sorted(sample_rng.sample(base_universe, size)))

    selected_literals: list[str] = []
    for base in selected_bases:
        variants = source.literals_by_base_task[base]
        variant_rng = _derived_rng(seed, size, replicate, f"variant:{base}")
        selected_literals.append(variants[variant_rng.randrange(len(variants))])

    return selected_bases, tuple(selected_literals)


_TIME_RE = re.compile(r"^(\d+):(\d{1,2})$")


def time_to_minutes(value: str) -> int:
    """Parse GIRO-style H:MM/HH:MM values, including hours beyond 24."""

    match = _TIME_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Invalid Start1 time {value!r}")
    hour, minute = int(match.group(1)), int(match.group(2))
    if minute >= 60:
        raise ValueError(f"Invalid Start1 time {value!r}")
    return 60 * hour + minute


def render_instance_csv(
    source: SourceRows,
    selected_literals: Sequence[str],
) -> tuple[bytes, int]:
    """Render a compact, chronological instance while preserving trip IDs."""

    selected_rows: list[dict[str, str]] = []
    for literal in selected_literals:
        try:
            selected_rows.extend(source.rows_by_literal_task[literal])
        except KeyError as exc:
            raise ValueError(f"Unknown literal VehicleTask {literal!r}") from exc

    selected_rows.sort(
        key=lambda row: (time_to_minutes(row["Start1"]), int(row["_source_index"]))
    )
    ordered_ids = [row["Ordered_Trip_ID"] for row in selected_rows]
    if any(not value.strip() for value in ordered_ids):
        raise ValueError("Every selected Regular row must have an Ordered_Trip_ID")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Selected Regular rows contain duplicate Ordered_Trip_ID values")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COMPACT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for local_trip_id, row in enumerate(selected_rows):
        compact = {column: row[column] for column in COMPACT_COLUMNS}
        compact["count_trip_id"] = str(local_trip_id)
        writer.writerow(compact)
    return output.getvalue().encode("utf-8"), len(selected_rows)


def _atomic_write(path: Path, payload: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing == payload:
            return
        if not force:
            raise FileExistsError(
                f"Refusing to replace different existing file {path}; pass --force explicitly"
            )

    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def generate_batch(
    *,
    source_path: Path,
    output_dir: Path,
    sizes: Sequence[int] = (10, 15, 20),
    replicates: int = 5,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> dict[str, object]:
    """Generate instance CSVs plus deterministic JSON and CSV manifests."""

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    normalized_sizes = tuple(sorted(set(int(size) for size in sizes)))
    if not normalized_sizes or any(size <= 0 for size in normalized_sizes):
        raise ValueError("sizes must contain positive integers")

    source = load_regular_source(source_path)
    output_dir = output_dir.resolve()
    instances: list[dict[str, object]] = []

    for size in normalized_sizes:
        for replicate in range(1, replicates + 1):
            selected_bases, selected_literals = select_vehicle_tasks(
                source, size=size, seed=seed, replicate=replicate
            )
            payload, trip_count = render_instance_csv(source, selected_literals)
            filename = (
                f"Practice_SyntheticRandom_{size}bus_s{seed}_r{replicate:02d}.csv"
            )
            output_path = output_dir / filename
            _atomic_write(output_path, payload, force=force)
            instances.append(
                {
                    "bus_count": size,
                    "replicate": replicate,
                    "seed": seed,
                    "selected_base_tasks": list(selected_bases),
                    "selected_literal_tasks": list(selected_literals),
                    "trip_count": trip_count,
                    "source_sha256": source.source_sha256,
                    "output_csv": filename,
                    "output_sha256": hashlib.sha256(payload).hexdigest(),
                    "synthetic": True,
                    "single_day_verified": False,
                    "note": SYNTHETIC_NOTE,
                }
            )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "generator": "src/generate_random_goal1_instances.py",
        "seed": seed,
        "replicates_per_size": replicates,
        "sizes": list(normalized_sizes),
        "synthetic": True,
        "single_day_verified": False,
        "note": SYNTHETIC_NOTE,
        "source": {
            "path": _portable_path(source.source_path),
            "sha256": source.source_sha256,
            "regular_rows": sum(len(rows) for rows in source.rows_by_literal_task.values()),
            "literal_task_count": len(source.rows_by_literal_task),
            "base_task_count": len(source.literals_by_base_task),
        },
        "variant_groups": {
            base: list(literals)
            for base, literals in source.literals_by_base_task.items()
            if len(literals) > 1
        },
        "instances": instances,
    }

    json_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(output_dir / "manifest.json", json_payload, force=force)

    csv_output = io.StringIO(newline="")
    manifest_columns = (
        "bus_count",
        "replicate",
        "seed",
        "trip_count",
        "selected_base_tasks",
        "selected_literal_tasks",
        "source_sha256",
        "output_csv",
        "output_sha256",
        "synthetic",
        "single_day_verified",
        "note",
    )
    writer = csv.DictWriter(csv_output, fieldnames=manifest_columns, lineterminator="\n")
    writer.writeheader()
    for instance in instances:
        row = dict(instance)
        row["selected_base_tasks"] = "|".join(instance["selected_base_tasks"])
        row["selected_literal_tasks"] = "|".join(instance["selected_literal_tasks"])
        writer.writerow({column: row[column] for column in manifest_columns})
    _atomic_write(
        output_dir / "manifest.csv",
        csv_output.getvalue().encode("utf-8"),
        force=force,
    )
    return manifest


def _parse_sizes(raw: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from exc
    if not sizes:
        raise argparse.ArgumentTypeError("sizes cannot be empty")
    return sizes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: data/random_goal1_instances/seed_<seed>",
    )
    parser.add_argument("--sizes", type=_parse_sizes, default=(10, 15, 20))
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a same-name file only when its bytes differ.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        REPO_ROOT / "data" / "random_goal1_instances" / f"seed_{args.seed}"
    )
    manifest = generate_batch(
        source_path=args.source,
        output_dir=output_dir,
        sizes=args.sizes,
        replicates=args.replicates,
        seed=args.seed,
        force=args.force,
    )
    print(f"Generated {len(manifest['instances'])} synthetic instances in {output_dir.resolve()}")
    print(f"Manifest: {(output_dir / 'manifest.json').resolve()}")
    print(SYNTHETIC_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

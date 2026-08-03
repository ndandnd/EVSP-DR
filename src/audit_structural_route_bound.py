#!/usr/bin/env python3
"""Audit structural lower bounds on fractional vehicle-route weight.

For each requested trip CSV, this script rebuilds the current pricing
compatibility graph, computes the ordinary peak-concurrency bound, and then
computes a stronger maximum antichain in the graph's reachability relation.

If ``A`` is that antichain, every feasible pricing route contains at most one
trip in ``A``.  Summing the master cover constraints for trips in ``A`` gives

    |A| <= sum_r x_r * sum_{i in A} a_ir <= sum_r x_r,

so ``|A|`` is a valid lower bound on route weight even when the master
variables ``x_r`` are fractional.  This is distinct from the minimum
vertex-disjoint path-cover count, which is not generally a fractional
set-cover bound when the direct compatibility graph is not transitively
closed.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from audit_giro_known_columns import (
    DEFAULT_DATA_DIR,
    HORIZON_MIN,
    MAX_DAILY_RECHARGES,
    STATIONS,
    ProblemData,
    build_problem,
)
from matching_init import _trip_successors, peak_trip_concurrency


SCHEMA_VERSION = 1
DEFAULT_BATTERY_KWH = 300.0
DEFAULT_CHARGE_RATE_KW = 300.0
DEFAULT_MAX_TRIP2TRIP_MIN = 57.0
DEFAULT_MAX_CHARGE2TRIP_MIN = HORIZON_MIN
DEFAULT_SUCCESSOR_CHARGE_TARGETS = True
LP_BOUND_EXPLANATION = (
    "Every feasible pricing column has a trip sequence that is a directed path "
    "in the audited compatibility graph. A reachability antichain intersects "
    "each such path at most once. Summing its trip-cover constraints therefore "
    "proves sum_r x_r >= antichain_size, including for fractional x."
)


@dataclass(frozen=True)
class AntichainCertificate:
    """Maximum-antichain certificate for one directed acyclic graph."""

    antichain: tuple[int, ...]
    direct_edge_count: int
    reachability_edge_count: int
    direct_matching_cardinality: int
    reachability_matching_cardinality: int
    direct_minimum_path_cover: int
    reachability_antichain_bound: int
    transitively_closed: bool
    pairwise_incomparable: bool


def _normal_successors(
    successors: Mapping[int, Iterable[int]],
) -> dict[int, tuple[int, ...]]:
    nodes = set(successors)
    for following in successors.values():
        nodes.update(following)
    if any(not isinstance(node, int) for node in nodes):
        raise TypeError("Structural-bound trip nodes must be integer local IDs")
    return {
        node: tuple(sorted(set(successors.get(node, ()))))
        for node in sorted(nodes)
    }


def reachability_closure(
    successors: Mapping[int, Iterable[int]],
) -> tuple[dict[int, frozenset[int]], tuple[int, ...]]:
    """Return strict reachability sets and a deterministic topological order."""

    graph = _normal_successors(successors)
    indegree = {node: 0 for node in graph}
    for following in graph.values():
        for node in following:
            indegree[node] += 1

    ready = deque(node for node in graph if indegree[node] == 0)
    topological: list[int] = []
    while ready:
        node = ready.popleft()
        topological.append(node)
        for following in graph[node]:
            indegree[following] -= 1
            if indegree[following] == 0:
                ready.append(following)

    if len(topological) != len(graph):
        raise ValueError("Compatibility graph must be acyclic")

    reachable: dict[int, frozenset[int]] = {}
    for node in reversed(topological):
        found: set[int] = set()
        for following in graph[node]:
            found.add(following)
            found.update(reachable[following])
        reachable[node] = frozenset(found)
    return reachable, tuple(topological)


def _maximum_matching(
    nodes: Sequence[int],
    successors: Mapping[int, Iterable[int]],
) -> dict[int, int]:
    if not nodes:
        return {}
    node_index = {node: index for index, node in enumerate(nodes)}
    rows: list[int] = []
    columns: list[int] = []
    for left in nodes:
        following = sorted(set(successors.get(left, ())))
        rows.extend([node_index[left]] * len(following))
        columns.extend(node_index[right] for right in following)
    if not rows:
        return {}

    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)),
        shape=(len(nodes), len(nodes)),
    )
    row_to_column = maximum_bipartite_matching(graph, perm_type="column")
    return {
        row: int(column)
        for row, column in enumerate(row_to_column)
        if column >= 0
    }


def maximum_reachability_antichain(
    successors: Mapping[int, Iterable[int]],
) -> AntichainCertificate:
    """Compute a maximum antichain and matching-based validation metadata.

    The bipartite graph used for the antichain calculation contains an edge
    from the left copy of ``u`` to the right copy of ``v`` exactly when ``v``
    is reachable from ``u``.  Dilworth's theorem then gives antichain size
    ``n - maximum_matching_cardinality``.  The alternating-path construction
    below extracts an explicit maximum antichain, not only its cardinality.
    """

    direct = _normal_successors(successors)
    reachable, topological = reachability_closure(direct)
    nodes = tuple(sorted(direct))
    node_index = {node: index for index, node in enumerate(nodes)}

    direct_matching = _maximum_matching(nodes, direct)
    reachability_matching = _maximum_matching(nodes, reachable)
    matched_left_by_right = {
        right: left for left, right in reachability_matching.items()
    }

    # Konig alternating search from unmatched left vertices.  For a poset's
    # reachability bipartite graph, {v: v_L in Z_L and v_R not in Z_R} is a
    # maximum antichain.
    reachable_indices = {
        node_index[left]: tuple(node_index[right] for right in reachable[left])
        for left in nodes
    }
    left_reached = {
        index for index in range(len(nodes))
        if index not in reachability_matching
    }
    right_reached: set[int] = set()
    queue = deque(("left", index) for index in sorted(left_reached))
    while queue:
        side, index = queue.popleft()
        if side == "left":
            matched_right = reachability_matching.get(index)
            for right in reachable_indices[index]:
                if right == matched_right:
                    continue
                if right not in right_reached:
                    right_reached.add(right)
                    queue.append(("right", right))
        else:
            matched_left = matched_left_by_right.get(index)
            if matched_left is not None and matched_left not in left_reached:
                left_reached.add(matched_left)
                queue.append(("left", matched_left))

    antichain = tuple(
        nodes[index]
        for index in range(len(nodes))
        if index in left_reached and index not in right_reached
    )
    incomparable = all(
        right not in reachable[left] and left not in reachable[right]
        for position, left in enumerate(antichain)
        for right in antichain[position + 1 :]
    )
    expected_size = len(nodes) - len(reachability_matching)
    if len(antichain) != expected_size or not incomparable:
        raise RuntimeError(
            "Failed to extract a valid maximum reachability antichain"
        )

    direct_edges = {
        (left, right) for left in nodes for right in direct[left]
    }
    reachability_edges = {
        (left, right) for left in nodes for right in reachable[left]
    }
    return AntichainCertificate(
        antichain=antichain,
        direct_edge_count=len(direct_edges),
        reachability_edge_count=len(reachability_edges),
        direct_matching_cardinality=len(direct_matching),
        reachability_matching_cardinality=len(reachability_matching),
        direct_minimum_path_cover=len(nodes) - len(direct_matching),
        reachability_antichain_bound=expected_size,
        transitively_closed=direct_edges == reachability_edges,
        pairwise_incomparable=incomparable,
    )


def build_current_trip_successors(
    problem: ProblemData,
    *,
    battery_kwh: float = DEFAULT_BATTERY_KWH,
    charge_rate_kw: float = DEFAULT_CHARGE_RATE_KW,
    max_charge2trip_min: float = DEFAULT_MAX_CHARGE2TRIP_MIN,
    successor_charge_targets: bool = DEFAULT_SUCCESSOR_CHARGE_TARGETS,
) -> dict[int, tuple[int, ...]]:
    """Build the trip compatibility supergraph used by current DP pricing."""

    horizon_min = float(HORIZON_MIN)
    soc_levels = [battery_kwh * index / 10.0 for index in range(1, 11)]
    return _trip_successors(
        problem.trips,
        problem.adjacency,
        stations=STATIONS,
        trip_start_min=problem.start_min,
        trip_end_min=problem.end_min,
        trip_energy_kwh=problem.trip_energy,
        battery_capacity_kwh=battery_kwh,
        charge_rate_kw=charge_rate_kw,
        soc_charge_levels=soc_levels,
        horizon_min=horizon_min,
        max_daily_recharges=MAX_DAILY_RECHARGES,
        max_station_to_trip_wait_min=max_charge2trip_min,
        successor_boundary_soc_target=successor_charge_targets,
        station_waiting_unrestricted=(
            max_charge2trip_min >= horizon_min - 1e-6
        ),
        direct_only=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _antichain_trip_record(problem: ProblemData, trip: int) -> dict[str, object]:
    row = problem.frame.iloc[trip]
    return {
        "local_trip_index": int(trip),
        "ordered_trip_id": _json_scalar(row.get("Ordered_Trip_ID")),
        "start": _json_scalar(row.get("ST")),
        "end": _json_scalar(row.get("ET")),
        "start_min": float(problem.start_min[trip]),
        "end_min": float(problem.end_min[trip]),
        "start_location": _json_scalar(row.get("SL")),
        "end_location": _json_scalar(row.get("EL")),
    }


def audit_instance(
    data_dir: Path,
    csv_name: str,
    *,
    battery_kwh: float = DEFAULT_BATTERY_KWH,
    charge_rate_kw: float = DEFAULT_CHARGE_RATE_KW,
    max_trip2trip_min: float = DEFAULT_MAX_TRIP2TRIP_MIN,
    max_charge2trip_min: float = DEFAULT_MAX_CHARGE2TRIP_MIN,
    successor_charge_targets: bool = DEFAULT_SUCCESSOR_CHARGE_TARGETS,
) -> dict[str, object]:
    """Return one deterministic structural-bound report."""

    problem = build_problem(
        data_dir,
        csv_name,
        max_trip2trip_min=max_trip2trip_min,
        max_station_to_trip_wait_min=max_charge2trip_min,
    )
    successors = build_current_trip_successors(
        problem,
        battery_kwh=battery_kwh,
        charge_rate_kw=charge_rate_kw,
        max_charge2trip_min=max_charge2trip_min,
        successor_charge_targets=successor_charge_targets,
    )
    certificate = maximum_reachability_antichain(successors)
    peak = peak_trip_concurrency(
        problem.trips,
        problem.start_min,
        problem.end_min,
    )
    instance_path = data_dir / csv_name
    antichain_records = [
        _antichain_trip_record(problem, trip)
        for trip in certificate.antichain
    ]
    return {
        "instance_csv": csv_name,
        "instance_sha256": _sha256(instance_path),
        "trip_count": len(problem.trips),
        "peak_trip_concurrency": int(peak),
        "reachability_antichain_lp_route_weight_bound": (
            certificate.reachability_antichain_bound
        ),
        "bound_gap_over_peak": (
            certificate.reachability_antichain_bound - int(peak)
        ),
        "antichain_local_trip_indices": list(certificate.antichain),
        "antichain_ordered_trip_ids": [
            record["ordered_trip_id"] for record in antichain_records
        ],
        "antichain_trips": antichain_records,
        "graph_validation": {
            "acyclic": True,
            "direct_edge_count": certificate.direct_edge_count,
            "reachability_edge_count": certificate.reachability_edge_count,
            "transitively_closed": certificate.transitively_closed,
            "direct_matching_cardinality": (
                certificate.direct_matching_cardinality
            ),
            "direct_minimum_vertex_disjoint_path_cover": (
                certificate.direct_minimum_path_cover
            ),
            "reachability_matching_cardinality": (
                certificate.reachability_matching_cardinality
            ),
            "antichain_pairwise_incomparable": (
                certificate.pairwise_incomparable
            ),
            "matching_antichain_identity_holds": (
                certificate.reachability_antichain_bound
                == len(problem.trips)
                - certificate.reachability_matching_cardinality
            ),
        },
        "compatibility_parameters": {
            "battery_kwh": float(battery_kwh),
            "charge_rate_kw": float(charge_rate_kw),
            "horizon_min": float(HORIZON_MIN),
            "max_trip2trip_min": float(max_trip2trip_min),
            "max_trip2charge_min": 61,
            "max_charge2trip_min": float(max_charge2trip_min),
            "max_daily_recharges": int(MAX_DAILY_RECHARGES),
            "successor_charge_targets": bool(successor_charge_targets),
            "soc_grid_levels": 10,
            "stations": list(STATIONS),
        },
        "validity": {
            "scope": "current_restricted_pricing_compatibility_graph",
            "lp_route_weight_bound": True,
            "explanation": LP_BOUND_EXPLANATION,
            "caution": (
                "The bound is structural for the configured compatibility "
                "graph. It is not a claim about a graph with relaxed gap, "
                "station, charging, or location assumptions."
            ),
        },
    }


def _resolve_instance_name(data_dir: Path, raw: str) -> str:
    data_root = data_dir.resolve()
    raw_path = Path(raw)
    candidates = [raw_path] if raw_path.is_absolute() else [data_dir / raw_path, raw_path]
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        try:
            return str(resolved.relative_to(data_root))
        except ValueError as exc:
            raise ValueError(
                f"Instance {resolved} must be inside data directory {data_root}"
            ) from exc
    raise FileNotFoundError(f"Instance CSV not found: {raw}")


def build_report(
    data_dir: Path,
    instance_names: Sequence[str],
    *,
    battery_kwh: float = DEFAULT_BATTERY_KWH,
    charge_rate_kw: float = DEFAULT_CHARGE_RATE_KW,
    max_trip2trip_min: float = DEFAULT_MAX_TRIP2TRIP_MIN,
    max_charge2trip_min: float = DEFAULT_MAX_CHARGE2TRIP_MIN,
    successor_charge_targets: bool = DEFAULT_SUCCESSOR_CHARGE_TARGETS,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "audit": "reachability_antichain_lp_route_weight_bound",
        "validity_explanation": LP_BOUND_EXPLANATION,
        "instances": [
            audit_instance(
                data_dir,
                name,
                battery_kwh=battery_kwh,
                charge_rate_kw=charge_rate_kw,
                max_trip2trip_min=max_trip2trip_min,
                max_charge2trip_min=max_charge2trip_min,
                successor_charge_targets=successor_charge_targets,
            )
            for name in instance_names
        ],
    }


def write_json_report(report: Mapping[str, object], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def print_report(report: Mapping[str, object]) -> None:
    print(LP_BOUND_EXPLANATION)
    for instance in report["instances"]:
        print()
        print(f"Instance: {instance['instance_csv']}")
        print(f"  SHA256: {instance['instance_sha256']}")
        print(f"  Trips: {instance['trip_count']}")
        print(f"  Peak concurrency bound: {instance['peak_trip_concurrency']}")
        print(
            "  Reachability-antichain LP route-weight bound: "
            f"{instance['reachability_antichain_lp_route_weight_bound']}"
        )
        validation = instance["graph_validation"]
        print(
            "  Graph: "
            f"{validation['direct_edge_count']} direct edges; "
            f"{validation['reachability_edge_count']} reachability edges; "
            f"transitively_closed={validation['transitively_closed']}"
        )
        print("  Maximum antichain trips:")
        for trip in instance["antichain_trips"]:
            print(
                "    local={local_trip_index} ordered={ordered_trip_id} "
                "{start}-{end} {start_location}->{end_location}".format(**trip)
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "instances",
        nargs="+",
        help="One or more CSV paths inside --data-dir",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--max-trip2trip",
        type=float,
        default=DEFAULT_MAX_TRIP2TRIP_MIN,
        help="Direct trip-end to next-trip-start cap in minutes (default: 57)",
    )
    parser.add_argument(
        "--max-charge2trip",
        type=float,
        default=DEFAULT_MAX_CHARGE2TRIP_MIN,
        help="Station-to-trip wait cap in minutes (current Goal-1 default: 1560)",
    )
    parser.add_argument(
        "--successor-charge-targets",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SUCCESSOR_CHARGE_TARGETS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_trip2trip <= 0 or args.max_charge2trip <= 0:
        raise SystemExit(
            "ERROR: --max-trip2trip and --max-charge2trip must be positive"
        )
    data_dir = args.data_dir.resolve()
    try:
        instance_names = [
            _resolve_instance_name(data_dir, raw) for raw in args.instances
        ]
        report = build_report(
            data_dir,
            instance_names,
            max_trip2trip_min=args.max_trip2trip,
            max_charge2trip_min=args.max_charge2trip,
            successor_charge_targets=args.successor_charge_targets,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print_report(report)
    if args.json_out is not None:
        write_json_report(report, args.json_out)
        print(f"\nJSON report: {args.json_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

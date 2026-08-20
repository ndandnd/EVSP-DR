"""Direct arc-flow oracle over the exact pricer's SOC-by-time DAG.

The network is imported from ``exact_pricer_expanded``; this module only
indexes it, removes arcs that provably lie on no source-to-sink path, and
builds the sparse LP/MIP.  Results are exact only for the named discretized
model, never for the continuous real-world problem.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix

from audit_giro_known_columns import DEPOT, HORIZON_MIN, build_problem
from config import BUS_COST_KX, CHARGING_STATIONS
from durable_io import read_jsonl_records
from exact_pricer_expanded import DATA_DIR, ExpandedNetwork, _block_minutes
from expanded_path_realization import realize_expanded_path
from run_exact_pool_mip import validate_injected_route
from utils_v2 import load_station_hourly_prices


STATUS = {0: "optimal", 1: "limit_reached", 2: "infeasible",
          3: "unbounded", 4: "solver_error"}


@dataclass(frozen=True)
class NetworkData:
    csv_name: str
    prices_csv: str
    problem: object
    network: ExpandedNetwork
    soc_step: float
    block_min: float
    g_kwh: float = 300.0
    charge_kw: float = 300.0
    reserve_kwh: float = 0.0


@dataclass(frozen=True)
class ArcTable:
    """Active arcs; inactive full-network arcs are implicit fixed zeros."""

    tail: np.ndarray
    head: np.ndarray
    cost: np.ndarray
    trip: np.ndarray
    out_start: np.ndarray
    node_row: np.ndarray
    full_nodes: int
    full_arcs: int

    @property
    def size(self) -> int:
        return int(self.tail.size)


@dataclass(frozen=True)
class ArcFlowModel:
    data: NetworkData
    arcs: ArcTable
    matrix: csc_matrix
    lower: np.ndarray
    upper: np.ndarray
    variable_upper: np.ndarray
    fleet_row: int


@dataclass(frozen=True)
class SolveResult:
    objective_kind: str
    integrality: str
    fixed_fleet: int | None
    status: str
    status_code: int
    objective: float | None
    vehicles: float | None
    charging_cost: float | None
    solve_s: float
    mip_gap: float | None
    dual_bound: float | None
    all_arcs_integral: bool | None
    max_row_violation: float | None


def build_network(csv_name: str, *, soc_step: float, block_min: float,
                  prices_csv: str = "hourly_prices_flat.csv",
                  g_kwh: float = 300.0, charge_kw: float = 300.0,
                  reserve_kwh: float = 0.0,
                  data_dir: Path = DATA_DIR) -> NetworkData:
    problem = build_problem(data_dir, csv_name,
                            max_station_to_trip_wait_min=HORIZON_MIN)
    prices = load_station_hourly_prices(data_dir / prices_csv,
                                        CHARGING_STATIONS)
    network = ExpandedNetwork(
        problem, prices, soc_step=soc_step, block_min=block_min,
        g_kwh=g_kwh, charge_kw=charge_kw, reserve_kwh=reserve_kwh,
    )
    parsed_block = _block_minutes(block_min)
    return NetworkData(csv_name, prices_csv, problem, network,
                       float(soc_step), parsed_block, float(g_kwh),
                       float(charge_kw), float(reserve_kwh))


def _source_sink_reachability(network: ExpandedNetwork) -> tuple[np.ndarray, np.ndarray]:
    """Nodes reachable from source and able to reach sink in the DAG."""

    n = len(network.node_meta)
    forward = np.zeros(n, dtype=bool)
    forward[0] = True
    for u in network.topo:
        if forward[u]:
            for v, _cost, _trip in network.out[u]:
                forward[v] = True
    backward = np.zeros(n, dtype=bool)
    backward[network.SINK] = True
    for u in reversed(network.topo):
        if any(backward[v] for v, _cost, _trip in network.out[u]):
            backward[u] = True
    return forward, backward


def index_active_arcs(network: ExpandedNetwork) -> ArcTable:
    """Exact presolve: omit only arcs fixed to zero by source/sink reachability."""

    forward, backward = _source_sink_reachability(network)
    full_nodes = len(network.node_meta)
    active_nodes = forward & backward
    active_nodes[0] = active_nodes[network.SINK] = True
    node_row = np.full(full_nodes, -1, dtype=np.int32)
    row = 0
    for node in range(full_nodes):
        if active_nodes[node] and node not in {0, network.SINK}:
            node_row[node] = row
            row += 1
    full_arcs = sum(len(outgoing) for outgoing in network.out)
    active_arcs = sum(
        1 for u, outgoing in enumerate(network.out) if forward[u]
        for v, _cost, _trip in outgoing if backward[v]
    )
    tail = np.empty(active_arcs, dtype=np.int32)
    head = np.empty(active_arcs, dtype=np.int32)
    cost = np.empty(active_arcs, dtype=np.float64)
    trip = np.empty(active_arcs, dtype=np.int32)
    out_start = np.empty(full_nodes + 1, dtype=np.int64)
    cursor = 0
    for u, outgoing in enumerate(network.out):
        out_start[u] = cursor
        if forward[u]:
            for v, arc_cost, trip_position in outgoing:
                if backward[v]:
                    tail[cursor], head[cursor] = u, v
                    cost[cursor], trip[cursor] = arc_cost, trip_position
                    cursor += 1
    out_start[full_nodes] = cursor
    if cursor != active_arcs or full_arcs != network.n_arcs:
        raise AssertionError("arc indexing disagrees with pricer counts")
    return ArcTable(tail, head, cost, trip, out_start, node_row,
                    full_nodes, full_arcs)


def gate_g1(data: NetworkData, arcs: ArcTable) -> dict:
    """Print both full counts and assert identity with the pricer network."""

    oracle_full_arcs = sum(len(outgoing) for outgoing in data.network.out)
    print(
        "[ARCFLOW] G1 network identity: "
        f"pricer={len(data.network.node_meta):,} nodes/"
        f"{data.network.n_arcs:,} arcs; oracle="
        f"{arcs.full_nodes:,} nodes/{oracle_full_arcs:,} arcs; "
        f"active after exact reachability presolve={arcs.size:,} arcs",
        flush=True,
    )
    if (len(data.network.node_meta), data.network.n_arcs) != (
            arcs.full_nodes, oracle_full_arcs):
        raise AssertionError("G1 FAILED: full network counts differ")
    return {"gate": "G1", "passed": True, "nodes": arcs.full_nodes,
            "arcs": arcs.full_arcs, "active_arcs": arcs.size}


def build_model(data: NetworkData, arcs: ArcTable) -> ArcFlowModel:
    """Flow rows, exact trip coverage, and a source-flow fleet row."""

    internal_rows = int(np.count_nonzero(arcs.node_row >= 0))
    n_trips = len(data.problem.trips)
    fleet_row = internal_rows + n_trips
    n_rows = fleet_row + 1
    counts = np.full(arcs.size, 2, dtype=np.int8)
    counts += (arcs.trip >= 0)
    counts += (arcs.tail == 0)
    counts -= (arcs.tail == 0) | (arcs.head == data.network.SINK)
    nnz = int(counts.sum())
    indices = np.empty(nnz, dtype=np.int32)
    values = np.empty(nnz, dtype=np.float64)
    indptr = np.empty(arcs.size + 1, dtype=np.int64)
    cursor = 0
    for arc in range(arcs.size):
        indptr[arc] = cursor
        entries: list[tuple[int, float]] = []
        tail_row = int(arcs.node_row[arcs.tail[arc]])
        head_row = int(arcs.node_row[arcs.head[arc]])
        if tail_row >= 0:
            entries.append((tail_row, 1.0))
        if head_row >= 0:
            entries.append((head_row, -1.0))
        if arcs.trip[arc] >= 0:
            entries.append((internal_rows + int(arcs.trip[arc]), 1.0))
        if arcs.tail[arc] == 0:
            entries.append((fleet_row, 1.0))
        entries.sort()
        for entry_row, value in entries:
            indices[cursor], values[cursor] = entry_row, value
            cursor += 1
    indptr[arcs.size] = cursor
    if cursor != nnz:
        raise AssertionError(f"sparse allocation {nnz} != emitted {cursor}")
    matrix = csc_matrix((values, indices, indptr),
                        shape=(n_rows, arcs.size))
    lower = np.zeros(n_rows)
    upper = np.zeros(n_rows)
    lower[internal_rows:fleet_row] = 1.0
    upper[internal_rows:fleet_row] = 1.0
    upper[fleet_row] = n_trips
    variable_upper = np.full(arcs.size, float(n_trips))
    # A trip is served once; every arc leaving a trip-state node is therefore
    # at most one. These are valid bounds, not a model restriction.
    service_or_trip_exit = arcs.trip >= 0
    for arc in np.flatnonzero(~service_or_trip_exit):
        if data.network.node_meta[int(arcs.tail[arc])][0] == "trip":
            service_or_trip_exit[arc] = True
    variable_upper[service_or_trip_exit] = 1.0
    return ArcFlowModel(data, arcs, matrix, lower, upper, variable_upper,
                        fleet_row)


def objectives(model: ArcFlowModel) -> tuple[np.ndarray, np.ndarray]:
    fleet = (model.arcs.tail == 0).astype(float)
    combined = np.array(model.arcs.cost, copy=True)
    if not np.allclose(combined[fleet == 1], BUS_COST_KX,
                       rtol=0.0, atol=1e-9):
        raise AssertionError("source arc cost differs from fixed bus cost")
    return fleet, combined


def solve(model: ArcFlowModel, *, objective_kind: str,
          integrality: str, fixed_fleet: int | None = None,
          time_limit_s: float | None = None,
          mip_rel_gap: float = 0.0, disp: bool = False
          ) -> tuple[SolveResult, np.ndarray | None]:
    """Solve one LP/MIP; ``service`` is an explicit integrality relaxation."""

    if objective_kind not in {"fleet", "combined", "feasibility"}:
        raise ValueError("invalid objective_kind")
    if integrality not in {"none", "all", "service"}:
        raise ValueError("integrality must be none, all, or service")
    fleet_obj, combined_obj = objectives(model)
    objective = (fleet_obj if objective_kind == "fleet" else combined_obj
                 if objective_kind == "combined"
                 else np.zeros(model.arcs.size))
    lower, upper = model.lower.copy(), model.upper.copy()
    if fixed_fleet is not None:
        lower[model.fleet_row] = upper[model.fleet_row] = fixed_fleet
    integer = None
    if integrality == "all":
        integer = np.ones(model.arcs.size, dtype=np.uint8)
    elif integrality == "service":
        integer = (model.arcs.trip >= 0).astype(np.uint8)
    options: dict[str, float | bool] = {"presolve": True, "disp": disp}
    if time_limit_s is not None:
        options["time_limit"] = float(time_limit_s)
    if integrality != "none":
        options["mip_rel_gap"] = float(mip_rel_gap)
    started = time.perf_counter()
    if integrality == "none":
        # Interior point avoids the severe dual-simplex degeneracy caused by
        # thousands of equivalent waiting/charging paths.
        equality_rows = (
            slice(None) if fixed_fleet is not None else slice(0, -1)
        )
        raw = linprog(
            c=objective,
            A_eq=model.matrix[equality_rows],
            b_eq=lower[equality_rows],
            bounds=np.column_stack((
                np.zeros(model.arcs.size), model.variable_upper
            )),
            method="highs-ipm",
            options=options,
        )
    else:
        raw = milp(
            c=objective, integrality=integer,
            bounds=Bounds(0.0, model.variable_upper),
            constraints=LinearConstraint(model.matrix, lower, upper),
            options=options,
        )
    elapsed = time.perf_counter() - started
    primal = None if raw.x is None else np.asarray(raw.x, dtype=float)
    vehicles = charging = row_violation = None
    all_integral = None
    if primal is not None:
        activity = np.asarray(model.matrix @ primal)
        row_violation = float(max(
            np.max(np.maximum(lower - activity, 0.0)),
            np.max(np.maximum(activity - upper, 0.0)),
        ))
        vehicles = float(fleet_obj @ primal)
        charging = float(combined_obj @ primal - BUS_COST_KX * vehicles)
        all_integral = bool(
            np.max(np.abs(primal - np.rint(primal))) <= 1e-6
        )
        if row_violation > 1e-6:
            raise RuntimeError("HiGHS returned a row-infeasible primal")
    result = SolveResult(
        objective_kind, integrality, fixed_fleet,
        STATUS.get(int(raw.status), f"unknown_{raw.status}"), int(raw.status),
        None if raw.fun is None else float(raw.fun), vehicles, charging,
        elapsed,
        None if getattr(raw, "mip_gap", None) is None else float(raw.mip_gap),
        None if getattr(raw, "mip_dual_bound", None) is None
        else float(raw.mip_dual_bound),
        all_integral, row_violation,
    )
    return result, primal


def _matching_arc(data: NetworkData, arcs: ArcTable, node: int, predicate) -> int:
    matches = [
        arc for arc in range(int(arcs.out_start[node]),
                             int(arcs.out_start[node + 1]))
        if predicate(int(arcs.head[arc]),
                     data.network.node_meta[int(arcs.head[arc])])
    ]
    if len(matches) != 1:
        raise ValueError(f"node {node} has {len(matches)} matching arcs")
    return matches[0]


def map_route_to_arcs(data: NetworkData, arcs: ArcTable,
                      record: dict) -> list[int]:
    """Map a persisted pricer route, including its SOC and charging blocks."""

    trips = list(record.get("trips") or [])
    nodes = list(record.get("route_nodes") or [])
    if [n for n in nodes if isinstance(n, int) and not isinstance(n, bool)] != trips:
        raise ValueError("route_nodes and trip incidence differ")
    stops = (record.get("expanded_grid_charging_stops")
             if record.get("expanded_grid_charging_stops") is not None
             else record.get("charging_stops") or {})
    fields = {key: list(stops.get(key, []))
              for key in ("stations", "cst", "cet", "kwh")}
    if len({len(value) for value in fields.values()}) != 1:
        raise ValueError("charging stop fields differ in length")
    if [n for n in nodes[1:-1] if isinstance(n, str)] != fields["stations"]:
        raise ValueError("route station sequence differs from charging stops")
    path, current, stop = [], 0, 0
    for event in nodes[1:-1]:
        if isinstance(event, int) and not isinstance(event, bool):
            arc = _matching_arc(
                data, arcs, current,
                lambda _v, meta, trip=event:
                    meta[0] == "trip" and meta[1] == trip,
            )
        else:
            first = int(round(float(fields["cst"][stop]) / data.block_min))
            after = int(round(float(fields["cet"][stop]) / data.block_min))
            if after <= first:
                raise ValueError("empty charging run")
            arc = _matching_arc(
                data, arcs, current,
                lambda _v, meta, station=event, block=first:
                    meta[0] == "charge" and meta[1] == (station, block),
            )
            path.append(arc)
            current = int(arcs.head[arc])
            for block in range(first + 1, after):
                arc = _matching_arc(
                    data, arcs, current,
                    lambda _v, meta, station=event, block=block:
                        meta[0] == "charge" and meta[1] == (station, block),
                )
                path.append(arc)
                current = int(arcs.head[arc])
            stop += 1
            continue
        path.append(arc)
        current = int(arcs.head[arc])
    final = _matching_arc(
        data, arcs, current,
        lambda successor, _meta: successor == data.network.SINK,
    )
    path.append(final)
    return path


def audit_route(data: NetworkData, arcs: ArcTable,
                record: dict, path: Sequence[int]) -> dict:
    if not path or arcs.tail[path[0]] != 0:
        raise ValueError("route path does not leave source")
    if any(arcs.head[a] != arcs.tail[b] for a, b in zip(path, path[1:])):
        raise ValueError("route path is discontinuous")
    if arcs.head[path[-1]] != data.network.SINK:
        raise ValueError("route path does not enter sink")
    serviced = [data.problem.trips[int(arcs.trip[a])] for a in path
                if arcs.trip[a] >= 0]
    if serviced != list(record["trips"]):
        raise ValueError("route path services different trips")
    mapped_cost = float(arcs.cost[np.asarray(path)].sum())
    stored_cost = float(record.get("expanded_grid_cost", record["cost"]))
    if not math.isclose(mapped_cost, stored_cost,
                        rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError(f"mapped cost {mapped_cost} != stored {stored_cost}")
    return {"trips": serviced, "arc_count": len(path),
            "mapped_cost": mapped_cost, "stored_cost": stored_cost}


def gate_g2(data: NetworkData, arcs: ArcTable, journal: Path,
            route_index: int | None = None) -> dict:
    """Round-trip a known feasible journal route at identical cost."""

    records = read_jsonl_records(journal, repair_trailing=False)
    if not records:
        raise ValueError("G2 journal is empty")
    if route_index is None:
        route_index, record = min(
            enumerate(records),
            key=lambda item: (
                not bool((item[1].get("expanded_grid_charging_stops") or {})
                         .get("stations")),
                -len(item[1].get("trips") or []), item[0],
            ),
        )
    else:
        record = records[route_index]
    detail = audit_route(data, arcs, record,
                         map_route_to_arcs(data, arcs, record))
    print(
        f"[ARCFLOW] G2 route round trip: route={route_index}, "
        f"trips={len(detail['trips'])}, arcs={detail['arc_count']}, "
        f"cost={detail['mapped_cost']:.6f} PASS", flush=True,
    )
    return {"gate": "G2", "passed": True, "journal": str(journal),
            "route_index": route_index, **detail}


def decompose(model: ArcFlowModel, primal: np.ndarray) -> list[list[int]]:
    """Decompose a fully integral acyclic flow into vehicle paths."""

    if np.max(np.abs(primal - np.rint(primal))) > 1e-6:
        raise ValueError("cannot decompose fractional arc flow")
    remaining = np.rint(primal).astype(np.int32)
    fleet = int(round(remaining[model.arcs.tail == 0].sum()))
    paths = []
    for _vehicle in range(fleet):
        node, path = 0, []
        while node != model.data.network.SINK:
            arc = next((
                a for a in range(int(model.arcs.out_start[node]),
                                 int(model.arcs.out_start[node + 1]))
                if remaining[a] > 0
            ), None)
            if arc is None:
                raise ValueError(f"flow decomposition stopped at node {node}")
            remaining[arc] -= 1
            path.append(arc)
            node = int(model.arcs.head[arc])
        paths.append(path)
    if remaining.any():
        raise ValueError("integer flow remains outside vehicle paths")
    return paths


def path_record(model: ArcFlowModel, path: Sequence[int]) -> dict:
    """Convert a decomposed path to the pricer's physical route schema."""

    data, arcs, net = model.data, model.arcs, model.data.network
    node_ids = [int(arcs.head[a]) for a in path[:-1]]
    trips, sequence, runs, run = [], [], [], None
    for node in node_ids:
        kind, key, level = net.node_meta[node]
        if kind == "trip":
            if run is not None:
                runs.append(run)
                run = None
            trips.append(key)
            sequence.append(("trip", key))
        elif kind == "charge":
            station, block = key
            if run is not None and run[0] == station and block == run[2] + 1:
                run[2] = block
            else:
                if run is not None:
                    runs.append(run)
                run = [station, block, block, level]
            if not sequence or sequence[-1] != ("station", station):
                sequence.append(("station", station))
    if run is not None:
        runs.append(run)
    charging = {"stations": [], "cst": [], "cet": [], "kwh": []}
    for station, first, last, level in runs:
        soc = net.grid[level]
        for _block in range(first, last + 1):
            soc = net.grid[net._floor(min(net.g, soc + net.block_kwh))]
        charging["stations"].append(station)
        charging["cst"].append(first * net.block_min)
        charging["cet"].append((last + 1) * net.block_min)
        charging["kwh"].append(round(soc - net.grid[level], 6))
    record = {
        "trips": trips,
        "route_nodes": [DEPOT] + [value for _kind, value in sequence] + [DEPOT],
        "charging_stops": charging,
        "expanded_grid_charging_stops": charging,
        "cost": float(arcs.cost[np.asarray(path)].sum()),
    }
    realized, detail = realize_expanded_path(
        data.problem, record, g_kwh=data.g_kwh, charge_kw=data.charge_kw,
        reserve_kwh=data.reserve_kwh, soc_step=data.soc_step,
        block_min=data.block_min, arc_map=net.continuous_arc_map,
    )
    if realized is None:
        raise ValueError(f"path realization failed: {detail.get('reason')}")
    record["charging_stops"] = realized["charging_stops"]
    return record


def gate_g4(model: ArcFlowModel, primal: np.ndarray) -> tuple[dict, list[dict]]:
    """Exact coverage and physical replay, matching the pool-MIP audit."""

    routes = [path_record(model, path) for path in decompose(model, primal)]
    counts = Counter(trip for route in routes for trip in route["trips"])
    wrong = {trip: counts[trip] for trip in model.data.problem.trips
             if counts[trip] != 1}
    if wrong:
        raise AssertionError(f"G4 FAILED: non-unit trip coverage {wrong}")
    for ordinal, route in enumerate(routes, 1):
        reason = validate_injected_route(
            model.data.problem, route, model.data.g_kwh,
            model.data.charge_kw, model.data.reserve_kwh, HORIZON_MIN,
            arc_map=model.data.network.continuous_arc_map,
        )
        if reason:
            raise AssertionError(f"G4 FAILED route {ordinal}: {reason}")
    print(f"[ARCFLOW] G4 integer audit: {len(routes)} routes, "
          f"{len(counts)} trips, exact coverage PASS", flush=True)
    return ({"gate": "G4", "passed": True, "routes": len(routes),
             "covered_trips": len(counts)}, routes)


def _scope(csv_name: str, soc_step: float, block_min: float) -> None:
    if not any(
        f"_k{k:02d}_" in Path(csv_name).name for k in (2, 3, 5, 8, 13, 20)
    ):
        raise ValueError("CLI scope is k2/k3/k5/k8/k13/k20")
    if float(soc_step) <= 0 or float(block_min) <= 0:
        raise ValueError("grid values must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--prices-csv", default="hourly_prices_flat.csv")
    parser.add_argument("--soc-step", type=float, required=True)
    parser.add_argument("--block-min", type=_block_minutes, required=True)
    parser.add_argument("--g-kwh", type=float, default=300.0)
    parser.add_argument("--charge-kw", type=float, default=300.0)
    parser.add_argument("--reserve-kwh", type=float, default=0.0)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--journal-route-index", type=int)
    parser.add_argument("--objective",
                        choices=("fleet", "combined", "feasibility"),
                        default="fleet")
    parser.add_argument("--integrality",
                        choices=("none", "all", "service"), default="none")
    parser.add_argument("--fixed-fleet", type=int)
    parser.add_argument("--setpart-lp", type=float)
    parser.add_argument(
        "--fleet-lower-bound", type=float,
        help="Certified lower bound used only to prove an integral witness optimal.",
    )
    parser.add_argument("--pool-mip", type=int)
    parser.add_argument("--time-limit-s", type=float)
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    parser.add_argument("--network-only", action="store_true")
    parser.add_argument("--disp", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    _scope(args.csv, args.soc_step, args.block_min)
    data = build_network(
        args.csv, prices_csv=args.prices_csv,
        soc_step=args.soc_step, block_min=args.block_min,
        g_kwh=args.g_kwh, charge_kw=args.charge_kw,
        reserve_kwh=args.reserve_kwh,
    )
    arcs = index_active_arcs(data.network)
    gates = [gate_g1(data, arcs)]
    if args.network_only:
        return 0
    if args.journal is None:
        parser.error("--journal is required: G2 must pass before solving")
    gates.append(gate_g2(data, arcs, args.journal,
                         args.journal_route_index))
    model = build_model(data, arcs)
    result, primal = solve(
        model, objective_kind=args.objective,
        integrality=args.integrality, fixed_fleet=args.fixed_fleet,
        time_limit_s=args.time_limit_s, mip_rel_gap=args.mip_rel_gap,
        disp=args.disp,
    )
    print(f"[ARCFLOW] solve: status={result.status}, "
          f"vehicles={result.vehicles}, charging={result.charging_cost}, "
          f"all_arcs_integral={result.all_arcs_integral}, "
          f"solve_s={result.solve_s:.3f}", flush=True)
    if (args.setpart_lp is not None and result.integrality == "none"
            and result.status == "optimal"):
        if result.vehicles is None or result.vehicles > args.setpart_lp + 1e-6:
            raise AssertionError("G3 FAILED: arcflow LP exceeds set-partitioning LP")
        gates.append({"gate": "G3", "passed": True,
                      "arcflow_lp": result.vehicles,
                      "setpart_lp": args.setpart_lp})
    routes = []
    if primal is not None and result.all_arcs_integral:
        g4, routes = gate_g4(model, primal)
        gates.append(g4)
    fleet_proven = bool(
        result.all_arcs_integral
        and result.vehicles is not None
        and args.fleet_lower_bound is not None
        and round(result.vehicles) == math.ceil(
            args.fleet_lower_bound - 1e-7
        )
    )
    if (args.pool_mip is not None and result.all_arcs_integral
            and result.vehicles is not None):
        if result.vehicles > args.pool_mip + 1e-6:
            raise AssertionError("G5 FAILED: arcflow optimum exceeds pool MIP")
        gates.append({"gate": "G5", "passed": True,
                      "arcflow_integer_witness": result.vehicles,
                      "fleet_proven": fleet_proven,
                      "pool_mip": args.pool_mip})
    payload = {
        "schema": "evsp-dr-arcflow-oracle-v1",
        "scope": "exact_for_named_discretized_model_only",
        "csv": data.csv_name,
        "grid": {"soc_step": data.soc_step, "block_min": data.block_min},
        "network": {"nodes": arcs.full_nodes, "arcs": arcs.full_arcs,
                    "active_arcs": arcs.size,
                    "rows": model.matrix.shape[0],
                    "nonzeros": int(model.matrix.nnz)},
        "gates": gates, "solve": asdict(result), "routes": routes,
        "fleet_proof": {
            "lower_bound": args.fleet_lower_bound,
            "integral_witness": result.vehicles
            if result.all_arcs_integral else None,
            "proven": fleet_proven,
        },
        "solver": {"interface": "scipy.optimize.milp", "backend": "HiGHS",
                   "scipy_version": scipy.__version__},
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

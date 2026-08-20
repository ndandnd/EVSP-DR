"""Exact column generation via an SOC x time expanded pricing network.

Instead of SPPRC labeling with dominance, discretized battery state lives in
the network NODES (de Vos, van Lieshout & Dollevoet, Transportation Science;
arXiv:2207.13734): trip nodes are (trip, SOC-at-trip-start) and charging nodes
are (station, time-block, SOC-before-charging). Pricing is then a plain
shortest path in a DAG, processed once in topological order per CG iteration:

  * no labels, no dominance, no queue starvation, no timeouts;
  * termination with min reduced cost >= -eps is a CERTIFICATE that no
    improving column exists in the expanded route space;
  * charging may start at ANY later block after arrival, so delayed-start
    (price-responsive) charging is native — the heuristic DP cannot express it.

Conservative rounding (SOC floored to the grid) keeps every generated duty
feasible for the continuous model. Two deliberate relaxations vs the runner's
restricted DP: recharge count is uncapped (the per-start fee discourages
excess; routes exceeding MAX_DAILY_RECHARGES are reported), and charging
energy per block is floored to the SOC grid (an effective-rate reduction).

Usage (from src/):

    python exact_pricer_expanded.py \
        --csv Practice_Custom_TwoDuty_13301_13302.csv \
        --prices_csv hourly_prices_flat.csv \
        --soc-step 15 --block-min 10 --max-iters 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from audit_giro_known_columns import (
    DEPOT,
    HORIZON_MIN,
    MAX_DAILY_RECHARGES,
    STATIONS,
    build_problem,
)
from config import BIG_M_PENALTY, BUS_COST_KX, CHARGE_RATE_KW, CHARGE_START_COST, CHARGING_STATIONS
from durable_io import (
    DurableFileError,
    atomic_copy,
    atomic_write_json,
    exclusive_output_lock,
    flush_and_fsync,
    read_jsonl_records,
    valid_json_object,
)
from exact_cg_telemetry import PhaseTelemetry
from expanded_path_realization import (
    _arc_map as continuous_arc_map,
    BLOCK_SCHEDULE_SCHEMA,
    charging_block_schedule_sha256,
    realize_expanded_path,
    realized_costs,
    validate_continuous_charging_blocks,
)
from master_lp_scipy import build_route_incidence, solve_restricted_master_lp
from run_exact_pool_mip import validate_injected_route
from utils_v2 import base_station_name, load_station_hourly_prices

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
G_KWH = 300.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_fixed_duty_seed_records(
    path: Path,
    problem,
    station_prices,
    *,
    tariff_path: Path,
    g_kwh: float,
    charge_kw: float,
    reserve_kwh: float,
    soc_step: float,
    block_min: int,
):
    """Load one tariff-specific exact partition as expanded-grid seed columns."""

    source = path.expanduser().resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    routes = payload.get("routes")
    certificates = payload.get("certificates")
    physics = payload.get("physics") or {}
    tariff = payload.get("tariff") or {}
    if (
        payload.get("schema") not in {
            "evsp-dr-tier1-giro40-tariff-v1",
            "evsp-dr-tier1-fixed-duty-partition-v1",
        }
        or not isinstance(routes, list)
        or not routes
        or not isinstance(certificates, list)
        or len(certificates) != len(routes)
        or payload.get("continuous_cost_pricing_certified") is not False
        or tariff.get("sha256") != _file_sha256(tariff_path)
        or any(
            not math.isclose(
                float(physics.get(key, math.nan)), expected,
                rel_tol=0.0, abs_tol=1e-9,
            )
            for key, expected in (
                ("g_kwh", g_kwh),
                ("charge_kw", charge_kw),
                ("reserve_kwh", reserve_kwh),
                ("soc_step", soc_step),
                ("block_min", block_min),
            )
        )
    ):
        raise ValueError("fixed-duty seed identity/physics/tariff mismatch")
    validation_prices = deepcopy(station_prices)
    required_hour = int(math.ceil(HORIZON_MIN / 60.0)) - 1
    if any(
        required_hour not in curve
        for curve in validation_prices.values()
    ):
        if tariff.get("coverage_policy") != (
            "historical_last_hour_extension_verified_constant"
        ):
            raise ValueError("fixed-duty seed tariff coverage is incomplete")
        for curve in validation_prices.values():
            if not curve or len(set(curve.values())) != 1:
                raise ValueError(
                    "fixed-duty seed tariff extension is not constant"
                )
            last = curve[max(curve)]
            for hour in range(required_hour + 1):
                curve.setdefault(hour, last)
    trip_set = set(problem.trips)
    counts = Counter()
    accepted = []
    certificate_by_duty = {
        certificate.get("duty_id"): certificate
        for certificate in certificates
        if isinstance(certificate, dict)
    }
    for ordinal, route in enumerate(routes, start=1):
        trips = list(route.get("trips") or [])
        counts.update(trips)
        if (
            not trips
            or len(trips) != len(set(trips))
            or not set(trips) <= trip_set
            or route.get("master_cost_semantics") != "expanded_grid_cost"
            or route.get("cost_tariff_sha256") != tariff["sha256"]
            or route.get("expanded_grid_cost") is None
            or not math.isclose(
                float(route.get("cost", math.nan)),
                float(route["expanded_grid_cost"]),
                rel_tol=1e-10, abs_tol=1e-6,
            )
        ):
            raise ValueError(f"invalid fixed-duty seed route {ordinal}")
        reason = validate_injected_route(
            problem, route, g_kwh, charge_kw, reserve_kwh, HORIZON_MIN
        )
        if reason is not None:
            raise ValueError(
                f"fixed-duty seed route {ordinal} failed replay: {reason}"
            )
        blocks = route.get("continuous_realized_charging_blocks")
        physical = route.get("physical_realization") or {}
        validation = validate_continuous_charging_blocks(
            route,
            blocks,
            station_prices=validation_prices,
            charge_kw=charge_kw,
            expected_continuous_cost=route.get(
                "continuous_realized_cost"
            ),
        )
        costs = realized_costs(
            route,
            physical,
            station_prices=validation_prices,
        )
        certificate = certificate_by_duty.get(route.get("duty_id"))
        certificate_payload = (
            {
                key: value for key, value in certificate.items()
                if key not in {"certificate_sha256", "duty_id"}
            }
            if isinstance(certificate, dict) else None
        )
        from fixed_duty_expanded_optimizer import optimize_fixed_duty
        recomputed = optimize_fixed_duty(
            problem,
            trips,
            validation_prices,
            g_kwh=g_kwh,
            charge_kw=charge_kw,
            reserve_kwh=reserve_kwh,
            soc_step=soc_step,
            block_min=block_min,
            tariff_id=tariff.get("tariff_id"),
            tariff_sha256=tariff["sha256"],
            instance_sha256=payload.get("instance_sha256"),
        )
        recomputed_certificate = recomputed.get("certificate") or {}
        if (
            physical.get("continuous_cost_pricing_certified") is not False
            or validation["block_schedule_sha256"] != physical.get(
                "continuous_realized_charging_blocks_sha256"
            )
            or not math.isclose(
                costs["recomputed_expanded_grid_cost"],
                float(route["expanded_grid_cost"]),
                rel_tol=1e-10, abs_tol=1e-6,
            )
            or certificate_payload is None
            or certificate.get("certified") is not True
            or certificate.get("scope")
            != "optimal_discretized_charging_for_fixed_trip_sequence"
            or certificate.get(
                "continuous_cost_optimality_certified"
            ) is not False
            or certificate.get("tariff_sha256") != tariff["sha256"]
            or not math.isclose(
                float(certificate.get("objective", math.nan)),
                float(route["expanded_grid_cost"]),
                rel_tol=1e-10, abs_tol=1e-6,
            )
            or certificate.get("certificate_sha256")
            != hashlib.sha256(json.dumps(
                certificate_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()).hexdigest()
            or route.get("fixed_duty_certificate_sha256")
            != certificate.get("certificate_sha256")
            or recomputed.get("feasible") is not True
            or recomputed_certificate.get("certificate_sha256")
            != certificate.get("certificate_sha256")
            or not math.isclose(
                float(recomputed["expanded_grid_objective"]),
                float(route["expanded_grid_cost"]),
                rel_tol=1e-10, abs_tol=1e-6,
            )
            or recomputed["route"].get("route_nodes")
            != route.get("route_nodes")
            or recomputed["route"].get("expanded_grid_charging_stops")
            != route.get("expanded_grid_charging_stops")
        ):
            raise ValueError(
                "fixed-duty seed cost/block/certificate provenance mismatch"
            )
        accepted.append({
            **route,
            "cost": float(route["expanded_grid_cost"]),
            "origin": "tariff_specific_fixed_duty_seed",
            "seed_source_sha256": hashlib.sha256(raw).hexdigest(),
        })
    if (
        set(counts) != trip_set
        or any(counts[trip] != 1 for trip in trip_set)
    ):
        raise ValueError("fixed-duty seeds are not an exact partition")
    return accepted, hashlib.sha256(raw).hexdigest()


def direct_singleton_seed_records(
    problem,
    *,
    g_kwh: float,
    soc_step: float,
    reserve_kwh: float,
) -> tuple[list[dict], list[int]]:
    """Build depot-trip-depot columns feasible in the expanded SOC grid.

    These deterministic one-trip routes form an integer partition whenever
    every trip can leave from and return directly to the depot.  They are a
    safe restricted-master initializer: expensive, but real model columns
    rather than BIG-M artificials.  Trips without a direct singleton are
    returned separately so callers never mistake a partial seed for a
    partition certificate.
    """

    g = float(g_kwh)
    step = float(soc_step)
    reserve = float(reserve_kwh)
    if g <= 0 or step <= 0 or reserve < 0:
        raise ValueError("g_kwh and soc_step must be positive; reserve_kwh >= 0")
    grid = [round(level * step, 6) for level in range(int(g / step) + 1)]

    def floor_soc(soc: float) -> float:
        level = min(
            max(int(math.floor((soc + 1e-9) / step)), 0),
            len(grid) - 1,
        )
        return grid[level]

    depot_trip: dict[int, tuple[float, float]] = {}
    trip_depot: dict[int, tuple[float, float]] = {}
    for node, arcs in problem.adjacency.items():
        for succ, travel_min, deadhead_kwh, arc_type in arcs:
            if arc_type == "depot_trip":
                depot_trip[succ] = (travel_min, deadhead_kwh)
            elif arc_type == "trip_depot":
                trip_depot[node] = (travel_min, deadhead_kwh)

    records: list[dict] = []
    missing: list[int] = []
    for trip in problem.trips:
        if trip not in depot_trip or trip not in trip_depot:
            missing.append(trip)
            continue
        outbound_min, outbound_kwh = depot_trip[trip]
        start_soc = floor_soc(g - outbound_kwh)
        if outbound_min > problem.start_min[trip] + 1e-9:
            missing.append(trip)
            continue
        if start_soc + 1e-9 < problem.trip_energy[trip] + reserve:
            missing.append(trip)
            continue
        return_min, return_kwh = trip_depot[trip]
        exit_soc = start_soc - problem.trip_energy[trip]
        if problem.end_min[trip] + return_min > HORIZON_MIN + 1e-9:
            missing.append(trip)
            continue
        if exit_soc - return_kwh < reserve - 1e-9:
            missing.append(trip)
            continue
        records.append({
            "trips": [trip],
            "cost": float(BUS_COST_KX),
            "route_nodes": [DEPOT, trip, DEPOT],
            "charging_stops": {
                "stations": [], "cst": [], "cet": [], "kwh": [],
            },
            "expanded_grid_charging_stops": {
                "stations": [], "cst": [], "cet": [], "kwh": [],
            },
            "continuous_realized_cost": float(BUS_COST_KX),
            "continuous_realized_charging_blocks": [],
            "continuous_realized_charging_blocks_json_bytes": 2,
            "cost_semantics": "expanded_grid_cost",
            "master_cost_semantics": "expanded_grid_cost",
            "continuous_cost_pricing_certified": False,
            "physical_realization": {
                "status": "valid_as_recorded_mapped",
                "continuous_realized_charging_blocks_sha256":
                    charging_block_schedule_sha256([]),
                "continuous_realized_charging_blocks_schema":
                    BLOCK_SCHEDULE_SCHEMA,
                "continuous_cost_pricing_certified": False,
            },
            "charges_started": 0,
            "found_iter": 0,
            "origin": "exact_direct_singleton_seed",
        })
    return records, missing


class ExpandedNetwork:
    """Static expanded DAG; arc costs are dual-free, trip duals applied on the fly."""

    def __init__(self, problem, station_prices, *, soc_step: float, block_min: int,
                 g_kwh: float = G_KWH, charge_kw: float = CHARGE_RATE_KW,
                 reserve_kwh: float = 0.0,
                 strict_tariff_coverage: bool = False):
        self.problem = problem
        self.trip_position = {
            trip: position for position, trip in enumerate(problem.trips)
        }
        self.soc_step = float(soc_step)
        self.block_min = int(block_min)
        self.g = float(g_kwh)
        self.charge_kw = float(charge_kw)
        self.reserve = float(reserve_kwh)
        self.n_blocks = int(HORIZON_MIN) // self.block_min
        self.block_kwh = float(charge_kw) * self.block_min / 60.0
        self.prices = station_prices  # base station -> {hour: $/kWh}
        self.strict_tariff_coverage = bool(strict_tariff_coverage)
        self.continuous_arc_map = continuous_arc_map(problem)

        self.grid = [round(k * self.soc_step, 6)
                     for k in range(int(self.g / self.soc_step) + 1)]
        self._floor = lambda soc: min(
            max(int(math.floor((soc + 1e-9) / self.soc_step)), 0),
            len(self.grid) - 1,
        )

        # adjacency split by arc type for direct access
        self.trip_trip: dict[int, list] = {}
        self.trip_station: dict[int, list] = {}
        self.station_trip: dict[str, list] = {}
        self.depot_trip: dict[int, tuple] = {}
        self.trip_depot: dict[int, tuple] = {}
        self.station_depot: dict[str, tuple] = {}
        for node, arcs in problem.adjacency.items():
            for succ, travel_min, dh_kwh, arc_type in arcs:
                if arc_type == "trip_trip":
                    self.trip_trip.setdefault(node, []).append((succ, travel_min, dh_kwh))
                elif arc_type == "trip_station":
                    self.trip_station.setdefault(node, []).append((succ, travel_min, dh_kwh))
                elif arc_type == "station_trip":
                    self.station_trip.setdefault(node, []).append((succ, travel_min, dh_kwh))
                elif arc_type == "depot_trip":
                    self.depot_trip[succ] = (travel_min, dh_kwh)
                elif arc_type == "trip_depot":
                    self.trip_depot[node] = (travel_min, dh_kwh)
                elif arc_type == "station_depot":
                    self.station_depot[node] = (travel_min, dh_kwh)

        self._build_nodes()
        self._build_arcs()

    # ── nodes ─────────────────────────────────────────────────────────────
    def _build_nodes(self):
        p = self.problem
        self.node_meta = [("source", None, None)]  # id 0
        self.SINK = 1
        self.node_meta.append(("sink", None, None))
        self.trip_node: dict[tuple[int, int], int] = {}
        self.charge_node: dict[tuple[str, int, int], int] = {}

        order = []  # (time_key, tiebreak, node_id)
        for trip in p.trips:
            for level in range(len(self.grid)):
                if self.grid[level] + 1e-9 >= p.trip_energy[trip] + self.reserve:
                    node_id = len(self.node_meta)
                    self.node_meta.append(("trip", trip, level))
                    self.trip_node[(trip, level)] = node_id
                    order.append((p.start_min[trip], 0, node_id))
        for station in STATIONS:
            for block in range(self.n_blocks):
                for level in range(len(self.grid)):
                    node_id = len(self.node_meta)
                    self.node_meta.append(("charge", (station, block), level))
                    self.charge_node[(station, block, level)] = node_id
                    order.append((block * self.block_min, 1, node_id))
        order.sort()
        self.topo = [0] + [node_id for _, _, node_id in order] + [1]

    # ── arcs ──────────────────────────────────────────────────────────────
    def _price(self, station: str, minute: float) -> float:
        curve = self.prices[base_station_name(station)]
        hour = int(minute // 60)
        if self.strict_tariff_coverage and hour not in curve:
            raise ValueError(
                f"tariff omits hour {hour} for {base_station_name(station)}"
            )
        return curve.get(hour, curve[max(curve)])

    def _charge_result(self, level: int) -> float:
        return self.grid[self._floor(min(self.g, self.grid[level] + self.block_kwh))]

    def _build_arcs(self):
        p = self.problem
        grid, floor = self.grid, self._floor
        # out[node] = list of (succ_id, base_cost, dense_trip_dual_index_or_-1)
        self.out: list[list] = [[] for _ in self.node_meta]
        def add(u, v, cost, trip=-1):
            dual_index = (
                self.trip_position[trip] if trip >= 0 else -1
            )
            self.out[u].append((v, cost, dual_index))

        # source -> trip
        for trip, (travel, dh) in self.depot_trip.items():
            if travel <= p.start_min[trip] + 1e-9:
                level = floor(self.g - dh)
                node = self.trip_node.get((trip, level))
                if node is not None:
                    add(0, node, BUS_COST_KX, trip)

        for (trip, level), u in self.trip_node.items():
            soc_exit = grid[level] - p.trip_energy[trip]
            depart = p.end_min[trip]
            # trip -> sink
            if trip in self.trip_depot:
                travel, dh = self.trip_depot[trip]
                if depart + travel <= HORIZON_MIN + 1e-9 and soc_exit - dh >= self.reserve - 1e-9:
                    add(u, 1, 0.0)
            # trip -> trip
            for succ, travel, dh in self.trip_trip.get(trip, ()):  # gap-filtered upstream
                nxt = floor(soc_exit - dh)
                if grid[nxt] > soc_exit - dh + 1e-9:
                    nxt -= 1
                if nxt < 0:
                    continue
                if depart + travel > p.start_min[succ] + 1e-9:
                    continue
                node = self.trip_node.get((succ, nxt))
                if node is not None:
                    add(u, node, 0.0, succ)
            # trip -> charge(station, any block starting after arrival)
            for station, travel, dh in self.trip_station.get(trip, ()):
                arrival = depart + travel
                soc_arr = soc_exit - dh
                if soc_arr < self.reserve - 1e-9:
                    continue
                lvl = floor(soc_arr)
                if grid[lvl] > soc_arr + 1e-9:
                    lvl -= 1
                if lvl < 0:
                    continue
                first_block = int(math.ceil(arrival / self.block_min - 1e-9))
                for block in range(max(first_block, 0), self.n_blocks):
                    add(u, self.charge_node[(station, block, lvl)], CHARGE_START_COST)

        for (station, block, level), u in self.charge_node.items():
            soc_after = self._charge_result(level)
            gained = soc_after - grid[level]
            cost = gained * self._price(station, block * self.block_min) if gained > 1e-9 else 0.0
            after_level = floor(soc_after)
            block_end = (block + 1) * self.block_min
            # continue charging next block (only if something was gained)
            if gained > 1e-9 and block + 1 < self.n_blocks:
                add(u, self.charge_node[(station, block + 1, after_level)], cost)
            # leave to a trip after this block
            for succ, travel, dh in self.station_trip.get(station, ()):
                if block_end + travel > p.start_min[succ] + 1e-9:
                    continue
                nxt = floor(soc_after - dh)
                if grid[nxt] > soc_after - dh + 1e-9:
                    nxt -= 1
                if nxt < 0:
                    continue
                node = self.trip_node.get((succ, nxt))
                if node is not None:
                    add(u, node, cost, succ)
            # leave to sink
            if station in self.station_depot:
                travel, dh = self.station_depot[station]
                if block_end + travel <= HORIZON_MIN + 1e-9 and soc_after - dh >= self.reserve - 1e-9:
                    add(u, 1, cost)

        self.n_arcs = sum(len(a) for a in self.out)
        self.sink_arcs = tuple(
            (u, cost)
            for u, arcs in enumerate(self.out[2:], start=2)
            for successor, cost, _dual_index in arcs
            if successor == self.SINK
        )

    # ── exact pricing pass ────────────────────────────────────────────────
    def min_reduced_cost_route(
        self,
        alpha: dict[int, float],
        *,
        objective: str = "combined-cost",
        route_dual: float = 0.0,
    ):
        if objective not in {
            "combined-cost", "artificial-elimination",
            "fleet-only", "charging-cost",
        }:
            raise ValueError(f"unsupported pricing objective: {objective}")
        INF = float("inf")
        dense_duals = [
            float(alpha.get(trip, 0.0)) for trip in self.problem.trips
        ]
        value = [INF] * len(self.node_meta)
        parent: list[tuple[int, int] | None] = [None] * len(self.node_meta)
        value[0] = 0.0
        for u in self.topo:
            vu = value[u]
            if vu == INF:
                continue
            for v, cost, dual_index in self.out[u]:
                if objective == "combined-cost" and route_dual == 0.0:
                    cand = (
                        vu + cost
                        - (dense_duals[dual_index] if dual_index >= 0 else 0.0)
                    )
                else:
                    objective_cost = (
                        0.0 if objective == "artificial-elimination"
                        else 1.0 if objective == "fleet-only" and u == 0
                        else 0.0 if objective == "fleet-only"
                        else cost - BUS_COST_KX
                        if objective == "charging-cost" and u == 0
                        else cost
                    )
                    cand = vu + objective_cost - (
                        dense_duals[dual_index]
                        if dual_index >= 0 else 0.0
                    ) - (route_dual if u == 0 else 0.0)
                if cand < value[v] - 1e-12:
                    value[v] = cand
                    parent[v] = (u, dual_index)
        if value[1] == INF:
            return None

        def _walk(from_node):
            """Reconstruct the full path: ordered trips + charging events."""
            nodes, node = [], from_node
            while node != 0:
                nodes.append(node)
                node = parent[node][0]
            nodes.reverse()  # source-side first (sink excluded when from_node=1? no: included)

            trips, stops = [], []
            run = None  # open charging run: [station, first_block, last_block, entry_level]
            for nid in nodes:
                kind, key, level = self.node_meta[nid]
                if kind == "trip":
                    if run is not None:
                        stops.append(run)
                        run = None
                    trips.append(key)
                elif kind == "charge":
                    station, block = key
                    if run is not None and run[0] == station and block == run[2] + 1:
                        run[2] = block
                    else:
                        if run is not None:
                            stops.append(run)
                        run = [station, block, block, level]
            if run is not None:
                stops.append(run)

            charging = {"stations": [], "cst": [], "cet": [], "kwh": []}
            route_nodes = [DEPOT]
            # interleave trips and stops in path order for the route node list
            seq = []
            for nid in nodes:
                kind, key, level = self.node_meta[nid]
                if kind == "trip":
                    seq.append(("t", key))
                elif kind == "charge":
                    if not seq or seq[-1] != ("s", key[0]):
                        seq.append(("s", key[0]))
            # collapse consecutive same-station markers (one per charging run)
            collapsed = []
            for item in seq:
                if not collapsed or item != collapsed[-1]:
                    collapsed.append(item)
            for tag, val in collapsed:
                route_nodes.append(val if tag == "t" else val)
            route_nodes.append(DEPOT)

            for station, b0, b1, lvl0 in stops:
                soc = self.grid[lvl0]
                for _ in range(b0, b1 + 1):
                    soc = self.grid[self._floor(min(self.g, soc + self.block_kwh))]
                charging["stations"].append(station)
                charging["cst"].append(b0 * self.block_min)
                charging["cet"].append((b1 + 1) * self.block_min)
                charging["kwh"].append(round(soc - self.grid[lvl0], 6))
            expanded_grid_charging = deepcopy(charging)
            realized, detail = realize_expanded_path(
                self.problem,
                {
                    "trips": trips,
                    "charging_stops": charging,
                    "route_nodes": route_nodes,
                },
                g_kwh=self.g,
                charge_kw=self.charge_kw,
                reserve_kwh=self.reserve,
                soc_step=self.soc_step,
                block_min=self.block_min,
                arc_map=self.continuous_arc_map,
            )
            if realized is None:
                raise RuntimeError(
                    "expanded path has no deterministic continuous "
                    f"realization: {detail.get('reason')}"
                )
            return (
                trips,
                realized["charging_stops"],
                route_nodes,
                detail["mapping"],
                expanded_grid_charging,
            )

        (
            best_trips,
            best_charging,
            best_nodes,
            best_mapping,
            best_grid_charging,
        ) = _walk(1)
        return {"rc": value[1], "trips": best_trips,
                "charging_stops": best_charging, "route_nodes": best_nodes,
                "charges_started": len(best_charging["stations"]),
                "_continuous_mapping": best_mapping,
                "_expanded_grid_charging": best_grid_charging,
                "_value": value, "_walk": _walk}

    def k_best_routes(
        self,
        alpha: dict[int, float],
        k: int = 30,
        *,
        phase_callback=None,
        objective: str = "combined-cost",
        route_dual: float = 0.0,
    ):
        """Best route plus up to k-1 additional negative columns from the same
        pass: min-cost paths ending at the k best distinct sink-predecessors."""
        started = time.perf_counter()
        if objective == "combined-cost" and route_dual == 0.0:
            best = self.min_reduced_cost_route(alpha)
        else:
            best = self.min_reduced_cost_route(
                alpha, objective=objective, route_dual=route_dual,
            )
        if phase_callback is not None:
            phase_callback(
                "pricing_shortest_path",
                time.perf_counter() - started,
                {"path_found": best is not None},
            )
        if best is None:
            if phase_callback is not None:
                phase_callback(
                    "pricing_extra_columns",
                    0.0,
                    {"sink_candidates": 0, "returned_routes": 0},
                )
            return []
        started = time.perf_counter()
        value, _walk = best.pop("_value"), best.pop("_walk")
        candidates = []
        for u, cost in self.sink_arcs:
            if value[u] == float("inf"):
                continue
            sink_cost = (
                cost if objective in {"combined-cost", "charging-cost"}
                else 0.0
            )
            candidates.append((value[u] + sink_cost, u))
        candidates.sort()
        routes, seen = [best], {frozenset(best["trips"])}
        for rc, u in candidates[: max(4 * k, 200)]:
            if len(routes) >= k or rc >= -1e-9:
                break
            (
                trips,
                charging,
                route_nodes,
                mapping,
                grid_charging,
            ) = _walk(u)
            key = frozenset(trips)
            if key in seen:
                continue
            seen.add(key)
            routes.append({"rc": rc, "trips": trips,
                           "charging_stops": charging, "route_nodes": route_nodes,
                           "charges_started": len(charging["stations"]),
                           "_continuous_mapping": mapping,
                           "_expanded_grid_charging": grid_charging})
        if phase_callback is not None:
            phase_callback(
                "pricing_extra_columns",
                time.perf_counter() - started,
                {
                    "sink_candidates": len(candidates),
                    "returned_routes": len(routes),
                },
            )
        return routes


def _provenance(args) -> dict:
    import platform
    import scipy
    import subprocess

    def _git(*a):
        r = subprocess.run(["git", *a], cwd=Path(__file__).resolve().parent,
                           text=True, capture_output=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else None

    def _sha(path):
        import hashlib
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "scipy": scipy.__version__,
        "instance_sha256": _sha(DATA_DIR / args.csv),
        "prices_sha256": _sha(DATA_DIR / args.prices_csv),
        "reference_sha256": _sha(DATA_DIR / "Ref_dict.csv"),
        "deadhead_sha256": _sha(DATA_DIR / "par_ref_dhd.csv"),
        "validated_seed_routes_sha256": (
            _sha(args.validated_seed_routes)
            if getattr(args, "validated_seed_routes", None) is not None
            else None
        ),
        "column_pool_treatment": (
            getattr(args, "augmentation_label", None)
            if getattr(args, "validated_seed_routes", None) is not None
            else "RAW"
        ),
        "rc_eps": args.rc_eps,
        "pricing_cost_semantics": "conservative_expanded_grid_cost",
        "charging_realization_schema":
            "evsp-dr-expanded-path-continuous-realization-v1",
        "continuous_cost_pricing_certified": False,
        "pricing_certificate_scope":
            "conservative_expanded_grid_model_only",
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key != "phase_telemetry"
        },
    }


ITERATION_LOG_HEADER = (
    "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,pool_columns"
)


def load_iteration_log(path: Path, *, repair_trailing: bool) -> list[list[str]]:
    """Read the append-only CG trajectory with narrow tail repair.

    Every data row has seven numeric fields.  Only an interrupted final row is
    repairable; an invalid header or interior row is evidence corruption.
    """

    path = Path(path)
    rows: list[list[str]] = []
    repair_offset = None
    last_valid_had_newline = True
    with open(path, "rb") as fh:
        header_offset = fh.tell()
        header = fh.readline()
        if not header:
            return rows
        try:
            decoded_header = header.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise DurableFileError(f"{path} has a non-UTF8 CSV header") from exc
        if decoded_header != ITERATION_LOG_HEADER:
            raise DurableFileError(
                f"{path} has unexpected iteration CSV header at byte "
                f"{header_offset}: {decoded_header!r}"
            )
        last_valid_had_newline = header.endswith(b"\n")
        while True:
            offset = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.strip():
                last_valid_had_newline = line.endswith(b"\n")
                continue
            valid = True
            try:
                fields = line.decode("utf-8").strip().split(",")
                if len(fields) != 7:
                    valid = False
                else:
                    for index, field in enumerate(fields):
                        value = float(field)
                        # A pricing iteration with no source-to-sink path is
                        # recorded as min_rc=+inf.  That is a meaningful
                        # terminal observation, not CSV corruption.  No other
                        # field may be infinite, and NaN/-inf are never valid.
                        if index == 5:
                            field_is_valid = (
                                math.isfinite(value)
                                or value == math.inf
                            )
                        else:
                            field_is_valid = math.isfinite(value)
                        if not field_is_valid:
                            valid = False
                            break
            except (UnicodeDecodeError, ValueError):
                valid = False
            if not valid:
                remainder = fh.read()
                if remainder.strip():
                    raise DurableFileError(
                        f"{path} has malformed iteration data before EOF at "
                        f"byte {offset}; refusing automatic repair"
                    )
                if not repair_trailing:
                    raise DurableFileError(
                        f"{path} has a malformed final iteration row at byte "
                        f"{offset}"
                    )
                repair_offset = offset
                break
            rows.append(fields)
            last_valid_had_newline = line.endswith(b"\n")

    if repair_trailing:
        if repair_offset is not None:
            with open(path, "r+b") as fh:
                fh.truncate(repair_offset)
                flush_and_fsync(fh)
        elif not last_valid_had_newline:
            with open(path, "ab") as fh:
                fh.write(b"\n")
                flush_and_fsync(fh)
    return rows


def resume_identity_mismatches(status, args, trips, provenance) -> list[str]:
    """Describe why persisted exact-CG state cannot belong to this run.

    This check deliberately depends only on the status and immutable model
    inputs.  It must run before a journal or iteration log is repaired so an
    incompatible or unidentified artifact is never modified as a side effect
    of a failed ``--resume`` attempt.
    """

    if not isinstance(status, dict):
        return ["status is missing, unreadable, or not a JSON object"]
    mismatches = []
    # Before this selector existed, exact CG unconditionally constructed the
    # direct-singleton pool.  Treat that one legacy absence as the historical
    # ``singletons`` identity while requiring explicit identity on every new
    # status written below.  An artificial-mode resume still fails closed.
    current_initial_pool = getattr(args, "initial_pool", "singletons")
    expected = {
        "csv": args.csv,
        "prices_csv": args.prices_csv,
        "soc_step": args.soc_step,
        "block_min": args.block_min,
        "strict_tariff_coverage": getattr(
            args, "strict_tariff_coverage", False
        ),
        "g_kwh": args.g_kwh,
        "charge_kw": args.charge_kw,
        "min_soc_frac": args.min_soc_frac,
        "master_sense": args.master_sense,
        "initial_pool": current_initial_pool,
        "validated_seed_routes_sha256": (
            _file_sha256(Path(args.validated_seed_routes))
            if getattr(args, "validated_seed_routes", None) is not None
            else None
        ),
        "column_pool_treatment": (
            getattr(args, "augmentation_label", None)
            if getattr(args, "validated_seed_routes", None) is not None
            else "RAW"
        ),
    }
    for key, value in expected.items():
        observed = status.get(key)
        if key == "initial_pool" and key not in status:
            observed = "singletons"
        if key == "strict_tariff_coverage" and key not in status:
            observed = False
        if key == "column_pool_treatment" and key not in status:
            observed = "RAW"
        if isinstance(value, float):
            try:
                matches = math.isclose(
                    float(observed), value, rel_tol=0.0, abs_tol=1e-9
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == value
        if not matches:
            mismatches.append(
                f"{key} differs (saved={observed!r}, current={value!r})"
            )
    if status.get("trip_ids") != trips:
        mismatches.append("trip_ids differ from the current instance")
    prior_provenance = status.get("provenance") or {}
    if not isinstance(prior_provenance, dict):
        mismatches.append("saved provenance is not a JSON object")
        prior_provenance = {}
    for key in (
        "instance_sha256", "prices_sha256",
        "reference_sha256", "deadhead_sha256",
    ):
        saved = prior_provenance.get(key)
        current = provenance.get(key)
        if saved != current:
            mismatches.append(
                f"{key} differs or is missing "
                f"(saved={saved!r}, current={current!r})"
            )
    saved_commit = prior_provenance.get("git_commit")
    current_commit = provenance.get("git_commit")
    parent = status.get("resume_parent") or {}
    if not isinstance(parent, dict):
        mismatches.append("saved resume_parent is not a JSON object")
        parent = {}
    if (str(parent.get("schema", "")).startswith(
            "evsp-dr-legacy-exact-pool-migration")
            and (not saved_commit or not current_commit)):
        mismatches.append(
            "attested legacy migration is missing saved or current "
            "git_commit identity"
        )
    if bool(saved_commit) != bool(current_commit):
        mismatches.append(
            "git_commit identity is present on only one side of resume "
            f"(saved={saved_commit!r}, current={current_commit!r})"
        )
    elif (saved_commit and current_commit
          and saved_commit != current_commit):
        mismatches.append(
            "git_commit differs; continue through an explicit, attested "
            f"migration (saved={saved_commit}, current={current_commit})"
        )
    return mismatches


def load_column_pool(records: list[dict], trip_ids: list[int]) -> dict:
    """Validate journal records and retain the cheapest realization per set."""

    pool: dict[frozenset, dict] = {}
    allowed = set(trip_ids)
    for index, record in enumerate(records, start=1):
        record_trips = record.get("trips")
        raw_cost = record.get("cost")
        if not isinstance(record_trips, list):
            raise DurableFileError(
                f"column journal record {index} has no trips list"
            )
        if not record_trips and allowed:
            raise DurableFileError(
                f"column journal record {index} contains no trips"
            )
        try:
            cost = float(raw_cost)
        except (TypeError, ValueError) as exc:
            raise DurableFileError(
                f"column journal record {index} has a non-numeric cost"
            ) from exc
        if not math.isfinite(cost):
            raise DurableFileError(
                f"column journal record {index} has a non-finite cost"
            )
        try:
            unique_trips = set(record_trips)
        except TypeError as exc:
            raise DurableFileError(
                f"column journal record {index} contains an unhashable trip"
            ) from exc
        if len(record_trips) != len(unique_trips):
            raise DurableFileError(
                f"column journal record {index} repeats a trip"
            )
        unknown = [trip for trip in record_trips if trip not in allowed]
        if unknown:
            raise DurableFileError(
                f"column journal record {index} contains trips outside the "
                f"current instance: {unknown[:10]}"
            )
        key = frozenset(record_trips)
        if key not in pool or cost < float(pool[key]["cost"]) - 1e-9:
            pool[key] = record
    return pool


def resume_pool_mismatches(status, pool: dict) -> list[str]:
    """Validate status claims that require the repaired journal contents."""

    mismatches = []
    try:
        recorded_columns = int(status.get("columns", 0))
    except (TypeError, ValueError):
        recorded_columns = -1
    if recorded_columns < 0:
        mismatches.append("saved column count is invalid")
    elif recorded_columns > len(pool):
        mismatches.append(
            f"saved status records {recorded_columns} columns but the journal "
            f"contains only {len(pool)} unique incidences"
        )
    pool_keys = set(pool)
    final_lp = status.get("final_lp") or {}
    if not isinstance(final_lp, dict):
        mismatches.append("saved final_lp is not a JSON object")
        return mismatches
    positive_routes = final_lp.get("positive_routes", [])
    if not isinstance(positive_routes, list):
        mismatches.append("saved final_lp positive_routes is not a list")
        return mismatches
    for route in positive_routes:
        if not isinstance(route, dict):
            mismatches.append("saved final_lp contains a non-object route")
            break
        try:
            route_key = frozenset(route.get("trips", []))
        except TypeError:
            mismatches.append("saved final_lp route has invalid trips")
            break
        if route_key not in pool_keys:
            mismatches.append(
                "journal does not contain every positive route in final_lp"
            )
            break
    return mismatches


def run_cg(args) -> dict:
    t0 = time.time()
    termination = {"requested": False, "signal": None}
    prior_signal_handlers = {}

    def request_termination(signum, _frame):
        termination["requested"] = True
        try:
            termination["signal"] = signal.Signals(signum).name
        except ValueError:
            termination["signal"] = str(signum)

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        prior_signal_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_termination)
    prior_status = None
    out_path = Path(args.out) if args.out else None
    journal_path = Path(str(args.out) + ".columns.jsonl") if args.out else None
    iters_path = Path(str(args.out) + ".iters.csv") if args.out else None
    if args.resume and out_path and out_path.exists():
        try:
            with open(out_path) as fh:
                prior_status = json.load(fh)
        except (OSError, ValueError):
            prior_status = None
    problem = build_problem(DATA_DIR, args.csv,
                            max_station_to_trip_wait_min=HORIZON_MIN)
    provenance = _provenance(args)
    trips = list(problem.trips)
    pool: dict[frozenset, dict] = {}
    history = []
    stall_hist = []  # (elapsed_s, lp_obj, min_rc) for --stall-window-min
    last_good_lp_detail = None
    certified = False
    stop_reason = "max_iters"
    stall_count = 0
    method_order = ("highs-ds", "highs-ipm", "highs")

    persisted_paths = [
        path for path in (out_path, journal_path, iters_path)
        if path is not None and path.exists()
    ]
    if persisted_paths and not args.resume:
        rendered = ", ".join(str(path) for path in persisted_paths)
        raise DurableFileError(
            f"refusing to overwrite persisted exact-CG artifacts without "
            f"--resume: {rendered}; use a new --out path"
        )

    identity_mismatches = resume_identity_mismatches(
        prior_status, args, trips, provenance
    )
    if args.resume and persisted_paths and identity_mismatches:
        raise DurableFileError(
            f"refusing --resume for {journal_path} before modifying persisted "
            "artifacts: " + "; ".join(identity_mismatches)
        )

    if args.resume and journal_path and journal_path.exists():
        # A hard preemption can interrupt only the last append.  Repair that
        # narrow case only after status/input identity has been established;
        # never hide interior corruption.
        journal_records = read_jsonl_records(
            journal_path, repair_trailing=True
        )
        pool = load_column_pool(journal_records, trips)
        print(f"[EXACT] resumed {len(pool)} columns from {journal_path.name}",
              flush=True)
    # Do not open append handles until resume identity and any interrupted
    # immutable snapshot publication have been validated.
    journal = None

    # Per-iteration stopping-rule instrumentation: append-only CSV so the
    # timing campaign can reconstruct the full LP trajectory with wall time.
    iters_csv = None
    iters_fresh = False
    iteration_rows = []
    elapsed_offset = 0.0
    iteration_offset = 0
    if out_path:
        iters_fresh = not (args.resume and iters_path.exists()
                           and iters_path.stat().st_size > 0)
        if not iters_fresh:
            iteration_rows = load_iteration_log(
                iters_path, repair_trailing=True
            )
            if iteration_rows:
                # Row count protects old resumed logs whose iteration number
                # restarted at one; the last recorded number preserves a
                # synthetic snapshot anchor such as iteration 1200.
                elapsed_offset = float(iteration_rows[-1][0])
                iteration_offset = max(
                    len(iteration_rows), int(float(iteration_rows[-1][1]))
                )
                stall_hist = [
                    (float(fields[0]), float(fields[2]), float(fields[5]))
                    for fields in iteration_rows
                    if math.isfinite(float(fields[5]))
                ]

    pool_mismatches = (
        resume_pool_mismatches(prior_status, pool)
        if args.resume and persisted_paths else []
    )
    if pool_mismatches:
        raise DurableFileError(
            f"refusing --resume for {journal_path}: "
            + "; ".join(pool_mismatches)
        )

    compatible_prior = bool(
        args.resume and persisted_paths
        and not identity_mismatches and not pool_mismatches
    )
    if compatible_prior:
        try:
            elapsed_offset = max(
                elapsed_offset, float(prior_status.get("wall_s", 0.0))
            )
            iteration_offset = max(
                iteration_offset, int(prior_status.get("iterations", 0))
            )
        except (TypeError, ValueError):
            pass

    telemetry = None
    telemetry_path = getattr(args, "phase_telemetry", None)
    if telemetry_path is not None:
        resolved_telemetry = Path(telemetry_path).expanduser().resolve()
        protected = {
            path.expanduser().resolve()
            for path in (out_path, journal_path, iters_path)
            if path is not None
        }
        protected.update({
            (DATA_DIR / args.csv).resolve(),
            (DATA_DIR / args.prices_csv).resolve(),
        })
        if resolved_telemetry in protected:
            raise DurableFileError(
                "phase telemetry must not overwrite model inputs or persisted "
                "status/journal/iteration artifacts"
            )
        telemetry = PhaseTelemetry(
            resolved_telemetry,
            identity={
                "output": str(out_path.resolve()) if out_path else None,
                "csv": args.csv,
                "prices_csv": args.prices_csv,
                "instance_sha256": provenance.get("instance_sha256"),
                "prices_sha256": provenance.get("prices_sha256"),
                "git_commit": provenance.get("git_commit"),
                "soc_step": args.soc_step,
                "block_min": args.block_min,
                "g_kwh": args.g_kwh,
                "charge_kw": args.charge_kw,
                "min_soc_frac": args.min_soc_frac,
                "master_sense": args.master_sense,
                "initial_pool": args.initial_pool,
            },
        )
    detached_telemetry_overhead_s = 0.0

    def _telemetry_overhead_s():
        return (
            detached_telemetry_overhead_s
            + (telemetry.overhead_s if telemetry is not None else 0.0)
        )

    def _attempt_elapsed_s():
        return max(
            0.0, time.time() - t0 - _telemetry_overhead_s()
        )

    def _cumulative_elapsed_s():
        return elapsed_offset + _attempt_elapsed_s()

    network_t0 = time.time()
    prices = load_station_hourly_prices(
        DATA_DIR / args.prices_csv, CHARGING_STATIONS
    )
    net = ExpandedNetwork(
        problem, prices,
        soc_step=args.soc_step,
        block_min=args.block_min,
        g_kwh=args.g_kwh,
        charge_kw=args.charge_kw,
        reserve_kwh=args.min_soc_frac * args.g_kwh,
        strict_tariff_coverage=getattr(
            args, "strict_tariff_coverage", False
        ),
    )
    build_s = time.time() - network_t0
    print(f"[EXACT] network: {len(net.node_meta):,} nodes, {net.n_arcs:,} arcs "
          f"(soc_step={args.soc_step}, block={args.block_min}min) "
          f"built in {build_s:.1f}s", flush=True)
    def _record_phase(
        name,
        duration_s,
        *,
        iteration=None,
        attempt=None,
        pool_columns=None,
        incidence_nnz=None,
        outcome="ok",
        details=None,
    ):
        nonlocal telemetry, detached_telemetry_overhead_s
        if telemetry is None:
            return
        try:
            resolved_details = details() if callable(details) else details
            telemetry.phase(
                name,
                duration_s,
                iteration=iteration,
                attempt=attempt,
                pool_columns=(
                    len(pool) if pool_columns is None else pool_columns
                ),
                incidence_nnz=incidence_nnz,
                network_nodes=len(net.node_meta),
                network_arcs=net.n_arcs,
                outcome=outcome,
                details=resolved_details,
            )
        except Exception as exc:
            detached_telemetry_overhead_s += telemetry.overhead_s
            telemetry = None
            warning_started = time.perf_counter()
            print(
                f"[EXACT] WARNING: phase telemetry disabled after I/O error: "
                f"{exc}",
                flush=True,
            )
            detached_telemetry_overhead_s += (
                time.perf_counter() - warning_started
            )

    _record_phase(
        "network_build",
        build_s,
        pool_columns=len(pool),
    )

    # Immutable timed pool snapshots (status + journal copy) for CG-vs-MIP
    # budget calibration, e.g. --snapshot-at-minutes 15,60,180,360.
    requested_snapshot_marks = sorted(
        float(m) for m in str(args.snapshot_at_minutes or "").split(",")
        if m.strip())
    snapshot_marks = []

    # Reconcile immutable snapshots only after the compatible prior status has
    # contributed its elapsed time.  This also handles the narrow publication
    # interruption where the frozen journal copy landed but its status JSON did
    # not: recovery is permitted only from the matching snapshot_mN status.
    orphan_snapshots = []
    if args.out:
        snapshot_stem = Path(str(args.out).replace(".json", ""))
        for mark in requested_snapshot_marks:
            snap_json = Path(f"{snapshot_stem}.m{int(mark)}.snapshot.json")
            snap_journal = Path(str(snap_json) + ".columns.jsonl")
            if mark * 60.0 <= elapsed_offset + 1e-9:
                if snap_json.exists():
                    if (not valid_json_object(
                            snap_json, ("trip_ids", "columns_journal"))
                            or not snap_journal.exists()):
                        if iters_csv:
                            iters_csv.close()
                        if journal:
                            journal.close()
                        raise DurableFileError(
                            f"immutable snapshot {snap_json} is incomplete or "
                            "corrupt; preserve it for diagnosis and choose a "
                            "new output stem before resuming"
                        )
                    read_jsonl_records(
                        snap_journal, repair_trailing=False, collect=False
                    )
                    print(f"[EXACT] snapshot {snap_json.name} already frozen — "
                          "keeping the original", flush=True)
                elif snap_journal.exists():
                    orphan_snapshots.append((mark, snap_json, snap_journal))
                else:
                    print(f"[EXACT] snapshot mark {mark:g} min was crossed in "
                          "an earlier allocation but no immutable snapshot "
                          "exists — recording it as missed, not fabricating a "
                          "late pool", flush=True)
                continue
            snapshot_marks.append(mark)
    else:
        snapshot_marks = requested_snapshot_marks

    for mark, snap_json, snap_journal in orphan_snapshots:
        expected_stop = f"snapshot_m{int(mark)}"
        if (not compatible_prior
                or prior_status.get("stop_reason") != expected_stop
                or prior_status.get("trip_ids") != trips):
            if iters_csv:
                iters_csv.close()
            if journal:
                journal.close()
            raise DurableFileError(
                f"orphan snapshot journal {snap_journal} cannot be published: "
                f"the compatible prior status must have stop_reason "
                f"{expected_stop!r} and the same trip_ids"
            )
        orphan_records = read_jsonl_records(
            snap_journal, repair_trailing=False
        )
        orphan_pool = set()
        try:
            for record in orphan_records:
                record_trips = record["trips"]
                record_cost = float(record["cost"])
                if (not isinstance(record_trips, list)
                        or not math.isfinite(record_cost)):
                    raise ValueError("invalid trips or cost")
                orphan_pool.add(frozenset(record_trips))
        except (KeyError, TypeError, ValueError) as exc:
            raise DurableFileError(
                f"orphan snapshot journal {snap_journal} contains a record "
                "without a valid trips list and finite cost"
            ) from exc
        try:
            recorded_columns = int(prior_status["columns"])
        except (KeyError, TypeError, ValueError) as exc:
            if iters_csv:
                iters_csv.close()
            if journal:
                journal.close()
            raise DurableFileError(
                f"orphan snapshot journal {snap_journal} has no trustworthy "
                "column count in its prior status"
            ) from exc
        if recorded_columns < 0 or len(orphan_pool) != recorded_columns:
            if iters_csv:
                iters_csv.close()
            if journal:
                journal.close()
            raise DurableFileError(
                f"orphan snapshot journal {snap_journal} contains "
                f"{len(orphan_pool)} unique route incidences, but the prior "
                f"status records {recorded_columns} columns"
            )
        for route in (prior_status.get("final_lp") or {}).get(
                "positive_routes", []):
            if frozenset(route.get("trips", [])) not in orphan_pool:
                raise DurableFileError(
                    f"orphan snapshot journal {snap_journal} does not contain "
                    "every positive route recorded by the prior status"
                )
        snapshot_status = dict(prior_status)
        snapshot_status["columns_journal"] = str(snap_journal)
        snapshot_status["snapshot_mark_minutes"] = mark
        atomic_write_json(snap_json, snapshot_status)
        print(f"[EXACT] recovered interrupted snapshot publication at "
              f"{mark:g} min: {snap_json.name}", flush=True)

    resume_parent = (prior_status or {}).get("resume_parent")
    if args.resume and compatible_prior and args.out:
        # A terminal status from the previous allocation must not remain
        # visible while this allocation is actively extending its journal.
        # Campaign discovery can therefore distinguish a live canonical pool
        # from an immutable *.snapshot.json file.
        resume_status = dict(prior_status)
        resume_status.update({
            "initial_pool": args.initial_pool,
            "validated_seed_routes_sha256": provenance.get(
                "validated_seed_routes_sha256"
            ),
            "column_pool_treatment": (
                getattr(args, "augmentation_label", None)
                if getattr(args, "validated_seed_routes", None)
                else "RAW"
            ),
            "trip_ids": trips,
            "columns": len(pool),
            "columns_journal": str(journal_path),
            "wall_s": _cumulative_elapsed_s(),
            "attempt_wall_s": _attempt_elapsed_s(),
            "stop_reason": "resume_starting",
            "provenance": provenance,
        })
        atomic_write_json(Path(args.out), resume_status)
        print("[EXACT] published live resume status before extending the "
              "journal", flush=True)
    elif args.out and not persisted_paths:
        # A fresh run can be preempted after its first journal append but
        # before the periodic checkpoint.  Publish identity first so that
        # journal-ahead-of-status is safely resumable from iteration zero.
        initial_status = {
            "csv": args.csv,
            "prices_csv": args.prices_csv,
            "soc_step": args.soc_step,
            "block_min": args.block_min,
            "strict_tariff_coverage": getattr(
                args, "strict_tariff_coverage", False
            ),
            "g_kwh": args.g_kwh,
            "charge_kw": args.charge_kw,
            "min_soc_frac": args.min_soc_frac,
            "master_sense": args.master_sense,
            "initial_pool": args.initial_pool,
            "validated_seed_routes_sha256": provenance.get(
                "validated_seed_routes_sha256"
            ),
            "column_pool_treatment": (
                getattr(args, "augmentation_label", None)
                if getattr(args, "validated_seed_routes", None)
                else "RAW"
            ),
            "trip_ids": trips,
            "iterations": 0,
            "attempt_iterations": 0,
            "certified_rc_optimal": False,
            "final": None,
            "columns": 0,
            "columns_journal": str(journal_path),
            "wall_s": _attempt_elapsed_s(),
            "attempt_wall_s": _attempt_elapsed_s(),
            "stop_reason": "initializing",
            "history_tail": [],
            "final_lp": None,
            "final_lp_source": None,
            "provenance": provenance,
            "resume_parent": None,
        }
        atomic_write_json(Path(args.out), initial_status)
        print("[EXACT] published initial identity before first journal append",
              flush=True)

    journal = open(journal_path, "a") if journal_path else None
    if iters_path is not None:
        iters_csv = open(iters_path, "a")
        if iters_fresh:
            iters_csv.write(ITERATION_LOG_HEADER + "\n")
            started = time.perf_counter()
            flush_and_fsync(iters_csv)
            _record_phase(
                "iteration_log_fsync",
                time.perf_counter() - started,
                iteration=iteration_offset,
                details=lambda: {"header": True},
            )
    if (compatible_prior
            and isinstance(prior_status.get("final_lp"), dict)):
        last_good_lp_detail = dict(prior_status["final_lp"])
        last_good_lp_detail["source"] = "compatible_prior_result"
        print("[EXACT] retained compatible prior final LP as a resume "
              "fallback", flush=True)
    def _serialize_lp(lp_result, lp_routes, *, source, iteration, pool_columns):
        return {
            "objective": lp_result.objective,
            "route_weight": lp_result.route_weight,
            "artificial_total": lp_result.artificial_total,
            "positive_routes": [
                {"trips": lp_routes[i]["trips"], "value": value,
                 "cost": lp_routes[i]["cost"]}
                for i, value in enumerate(lp_result.route_values)
                if value > 0.0
            ],
            "trip_duals": {
                str(key): value for key, value in lp_result.trip_duals.items()
            },
            "source": source,
            "iteration": iteration,
            "pool_columns": pool_columns,
            "max_row_violation": lp_result.max_row_violation,
            "max_bound_violation": lp_result.max_bound_violation,
            "feasibility_tolerance": lp_result.feasibility_tolerance,
            "master_method": lp_result.backend.method,
        }

    def _freeze_snapshot_impl(mark):
        if not args.out:
            return
        stem = Path(str(args.out).replace(".json", ""))
        snap_json = Path(f"{stem}.m{int(mark)}.snapshot.json")
        snap_journal = Path(str(snap_json) + ".columns.jsonl")
        if snap_json.exists():
            # snapshots are immutable: a requeued run must never overwrite the
            # original N-minute pool with a later one
            if (not valid_json_object(
                    snap_json, ("trip_ids", "columns_journal"))
                    or not snap_journal.exists()):
                raise DurableFileError(
                    f"immutable snapshot {snap_json} is incomplete or corrupt; "
                    "refusing to replace it silently"
                )
            read_jsonl_records(
                snap_journal, repair_trailing=False, collect=False
            )
            print(f"[EXACT] snapshot {snap_json.name} already frozen — keeping "
                  "the original", flush=True)
            return
        _write_partial(f"snapshot_m{int(mark)}")
        if journal:
            flush_and_fsync(journal)
        if journal_path and journal_path.exists():
            atomic_copy(journal_path, snap_journal)
            # Validate the frozen copy before publishing the status JSON that
            # makes the pair discoverable to launchers.
            read_jsonl_records(
                snap_journal, repair_trailing=False, collect=False
            )
        with open(args.out) as fh:
            snap = json.load(fh)
        snap["trip_ids"] = trips
        snap["columns_journal"] = str(snap_journal)
        snap["snapshot_mark_minutes"] = mark
        atomic_write_json(snap_json, snap)
        print(f"[EXACT] froze snapshot at {mark:g} min: {snap_json.name}",
              flush=True)

    def _freeze_snapshot(mark):
        started = time.perf_counter()
        try:
            _freeze_snapshot_impl(mark)
        except Exception as exc:
            _record_phase(
                "snapshot",
                time.perf_counter() - started,
                outcome="error",
                details=lambda: {
                    "mark_minutes": mark, "error": repr(exc),
                },
            )
            raise
        _record_phase(
            "snapshot",
            time.perf_counter() - started,
            details=lambda: {"mark_minutes": mark},
        )

    def _elapsed_s():
        return _cumulative_elapsed_s()

    def _freeze_crossed_snapshots():
        """Publish every due mark from the last completed durable state."""

        frozen = []
        while snapshot_marks and _elapsed_s() >= snapshot_marks[0] * 60.0:
            mark = snapshot_marks.pop(0)
            _freeze_snapshot(mark)
            frozen.append(mark)
        return frozen

    validated_seed_sha256 = None
    if getattr(args, "validated_seed_routes", None) is not None:
        seed_records, validated_seed_sha256 = (
            validated_fixed_duty_seed_records(
                Path(args.validated_seed_routes),
                problem,
                prices,
                tariff_path=DATA_DIR / args.prices_csv,
                g_kwh=args.g_kwh,
                charge_kw=args.charge_kw,
                reserve_kwh=args.min_soc_frac * args.g_kwh,
                soc_step=args.soc_step,
                block_min=args.block_min,
            )
        )
        seed_added = 0
        for record in seed_records:
            key = frozenset(record["trips"])
            if key not in pool or record["cost"] < pool[key]["cost"] - 1e-9:
                pool[key] = record
                if journal:
                    journal.write(json.dumps(record) + "\n")
                seed_added += 1
        if journal and seed_added:
            flush_and_fsync(journal)
        print(
            "[EXACT] tariff-specific validated fixed-duty seeds: "
            f"{len(seed_records)} routes ({seed_added} added), "
            f"sha256={validated_seed_sha256}",
            flush=True,
        )

    if args.initial_pool == "singletons":
        singleton_seeds, missing_singletons = direct_singleton_seed_records(
            problem,
            g_kwh=args.g_kwh,
            soc_step=args.soc_step,
            reserve_kwh=args.min_soc_frac * args.g_kwh,
        )
        seeds_added = 0
        for record in singleton_seeds:
            record["cost_tariff_sha256"] = provenance["prices_sha256"]
            key = frozenset(record["trips"])
            if key not in pool or record["cost"] < pool[key]["cost"] - 1e-9:
                pool[key] = record
                if journal:
                    journal.write(json.dumps(record) + "\n")
                seeds_added += 1
        if journal and seeds_added:
            started = time.perf_counter()
            flush_and_fsync(journal)
            _record_phase(
                "journal_fsync",
                time.perf_counter() - started,
                iteration=iteration_offset,
                details=lambda: {
                    "records": seeds_added, "origin": "singleton_seed",
                },
            )
        print(
            f"[EXACT] direct-singleton seed: "
            f"{len(singleton_seeds)}/{len(trips)} trips feasible "
            f"({seeds_added} added to pool)",
            flush=True,
        )
        if missing_singletons:
            print(
                "[EXACT] WARNING: direct-singleton seed is not a full "
                f"partition; missing {len(missing_singletons)} trips "
                f"({missing_singletons[:15]}).",
                flush=True,
            )
    else:
        if pool:
            print(
                "[EXACT] initial-pool policy: artificial; direct singleton "
                f"construction is disabled (continuing {len(pool)} resumed "
                "columns)",
                flush=True,
            )
        else:
            print(
                "[EXACT] initial pool: artificial-only; direct singleton "
                "construction is disabled and the first pricing iteration "
                "uses Big-M artificial duals",
                flush=True,
            )

    def _write_partial(status):
        if not args.out:
            return
        started = time.perf_counter()
        partial = {
            "csv": args.csv, "prices_csv": args.prices_csv,
            "soc_step": args.soc_step, "block_min": args.block_min,
            "strict_tariff_coverage": getattr(
                args, "strict_tariff_coverage", False
            ),
            "g_kwh": args.g_kwh, "charge_kw": args.charge_kw,
            "min_soc_frac": args.min_soc_frac,
            "master_sense": args.master_sense,
            "initial_pool": args.initial_pool,
            "validated_seed_routes_sha256": provenance.get(
                "validated_seed_routes_sha256"
            ),
            "column_pool_treatment": (
                getattr(args, "augmentation_label", None)
                if getattr(args, "validated_seed_routes", None)
                else "RAW"
            ),
            "trip_ids": trips,
            "iterations": iteration_offset + len(history),
            "attempt_iterations": len(history),
            "certified_rc_optimal": certified,
            "final": history[-1] if history else None,
            "columns": len(pool),
            "columns_journal": str(journal_path) if journal_path else None,
            "wall_s": _cumulative_elapsed_s(),
            "attempt_wall_s": _attempt_elapsed_s(),
            "stop_reason": status, "history_tail": history[-5:],
            "final_lp": last_good_lp_detail,
            "final_lp_source": (last_good_lp_detail or {}).get("source"),
            "provenance": provenance,
            "resume_parent": resume_parent,
        }
        atomic_write_json(args.out, partial)
        _record_phase(
            "status_checkpoint",
            time.perf_counter() - started,
            iteration=iteration_offset + len(history),
            details=lambda: {"stop_reason": status},
        )
    class _ArtificialOnlyLP:
        objective = len(trips) * BIG_M_PENALTY
        route_weight = 0.0
        artificial_total = float(len(trips))
        trip_duals = {t: float(BIG_M_PENALTY) for t in trips}

    def _remaining_wall_s(reserve_s=0.0):
        if not args.wall_limit_s:
            return None
        remaining = (
            args.wall_limit_s - _cumulative_elapsed_s() - reserve_s
        )
        return max(0.0, remaining)

    def _master_attempt_time_limit(reserve_s=0.0):
        """Return ``(seconds, snapshot_limited)`` for one master attempt."""

        while True:
            _freeze_crossed_snapshots()
            wall_budget = _remaining_wall_s(reserve_s=reserve_s)
            if wall_budget is not None and wall_budget <= 0.0:
                return 0.0, False
            if not snapshot_marks:
                return wall_budget, False
            snapshot_budget = snapshot_marks[0] * 60.0 - _elapsed_s()
            if snapshot_budget <= 0.0:
                # The clock crossed between the publication check and budget
                # calculation.  Publish that mark before allowing a solve.
                continue
            if wall_budget is None:
                return snapshot_budget, True
            snapshot_limited = snapshot_budget < wall_budget
            return min(wall_budget, snapshot_budget), snapshot_limited

    for iteration in range(1, args.max_iters + 1):
        global_iteration = iteration_offset + iteration
        if termination["requested"]:
            print(
                "[EXACT] external termination requested; publishing the "
                "last complete durable state",
                flush=True,
            )
            stop_reason = "external_signal"
            break
        if args.wall_limit_s and _remaining_wall_s(reserve_s=60.0) <= 0.0:
            print(f"[EXACT] cumulative wall limit {args.wall_limit_s}s "
                  "reached (with a 60s serialization margin) — stopping "
                  "gracefully (partial result saved)", flush=True)
            stop_reason = "wall_limit"
            break
        if args.out and iteration % args.checkpoint_every == 0:
            _write_partial("running")
        routes = list(pool.values())
        incidence_nnz = 0
        if routes:
            started = time.perf_counter()
            incidence = build_route_incidence(
                trip_ids=trips,
                route_trip_ids=[r["trips"] for r in routes],
            )
            incidence_nnz = int(getattr(incidence, "nnz", 0))
            incidence_shape = getattr(
                incidence, "shape", (len(trips), len(routes))
            )
            _record_phase(
                "incidence_construction",
                time.perf_counter() - started,
                iteration=global_iteration,
                pool_columns=len(routes),
                incidence_nnz=incidence_nnz,
                details=lambda: {
                    "rows": incidence_shape[0],
                    "columns": incidence_shape[1],
                },
            )
            lp = None
            master_attempt = 0
            for method in method_order:
                while True:
                    method_limit, snapshot_limited = (
                        _master_attempt_time_limit(reserve_s=30.0)
                    )
                    if method_limit is not None and method_limit <= 0.0:
                        break
                    master_attempt += 1
                    started = time.perf_counter()
                    try:
                        lp = solve_restricted_master_lp(
                            trip_ids=trips,
                            route_incidence=incidence,
                            route_costs=[r["cost"] for r in routes],
                            artificial_penalty=BIG_M_PENALTY,
                            method=method,
                            coverage_sense=args.master_sense,
                            time_limit_s=method_limit,
                        )
                        _record_phase(
                            "master_attempt",
                            time.perf_counter() - started,
                            iteration=global_iteration,
                            attempt=master_attempt,
                            pool_columns=len(routes),
                            incidence_nnz=incidence_nnz,
                            details=lambda: {
                                "method": method,
                                "time_limit_s": method_limit,
                                "snapshot_limited": snapshot_limited,
                                "backend_runtime_s": lp.runtime_s,
                                "objective": lp.objective,
                            },
                        )
                        # A solver may overrun its requested limit.  Freeze any
                        # crossed mark before this newly completed LP becomes
                        # the checkpoint fallback.
                        _freeze_crossed_snapshots()
                        break
                    except Exception as exc:  # degenerate masters can stall
                        _record_phase(
                            "master_attempt",
                            time.perf_counter() - started,
                            iteration=global_iteration,
                            attempt=master_attempt,
                            pool_columns=len(routes),
                            incidence_nnz=incidence_nnz,
                            outcome="error",
                            details=lambda: {
                                "method": method,
                                "time_limit_s": method_limit,
                                "snapshot_limited": snapshot_limited,
                                "error": repr(exc),
                            },
                        )
                        # Preserve the pre-attempt pool/LP before either this
                        # method or another one is allowed to run again.
                        crossed = _freeze_crossed_snapshots()
                        wall_remaining = _remaining_wall_s(reserve_s=30.0)
                        retry_same_method = (
                            snapshot_limited
                            and bool(crossed)
                            and (wall_remaining is None
                                 or wall_remaining > 0.0)
                        )
                        if retry_same_method:
                            print(
                                f"[EXACT] master reached snapshot boundary "
                                f"with {method}: {exc}; snapshot frozen, "
                                "retrying the same method",
                                flush=True,
                            )
                            continue
                        print(f"[EXACT] master failed with {method}: {exc}; "
                              "retrying with next method", flush=True)
                        break
                if lp is not None:
                    break
            if lp is None:
                # Evaluate wall exhaustion at this exit path itself: the final
                # method attempt may have consumed the remaining budget before
                # raising, and there is no later loop iteration to notice.  A
                # timed-out attempt is a graceful, resumable wall stop, not
                # evidence that every master method failed.
                remaining_wall_s = _remaining_wall_s(reserve_s=30.0)
                if remaining_wall_s is not None and remaining_wall_s <= 0.0:
                    print(f"[EXACT] cumulative wall limit {args.wall_limit_s}s "
                          "reached during the master solve — stopping "
                          "gracefully (partial result saved)", flush=True)
                    stop_reason = "wall_limit"
                else:
                    print("[EXACT] all master methods failed — stopping "
                          "uncertified", flush=True)
                    stop_reason = "master_failed"
                break
            last_good_lp_detail = _serialize_lp(
                lp, routes, source="last_good_iterate",
                iteration=global_iteration, pool_columns=len(routes),
            )
        else:
            lp = _ArtificialOnlyLP()
        if telemetry is None:
            batch = net.k_best_routes(
                lp.trip_duals, k=args.columns_per_iter
            )
        else:
            def pricing_phase(name, duration_s, details):
                _record_phase(
                    name,
                    duration_s,
                    iteration=global_iteration,
                    pool_columns=len(routes),
                    incidence_nnz=incidence_nnz,
                    details=details,
                )

            batch = net.k_best_routes(
                lp.trip_duals,
                k=args.columns_per_iter,
                phase_callback=pricing_phase,
            )
        best = batch[0] if batch else None
        min_rc = best["rc"] if best else float("inf")
        history.append({"iter": global_iteration,
                        "attempt_iter": iteration,
                        "lp_obj": lp.objective,
                        "route_weight": lp.route_weight,
                        "artificials": lp.artificial_total, "min_rc": min_rc,
                        "max_row_violation": getattr(
                            lp, "max_row_violation", 0.0),
                        "max_bound_violation": getattr(
                            lp, "max_bound_violation", 0.0)})
        if iters_csv:
            iters_csv.write(f"{_cumulative_elapsed_s():.2f},"
                            f"{global_iteration},"
                            f"{lp.objective:.6f},{lp.route_weight:.9f},"
                            f"{lp.artificial_total:.6f},{min_rc:.6f},{len(pool)}\n")
            started = time.perf_counter()
            flush_and_fsync(iters_csv)
            _record_phase(
                "iteration_log_fsync",
                time.perf_counter() - started,
                iteration=global_iteration,
                incidence_nnz=incidence_nnz,
            )
        _freeze_crossed_snapshots()
        if iteration % 10 == 0 or min_rc >= -args.rc_eps:
            print(f"[EXACT] it {global_iteration:3d}: obj={lp.objective:,.2f} "
                  f"weight={lp.route_weight:.4f} art={lp.artificial_total:.2f} "
                  f"min_rc={min_rc:,.3f}", flush=True)
        if best is None or min_rc >= -args.rc_eps:
            certified = best is not None
            stop_reason = "certified" if certified else "no_path"
            break
        if args.stall_window_min and lp.artificial_total < 1e-6:
            now = _cumulative_elapsed_s()
            quarter = args.stall_window_min * 60.0 / 4.0
            recent = [h for h in stall_hist if h[0] >= now - quarter]
            old = [h for h in stall_hist
                   if now - 4 * quarter <= h[0] <= now - 3 * quarter]
            if recent and old:
                rc_rec = min(h[2] for h in recent)   # most negative
                rc_old = min(h[2] for h in old)
                obj_rec = min(h[1] for h in recent)
                obj_old = min(h[1] for h in old)
                rc_impr = (abs(rc_old) - abs(rc_rec)) / max(1e-9, abs(rc_old))
                obj_impr = (obj_old - obj_rec) / max(1.0, abs(obj_old))
                # A negative rc_impr means the reduced-cost signal became
                # stronger (more negative), which is evidence to continue,
                # not evidence of a stall.
                if (0.0 <= rc_impr < args.stall_rc_frac
                        and obj_impr < args.stall_obj_frac):
                    print(f"[EXACT] marginal returns stalled over "
                          f"{args.stall_window_min:g} min: |min_rc| "
                          f"{abs(rc_old):,.2f}->{abs(rc_rec):,.2f} "
                          f"({100 * rc_impr:.1f}%), obj "
                          f"{obj_old:,.2f}->{obj_rec:,.2f} "
                          f"({100 * obj_impr:.4f}%) — stopping with an "
                          "INCUMBENT (not a certificate)", flush=True)
                    stop_reason = "stalled_marginal_returns"
                    break
            stall_hist.append((now, lp.objective, min_rc))
        elif args.stall_window_min:
            stall_hist.append((_cumulative_elapsed_s(),
                               lp.objective, min_rc))
        added = 0
        added_charge_starts = []
        started = time.perf_counter()
        for route in batch:
            cost = route["rc"] + sum(lp.trip_duals.get(t, 0.0) for t in route["trips"])
            key = frozenset(route["trips"])
            if key not in pool or cost < pool[key]["cost"] - 1e-9:
                record = {
                    "trips": route["trips"],           # ordered
                    "cost": cost,
                    "route_nodes": route["route_nodes"],
                    "charging_stops": route["charging_stops"],
                    "expanded_grid_charging_stops":
                        route["_expanded_grid_charging"],
                    "charges_started": route["charges_started"],
                    "found_iter": global_iteration,
                }
                mapping = route["_continuous_mapping"]
                costs = realized_costs(
                    record, mapping, station_prices=prices
                )
                record.update({
                    "expanded_grid_cost": cost,
                    "continuous_realized_cost":
                        costs["continuous_realized_cost"],
                    "continuous_realized_charging_blocks":
                        costs["continuous_realized_charging_blocks"],
                    "continuous_realized_charging_blocks_json_bytes":
                        len(json.dumps(
                            costs["continuous_realized_charging_blocks"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()),
                    "cost_semantics": "expanded_grid_cost",
                    "master_cost_semantics": "expanded_grid_cost",
                    "continuous_cost_pricing_certified": False,
                    "cost_tariff_sha256": provenance["prices_sha256"],
                    "physical_realization": {
                        key: value for key, value in mapping.items()
                        if key != "trace"
                    },
                })
                record["physical_realization"][
                    "continuous_realized_charging_blocks_sha256"
                ] = costs[
                    "continuous_realized_charging_blocks_sha256"
                ]
                record["physical_realization"][
                    "continuous_realized_charging_blocks_schema"
                ] = BLOCK_SCHEDULE_SCHEMA
                pool[key] = record
                if journal:
                    journal.write(json.dumps(record) + "\n")
                added += 1
                added_charge_starts.append(route["charges_started"])
        _record_phase(
            "route_insertion",
            time.perf_counter() - started,
            iteration=global_iteration,
            pool_columns=len(pool),
            incidence_nnz=incidence_nnz,
            details=lambda: {
                "candidate_routes": len(batch),
                "inserted_or_replaced": added,
            },
        )
        if journal and added:
            started = time.perf_counter()
            flush_and_fsync(journal)
            _record_phase(
                "journal_fsync",
                time.perf_counter() - started,
                iteration=global_iteration,
                pool_columns=len(pool),
                incidence_nnz=incidence_nnz,
                details=lambda: {
                    "records": added, "origin": "pricing",
                },
            )
            max_charge_starts = max(added_charge_starts)
            if max_charge_starts > MAX_DAILY_RECHARGES:
                print(f"[EXACT] note: column uses {max_charge_starts} charge "
                      f"starts (> cap {MAX_DAILY_RECHARGES}).", flush=True)
        if added == 0:
            # Every returned incidence already in the pool at equal cost: the
            # duals are frozen at a degenerate vertex. Interior-point duals
            # (analytic center of the optimal face) usually break the cycle;
            # only give up if the stall repeats under both dual sources.
            stall_count += 1
            if stall_count == 1:
                print("[EXACT] degenerate stall — switching to interior-point "
                      "duals and continuing", flush=True)
                method_order = ("highs-ipm", "highs-ds", "highs")
                continue
            print("[EXACT] stall persists under alternate duals — stopping "
                  "uncertified.", flush=True)
            stop_reason = "degenerate_stall"
            break
        stall_count = 0

    if args.diversify_rounds and pool:
        import random as _random
        rng = _random.Random(20260807)
        base_lp = None
        diversify_attempt = 0
        try:
            routes_now = list(pool.values())
            while True:
                method_limit, snapshot_limited = (
                    _master_attempt_time_limit(reserve_s=30.0)
                )
                if method_limit is not None and method_limit <= 0.0:
                    raise TimeoutError("no application time remains")
                started = time.perf_counter()
                diversify_incidence = build_route_incidence(
                    trip_ids=trips,
                    route_trip_ids=[r["trips"] for r in routes_now],
                )
                diversify_nnz = int(getattr(diversify_incidence, "nnz", 0))
                _record_phase(
                    "incidence_construction",
                    time.perf_counter() - started,
                    iteration=iteration_offset + len(history),
                    pool_columns=len(routes_now),
                    incidence_nnz=diversify_nnz,
                    details=lambda: {"purpose": "diversify"},
                )
                diversify_attempt += 1
                started = time.perf_counter()
                try:
                    candidate_lp = solve_restricted_master_lp(
                        trip_ids=trips,
                        route_incidence=diversify_incidence,
                        route_costs=[r["cost"] for r in routes_now],
                        artificial_penalty=BIG_M_PENALTY,
                        time_limit_s=method_limit,
                    )
                    _record_phase(
                        "master_attempt",
                        time.perf_counter() - started,
                        iteration=iteration_offset + len(history),
                        attempt=diversify_attempt,
                        pool_columns=len(routes_now),
                        incidence_nnz=diversify_nnz,
                        details=lambda: {
                            "method": "highs-ds",
                            "purpose": "diversify",
                            "time_limit_s": method_limit,
                            "snapshot_limited": snapshot_limited,
                            "backend_runtime_s": candidate_lp.runtime_s,
                        },
                    )
                    _freeze_crossed_snapshots()
                    base_lp = candidate_lp
                    break
                except Exception as exc:
                    _record_phase(
                        "master_attempt",
                        time.perf_counter() - started,
                        iteration=iteration_offset + len(history),
                        attempt=diversify_attempt,
                        pool_columns=len(routes_now),
                        incidence_nnz=diversify_nnz,
                        outcome="error",
                        details=lambda: {
                            "method": "highs-ds",
                            "purpose": "diversify",
                            "time_limit_s": method_limit,
                            "snapshot_limited": snapshot_limited,
                            "error": repr(exc),
                        },
                    )
                    crossed = _freeze_crossed_snapshots()
                    wall_remaining = _remaining_wall_s(reserve_s=30.0)
                    if (snapshot_limited and crossed
                            and (wall_remaining is None
                                 or wall_remaining > 0.0)):
                        continue
                    raise
        except Exception as exc:
            _freeze_crossed_snapshots()
            print(f"[EXACT] diversify: base LP failed ({exc}); skipping", flush=True)
        if base_lp is not None:
            added_div = 0
            for rnd in range(1, args.diversify_rounds + 1):
                alpha = {t_: v * (1.0 + rng.uniform(-args.diversify_delta,
                                                    args.diversify_delta))
                         for t_, v in base_lp.trip_duals.items()}
                if telemetry is None:
                    diversify_routes = net.k_best_routes(
                        alpha, k=args.columns_per_iter
                    )
                else:
                    def diversify_pricing_phase(name, duration_s, details):
                        _record_phase(
                            name,
                            duration_s,
                            iteration=iteration_offset + len(history),
                            pool_columns=len(pool),
                            details=lambda: {
                                **details,
                                "purpose": "diversify",
                                "round": rnd,
                            },
                        )

                    diversify_routes = net.k_best_routes(
                        alpha,
                        k=args.columns_per_iter,
                        phase_callback=diversify_pricing_phase,
                    )
                for route in diversify_routes:
                    cost = route["rc"] + sum(alpha.get(t_, 0.0)
                                             for t_ in route["trips"])
                    key = frozenset(route["trips"])
                    if key not in pool or cost < pool[key]["cost"] - 1e-9:
                        record = {
                            "trips": route["trips"], "cost": cost,
                            "route_nodes": route["route_nodes"],
                            "charging_stops": route["charging_stops"],
                            "expanded_grid_charging_stops":
                                route["_expanded_grid_charging"],
                            "charges_started": route["charges_started"],
                            "found_iter": -rnd,
                            "origin": "diversify",
                        }
                        mapping = route["_continuous_mapping"]
                        costs = realized_costs(
                            record, mapping, station_prices=prices
                        )
                        record.update({
                            "expanded_grid_cost": cost,
                            "continuous_realized_cost":
                                costs["continuous_realized_cost"],
                            "continuous_realized_charging_blocks":
                                costs[
                                    "continuous_realized_charging_blocks"
                                ],
                            "continuous_realized_charging_blocks_json_bytes":
                                len(json.dumps(
                                    costs[
                                        "continuous_realized_charging_blocks"
                                    ],
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode()),
                            "cost_semantics": "expanded_grid_cost",
                            "master_cost_semantics":
                                "expanded_grid_cost",
                            "continuous_cost_pricing_certified": False,
                            "cost_tariff_sha256":
                                provenance["prices_sha256"],
                            "physical_realization": {
                                key_: value for key_, value
                                in mapping.items() if key_ != "trace"
                            },
                        })
                        record["physical_realization"][
                            "continuous_realized_charging_blocks_sha256"
                        ] = costs[
                            "continuous_realized_charging_blocks_sha256"
                        ]
                        record["physical_realization"][
                            "continuous_realized_charging_blocks_schema"
                        ] = BLOCK_SCHEDULE_SCHEMA
                        pool[key] = record
                        if journal:
                            journal.write(json.dumps(record) + "\n")
                        added_div += 1
            if journal:
                started = time.perf_counter()
                flush_and_fsync(journal)
                _record_phase(
                    "journal_fsync",
                    time.perf_counter() - started,
                    iteration=iteration_offset + len(history),
                    details=lambda: {
                        "records": added_div,
                        "origin": "diversify",
                    },
                )
            print(f"[EXACT] diversify: {args.diversify_rounds} rounds added "
                  f"{added_div} complementary columns", flush=True)

    # Final LP over the persisted pool: store route values + duals so the
    # fractional solution is reconstructable without re-solving.
    final_lp_detail = None
    final_lp_source = None
    routes = list(pool.values())
    if routes:
        final_errors = []
        final_attempt = 0
        for method in ("highs-ds", "highs-ipm", "highs"):
            while True:
                method_limit, snapshot_limited = (
                    _master_attempt_time_limit(reserve_s=10.0)
                )
                if method_limit is not None and method_limit <= 0.0:
                    final_errors.append("no application time remains")
                    break
                started = time.perf_counter()
                final_incidence = build_route_incidence(
                    trip_ids=trips,
                    route_trip_ids=[r["trips"] for r in routes],
                )
                final_nnz = int(getattr(final_incidence, "nnz", 0))
                _record_phase(
                    "incidence_construction",
                    time.perf_counter() - started,
                    iteration=iteration_offset + len(history),
                    pool_columns=len(routes),
                    incidence_nnz=final_nnz,
                    details=lambda: {
                        "purpose": "final_resolve", "method": method,
                    },
                )
                final_attempt += 1
                started = time.perf_counter()
                try:
                    lp_final = solve_restricted_master_lp(
                        trip_ids=trips,
                        route_incidence=final_incidence,
                        route_costs=[r["cost"] for r in routes],
                        artificial_penalty=BIG_M_PENALTY,
                        coverage_sense=args.master_sense,
                        method=method,
                        time_limit_s=method_limit,
                    )
                    _record_phase(
                        "master_attempt",
                        time.perf_counter() - started,
                        iteration=iteration_offset + len(history),
                        attempt=final_attempt,
                        pool_columns=len(routes),
                        incidence_nnz=final_nnz,
                        details=lambda: {
                            "purpose": "final_resolve",
                            "method": method,
                            "time_limit_s": method_limit,
                            "snapshot_limited": snapshot_limited,
                            "backend_runtime_s": lp_final.runtime_s,
                        },
                    )
                    _freeze_crossed_snapshots()
                    final_lp_detail = _serialize_lp(
                        lp_final, routes, source="final_pool_resolve",
                        iteration=iteration_offset + len(history),
                        pool_columns=len(routes),
                    )
                    final_lp_source = "final_pool_resolve"
                    break
                except Exception as exc:
                    _record_phase(
                        "master_attempt",
                        time.perf_counter() - started,
                        iteration=iteration_offset + len(history),
                        attempt=final_attempt,
                        pool_columns=len(routes),
                        incidence_nnz=final_nnz,
                        outcome="error",
                        details=lambda: {
                            "purpose": "final_resolve",
                            "method": method,
                            "time_limit_s": method_limit,
                            "snapshot_limited": snapshot_limited,
                            "error": repr(exc),
                        },
                    )
                    crossed = _freeze_crossed_snapshots()
                    wall_remaining = _remaining_wall_s(reserve_s=10.0)
                    if (snapshot_limited and crossed
                            and (wall_remaining is None
                                 or wall_remaining > 0.0)):
                        continue
                    final_errors.append(f"{method}: {exc}")
                    break
            if final_lp_detail is not None:
                break
        if final_lp_detail is None:
            print("[EXACT] final LP re-solve failed with all methods: "
                  + " | ".join(final_errors), flush=True)
    if final_lp_detail is None and last_good_lp_detail is not None:
        # This LP is a valid solution over an earlier/compatible restricted
        # pool, but is not claimed optimal over the final enlarged pool.
        final_lp_detail = dict(last_good_lp_detail)
        final_lp_source = final_lp_detail.get("source", "last_good_iterate")
        final_lp_detail["source"] = final_lp_source

    _freeze_crossed_snapshots()
    snapshot_availability = {}
    if args.out:
        final_snapshot_stem = Path(str(args.out).replace(".json", ""))
        for mark in requested_snapshot_marks:
            snapshot = Path(
                f"{final_snapshot_stem}.m{int(mark)}.snapshot.json"
            )
            journal_path_for_snapshot = Path(
                str(snapshot) + ".columns.jsonl"
            )
            snapshot_availability[str(int(mark))] = (
                "available"
                if snapshot.is_file() and journal_path_for_snapshot.is_file()
                else "censored_solver_terminated_before_mark"
                if mark in snapshot_marks
                else "missed_in_prior_allocation"
            )
    if iters_csv:
        iters_csv.close()
    if journal:
        journal.close()

    result = {
        "csv": args.csv,
        "prices_csv": args.prices_csv,
        "soc_step": args.soc_step,
        "block_min": args.block_min,
        "strict_tariff_coverage": getattr(
            args, "strict_tariff_coverage", False
        ),
        "g_kwh": args.g_kwh,
        "charge_kw": args.charge_kw,
        "min_soc_frac": args.min_soc_frac,
        "master_sense": args.master_sense,
        "initial_pool": args.initial_pool,
        "validated_seed_routes_sha256": provenance.get(
            "validated_seed_routes_sha256"
        ),
        "column_pool_treatment": (
            getattr(args, "augmentation_label", None)
            if getattr(args, "validated_seed_routes", None)
            else "RAW"
        ),
        "trip_ids": trips,
        "iterations": iteration_offset + len(history),
        "attempt_iterations": len(history),
        "certified_rc_optimal": certified,
        "final": history[-1] if history else None,
        "columns": len(pool),
        "columns_journal": str(journal_path) if journal_path else None,
        "wall_s": _cumulative_elapsed_s(),
        "attempt_wall_s": _attempt_elapsed_s(),
        "stop_reason": stop_reason,
        "termination_signal": termination["signal"],
        "snapshot_availability": snapshot_availability,
        "history_tail": history[-5:],
        "final_lp": final_lp_detail,
        "final_lp_source": final_lp_source,
        "provenance": provenance,
        "resume_parent": resume_parent,
    }
    print(f"[EXACT] DONE: {json.dumps(result['final'], default=float)} "
          f"certified={certified} columns={len(pool)} "
          f"wall={result['wall_s']:.0f}s", flush=True)
    for signum, previous in prior_signal_handlers.items():
        signal.signal(signum, previous)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--prices_csv", default="hourly_prices_flat.csv")
    parser.add_argument("--soc-step", type=float, default=15.0)
    parser.add_argument("--block-min", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--columns_per_iter", type=int, default=30)
    parser.add_argument("--rc-eps", type=float, default=1e-4)
    parser.add_argument(
        "--master-sense",
        choices=("partition", "cover"),
        default="partition",
        help="Trip-row sense in the exact-CG restricted master. Partition is "
             "the operational default; cover reproduces legacy campaigns.",
    )
    parser.add_argument(
        "--initial-pool",
        choices=("singletons", "artificial"),
        default="singletons",
        help="Initial real-column pool. Singletons preserves the operational "
             "default; artificial starts pricing from Big-M artificials only.",
    )
    parser.add_argument(
        "--objective",
        choices=("combined-cost", "lexicographic-fleet"),
        default=argparse.SUPPRESS,
        help="Opt in to three-phase artificial/fleet/charging exact CG. "
             "Omitting this flag preserves the combined-cost path exactly.",
    )
    parser.add_argument(
        "--validated-seed-routes",
        type=Path,
        default=None,
        help=(
            "Tariff-specific Tier-1 exact partition to add as "
            "GIRO-AUGMENTED expanded-grid columns."
        ),
    )
    parser.add_argument(
        "--augmentation-label",
        choices=("GIRO-AUGMENTED", "GIRO40-AUGMENTED"),
        default=None,
        help="Scientific column-pool label required with validated seeds.",
    )
    parser.add_argument(
        "--strict-tariff-coverage",
        action="store_true",
        help="Reject any expanded block without an explicitly defined price.",
    )
    parser.add_argument("--stall-window-min", type=float, default=None,
                        help="Enable marginal-returns stopping: compare the "
                             "best min_rc and LP objective of the most recent "
                             "quarter-window against the quarter-window one "
                             "full window ago; stop when both improved less "
                             "than their thresholds and no artificials "
                             "remain. Off by default.")
    parser.add_argument("--stall-rc-frac", type=float, default=0.05,
                        help="Relative |min_rc| improvement below which the "
                             "pricing signal counts as stalled.")
    parser.add_argument("--stall-obj-frac", type=float, default=1e-5,
                        help="Relative LP-objective improvement below which "
                             "the master counts as stalled.")
    parser.add_argument("--wall-limit-s", type=int, default=None,
                        help="Stop gracefully after this many cumulative "
                             "journaled seconds across resumes (set below the "
                             "Slurm limit so results get written).")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Write the partial --out JSON every N iterations.")
    parser.add_argument("--g-kwh", type=float, default=300.0,
                        help="Battery capacity. GIRO telemetry implies ~239 kWh "
                             "usable; 300 is the historical model convention.")
    parser.add_argument("--charge-kw", type=float, default=CHARGE_RATE_KW,
                        help="Charger power. GIRO telemetry implies ~220 kW; "
                             "300 is the historical model convention.")
    parser.add_argument("--min-soc-frac", type=float, default=0.0,
                        help="SOC reserve as a fraction of capacity (FDL notes "
                             "require 0.2 for duties over 20h).")
    parser.add_argument("--diversify-rounds", type=int, default=0,
                        help="After the main loop, run N extra pricing rounds "
                             "against randomly perturbed duals to harvest "
                             "complementary columns (integrality repair; "
                             "columns are journaled, certification claims "
                             "are unaffected).")
    parser.add_argument("--diversify-delta", type=float, default=0.15,
                        help="Relative dual perturbation for diversify rounds.")
    parser.add_argument("--snapshot-at-minutes", default=None,
                        help="Comma-separated elapsed-minute marks at which to "
                             "freeze immutable pool snapshots (status+journal), "
                             "e.g. 15,60,180,360 for MIP-budget calibration.")
    parser.add_argument(
        "--phase-telemetry",
        type=Path,
        default=None,
        help="Optional durable JSONL sidecar for phase timing/RSS evidence. "
             "Operational only; excluded from model/resume identity.",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Reload the column journal next to --out and "
                             "continue from that pool.")
    parser.add_argument("--out", "--o", type=Path, default=None)
    args = parser.parse_args(argv)
    if bool(args.validated_seed_routes) != bool(args.augmentation_label):
        parser.error(
            "--validated-seed-routes and --augmentation-label are required "
            "together"
        )
    runner = run_cg
    if getattr(args, "objective", "combined-cost") == "lexicographic-fleet":
        from lexicographic_fleet_cg import run_lexicographic_fleet_cg
        runner = run_lexicographic_fleet_cg
    if args.out:
        lock_metadata = {
            "pid": os.getpid(),
            "host": os.uname().nodename,
            "started_epoch": time.time(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT"),
            "expected_commit": os.environ.get("EVSP_EXPECTED_COMMIT"),
        }
        with exclusive_output_lock(args.out, lock_metadata):
            result = runner(args)
            atomic_write_json(args.out, result)
    else:
        result = runner(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

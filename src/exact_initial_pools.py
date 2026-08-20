"""Deterministic non-historical initial pools for exact expanded-grid CG."""

from __future__ import annotations

import hashlib
import json
import math

from config import CHARGE_START_COST
from fixed_duty_expanded_optimizer import optimize_fixed_duty
from greedy_init import build_greedy_routes
from matching_init import build_matching_initial_routes


SCHEMA = "evsp-dr-exact-initial-pool-v1"


def _canonical(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def pool_sha256(records):
    """Hash the ordered, fully realized initial column records."""

    return hashlib.sha256(_canonical(records)).hexdigest()


def pool_provenance(mode, records, **details):
    return {
        "schema": SCHEMA,
        "initial_pool": mode,
        "generated_pool_sha256": pool_sha256(records),
        "generated_route_count": len(records),
        "uses_historical_partition": False,
        "uses_giro_partition": False,
        **details,
    }


def _trip_sequences(problem, mode, *, depot, stations, g_kwh, charge_kw,
                    reserve_kwh, soc_step, block_min):
    trips = tuple(problem.trips)
    if mode == "matching":
        levels = [
            round(index * soc_step, 6)
            for index in range(1, int(g_kwh / soc_step) + 1)
        ]
        proposed = build_matching_initial_routes(
            trips=trips,
            adjacency=problem.adjacency,
            depot=depot,
            stations=stations,
            trip_start_min=problem.start_min,
            trip_end_min=problem.end_min,
            trip_energy_kwh=problem.trip_energy,
            battery_capacity_kwh=g_kwh,
            charge_rate_kw=charge_kw,
            soc_charge_levels=levels,
            horizon_min=1560.0,
            max_station_to_trip_wait_min=1560.0,
            successor_boundary_soc_target=True,
            station_waiting_unrestricted=True,
            charge_start_cost=CHARGE_START_COST,
        )
        generator = "matching_init.build_matching_initial_routes"
        matching = dict((proposed[0].get("_matching_init") or {}))
        generator_details = {
            "heuristic_relaxed_route_count": matching.get(
                "relaxed_minimum_path_count", len(proposed)
            ),
            "heuristic_realized_route_count": len(proposed),
            "heuristic_contiguous_splits_added": matching.get(
                "contiguous_splits_added", 0
            ),
            "heuristic_resource_repair_mode": matching.get(
                "resource_repair_mode", "none"
            ),
            "matching_initialization": matching,
        }
    elif mode == "greedy":
        tau, tau_min, energy = {}, {}, {}
        for source, entries in problem.adjacency.items():
            for target, minutes, kwh, _kind in entries:
                tau[(source, target)] = int(math.ceil(minutes / block_min))
                tau_min[(source, target)] = float(minutes)
                energy[(source, target)] = float(kwh)
        proposed = build_greedy_routes(
            T=list(trips),
            S_use=list(stations),
            DEPOT=depot,
            tau=tau,
            tau_min=tau_min,
            d=energy,
            st={
                trip: int(problem.start_min[trip] // block_min)
                for trip in trips
            },
            et={
                trip: int(math.ceil(problem.end_min[trip] / block_min))
                for trip in trips
            },
            st_min=problem.start_min,
            et_min=problem.end_min,
            epsilon=problem.trip_energy,
            G=g_kwh,
            bar_t=int(1560 // block_min),
            TB_MIN=block_min,
            CHARGE_RATE_KW=charge_kw,
            max_trip2trip=1560.0,
            max_trip2charge=1560.0,
            max_charge2trip=1560.0,
            min_soc_fraction=reserve_kwh / g_kwh,
        )
        generator = "greedy_init.build_greedy_routes"
        generator_details = {
            "heuristic_relaxed_route_count": len(proposed),
            "heuristic_realized_route_count": len(proposed),
            "heuristic_contiguous_splits_added": 0,
            "heuristic_resource_repair_mode": "none",
        }
    else:
        raise ValueError(f"unsupported heuristic initial pool: {mode}")
    trip_set = set(trips)
    sequences = [
        tuple(node for node in route["route"] if node in trip_set)
        for route in proposed
    ]
    covered = [trip for sequence in sequences for trip in sequence]
    if (
        not sequences
        or len(covered) != len(trips)
        or len(set(covered)) != len(trips)
        or set(covered) != trip_set
    ):
        raise ValueError(f"{mode} initializer is not an exact trip partition")
    return sequences, generator, generator_details


def build_heuristic_initial_pool(
    problem,
    station_prices,
    *,
    mode,
    depot,
    stations,
    g_kwh,
    charge_kw,
    reserve_kwh,
    soc_step,
    block_min,
    tariff_sha256,
    instance_sha256,
):
    """Generate heuristic sequences and realize them in the exact route space."""

    sequences, generator, generator_details = _trip_sequences(
        problem,
        mode,
        depot=depot,
        stations=stations,
        g_kwh=g_kwh,
        charge_kw=charge_kw,
        reserve_kwh=reserve_kwh,
        soc_step=soc_step,
        block_min=block_min,
    )
    realization_prices = {
        station: {
            hour: curve.get(hour, curve[max(curve)])
            for hour in range(int(math.ceil(1560 / 60)))
        }
        for station, curve in station_prices.items()
    }
    cache = {}

    def realize(sequence):
        sequence = tuple(sequence)
        if sequence not in cache:
            cache[sequence] = optimize_fixed_duty(
                problem,
                sequence,
                realization_prices,
                g_kwh=g_kwh,
                charge_kw=charge_kw,
                reserve_kwh=reserve_kwh,
                soc_step=soc_step,
                block_min=block_min,
                tariff_id="exact-cg-initial-pool",
                tariff_sha256=tariff_sha256,
                instance_sha256=instance_sha256,
                allow_diagnostic_grid=(soc_step, block_min) != (15.0, 10),
            )
        result = cache[sequence]
        return result["route"] if result.get("feasible") is True else None

    def split(sequence):
        route = realize(sequence)
        if route is not None:
            return [route]
        for cut in range(len(sequence) - 1, 0, -1):
            prefix = realize(sequence[:cut])
            if prefix is not None:
                return [prefix, *split(sequence[cut:])]
        raise ValueError(
            f"{mode} initial sequence cannot be split into exact routes: "
            f"{list(sequence)}"
        )

    records = []
    for sequence in sequences:
        records.extend(split(sequence))
    for record in records:
        record.update({
            "origin": f"exact_{mode}_initial_seed",
            "initial_pool": mode,
            "found_iter": 0,
            "charges_started": len(
                (record.get("charging_stops") or {}).get("stations") or []
            ),
            "cost_semantics": "expanded_grid_cost",
            "continuous_cost_pricing_certified": False,
            "continuous_realized_charging_blocks_json_bytes": len(json.dumps(
                record.get("continuous_realized_charging_blocks") or [],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()),
        })
    trips = list(problem.trips)
    covered = [trip for record in records for trip in record["trips"]]
    if len(covered) != len(trips) or set(covered) != set(trips):
        raise ValueError(f"{mode} exact initial pool does not cover every trip")
    provenance = pool_provenance(
        mode,
        records,
        generator=generator,
        heuristic_route_count=generator_details[
            "heuristic_relaxed_route_count"
        ],
        **generator_details,
        exact_route_count=len(records),
        expanded_grid_contiguous_splits_added=(
            len(records) - len(sequences)
        ),
        total_contiguous_splits_added=(
            generator_details["heuristic_contiguous_splits_added"]
            + len(records) - len(sequences)
        ),
        realization="fixed_duty_expanded_optimizer",
        physics={
            "g_kwh": g_kwh,
            "charge_kw": charge_kw,
            "reserve_kwh": reserve_kwh,
            "soc_step": soc_step,
            "block_min": block_min,
        },
    )
    return records, provenance

import csv
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_giro_known_columns import (  # noqa: E402
    DEPOT,
    HORIZON_MIN,
    STATIONS,
    build_problem,
)
from config import BUS_COST_KX, CHARGE_START_COST  # noqa: E402
from fixed_duty_continuous_optimizer import (  # noqa: E402
    _Model,
    optimize_fixed_duty_continuous,
)
from fixed_duty_expanded_optimizer import (  # noqa: E402
    _arc_groups,
    optimize_fixed_duty,
)
from utils_v2 import base_station_name  # noqa: E402
from tariff_response_core import (  # noqa: E402
    giro_routes_for_instance,
    load_tariff_manifest,
    tariff_prices,
)


STATION = STATIONS[0]


def prices(default=1.0, overrides=None):
    overrides = overrides or {}
    return {
        station.rsplit("_", 1)[0]: {
            hour: float(overrides.get(hour, default))
            for hour in range(27)
        }
        for station in STATIONS
    }


def two_trip_problem(first_energy=190.0, second_energy=100.0):
    return SimpleNamespace(
        trips=[0, 1],
        trip_energy={0: first_energy, 1: second_energy},
        start_min={0: 0.0, 1: 180.0},
        end_min={0: 10.0, 1: 190.0},
        adjacency={
            DEPOT: [(0, 0.0, 0.0, "depot_trip")],
            0: [(STATION, 0.0, 0.0, "trip_station")],
            STATION: [
                (1, 0.0, 0.0, "station_trip"),
                (DEPOT, 0.0, 0.0, "station_depot"),
            ],
            1: [(DEPOT, 0.0, 0.0, "trip_depot")],
        },
    )


def grid_floor(grid, step, soc):
    level = min(
        max(int(math.floor((soc + 1e-9) / step)), 0), len(grid) - 1
    )
    return level - 1 if grid[level] > soc + 1e-9 else level


def lattice_candidates(
    problem, arcs, trip, successor, final, level, grid, step, block_min,
    g_kwh, charge_kw, reserve_kwh, station_prices,
):
    result = []
    after_trip = grid[level] - float(problem.trip_energy[trip])
    deadline = HORIZON_MIN if final else float(problem.start_min[successor])
    successor_energy = 0.0 if final else float(problem.trip_energy[successor])
    direct = (
        arcs["trip_depot"].get(trip) if final
        else arcs["trip_trip"].get(trip, {}).get(successor)
    )
    if direct is not None:
        remaining = after_trip - direct.deadhead_kwh
        target = grid_floor(grid, step, remaining)
        if (
            float(problem.end_min[trip]) + direct.travel_min <= deadline + 1e-9
            and remaining >= reserve_kwh - 1e-9
            and (
                final
                or grid[target] >= successor_energy + reserve_kwh - 1e-9
            )
        ):
            result.append((None if final else target, 0.0))
    inbound_by_station = arcs["trip_station"].get(trip, {})
    for station in sorted(set(STATIONS) | set(inbound_by_station)):
        inbound = inbound_by_station.get(station)
        outbound = (
            arcs["station_depot"].get(station)
            if final
            else arcs["station_trip"].get(station, {}).get(successor)
        )
        if inbound is None or outbound is None:
            continue
        arrival = float(problem.end_min[trip]) + inbound.travel_min
        arrival_soc = after_trip - inbound.deadhead_kwh
        entry = grid_floor(grid, step, arrival_soc)
        first = max(0, int(math.ceil(arrival / block_min - 1e-9)))
        last = min(
            int(HORIZON_MIN) // block_min - 1,
            int(math.floor(
                (deadline - outbound.travel_min) / block_min + 1e-9
            )) - 1,
        )
        if arrival_soc < reserve_kwh - 1e-9 or entry < 0 or last < first:
            continue
        curve = station_prices[base_station_name(station)]
        for start in range(first, last + 1):
            charge_level, energy_cost = entry, 0.0
            for end in range(start, last + 1):
                before = grid[charge_level]
                after_level = grid_floor(
                    grid,
                    step,
                    min(g_kwh, before + charge_kw * block_min / 60.0),
                )
                gain = grid[after_level] - before
                energy_cost += gain * float(curve[end * block_min // 60])
                remaining = grid[after_level] - outbound.deadhead_kwh
                target = grid_floor(grid, step, remaining)
                if (
                    remaining >= reserve_kwh - 1e-9
                    and (
                        final
                        or grid[target]
                        >= successor_energy + reserve_kwh - 1e-9
                    )
                ):
                    result.append((
                        None if final else target,
                        CHARGE_START_COST + energy_cost,
                    ))
                if gain <= 1e-9:
                    break
                charge_level = after_level
    return result


def solve_lattice(
    problem, trips, station_prices, *, g_kwh=300.0, charge_kw=300.0,
    reserve_kwh=0.0, step=15.0, block_min=10,
):
    trips = tuple(trips)
    grid = [
        round(index * step, 6)
        for index in range(int(g_kwh / step) + 1)
    ]
    arcs = _arc_groups(problem)
    first = arcs["depot_trip"][trips[0]]
    first_level = grid_floor(grid, step, g_kwh - first.deadhead_kwh)
    reachable, transitions = {first_level}, []
    for position, trip in enumerate(trips):
        final = position == len(trips) - 1
        successor = None if final else trips[position + 1]
        following = set()
        for level in sorted(reachable):
            cheapest = {}
            for target, cost in lattice_candidates(
                problem, arcs, trip, successor, final, level, grid, step,
                block_min, g_kwh, charge_kw, reserve_kwh, station_prices,
            ):
                cheapest[target] = min(cheapest.get(target, math.inf), cost)
            for target, cost in cheapest.items():
                transitions.append((position, level, target, cost))
                if target is not None:
                    following.add(target)
        reachable = following
    model = _Model()
    variables = [
        model.variable(cost=row[3], ub=1, integer=True)
        for row in transitions
    ]
    nodes = (
        {(row[0], row[1]) for row in transitions}
        | {
            (row[0] + 1, row[2])
            for row in transitions
            if row[2] is not None
        }
    )
    for position, level in sorted(nodes):
        incoming = [
            (variables[index], 1)
            for index, row in enumerate(transitions)
            if row[0] == position - 1 and row[2] == level
        ]
        outgoing = [
            (variables[index], -1)
            for index, row in enumerate(transitions)
            if row[0] == position and row[1] == level
        ]
        rhs = -1 if (position, level) == (0, first_level) else 0
        model.constraint(incoming + outgoing, lower=rhs, upper=rhs)
    terminal = [
        (variables[index], 1)
        for index, row in enumerate(transitions)
        if row[0] == len(trips) - 1 and row[2] is None
    ]
    if not terminal:
        return None
    model.constraint(terminal, lower=1, upper=1)
    solved = model.solve()
    return BUS_COST_KX + float(solved.fun) if solved.success else None


class ContinuousFixedDutyTests(unittest.TestCase):
    def test_non_frozen_physics_require_diagnostic_scope(self):
        with self.assertRaisesRegex(ValueError, "diagnostic_physics"):
            optimize_fixed_duty_continuous(
                two_trip_problem(),
                [0, 1],
                prices(),
                charge_kw=60.0,
            )

    def test_solver_limit_is_unknown_not_physical_infeasibility(self):
        limited = SimpleNamespace(
            success=False,
            x=None,
            status=1,
            message="Time limit reached",
        )
        with patch(
            "fixed_duty_continuous_optimizer._Model.solve",
            return_value=limited,
        ):
            result = optimize_fixed_duty_continuous(
                two_trip_problem(), [0, 1], prices()
            )
        self.assertIsNone(result["feasible"])
        self.assertEqual(result["classification"], "solver_limit")

    def test_delays_charge_into_cheap_hour_and_replays_cost(self):
        result = optimize_fixed_duty_continuous(
            two_trip_problem(),
            [0, 1],
            prices(overrides={0: 10.0}),
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["charge_events"], 1)
        self.assertEqual(result["delayed_starts"], 1)
        self.assertGreaterEqual(result["charging_events"][0]["start_min"], 60)
        self.assertAlmostEqual(result["objective"], 100055.0)
        self.assertAlmostEqual(
            result["objective"],
            result["physical_replay"]["replayed_objective"],
            places=6,
        )
        self.assertEqual(result["peak_kw"], 240.0)
        self.assertEqual(result["charger_concurrency_max"], 1)
        self.assertEqual(
            len(result["certificate"]["problem_identity_sha256"]), 64
        )
        self.assertEqual(
            len(result["certificate"]["tariff_curve_identity_sha256"]), 64
        )

    def test_fixed_cost_counts_two_events_around_expensive_hour(self):
        result = optimize_fixed_duty_continuous(
            two_trip_problem(first_energy=180.0, second_energy=160.0),
            [0, 1],
            prices(default=1.0, overrides={1: 100.0}),
            charge_kw=60.0,
            allow_diagnostic_physics=True,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["charge_events"], 2)
        self.assertEqual(
            [event["start_min"] for event in result["charging_events"]],
            [10.0, 120.0],
        )
        self.assertAlmostEqual(result["charging_cost"], 110.0, places=6)
        self.assertAlmostEqual(
            sum(event["delivered_kwh"] for event in result["charging_events"]),
            100.0,
            places=6,
        )

    def test_event_cap_limits_zero_cost_charge_fragmentation(self):
        arguments = {
            "problem": two_trip_problem(
                first_energy=180.0, second_energy=160.0
            ),
            "trip_sequence": [0, 1],
            "station_prices": prices(default=1.0, overrides={1: 100.0}),
            "charge_kw": 60.0,
            "charge_start_cost": 0.0,
            "allow_diagnostic_physics": True,
        }
        uncapped = optimize_fixed_duty_continuous(**arguments)
        capped = optimize_fixed_duty_continuous(
            **arguments, max_charge_events=1
        )
        self.assertEqual(uncapped["charge_events"], 2)
        self.assertEqual(capped["charge_events"], 1)
        self.assertGreater(capped["charging_cost"], uncapped["charging_cost"])
        self.assertEqual(capped["certificate"]["max_charge_events"], 1)
        with self.assertRaisesRegex(ValueError, "max_charge_events"):
            optimize_fixed_duty_continuous(
                **arguments, max_charge_events=-1
            )

    def test_terminal_policies_are_explicit_and_switchable(self):
        problem = SimpleNamespace(
            trips=[0],
            trip_energy={0: 100.0},
            start_min={0: 0.0},
            end_min={0: 10.0},
            adjacency={
                DEPOT: [(0, 0.0, 0.0, "depot_trip")],
                0: [
                    (DEPOT, 0.0, 0.0, "trip_depot"),
                    (STATION, 0.0, 0.0, "trip_station"),
                ],
                STATION: [(DEPOT, 0.0, 0.0, "station_depot")],
            },
        )
        free = optimize_fixed_duty_continuous(
            problem, [0], prices(), terminal_soc_policy="free"
        )
        restored = optimize_fixed_duty_continuous(
            problem, [0], prices(), terminal_soc_policy=">= start"
        )
        priced = optimize_fixed_duty_continuous(
            problem,
            [0],
            prices(),
            terminal_soc_policy="priced terminal energy",
            terminal_energy_price=2.0,
        )
        self.assertAlmostEqual(free["terminal_soc_kwh"], 140.0)
        self.assertEqual(free["charge_events"], 0)
        self.assertAlmostEqual(restored["terminal_soc_kwh"], 240.0)
        self.assertEqual(restored["charge_events"], 1)
        self.assertAlmostEqual(priced["terminal_soc_kwh"], 240.0)
        self.assertLess(priced["objective"], free["objective"])

    def test_flat_tariff_has_no_delayed_timing_advantage(self):
        problem = two_trip_problem()
        optimized = optimize_fixed_duty_continuous(
            problem, [0, 1], prices(), timing_mode="optimized"
        )
        arrival = optimize_fixed_duty_continuous(
            problem, [0, 1], prices(), timing_mode="arrival"
        )
        self.assertTrue(optimized["feasible"] and arrival["feasible"])
        self.assertTrue(math.isclose(
            optimized["objective"],
            arrival["objective"],
            rel_tol=0.0,
            abs_tol=1e-6,
        ))
        self.assertEqual(
            optimized["charge_events"], arrival["charge_events"]
        )

    def test_zero_charge_station_waypoint_keeps_replay_alignment(self):
        problem = SimpleNamespace(
            trips=[0, 1, 2],
            trip_energy={0: 10.0, 1: 200.0, 2: 100.0},
            start_min={0: 0.0, 1: 10.0, 2: 120.0},
            end_min={0: 10.0, 1: 20.0, 2: 130.0},
            adjacency={
                DEPOT: [(0, 0.0, 0.0, "depot_trip")],
                0: [(STATION, 0.0, 0.0, "trip_station")],
                1: [(STATION, 0.0, 0.0, "trip_station")],
                2: [(DEPOT, 0.0, 0.0, "trip_depot")],
                STATION: [
                    (1, 0.0, 0.0, "station_trip"),
                    (2, 0.0, 0.0, "station_trip"),
                ],
            },
        )
        result = optimize_fixed_duty_continuous(
            problem, [0, 1, 2], prices()
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["physical_replay_status"], "validated")
        self.assertFalse(any(
            kwh == 0.0 for kwh in result["route"]["charging_stops"]["kwh"]
        ))
        self.assertEqual(
            result["physical_replay"]["station_visits"][0][
                "charge_event_indices"
            ],
            [],
        )
        self.assertEqual(result["charge_events"], 1)

    def test_legacy_lattice_milp_reproduces_dynamic_program(self):
        problem = two_trip_problem(first_energy=250.0, second_energy=40.0)
        station_prices = prices(default=1.0)
        lattice = solve_lattice(problem, [0, 1], station_prices)
        reference = optimize_fixed_duty(
            problem, [0, 1], station_prices
        )
        self.assertTrue(reference["feasible"])
        self.assertAlmostEqual(
            lattice, reference["expanded_grid_objective"], places=7
        )


class RealDutyContinuousGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instance = (
            REPO_ROOT
            / "data/scale_ladder/instances/"
            "Practice_Custom_DutyUnion_k05_r1.csv"
        )
        cls.problem = build_problem(
            cls.instance.parent,
            cls.instance.name,
            max_station_to_trip_wait_min=HORIZON_MIN,
            reference_data_dir=REPO_ROOT / "data",
        )
        routes = giro_routes_for_instance(
            REPO_ROOT / "data/Par_VehicleDetails_Updated.csv",
            cls.instance,
        )
        cls.routes = {route["duty_id"]: route["trips"] for route in routes}
        manifest = load_tariff_manifest(
            REPO_ROOT / "data/tariff_response/tariff_manifest.csv"
        )
        by_id = {row["tariff_id"]: row for row in manifest}
        cls.prices = {
            tariff: tariff_prices(by_id[tariff])
            for tariff in ("flat", "peak12")
        }

    def test_g1_real_duty_lattice_reproduction(self):
        trips = self.routes["13412"]
        lattice = solve_lattice(
            self.problem, trips, self.prices["flat"]
        )
        reference = optimize_fixed_duty(
            self.problem, trips, self.prices["flat"]
        )
        self.assertEqual(lattice, 100058.152)
        self.assertEqual(lattice, reference["expanded_grid_objective"])

    def test_g2_relaxation_ordering(self):
        trips = self.routes["13412"]
        for tariff in ("flat", "peak12"):
            grid = optimize_fixed_duty(
                self.problem, trips, self.prices[tariff]
            )
            continuous = optimize_fixed_duty_continuous(
                self.problem,
                trips,
                self.prices[tariff],
                g_kwh=300,
                charge_kw=300,
                allow_diagnostic_physics=True,
            )
            self.assertTrue(grid["feasible"] and continuous["feasible"])
            self.assertLessEqual(
                continuous["objective"],
                grid["expanded_grid_objective"] + 1e-6,
            )

    def test_g3_g4_duty_13411_is_feasible_and_replays(self):
        result = optimize_fixed_duty_continuous(
            self.problem,
            self.routes["13411"],
            self.prices["peak12"],
            terminal_soc_policy=">= reserve",
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["physical_replay_status"], "validated")
        self.assertAlmostEqual(
            result["objective"],
            result["physical_replay"]["replayed_objective"],
            places=6,
        )

    def test_g5_flat_tariff_invariance(self):
        optimized = optimize_fixed_duty_continuous(
            self.problem, self.routes["13411"], self.prices["flat"]
        )
        arrival = optimize_fixed_duty_continuous(
            self.problem,
            self.routes["13411"],
            self.prices["flat"],
            timing_mode="arrival",
        )
        self.assertAlmostEqual(
            optimized["objective"], arrival["objective"], places=6
        )

    def test_all_k2_k3_k5_duties_replay_under_both_tariffs(self):
        manifest_path = (
            REPO_ROOT
            / "data/scale_ladder/instances/scale_ladder_instance_manifest.csv"
        )
        with manifest_path.open(newline="") as handle:
            instances = [
                row for row in csv.DictReader(handle)
                if int(row["scale"]) in {2, 3, 5}
            ]
        for item in instances:
            path = REPO_ROOT / item["relative_path"]
            problem = build_problem(
                path.parent,
                path.name,
                max_station_to_trip_wait_min=HORIZON_MIN,
                reference_data_dir=REPO_ROOT / "data",
            )
            routes = giro_routes_for_instance(
                REPO_ROOT / "data/Par_VehicleDetails_Updated.csv", path
            )
            for route in routes:
                for tariff in ("flat", "peak12"):
                    result = optimize_fixed_duty_continuous(
                        problem, route["trips"], self.prices[tariff]
                    )
                    self.assertTrue(
                        result["feasible"],
                        f"{item['scale']}/{item['selection_replicate']} "
                        f"{route['duty_id']} {tariff}: "
                        f"{result.get('reason')}",
                    )
                    self.assertEqual(
                        result["physical_replay_status"], "validated"
                    )


if __name__ == "__main__":
    unittest.main()

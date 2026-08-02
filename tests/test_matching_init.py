import inspect
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matching_init  # noqa: E402
from audit_giro_known_columns import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEPOT,
    STATIONS,
    build_problem,
)
from matching_init import (  # noqa: E402
    MatchingInitializationError,
    RouteRealizationError,
    build_matching_initial_routes,
    minimum_trip_path_cover,
    peak_trip_concurrency,
    realize_fixed_trip_path,
)


class MinimumPathCoverTests(unittest.TestCase):
    def test_minimum_cover_uses_only_active_trip_trip_arcs(self):
        adjacency = {
            "D": [(0, 0.0, 0.0, "depot_trip")],
            0: [
                (1, 0.0, 0.0, "trip_trip"),
                (2, 0.0, 0.0, "trip_trip"),
                ("S", 0.0, 0.0, "trip_station"),
            ],
            1: [(3, 0.0, 0.0, "trip_trip")],
            2: [(3, 0.0, 0.0, "trip_trip")],
            3: [(99, 0.0, 0.0, "trip_trip")],
        }

        paths = minimum_trip_path_cover(
            [0, 1, 2, 3],
            adjacency,
            trip_start_min={0: 0.0, 1: 10.0, 2: 10.0, 3: 20.0},
        )

        self.assertEqual(len(paths), 2)
        flattened = [trip for path in paths for trip in path]
        self.assertCountEqual(flattened, [0, 1, 2, 3])
        active_arcs = {
            (left, right)
            for left in (0, 1, 2, 3)
            for right, _time, _energy, kind in adjacency.get(left, ())
            if kind == "trip_trip" and right in {0, 1, 2, 3}
        }
        for path in paths:
            for edge in zip(path, path[1:]):
                self.assertIn(edge, active_arcs)

    def test_empty_graph_gives_one_path_per_trip(self):
        paths = minimum_trip_path_cover([4, 7, 9], {4: [], 7: [], 9: []})
        self.assertEqual(paths, [(4,), (7,), (9,)])

    def test_cycle_is_rejected_instead_of_misreported_as_a_cover(self):
        adjacency = {
            0: [(1, 0.0, 0.0, "trip_trip")],
            1: [(0, 0.0, 0.0, "trip_trip")],
        }
        with self.assertRaisesRegex(MatchingInitializationError, "not acyclic"):
            minimum_trip_path_cover([0, 1], adjacency)

    def test_station_bridge_reduces_path_count_but_requires_positive_charge(self):
        adjacency = {
            "D": [(0, 0.0, 0.0, "depot_trip")],
            0: [
                ("S", 5.0, 0.0, "trip_station"),
                ("D", 0.0, 0.0, "trip_depot"),
            ],
            "S": [(1, 5.0, 0.0, "station_trip")],
            1: [("D", 0.0, 0.0, "trip_depot")],
        }
        options = dict(
            trips=[0, 1],
            adjacency=adjacency,
            stations=["S"],
            trip_start_min={0: 0.0, 1: 100.0},
            trip_end_min={0: 10.0, 1: 110.0},
            trip_energy_kwh={0: 20.0, 1: 20.0},
            battery_capacity_kwh=100.0,
            charge_rate_kw=600.0,
            soc_charge_levels=[50.0, 100.0],
            horizon_min=200.0,
            max_station_to_trip_wait_min=100.0,
        )

        direct = minimum_trip_path_cover(**options, direct_only=True)
        bridged = minimum_trip_path_cover(**options)

        self.assertEqual(direct, [(0,), (1,)])
        self.assertEqual(bridged, [(0, 1)])

        # With a ten-minute charger-capacity window but an eighty-minute wait
        # that must be filled by charging, no positive-charge bridge exists.
        too_short = minimum_trip_path_cover(
            **{**options, "max_station_to_trip_wait_min": 10.0}
        )
        self.assertEqual(too_short, [(0,), (1,)])


class PeakConcurrencyTests(unittest.TestCase):
    def test_half_open_intervals_end_before_equal_time_starts(self):
        peak = peak_trip_concurrency(
            [0, 1, 2, 3],
            {0: 0.0, 1: 10.0, 2: 9.0, 3: 10.0},
            {0: 10.0, 1: 20.0, 2: 11.0, 3: 10.0},
        )
        # Trips 0 and 2 overlap before t=10.  Trip 0 ends exactly when trip 1
        # starts, and zero-duration trip 3 contributes no occupancy.
        self.assertEqual(peak, 2)

    def test_invalid_service_interval_is_rejected(self):
        with self.assertRaisesRegex(MatchingInitializationError, "ends before"):
            peak_trip_concurrency([0], {0: 5.0}, {0: 4.0})


class FixedPathRealizationTests(unittest.TestCase):
    def test_charge_realization_has_r_truck_schema_and_resources(self):
        adjacency = {
            "D": [(0, 0.0, 0.0, "depot_trip")],
            0: [
                (1, 0.0, 0.0, "trip_trip"),
                ("S", 0.0, 0.0, "trip_station"),
            ],
            "S": [(1, 0.0, 0.0, "station_trip")],
            1: [("D", 0.0, 0.0, "trip_depot")],
        }

        route = realize_fixed_trip_path(
            [0, 1],
            adjacency=adjacency,
            depot="D",
            stations=["S"],
            trip_start_min={0: 0.0, 1: 20.0},
            trip_end_min={0: 10.0, 1: 30.0},
            trip_energy_kwh={0: 70.0, 1: 50.0},
            battery_capacity_kwh=100.0,
            charge_rate_kw=600.0,
            soc_charge_levels=[50.0, 100.0],
            horizon_min=60.0,
            max_station_to_trip_wait_min=220.0,
        )

        self.assertEqual(route["route"], ["D", 0, "S", 1, "D"])
        self.assertEqual(route["charging_activities"], 1)
        self.assertEqual(route["charging_stops"]["stations"], ["S"])
        self.assertAlmostEqual(route["charging_stops"]["cst"][0], 10.0)
        self.assertAlmostEqual(route["charging_stops"]["cet"][0], 12.0)
        self.assertAlmostEqual(route["charging_stops"]["kwh"][0], 20.0)
        self.assertAlmostEqual(route["deadhead_kwh"], 0.0)
        self.assertEqual(route["type"], "truck")

    def test_resource_infeasible_minimum_cover_splits_into_singletons(self):
        adjacency = {
            "D": [
                (0, 0.0, 0.0, "depot_trip"),
                (1, 0.0, 0.0, "depot_trip"),
            ],
            0: [
                (1, 0.0, 0.0, "trip_trip"),
                ("D", 0.0, 0.0, "trip_depot"),
            ],
            1: [("D", 0.0, 0.0, "trip_depot")],
        }
        routes = build_matching_initial_routes(
            trips=[0, 1],
            adjacency=adjacency,
            depot="D",
            stations=[],
            trip_start_min={0: 0.0, 1: 10.0},
            trip_end_min={0: 5.0, 1: 15.0},
            trip_energy_kwh={0: 60.0, 1: 60.0},
            battery_capacity_kwh=100.0,
            horizon_min=30.0,
        )

        self.assertEqual(
            [[node for node in route["route"] if isinstance(node, int)] for route in routes],
            [[0], [1]],
        )
        provenance = routes[0]["_matching_init"]
        self.assertEqual(provenance["relaxed_minimum_path_count"], 1)
        self.assertEqual(provenance["resource_feasible_path_count"], 2)
        self.assertEqual(provenance["resource_repair_mode"], "contiguous_split")
        self.assertFalse(provenance["is_exact_minimum_path_cover"])
        self.assertEqual(provenance["contiguous_splits_added"], 1)

    def test_singleton_resource_infeasibility_still_fails_loudly(self):
        adjacency = {
            "D": [(0, 0.0, 0.0, "depot_trip")],
            0: [("D", 0.0, 0.0, "trip_depot")],
        }
        with self.assertRaises(RouteRealizationError) as caught:
            build_matching_initial_routes(
                trips=[0],
                adjacency=adjacency,
                depot="D",
                stations=[],
                trip_start_min={0: 0.0},
                trip_end_min={0: 5.0},
                trip_energy_kwh={0: 101.0},
                battery_capacity_kwh=100.0,
                horizon_min=30.0,
            )
        self.assertEqual(caught.exception.failed_transition, ("D", 0))

    def test_alternate_maximum_matching_repairs_first_energy_infeasible_tie(self):
        adjacency = {
            "D": [
                (0, 0.0, 0.0, "depot_trip"),
                (1, 0.0, 0.0, "depot_trip"),
                (2, 0.0, 0.0, "depot_trip"),
                (3, 0.0, 0.0, "depot_trip"),
            ],
            0: [
                (2, 0.0, 0.0, "trip_trip"),
                (3, 0.0, 0.0, "trip_trip"),
                ("D", 0.0, 0.0, "trip_depot"),
            ],
            1: [
                (2, 0.0, 0.0, "trip_trip"),
                (3, 0.0, 0.0, "trip_trip"),
                ("D", 0.0, 0.0, "trip_depot"),
            ],
            2: [("D", 0.0, 0.0, "trip_depot")],
            3: [("D", 0.0, 0.0, "trip_depot")],
        }
        options = dict(
            trips=[0, 1, 2, 3],
            adjacency=adjacency,
            depot="D",
            stations=[],
            trip_start_min={0: 0.0, 1: 0.0, 2: 10.0, 3: 10.0},
            trip_end_min={0: 5.0, 1: 5.0, 2: 15.0, 3: 15.0},
            trip_energy_kwh={0: 60.0, 1: 30.0, 2: 60.0, 3: 30.0},
            battery_capacity_kwh=100.0,
            horizon_min=30.0,
            direct_only=True,
        )

        first_attempt_routes = build_matching_initial_routes(
            **options,
            max_matching_attempts=1,
        )
        self.assertEqual(len(first_attempt_routes), 3)
        self.assertFalse(
            first_attempt_routes[0]["_matching_init"]["is_exact_minimum_path_cover"]
        )

        routes = build_matching_initial_routes(
            **options,
            max_matching_attempts=2,
            matching_order_seed=17,
        )

        self.assertEqual(len(routes), 2)
        self.assertCountEqual(
            [node for route in routes for node in route["route"] if isinstance(node, int)],
            [0, 1, 2, 3],
        )
        provenance = routes[0]["_matching_init"]
        self.assertEqual(provenance["matching_retry_count"], 1)
        self.assertEqual(provenance["matching_order_seed"], 17)
        self.assertEqual(provenance["matching_order"], "columns_reversed")
        self.assertEqual(provenance["resource_repair_mode"], "none")
        self.assertTrue(provenance["is_exact_minimum_path_cover"])
        self.assertEqual(provenance["contiguous_splits_added"], 0)
        self.assertIn("retry=1, seed=17, order=columns_reversed", routes[0]["desc"])

    def test_capped_successor_targets_fall_back_to_feasible_route_cover(self):
        adjacency = {
            "D": [
                (0, 0.0, 0.0, "depot_trip"),
                (1, 0.0, 0.0, "depot_trip"),
                (2, 0.0, 0.0, "depot_trip"),
            ],
            0: [
                ("S", 0.0, 0.0, "trip_station"),
                ("D", 0.0, 0.0, "trip_depot"),
            ],
            1: [("D", 0.0, 0.0, "trip_depot")],
            2: [("D", 0.0, 0.0, "trip_depot")],
            "S": [
                (1, 0.0, 0.0, "station_trip"),
                (2, 100.0, 60.0, "station_trip"),
            ],
        }

        routes = build_matching_initial_routes(
            trips=[0, 1, 2],
            adjacency=adjacency,
            depot="D",
            stations=["S"],
            trip_start_min={0: 0.0, 1: 5.0, 2: 200.0},
            trip_end_min={0: 1.0, 1: 6.0, 2: 201.0},
            trip_energy_kwh={0: 90.0, 1: 30.0, 2: 50.0},
            battery_capacity_kwh=100.0,
            charge_rate_kw=300.0,
            soc_charge_levels=[50.0, 100.0],
            horizon_min=300.0,
            max_daily_recharges=2,
            max_station_to_trip_wait_min=300.0,
            successor_boundary_soc_target=True,
            max_successor_charge_targets=1,
            station_waiting_unrestricted=True,
            max_matching_attempts=8,
        )

        self.assertEqual(len(routes), 3)
        self.assertCountEqual(
            [node for route in routes for node in route["route"] if isinstance(node, int)],
            [0, 1, 2],
        )
        provenance = routes[0]["_matching_init"]
        self.assertEqual(provenance["relaxed_minimum_path_count"], 2)
        self.assertEqual(provenance["resource_feasible_path_count"], 3)
        self.assertEqual(provenance["resource_repair_mode"], "contiguous_split")
        self.assertFalse(provenance["is_exact_minimum_path_cover"])
        self.assertEqual(provenance["contiguous_splits_added"], 1)


class PracticeTenBusMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = build_problem(DEFAULT_DATA_DIR, "Practice_10bus.csv")
        cls.routes = build_matching_initial_routes(
            trips=cls.problem.trips,
            adjacency=cls.problem.adjacency,
            depot=DEPOT,
            stations=STATIONS,
            trip_start_min=cls.problem.start_min,
            trip_end_min=cls.problem.end_min,
            trip_energy_kwh=cls.problem.trip_energy,
            battery_capacity_kwh=300.0,
            charge_rate_kw=300.0,
            soc_charge_levels=[30.0 * index for index in range(1, 11)],
            horizon_min=1560.0,
            max_daily_recharges=15,
            max_station_to_trip_wait_min=220.0,
            charge_start_cost=5.0,
            charging_cost=lambda _station, _start, energy: 0.0992 * energy,
        )
        cls.arc = {
            (left, right): (minutes, energy, arc_type)
            for left, outgoing in cls.problem.adjacency.items()
            for right, minutes, energy, arc_type in outgoing
        }

    def assert_route_resources_independently(self, route):
        nodes = route["route"]
        self.assertEqual(nodes[0], DEPOT)
        self.assertEqual(nodes[-1], DEPOT)

        stops = route["charging_stops"]
        self.assertEqual(
            len(stops["stations"]),
            len(stops["cst"]),
        )
        self.assertEqual(len(stops["stations"]), len(stops["cet"]))
        self.assertEqual(len(stops["stations"]), len(stops["kwh"]))

        soc = 300.0
        time_min = 0.0
        charge_index = 0
        deadhead_total = 0.0
        active = set(self.problem.trips)

        for left, right in zip(nodes, nodes[1:]):
            self.assertIn((left, right), self.arc)
            travel_min, deadhead_kwh, arc_type = self.arc[left, right]
            deadhead_total += deadhead_kwh

            if right in active:
                self.assertIn(
                    arc_type,
                    {"depot_trip", "trip_trip", "station_trip"},
                )
                self.assertLessEqual(
                    time_min + travel_min,
                    self.problem.start_min[right] + 1e-6,
                )
                if arc_type == "station_trip":
                    self.assertLessEqual(
                        self.problem.start_min[right] - time_min,
                        220.0 + 1e-6,
                    )
                soc -= deadhead_kwh + self.problem.trip_energy[right]
                self.assertGreaterEqual(soc, -1e-6)
                time_min = self.problem.end_min[right]
            elif right in STATIONS:
                self.assertEqual(arc_type, "trip_station")
                soc -= deadhead_kwh
                self.assertGreaterEqual(soc, -1e-6)
                time_min += travel_min

                self.assertLess(charge_index, len(stops["stations"]))
                self.assertEqual(stops["stations"][charge_index], right)
                self.assertAlmostEqual(stops["cst"][charge_index], time_min)
                energy = stops["kwh"][charge_index]
                self.assertGreater(energy, 0.0)
                expected_end = time_min + energy / 300.0 * 60.0
                self.assertAlmostEqual(stops["cet"][charge_index], expected_end)
                soc += energy
                self.assertLessEqual(soc, 300.0 + 1e-6)
                self.assertTrue(
                    any(abs(soc - 30.0 * level) <= 1e-6 for level in range(1, 11))
                )
                time_min = expected_end
                charge_index += 1
            else:
                self.assertEqual(right, DEPOT)
                self.assertIn(arc_type, {"trip_depot", "station_depot"})
                soc -= deadhead_kwh
                self.assertGreaterEqual(soc, -1e-6)
                time_min += travel_min
                self.assertLessEqual(time_min, 1560.0 + 1e-6)

        self.assertEqual(charge_index, len(stops["stations"]))
        self.assertAlmostEqual(route["deadhead_kwh"], deadhead_total)
        self.assertLessEqual(route["charging_activities"], 15)

    def test_practice_10bus_has_ten_resource_feasible_matching_routes(self):
        self.assertEqual(len(self.routes), 10)
        self.assertEqual(
            peak_trip_concurrency(
                self.problem.trips,
                self.problem.start_min,
                self.problem.end_min,
            ),
            10,
        )
        all_trips = [
            node
            for route in self.routes
            for node in route["route"]
            if node in set(self.problem.trips)
        ]
        self.assertEqual(len(all_trips), 329)
        self.assertEqual(len(set(all_trips)), 329)
        self.assertEqual(set(all_trips), set(self.problem.trips))

        for route in self.routes:
            self.assert_route_resources_independently(route)
            self.assertTrue(route["_matching_init"]["is_exact_minimum_path_cover"])
            self.assertEqual(route["_matching_init"]["resource_repair_mode"], "none")

    def test_production_initializer_has_no_historical_schedule_dependency(self):
        source = inspect.getsource(matching_init)
        self.assertNotIn("VehicleTask", source)
        self.assertNotIn("Ordered_Trip_ID", source)
        self.assertNotIn("Par_VehicleDetails", source)


class PracticeFifteenBusMatchingTests(unittest.TestCase):
    def test_successor_boundary_realizes_exact_fifteen_route_cover(self):
        problem = build_problem(DEFAULT_DATA_DIR, "Practice_15bus.csv")
        routes = build_matching_initial_routes(
            trips=problem.trips,
            adjacency=problem.adjacency,
            depot=DEPOT,
            stations=STATIONS,
            trip_start_min=problem.start_min,
            trip_end_min=problem.end_min,
            trip_energy_kwh=problem.trip_energy,
            battery_capacity_kwh=300.0,
            charge_rate_kw=300.0,
            soc_charge_levels=[30.0 * index for index in range(1, 11)],
            horizon_min=1560.0,
            max_daily_recharges=15,
            max_station_to_trip_wait_min=220.0,
            successor_boundary_soc_target=True,
            charge_start_cost=5.0,
            charging_cost=lambda _station, _start, energy: 0.0992 * energy,
        )

        self.assertEqual(
            peak_trip_concurrency(
                problem.trips,
                problem.start_min,
                problem.end_min,
            ),
            15,
        )
        self.assertEqual(len(routes), 15)
        active = set(problem.trips)
        covered = [
            node
            for route in routes
            for node in route["route"]
            if node in active
        ]
        self.assertEqual(len(covered), len(problem.trips))
        self.assertEqual(len(set(covered)), len(problem.trips))
        self.assertEqual(set(covered), active)
        self.assertTrue(all(route["charging_activities"] <= 15 for route in routes))
        self.assertTrue(
            all(stop > 0.0 for route in routes for stop in route["charging_stops"]["kwh"])
        )
        self.assertTrue(
            all(route["_matching_init"]["compatibility_mode"] == "full" for route in routes)
        )
        self.assertTrue(
            all(route["_matching_init"]["is_exact_minimum_path_cover"] for route in routes)
        )
        self.assertTrue(
            all(route["_matching_init"]["resource_repair_mode"] == "none" for route in routes)
        )


class MatchingPricingUniverseTests(unittest.TestCase):
    def test_successor_boundary_cap_matches_pricer_subsampling(self):
        options = dict(
            trip_order=[0, 1],
            adjacency={
                "D": [(0, 0.0, 0.0, "depot_trip")],
                0: [("S", 408.0, 0.0, "trip_station")],
                "S": [
                    (1, 0.0, 0.0, "station_trip"),
                    (2, 0.0, 0.0, "station_trip"),
                ],
                1: [("D", 0.0, 0.0, "trip_depot")],
            },
            depot="D",
            stations=["S"],
            trip_start_min={0: 0.0, 1: 413.0, 2: 500.0},
            trip_end_min={0: 1.0, 1: 414.0, 2: 501.0},
            trip_energy_kwh={0: 83.8670001, 1: 236.0},
            battery_capacity_kwh=300.0,
            charge_rate_kw=300.0,
            soc_charge_levels=[30.0 * index for index in range(1, 11)],
            horizon_min=600.0,
            max_daily_recharges=2,
            max_station_to_trip_wait_min=600.0,
            successor_boundary_soc_target=True,
            station_waiting_unrestricted=True,
        )

        # With a cap of one, the common DP rule keeps only the latest/full-SOC
        # boundary, which cannot be reached before trip 1.  A cap of two also
        # retains trip 1's exact partial-charge boundary and realizes the path.
        with self.assertRaises(RouteRealizationError):
            realize_fixed_trip_path(**options, max_successor_charge_targets=1)

        route = realize_fixed_trip_path(**options, max_successor_charge_targets=2)
        self.assertEqual(
            [node for node in route["route"] if isinstance(node, int)],
            [0, 1],
        )
        self.assertAlmostEqual(route["charging_stops"]["kwh"][0], 20.0)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_giro_known_columns import (  # noqa: E402
    DEFAULT_DATA_DIR,
    HORIZON_MIN,
    build_problem,
    charge_target_levels,
    realize_current_runner_trip_order,
    realize_fixed_trip_order,
    reconstruct_historical_duties,
)


class KnownGiroColumnAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = build_problem(DEFAULT_DATA_DIR, "Practice_10bus.csv")
        cls.current_runner_problem = build_problem(
            DEFAULT_DATA_DIR,
            "Practice_10bus.csv",
            max_station_to_trip_wait_min=HORIZON_MIN,
        )
        cls.duties = {
            duty["bus"]: duty
            for duty in reconstruct_historical_duties(
                DEFAULT_DATA_DIR,
                "Practice_10bus.csv",
            )
        }

    def trip_order(self, bus):
        return [
            node
            for node in self.duties[bus]["route"]
            if isinstance(node, int)
        ]

    def test_reconstruction_is_the_expected_ten_bus_partition(self):
        self.assertEqual(
            list(self.duties),
            [str(13300 + index) for index in range(1, 11)],
        )
        all_trips = [
            trip
            for bus in self.duties
            for trip in self.trip_order(bus)
        ]
        self.assertEqual(len(all_trips), 329)
        self.assertEqual(set(all_trips), set(range(329)))

    def test_successor_boundary_target_is_not_the_existing_soc_grid(self):
        current = charge_target_levels(
            arrival_soc=216.1329999,
            arrival_time=409.0,
            latest_departure=413.0,
            include_successor_boundary=False,
        )
        counterfactual = charge_target_levels(
            arrival_soc=216.1329999,
            arrival_time=409.0,
            latest_departure=413.0,
            include_successor_boundary=True,
        )

        self.assertNotIn(236.1329999, current)
        self.assertIn(236.1329999, counterfactual)

    def test_bus_13301_is_current_dp_feasible(self):
        result = realize_fixed_trip_order(
            self.problem,
            self.trip_order("13301"),
        )

        self.assertTrue(result["feasible"])
        self.assertEqual(result["charges_used"], 3)

    def test_bus_13303_fails_soc_grid_and_successor_boundary_repairs_it(self):
        current = realize_fixed_trip_order(
            self.problem,
            self.trip_order("13303"),
        )
        boundary = realize_fixed_trip_order(
            self.problem,
            self.trip_order("13303"),
            include_successor_boundary_target=True,
        )

        self.assertFalse(current["feasible"])
        self.assertEqual(
            (current["failure"]["previous_trip"], current["failure"]["following_trip"]),
            (134, 135),
        )
        self.assertAlmostEqual(
            current["failure"]["direct_energy_shortfall"],
            4.41100022,
            places=6,
        )
        self.assertTrue(boundary["feasible"])
        target_socs = {
            round(charge["target_soc_kwh"], 6)
            for charge in boundary["charges"]
        }
        self.assertIn(round(236.1329999, 6), target_socs)

    def test_bus_13304_needs_waiting_not_a_finer_soc_target(self):
        trip_order = self.trip_order("13304")
        current = realize_fixed_trip_order(self.problem, trip_order)
        boundary = realize_fixed_trip_order(
            self.problem,
            trip_order,
            include_successor_boundary_target=True,
        )
        waiting = realize_fixed_trip_order(
            self.problem,
            trip_order,
            allow_station_waiting=True,
        )

        self.assertFalse(current["feasible"])
        self.assertFalse(boundary["feasible"])
        self.assertTrue(waiting["feasible"])
        self.assertEqual(current["failure"]["raw_gap_min"], 396)
        parx = next(
            window
            for window in current["failure"]["station_windows"]
            if window["station"] == "PARX_1"
        )
        self.assertLess(
            parx["latest_immediate_full_charge_end"],
            parx["earliest_departure_from_220_limit"],
        )

    def test_all_ten_known_duties_fit_the_current_runner_action_set(self):
        results = [
            realize_current_runner_trip_order(
                self.current_runner_problem,
                self.trip_order(bus),
            )
            for bus in self.duties
        ]

        self.assertTrue(all(result["feasible"] for result in results))


if __name__ == "__main__":
    unittest.main()

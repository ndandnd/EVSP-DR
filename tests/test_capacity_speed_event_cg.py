import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from run_capacity_speed_event_cg import (  # noqa: E402
    ExactCapacityMaster,
    duplicate_service_audit,
    physical_capacity_audit,
)


def route(trips, cost, start=None, end=None):
    stops = {"stations": [], "cst": [], "cet": [], "kwh": []}
    if start is not None:
        stops = {
            "stations": ["2190L_0"], "cst": [start], "cet": [end],
            "kwh": [10.0],
        }
    return {
        "trips": trips,
        "cost": cost,
        "expanded_grid_charging_stops": stops,
    }


class CapacitySpeedPilotTests(unittest.TestCase):
    def test_capacity_rows_change_restricted_master_selection(self):
        routes = [
            route([0], 1.0, 10.0, 20.0),
            route([1], 1.0, 10.0, 20.0),
            route([0, 1], 3.0),
        ]
        raw = ExactCapacityMaster((0, 1), capacity=False, threads=1)
        constrained = ExactCapacityMaster((0, 1), capacity=True, threads=1)
        for candidate in routes:
            raw.add_route(candidate)
            constrained.add_route(candidate)
        self.assertAlmostEqual(raw.solve()["objective"], 2.0)
        self.assertAlmostEqual(constrained.solve()["objective"], 2.5)
        self.assertTrue(constrained.solve()["capacity_duals"])

    def test_continuous_half_open_capacity_audit(self):
        touching = [
            route([0], 1.0, 10.0, 20.0),
            route([1], 1.0, 20.0, 30.0),
        ]
        overlapping = [
            route([0], 1.0, 10.0, 20.1),
            route([1], 1.0, 20.0, 30.0),
        ]
        self.assertTrue(physical_capacity_audit(touching, [0, 1])["valid"])
        self.assertFalse(physical_capacity_audit(overlapping, [0, 1])["valid"])

    def test_covering_duplicate_service_is_reported_separately(self):
        routes = [route([0, 1], 1.0), route([1, 2], 1.0)]
        audit = duplicate_service_audit(routes, [0, 1], (0, 1, 2))
        self.assertTrue(audit["all_trips_covered"])
        self.assertEqual(audit["overcovered_trip_count"], 1)
        self.assertEqual(audit["overcovered_trips"], {"1": 2})


if __name__ == "__main__":
    unittest.main()

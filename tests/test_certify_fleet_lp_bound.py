import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from certify_fleet_lp_bound import solve_fleet_master  # noqa: E402


class FleetLPPhaseTwoTests(unittest.TestCase):
    def test_fleet_master_uses_unit_cost_per_real_route(self):
        trips = [0, 1]
        routes = [
            {"trips": [0], "cost": 900000.0},
            {"trips": [1], "cost": 1.0},
            {"trips": [0, 1], "cost": 999999999.0},
        ]
        solved = solve_fleet_master(trips, routes)
        self.assertAlmostEqual(solved.objective, 1.0)
        self.assertAlmostEqual(sum(solved.route_values), 1.0)
        self.assertLessEqual(solved.max_row_violation, 1e-7)
        self.assertLessEqual(solved.max_bound_violation, 1e-7)

    def test_fractional_fleet_master_bound(self):
        trips = [0, 1, 2]
        routes = [
            {"trips": [0, 1], "cost": 0.0},
            {"trips": [1, 2], "cost": 0.0},
            {"trips": [0, 2], "cost": 0.0},
        ]
        solved = solve_fleet_master(trips, routes)
        self.assertAlmostEqual(solved.objective, 1.5)
        self.assertAlmostEqual(sum(solved.trip_duals.values()), 1.5)


if __name__ == "__main__":
    unittest.main()

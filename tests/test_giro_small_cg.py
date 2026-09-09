import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_giro_known_columns import ProblemData  # noqa: E402
from giro_partille_physics import PARTILLE_PROFILES  # noqa: E402
from giro_weighted_pricing import weighted_price_route  # noqa: E402
from run_giro_small_cg import _master  # noqa: E402


def toy_problem():
    return ProblemData(
        frame=None,
        trips=(0, 1),
        adjacency={
            "PARX_0": [
                (0, 0.0, 0.0, "depot_trip"),
                (1, 0.0, 0.0, "depot_trip"),
            ],
            0: [
                (1, 0.0, 0.0, "trip_trip"),
                ("PARX_0", 0.0, 0.0, "trip_depot"),
            ],
            1: [("PARX_0", 0.0, 0.0, "trip_depot")],
        },
        start_min={0: 0.0, 1: 60.0},
        end_min={0: 30.0, 1: 90.0},
        trip_energy={0: 10.0, 1: 10.0},
    )


def route(trips, start=None, end=None):
    actions = [{"kind": "source"}]
    if start is not None:
        actions.append({
            "kind": "charge", "station": "2190L",
            "setup_start_min": start, "connection_end_min": end,
        })
    return {"trips": trips, "actions": actions, "cost": 1.0}


class WeightedPricingTests(unittest.TestCase):
    def test_trip_duals_drive_full_two_trip_route(self):
        priced = weighted_price_route(
            toy_problem(), PARTILLE_PROFILES["18E2"], {0: 1.0, 1: 1.0},
            horizon_min=120.0, wall_limit_s=2.0, label_limit=100,
        )
        self.assertIsNone(priced["guard"])
        self.assertEqual(priced["route"]["trips"], [0, 1])
        self.assertAlmostEqual(priced["reduced_cost_without_capacity_duals"], -1.0)


class SharedPoolMasterTests(unittest.TestCase):
    def test_capacity_row_changes_selected_route_and_cover_le_partition(self):
        routes = [
            route([0], 0.0, 1.0),
            route([1], 0.0, 1.0),
            route([1], 2.0, 3.0),
        ]
        constrained = _master(
            routes, [0, 1], sense="partition", binary=True,
            capacity=True, threads=1,
        )
        self.assertEqual(constrained["status"], "OPTIMAL")
        self.assertEqual(set(constrained["selected_indices"]), {0, 2})
        cover = _master(
            routes, [0, 1], sense="cover", binary=False,
            capacity=True, threads=1,
        )
        partition = _master(
            routes, [0, 1], sense="partition", binary=False,
            capacity=True, threads=1,
        )
        self.assertLessEqual(cover["restricted_pool_fleet_lp"],
                             partition["restricted_pool_fleet_lp"] + 1e-9)
        self.assertEqual(cover["capacity_time_grid_min"], 1)


if __name__ == "__main__":
    unittest.main()

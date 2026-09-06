import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from master_lp_gurobi import (  # noqa: E402
    GurobiRestrictedMaster,
    RestrictedMasterInputError,
    gurobi_preflight,
)


class GurobiRestrictedMasterTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("GRB_LICENSE_FILE"):
            self.skipTest("GRB_LICENSE_FILE is not configured")
        try:
            gurobi_preflight()
        except Exception as exc:  # pragma: no cover - host-dependent
            self.skipTest(f"Gurobi preflight unavailable: {exc}")

    def test_persistent_partition_master_improves_incrementally(self):
        master = GurobiRestrictedMaster(
            trip_ids=[1, 2],
            artificial_penalty=100.0,
            coverage_sense="partition",
        )
        try:
            self.assertEqual(master.add_routes([[1], [2]], [2.0, 2.0]), 2)
            model_id = id(master.model)
            first = master.solve()
            self.assertAlmostEqual(first.objective, 4.0, places=7)
            self.assertAlmostEqual(first.route_weight, 2.0, places=7)
            self.assertAlmostEqual(first.artificial_total, 0.0, places=7)

            self.assertEqual(
                master.add_routes(
                    [[1], [2], [1, 2]], [2.0, 2.0, 3.0]
                ),
                1,
            )
            second = master.solve()
            self.assertEqual(id(master.model), model_id)
            self.assertAlmostEqual(second.objective, 3.0, places=7)
            self.assertAlmostEqual(second.route_weight, 1.0, places=7)
            self.assertAlmostEqual(second.artificial_total, 0.0, places=7)
            self.assertIn("Gurobi", second.backend.solver)
            self.assertEqual(second.backend.threads, 1)
        finally:
            master.close()

    def test_artificial_columns_can_be_replaced_incrementally(self):
        master = GurobiRestrictedMaster(
            trip_ids=[1, 2],
            artificial_penalty=100.0,
            coverage_sense="partition",
        )
        try:
            self.assertEqual(master.add_routes([], []), 0)
            artificial = master.solve()
            self.assertAlmostEqual(artificial.objective, 200.0, places=7)
            self.assertAlmostEqual(artificial.route_weight, 0.0, places=7)
            self.assertAlmostEqual(artificial.artificial_total, 2.0, places=7)

            master.add_routes([[1, 2]], [3.0])
            replaced = master.solve()
            self.assertAlmostEqual(replaced.objective, 3.0, places=7)
            self.assertAlmostEqual(replaced.route_weight, 1.0, places=7)
            self.assertAlmostEqual(replaced.artificial_total, 0.0, places=7)
        finally:
            master.close()

    def test_sync_routes_replaces_route_with_cheaper_charging_realization(self):
        master = GurobiRestrictedMaster(
            trip_ids=[1, 2],
            artificial_penalty=100.0,
            coverage_sense="partition",
        )
        try:
            master.sync_routes([{"trips": [1, 2], "cost": 12.0}])
            initial = master.solve()
            self.assertAlmostEqual(initial.objective, 12.0, places=7)

            # The ordered trip incidence is unchanged; only the charging
            # realization made the column cheaper.
            self.assertEqual(
                master.sync_routes([{"trips": [1, 2], "cost": 7.0}]), 0
            )
            cheaper = master.solve()
            self.assertAlmostEqual(cheaper.objective, 7.0, places=7)
            self.assertAlmostEqual(cheaper.route_weight, 1.0, places=7)
            self.assertAlmostEqual(master._routes[0][1], 7.0, places=7)
            self.assertAlmostEqual(master._routes[0][2].Obj, 7.0, places=7)

            with self.assertRaises(RestrictedMasterInputError):
                master.sync_routes([{"trips": [2, 1], "cost": 6.0}])
            with self.assertRaises(RestrictedMasterInputError):
                master.sync_routes([{"trips": [1, 2], "cost": 8.0}])
            with self.assertRaises(RestrictedMasterInputError):
                master.sync_routes([])
        finally:
            master.close()

    def test_route_prefix_identity_and_cost_are_immutable(self):
        master = GurobiRestrictedMaster(
            trip_ids=[1, 2],
            artificial_penalty=100.0,
            coverage_sense="partition",
        )
        try:
            master.add_routes([[1], [2]], [2.0, 2.0])
            with self.assertRaises(RestrictedMasterInputError):
                master.add_routes([[2], [2]], [2.0, 2.0])
            with self.assertRaises(RestrictedMasterInputError):
                master.add_routes([[1], [2]], [2.0, 3.0])
            with self.assertRaises(RestrictedMasterInputError):
                master.add_routes([[1]], [2.0])
        finally:
            master.close()

    def test_cover_duals_use_nonnegative_public_convention(self):
        master = GurobiRestrictedMaster(
            trip_ids=[1],
            artificial_penalty=100.0,
            coverage_sense="cover",
        )
        try:
            master.add_routes([[1]], [3.0])
            result = master.solve()
            self.assertAlmostEqual(result.trip_duals[1], 3.0, places=7)
        finally:
            master.close()


if __name__ == "__main__":
    unittest.main()

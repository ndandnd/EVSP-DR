import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exact_pricer_expanded import prepare_master_incidence
from master_lp_gurobi import GurobiRestrictedMaster, RestrictedMasterInputError, gurobi_preflight


class MasterIncidenceInputTests(unittest.TestCase):
    def test_scipy_keeps_matrix_even_with_optimization_flag(self):
        routes = [{"trips": [4], "cost": 3}, {"trips": [4, 8], "cost": 5}]
        matrix, nnz, shape = prepare_master_incidence([4, 8], routes, backend="scipy", skip_gurobi=True)
        self.assertEqual(matrix.toarray().tolist(), [[1, 1], [0, 1]])
        self.assertEqual((nnz, shape), (3, (2, 2)))

    def test_gurobi_opt_in_avoids_allocation_and_preserves_telemetry(self):
        routes = [{"trips": [4], "cost": 3}, {"trips": [4, 8], "cost": 5}]
        original, nnz, shape = prepare_master_incidence([4, 8], routes, backend="gurobi")
        with patch("exact_pricer_expanded.build_route_incidence", side_effect=AssertionError("unused allocation")):
            optimized = prepare_master_incidence([4, 8], routes, backend="gurobi", skip_gurobi=True)
        self.assertEqual(optimized, (None, nnz, shape))
        self.assertEqual(original.nnz, 3)

    def test_skip_retains_persistent_master_validation_and_cheaper_rewrite(self):
        if not os.environ.get("GRB_LICENSE_FILE"):
            self.skipTest("Gurobi license not configured")
        gurobi_preflight()
        for sense in ("cover", "partition"):
            solutions = []
            for skip in (False, True):
                master = GurobiRestrictedMaster(trip_ids=[0, 1], artificial_penalty=100, coverage_sense=sense)
                try:
                    for cost in (7.0, 5.0):
                        routes = [{"trips": [0], "cost": 4}, {"trips": [1], "cost": 4}, {"trips": [0, 1], "cost": cost}]
                        prepare_master_incidence([0, 1], routes, backend="gurobi", skip_gurobi=skip)
                        master.sync_routes(routes)
                        result = master.solve()
                        solutions.append((result.objective, result.route_values, result.trip_duals, result.artificial_total))
                    with self.assertRaises(RestrictedMasterInputError):
                        master.sync_routes([{**routes[0], "trips": [9]}, *routes[1:]])
                finally:
                    master.close()
            self.assertEqual(solutions[:2], solutions[2:])
            self.assertEqual([s[0] for s in solutions[:2]], [7, 5])


if __name__ == "__main__":
    unittest.main()

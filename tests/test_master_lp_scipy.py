import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from master_lp_scipy import (  # noqa: E402
    RestrictedMasterInputError,
    RestrictedMasterSolveError,
    build_route_incidence,
    solve_restricted_master_lp,
)


class RestrictedMasterLPTests(unittest.TestCase):
    def solve(self, trips, routes, costs, penalty=500.0, coverage_sense="cover"):
        incidence = build_route_incidence(trips, routes)
        return solve_restricted_master_lp(
            trip_ids=trips,
            route_incidence=incidence,
            route_costs=costs,
            artificial_penalty=penalty,
            coverage_sense=coverage_sense,
        )

    def test_artificial_only_has_positive_coverage_dual(self):
        result = solve_restricted_master_lp(
            trip_ids=[7],
            route_incidence=np.empty((1, 0)),
            route_costs=[],
            artificial_penalty=500.0,
        )

        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.route_values, ())
        self.assertAlmostEqual(result.objective, 500.0)
        self.assertAlmostEqual(result.artificial_values[7], 1.0)
        self.assertAlmostEqual(result.trip_duals[7], 500.0)
        self.assertIn("HiGHS", result.backend.solver)
        self.assertGreaterEqual(result.runtime_s, 0.0)

    def test_real_route_replaces_artificial(self):
        result = self.solve([0], [[0]], [100.0])

        self.assertAlmostEqual(result.objective, 100.0)
        self.assertAlmostEqual(result.route_values[0], 1.0)
        self.assertAlmostEqual(result.artificial_values[0], 0.0)
        self.assertAlmostEqual(result.trip_duals[0], 100.0)

    def test_fractional_triangle_set_cover(self):
        trips = [0, 1, 2]
        routes = [[0, 1], [1, 2], [0, 2]]
        result = self.solve(trips, routes, [1.0, 1.0, 1.0], penalty=100.0)

        self.assertAlmostEqual(result.objective, 1.5)
        self.assertAlmostEqual(result.route_weight, 1.5)
        self.assertEqual(len(result.route_values), 3)
        for value in result.route_values:
            self.assertAlmostEqual(value, 0.5)
        self.assertAlmostEqual(result.artificial_total, 0.0)

    def test_missing_trip_stays_artificial(self):
        result = self.solve([0, 1], [[0]], [2.0], penalty=50.0)

        self.assertAlmostEqual(result.objective, 52.0)
        self.assertAlmostEqual(result.route_values[0], 1.0)
        self.assertAlmostEqual(result.artificial_values[0], 0.0)
        self.assertAlmostEqual(result.artificial_values[1], 1.0)
        self.assertAlmostEqual(result.trip_duals[0], 2.0)
        self.assertAlmostEqual(result.trip_duals[1], 50.0)

    def test_partition_mode_does_not_accept_cheap_overcoverage(self):
        trips = [0, 1, 2]
        routes = [[0, 1], [1, 2]]
        costs = [1.0, 1.0]

        covering = self.solve(trips, routes, costs, penalty=100.0)
        partitioning = self.solve(
            trips,
            routes,
            costs,
            penalty=100.0,
            coverage_sense="partition",
        )

        self.assertAlmostEqual(covering.objective, 2.0)
        self.assertAlmostEqual(partitioning.objective, 101.0)
        self.assertAlmostEqual(partitioning.artificial_total, 1.0)

        incidence = build_route_incidence(trips, routes)
        alpha = np.array([partitioning.trip_duals[trip] for trip in trips])
        reduced_costs = np.array(costs) - incidence.T @ alpha
        self.assertGreaterEqual(float(reduced_costs.min()), -1e-8)
        self.assertGreaterEqual(float((100.0 - alpha).min()), -1e-8)

    def test_duals_satisfy_route_and_artificial_reduced_costs(self):
        trips = [0, 1, 2]
        routes = [[0, 1], [1, 2], [0, 2], [0]]
        costs = np.array([1.0, 1.0, 1.0, 0.75])
        penalty = 100.0
        incidence = build_route_incidence(trips, routes)
        result = solve_restricted_master_lp(
            trip_ids=trips,
            route_incidence=incidence,
            route_costs=costs,
            artificial_penalty=penalty,
        )

        alpha = np.array([result.trip_duals[trip] for trip in trips])
        route_reduced_costs = costs - incidence.T @ alpha
        artificial_reduced_costs = penalty - alpha
        self.assertGreaterEqual(float(route_reduced_costs.min()), -1e-8)
        self.assertGreaterEqual(float(artificial_reduced_costs.min()), -1e-8)

        for value, reduced_cost in zip(result.route_values, route_reduced_costs):
            if value > 1e-8:
                self.assertAlmostEqual(float(reduced_cost), 0.0, places=8)

    @staticmethod
    def fake_partition_result(x, marginal=5e-7, objective=1.0):
        return SimpleNamespace(
            success=True,
            status=0,
            x=np.asarray(x, dtype=float),
            fun=float(objective),
            message="synthetic optimal result",
            eqlin=SimpleNamespace(marginals=np.asarray([marginal], dtype=float)),
            ineqlin=None,
        )

    def test_raw_tiny_primal_values_are_not_deleted_before_feasibility_check(self):
        # Raw coverage is exactly one.  The old implementation deleted both
        # 0.75e-6 route values before checking coverage and then rejected its
        # own modified solution with a 1.5e-6 row violation.
        fake = self.fake_partition_result(
            [0.9999985, 0.75e-6, 0.75e-6, 0.0],
            marginal=5e-7,
            objective=1.0,
        )
        incidence = np.ones((1, 3), dtype=float)
        with patch("master_lp_scipy.linprog", return_value=fake):
            result = solve_restricted_master_lp(
                trip_ids=[11],
                route_incidence=incidence,
                route_costs=[1.0, 1.0, 1.0],
                artificial_penalty=500.0,
                coverage_sense="partition",
                feasibility_tolerance=1e-6,
            )

        self.assertEqual(result.route_values, (0.9999985, 0.75e-6, 0.75e-6))
        self.assertAlmostEqual(result.route_weight, 1.0)
        self.assertAlmostEqual(result.trip_duals[11], 5e-7)
        self.assertLessEqual(result.max_row_violation, 1e-12)
        self.assertEqual(result.max_bound_violation, 0.0)
        self.assertEqual(fake.x[1], 0.75e-6, "solver output must not be mutated")

    def test_genuine_raw_partition_violation_is_still_rejected(self):
        fake = self.fake_partition_result([0.999998, 0.0], marginal=1.0)
        with patch("master_lp_scipy.linprog", return_value=fake):
            with self.assertRaisesRegex(
                RestrictedMasterSolveError,
                r"maximum row violation=.*tolerance=1e-06",
            ):
                solve_restricted_master_lp(
                    trip_ids=[11],
                    route_incidence=np.ones((1, 1), dtype=float),
                    route_costs=[1.0],
                    artificial_penalty=500.0,
                    coverage_sense="partition",
                    feasibility_tolerance=1e-6,
                )

    def test_genuine_raw_lower_bound_violation_is_rejected(self):
        # Coverage is exact, but the artificial variable violates its
        # nonnegative lower bound by more than the audit tolerance.
        fake = self.fake_partition_result(
            [1.000002, -0.000002], marginal=1.0
        )
        with patch("master_lp_scipy.linprog", return_value=fake):
            with self.assertRaisesRegex(
                RestrictedMasterSolveError,
                r"maximum bound violation=.*tolerance=1e-06",
            ):
                solve_restricted_master_lp(
                    trip_ids=[11],
                    route_incidence=np.ones((1, 1), dtype=float),
                    route_costs=[1.0],
                    artificial_penalty=500.0,
                    coverage_sense="partition",
                    feasibility_tolerance=1e-6,
                )

    def test_input_errors_name_the_inconsistent_field(self):
        with self.assertRaisesRegex(RestrictedMasterInputError, "outside trip_ids"):
            build_route_incidence([0], [[1]])

        with self.assertRaisesRegex(RestrictedMasterInputError, "shape"):
            solve_restricted_master_lp(
                trip_ids=[0, 1],
                route_incidence=np.ones((2, 2)),
                route_costs=[1.0],
                artificial_penalty=100.0,
            )

        with self.assertRaisesRegex(RestrictedMasterInputError, "coverage_sense"):
            self.solve([0], [[0]], [1.0], coverage_sense="invalid")


if __name__ == "__main__":
    unittest.main()

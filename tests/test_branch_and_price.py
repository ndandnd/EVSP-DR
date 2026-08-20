import math
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from branch_and_price import (  # noqa: E402
    BranchConstraint,
    ConstrainedDAGPricer,
    ValidationGateError,
    assert_child_bound,
    assert_integral_solution,
    audit_exact_partition,
    choose_ryan_foster_pair,
    expand_constraint_assignments,
    route_satisfies,
)


@dataclass
class TinyProblem:
    trips: tuple[int, ...]
    start_min: dict[int, float]


class TinyExpandedNetwork:
    """A one-SOC-state time-expanded DAG with seven feasible routes."""

    SINK = 1

    def __init__(self):
        # source, sink, then time-ordered trip nodes A/B/C.
        self.problem = TinyProblem(
            trips=(10, 20, 30),
            start_min={10: 10.0, 20: 20.0, 30: 30.0},
        )
        self.node_meta = [
            ("source", None, None),
            ("sink", None, None),
            ("trip", 10, 0),
            ("trip", 20, 0),
            ("trip", 30, 0),
        ]
        self.topo = [0, 2, 3, 4, 1]
        # Arc fields are successor, base cost, dense trip-dual index.
        self.out = [
            [(2, 6.0, 0), (3, 5.0, 1), (4, 7.0, 2)],
            [],
            [(1, 0.0, -1), (3, 1.0, 1), (4, 3.0, 2)],
            [(1, 0.0, -1), (4, 1.0, 2)],
            [(1, 0.0, -1)],
        ]
        self.sink_arcs = (
            (2, 0.0),
            (3, 0.0),
            (4, 0.0),
        )


def brute_force_routes(network, duals):
    """Enumerate every source-to-sink path in the deliberately tiny DAG."""

    dense = [duals.get(trip, 0.0) for trip in network.problem.trips]
    routes = []

    def visit(node, cost, trips):
        for successor, base_cost, dual_index in network.out[node]:
            next_cost = cost + base_cost
            if dual_index >= 0:
                next_cost -= dense[dual_index]
            if successor == network.SINK:
                routes.append((next_cost, tuple(trips)))
                continue
            kind, key, _level = network.node_meta[successor]
            visit(
                successor,
                next_cost,
                trips + ([key] if kind == "trip" else []),
            )

    visit(0, 0.0, [])
    return routes


class BranchAndPriceGateTests(unittest.TestCase):
    def setUp(self):
        self.network = TinyExpandedNetwork()
        self.pricer = ConstrainedDAGPricer(self.network)
        self.duals = {10: 4.0, 20: 4.0, 30: 4.0}

    def assert_pricing_matches_bruteforce(self, constraints):
        priced, solves = self.pricer.price(
            self.duals,
            constraints,
            max_candidates=20,
        )
        feasible = [
            (cost, trips)
            for cost, trips in brute_force_routes(self.network, self.duals)
            if route_satisfies(trips, constraints)
        ]
        self.assertTrue(feasible)
        self.assertTrue(priced)
        self.assertAlmostEqual(
            priced[0]["rc"],
            min(cost for cost, _trips in feasible),
            places=12,
        )
        self.assertLessEqual(solves, 2 ** len(constraints))

    def test_g3_apart_pricing_matches_complete_route_enumeration(self):
        self.assert_pricing_matches_bruteforce(
            [BranchConstraint("apart", 10, 20)]
        )

    def test_g3_together_pricing_matches_complete_route_enumeration(self):
        self.assert_pricing_matches_bruteforce(
            [BranchConstraint("together", 10, 20)]
        )

    def test_g3_multiple_constraints_match_complete_route_enumeration(self):
        self.assert_pricing_matches_bruteforce([
            BranchConstraint("together", 10, 20),
            BranchConstraint("apart", 20, 30),
        ])

    def test_disjunctive_expansion_removes_conflicts_and_duplicates(self):
        assignments = expand_constraint_assignments([
            BranchConstraint("together", 10, 20),
            BranchConstraint("apart", 10, 30),
        ])
        self.assertGreater(len(assignments), 0)
        self.assertLessEqual(len(assignments), 4)
        for required, forbidden in assignments:
            self.assertTrue(required.isdisjoint(forbidden))

    def test_g2_child_bound_monotonicity_is_a_runtime_assertion(self):
        assert_child_bound(100.0, 100.0)
        assert_child_bound(100.0, 100.00001)
        with self.assertRaisesRegex(ValidationGateError, "G2"):
            assert_child_bound(100.0, 99.0)

    def test_g4_fractional_pair_is_selected_nearest_half(self):
        routes = [
            {"trips": [10, 20]},
            {"trips": [20, 30]},
            {"trips": [10, 30]},
        ]
        pair, alpha = choose_ryan_foster_pair(routes, [0.5, 0.5, 0.5])
        self.assertEqual(pair, (10, 20))
        self.assertAlmostEqual(alpha, 0.5)

    def test_g4_pair_integral_solution_is_genuinely_integral(self):
        trips = [10, 20, 30]
        routes = [{"trips": [10, 20]}, {"trips": [30]}]
        self.assertIsNone(choose_ryan_foster_pair(routes, [1.0, 1.0]))
        selected = assert_integral_solution(trips, routes, [1.0, 1.0])
        self.assertEqual(selected, routes)

    def test_g4_rejects_pair_integral_but_fractional_route_values(self):
        with self.assertRaisesRegex(ValidationGateError, "G4"):
            assert_integral_solution(
                [10],
                [{"trips": [10]}],
                [0.5],
            )

    def test_g5_partition_audit_rejects_overcoverage(self):
        with self.assertRaisesRegex(ValidationGateError, "G5"):
            audit_exact_partition(
                [10, 20, 30],
                [{"trips": [10, 20]}, {"trips": [20, 30]}],
            )

    def test_g5_partition_audit_accepts_exact_cover(self):
        audit_exact_partition(
            [10, 20, 30],
            [{"trips": [10, 20]}, {"trips": [30]}],
        )


if __name__ == "__main__":
    unittest.main()

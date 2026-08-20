import json
import math
import random
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from branch_and_price import (  # noqa: E402
    BranchConstraint,
    BranchAndPriceSolver,
    _baseline_identity,
    ConstrainedDAGPricer,
    ValidationGateError,
    assert_child_bound,
    assert_integral_solution,
    audit_exact_partition,
    choose_ryan_foster_pair,
    conservative_dual_lower_bound,
    expand_constraint_assignments,
    route_satisfies,
    solve_phase_master,
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


def brute_force_routes(network, duals, objective="combined-cost"):
    """Enumerate every source-to-sink path in the deliberately tiny DAG."""

    dense = [duals.get(trip, 0.0) for trip in network.problem.trips]
    routes = []

    def visit(node, cost, trips):
        for successor, base_cost, dual_index in network.out[node]:
            objective_cost = (
                0.0 if objective == "artificial-elimination"
                else 1.0 if objective == "fleet-only" and node == 0
                else 0.0 if objective == "fleet-only"
                else base_cost
            )
            next_cost = cost + objective_cost
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


class RandomMultiSOCNetwork:
    """Random tiny expanded DAG: three trips, two SOC states, charge states."""

    SINK = 1

    def __init__(self, seed):
        rng = random.Random(seed)
        self.problem = TinyProblem(
            trips=(10, 20, 30),
            start_min={10: 10.0, 20: 20.0, 30: 30.0},
        )
        self.node_meta = [("source", None, None), ("sink", None, None)]
        layer_specs = [
            (("trip", 10), 2),
            (("charge", ("S", 0)), 2),
            (("trip", 20), 2),
            (("charge", ("S", 1)), 2),
            (("trip", 30), 2),
        ]
        layers = []
        for (kind, key), width in layer_specs:
            layer = []
            for level in range(width):
                layer.append(len(self.node_meta))
                self.node_meta.append((kind, key, level))
            layers.append(layer)
        self.topo = [0] + [node for layer in layers for node in layer] + [1]
        self.out = [[] for _ in self.node_meta]
        trip_position = {trip: index for index, trip in enumerate(self.problem.trips)}

        def add(left, right):
            kind, key, _level = self.node_meta[right]
            dual = trip_position[key] if kind == "trip" else -1
            self.out[left].append((right, rng.uniform(0.1, 5.0), dual))

        trip_layers = (layers[0], layers[2], layers[4])
        for layer in trip_layers:
            for node in layer:
                add(0, node)
                self.out[node].append((1, rng.uniform(0.0, 2.0), -1))
        for left_index, left_layer in enumerate(layers):
            for right_layer in layers[left_index + 1:]:
                for left in left_layer:
                    for right in right_layer:
                        if rng.random() < 0.4:
                            add(left, right)
        self.sink_arcs = tuple(
            (left, cost) for left, arcs in enumerate(self.out)
            for successor, cost, _dual in arcs if successor == self.SINK
        )


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

    def test_g3_randomized_multi_soc_oracle_all_objectives(self):
        constraint_sets = [
            [BranchConstraint(kind, left, right)]
            for kind in ("apart", "together")
            for left, right in ((10, 20), (10, 30), (20, 30))
        ] + [[
            BranchConstraint("together", 10, 20),
            BranchConstraint("apart", 20, 30),
        ]]
        for seed in range(20):
            network = RandomMultiSOCNetwork(seed)
            pricer = ConstrainedDAGPricer(network)
            rng = random.Random(1000 + seed)
            duals = {trip: rng.uniform(-2.0, 4.0)
                     for trip in network.problem.trips}
            for objective in (
                "combined-cost", "artificial-elimination", "fleet-only",
            ):
                enumerated = brute_force_routes(network, duals, objective)
                for constraints in constraint_sets:
                    feasible = [
                        cost for cost, trips in enumerated
                        if route_satisfies(trips, constraints)
                    ]
                    priced, _solves = pricer.price(
                        duals, constraints, max_candidates=100,
                        objective=objective,
                    )
                    with self.subTest(
                        seed=seed, objective=objective,
                        constraints=constraints,
                    ):
                        self.assertTrue(feasible)
                        self.assertTrue(priced)
                        self.assertAlmostEqual(
                            priced[0]["rc"], min(feasible), places=10,
                        )

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

    def test_phase_one_ignores_finite_real_route_cost(self):
        expensive = [{"trips": [10], "cost": 10**12}]
        phase_one = solve_phase_master([10], expensive, 1)
        phase_two = solve_phase_master([10], expensive, 2)
        self.assertEqual(phase_one.artificial_total, 0.0)
        self.assertEqual(phase_one.objective, 0.0)
        self.assertEqual(phase_two.artificial_values, ())
        self.assertEqual(phase_two.objective, 1.0)

    def test_g1_baseline_binds_all_scientific_identity_fields(self):
        args = SimpleNamespace(
            csv="scale_ladder/instances/Practice_Custom_DutyUnion_k02_r2.csv",
            prices_csv="hourly_prices_flat.csv", soc_step=15.0, block_min=10,
            g_kwh=300.0, charge_kw=300.0, min_soc_frac=0.0,
            expected_root_weight=2.1875,
        )
        provenance = {
            "instance_sha256":
                "6ca7e2db690120699d59fc81428b0f1af00c5cc3889770e8f2860af040244932",
            "prices_sha256":
                "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200",
            "reference_sha256":
                "7bda0e1f439dc8bf5081499566eb2c6a0314190ef27294707f1403fd2c13e3a0",
            "deadhead_sha256":
                "5993e922c671f053611635578b32a1be13bab87b3b5fd8c02b699b81fe0eb66c",
        }
        bound = _baseline_identity(args, provenance)
        self.assertEqual(bound["cell"], "k02_s2")
        self.assertEqual(bound["route_weight"], 2.1875)
        with self.assertRaisesRegex(ValidationGateError, "prices_sha256"):
            _baseline_identity(args, {**provenance, "prices_sha256": "wrong"})

    def test_positive_phase_one_dual_bound_is_strict_certificate(self):
        phase_one = solve_phase_master([10], [], 1)
        lower_bound = conservative_dual_lower_bound(
            phase_one, trip_count=1, pricing_tolerance=1e-9,
        )
        self.assertGreater(lower_bound, 0.0)
        self.assertAlmostEqual(phase_one.artificial_total, 1.0)

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

    def test_durable_root_checkpoint_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "root.json"
            values = dict(
                csv="Practice_Selected_1buses.csv",
                prices_csv="hourly_prices_flat.csv", target_fleet=1,
                soc_step=15.0, block_min=10, g_kwh=300.0,
                charge_kw=300.0, min_soc_frac=0.0, columns_per_iter=30,
                rc_eps=1e-9, integrality_tol=1e-7,
                phase_1_positive_tol=1e-8, bound_tolerance=1e-5,
                max_cg_iters=400, max_depth=8, node_limit=100,
                wall_limit_s=300.0, root_mip_s=0.0,
                pricing_slowdown_limit=10.0, pricing_slow_nodes=3,
                expected_root_weight=None, root_only=True,
                root_pool_result=None, resume=False, out=output,
            )
            fresh = BranchAndPriceSolver(SimpleNamespace(**values))
            first = fresh.run()
            fresh.close()
            self.assertTrue(first["root_certified"])
            self.assertTrue(output.is_file())
            ledger_path = Path(str(output) + ".nodes.jsonl")
            events = [json.loads(line)["event"]
                      for line in ledger_path.read_text().splitlines()]
            self.assertIn("pricing_iteration", events)
            self.assertIn("root_certified", events)
            self.assertEqual(first["search_checkpoint"]["stack"], [])
            values["resume"] = True
            resumed = BranchAndPriceSolver(SimpleNamespace(**values))
            second = resumed.run()
            resumed.close()
            self.assertTrue(second["root_certified"])
            self.assertEqual(first["run_identity"], second["run_identity"])


if __name__ == "__main__":
    unittest.main()

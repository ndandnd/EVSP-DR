import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arcflow_oracle import (  # noqa: E402
    NetworkData,
    audit_route,
    build_model,
    build_network,
    decompose,
    gate_g1,
    index_active_arcs,
    map_route_to_arcs,
    solve,
)


class FakeNetwork:
    def __init__(self):
        self.problem = SimpleNamespace(trips=(0,))
        self.node_meta = [
            ("source", None, None),
            ("sink", None, None),
            ("trip", 0, 0),
            ("charge", ("dead", 0), 0),
        ]
        self.SINK = 1
        self.topo = [0, 2, 3, 1]
        self.out = [
            [(2, 100000.0, 0)],
            [],
            [(1, 0.0, -1)],
            [(1, 0.0, -1)],  # Cannot be reached from the source.
        ]
        self.n_arcs = 3


def fake_data():
    network = FakeNetwork()
    return NetworkData(
        "fake.csv", "prices.csv", network.problem, network, 15.0, 10
    )


class ArcFlowOracleTests(unittest.TestCase):
    def test_exact_reachability_presolve_and_tiny_solve(self):
        data = fake_data()
        arcs = index_active_arcs(data.network)
        self.assertEqual(arcs.full_arcs, 3)
        self.assertEqual(arcs.size, 2)
        self.assertTrue(gate_g1(data, arcs)["passed"])
        model = build_model(data, arcs)

        lp, primal = solve(
            model, objective_kind="fleet", integrality="none"
        )
        self.assertEqual(lp.status, "optimal")
        self.assertAlmostEqual(lp.vehicles, 1.0)
        np.testing.assert_allclose(primal, [1.0, 1.0])

        mip, primal = solve(
            model, objective_kind="combined", integrality="all"
        )
        self.assertEqual(mip.status, "optimal")
        self.assertTrue(mip.all_arcs_integral)
        self.assertAlmostEqual(mip.objective, 100000.0)
        self.assertEqual(decompose(model, primal), [[0, 1]])

    def test_service_integrality_relaxation_is_labeled(self):
        data = fake_data()
        model = build_model(data, index_active_arcs(data.network))
        result, _primal = solve(
            model, objective_kind="feasibility", integrality="service",
            fixed_fleet=1,
        )
        self.assertEqual(result.integrality, "service")
        self.assertTrue(result.all_arcs_integral)

    def test_real_pricer_route_maps_to_identical_arcs_and_cost(self):
        data = build_network(
            "scale_ladder/instances/Practice_Custom_DutyUnion_k02_r2.csv",
            soc_step=15.0,
            block_min=10,
        )
        arcs = index_active_arcs(data.network)
        gate_g1(data, arcs)
        duals = {trip: 500000.0 for trip in data.problem.trips}
        priced = data.network.min_reduced_cost_route(duals)
        cost = priced["rc"] + sum(duals[t] for t in priced["trips"])
        record = {
            "trips": priced["trips"],
            "route_nodes": priced["route_nodes"],
            "charging_stops": priced["charging_stops"],
            "expanded_grid_charging_stops":
                priced["_expanded_grid_charging"],
            "cost": cost,
            "expanded_grid_cost": cost,
        }
        path = map_route_to_arcs(data, arcs, record)
        detail = audit_route(data, arcs, record, path)
        self.assertEqual(detail["trips"], priced["trips"])
        self.assertTrue(math.isclose(
            detail["mapped_cost"], cost, abs_tol=1e-6
        ))


if __name__ == "__main__":
    unittest.main()

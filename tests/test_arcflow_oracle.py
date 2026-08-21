import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arcflow_oracle import (  # noqa: E402
    _validate_journal_identity,
    NetworkData,
    audit_route,
    build_model,
    build_network,
    decompose,
    gate_g1,
    index_active_arcs,
    map_route_to_arcs,
    solve,
    validate_configuration,
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
        self.assertEqual(data.g_kwh, 300.0)
        self.assertEqual(data.charge_kw, 300.0)
        self.assertEqual(data.reserve_kwh, 0.0)
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

    def test_frozen_physics_and_commensurate_grid(self):
        data = build_network(
            "scale_ladder/instances/Practice_Custom_DutyUnion_k02_r2.csv",
            soc_step=10.0,
            block_min=10,
            g_kwh=240.0,
            charge_kw=240.0,
            min_soc_frac=0.0,
        )
        self.assertEqual(data.g_kwh, 240.0)
        self.assertEqual(data.charge_kw, 240.0)
        self.assertEqual(data.reserve_kwh, 0.0)
        self.assertEqual(data.network.grid[-1], 240.0)
        self.assertEqual(data.network.block_kwh, 40.0)

    def test_grid_validation_preserves_legacy_and_rejects_noncommensurate(self):
        validate_configuration(
            g_kwh=300.0, charge_kw=300.0, min_soc_frac=0.0,
            soc_step=15.0, block_min=10,
        )
        validate_configuration(
            g_kwh=240.0, charge_kw=240.0, min_soc_frac=0.0,
            soc_step=10.0, block_min=10,
        )
        validate_configuration(
            g_kwh=240.0, charge_kw=240.0, min_soc_frac=0.0,
            soc_step=2.5, block_min=5,
        )
        invalid = [
            dict(g_kwh=240, charge_kw=240, min_soc_frac=0,
                 soc_step=15, block_min=10),
            dict(g_kwh=300, charge_kw=300, min_soc_frac=0,
                 soc_step=10, block_min=7),
            dict(g_kwh=240, charge_kw=240, min_soc_frac=1.1,
                 soc_step=10, block_min=10),
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_configuration(**values)

    def test_g2_status_binds_physics(self):
        data = fake_data()
        with tempfile.TemporaryDirectory() as folder:
            status_path = Path(folder) / "run.json"
            journal = Path(str(status_path) + ".columns.jsonl")
            journal.write_text("")
            status = {
                "csv": data.csv_name,
                "prices_csv": data.prices_csv,
                "soc_step": data.soc_step,
                "block_min": data.block_min,
                "g_kwh": data.g_kwh,
                "charge_kw": data.charge_kw,
                "min_soc_frac": 0.0,
                "trip_ids": [0],
            }
            status_path.write_text(json.dumps(status))
            self.assertEqual(
                _validate_journal_identity(data, journal), status_path
            )
            status["g_kwh"] = 240.0
            status_path.write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, "g_kwh"):
                _validate_journal_identity(data, journal)


if __name__ == "__main__":
    unittest.main()

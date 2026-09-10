import sys
import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gurobipy as gp
from gurobipy import GRB


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from run_capacity_speed_event_cg import (  # noqa: E402
    ExactCapacityMaster,
    add_fleet_incumbent_cap,
    classify_saved_pool,
    duplicate_service_audit,
    fleet_cap_from_stage1,
    physical_capacity_audit,
    solve_mip,
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

    def test_unproved_incumbent_is_valid_stage2_upper_cap(self):
        stage1 = {
            "has_solution": True,
            "incumbent_fleet": 3,
            "fleet_proven": False,
        }
        cap = fleet_cap_from_stage1(stage1)
        self.assertEqual(cap["sense"], "<=")
        self.assertEqual(cap["rhs"], 3)
        self.assertFalse(cap["source_fleet_proven"])

        model = gp.Model("test_fleet_cap_direction")
        model.Params.OutputFlag = 0
        x = model.addVars(3, vtype=GRB.BINARY)
        constraint, _cap = add_fleet_incumbent_cap(
            model, gp.quicksum(x.values()), stage1,
        )
        model.addConstr(x[0] + x[1] >= 2)
        model.setObjective(gp.quicksum(x.values()), GRB.MINIMIZE)
        model.optimize()
        self.assertEqual(constraint.Sense, "<")
        self.assertAlmostEqual(model.ObjVal, 2.0)

    def test_timed_cg_pool_is_usable_but_explicitly_uncertified(self):
        status = {
            "certified_rc_optimal": False,
            "status": "incomplete",
            "stop_reason": "cg_wall_limit",
        }
        accepted = classify_saved_pool(
            status, [route([0], 1.0), route([1], 1.0)], (0, 1),
        )
        self.assertEqual(
            accepted["classification"],
            "timed_uncertified_exact_event_cg_pool",
        )
        self.assertFalse(accepted["cg_pricing_certified"])
        with self.assertRaisesRegex(ValueError, "no usable saved pool"):
            classify_saved_pool(status, [route([0], 1.0)], (0, 1))

    def test_infeasible_capacity_pool_is_strict_json_serializable(self):
        problem = SimpleNamespace(trips=(0, 1))
        routes = [
            route([0], 1.0, 10.0, 20.0),
            route([1], 1.0, 10.0, 20.0),
        ]
        args = SimpleNamespace(
            threads=1, mip_gap=1e-4, mip_wall_s=1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = solve_mip(
                args, problem, routes, True,
                Path(directory) / "capacity_infeasible.log",
            )
        self.assertFalse(result["has_solution"])
        self.assertEqual(
            result["stage2"]["skip_reason"],
            "no_usable_stage1_fleet_incumbent",
        )
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

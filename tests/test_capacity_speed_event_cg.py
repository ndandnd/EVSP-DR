import json
import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gurobipy as gp
from gurobipy import GRB


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from run_capacity_speed_event_cg import (  # noqa: E402
    ExactCapacityMaster,
    add_fleet_incumbent_cap,
    atomic_pool,
    classify_saved_pool,
    duplicate_service_audit,
    fleet_cap_from_stage1,
    load_resume_pool,
    physical_capacity_audit,
    run_cg,
    solve_mip,
)
from event_pricer_network import (  # noqa: E402
    EventExpandedNetwork,
    PricingDeadlineExceeded,
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
    def test_atomic_pool_keeps_previous_checkpoint_if_replace_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "pool.jsonl"
            atomic_pool(path, [route([0], 1.0)])
            previous = path.read_bytes()
            with mock.patch(
                "run_capacity_speed_event_cg.os.replace",
                side_effect=OSError("synthetic interrupted replace"),
            ):
                with self.assertRaisesRegex(OSError, "synthetic"):
                    atomic_pool(path, [route([0], 2.0)])
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(path.parent.glob(".pool.jsonl.*")), [])

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
        for stop_reason in ("cg_wall_limit", "pricing_deadline"):
            status = {
                "certified_rc_optimal": False,
                "status": "incomplete",
                "stop_reason": stop_reason,
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

    def test_exact_event_pricing_honors_expired_deadline(self):
        network = EventExpandedNetwork.__new__(EventExpandedNetwork)
        network.problem = SimpleNamespace(trips=(0,))
        network.arc_mode = "explicit"
        network.node_meta = [("source", None, None), ("sink", None, None)]
        network.topo = [0, 1]
        network.out = [[(1, 0.0, -1, {"kind": "direct"})], []]
        with self.assertRaises(PricingDeadlineExceeded):
            network.min_reduced_cost_route(
                {}, deadline=10.0, clock=lambda: 10.0,
            )

    def test_interrupted_pool_resumes_to_uninterrupted_equivalent(self):
        class FakeMaster:
            def __init__(self, trips, **_kwargs):
                self.routes = []

            def add_route(self, candidate):
                self.routes.append(candidate)

            def solve(self):
                count = len(self.routes)
                dual = {1: 10.0, 2: 9.0}.get(count, float(count))
                return {
                    "objective": float(count), "runtime_s": 0.0,
                    "rows": 1, "columns": count, "nonzeros": count,
                    "artificial_total": 0.0, "route_weight": 1.0,
                    "trip_duals": {0: dual, "_route_count": count},
                    "capacity_duals": {},
                }

        class FakeNetwork:
            def __init__(self, interrupt_at=None):
                self.interrupt_at = interrupt_at

            def fixed_sequence_record(self, trips):
                return route(list(trips), 10.0)

            def min_reduced_cost_route(self, duals, **_kwargs):
                count = int(duals["_route_count"])
                if count == self.interrupt_at:
                    raise PricingDeadlineExceeded("synthetic deadline")
                if count == 1:
                    record = route([0], 9.0, 10.0, 20.0)
                    return {"rc": -1.0, "_event_record": record}
                if count == 2:
                    record = route([0], 8.0, 20.0, 30.0)
                    return {"rc": -1.0, "_event_record": record}
                record = route([0], float(count))
                return {"rc": 0.0, "_event_record": record}

            def metrics(self):
                return {"synthetic": True}

        args = SimpleNamespace(
            arm="baseline", threads=1, cg_wall_s=100.0, max_iters=10,
            rc_eps=1e-5, battery_kwh=240.0, reserve_kwh=0.0,
            soc_step=2.5, block_min=5, non_parx_kw=240.0, resume=False,
        )
        problem = SimpleNamespace(trips=(0,))
        prov = {
            "git_commit": "immutable",
            "instance_sha256": "instance", "prices_sha256": "prices",
            "reference_sha256": "reference", "deadhead_sha256": "deadhead",
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            resumed_pool = root / "resumed.pool.jsonl"
            with (
                mock.patch(
                    "run_capacity_speed_event_cg.ExactCapacityMaster", FakeMaster,
                ),
                mock.patch(
                    "run_capacity_speed_event_cg.build_network",
                    return_value=FakeNetwork(interrupt_at=2),
                ),
            ):
                interrupted = run_cg(
                    args, problem, {}, prov, root / "interrupted.json",
                    resumed_pool,
                )
            self.assertEqual(interrupted["stop_reason"], "pricing_deadline")
            self.assertFalse(interrupted["certified_rc_optimal"])
            self.assertIsNone(interrupted["terminal_exact_min_reduced_cost"])
            checkpoint = [
                json.loads(line) for line in resumed_pool.read_text().splitlines()
            ]
            self.assertEqual(len(checkpoint), 2)
            self.assertTrue(all(row.get("cg_checkpoint_id") for row in checkpoint))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_resume_pool(
                    resumed_pool, expected_id="wrong", trips=problem.trips,
                )
            with self.assertRaisesRegex(ValueError, "physical replay"):
                load_resume_pool(
                    resumed_pool,
                    expected_id=checkpoint[0]["cg_checkpoint_id"],
                    trips=problem.trips,
                    route_validator=lambda _route: "synthetic invalid route",
                )

            args.resume = True
            with (
                mock.patch(
                    "run_capacity_speed_event_cg.ExactCapacityMaster", FakeMaster,
                ),
                mock.patch(
                    "run_capacity_speed_event_cg.build_network",
                    return_value=FakeNetwork(),
                ),
                mock.patch(
                    "run_capacity_speed_event_cg.validate_injected_route",
                    return_value=None,
                ),
            ):
                resumed = run_cg(
                    args, problem, {}, prov, root / "resumed.json", resumed_pool,
                )
            self.assertTrue(resumed["certified_rc_optimal"])
            resumed_bytes = resumed_pool.read_bytes()

            args.resume = False
            uninterrupted_pool = root / "uninterrupted.pool.jsonl"
            with (
                mock.patch(
                    "run_capacity_speed_event_cg.ExactCapacityMaster", FakeMaster,
                ),
                mock.patch(
                    "run_capacity_speed_event_cg.build_network",
                    return_value=FakeNetwork(),
                ),
            ):
                uninterrupted = run_cg(
                    args, problem, {}, prov, root / "uninterrupted.json",
                    uninterrupted_pool,
                )
            self.assertTrue(uninterrupted["certified_rc_optimal"])
            self.assertEqual(resumed_bytes, uninterrupted_pool.read_bytes())


if __name__ == "__main__":
    unittest.main()

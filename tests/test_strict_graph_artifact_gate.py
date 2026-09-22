import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_strict_graph_artifact as gate


class FullGraphGateTests(unittest.TestCase):
    def test_exactly_two_distinct_deterministic_diagnostic_duals(self):
        problem = SimpleNamespace(trips=(0, 1, 2))
        saved = [{"trips": [0, 1]}, {"trips": [0, 2]}]
        first = gate.fixed_duals(problem, saved, 100000.0)
        self.assertEqual(first, gate.fixed_duals(problem, saved, 100000.0))
        self.assertEqual(len(first), 2)
        self.assertEqual(set(first[0][1]), set(problem.trips))
        self.assertTrue(all(value == 0 for value in first[0][1].values()))
        self.assertTrue(all(value > 0 for value in first[1][1].values()))

    def test_partial_preparation_never_triggers_second_build(self):
        self.assertEqual(gate.build_action(cache_exists=False, cold_exists=False,
            marker_exists=False, seal_exists=False), "cold_then_reload")
        self.assertEqual(gate.build_action(cache_exists=True, cold_exists=True,
            marker_exists=True, seal_exists=True), "reload_only")
        for state in ((True, False, True, False), (True, True, True, False),
                      (False, False, True, False), (False, True, False, False)):
            with self.assertRaisesRegex(ValueError, "no automatic second build"):
                gate.build_action(**dict(zip(("cache_exists", "cold_exists", "marker_exists", "seal_exists"), state)))

    def test_cold_baseline_identity_checked_before_reload(self):
        cold = {"schema": gate.SCHEMA, "phase": "cold", "status": "passed",
            "manifest_sha256": "m", "wrapper_commit": "w",
            "model_provenance": {"git_commit": gate.MODEL_COMMIT},
            "initial_pool": {"initial_pool_columns": 8397}}
        gate.validate_cold_baseline(cold, "m", "w")
        for key, value in (("phase", "reload"), ("status", "failed"), ("manifest_sha256", "wrong"),
                           ("wrapper_commit", "wrong"), ("model_provenance", {"git_commit": "wrong"})):
            bad = {**cold, key: value}
            with self.assertRaises(ValueError):
                gate.validate_cold_baseline(bad, "m", "w")

    def test_phase_comparison_rejects_pricing_graph_and_initializer_drift(self):
        row = {"name": "x", "duals_sha256": "d", "reduced_cost": 1.0,
            "independent_reduced_cost": 1.0, "route_sha256": "r", "physical_replay": "passed"}
        cold = {"identity_sha256": "i", "payload_sha256": "p", "graph_fingerprint": {"a": "h"},
            "network": {"dag_arcs": 3}, "queries": [row, {**row, "name": "y"}],
            "initial_pool": {"ordered_record_sha256": "initial", "singleton_checks": [{"record_sha256": "s"}]}}
        gate.compare_phases(cold, copy.deepcopy(cold))
        for mutate in (lambda x: x.update(payload_sha256="bad"),
                       lambda x: x["graph_fingerprint"].update(a="bad"),
                       lambda x: x["queries"][0].update(reduced_cost=2),
                       lambda x: x["initial_pool"].update(ordered_record_sha256="bad"),
                       lambda x: x["initial_pool"]["singleton_checks"][0].update(record_sha256="bad")):
            bad = copy.deepcopy(cold); mutate(bad)
            with self.assertRaises(ValueError):
                gate.compare_phases(cold, bad)

    def test_query_gate_rejects_independent_rc_and_physical_errors(self):
        problem = SimpleNamespace(trips=(0,))
        record = {"trips": [0], "cost": 100000.0}
        class Network:
            def min_reduced_cost_route(self, duals, **kwargs):
                return {"rc": record["cost"] - duals[0] + 1.0, "_event_record": record}
        with self.assertRaisesRegex(ValueError, "reduced-cost mismatch"):
            gate.query_parity_records(Network(), problem, [{"trips": [0]}], lambda r: None,
                bus_cost=100000.0, query_limit_s=600)
        with self.assertRaisesRegex(ValueError, "physical replay failed"):
            gate.query_parity_records(Network(), problem, [{"trips": [0]}], lambda r: "low SOC",
                bus_cost=100000.0, query_limit_s=600)


if __name__ == "__main__":
    unittest.main()

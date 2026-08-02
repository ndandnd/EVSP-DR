import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_giro_known_columns import DEFAULT_DATA_DIR  # noqa: E402
from audit_structural_route_bound import (  # noqa: E402
    LP_BOUND_EXPLANATION,
    audit_instance,
    maximum_reachability_antichain,
    reachability_closure,
    write_json_report,
)


class ReachabilityAntichainTests(unittest.TestCase):
    def setUp(self):
        # A non-transitively-closed DAG where the minimum vertex-disjoint path
        # cover has value 3, but four half-weight overlapping paths give a
        # fractional set cover of value 2.
        self.successors = {
            0: (2,),
            1: (2,),
            2: (3, 4),
            3: (),
            4: (),
        }

    def test_reachability_closure_adds_missing_comparabilities(self):
        closure, topological = reachability_closure(self.successors)

        self.assertEqual(set(topological), set(self.successors))
        self.assertEqual(closure[0], frozenset({2, 3, 4}))
        self.assertEqual(closure[1], frozenset({2, 3, 4}))
        self.assertEqual(closure[2], frozenset({3, 4}))

    def test_explicit_antichain_is_fractional_route_weight_certificate(self):
        certificate = maximum_reachability_antichain(self.successors)

        self.assertEqual(certificate.direct_minimum_path_cover, 3)
        self.assertEqual(certificate.reachability_antichain_bound, 2)
        self.assertEqual(len(certificate.antichain), 2)
        self.assertFalse(certificate.transitively_closed)
        self.assertTrue(certificate.pairwise_incomparable)

        closure, _ = reachability_closure(self.successors)
        feasible_paths = (
            (0, 2, 3),
            (0, 2, 4),
            (1, 2, 3),
            (1, 2, 4),
        )
        for path in feasible_paths:
            self.assertLessEqual(
                len(set(path).intersection(certificate.antichain)),
                1,
            )
            for left, right in zip(path, path[1:]):
                self.assertIn(right, closure[left])

        fractional_cover = {path: 0.5 for path in feasible_paths}
        for node in self.successors:
            coverage = sum(
                weight for path, weight in fractional_cover.items()
                if node in path
            )
            self.assertGreaterEqual(coverage, 1.0)
        self.assertEqual(sum(fractional_cover.values()), 2.0)

    def test_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "acyclic"):
            reachability_closure({0: (1,), 1: (0,)})


class CurrentInstanceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit_instance(DEFAULT_DATA_DIR, "Practice_10bus.csv")

    def test_tracked_hard_ten_has_valid_exact_antichain(self):
        report = self.report
        bound = report["reachability_antichain_lp_route_weight_bound"]

        self.assertEqual(report["trip_count"], 329)
        self.assertEqual(report["peak_trip_concurrency"], 10)
        self.assertEqual(bound, 10)
        self.assertEqual(len(report["antichain_trips"]), bound)
        self.assertTrue(
            all(
                isinstance(trip_id, int)
                for trip_id in report["antichain_ordered_trip_ids"]
            )
        )
        self.assertTrue(
            report["graph_validation"]["antichain_pairwise_incomparable"]
        )
        self.assertTrue(
            report["graph_validation"]["matching_antichain_identity_holds"]
        )
        self.assertTrue(report["validity"]["lp_route_weight_bound"])
        self.assertEqual(report["validity"]["explanation"], LP_BOUND_EXPLANATION)

    def test_json_writer_preserves_exact_antichain_ids(self):
        bundle = {
            "schema_version": 1,
            "instances": [self.report],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "bound.json"
            write_json_report(bundle, output)
            restored = json.loads(output.read_text(encoding="utf-8"))

        restored_report = restored["instances"][0]
        self.assertEqual(
            restored_report["antichain_local_trip_indices"],
            self.report["antichain_local_trip_indices"],
        )
        self.assertEqual(
            restored_report["antichain_ordered_trip_ids"],
            self.report["antichain_ordered_trip_ids"],
        )


if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_enumeration_oracle import (  # noqa: E402
    DEFAULT_SEED,
    TinyNetwork,
    compare_spec,
    enumerate_route_masks,
    generate_spec,
    mutation_specs,
    run_campaign,
)


class TinyEnumerationOracleTests(unittest.TestCase):
    def test_generation_and_enumeration_are_deterministic(self):
        left = generate_spec(DEFAULT_SEED, 7)
        right = generate_spec(DEFAULT_SEED, 7)
        self.assertEqual(left, right)
        self.assertEqual(
            enumerate_route_masks(TinyNetwork(left)),
            enumerate_route_masks(TinyNetwork(right)),
        )
        self.assertGreaterEqual(left.trip_count, 8)
        self.assertLessEqual(left.trip_count, 14)
        self.assertIn(left.station_count, {1, 2})

    def test_four_methods_agree_on_seeded_case(self):
        result = compare_spec(generate_spec(DEFAULT_SEED, 0))
        self.assertTrue(result["lp_agrees"], result)
        self.assertTrue(result["integer_agrees"], result)
        self.assertTrue(result["agreement"], result)

    def test_each_targeted_mutation_changes_optimum(self):
        for name, (baseline, mutated) in mutation_specs().items():
            with self.subTest(name=name):
                left = compare_spec(baseline)
                right = compare_spec(mutated)
                self.assertTrue(left["agreement"], left)
                self.assertTrue(right["agreement"], right)
                self.assertNotEqual(
                    left["integer"]["brute_force"],
                    right["integer"]["brute_force"],
                )

    def test_small_campaign_emits_every_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_campaign(
                DEFAULT_SEED, 5, Path(temporary),
            )
            self.assertEqual(summary["cases"], 5)
            self.assertEqual(
                summary["agreements"] + summary["disagreements"], 5,
            )
            self.assertTrue((Path(temporary) / "agreement.csv").is_file())
            self.assertTrue((Path(temporary) / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()

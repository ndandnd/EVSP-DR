import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_giro_known_columns import DEPOT  # noqa: E402
from exact_pricer_expanded import direct_singleton_seed_records  # noqa: E402
from prepare_exact_pool_mip import (  # noqa: E402
    default_output_path,
    journal_for_output,
    merge_pool,
)


class ExactPoolSeedTests(unittest.TestCase):
    def problem(self):
        return SimpleNamespace(
            trips=[1, 2],
            start_min={1: 60.0, 2: 120.0},
            end_min={1: 90.0, 2: 150.0},
            trip_energy={1: 20.0, 2: 95.0},
            adjacency={
                DEPOT: [
                    (1, 10.0, 10.0, "depot_trip"),
                    (2, 10.0, 10.0, "depot_trip"),
                ],
                1: [(DEPOT, 10.0, 10.0, "trip_depot")],
                2: [(DEPOT, 10.0, 10.0, "trip_depot")],
            },
        )

    def test_direct_singletons_are_real_grid_feasible_columns(self):
        seeds, missing = direct_singleton_seed_records(
            self.problem(), g_kwh=100.0, soc_step=10.0, reserve_kwh=0.0
        )

        self.assertEqual([route["trips"] for route in seeds], [[1]])
        self.assertEqual(missing, [2])
        self.assertEqual(seeds[0]["route_nodes"], [DEPOT, 1, DEPOT])
        self.assertEqual(seeds[0]["cost"], 100000.0)
        self.assertEqual(seeds[0]["origin"], "exact_direct_singleton_seed")

    def test_merging_seeds_is_idempotent(self):
        seed = {"trips": [1], "cost": 100000.0, "origin": "seed"}
        merged, added = merge_pool([], [seed])
        merged_again, added_again = merge_pool(merged, [seed])

        self.assertEqual(added, 1)
        self.assertEqual(added_again, 0)
        self.assertEqual(merged_again, [seed])

    def test_prepared_pool_names_keep_snapshot_and_journal_adjacent(self):
        source = Path("/tmp/example.snapshot.json")
        output = default_output_path(source)

        self.assertEqual(output.name, "example.partition_ready.snapshot.json")
        self.assertEqual(
            journal_for_output(output).name,
            "example.partition_ready.columns.jsonl",
        )


if __name__ == "__main__":
    unittest.main()

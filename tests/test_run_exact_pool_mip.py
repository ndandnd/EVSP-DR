import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_exact_pool_mip import (  # noqa: E402
    finite_solver_value,
    greedy_partition_start_indices,
    load_pool,
    main,
    singleton_partition_indices,
)


class ExactPoolMipTests(unittest.TestCase):
    def test_singletons_are_a_strict_partition_seed(self):
        routes = [
            {"trips": [1, 2], "cost": 1.0},
            {"trips": [1], "cost": 2.0},
            {"trips": [2], "cost": 2.0},
        ]
        self.assertEqual(singleton_partition_indices(routes, [1, 2]), [1, 2])
        self.assertEqual(singleton_partition_indices(routes[:-1], [1, 2]), [])

    def test_solver_infinity_is_serialized_as_null(self):
        self.assertIsNone(finite_solver_value(float("inf")))
        self.assertIsNone(finite_solver_value(1.7976931348623157e308))
        self.assertEqual(finite_solver_value(42.5), 42.5)

    def test_greedy_start_replaces_singletons_with_disjoint_routes(self):
        routes = [
            {"trips": [1], "cost": 100000.0},
            {"trips": [2], "cost": 100000.0},
            {"trips": [3], "cost": 100000.0},
            {"trips": [1, 2], "cost": 100010.0},
            {"trips": [2, 3], "cost": 100020.0},
        ]

        start = greedy_partition_start_indices(routes, [1, 2, 3], [0, 1, 2])

        self.assertEqual(start, [3, 2])
        covered = [trip for index in start for trip in routes[index]["trips"]]
        self.assertEqual(sorted(covered), [1, 2, 3])

    def test_copied_snapshot_finds_adjacent_recorded_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = folder / "sample.snapshot.json"
            journal = folder / "sample.columns.jsonl"
            result.write_text(json.dumps({
                "csv": "sample.csv",
                "soc_step": 5,
                "trip_ids": [11, 12],
                "columns_journal": "/unavailable/cluster/sample.columns.jsonl",
            }))
            journal.write_text(
                json.dumps({"trips": [11], "cost": 1.0}) + "\n" +
                json.dumps({"trips": [12], "cost": 1.0}) + "\n"
            )

            _, routes, trips = load_pool(result)

            self.assertEqual(trips, [11, 12])
            self.assertEqual(len(routes), 2)

    def test_snapshot_prefers_frozen_sibling_over_recorded_live_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            live_folder = folder / "live"
            frozen_folder = folder / "frozen"
            live_folder.mkdir()
            frozen_folder.mkdir()
            live_journal = live_folder / "sample.columns.jsonl"
            frozen_journal = frozen_folder / "sample.columns.jsonl"
            result = frozen_folder / "sample.snapshot.json"
            result.write_text(json.dumps({
                "csv": "sample.csv",
                "soc_step": 5,
                "trip_ids": [11],
                "columns_journal": str(live_journal),
            }))
            live_journal.write_text(
                json.dumps({"trips": [11], "cost": 99.0}) + "\n"
            )
            frozen_journal.write_text(
                json.dumps({"trips": [11], "cost": 1.0}) + "\n"
            )

            _, routes, _ = load_pool(result)

            self.assertEqual(routes[0]["cost"], 1.0)

    def test_runner_refuses_to_overwrite_input_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "sample.json"
            result.write_text("{}")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["--result", str(result), "--out", str(result),
                          "--validate-only"])
            self.assertEqual(raised.exception.code, 2)

    def test_required_singleton_partition_rejects_coverage_only_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = folder / "triangle.json"
            journal = Path(str(result) + ".columns.jsonl")
            result.write_text(json.dumps({
                "csv": "triangle.csv",
                "soc_step": 5,
                "trip_ids": [1, 2, 3],
                "columns_journal": str(journal),
            }))
            journal.write_text("\n".join(json.dumps(route) for route in (
                {"trips": [1, 2], "cost": 1.0},
                {"trips": [1, 3], "cost": 1.0},
                {"trips": [2, 3], "cost": 1.0},
            )) + "\n")

            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    SystemExit, "singleton partition required"
                ):
                    main([
                        "--result", str(result),
                        "--require-singleton-partition",
                        "--validate-only",
                    ])


if __name__ == "__main__":
    unittest.main()

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freeze_exact_cg_at_wall import freeze  # noqa: E402


class MatchedWallSnapshotTests(unittest.TestCase):
    def test_snapshot_excludes_same_iteration_insertions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "cg.json"
            journal = Path(str(source) + ".columns.jsonl")
            iterations = Path(str(source) + ".iters.csv")
            records = [
                {"trips": [0], "cost": 1.0, "found_iter": 0},
                {"trips": [1], "cost": 1.0, "found_iter": 0},
                {"trips": [0, 1], "cost": 1.0, "found_iter": 1},
            ]
            journal.write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            source.write_text(json.dumps({
                "trip_ids": [0, 1],
                "columns_journal": str(journal),
                "certified_rc_optimal": True,
                "iterations": 2,
                "columns": 3,
                "final": {
                    "iter": 2, "lp_obj": 1.0, "route_weight": 1.0,
                    "artificials": 0.0, "min_rc": 0.0,
                },
            }))
            fields = (
                "elapsed_s", "iteration", "lp_obj", "route_weight",
                "artificials", "min_rc", "pool_columns",
            )
            with iterations.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerow({
                    "elapsed_s": 10, "iteration": 1, "lp_obj": 2,
                    "route_weight": 2, "artificials": 0,
                    "min_rc": -1, "pool_columns": 2,
                })
                writer.writerow({
                    "elapsed_s": 20, "iteration": 2, "lp_obj": 1,
                    "route_weight": 1, "artificials": 0,
                    "min_rc": 0, "pool_columns": 3,
                })
            output = root / "snapshot.json"
            snapshot = freeze(SimpleNamespace(
                result=source,
                out=output,
                budget_s=15.0,
                telemetry=None,
            ))
            self.assertEqual(snapshot["iterations"], 1)
            self.assertEqual(snapshot["columns"], 2)
            self.assertFalse(snapshot["certified_rc_optimal"])
            frozen = [
                json.loads(line)
                for line in Path(
                    snapshot["columns_journal"]
                ).read_text().splitlines()
            ]
            self.assertEqual([record["trips"] for record in frozen], [[0], [1]])
            self.assertEqual(
                snapshot["matched_wall_snapshot"][
                    "conservative_boundary"
                ],
                "columns_found_at_included_iteration_are_excluded",
            )


if __name__ == "__main__":
    unittest.main()

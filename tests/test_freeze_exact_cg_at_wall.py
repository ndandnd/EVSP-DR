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
    def test_snapshot_includes_same_iteration_after_durable_fsync(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "cg.json"
            journal = Path(str(source) + ".columns.jsonl")
            iterations = Path(str(source) + ".iters.csv")
            telemetry = root / "phases.jsonl"
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
            telemetry.write_text("\n".join(json.dumps(record) for record in (
                {
                    "record_type": "phase",
                    "phase": "route_insertion",
                    "iteration": 1,
                    "elapsed_session_s": 11.0,
                    "pool_columns": 3,
                    "peak_rss_bytes": 1024,
                    "details": {"inserted_or_replaced": 1},
                },
                {
                    "record_type": "phase",
                    "phase": "journal_fsync",
                    "iteration": 1,
                    "elapsed_session_s": 12.0,
                    "pool_columns": 3,
                    "peak_rss_bytes": 2048,
                    "details": {"records": 1},
                },
                {
                    "record_type": "phase",
                    "phase": "route_insertion",
                    "iteration": 2,
                    "elapsed_session_s": 21.0,
                    "pool_columns": 3,
                    "peak_rss_bytes": 4096,
                    "details": {"inserted_or_replaced": 0},
                },
            )) + "\n")
            output = root / "snapshot.json"
            snapshot = freeze(SimpleNamespace(
                result=source,
                out=output,
                budget_s=15.0,
                telemetry=telemetry,
            ))
            self.assertEqual(snapshot["iterations"], 1)
            self.assertEqual(snapshot["columns"], 3)
            self.assertFalse(snapshot["certified_rc_optimal"])
            frozen = [
                json.loads(line)
                for line in Path(
                    snapshot["columns_journal"]
                ).read_text().splitlines()
            ]
            self.assertEqual(
                [record["trips"] for record in frozen],
                [[0], [1], [0, 1]],
            )
            self.assertEqual(
                snapshot["matched_wall_snapshot"][
                    "conservative_boundary"
                ],
                "include_columns_only_through_last_durably_completed_iteration",
            )
            self.assertEqual(
                snapshot["matched_wall_snapshot"][
                    "durable_completion_elapsed_s"
                ],
                12.0,
            )


if __name__ == "__main__":
    unittest.main()

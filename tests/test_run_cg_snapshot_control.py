import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_cg_snapshot_control import (  # noqa: E402
    control_result_complete,
    prepare_snapshot_resume,
)


class SnapshotControlTests(unittest.TestCase):
    def test_preparation_copies_immutable_pool_and_anchors_elapsed_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            snapshot = folder / "source.m360.snapshot.json"
            source_journal = folder / "source.m360.columns.jsonl"
            source_journal.write_text(json.dumps({
                "trips": [1], "cost": 100000.0
            }) + "\n")
            snapshot.write_text(json.dumps({
                "csv": "sample.csv",
                "prices_csv": "hourly_prices_flat.csv",
                "soc_step": 15.0,
                "block_min": 10,
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "min_soc_frac": 0.0,
                "master_sense": "partition",
                "trip_ids": [1],
                "columns": 1,
                "iterations": 12,
                "wall_s": 21637.5,
                "columns_journal": str(source_journal),
                "final": {
                    "lp_obj": 100000.0,
                    "route_weight": 1.0,
                    "artificials": 0.0,
                    "min_rc": -2.0,
                },
            }))
            out = folder / "control.json"

            prepare_snapshot_resume(snapshot, out, 360.0)

            prepared = json.loads(out.read_text())
            self.assertEqual(prepared["stop_reason"],
                             "prepared_snapshot_resume")
            self.assertEqual(prepared["resume_parent"]["snapshot_minutes"],
                             360.0)
            self.assertTrue(Path(str(out) + ".columns.jsonl").exists())
            rows = Path(str(out) + ".iters.csv").read_text().splitlines()
            self.assertTrue(rows[1].startswith("21637.50,12,100000.0,1.0"))
            self.assertEqual(
                prepared["resume_parent"]["snapshot_actual_wall_s"],
                21637.5,
            )

            # A preemption between publishing the prepared status and writing
            # its synthetic trajectory anchor is narrowly recoverable.
            journal_copy = Path(str(out) + ".columns.jsonl")
            iters = Path(str(out) + ".iters.csv")
            iters.unlink()
            prepare_snapshot_resume(snapshot, out, 360.0)
            self.assertEqual(journal_copy.read_text(), source_journal.read_text())
            self.assertTrue(iters.exists())

            # Once status exists, a journal that lost its source prefix is not
            # silently reconstructed: it may contain uncheckpointed work.
            journal_copy.write_text('{"trips":')
            with self.assertRaisesRegex(ValueError, "source journal as a prefix"):
                prepare_snapshot_resume(snapshot, out, 360.0)

    def test_preemption_before_status_checkpoint_preserves_pricing_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            snapshot = folder / "source.m360.snapshot.json"
            source_journal = folder / "source.m360.columns.jsonl"
            source_journal.write_text(
                json.dumps({"trips": [1], "cost": 100000.0}) + "\n"
            )
            snapshot.write_text(json.dumps({
                "csv": "sample.csv",
                "prices_csv": "hourly_prices_flat.csv",
                "soc_step": 15.0,
                "block_min": 10,
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "min_soc_frac": 0.0,
                "master_sense": "partition",
                "trip_ids": [1, 2],
                "columns": 1,
                "iterations": 12,
                "wall_s": 21600.0,
                "columns_journal": str(source_journal),
                "final": {"lp_obj": 100000.0, "route_weight": 1.0,
                          "artificials": 0.0, "min_rc": -2.0},
            }))
            out = folder / "control.json"
            prepare_snapshot_resume(snapshot, out, 360.0,
                                    continuation_commit="commit-a")

            copied = Path(str(out) + ".columns.jsonl")
            appended = json.dumps({"trips": [2], "cost": 100001.0}) + "\n"
            with copied.open("a") as fh:
                fh.write(appended)
            with Path(str(out) + ".iters.csv").open("a") as fh:
                fh.write("21640,13,99999,1,0,-1,2\n")

            prepare_snapshot_resume(snapshot, out, 360.0,
                                    continuation_commit="commit-a")

            self.assertTrue(copied.read_text().endswith(appended))
            rows = Path(str(out) + ".iters.csv").read_text().splitlines()
            self.assertEqual(len(rows), 3)

            with self.assertRaisesRegex(ValueError, "mixing algorithms"):
                prepare_snapshot_resume(
                    snapshot, out, 360.0,
                    continuation_commit="commit-b",
                )

    def test_running_control_refuses_missing_isolated_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            snapshot = folder / "source.m60.snapshot.json"
            source_journal = folder / "source.columns.jsonl"
            source_journal.write_text(
                json.dumps({"trips": [1], "cost": 100000.0}) + "\n"
            )
            snapshot.write_text(json.dumps({
                "csv": "sample.csv", "prices_csv": "prices.csv",
                "soc_step": 5.0, "block_min": 10,
                "g_kwh": 300.0, "charge_kw": 300.0,
                "min_soc_frac": 0.0, "master_sense": "partition",
                "trip_ids": [1], "columns": 1, "iterations": 1,
                "wall_s": 3600.0, "columns_journal": str(source_journal),
                "final": {"lp_obj": 100000.0, "route_weight": 1.0,
                          "artificials": 0.0, "min_rc": -1.0},
            }))
            out = folder / "control.json"
            prepare_snapshot_resume(snapshot, out, 60.0)
            status = json.loads(out.read_text())
            status["stop_reason"] = "running"
            out.write_text(json.dumps(status))

            out_journal = Path(str(out) + ".columns.jsonl")
            out_journal.unlink()
            with self.assertRaisesRegex(ValueError, "lost its isolated journal"):
                prepare_snapshot_resume(snapshot, out, 60.0)

            # Restore the journal but remove the trajectory. A running status
            # cannot be downgraded to a fresh preparation.
            out_journal.write_text(source_journal.read_text())
            Path(str(out) + ".iters.csv").unlink()
            with self.assertRaisesRegex(ValueError, "lost its iteration"):
                prepare_snapshot_resume(snapshot, out, 60.0)

    def test_incompatible_commit_does_not_repair_truncated_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            snapshot = folder / "source.m60.snapshot.json"
            source_journal = folder / "source.columns.jsonl"
            source_journal.write_text(
                json.dumps({"trips": [1], "cost": 100000.0}) + "\n"
            )
            snapshot.write_text(json.dumps({
                "csv": "sample.csv", "prices_csv": "prices.csv",
                "soc_step": 5.0, "block_min": 10,
                "g_kwh": 300.0, "charge_kw": 300.0,
                "min_soc_frac": 0.0, "master_sense": "partition",
                "trip_ids": [1], "columns": 1, "iterations": 1,
                "wall_s": 3600.0, "columns_journal": str(source_journal),
                "final": {"lp_obj": 100000.0, "route_weight": 1.0,
                          "artificials": 0.0, "min_rc": -1.0},
            }))
            out = folder / "control.json"
            prepare_snapshot_resume(
                snapshot, out, 60.0, continuation_commit="commit-a"
            )
            out_journal = Path(str(out) + ".columns.jsonl")
            with out_journal.open("ab") as handle:
                handle.write(b'{"trips":')
            iters = Path(str(out) + ".iters.csv")
            with iters.open("ab") as handle:
                handle.write(b"3700,2,")
            journal_before = out_journal.read_bytes()
            iters_before = iters.read_bytes()

            with self.assertRaisesRegex(ValueError, "no artifact was repaired"):
                prepare_snapshot_resume(
                    snapshot, out, 60.0, continuation_commit="commit-b"
                )

            self.assertEqual(out_journal.read_bytes(), journal_before)
            self.assertEqual(iters.read_bytes(), iters_before)

    def test_completion_requires_semantic_terminal_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "control.json"
            path.write_text(json.dumps({
                "stop_reason": "prepared_snapshot_resume",
                "wall_s": 21600,
            }))
            self.assertFalse(control_result_complete(path, 172800))

            path.write_text(json.dumps({
                "stop_reason": "wall_limit",
                "wall_s": 172740,
                "provenance": {
                    "git_commit": "campaign",
                    "args": {"wall_limit_s": 172800},
                },
            }))
            self.assertTrue(control_result_complete(path, 172800))
            self.assertTrue(control_result_complete(
                path, 172800, expected_commit="campaign"
            ))
            self.assertFalse(control_result_complete(
                path, 172800, expected_commit="different"
            ))

            path.write_text(json.dumps({
                "stop_reason": "wall_limit",
                "wall_s": 172740,
                "provenance": {"args": {"wall_limit_s": 200000}},
            }))
            self.assertFalse(control_result_complete(path, 172800))

            path.write_text(json.dumps({
                "stop_reason": "master_failed",
                "wall_s": 100000,
            }))
            self.assertFalse(control_result_complete(path, 172800))


if __name__ == "__main__":
    unittest.main()

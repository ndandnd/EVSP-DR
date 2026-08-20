import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
import sys
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from monitor_legacy_recovery import (
    ITERATION_HEADER,
    SlurmRecord,
    array_task_records,
    build_report,
    choose_recovery_records,
    iteration_trend,
    parse_iteration_rows,
    parse_sacct,
    query_array,
    unsuccessful_array_records,
    verify_attestation,
)


class LegacyRecoveryMonitorTests(unittest.TestCase):
    def test_array_identity_uses_job_id_not_job_id_raw(self):
        records = parse_sacct(
            "867334_22|867359|cg-bigtar|FAILED|00:00:24|1:0|start|end|\n"
            "867334_24|867361|cg-bigtar|FAILED|00:00:21|1:0|start|end|\n"
        )

        tasks = array_task_records(records, 867334)

        self.assertEqual(set(tasks), {22, 24})
        self.assertEqual(tasks[22].job_id_raw, "867359")
        self.assertNotIn(867359, tasks)

    def test_all_unsuccessful_array_tasks_keep_exact_array_identities(self):
        records = parse_sacct(
            "867334_22|867359|cg-bigtar|FAILED|00:00:24|1:0|start|end|\n"
            "867334_24|867361|cg-bigtar|FAILED|00:00:21|1:0|start|end|\n"
            "867334_32|867369|cg-bigtar|FAILED|00:00:39|1:0|start|end|\n"
            "867334_34|867371|cg-bigtar|FAILED|00:01:01|1:0|start|end|\n"
            "867334_40|867377|cg-bigtar|COMPLETED|08:00:00|0:0|start|end|\n"
        )
        failures = unsuccessful_array_records(records, 867334)
        self.assertEqual([item["task"] for item in failures], [22, 24, 32, 34])
        self.assertEqual(
            [item["job_id"] for item in failures],
            ["867334_22", "867334_24", "867334_32", "867334_34"],
        )
        self.assertEqual(
            [item["job_id_raw"] for item in failures],
            ["867359", "867361", "867369", "867371"],
        )

    def test_array_query_requests_expanded_array_records(self):
        with patch(
            "monitor_legacy_recovery.run_command", return_value=("", None)
        ) as command:
            query_array(867334)
        arguments = command.call_args.args[0]
        self.assertIn("--array", arguments)
        self.assertIn("JobID%64", arguments[-1])

    def test_recovery_discovery_does_not_fall_back_to_old_commit(self):
        queue = [
            SlurmRecord(
                "900001", "", "R22-30r2-p18-caaaaaa", "RUNNING",
                source="squeue",
            )
        ]
        self.assertEqual(
            choose_recovery_records(
                queue,
                task=22,
                continuation_commit="b" * 40,
                explicit_job=None,
            ),
            [],
        )

    def test_recovery_discovery_requires_valid_continuation_identity(self):
        queue = [
            SlurmRecord(
                "900001", "", "R22-30r2-p18-caaaaaa", "RUNNING",
                source="squeue",
            )
        ]
        for claim in (None, "", "not-a-commit"):
            with self.subTest(claim=claim):
                self.assertEqual(
                    choose_recovery_records(
                        queue,
                        task=22,
                        continuation_commit=claim,
                        explicit_job=None,
                    ),
                    [],
                )

    def test_recent_iteration_trend_uses_elapsed_time(self):
        rows = parse_iteration_rows(
            [
                ITERATION_HEADER,
                "3600,100,3000000,30,0,-50000,10000",
                "5400,110,2990000,29.9,0,-40000,10200",
                "7200,120,2980000,29.8,0,-30000,10400",
            ]
        )

        trend = iteration_trend(rows, 60)

        self.assertEqual(trend["first_iteration"], 100)
        self.assertEqual(trend["last_iteration"], 120)
        self.assertAlmostEqual(trend["iterations_per_hour"], 20.0)
        self.assertAlmostEqual(trend["objective_drop_per_hour"], 20000.0)

    def _fixture(self, root: Path):
        result = (
            root
            / "src/results/legacy_recovery/job867334/cabc123def456/task22"
            / "cell.json"
        )
        result.parent.mkdir(parents=True)
        journal = Path(str(result) + ".columns.jsonl")
        iters = Path(str(result) + ".iters.csv")
        attestation_path = Path(str(result) + ".migration_attestation.json")
        raw_dir = result.parent / f"{result.name}.legacy_raw"
        raw_dir.mkdir()
        initial_journal = b'{"trips":[1],"cost":100000}\n'
        initial_iters = (
            ITERATION_HEADER + "\n3600,100,3000000,30,0,-50000,10000\n"
        ).encode()
        journal.write_bytes(initial_journal + b'{"trips":[2],"cost":100000}\n')
        iters.write_bytes(
            initial_iters
            + b"5400,110,2990000,29.9,0,-40000,10200\n"
            + b"7200,120,2980000,29.8,0,-30000,10400\n"
        )
        migration_id = "migration-1"
        status = {
            "stop_reason": "running",
            "certified_rc_optimal": False,
            "iterations": 120,
            "attempt_iterations": 20,
            "columns": 10400,
            "wall_s": 7200,
            "attempt_wall_s": 3600,
            "final": {
                "lp_obj": 2980000,
                "route_weight": 29.8,
                "artificials": 0,
                "min_rc": -30000,
            },
            "provenance": {"git_commit": "b" * 40},
            "resume_parent": {
                "schema": "evsp-dr-legacy-exact-pool-migration-v1",
                "migration_id": migration_id,
                "tool_commit": "a" * 40,
            },
        }
        result.write_text(json.dumps(status))
        attestation = {
            "schema": "evsp-dr-legacy-exact-pool-migration-v1",
            "migration_id": migration_id,
            "tool": {"commit": "a" * 40},
            "source": {"slurm_array_job": 867334, "slurm_task": 22, "logs": []},
            "validation": {
                "source_unchanged_during_copy": True,
                "copy_has_distinct_inode": True,
                "pricing_record_fields_valid": True,
                "trip_ids_witnessed": True,
                "elapsed_time_monotone": True,
            },
            "destination": {
                "result": str(result),
                "journal": str(journal),
                "journal_initial_bytes": len(initial_journal),
                "journal_initial_sha256": hashlib.sha256(initial_journal).hexdigest(),
                "iters": str(iters),
                "iters_initial_bytes": len(initial_iters),
                "iters_initial_sha256": hashlib.sha256(initial_iters).hexdigest(),
            },
            "repairs": {
                "journal": {"applied": True, "legacy_line_normalizations": []},
                "iters": {"applied": False},
            },
        }
        attestation_path.write_text(json.dumps(attestation))
        (raw_dir / "raw_manifest.json").write_text(json.dumps(attestation))
        return result, status

    def test_prefix_attestation_accepts_extended_append_only_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, status = self._fixture(Path(tmp))

            audit = verify_attestation(
                result,
                status,
                array_job=867334,
                task=22,
                mode="prefix",
            )

            self.assertTrue(audit["ok"], audit["errors"])
            self.assertIn("journal-prefix-sha256", audit["checks"])
            self.assertIn("iters-prefix-sha256", audit["checks"])
            self.assertEqual(audit["tool_commit"], "a" * 40)
            self.assertEqual(audit["continuation_commit"], "b" * 40)

    def test_attestation_fails_without_valid_current_continuation_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, status = self._fixture(Path(tmp))
            for claim in (None, "", "not-a-commit"):
                with self.subTest(claim=claim):
                    changed = json.loads(json.dumps(status))
                    if claim is None:
                        changed["provenance"].pop("git_commit")
                    else:
                        changed["provenance"]["git_commit"] = claim
                    audit = verify_attestation(
                        result,
                        changed,
                        array_job=867334,
                        task=22,
                        mode="prefix",
                    )
                    self.assertFalse(audit["ok"])
                    self.assertIsNone(audit["continuation_commit"])
                    self.assertIn(
                        "status provenance has no valid 40-character "
                        "continuation commit",
                        audit["errors"],
                    )

    def test_report_is_healthy_for_live_fresh_attested_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, _ = self._fixture(root)
            args = Namespace(
                root=root,
                result=result,
                array_job=867334,
                task=22,
                recovery_job="904367",
                user="nc437",
                attestation_mode="prefix",
                trend_minutes=60.0,
                trend_rows=500,
                stale_minutes=60.0,
                no_slurm=False,
            )
            array = [
                SlurmRecord(
                    "867334_22", "867359", "cg-bigtar", "FAILED",
                    "00:00:24", "1:0",
                )
            ]
            queue = [
                SlurmRecord(
                    "904367", "", "R22-30r2-p18-cbbbbbb", "RUNNING",
                    "1-01:00", source="squeue",
                )
            ]
            with (
                patch("monitor_legacy_recovery.query_array", return_value=(array, None)),
                patch("monitor_legacy_recovery.query_queue", return_value=(queue, None)),
                patch("monitor_legacy_recovery.scan_log_errors", return_value=[]),
            ):
                report = build_report(args)

            self.assertEqual(report["verdict"], "HEALTHY", report)
            self.assertEqual(report["original_task"]["job_id"], "867334_22")
            self.assertEqual(
                [item["task"] for item in report["original_unsuccessful_tasks"]],
                [22],
            )
            self.assertEqual(report["recovery_job"]["job_id"], "904367")
            self.assertEqual(report["latest_iteration"]["iteration"], 120)
            self.assertGreater(report["trend"]["iterations_per_hour"], 0)

    def test_explicit_job_does_not_hide_missing_continuation_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, status = self._fixture(root)
            status["provenance"].pop("git_commit")
            result.write_text(json.dumps(status))
            args = Namespace(
                root=root,
                result=result,
                array_job=867334,
                task=22,
                recovery_job="904367",
                user="nc437",
                attestation_mode="prefix",
                trend_minutes=60.0,
                trend_rows=500,
                stale_minutes=60.0,
                no_slurm=False,
            )
            queue = [
                SlurmRecord(
                    "904367", "", "R22-30r2-p18-cbbbbbb", "RUNNING",
                    source="squeue",
                )
            ]
            with (
                patch("monitor_legacy_recovery.query_array", return_value=([], None)),
                patch("monitor_legacy_recovery.query_queue", return_value=(queue, None)),
                patch("monitor_legacy_recovery.scan_log_errors", return_value=[]),
            ):
                report = build_report(args)

            self.assertEqual(report["recovery_job"]["job_id"], "904367")
            self.assertEqual(report["verdict"], "FAIL")
            self.assertIn(
                "status provenance has no valid 40-character continuation commit",
                report["errors"],
            )

    def test_bad_migrated_prefix_is_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, status = self._fixture(Path(tmp))
            journal = Path(str(result) + ".columns.jsonl")
            journal.write_bytes(b"X" + journal.read_bytes()[1:])

            audit = verify_attestation(
                result,
                status,
                array_job=867334,
                task=22,
                mode="prefix",
            )

            self.assertFalse(audit["ok"])
            self.assertIn("journal migrated prefix hash mismatch", audit["errors"])


if __name__ == "__main__":
    unittest.main()

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import archive_exact_cg_profile_campaign as archiver  # noqa: E402
import launch_exact_cg_profile_campaign as launcher  # noqa: E402
import monitor_exact_cg_profile_campaign as monitor  # noqa: E402
import summarize_exact_cg_profiles as summarizer  # noqa: E402


class ExactCgProfileCampaignTests(unittest.TestCase):
    def _campaign_fixture(self, root: Path):
        repo = root / "repo"
        data = repo / "data"
        data.mkdir(parents=True)
        instance = data / "instance.csv"
        prices = data / "prices.csv"
        instance.write_text("instance\n")
        prices.write_text("prices\n")
        instance_sha = hashlib.sha256(instance.read_bytes()).hexdigest()
        prices_sha = hashlib.sha256(prices.read_bytes()).hexdigest()
        snapshots = {}
        for label in ("historical", "ca", "cs", "pa", "ps"):
            folder = repo / "source" / label
            folder.mkdir(parents=True)
            result = folder / f"{label}.snapshot.json"
            journal = Path(str(result) + ".columns.jsonl")
            journal.write_text(
                json.dumps({"trips": [1], "cost": 100000.0}) + "\n"
            )
            result.write_text(json.dumps({
                "csv": "instance.csv",
                "prices_csv": "prices.csv",
                "trip_ids": [1],
                "columns": 1,
                "columns_journal": str(journal),
                "provenance": {
                    "instance_sha256": instance_sha,
                    "prices_sha256": prices_sha,
                },
            }))
            snapshots[label] = result
        return repo, snapshots

    def _args(self, snapshots):
        return Namespace(
            historical=snapshots["historical"],
            ca=snapshots["ca"],
            cs=snapshots["cs"],
            pa=snapshots["pa"],
            ps=snapshots["ps"],
            campaign="profile_dry_run",
            python=Path(sys.executable),
            solve_limit_s=120.0,
            repeat=3,
            mem_gb=64,
            job_hours=24,
            submit=False,
        )

    def test_launcher_is_dry_run_with_unique_hash_bound_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, snapshots = self._campaign_fixture(Path(tmp))
            args = self._args(snapshots)
            identity = {
                "expected_commit": "a" * 40,
                "observed_commit": "a" * 40,
                "detached": True,
                "tracked_clean": True,
                "profile_core_commit": launcher.PROFILE_CORE_COMMIT,
            }
            output = io.StringIO()
            with (
                patch.object(launcher, "REPO_ROOT", repo),
                patch.object(
                    launcher,
                    "reviewed_checkout_identity",
                    return_value=identity,
                ),
                patch.object(
                    launcher,
                    "validated_python",
                    return_value={
                        "path": str(Path(sys.executable).resolve()),
                        "version": "3.12.test",
                    },
                ),
                patch.object(
                    launcher,
                    "reviewed_worker_bytes",
                    return_value=b"#!/bin/bash\n",
                ),
                patch.object(
                    launcher.subprocess,
                    "run",
                    side_effect=AssertionError(
                        "dry-run must not invoke sbatch"
                    ),
                ),
                contextlib.redirect_stdout(output),
            ):
                manifest = launcher.launch(args)

            campaign_root = (
                repo / "src/results/exact_cg_profiles/profile_dry_run"
            )
            self.assertFalse(campaign_root.exists())
            self.assertFalse(manifest["submitted"])
            self.assertEqual(len(manifest["jobs"]), 5)
            outputs = [job["output"] for job in manifest["jobs"]]
            self.assertEqual(len(outputs), len(set(outputs)))
            self.assertEqual(
                [job["label"] for job in manifest["jobs"]],
                ["historical", "ca", "cs", "pa", "ps"],
            )
            for job in manifest["jobs"]:
                spec = job["job_spec"]
                self.assertEqual(
                    spec["source_hashes"], job["source_hashes"]
                )
                self.assertEqual(
                    spec["staged_journal_sha256"],
                    job["source_hashes"]["journal"],
                )
                command = job["command"]
                self.assertIn("--no-requeue", command)
                self.assertIn("--cpus-per-task=1", command)
                self.assertIn("--mem=64G", command)
                export = next(
                    item for item in command if item.startswith("--export=")
                )
                self.assertFalse(export.startswith("--export=ALL"))
                self.assertIn("EVSP_EXPECTED_COMMIT=" + "a" * 40, export)
                self.assertIn("EVSP_PROFILE_PYTHON=", export)
            self.assertIn("[dry-run]", output.getvalue())

    def test_python_environment_failure_is_fatal(self):
        with self.assertRaisesRegex(SystemExit, "validation failed"):
            launcher.validated_python(Path("/bin/false"))

    def test_worker_declares_nonresumable_single_thread_invariants(self):
        text = (
            REPO_ROOT / "src/submit_exact_cg_profile.sub"
        ).read_text()
        self.assertIn("set -euo pipefail", text)
        self.assertIn("#SBATCH --no-requeue", text)
        self.assertNotIn("#SBATCH --requeue", text)
        self.assertIn("#SBATCH --cpus-per-task=1", text)
        self.assertIn("OMP_NUM_THREADS=1", text)
        self.assertIn("OPENBLAS_NUM_THREADS=1", text)
        self.assertIn(launcher.PROFILE_CORE_COMMIT, text)
        self.assertNotIn("--phase-telemetry", text)

    def test_summary_and_monitor_parse_profile_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            root.mkdir()
            output = root / "historical.profile.json"
            output.write_text(json.dumps({
                "schema": "evsp-dr-frozen-pool-prefix-profile-v2",
                "source_unchanged": True,
                "profiles": [{
                    "prefix_columns": 5000,
                    "available": True,
                    "methods": [{
                        "method": "highs-ds",
                        "outcome": "error",
                        "successful_repetitions": 2,
                        "requested_repetitions": 3,
                        "timing": {
                            "total_min_s": 1.0,
                            "total_median_s": 1.2,
                            "total_max_s": 1.4,
                            "backend_median_s": 1.1,
                        },
                        "solution": {
                            "objective": 2900000.0,
                            "route_weight": 29.0,
                            "artificial_total": 0.0,
                            "max_row_violation": 1e-8,
                            "max_bound_violation": 0.0,
                        },
                        "repetitions": [
                            {
                                "outcome": "ok",
                                "peak_rss_bytes": 100,
                            },
                            {
                                "outcome": "error",
                                "error": "timeout",
                                "peak_rss_bytes": 120,
                            },
                            {
                                "outcome": "ok",
                                "peak_rss_bytes": 110,
                            },
                        ],
                    }],
                }],
            }))
            manifest = {
                "campaign": "test",
                "jobs": [{
                    "label": "historical",
                    "job_id": "123",
                    "job_name": "PFhist-abcdef",
                    "submission_state": "submitted",
                    "output": str(output),
                }],
                "checkout_identity": {"expected_commit": "a" * 40},
                "profile_core_commit": launcher.PROFILE_CORE_COMMIT,
            }
            (root / "campaign.json").write_text(json.dumps(manifest))

            rows = summarizer.summarize(root)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["prefix"], 5000)
            self.assertEqual(row["method"], "highs-ds")
            self.assertEqual(row["median_total_s"], 1.2)
            self.assertEqual(row["objective"], 2900000.0)
            self.assertEqual(row["route_weight"], 29.0)
            self.assertEqual(row["artificials"], 0.0)
            self.assertEqual(row["peak_rss_bytes"], 120)
            self.assertEqual(row["failure_count"], 1)
            self.assertIn("timeout", row["failures"])

            monitored = monitor.monitor(root, query_slurm=False)
            self.assertTrue(monitored[0]["output_exists"])
            self.assertTrue(monitored[0]["artifact"]["valid_json"])

    def test_archive_records_commit_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            root.mkdir()
            manifest = {
                "campaign": "archive-test",
                "jobs": [],
                "checkout_identity": {"expected_commit": "b" * 40},
                "profile_core_commit": launcher.PROFILE_CORE_COMMIT,
            }
            (root / "campaign.json").write_text(json.dumps(manifest))
            (root / "result.json").write_text("{}\n")
            output = Path(tmp) / "archive.tar.gz"

            record = archiver.archive(root, output)

            self.assertTrue(output.is_file())
            self.assertEqual(record["expected_commit"], "b" * 40)
            self.assertIn("campaign.json", record["files"])
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(),
                record["archive_sha256"],
            )


if __name__ == "__main__":
    unittest.main()

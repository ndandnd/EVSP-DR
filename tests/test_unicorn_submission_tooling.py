import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cluster_campaign  # noqa: E402
from unicorn_preflight import parse_args  # noqa: E402


class UnicornSubmissionToolingTests(unittest.TestCase):
    def test_matching_is_the_preflight_default(self):
        args = parse_args(["--csv", "Practice_10bus.csv"])
        self.assertEqual(args.mode, "MATCHING")

    def test_preflight_accepts_all_submission_modes(self):
        for mode in ("MATCHING", "GREEDY", "NO_CHEAT", "CHEAT"):
            with self.subTest(mode=mode):
                args = parse_args(["--csv", "Practice_10bus.csv", "--mode", mode])
                self.assertEqual(args.mode, mode)

    def test_cluster_job_maps_matching_mode_to_runner_flag(self):
        job_text = (REPO_ROOT / "src" / "submit_goal1_colgen.sub").read_text()
        self.assertIn("NO_CHEAT|CHEAT|GREEDY|MATCHING", job_text)
        self.assertIn("MATCHING) command+=(--matching)", job_text)

    def test_cluster_column_generation_defaults_to_free_master_backend(self):
        job_text = (REPO_ROOT / "src" / "submit_goal1_colgen.sub").read_text()
        self.assertIn("EVSP_MASTER_BACKEND:-scipy", job_text)
        self.assertIn('--master_backend "$MASTER_BACKEND"', job_text)
        self.assertIn("preflight_command+=(--skip_gurobi)", job_text)

    def test_exact_jobs_requeue_and_resume_persisted_pools(self):
        for script_name in (
            "submit_exact_pairs.sub",
            "submit_exact_unions.sub",
            "submit_exact_big.sub",
        ):
            with self.subTest(script_name=script_name):
                job_text = (REPO_ROOT / "src" / script_name).read_text()
                self.assertIn("#SBATCH --requeue", job_text)
                self.assertIn("#SBATCH --open-mode=append", job_text)
                self.assertIn("#SBATCH --mail-type=REQUEUE,FAIL", job_text)
                self.assertIn("--resume", job_text)

    def test_exact_pool_mip_uses_scaglione_without_requeue(self):
        script = REPO_ROOT / "src" / "submit_exact_pool_mip.sub"
        job_text = script.read_text()
        self.assertTrue(job_text.startswith("#!/bin/bash\n"))
        self.assertIn("set -euo pipefail", job_text)
        self.assertIn("#SBATCH --partition=scaglione", job_text)
        self.assertIn("#SBATCH --no-requeue", job_text)
        self.assertNotIn("#SBATCH --requeue", job_text)
        self.assertIn("#SBATCH --open-mode=append", job_text)
        self.assertIn("#SBATCH --mail-type=FAIL,TIME_LIMIT", job_text)
        self.assertIn("EVSP_ALLOW_PREEMPTIBLE_MIP", job_text)
        self.assertIn("#SBATCH --job-name=MIP-UNNAMED", job_text)
        self.assertIn("non-semantic MIP job name", job_text)
        self.assertIn("run_exact_pool_mip.py", job_text)
        self.assertIn("--require-singleton-partition", job_text)
        self.assertIn("EXACT_MIP_TWO_STAGE", job_text)
        self.assertIn("--two-stage", job_text)
        self.assertIn("--initial-partition-routes", job_text)
        self.assertNotIn(
            '${4:-${EXACT_MIP_INITIAL_PARTITION:-}}', job_text
        )
        self.assertIn('INITIAL_PARTITION_ARG=${4:-}', job_text)
        self.assertIn("EVSP_MIP_EXPECTED_RESULT_SHA256", job_text)
        self.assertIn("EVSP_MIP_EXPECTED_JOURNAL_SHA256", job_text)
        self.assertIn(
            "EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256", job_text
        )
        self.assertGreaterEqual(job_text.count("verify_campaign_inputs"), 3)
        self.assertNotIn("sbatch --wrap=", job_text)
        subprocess.run(["/bin/bash", "-n", str(script)], check=True)

    def test_validated_cluster_launcher_rejects_placeholder_and_missing_paths(self):
        launcher = REPO_ROOT / "src" / "cluster_campaign.py"
        for result in (
            "/absolute/path/to/pool.snapshot.json",
            "missing-pool.snapshot.json",
        ):
            with self.subTest(result=result):
                completed = subprocess.run(
                    [sys.executable, str(launcher), "mip", "--result", result,
                     "--minutes", "5"],
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("error: argument --result:", completed.stderr)

    def test_validated_cluster_launcher_rejects_live_status_json(self):
        launcher = REPO_ROOT / "src" / "cluster_campaign.py"
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "live.json"
            result.write_text("{}\n")
            completed = subprocess.run(
                [sys.executable, str(launcher), "mip", "--result", str(result),
                 "--minutes", "5"],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("immutable *.snapshot.json", completed.stderr)

    def test_validated_cluster_launcher_is_dry_run_and_scaglione_only(self):
        launcher = REPO_ROOT / "src" / "cluster_campaign.py"
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = folder / "sample.snapshot.json"
            journal = folder / "sample.columns.jsonl"
            result.write_text(json.dumps({
                "csv": "sample.csv",
                "soc_step": 5,
                "trip_ids": [1],
                "columns_journal": str(journal),
            }))
            journal.write_text(json.dumps({"trips": [1], "cost": 100000}) + "\n")
            completed = subprocess.run(
                [sys.executable, str(launcher), "mip", "--result", str(result),
                 "--minutes", "5"],
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("--partition=scaglione", completed.stdout)
            self.assertIn("--no-requeue", completed.stdout)
            self.assertIn("--job-name=MP", completed.stdout)
            self.assertNotIn("EXACTMIP", completed.stdout)
            self.assertNotIn("FULLCOVER", completed.stdout)
            self.assertIn("[dry-run]", completed.stdout)

    def test_submitted_campaign_stages_and_hashes_immutable_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            source = Path(tmp) / "source"
            source.mkdir()
            result = source / "sample.partition_ready.snapshot.json"
            journal = source / "sample.partition_ready.columns.jsonl"
            result.write_text(json.dumps({
                "csv": "sample.csv",
                "soc_step": 5,
                "trip_ids": [1],
                "columns_journal": str(journal),
            }))
            journal.write_text(json.dumps({"trips": [1], "cost": 100000}) + "\n")
            partition = source / "validated_partition.json"
            partition.write_text(json.dumps({
                "routes": [{
                    "route": ["PARX_0", 1, "PARX_0"],
                    "charging_stops": {},
                }],
            }))
            args = argparse.Namespace(
                result=result,
                minutes=5,
                cover=False,
                two_stage=True,
                initial_partition_routes=partition,
                campaign="safe_campaign",
                submit=True,
            )
            completed = subprocess.CompletedProcess(
                args=["sbatch"], returncode=0, stdout="12345\n", stderr=""
            )
            with (
                mock.patch.object(cluster_campaign, "REPO_ROOT", repo),
                mock.patch.object(
                    cluster_campaign,
                    "MIP_WORKER",
                    repo / "src" / "submit_exact_pool_mip.sub",
                ),
                mock.patch.object(
                    cluster_campaign,
                    "MIP_RUNNER",
                    repo / "src" / "run_exact_pool_mip.py",
                ),
                mock.patch.object(
                    cluster_campaign, "_run_checked"
                ) as run_checked,
                mock.patch.object(cluster_campaign.subprocess, "run", return_value=completed),
            ):
                self.assertEqual(cluster_campaign.submit_mip(args), 0)

            campaign = repo / "src" / "results" / "cluster_campaigns" / "safe_campaign"
            manifest = json.loads((campaign / "submission.json").read_text())
            staged_result = Path(manifest["input_result"])
            staged_journal = Path(manifest["input_journal"])
            staged_partition = Path(manifest["input_initial_partition"])
            self.assertTrue(staged_result.is_file())
            self.assertTrue(staged_journal.is_file())
            self.assertTrue(staged_partition.is_file())
            self.assertEqual(manifest["job_id"], "12345")
            self.assertTrue(manifest["job_name"].startswith("MP"))
            self.assertTrue(manifest["submitted"])
            self.assertTrue(manifest["two_stage"])
            self.assertTrue(any(
                "EXACT_MIP_TWO_STAGE=1" in argument
                for argument in manifest["command"]
            ))
            export_argument = next(
                argument for argument in manifest["command"]
                if argument.startswith("--export=")
            )
            self.assertIn(
                "EXACT_MIP_INITIAL_PARTITION=,", export_argument
            )
            self.assertIn(
                "EVSP_MIP_EXPECTED_RESULT_SHA256="
                + manifest["input_result_sha256"],
                export_argument,
            )
            self.assertIn(
                "EVSP_MIP_EXPECTED_JOURNAL_SHA256="
                + manifest["input_journal_sha256"],
                export_argument,
            )
            self.assertIn(
                "EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256="
                + manifest["input_initial_partition_sha256"],
                export_argument,
            )
            self.assertEqual(
                manifest["command"][-1], str(staged_partition)
            )
            self.assertEqual(
                manifest["source_journal_sha256"],
                manifest["input_journal_sha256"],
            )
            self.assertEqual(
                manifest["initial_partition_source_sha256"],
                manifest["input_initial_partition_sha256"],
            )
            self.assertEqual(
                json.loads(staged_result.read_text())["columns_journal"],
                str(staged_journal),
            )
            validation_commands = [
                call.args[0] for call in run_checked.call_args_list
            ]
            self.assertEqual(len(validation_commands), 2)
            for validation_command in validation_commands:
                self.assertIn(
                    "--initial-partition-routes", validation_command
                )
                self.assertNotIn(
                    "--require-singleton-partition", validation_command
                )
            self.assertIn(str(partition), validation_commands[0])
            self.assertIn(str(staged_partition), validation_commands[1])

    def test_cluster_campaign_refuses_existing_or_dot_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "sample.snapshot.json"
            journal = Path(tmp) / "sample.columns.jsonl"
            result.write_text(json.dumps({
                "csv": "sample.csv",
                "soc_step": 5,
                "trip_ids": [1],
                "columns_journal": str(journal),
            }))
            journal.write_text(json.dumps({"trips": [1], "cost": 100000}) + "\n")
            args = argparse.Namespace(
                result=result,
                minutes=5,
                cover=False,
                campaign=".",
                submit=False,
            )
            with self.assertRaises(SystemExit):
                cluster_campaign.submit_mip(args)

            repo = Path(tmp) / "repo"
            existing = (
                repo / "src" / "results" / "cluster_campaigns" / "already"
            )
            existing.mkdir(parents=True)
            args.campaign = "already"
            with mock.patch.object(cluster_campaign, "REPO_ROOT", repo):
                with self.assertRaises(SystemExit):
                    cluster_campaign.submit_mip(args)

    def test_cluster_job_exposes_controlled_heap_and_nested_instances(self):
        job_text = (REPO_ROOT / "src" / "submit_goal1_colgen.sub").read_text()
        matrix_text = (REPO_ROOT / "src" / "submit_goal1_matrix.sh").read_text()
        self.assertIn("EVSP_QUEUE_ORDER:-reduced_cost_bound", job_text)
        self.assertIn("reduced_cost_bound|start_fair_bound", job_text)
        self.assertIn('--queue_order "$QUEUE_ORDER"', job_text)
        self.assertIn("EVSP_PRICING_OUTPUT_SELECTION:-reduced_cost", job_text)
        self.assertIn(
            '--pricing_output_selection "$PRICING_OUTPUT_SELECTION"',
            job_text,
        )
        self.assertIn("EVSP_DOMINANCE_MODE:-resource", job_text)
        self.assertIn('--dominance_mode "$DOMINANCE_MODE"', job_text)
        self.assertIn("EVSP_MAX_TRIP2TRIP:-57", job_text)
        self.assertIn('--max_trip2trip "$MAX_TRIP2TRIP"', job_text)
        self.assertIn('--max_charge2trip "$MAX_CHARGE2TRIP"', job_text)
        self.assertNotIn("INSTANCE must be a filename under data/", job_text)
        self.assertIn('instance_tag=${instance_tag//\\//_}', matrix_text)

    def test_matrix_defaults_to_matching_with_greedy_control(self):
        matrix_text = (REPO_ROOT / "src" / "submit_goal1_matrix.sh").read_text()
        self.assertIn("EVSP_MODES:-MATCHING}", matrix_text)
        self.assertIn("EVSP_MODES:-MATCHING,GREEDY}", matrix_text)
        self.assertIn("NO_CHEAT|CHEAT|GREEDY|MATCHING", matrix_text)

    def test_matrix_has_short_data_collection_profiles(self):
        matrix_text = (REPO_ROOT / "src" / "submit_goal1_matrix.sh").read_text()
        self.assertIn("    5m)", matrix_text)
        self.assertIn("ACTIVE_HOURS=0.0833333333333", matrix_text)
        self.assertIn("    30m)", matrix_text)
        self.assertIn("ACTIVE_HOURS=0.5", matrix_text)

    def test_portfolio_launcher_submits_three_complementary_policies(self):
        portfolio_text = (
            REPO_ROOT / "src" / "submit_goal1_portfolio_matrix.sh"
        ).read_text()
        self.assertIn(
            "submit_component bound_resource reduced_cost_bound resource",
            portfolio_text,
        )
        self.assertIn(
            "submit_component fair_resource start_fair_bound resource",
            portfolio_text,
        )
        self.assertIn(
            "submit_component fair_incidence start_fair_bound incidence_diverse",
            portfolio_text,
        )
        self.assertIn("EVSP_PRICING_OUTPUT_SELECTION=diversified", portfolio_text)
        self.assertIn("EVSP_MODES=GREEDY", portfolio_text)


if __name__ == "__main__":
    unittest.main()

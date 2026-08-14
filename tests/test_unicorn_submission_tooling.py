import argparse
import contextlib
import io
import json
import os
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

    def test_mip_job_names_distinguish_all_four_controlled_arms(self):
        status = {
            "csv": "duty_unions_big/Practice_Custom_DutyUnion_k30_r2.csv",
            "g_kwh": 300.0,
            "min_soc_frac": 0.0,
        }
        names = {}
        for two_stage, validated_start, arm in (
            (False, False, "A"),
            (True, False, "B"),
            (False, True, "C"),
            (True, True, "D"),
        ):
            name = cluster_campaign._mip_job_name(
                status,
                "partition",
                30,
                two_stage=two_stage,
                validated_start=validated_start,
            )
            names[arm] = name
            self.assertLessEqual(len(name), 15)
            self.assertTrue(name.startswith(f"MP{arm}"), name)
        self.assertEqual(len(set(names.values())), 4)

    def test_reviewed_checkout_must_be_detached_and_tracked_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=repo, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo,
                check=True,
            )
            tracked = repo / "solver.py"
            tracked.write_text("reviewed = True\n")
            subprocess.run(["git", "add", "solver.py"], cwd=repo, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "reviewed"],
                cwd=repo,
                check=True,
            )

            with self.assertRaisesRegex(SystemExit, "detached checkout"):
                cluster_campaign._reviewed_checkout_identity(repo)

            subprocess.run(
                ["git", "checkout", "-q", "--detach"], cwd=repo, check=True
            )
            identity = cluster_campaign._reviewed_checkout_identity(repo)
            self.assertTrue(identity["detached"])
            self.assertTrue(identity["tracked_clean"])

            tracked.write_text("reviewed = False\n")
            with self.assertRaisesRegex(SystemExit, "tracked modifications"):
                cluster_campaign._reviewed_checkout_identity(repo)

    def test_local_preflight_drops_slurm_and_submission_identity(self):
        completed = subprocess.CompletedProcess(
            args=["python"], returncode=0
        )
        with (
            mock.patch.dict(os.environ, {
                "SLURM_JOB_ID": "interactive-allocation",
                "EVSP_EXPECTED_COMMIT": "f" * 40,
                "EVSP_REQUIRE_DETACHED": "1",
            }, clear=False),
            mock.patch.object(
                cluster_campaign.subprocess,
                "run",
                return_value=completed,
            ) as run,
        ):
            cluster_campaign._run_checked(["python", "--version"])

        environment = run.call_args.kwargs["env"]
        self.assertNotIn("SLURM_JOB_ID", environment)
        self.assertNotIn("EVSP_EXPECTED_COMMIT", environment)
        self.assertNotIn("EVSP_REQUIRE_DETACHED", environment)

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

    def test_k40_factorial_has_four_named_isolated_resume_arms(self):
        launcher = REPO_ROOT / "src" / "launch_k40_factorial.sh"
        worker = REPO_ROOT / "src" / "submit_k40_factorial.sub"
        prep = REPO_ROOT / "src" / "submit_k40_factorial_prep.sub"
        monitor = REPO_ROOT / "src" / "monitor_k40_factorial.sh"

        launch_text = launcher.read_text()
        worker_text = worker.read_text()
        prep_text = prep.read_text()

        for name, sense, initial in (
            ("K40-CA24", "cover", "artificial"),
            ("K40-CS24", "cover", "singletons"),
            ("K40-PA24", "partition", "artificial"),
            ("K40-PS24", "partition", "singletons"),
        ):
            self.assertIn(
                f"submit_arm {name} {sense} {initial}", launch_text
            )
        self.assertNotIn("--export=ALL", launch_text)
        self.assertIn("--dependency=\"afterok:$PREP_JOB\"", launch_text)
        self.assertIn("--initial-pool \"$INITIAL_POOL\"", worker_text)
        self.assertIn("--master-sense \"$MASTER_SENSE\"", worker_text)
        self.assertIn("--wall-limit-s 90000", worker_text)
        self.assertIn(
            "--snapshot-at-minutes 60,180,360,720,1320,1440", worker_text
        )
        self.assertIn("--resume", worker_text)
        self.assertIn("#SBATCH --requeue", worker_text)
        self.assertIn(".allocations.tsv", worker_text)
        self.assertIn("historical 22-hour comparison", monitor.read_text())
        self.assertIn("primary 24-hour snapshots", monitor.read_text())
        self.assertIn("--union-sizes 15,20,30,40", prep_text)
        self.assertIn("--per-size 6", prep_text)
        self.assertIn("--seed 20260803", prep_text)
        self.assertIn(
            "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd",
            prep_text,
        )
        for script in (launcher, worker, prep, monitor):
            subprocess.run(["/bin/bash", "-n", str(script)], check=True)

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
        self.assertIn("EVSP_EXPECTED_COMMIT", job_text)
        self.assertIn("symbolic-ref -q HEAD", job_text)
        self.assertIn("status --porcelain --untracked-files=no", job_text)
        self.assertIn('--mipgap "$MIP_GAP"', job_text)
        self.assertNotIn("EXACT_MIP_GAP:-", job_text)
        self.assertNotIn(
            '${4:-${EXACT_MIP_INITIAL_PARTITION:-}}', job_text
        )
        self.assertIn('MIP_GAP=${4:-0.0001}', job_text)
        self.assertIn('INITIAL_PARTITION_ARG=${5:-}', job_text)
        self.assertIn("EVSP_MIP_EXPECTED_RESULT_SHA256", job_text)
        self.assertIn("EVSP_MIP_EXPECTED_JOURNAL_SHA256", job_text)
        self.assertIn("EVSP_MIP_EXPECTED_WORKER_SHA256", job_text)
        self.assertIn("EVSP_MIP_EXPECTED_RUNNER_SHA256", job_text)
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

    def test_mip_gap_has_fixed_default_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "sample.snapshot.json"
            result.write_text("{}\n")
            args = cluster_campaign.parse_args([
                "mip", "--result", str(result), "--minutes", "30",
            ])
            self.assertEqual(
                args.mip_gap, cluster_campaign.DEFAULT_MIP_GAP
            )
            with self.assertRaises(SystemExit):
                cluster_campaign.parse_args([
                    "mip", "--result", str(result), "--minutes", "30",
                    "--mip-gap", "nan",
                ])

    def test_validated_cluster_launcher_is_dry_run_and_scaglione_only(self):
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
            args = argparse.Namespace(
                result=result,
                minutes=5,
                mip_gap=0.0001,
                cover=False,
                two_stage=False,
                initial_partition_routes=None,
                campaign="dry_run",
                submit=False,
            )
            identity = {
                "expected_commit": "a" * 40,
                "observed_commit": "a" * 40,
                "detached": True,
                "tracked_clean": True,
            }
            output = io.StringIO()
            with (
                mock.patch.object(
                    cluster_campaign,
                    "_reviewed_checkout_identity",
                    return_value=identity,
                ),
                mock.patch.object(
                    cluster_campaign,
                    "_reviewed_git_blob",
                    side_effect=lambda _root, _commit, relative: (
                        REPO_ROOT / relative
                    ).read_bytes(),
                ),
                mock.patch.object(cluster_campaign, "_run_checked"),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(cluster_campaign.submit_mip(args), 0)
            rendered = output.getvalue()
            self.assertIn("--partition=scaglione", rendered)
            self.assertIn("--no-requeue", rendered)
            self.assertIn("--job-name=MPA", rendered)
            self.assertIn("EVSP_EXPECTED_COMMIT=" + "a" * 40, rendered)
            self.assertIn("0.0001", rendered)
            self.assertIn("[dry-run]", rendered)

    def test_submitted_campaign_stages_and_hashes_immutable_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            (repo / "src").mkdir()
            (repo / "src" / "submit_exact_pool_mip.sub").write_bytes(
                b"#!/bin/bash\n"
            )
            (repo / "src" / "run_exact_pool_mip.py").write_bytes(
                b"#!/bin/bash\n"
            )
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
                mip_gap=0.0125,
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
                mock.patch.object(
                    cluster_campaign,
                    "_reviewed_checkout_identity",
                    return_value={
                        "expected_commit": "a" * 40,
                        "observed_commit": "a" * 40,
                        "detached": True,
                        "tracked_clean": True,
                    },
                ),
                mock.patch.object(
                    cluster_campaign,
                    "_reviewed_git_blob",
                    return_value=b"#!/bin/bash\n",
                ),
                mock.patch.object(cluster_campaign.subprocess, "run", return_value=completed),
            ):
                self.assertEqual(cluster_campaign.submit_mip(args), 0)

            campaign = repo / "src" / "results" / "cluster_campaigns" / "safe_campaign"
            manifest = json.loads((campaign / "submission.json").read_text())
            staged_result = Path(manifest["input_result"])
            staged_journal = Path(manifest["input_journal"])
            staged_partition = Path(manifest["input_initial_partition"])
            staged_worker = Path(manifest["input_worker"])
            self.assertTrue(staged_result.is_file())
            self.assertTrue(staged_journal.is_file())
            self.assertTrue(staged_partition.is_file())
            self.assertTrue(staged_worker.is_file())
            self.assertEqual(
                manifest["reviewed_worker_sha256"],
                manifest["input_worker_sha256"],
            )
            self.assertIn(str(staged_worker), manifest["command"])
            self.assertEqual(manifest["job_id"], "12345")
            self.assertEqual(manifest["submission_state"], "submitted")
            self.assertTrue(manifest["job_name"].startswith("MP"))
            self.assertTrue(manifest["submitted"])
            self.assertTrue(manifest["two_stage"])
            self.assertEqual(manifest["experiment_arm"], "D")
            self.assertEqual(manifest["requested_mip_gap"], 0.0125)
            self.assertEqual(manifest["expected_git_commit"], "a" * 40)
            self.assertEqual(
                manifest["launcher_observed_git_commit"], "a" * 40
            )
            self.assertEqual(
                manifest["pre_submission_observed_git_commit"], "a" * 40
            )
            self.assertTrue(any(
                "EXACT_MIP_TWO_STAGE=1" in argument
                for argument in manifest["command"]
            ))
            export_argument = next(
                argument for argument in manifest["command"]
                if argument.startswith("--export=")
            )
            self.assertFalse(export_argument.startswith("--export=ALL"))
            self.assertNotIn(",ALL,", export_argument)
            self.assertNotIn("EXACT_MIP_INITIAL_PARTITION", export_argument)
            self.assertNotIn("EXACT_MIP_GAP", export_argument)
            self.assertIn(
                "EVSP_EXPECTED_COMMIT=" + "a" * 40, export_argument
            )
            self.assertIn("EVSP_REQUIRE_DETACHED=1", export_argument)
            self.assertIn(
                "EVSP_MIP_EXPECTED_WORKER_SHA256="
                + manifest["reviewed_worker_sha256"],
                export_argument,
            )
            self.assertIn(
                "EVSP_MIP_EXPECTED_RUNNER_SHA256="
                + manifest["reviewed_runner_sha256"],
                export_argument,
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
            self.assertEqual(manifest["command"][-2], "0.012500000000000001")
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

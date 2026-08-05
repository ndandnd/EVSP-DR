import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

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

    def test_exact_pool_mip_uses_a_real_bash_worker(self):
        script = REPO_ROOT / "src" / "submit_exact_pool_mip.sub"
        job_text = script.read_text()
        self.assertTrue(job_text.startswith("#!/bin/bash\n"))
        self.assertIn("set -euo pipefail", job_text)
        self.assertIn("#SBATCH --requeue", job_text)
        self.assertIn("#SBATCH --open-mode=append", job_text)
        self.assertIn("#SBATCH --mail-type=REQUEUE,FAIL", job_text)
        self.assertIn("run_exact_pool_mip.py", job_text)
        self.assertIn("--require-singleton-partition", job_text)
        self.assertNotIn("sbatch --wrap=", job_text)
        subprocess.run(["/bin/bash", "-n", str(script)], check=True)

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

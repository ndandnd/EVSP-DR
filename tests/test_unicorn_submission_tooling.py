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

    def test_cluster_job_exposes_controlled_heap_and_nested_instances(self):
        job_text = (REPO_ROOT / "src" / "submit_goal1_colgen.sub").read_text()
        matrix_text = (REPO_ROOT / "src" / "submit_goal1_matrix.sh").read_text()
        self.assertIn("EVSP_QUEUE_ORDER:-reduced_cost_bound", job_text)
        self.assertIn('--queue_order "$QUEUE_ORDER"', job_text)
        self.assertIn("EVSP_PRICING_OUTPUT_SELECTION:-reduced_cost", job_text)
        self.assertIn(
            '--pricing_output_selection "$PRICING_OUTPUT_SELECTION"',
            job_text,
        )
        self.assertIn('--max_charge2trip "$MAX_CHARGE2TRIP"', job_text)
        self.assertNotIn("INSTANCE must be a filename under data/", job_text)
        self.assertIn('instance_tag=${instance_tag//\\//_}', matrix_text)

    def test_matrix_defaults_to_matching_with_greedy_control(self):
        matrix_text = (REPO_ROOT / "src" / "submit_goal1_matrix.sh").read_text()
        self.assertIn("EVSP_MODES:-MATCHING}", matrix_text)
        self.assertIn("EVSP_MODES:-MATCHING,GREEDY}", matrix_text)
        self.assertIn("NO_CHEAT|CHEAT|GREEDY|MATCHING", matrix_text)


if __name__ == "__main__":
    unittest.main()

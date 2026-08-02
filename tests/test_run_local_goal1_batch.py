import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_local_goal1_batch import (  # noqa: E402
    BenchmarkCase,
    build_command,
    parse_args,
)


class LocalGoal1LauncherTests(unittest.TestCase):
    def make_case(self):
        return BenchmarkCase(
            name="tiny",
            csv_path=REPO_ROOT / "data" / "Practice_10bus.csv",
            csv_sha256="abc",
            bus_count=10,
            replicate=1,
            seed=7,
        )

    def test_matching_is_the_default_initializer(self):
        args = parse_args(["5m", "--manifest", "manifest.json"])
        self.assertEqual(args.initializer, "matching")
        self.assertEqual(args.dominance_mode, "resource")

    def test_command_uses_selected_initializer_exclusively(self):
        common = dict(
            case=self.make_case(),
            python=Path(sys.executable),
            profile_name="5m",
            results_root=REPO_ROOT / "src" / "results" / "test",
            batch_tag="test",
            queue_order="reduced_cost_bound",
            pricing_output_selection="reduced_cost",
            dominance_mode="resource",
            max_charge2trip=1560,
        )

        matching = build_command(**common, initializer="matching")
        greedy = build_command(**common, initializer="greedy")

        self.assertIn("--matching", matching)
        self.assertNotIn("--greedy", matching)
        self.assertIn("--greedy", greedy)
        self.assertNotIn("--matching", greedy)

    def test_command_threads_diversified_output_selection(self):
        command = build_command(
            case=self.make_case(),
            python=Path(sys.executable),
            profile_name="5m",
            results_root=REPO_ROOT / "src" / "results" / "test",
            batch_tag="test",
            initializer="greedy",
            queue_order="reduced_cost_bound",
            pricing_output_selection="diversified",
            dominance_mode="resource",
            max_charge2trip=1560,
        )

        flag_index = command.index("--pricing_output_selection")
        self.assertEqual(command[flag_index + 1], "diversified")

    def test_command_threads_incidence_diverse_dominance(self):
        command = build_command(
            case=self.make_case(),
            python=Path(sys.executable),
            profile_name="5m",
            results_root=REPO_ROOT / "src" / "results" / "test",
            batch_tag="test",
            initializer="matching",
            queue_order="reduced_cost_bound",
            pricing_output_selection="reduced_cost",
            dominance_mode="incidence_diverse",
            max_charge2trip=1560,
        )

        flag_index = command.index("--dominance_mode")
        self.assertEqual(command[flag_index + 1], "incidence_diverse")


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_local_goal1_batch as launcher  # noqa: E402
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

    def test_local_launcher_accepts_start_fair_bound_queue(self):
        args = parse_args([
            "5m",
            "--manifest",
            "manifest.json",
            "--queue-order",
            "start_fair_bound",
        ])
        self.assertEqual(args.queue_order, "start_fair_bound")

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
        gap_index = matching.index("--max_trip2trip")
        self.assertEqual(matching[gap_index + 1], "57")

    def test_command_threads_relaxed_trip_gap(self):
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
            max_trip2trip=180,
        )

        flag_index = command.index("--max_trip2trip")
        self.assertEqual(command[flag_index + 1], "180")

    def test_preflight_requires_trip_gap_interface(self):
        advertised_without_trip_gap = " ".join(
            (
                "--master_backend",
                "--matching",
                launcher.RUNNER_QUEUE_FLAG,
                launcher.RUNNER_OUTPUT_SELECTION_FLAG,
                launcher.RUNNER_DOMINANCE_FLAG,
                launcher.RUNNER_GAP_FLAG,
                "reduced_cost_bound",
                "diversified",
                "resource",
            )
        )
        completed = SimpleNamespace(
            returncode=0,
            stdout=advertised_without_trip_gap,
        )

        with patch.object(launcher.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "--max_trip2trip"):
                launcher._runner_preflight(
                    Path(sys.executable),
                    "reduced_cost_bound",
                    "diversified",
                    "resource",
                )

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

    def test_command_threads_start_fair_bound_queue(self):
        command = build_command(
            case=self.make_case(),
            python=Path(sys.executable),
            profile_name="5m",
            results_root=REPO_ROOT / "src" / "results" / "test",
            batch_tag="test",
            initializer="matching",
            queue_order="start_fair_bound",
            pricing_output_selection="reduced_cost",
            dominance_mode="incidence_diverse",
            max_charge2trip=1560,
        )

        flag_index = command.index("--queue_order")
        self.assertEqual(command[flag_index + 1], "start_fair_bound")


if __name__ == "__main__":
    unittest.main()

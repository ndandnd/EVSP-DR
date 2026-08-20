import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from target_pool_feasibility import (  # noqa: E402
    classify_outcome,
    evaluate,
    solve_target_feasibility,
)


class TargetPoolFeasibilityTests(unittest.TestCase):
    def test_feasible_returns_target_partition_witness(self):
        routes = [
            {"trips": [0]},
            {"trips": [1]},
            {"trips": [0, 1]},
        ]
        result = solve_target_feasibility(
            routes, [0, 1], 1, timelimit=30, threads=1,
        )
        self.assertEqual(result["outcome"], "FEASIBLE")
        selected = result["selected_indices"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(routes[selected[0]]["trips"], [0, 1])
        self.assertEqual(
            result["parameters"]["objective"],
            "constant_zero_pure_feasibility",
        )

    def test_infeasible_is_proved_for_too_small_target(self):
        result = solve_target_feasibility(
            [{"trips": [0]}, {"trips": [1]}],
            [0, 1],
            1,
            timelimit=30,
            threads=1,
        )
        self.assertEqual(result["outcome"], "INFEASIBLE")
        self.assertEqual(result["selected_indices"], [])
        self.assertEqual(result["solution_count"], 0)

    def test_time_limit_without_incumbent_is_censored(self):
        GRB = SimpleNamespace(
            INFEASIBLE=3,
            TIME_LIMIT=9,
        )
        self.assertEqual(classify_outcome(9, 0, GRB), "TIME_LIMIT")
        self.assertEqual(classify_outcome(9, 1, GRB), "FEASIBLE")
        self.assertEqual(classify_outcome(3, 0, GRB), "INFEASIBLE")

    def test_unexpected_no_solution_status_is_not_misclassified(self):
        GRB = SimpleNamespace(INFEASIBLE=3, TIME_LIMIT=9)
        with self.assertRaisesRegex(RuntimeError, "without a classified"):
            classify_outcome(12, 0, GRB)

    def test_pool_omission_and_invalid_target_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "omits trips"):
            solve_target_feasibility(
                [{"trips": [0]}], [0, 1], 1,
                timelimit=30, threads=1,
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            solve_target_feasibility(
                [{"trips": [0]}], [0], 0,
                timelimit=30, threads=1,
            )

    def test_evaluate_rejects_source_swap_before_physical_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "pool.jsonl"
            journal.write_text('{"trips":[0],"cost":1}\n')
            status = {"columns_journal": str(journal), "trip_ids": [0]}
            result = root / "status.json"
            result.write_text(json.dumps(status))
            args = SimpleNamespace(
                result=result, out=root/"out.json", target=1,
                timelimit=30, threads=1, seed=0,
                data_dir=None, reference_data_dir=None,
            )

            def swapped(*_args, **_kwargs):
                journal.write_text('{"trips":[0],"cost":2}\n')
                return status, [{"trips":[0],"cost":2}], [0]

            with (
                patch(
                    "target_pool_feasibility.resolve_pool_journal",
                    return_value=journal,
                ),
                patch(
                    "target_pool_feasibility.load_pool",
                    side_effect=swapped,
                ),
                patch(
                    "target_pool_feasibility.verified_mip_code_identity",
                    return_value={"observed_commit":"a"*40},
                ),
                self.assertRaisesRegex(RuntimeError, "changed while loading"),
            ):
                evaluate(args)
            self.assertFalse((root/"out.json").exists())

    def test_evaluate_refuses_dangling_output_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/"out.json"
            os.symlink(root/"missing",output)
            args=SimpleNamespace(result=root/"unused",out=output)
            with self.assertRaises(FileExistsError):
                evaluate(args)


if __name__ == "__main__":
    unittest.main()

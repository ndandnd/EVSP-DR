#!/usr/bin/env python3
import copy
import json
import tempfile
import unittest
from pathlib import Path

import campaign
import collect as collector


class FixedDualTests(unittest.TestCase):
    def case(self):
        return {
            "case_id": "case", "pair_id": "pair", "capacity_selector": "reference",
            "starting_dual_vector_sha256": "d" * 64,
            "starting_raw_dual_vector_sha256": "r" * 64,
            "starting_normalized_9dp_dual_vector_sha256": "d" * 64,
            "source_pool_sha256": "p" * 64, "checkpoint_id": "c" * 64,
            "max_iters": 16,
        }

    def start(self):
        return {
            "dual_vector_sha256": "d" * 64, "source_pool_sha256": "p" * 64,
            "raw_dual_vector_sha256": "r" * 64,
            "normalized_9dp_dual_vector_sha256": "d" * 64,
            "next_iteration": 16,
        }

    def test_complete_and_censored_contracts(self):
        complete_call = {
            "status": "complete", "attempted_calls": 1, "completed_calls": 1,
            "min_reduced_cost": -2.5, "elapsed_s": 10,
            "candidate_route_key_sha256": "r" * 64,
            "candidate_payload_sha256": "q" * 64,
            "raw_dual_vector_sha256": "r" * 64,
            "normalized_9dp_dual_vector_sha256": "d" * 64,
        }
        complete_result = {
            "iterations": [{"iteration": 16, "min_reduced_cost": -2.5}],
            "status": "incomplete", "stop_reason": "max_iters",
            "certified_rc_optimal": False, "terminal_exact_min_reduced_cost": None,
            "runtime_s": 11, "network_build_s": 1,
            "final": {"pool_columns": 20}, "pool_sha256": "x" * 64,
        }
        value = campaign.validate_endpoint(
            self.case(), self.start(), complete_call, complete_result,
        )
        self.assertEqual(value["pricing_call"]["completed_calls"], 1)

        censored_call = {
            "status": "censored", "attempted_calls": 1, "completed_calls": 0,
            "min_reduced_cost": None, "elapsed_s": 100,
            "raw_dual_vector_sha256": "r" * 64,
            "normalized_9dp_dual_vector_sha256": "d" * 64,
        }
        censored_result = copy.deepcopy(complete_result)
        censored_result.update(
            iterations=[], stop_reason="pricing_deadline",
            terminal_exact_min_reduced_cost=None, certified_rc_optimal=False,
        )
        campaign.validate_endpoint(
            self.case(), self.start(), censored_call, censored_result,
        )
        censored_result["certified_rc_optimal"] = True
        with self.assertRaises(ValueError):
            campaign.validate_endpoint(
                self.case(), self.start(), censored_call, censored_result,
            )

    def test_atomic_copy_rejects_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_text("fixed\n")
            expected = campaign.sha256_file(source)
            destination = root / "copy"
            self.assertEqual(campaign.atomic_copy(source, destination, expected), expected)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            with self.assertRaises(ValueError):
                campaign.atomic_copy(source, root / "bad", "0" * 64)

    def test_incomplete_pair_is_not_vacuously_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "tooling_sha256": {},
                "cases": {
                    "a": {"pair_id": "p", "capacity_selector": "reference"},
                    "b": {"pair_id": "p", "capacity_selector": "prefix-memo"},
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            state = root / "results/a/123_r0/worker_status.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"status": "failed", "error": "preempted"}))
            value = collector.collect(root)
            self.assertIsNone(value["pair_checks"]["p"]["identical_starting_dual_hash"])
            self.assertEqual(
                value["workflow"]["attempt_progress"][0]["attempts"][0]["status"],
                "failed",
            )


if __name__ == "__main__":
    unittest.main()

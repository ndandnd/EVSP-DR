import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get(
    "CAPACITY_CODE_ROOT",
    HERE.parents[2] / ".codex-work/capacity-parallel-followup-20260914",
)).resolve()
SPEC = importlib.util.spec_from_file_location("boundary_campaign", HERE / "campaign.py")
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


class BoundaryCampaignTests(unittest.TestCase):
    def setUp(self):
        self.manifest = CAMPAIGN.load_manifest()

    def test_exact_matched_matrix_and_budgets(self):
        cases = self.manifest["cases"]
        self.assertEqual(len(cases), 8)
        self.assertEqual([c["index"] for c in cases], list(range(8)))
        for duty in range(13405, 13409):
            pair = [c for c in cases if c["instance"] == f"k1_duty{duty}"]
            self.assertEqual({c["capacity_selector"] for c in pair},
                             {"reference", "prefix-memo"})
            self.assertEqual(len(pair), 2)
        for case in cases:
            self.assertEqual(case["arm"], "capacity")
            self.assertEqual(case["prices"], "flat")
            self.assertEqual((case["battery_kwh"], case["reserve_kwh"]), (240.0, 0.0))
            self.assertEqual((case["cg_wall_s"], case["mip_wall_s"],
                              case["slurm_time_s"]), (13200, 600, 14400))

    def test_resource_and_proof_contract(self):
        policy = self.manifest["resource_policy"]
        self.assertEqual(policy["array_concurrency"], 8)
        self.assertEqual(policy["cpus_per_task"], 1)
        self.assertEqual(policy["memory_gb"], 24)
        self.assertIn("scaglione-compute-01", policy["exclude"])
        self.assertTrue(policy["submission_authorized"])
        reporting = self.manifest["reporting_contract"]
        self.assertIn("never a pricing certificate", reporting["interrupted_pricing"])
        self.assertIn("diagnostic", reporting["integer_proof"])
        self.assertFalse(self.manifest["common_model"]["terminal_65_percent_floor"])

    def test_input_hashes_and_actual_driver_parser(self):
        result = CAMPAIGN.validate_manifest(self.manifest, CODE)
        self.assertTrue(result["valid"], result["errors"])
        parsed = CAMPAIGN.validate_driver_commands(self.manifest, CODE)
        self.assertTrue(parsed["valid"], parsed["failures"])
        self.assertEqual(parsed["parsed_command_count"], 16)

    def test_commands_are_dedicated_and_one_thread(self):
        with tempfile.TemporaryDirectory() as raw:
            for case in self.manifest["cases"]:
                commands = CAMPAIGN.build_commands(
                    self.manifest, CODE, Path(raw) / case["case_id"], case,
                    python="/verified/python",
                )
                for command in commands.values():
                    self.assertIn("src/run_capacity_speed_event_cg.py", command)
                    self.assertIn(CAMPAIGN.DRIVER_COMMIT, command)
                    self.assertIn("--threads", command)
                    self.assertEqual(command[command.index("--threads") + 1], "1")
                self.assertNotIn("--cg-wall-s", commands["mip"])

    def test_strict_json(self):
        self.assertEqual(json.loads(json.dumps(self.manifest, allow_nan=False))["schema"],
                         "evsp-dr-capacity-pricing-boundary-v1")


if __name__ == "__main__":
    unittest.main()

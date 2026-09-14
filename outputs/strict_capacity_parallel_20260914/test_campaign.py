import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("CAPACITY_CODE_ROOT", HERE.parents[2])).resolve()
SPEC = importlib.util.spec_from_file_location("capacity_campaign", HERE / "campaign.py")
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


class CapacityCampaignTests(unittest.TestCase):
    def setUp(self):
        self.manifest = CAMPAIGN.load_manifest()

    def test_frozen_inputs_and_campaign_invariants(self):
        result = CAMPAIGN.validate_manifest(self.manifest, REPO)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["case_count"], 12)
        self.assertTrue(all(row["valid"] for row in result["input_checks"].values()))

    def test_plan_is_independent_bounded_and_non_submitting(self):
        policy = self.manifest["resource_policy"]
        self.assertEqual(policy["array_concurrency"], len(self.manifest["cases"]))
        self.assertFalse(policy["submission_authorized"])
        self.assertIn("scaglione-compute-01", policy["exclude"])
        source = (HERE / "campaign.py").read_text()
        self.assertNotIn("sbatch", source)
        self.assertNotIn("scontrol", source)
        for case in self.manifest["cases"]:
            self.assertGreaterEqual(case["slurm_time_s"], 3600)
            self.assertLessEqual(case["slurm_time_s"], 14400)
            self.assertGreaterEqual(
                case["slurm_time_s"] - case["cg_wall_s"] - case["mip_wall_s"],
                300,
            )

    def test_commands_use_only_dedicated_capacity_driver(self):
        with tempfile.TemporaryDirectory() as raw:
            for case in self.manifest["cases"]:
                commands = CAMPAIGN.build_commands(
                    self.manifest, REPO, Path(raw) / case["case_id"], case,
                    python="/verified/python",
                )
                for stage in ("cg", "mip"):
                    command = commands[stage]
                    self.assertIn("src/run_capacity_speed_event_cg.py", command)
                    self.assertNotIn("src/run_exact_pool_mip.py", command)
                    self.assertIn(CAMPAIGN.DRIVER_COMMIT, command)
                    self.assertIn(str(case["battery_kwh"]), command)
                    self.assertIn(str(case["reserve_kwh"]), command)
                self.assertIn(case["capacity_selector"], commands["cg"])

    def test_all_24_commands_parse_with_actual_driver(self):
        result = CAMPAIGN.validate_driver_commands(self.manifest, REPO)
        self.assertTrue(result["valid"], result["failures"])
        self.assertEqual(result["parsed_command_count"], 24)

    def test_clean_execution_environment_is_bounded(self):
        with mock.patch.dict(
            CAMPAIGN.os.environ,
            {"PYTHONPATH": "bad", "LD_LIBRARY_PATH": "bad",
             "LM_LICENSE_FILE": "bad"},
        ):
            environment = CAMPAIGN.execution_environment()
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("LD_LIBRARY_PATH", environment)
        self.assertNotIn("LM_LICENSE_FILE", environment)
        self.assertEqual(environment["GRB_LICENSE_FILE"], CAMPAIGN.GUROBI_LICENSE)
        self.assertEqual(environment["OMP_NUM_THREADS"], "1")

    def test_factorial_and_no_terminal_floor(self):
        cases = self.manifest["cases"]
        k1 = [case for case in cases if case["instance"] == "k1_duty13408"]
        self.assertEqual(
            {(case["prices"], case["capacity_selector"]) for case in k1},
            {
                ("flat", "reference"), ("flat", "prefix-memo"),
                ("peak12", "reference"), ("peak12", "prefix-memo"),
            },
        )
        k2 = [case for case in cases if case["instance"] == "e1_short_k2"]
        self.assertEqual(len(k2), 8)
        self.assertEqual(
            {(case["battery_kwh"], case["reserve_kwh"]) for case in k2},
            {(240.0, 0.0), (236.44, 35.466)},
        )
        for physics in {(case["battery_kwh"], case["reserve_kwh"]) for case in k2}:
            self.assertEqual(
                {case["arm"] for case in k2
                 if (case["battery_kwh"], case["reserve_kwh"]) == physics},
                {"baseline", "capacity", "parx60", "combined"},
            )
        self.assertAlmostEqual(35.466, 0.15 * 236.44)
        self.assertFalse(self.manifest["common_model"]["terminal_65_percent_floor"])

    def test_manifest_is_strict_json(self):
        encoded = json.dumps(self.manifest, allow_nan=False, sort_keys=True)
        self.assertEqual(json.loads(encoded)["schema"], self.manifest["schema"])

    def test_collector_keeps_proof_scopes_separate(self):
        case = self.manifest["cases"][0]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            attempt = root / "results" / case["case_id"] / "job1_r0"
            attempt.mkdir(parents=True)
            (attempt / "pool.jsonl").write_text("{}\n")
            (attempt / "worker_status.json").write_text(json.dumps({
                "returncode": 0,
                "manifest_sha256": "manifest",
                "stages": {
                    "cg": {"returncode": 0},
                    "mip": {"returncode": 0},
                },
            }))
            (attempt / "cg.json").write_text(json.dumps({
                "status": "incomplete",
                "stop_reason": "pricing_deadline",
                "certified_rc_optimal": False,
                "terminal_exact_min_reduced_cost": None,
                "iterations": [{"iteration": 1}],
                "final": {"pool_columns": 2},
                "pool_sha256": "pool",
            }))
            (attempt / "mip.json").write_text(json.dumps({
                "pool_acceptance": {
                    "classification": "timed_uncertified_exact_event_cg_pool",
                },
                "result": {
                    "status": "OPTIMAL", "has_solution": True, "fleet": 1,
                    "stage1": {
                        "fleet_proven": True,
                        "fleet_integer_lower_bound": 1,
                    },
                    "stage2": {"status": "OPTIMAL"},
                },
                "physical_station_capacity_audit": {"valid": True},
            }))
            collected = CAMPAIGN.collect_campaign(self.manifest, root)
        self.assertEqual(
            collected["schema"],
            "evsp-dr-strict-capacity-parallel-collection-v1",
        )
        self.assertEqual(collected["observed_attempt_count"], 1)
        self.assertEqual(len(collected["cg"]), 1)
        self.assertEqual(len(collected["mip"]), 1)
        self.assertEqual(len(collected["records"]), 2)
        self.assertTrue(collected["cg"][0]["provisional"])
        self.assertFalse(collected["mip"][0]["provisional"])
        self.assertNotIn("iterations", collected["cg"][0]["result"])
        self.assertIn("stage1", collected["mip"][0]["result"]["result"])
        self.assertNotIn(
            "selected_indices",
            collected["mip"][0]["result"]["result"]["stage1"],
        )
        self.assertIn("attempt_progress", collected["workflow"])
        row = collected["rows"][0]
        self.assertFalse(row["cg"]["certified_rc_optimal"])
        self.assertIsNone(row["cg"]["terminal_exact_min_reduced_cost"])
        self.assertTrue(row["finite_pool_mip"]["stage1_fleet_proven"])
        self.assertTrue(
            row["finite_pool_mip"]["physical_station_capacity_audit"]["valid"]
        )


if __name__ == "__main__":
    unittest.main()

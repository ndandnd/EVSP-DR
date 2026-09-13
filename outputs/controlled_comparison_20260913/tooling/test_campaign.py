#!/usr/bin/env python3
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("controlled_campaign", HERE / "campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def contract(root):
    return {
        "schema": "evsp-controlled-comparison-contract-v1", "root": str(root),
        "python": sys.executable, "code": "/code", "mip_code": "/mip",
        "cg_commit": "c" * 40, "mip_commit": "m" * 40,
        "common": {"cg_seconds": 7200, "mip_seconds": 3600, "stage1_seconds": 1800,
                   "threads": 8, "soc_step_kwh": 2.5, "block_minutes": 5,
                   "columns_per_iter": 30, "rc_epsilon": 0.0001,
                   "battery_kwh": 240, "charge_kw": 240,
                   "inherit_workers": 8, "inherit_time_limit_s": 0},
        "arms": {
            "A": {"fixed_sequence_index": False, "inherit_max_columns": 512, "skip_gurobi_incidence": False},
            "B": {"fixed_sequence_index": True, "inherit_max_columns": 512, "skip_gurobi_incidence": False},
            "C": {"fixed_sequence_index": True, "inherit_max_columns": 0, "skip_gurobi_incidence": False},
            "D": {"fixed_sequence_index": True, "inherit_max_columns": 0, "skip_gurobi_incidence": True},
            "E": {"fixed_sequence_index": False, "inherit_max_columns": 0, "skip_gurobi_incidence": False}},
        "inputs": {"source": {"csv": "case.csv", "cache": "/cache.pkl",
                                "parent_descriptor": "/parent.json"}},
        "pairs": [{"id": "pair", "case_id": "source", "contrast": "index", "repetition": 1,
                   "order": ["A", "B"], "cpus": 8, "mem": "96G", "slurm_time": "07:00:00"}],
    }


class CampaignTests(unittest.TestCase):
    def test_arm_commands_vary_only_declared_treatments(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest = contract(root); pair = manifest["pairs"][0]
            flags = {arm: campaign.command_flags(campaign.cg_command(manifest, pair, arm, root / arm))
                     for arm in "ABCDE"}
            self.assertNotIn("--fixed-sequence-index", flags["A"])
            self.assertIn("--fixed-sequence-index", flags["B"])
            self.assertEqual(flags["A"]["--inherit-max-columns"], "512")
            self.assertEqual(flags["B"]["--inherit-max-columns"], "512")
            self.assertEqual(flags["C"]["--inherit-max-columns"], "0")
            self.assertNotIn("--skip-gurobi-incidence", flags["C"])
            self.assertIn("--skip-gurobi-incidence", flags["D"])
            self.assertNotIn("--fixed-sequence-index", flags["E"])
            self.assertEqual(flags["E"]["--inherit-max-columns"], "0")
            self.assertNotIn("--skip-gurobi-incidence", flags["E"])
            for arm in "ABCDE":
                self.assertEqual(flags[arm]["--inherit-time-limit-s"], "0")
                self.assertNotIn("--validated-seed-routes", flags[arm])

    def test_mip_command_has_bounded_two_stage_cover_and_fleet_cap_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest = contract(root)
            flags = campaign.command_flags(campaign.mip_command(manifest, root / "arm"))
            self.assertIs(flags["--cover"], True)
            self.assertIs(flags["--two-stage"], True)
            self.assertEqual(flags["--timelimit"], "3600")
            self.assertEqual(flags["--stage1-timelimit"], "1800")
            self.assertEqual(flags["--threads"], "8")
            self.assertEqual(manifest["mip_commit"], "m" * 40)

    def test_pool_gate_and_collector_keep_process_and_certificate_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); journal = root / "cg.json.columns.jsonl"; journal.write_text("{}\n")
            status_path = root / "cg.json"
            status_path.write_text(json.dumps({"columns_journal": str(journal), "certified_rc_optimal": False,
                                               "final": {"iter": 3, "artificials": 0}}))
            gate = campaign.cg_pool_gate({"returncode": 0, "watchdog_triggered": False}, status_path)
            self.assertTrue(gate["allow_mip"])
            self.assertFalse(gate["cg_certified_rc_optimal"])
            status_path.write_text(json.dumps({"columns_journal": str(journal),
                                               "final": {"iter": 3, "artificials": 1}}))
            self.assertFalse(campaign.cg_pool_gate({"returncode": 0}, status_path)["allow_mip"])

            manifest = contract(root); (root / "manifest.json").write_text(json.dumps(manifest))
            attempt = root / "cases/pair/1_r0"; arm = attempt / "A"
            arm.joinpath("cg").mkdir(parents=True); arm.joinpath("mip").mkdir()
            (attempt / "pair_status.json").write_text(json.dumps({"status": "finished", "order": ["A"], "arms": {}}))
            (arm / "cg/execution.json").write_text(json.dumps({"argv": ["solver"], "returncode": 0,
                                                                 "watchdog_triggered": False}))
            (arm / "cg.json").write_text(json.dumps({"certified_rc_optimal": False,
                                                       "final": {"iter": 2, "artificials": 0, "lp_obj": 4}}))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                campaign.collect(type("Args", (), {"root": root})())
            collected = json.loads(output.getvalue())
            row = collected["cg"][0]
            self.assertTrue(row["process"]["process_success"])
            self.assertFalse(row["certified_rc_optimal"])
            self.assertNotIn("final_lp", row)

    def test_process_directory_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); env = campaign.clean_environment()
            first = campaign.run_process([sys.executable, "-c", "print('ok')"], root,
                                         root / "run", 5, env, {"cpus": 1})
            self.assertEqual(first["returncode"], 0)
            with self.assertRaises(FileExistsError):
                campaign.run_process([sys.executable, "-c", "print('again')"], root,
                                     root / "run", 5, env, {"cpus": 1})


if __name__ == "__main__":
    unittest.main()

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cluster_campaign  # noqa: E402
import slurm_campaign  # noqa: E402


class SemanticSlurmNameTests(unittest.TestCase):
    def test_every_exact_campaign_name_is_short_unique_and_semantic(self):
        for campaign, specs in slurm_campaign.CAMPAIGNS.items():
            with self.subTest(campaign=campaign):
                names = [slurm_campaign.job_name(spec) for spec in specs]
                self.assertEqual(len(names), len(set(names)))
                self.assertTrue(all(len(name) <= 15 for name in names))
                self.assertTrue(all(slurm_campaign.JOB_NAME_RE.fullmatch(name)
                                    for name in names))

    def test_failed_peak_tasks_share_the_visible_k15_case(self):
        expected = {
            3: "XP-15r6-f",
            6: "XP-15r6-p08",
            9: "XP-15r6-p12",
            12: "XP-15r6-p18",
        }
        actual = {
            task: slurm_campaign.job_name(
                slurm_campaign.task_spec("peaks", task)
            )
            for task in expected
        }
        self.assertEqual(actual, expected)
        self.assertEqual(
            slurm_campaign.task_spec("peaks", 3).csv,
            "duty_unions_big/Practice_Custom_DutyUnion_k15_r6.csv",
        )

    def test_mip_names_expose_mode_case_battery_reserve_and_limit(self):
        status_300 = {
            "csv": "duty_unions_big/Practice_Custom_DutyUnion_k30_r2.csv",
            "g_kwh": 300,
            "min_soc_frac": 0,
        }
        status_240 = {
            "csv": "duty_unions_big/Practice_Custom_DutyUnion_k40_r1.csv",
            "g_kwh": 240,
            "min_soc_frac": 0.2,
        }
        self.assertEqual(
            cluster_campaign._mip_job_name(status_300, "cover", 60),
            "MC30r2G30R0T60",
        )
        self.assertEqual(
            cluster_campaign._mip_job_name(status_240, "partition", 120),
            "MP40r1G24R2T120",
        )
        arbitrary = {
            "csv": "duty_pairs/Practice_Custom_DutyPair_13302_13325.csv",
            "g_kwh": 300,
            "min_soc_frac": 0,
        }
        for minutes in (100, 1440, 1000000):
            with self.subTest(minutes=minutes):
                name = cluster_campaign._mip_job_name(
                    arbitrary, "partition", minutes
                )
                self.assertLessEqual(len(name), 15)
                self.assertRegex(name, r"^MP[A-Za-z0-9]+$")

    def test_peak_preflight_stops_before_sbatch_when_k15_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "submit_exact_peaks.sub").write_text("#!/bin/bash\n")
            (root / "data" / "duty_unions").mkdir(parents=True)
            for name in (
                "Practice_Custom_DutyUnion_k08_r3.csv",
                "Practice_Custom_DutyUnion_k13_r2.csv",
            ):
                (root / "data" / "duty_unions" / name).write_text("test\n")
            for name in (
                "hourly_prices_flat.csv",
                "hourly_prices_single_peak_08.csv",
                "hourly_prices_single_peak_12.csv",
                "hourly_prices_single_peak_18.csv",
            ):
                (root / "data" / name).write_text("test\n")

            args = argparse.Namespace(
                root=root,
                campaign="peaks",
                tasks=None,
                run_tag=None,
                submit=True,
            )
            with mock.patch.object(slurm_campaign.subprocess, "run") as run:
                with self.assertRaisesRegex(SystemExit, "k15_r6"):
                    slurm_campaign.submit_campaign(args)
                run.assert_not_called()

    def test_dive_dry_run_uses_one_named_submission_per_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "submit_exact_dive.sub").write_text("#!/bin/bash\n")
            for spec in slurm_campaign.CAMPAIGNS["dive"]:
                path = root / "data" / spec.csv
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test\n")
                price = root / "data" / spec.prices_csv
                price.parent.mkdir(parents=True, exist_ok=True)
                price.write_text("test\n")
            args = argparse.Namespace(
                root=root,
                campaign="dive",
                tasks=None,
                run_tag="test_dive",
                submit=False,
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(slurm_campaign.submit_campaign(args), 0)
            text = output.getvalue()
            self.assertEqual(text.count("[name]"), 10)
            self.assertEqual(text.count("[slurm]"), 1)
            self.assertIn("job=XD-30r1-t70", text)
            self.assertIn("job=XD-40r4-t70", text)
            self.assertIn("--hold", text)
            self.assertIn("--job-name=XD-HELD", text)
            self.assertNotIn("EXACTDIVE", text)
            self.assertFalse((root / "src" / "logs").exists())

    def test_named_submission_records_the_job_id_and_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "submit_exact_dive.sub").write_text("#!/bin/bash\n")
            spec = slurm_campaign.task_spec("dive", 1)
            csv = root / "data" / spec.csv
            csv.parent.mkdir(parents=True)
            csv.write_text("test\n")
            price = root / "data" / spec.prices_csv
            price.write_text("test\n")
            args = argparse.Namespace(
                root=root,
                campaign="dive",
                tasks="1",
                run_tag="submitted_dive",
                submit=True,
            )
            git_result = subprocess.CompletedProcess(
                args=["git"], returncode=0, stdout="a" * 40 + "\n", stderr=""
            )
            status_result = subprocess.CompletedProcess(
                args=["git"], returncode=0, stdout="", stderr=""
            )
            sbatch_result = subprocess.CompletedProcess(
                args=["sbatch"], returncode=0, stdout="98765\n", stderr=""
            )
            scontrol_result = subprocess.CompletedProcess(
                args=["scontrol"], returncode=0, stdout="", stderr=""
            )
            with mock.patch.object(
                slurm_campaign.subprocess,
                "run",
                side_effect=[
                    git_result,
                    status_result,
                    sbatch_result,
                    scontrol_result,
                    scontrol_result,
                ],
            ):
                self.assertEqual(slurm_campaign.submit_campaign(args), 0)

            manifest_path = (
                root / "src" / "results" / "slurm_campaigns" /
                "submitted_dive" / "submission.json"
            )
            manifest = json.loads(manifest_path.read_text())
            task = manifest["tasks"][0]
            self.assertTrue(manifest["submitted"])
            self.assertEqual(manifest["array_job_id"], "98765")
            self.assertEqual(manifest["state"], "released")
            self.assertEqual(task["element_job_id"], "98765_1")
            self.assertTrue(task["renamed"])
            self.assertIn("--array=1", manifest["sbatch_command"])
            self.assertIn("--job-name=XD-HELD", manifest["sbatch_command"])
            self.assertTrue(
                any(
                    f"EVSP_DR_ROOT={root.resolve()},"
                    "EVSP_SLURM_RUN_TAG=submitted_dive" in part
                    for part in manifest["sbatch_command"]
                )
            )

    def test_workers_resolve_tracked_manifest_and_reject_generic_names(self):
        dive = (REPO_ROOT / "src" / "submit_exact_dive.sub").read_text()
        peaks = (REPO_ROOT / "src" / "submit_exact_peaks.sub").read_text()
        mip = (REPO_ROOT / "src" / "submit_exact_pool_mip.sub").read_text()
        self.assertIn("slurm_campaign.py task dive", dive)
        self.assertIn("slurm_campaign.py task peaks", peaks)
        self.assertIn("scontrol update", dive)
        self.assertIn("scontrol update", peaks)
        self.assertIn("EVSP_SLURM_RUN_TAG", dive)
        self.assertIn("EVSP_SLURM_RUN_TAG", peaks)
        self.assertNotIn("#SBATCH -J EXACTDIVE", dive)
        self.assertNotIn("#SBATCH -J EXACTPEAK", peaks)
        self.assertIn("non-semantic MIP job name", mip)


if __name__ == "__main__":
    unittest.main()

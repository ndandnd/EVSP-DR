import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import launch_scale_ladder as ladder  # noqa: E402


class LadderLiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.plan = ladder.build_plan(
            "ll_test", Path(sys.executable), root / "reservations"
        )
        cls.plan_path = root / "approved-plan.json"
        cls.plan_path.write_bytes(ladder.canonical(cls.plan))
        cls.python = sys.executable

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _printed_command(self, group, index):
        environment = dict(os.environ)
        environment.update({
            "SLURM_ARRAY_TASK_ID": str(index),
            "LL_PYTHON": self.python,
            "LL_PRINT_COMMAND": "1",
        })
        completed = subprocess.run(
            [
                "bash", str(REPO / "scripts/ladder_lite/run_cell.sh"),
                str(self.plan_path), group,
            ],
            cwd=REPO,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return shlex.split(completed.stdout)

    def test_all_138_array_indices_resolve_to_plan_jobs(self):
        jobs = {job["job_key"]: job for job in self.plan["jobs"]}
        observed = []
        for group, keys in self.plan["task_groups"].items():
            for index, key in enumerate(keys):
                command = self._printed_command(group, index)
                self.assertTrue(command)
                self.assertIn(key, jobs)
                observed.append(key)
        self.assertEqual(len(observed), 138)
        self.assertEqual(set(observed), set(jobs))

    def test_science_commands_match_reviewed_worker(self):
        jobs = {job["job_key"]: job for job in self.plan["jobs"]}
        for group in ("PREFLIGHT", "SEED", "CG", "MIP_RAW"):
            job = jobs[self.plan["task_groups"][group][0]]
            actual = self._printed_command(group, 0)
            instance = job["instance"]
            if group == "PREFLIGHT":
                expected = [
                    self.python, "-B",
                    str(REPO / "src/audit_scale_ladder_known_membership.py"),
                    "--instance", str(REPO / "data" / instance["relative_path"]),
                    "--instance-sha256", instance["instance_file_sha256"],
                    "--scale", str(job["scale"]), "--selection-replicate",
                    str(job["selection_replicate"]), "--out", job["output"],
                    "--csv-out", str(Path(job["output"]).with_suffix(".csv")),
                ]
            elif group == "SEED":
                expected = [
                    self.python, "-B",
                    str(REPO / "src/prepare_scale_ladder_known_partition.py"),
                    "--instance", str(REPO / "data" / instance["relative_path"]),
                    "--instance-sha256", instance["instance_file_sha256"],
                    "--out", job["output"],
                ]
            elif group == "CG":
                expected = [
                    self.python, "-u", str(REPO / "src/exact_pricer_expanded.py"),
                    "--csv", instance["relative_path"], "--prices_csv",
                    "hourly_prices_flat.csv", "--g-kwh", "300", "--charge-kw",
                    "300", "--min-soc-frac", "0", "--soc-step",
                    str(job["soc_step"]), "--block-min", str(job["block_min"]),
                    "--master-sense", "partition", "--initial-pool", "singletons",
                    "--wall-limit-s", str(job["budget_s"] + 60),
                    "--checkpoint-every", "25", "--resume",
                    "--snapshot-at-minutes",
                    ",".join(map(str, job["snapshot_minutes"])),
                    "--out", job["output"],
                ]
                if job.get("telemetry"):
                    expected += ["--phase-telemetry", job["telemetry"]]
            else:
                source = jobs[job["dependency_cg"]]
                expected = [
                    self.python, "-u", str(REPO / "src/run_exact_pool_mip.py"),
                    "--result", source["output"], "--two-stage", "--threads",
                    str(job["threads"]), "--timelimit", str(job["budget_s"]),
                    "--mipgap", "0.0001", "--progress-dir",
                    job["progress_dir"], "--out", job["output"],
                ]
            self.assertEqual(actual, expected, group)

    def test_submit_dry_run_groups_budget_memory_and_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "campaign").mkdir()
            plan = {
                "task_groups": {"CG": ["a", "b", "c", "d"]},
                "jobs": [
                    {"job_key": "a", "scale": 2, "budget_s": 60,
                     "partition": "default_partition", "threads": 2},
                    {"job_key": "b", "scale": 3, "budget_s": 120,
                     "partition": "default_partition", "threads": 2},
                    {"job_key": "c", "scale": 2, "budget_s": 60,
                     "partition": "default_partition", "threads": 2},
                    {"job_key": "d", "scale": 8, "budget_s": 180,
                     "partition": "default_partition", "threads": 2},
                ],
            }
            (root / "campaign/approved-plan.json").write_text(json.dumps(plan))
            environment = {
                **os.environ, "LL_ROOT": str(root), "LL_PYTHON": self.python
            }
            completed = subprocess.run(
                [
                    "bash", str(REPO / "scripts/ladder_lite/submit.sh"),
                    "CG", "--scales", "2,3", "--dry-run",
                ],
                cwd=REPO, env=environment, text=True,
                capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            lines = completed.stdout.splitlines()
            commands = [line for line in lines if line.startswith("sbatch ")]
            self.assertEqual(len(commands), 2)
            self.assertTrue(any("--array=0,2%16" in line for line in commands))
            self.assertTrue(any("--array=1%16" in line for line in commands))
            self.assertTrue(all("--mem=16G" in line and "-c 2" in line
                                for line in commands))
            self.assertIn("total_tasks=3", lines)

    def test_record_results_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); records = root / "records"
            (root / "campaign").mkdir(parents=True); (root / "normalized").mkdir()
            records.mkdir()
            (records / "RESULTS_LOG.csv").write_bytes(
                (REPO / "records/RESULTS_LOG.csv").read_bytes()
            )
            output = root / "cg.json"; output.write_text("{}")
            job = {
                "job_key": "cg_a", "phase": "CG", "arm": None, "scale": 2,
                "selection_replicate": 1, "cg_replicate": 1,
                "soc_step": 15.0, "block_min": 10, "budget_s": 180,
                "target_fleet": 2, "cell_id": "k02_s1_c1",
                "output": str(output), "scientific_role": None,
            }
            (root / "campaign/approved-plan.json").write_text(
                json.dumps({"jobs": [job]})
            )
            (root / "campaign/campaign.json").write_text(json.dumps({
                "execution_mode": "ladder_lite_direct_array",
                "commit": "a" * 40,
            }))
            with (root / "normalized/cg_run_summary.csv").open("w", newline="") as h:
                w=csv.DictWriter(h,fieldnames=("cell_id","campaign_role","soc_step","block_min","cg_replicate","final_route_weight","final_min_reduced_cost","pricing_certified","final_artificial_mass","pool_columns","iterations","elapsed_s","stopping_reason","censored"));w.writeheader();w.writerow({"cell_id":"k02_s1_c1","campaign_role":"primary","soc_step":"15.0","block_min":"10","cg_replicate":"1","final_route_weight":"2","censored":"False"})
            for name in ("mip_run_summary.csv","cg_iteration_long.csv","mip_checkpoint_long.csv"):
                (root / "normalized" / name).write_text("cell_id\n")
            env={**os.environ,"LL_ROOT":str(root),"LL_PYTHON":self.python,
                 "LL_RECORDS_ROOT":str(records)}
            command=["bash",str(REPO/"scripts/ladder_lite/record_results.sh"),"run1"]
            first=subprocess.run(command,cwd=REPO,env=env,text=True,capture_output=True)
            second=subprocess.run(command,cwd=REPO,env=env,text=True,capture_output=True)
            self.assertEqual(first.returncode,0,first.stderr)
            self.assertEqual(second.returncode,0,second.stderr)
            self.assertIn("appended=1 skipped=0",first.stdout)
            self.assertIn("appended=0 skipped=1",second.stdout)
            rows=list(csv.DictReader((records/"RESULTS_LOG.csv").open()))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]["route_weight_meaning"],
                             "combined-cost-master route weight")


if __name__ == "__main__":
    unittest.main()

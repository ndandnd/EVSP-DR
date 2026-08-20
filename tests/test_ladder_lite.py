import csv
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import launch_scale_ladder as ladder  # noqa: E402
import summarize_scale_ladder_lite as lite_summary  # noqa: E402


class LadderLiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        executable = Path(sys.executable).resolve()
        identity = {
            "schema": "synthetic-test-environment",
            "portable": {
                "python": "3.12.test",
                "executable": str(executable),
                "executable_sha256": hashlib.sha256(
                    executable.read_bytes()
                ).hexdigest(),
            },
            "portable_identity_sha256": "a" * 64,
        }
        with patch.object(ladder, "_environment", return_value=identity):
            cls.plan = ladder.build_plan(
                "ll_test", executable, root / "reservations"
            )
        cls.plan_path = root / "approved-plan.json"
        cls.plan_path.write_bytes(ladder.canonical(cls.plan))
        cls.python = sys.executable

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _printed_command(self, group, index, extra_environment=None):
        environment = dict(os.environ)
        environment.update({
            "SLURM_ARRAY_TASK_ID": str(index),
            "LL_PYTHON": self.python,
            "LL_PRINT_COMMAND": "1",
        })
        environment.update(extra_environment or {})
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

    def test_staged_worker_without_repo_fails_with_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = root / "slurm_script"
            staged.write_bytes(
                (REPO / "scripts/ladder_lite/run_cell.sh").read_bytes()
            )
            plan = copy.deepcopy(self.plan)
            key = plan["task_groups"]["PREFLIGHT"][0]
            job = next(row for row in plan["jobs"] if row["job_key"] == key)
            output = root / "preflight.json"
            job["output"] = str(output)
            plan["task_groups"] = {"PREFLIGHT": [key]}
            plan["jobs"] = [job]
            plan_path = root / "approved-plan.json"
            plan_path.write_text(json.dumps(plan))
            environment = {
                **os.environ,
                "SLURM_ARRAY_TASK_ID": "0",
                "LL_PYTHON": self.python,
            }
            environment.pop("LL_REPO", None)
            completed = subprocess.run(
                ["bash", str(staged), str(plan_path), "PREFLIGHT"],
                cwd=root, env=environment, text=True,
                capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("unresolved repository root", completed.stderr)
            self.assertIn("[ll] host=", completed.stdout)
            self.assertIn(" task=0 repo=", completed.stdout)
            self.assertNotIn(
                "rev-parse HEAD 2>/dev/null", staged.read_text()
            )
            failed = Path(f"{output}.failed")
            self.assertTrue(failed.is_file())
            self.assertIn("exit_code=2", failed.read_text())

    def test_science_commands_match_reviewed_worker(self):
        jobs = {job["job_key"]: job for job in self.plan["jobs"]}
        for group in (
            "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
            "MIP_RAW", "MIP_KNOWN",
        ):
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
            elif group in {"CG", "CG_SENSITIVITY"}:
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
                if job["arm"] == "KNOWN-PARTITION":
                    expected += [
                        "--initial-partition-routes",
                        jobs[job["dependency_seed"]]["output"],
                    ]
            self.assertEqual(actual, expected, group)
        overridden = self._printed_command(
            "CG", 0, {"LL_BUDGET_OVERRIDE_S": "180"}
        )
        self.assertEqual(
            overridden[overridden.index("--wall-limit-s") + 1], "180"
        )

    def test_submit_dry_run_groups_budget_memory_and_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); campaign = "test-campaign"
            (root / "campaign" / campaign).mkdir(parents=True)
            plan = {
                "task_groups": {
                    "CG": ["a", "b", "c", "d"],
                    "PREFLIGHT": ["d"],
                },
                "jobs": [
                    {"job_key": "a", "scale": 2, "budget_s": 60,
                     "partition": "default_partition", "threads": 2},
                    {"job_key": "b", "scale": 3, "budget_s": 120,
                     "partition": "default_partition", "threads": 2},
                    {"job_key": "c", "scale": 2, "budget_s": 60,
                     "partition": "default_partition", "threads": 2},
                    {"job_key": "d", "scale": 20, "budget_s": 180,
                     "partition": "default_partition", "threads": 2},
                ],
            }
            (root / "campaign" / campaign / "approved-plan.json").write_text(
                json.dumps(plan)
            )
            environment = {
                **os.environ, "LL_ROOT": str(root), "LL_PYTHON": self.python,
                "LL_CAMPAIGN": campaign,
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
            tokens = [shlex.split(line) for line in commands]
            self.assertTrue(any("--array=0,2%16" in row for row in tokens))
            self.assertTrue(any("--array=1%16" in row for row in tokens))
            self.assertTrue(all("--mem=16G" in row and "-c" in row
                                and row[row.index("-c")+1]=="2" for row in tokens))
            self.assertTrue(all(
                any(token == f"--export=ALL,LL_PYTHON={self.python},LL_REPO={REPO}"
                    for token in row)
                for row in tokens
            ))
            self.assertIn("total_tasks=3", lines)
            for group, expected_mem in (("CG", "24G"), ("PREFLIGHT", "16G")):
                long_run = subprocess.run(
                    [
                        "bash", str(REPO / "scripts/ladder_lite/submit.sh"),
                        group, "--scales", "20", "--dry-run",
                    ],
                    cwd=REPO, env=environment, text=True,
                    capture_output=True, check=False,
                )
                self.assertEqual(long_run.returncode, 0, long_run.stderr)
                self.assertIn(
                    f"--mem={expected_mem}", shlex.split(
                        long_run.stdout.splitlines()[0]
                    ),
                )

    def test_plan_is_campaign_scoped_and_logs_verbose_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkout = root / "repo"; state = root / "state"
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(checkout), "HEAD"],
                cwd=REPO, check=True, text=True, capture_output=True,
            )
            try:
                observed = []
                for campaign in ("campaign-one", "campaign-two"):
                    environment = {
                        **os.environ,
                        "LL_ROOT": str(state),
                        "LL_PYTHON": self.python,
                        "LL_CAMPAIGN": campaign,
                    }
                    completed = subprocess.run(
                        ["bash", str(
                            checkout / "scripts/ladder_lite/plan.sh"
                        )],
                        cwd=checkout, env=environment, text=True,
                        capture_output=True, check=False,
                    )
                    self.assertEqual(
                        completed.returncode, 0, completed.stderr
                    )
                    campaign_dir = state / "campaign" / campaign
                    plan = campaign_dir / "approved-plan.json"
                    self.assertTrue(plan.is_file())
                    self.assertTrue((campaign_dir / "task_matrix.csv").is_file())
                    log = (campaign_dir / "plan.log").read_text()
                    self.assertIn('"task_groups"', log)
                    self.assertNotIn('"task_groups"', completed.stdout)
                    self.assertIn("[ll] staging scientific inputs", completed.stdout)
                    self.assertIn("total tasks  : 138", completed.stdout)
                    observed.append(json.loads(plan.read_text())["campaign"])
                self.assertEqual(observed, ["campaign-one", "campaign-two"])
            finally:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(checkout)],
                    cwd=REPO, check=True, text=True, capture_output=True,
                )

    def test_record_results_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); records = root / "records"; campaign = "test-campaign"
            (root / "campaign" / campaign).mkdir(parents=True)
            (root / "normalized").mkdir()
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
            missing_job = {
                **job,
                "job_key": "cg_b",
                "cell_id": "k02_s2_c1",
                "selection_replicate": 2,
                "output": str(root / "missing.json"),
            }
            mip_output=root/"mip.json";mip_output.write_text("{}")
            mip_job={
                **job,"job_key":"mip_a","phase":"MIP","arm":"RAW",
                "dependency_cg":"cg_a","output":str(mip_output),
                "progress_dir":str(root/"progress"),"threads":8,
            }
            (root / "campaign" / campaign / "approved-plan.json").write_text(
                json.dumps({"jobs": [job, missing_job, mip_job]})
            )
            (root / "campaign" / campaign / "campaign.json").write_text(json.dumps({
                "execution_mode": "ladder_lite_direct_array",
                "commit": "a" * 40,
            }))
            with (root / "normalized/cg_run_summary.csv").open("w", newline="") as h:
                w=csv.DictWriter(h,fieldnames=("cell_id","campaign_role","soc_step","block_min","cg_replicate","final_route_weight","final_min_reduced_cost","pricing_certified","final_artificial_mass","pool_columns","iterations","elapsed_s","stopping_reason","censored"));w.writeheader();w.writerow({"cell_id":"k02_s1_c1","campaign_role":"primary","soc_step":"15.0","block_min":"10","cg_replicate":"1","final_route_weight":"2","censored":"False"})
            with (root/"normalized/mip_run_summary.csv").open("w",newline="") as h:
                w=csv.DictWriter(h,fieldnames=("cell_id","arm","cg_replicate","scale","budget_s","output_available","censored","buses","fleet_bound","mip_gap","runtime_s","status_name","missing_reason"));w.writeheader();w.writerows([
                    {"cell_id":"k02_s1_c1","arm":"RAW","cg_replicate":"1","scale":"2","budget_s":"60","output_available":"True","censored":"False","buses":"2","fleet_bound":"2","mip_gap":"0"},
                    {"cell_id":"k40_s2_c1","arm":"RAW","cg_replicate":"1","scale":"40","output_available":"False","censored":"True","missing_reason":"reuse missing"}])
            (root/"normalized/cg_iteration_long.csv").write_text("cell_id\n")
            (root/"normalized/mip_checkpoint_long.csv").write_text("cell_id,arm,cg_replicate,node_count\nk02_s1_c1,RAW,1,7\n")
            env={**os.environ,"LL_ROOT":str(root),"LL_PYTHON":self.python,
                 "LL_RECORDS_ROOT":str(records),"LL_CAMPAIGN":campaign}
            command=["bash",str(REPO/"scripts/ladder_lite/record_results.sh"),"run1"]
            first=subprocess.run(command,cwd=REPO,env=env,text=True,capture_output=True)
            second=subprocess.run(command,cwd=REPO,env=env,text=True,capture_output=True)
            self.assertEqual(first.returncode,0,first.stderr)
            self.assertEqual(second.returncode,0,second.stderr)
            self.assertIn("appended=4 skipped=0",first.stdout)
            self.assertIn("appended=0 skipped=4",second.stdout)
            rows=list(csv.DictReader((records/"RESULTS_LOG.csv").open()))
            self.assertEqual(len(rows),4)
            self.assertEqual(rows[0]["route_weight_meaning"],
                             "combined-cost-master route weight")
            missing=next(row for row in rows if row["cell_id"]=="cg_b")
            self.assertEqual(missing["status"],"missing")
            self.assertEqual(missing["censor_reason"],"normalized row missing")
            mip=next(row for row in rows if row["cell_id"]=="mip_a")
            self.assertEqual(mip["arm"],"RAW")
            self.assertEqual(mip["route_weight"],"2")
            self.assertEqual(mip["mip_nodes"],"7")
            reuse=next(row for row in rows if row["group"]=="MIP_REUSE")
            self.assertEqual(reuse["censor_reason"],"reuse missing")

    def test_partial_normalization_emits_missing_rows_and_rejects_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); campaign=root/"campaign"; campaign.mkdir()
            plan=copy.deepcopy(self.plan)
            for job in plan["jobs"]:
                job["output"]=str(root/"outputs"/f"{job['job_key']}.json")
                if job.get("progress_dir"):
                    job["progress_dir"]=str(root/"progress"/job["job_key"])
                if job.get("telemetry"):
                    job["telemetry"]=str(root/"telemetry"/f"{job['job_key']}.jsonl")
            raw=ladder.canonical(plan); (campaign/"approved-plan.json").write_bytes(raw)
            (campaign/"campaign.json").write_text(json.dumps({
                "approval_sha256":hashlib.sha256(raw).hexdigest(),
                "execution_mode":"ladder_lite_direct_array",
                "commit":plan["checkout_identity"]["commit"],
            }))
            output=root/"normalized"
            result=lite_summary.summarize(campaign,output)
            self.assertEqual(result["completed"],0)
            self.assertEqual(result["omitted"],138)
            with (output/"cg_run_summary.csv").open(newline="") as h:
                cg=list(csv.DictReader(h))
            with (output/"mip_run_summary.csv").open(newline="") as h:
                mip=list(csv.DictReader(h))
            self.assertEqual(len(cg),53)
            self.assertGreaterEqual(len(mip),42)
            self.assertTrue(all(row["censored"]=="True" for row in cg))
            provenance=json.loads((output/"provenance.json").read_text())
            self.assertEqual(provenance["execution_mode"],"ladder_lite_direct_array")
            self.assertIn(
                "src/run_scale_ladder_local_diagnostics.py",
                provenance["code_hashes"],
            )
            self.assertFalse(any(
                b'"local_diagnostic"' in path.read_bytes()
                for path in output.iterdir() if path.is_file()
            ))
            overridden=next(job for job in plan["jobs"] if job["phase"]=="CG")
            shutil_target=Path(overridden["output"]+".override.json")
            shutil_target.parent.mkdir(parents=True,exist_ok=True);shutil_target.write_text("{}")
            second=root/"second"
            result=lite_summary.summarize(campaign,second)
            self.assertEqual(result["omitted"],138)
            with (second/"cg_run_summary.csv").open(newline="") as h:
                excluded=next(
                    row for row in csv.DictReader(h)
                    if row["cell_id"]==overridden["cell_id"]
                    and row["campaign_role"]=="primary"
                    and row["cg_replicate"]==str(overridden["cg_replicate"])
                )
            self.assertEqual(excluded["stopping_reason"],"excluded")
            self.assertEqual(
                excluded["grid_interpretation"],"excluded: budget_overridden"
            )

    def test_lite_validator_rejects_orphan_snapshots_and_bad_mip_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); commit="a"*40
            lite_summary.PLAN={"checkout_identity":{"commit":commit}}
            cg=root/"cg.json"; cg.write_text(json.dumps({
                "provenance":{"git_commit":commit},"stop_reason":"certified",
                "snapshot_availability":{"5":"censored_solver_terminated_before_mark"},
            }))
            Path(str(cg)+".done").touch();Path(str(cg)+".columns.jsonl").touch()
            Path(str(cg)+".iters.csv").write_text("iteration\n")
            (root/"cg.m5.snapshot.json").touch()
            with self.assertRaisesRegex(ValueError,"orphan"):
                lite_summary._validate({
                    "job_key":"cg","phase":"CG","output":str(cg),
                    "telemetry":None,"snapshot_minutes":[5],
                },"")
            mip=root/"mip.json";mip.write_text(json.dumps({
                "mip_provenance":{
                    "expected_git_commit":commit,"observed_git_commit":commit,
                    "final_observed_git_commit":commit,"git_dirty":False,
                    "tracked_clean_at_end":True,
                    "arguments":{"two_stage":True,"cover":False,"threads":8,
                                 "timelimit":60,"mipgap":0.0001}},
                "progress":{},
            }))
            Path(str(mip)+".done").touch();progress=root/"progress";progress.mkdir()
            (progress/"final.json").touch()
            with self.assertRaisesRegex(ValueError,"schedule"):
                lite_summary._validate({
                    "job_key":"mip","phase":"MIP","output":str(mip),
                    "progress_dir":str(progress),"threads":8,"budget_s":60,
                },"")

    def test_partial_cg_values_are_preserved_and_marked_censored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"cg_run_summary.csv").write_text(
                "cell_id,campaign_role,soc_step,block_min,cg_replicate,final_route_weight,censored,stopping_reason\n"
                "k02_s1_c1,primary,15.0,10,1,2.181818,False,certified\n"
                "k02_s1_c1,small_grid_sensitivity,5.0,10,1,2.0,False,certified\n"
            )
            (root/"mip_run_summary.csv").write_text(
                "cell_id,arm,cg_replicate,censored,missing_reason\n"
            )
            (root/"scale_progress_summary.csv").write_text(
                "scale,selection_replicate,cg_replicate,cg_censored,cg_stopping_reason,missing_reason\n"
                "2,1,1,False,certified,\n"
            )
            lite_summary._mark_censored(root,[{
                "cell_id":"k02_s1_c1","cg_replicate":1,
                "phase":"CG","arm":None,"soc_step":15.0,"block_min":10,
                "scale":2,"selection_replicate":1,
            }])
            row=next(csv.DictReader(
                (root/"cg_run_summary.csv").open()
            ))
            self.assertEqual(row["final_route_weight"],"2.181818")
            self.assertEqual(row["censored"],"True")
            self.assertIn("without .done",row["stopping_reason"])
            rows=list(csv.DictReader((root/"cg_run_summary.csv").open()))
            self.assertEqual(rows[1]["censored"],"False")
            progress=next(csv.DictReader(
                (root/"scale_progress_summary.csv").open()
            ))
            self.assertEqual(progress["cg_censored"],"True")


if __name__ == "__main__":
    unittest.main()

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

    def _printed_command(
        self, group, index, extra_environment=None, plan_path=None,
    ):
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
                str(plan_path or self.plan_path), group,
            ],
            cwd=REPO,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return shlex.split(completed.stdout.splitlines()[-1])

    def test_all_362_array_indices_resolve_to_plan_jobs(self):
        jobs = {job["job_key"]: job for job in self.plan["jobs"]}
        observed = []
        for group, keys in self.plan["task_groups"].items():
            for index, key in enumerate(keys):
                command = self._printed_command(group, index)
                self.assertTrue(command)
                self.assertIn(key, jobs)
                observed.append(key)
        self.assertEqual(len(observed), 362)
        self.assertEqual(set(observed), set(jobs))

    def test_declared_grids_cross_every_scale_cell(self):
        expected = {
            ("soc15_b10", 15.0, 10, "primary"),
            ("soc5_b10", 5.0, 10, "resolution"),
            ("soc2p5_b10", 2.5, 10, "resolution"),
            ("soc1_b10", 1.0, 10, "resolution"),
            ("soc1_b5", 1.0, 5, "resolution"),
        }
        self.assertEqual({
            (grid["grid_id"], grid["soc_step"], grid["block_min"],
             grid["grid_role"])
            for grid in self.plan["cg_grids"]
        }, expected)
        cg = [
            job for job in self.plan["jobs"]
            if job["phase"] in {"CG", "CG_SENSITIVITY"}
        ]
        by_cell = {}
        for job in cg:
            by_cell.setdefault(job["cell_id"], set()).add((
                job["grid_id"], job["soc_step"], job["block_min"],
                job["grid_role"],
            ))
            self.assertFalse(job["diagnostic_only"])
        self.assertEqual(len(by_cell), 41)
        self.assertTrue(all(grids == expected for grids in by_cell.values()))
        fine_large = next(
            job for job in cg
            if job["scale"] == 40 and job["grid_id"] == "soc1_b5"
        )
        self.assertEqual(
            (fine_large["memory_gb"], fine_large["max_concurrency"]),
            (128, 1),
        )
        primary = {
            job["job_key"]: job for job in cg if job["grid_role"] == "primary"
        }
        for job in self.plan["jobs"]:
            if job["phase"] == "MIP":
                self.assertIn(job["dependency_cg"], primary)

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
                    "hourly_prices_flat.csv", "--g-kwh", str(job["g_kwh"]),
                    "--charge-kw", str(job["charge_kw"]),
                    "--min-soc-frac", str(job["min_soc_frac"]), "--soc-step",
                    str(job["soc_step"]), "--block-min", str(job["block_min"]),
                    "--master-sense", job["master_sense"],
                    "--initial-pool", job["initial_pool"],
                    "--objective", job["objective"],
                    "--columns_per_iter", str(job["columns_per_iter"]),
                    "--max-iters", str(job["max_iters"]),
                    "--diversify-rounds", str(job["diversify_rounds"]),
                    "--wall-limit-s", str(job["budget_s"] + 60),
                    "--checkpoint-every", str(job["checkpoint_every"]), "--resume",
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

    def test_lexicographic_plan_omits_unsupported_operational_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = copy.deepcopy(self.plan)
            job_key = plan["task_groups"]["CG"][0]
            job = next(
                row for row in plan["jobs"] if row["job_key"] == job_key
            )
            job["objective"] = "lexicographic-fleet"
            plan_path = Path(tmp) / "approved-plan.json"
            plan_path.write_text(json.dumps(plan))
            command = self._printed_command(
                "CG", 0, plan_path=plan_path,
            )
        self.assertEqual(
            command[command.index("--objective") + 1],
            "lexicographic-fleet",
        )
        for unsupported in (
            "--resume", "--snapshot-at-minutes", "--phase-telemetry",
            "--checkpoint-every",
        ):
            self.assertNotIn(unsupported, command)
        self.assertEqual(command[-2:], ["--out", job["output"]])

    def test_submit_dry_run_groups_budget_memory_and_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); campaign = "test-campaign"
            (root / "campaign" / campaign).mkdir(parents=True)
            plan = {
                "task_groups": {
                    "CG": ["a", "b", "c", "d"],
                    "PREFLIGHT": ["e"],
                },
                "jobs": [
                    {"job_key": "a", "scale": 2, "budget_s": 60,
                     "partition": "default_partition", "threads": 2,
                     "memory_gb": 16, "max_concurrency": 16},
                    {"job_key": "b", "scale": 3, "budget_s": 120,
                     "partition": "default_partition", "threads": 2,
                     "memory_gb": 16, "max_concurrency": 16},
                    {"job_key": "c", "scale": 2, "budget_s": 60,
                     "partition": "default_partition", "threads": 2,
                     "memory_gb": 16, "max_concurrency": 16},
                    {"job_key": "d", "scale": 20, "budget_s": 180,
                     "partition": "default_partition", "threads": 2,
                     "memory_gb": 24, "max_concurrency": 8},
                    {"job_key": "e", "scale": 20, "budget_s": 180,
                     "partition": "default_partition", "threads": 2,
                     "memory_gb": 16, "max_concurrency": 16},
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
            cg_long = subprocess.run(
                ["bash", str(REPO / "scripts/ladder_lite/submit.sh"),
                 "CG", "--scales", "20", "--dry-run"],
                cwd=REPO, env=environment, text=True,
                capture_output=True, check=False,
            )
            self.assertIn("--array=3%8", shlex.split(
                cg_long.stdout.splitlines()[0]
            ))

    def test_plan_is_campaign_scoped_and_logs_verbose_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkout = root / "repo"; state = root / "state"
            fake_python = root / "python3.12"
            fake_python.write_text(f"""#!{self.python}
import json,os,sys
args=sys.argv[1:]
if args and args[0]=="-B":args=args[1:]
if args and args[0].endswith("launch_scale_ladder.py"):
 def value(flag):return args[args.index(flag)+1]
 plan={{"campaign":value("--campaign"),"task_groups":{{"TEST":list(range(362))}}}}
 open(value("--plan-out"),"x").write(json.dumps(plan))
 open(value("--matrix-out"),"x").write("task\\n")
 print(json.dumps(plan,indent=2));raise SystemExit(0)
if args and args[0]=="-c":raise SystemExit(0)
os.execv({self.python!r},[{self.python!r},"-B",*args])
""")
            fake_python.chmod(0o755)
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
                        "LL_PYTHON": str(fake_python),
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
                    self.assertIn("total tasks  : 362", completed.stdout)
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
                w=csv.DictWriter(h,fieldnames=("cell_id","campaign_role","soc_step","block_min","cg_replicate","final_route_weight","final_min_reduced_cost","pricing_certified","final_artificial_mass","pool_columns","iterations","elapsed_s","stopping_reason","censored"));w.writeheader();w.writerow({"cell_id":"k02_s1_c1","campaign_role":"primary","soc_step":"15.0","block_min":"10","cg_replicate":"1","final_route_weight":"2","pricing_certified":"True","censored":"False"})
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
                             "fleet LP lower bound (certified discretized model; grid stated; D0019)")
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
            self.assertEqual(result["omitted"],362)
            with (output/"cg_run_summary.csv").open(newline="") as h:
                cg=list(csv.DictReader(h))
            with (output/"mip_run_summary.csv").open(newline="") as h:
                mip=list(csv.DictReader(h))
            self.assertEqual(len(cg),205)
            self.assertGreaterEqual(len(mip),78)
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
            self.assertEqual(result["omitted"],362)
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
            lex=root/"lex.json";lex.write_text(json.dumps({
                "objective":"lexicographic-fleet",
                "provenance":{"git_commit":commit},
                "phases":[{"phase":2,"stop_reason":"certified",
                           "certified":True,"iterations":1,"pool_columns":1}],
            }))
            Path(str(lex)+".done").touch()
            Path(str(lex)+".columns.jsonl").touch()
            Path(str(lex)+".lexicographic.iters.csv").write_text(
                "phase,iteration,objective,route_weight,artificial_mass,minimum_reduced_cost,pool_columns\n"
                "1,1,1000000,10,1,-100,5\n"
                "2,1,2.0,2.0,0,-0.0,7\n"
            )
            lite_summary._validate({
                "job_key":"lex","phase":"CG","output":str(lex),
                "telemetry":str(root/"unsupported-lex-telemetry.jsonl"),
                "snapshot_minutes":[5],
            },"")
            compatible,mapping,payload,is_lex=lite_summary._compat_cg({
                "job_key":"lex","output":str(lex),
            },root/"temp")
            self.assertTrue(is_lex);self.assertTrue(payload["certified_rc_optimal"])
            self.assertEqual(mapping[compatible["output"]],lex)
            with Path(str(compatible["output"])+".iters.csv").open(newline="") as h:
                reader=csv.DictReader(h);rows=list(reader)
                self.assertEqual(reader.fieldnames,[
                    "elapsed_s","iteration","lp_obj","route_weight",
                    "artificials","min_rc","pool_columns",
                ])
            self.assertEqual(len(rows),1)
            self.assertEqual(
                {key:rows[0][key] for key in (
                    "lp_obj","route_weight","artificials","min_rc",
                    "pool_columns",
                )},
                {"lp_obj":"2.0","route_weight":"2.0","artificials":"0",
                 "min_rc":"-0.0","pool_columns":"7"},
            )
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

    def test_lite_validator_enforces_every_declared_cg_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); commit="b"*40; output=root/"cg.json"
            lite_summary.PLAN={"checkout_identity":{"commit":commit}}
            output.write_text(json.dumps({
                "g_kwh":240.0,"charge_kw":240.0,"min_soc_frac":0.2,
                "initial_pool":"matching","master_sense":"partition",
                "stop_reason":"certified","provenance":{
                    "git_commit":commit,"args":{
                        "columns_per_iter":100,"max_iters":100000,
                        "diversify_rounds":3,"objective":"combined-cost",
                        "checkpoint_every":25,
                    },
                },
            }))
            Path(str(output)+".done").touch()
            Path(str(output)+".columns.jsonl").touch()
            Path(str(output)+".iters.csv").write_text("iteration\n")
            job={
                "job_key":"cg","phase":"CG","output":str(output),
                "telemetry":None,"snapshot_minutes":[],"g_kwh":240.0,
                "charge_kw":240.0,"min_soc_frac":0.2,
                "columns_per_iter":100,"max_iters":100000,
                "diversify_rounds":3,"initial_pool":"matching",
                "objective":"combined-cost","master_sense":"partition",
                "checkpoint_every":25,
            }
            lite_summary._validate(job,"")
            mismatches={
                "g_kwh":300.0,"charge_kw":300.0,"min_soc_frac":0.0,
                "columns_per_iter":101,"max_iters":99999,
                "diversify_rounds":4,"initial_pool":"greedy",
                "objective":"lexicographic-fleet",
            }
            for field,value in mismatches.items():
                with self.subTest(field=field):
                    changed={**job,field:value}
                    with self.assertRaisesRegex(
                        ValueError,"CG identity/artifacts invalid",
                    ):
                        lite_summary._validate(changed,"")

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

    def test_resolution_postpass_removes_legacy_fallback_claim(self):
        self.assertEqual(
            lite_summary._labels(
                "known_duties_contained_fallback_grid,local_diagnostic"
            ),
            "declared_resolution_scale_grid,local_diagnostic",
        )

    def test_route_weight_labels_split_certified_from_uncertified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fields=("cell_id","campaign_role","soc_step","block_min",
                    "cg_replicate","pricing_certified")
            for name in ("cg_iteration_long.csv","cg_run_summary.csv"):
                with (root/name).open("w",newline="") as h:
                    writer=csv.DictWriter(h,fieldnames=fields,lineterminator="\n")
                    writer.writeheader();writer.writerows([
                        {"cell_id":"cert","pricing_certified":"True"},
                        {"cell_id":"open","pricing_certified":"False"},
                    ])
            lite_summary._postprocess_cg_tables(root,set())
            with (root/"cg_run_summary.csv").open(newline="") as h:
                rows={row["cell_id"]:row for row in csv.DictReader(h)}
            self.assertIn("D0019",rows["cert"]["route_weight_meaning"])
            self.assertEqual(
                rows["open"]["route_weight_meaning"],
                "upper bound on LP optimum only; no fleet LP lower bound",
            )


if __name__ == "__main__":
    unittest.main()

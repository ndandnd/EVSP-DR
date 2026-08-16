import contextlib
import concurrent.futures
import hashlib
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import archive_exact_cg_profile_campaign as archiver  # noqa: E402
import exact_cg_profile_results as result_validation  # noqa: E402
import install_exact_cg_profile_input as input_installer  # noqa: E402
import launch_exact_cg_profile_campaign as launcher  # noqa: E402
import monitor_exact_cg_profile_campaign as monitor  # noqa: E402
import reconcile_exact_cg_profile_campaign as reconciler  # noqa: E402
import summarize_exact_cg_profiles as summarizer  # noqa: E402


class ExactCgProfileCampaignTests(unittest.TestCase):
    @staticmethod
    def _python_identity():
        return {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_executable_sha256": "e" * 64,
            "python_version": "3.12.test",
            "numpy_version": "test",
            "pandas_version": "test",
            "scipy_version": "test",
            "highs_version": "test",
            "observed_platform": "test",
            "identity_sha256": "f" * 64,
        }

    def _campaign_fixture(self, root: Path):
        repo = root / "repo"
        data = repo / "data"
        data.mkdir(parents=True)
        instance = data / "instance.csv"
        prices = data / "prices.csv"
        instance.write_text("instance\n")
        prices.write_text("prices\n")
        instance_sha = hashlib.sha256(instance.read_bytes()).hexdigest()
        prices_sha = hashlib.sha256(prices.read_bytes()).hexdigest()
        snapshots = {}
        treatments = {
            "historical": (None, None),
            "ca": ("cover", "artificial"),
            "cs": ("cover", "singletons"),
            "pa": ("partition", "artificial"),
            "ps": ("partition", "singletons"),
        }
        for index, label in enumerate(
                ("historical", "ca", "cs", "pa", "ps")):
            folder = repo / "source" / label
            folder.mkdir(parents=True)
            result = folder / f"{label}.snapshot.json"
            journal = Path(str(result) + ".columns.jsonl")
            journal.write_text(
                json.dumps({
                    "trips": [1], "cost": 100000.0 + index,
                }) + "\n"
            )
            result.write_text(json.dumps({
                "csv": "instance.csv",
                "prices_csv": "prices.csv",
                "trip_ids": [1],
                "columns": 1,
                "soc_step": 15.0,
                "block_min": 10,
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "min_soc_frac": 0.0,
                "master_sense": treatments[label][0],
                "initial_pool": treatments[label][1],
                "snapshot_mark_minutes": (
                    None if label == "historical" else 360.0
                ),
                "stop_reason": (
                    "wall_limit" if label == "historical" else "running"
                ),
                "wall_s": 79348.0 if label == "historical" else 21600.0,
                "columns_journal": str(journal),
                "provenance": {
                    "git_commit": (
                        launcher.HISTORICAL_COMPARATOR_COMMIT
                        if label == "historical" else "a" * 40
                    ),
                    "instance_sha256": instance_sha,
                    "prices_sha256": prices_sha,
                    "args": {
                        "columns_per_iter": 30,
                        "rc_eps": 0.0001,
                    },
                },
            }))
            snapshots[label] = result
        return repo, snapshots

    def _args(self, snapshots):
        return Namespace(
            historical=snapshots["historical"],
            ca=snapshots["ca"],
            cs=snapshots["cs"],
            pa=snapshots["pa"],
            ps=snapshots["ps"],
            campaign="profile_dry_run",
            python=Path(sys.executable),
            solve_limit_s=120.0,
            repeat=3,
            mem_gb=64,
            job_hours=24,
            plan_out=None,
            approved_plan_sha256=None,
            submit=False,
        )

    def _valid_profile(self, job, commit, *, inject_failure=False):
        profiles = []
        for prefix in launcher.PREFIXES:
            methods = []
            for method_name in launcher.METHODS:
                repetitions = []
                for repetition in range(1, 4):
                    failed = (
                        inject_failure
                        and prefix == 5000
                        and method_name == "highs-ds"
                        and repetition == 2
                    )
                    repetitions.append({
                        "repetition": repetition,
                        "outcome": "error" if failed else "ok",
                        "error": "timeout" if failed else None,
                        "total_s": 1.0 + repetition / 10,
                        "backend_s": None if failed else 1.0,
                        "peak_rss_bytes": 100 + repetition * 10,
                        **({} if failed else {
                            "objective": 2900000.0,
                            "route_weight": 29.0,
                            "artificial_total": 0.0,
                            "max_row_violation": 1e-8,
                            "max_bound_violation": 0.0,
                        }),
                    })
                methods.append({
                    "method": method_name,
                    "outcome": "error" if any(
                        rep["outcome"] == "error" for rep in repetitions
                    ) else "ok",
                    "successful_repetitions": sum(
                        rep["outcome"] == "ok" for rep in repetitions
                    ),
                    "requested_repetitions": 3,
                    "repetitions": repetitions,
                    "timing": {
                        "total_min_s": 1.1,
                        "total_median_s": 1.2,
                        "total_max_s": 1.3,
                        "backend_min_s": 1.0,
                        "backend_median_s": 1.0,
                        "backend_max_s": 1.0,
                    },
                    "solution": {
                        "objective": 2900000.0,
                        "route_weight": 29.0,
                        "artificial_total": 0.0,
                        "max_row_violation": 1e-8,
                        "max_bound_violation": 0.0,
                    },
                })
            profiles.append({
                "prefix_columns": prefix,
                "available": True,
                "methods": methods,
            })
        hashes = {
            "result": job["job_spec"]["staged_result_sha256"],
            "journal": job["job_spec"]["staged_journal_sha256"],
            "instance": job["job_spec"]["staged_instance_sha256"],
            "prices": job["job_spec"]["staged_prices_sha256"],
        }
        return {
            "schema": "evsp-dr-frozen-pool-prefix-profile-v2",
            "source_unchanged": True,
            "source_hashes_before": hashes,
            "source_hashes_after": hashes,
            "requested_prefixes": launcher.PREFIXES,
            "methods": launcher.METHODS,
            "repeat": 3,
            "time_limit_s": job["job_spec"]["time_limit_s"],
            "profiles": profiles,
            "provenance": {
                "git_commit": commit,
                "git_dirty": False,
            },
        }

    def test_launcher_is_dry_run_with_unique_hash_bound_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, snapshots = self._campaign_fixture(Path(tmp))
            args = self._args(snapshots)
            args.plan_out = Path(tmp) / "approved-plan.json"
            identity = {
                "expected_commit": "a" * 40,
                "observed_commit": "a" * 40,
                "detached": True,
                "tracked_clean": True,
                "profile_core_commit": launcher.PROFILE_CORE_COMMIT,
            }
            output = io.StringIO()
            with (
                patch.object(launcher, "REPO_ROOT", repo),
                patch.object(
                    launcher,
                    "reviewed_checkout_identity",
                    return_value=identity,
                ),
                patch.object(
                    launcher,
                    "validated_python",
                    return_value=self._python_identity(),
                ),
                patch.object(
                    launcher,
                    "reviewed_worker_bytes",
                    return_value=b"#!/bin/bash\n",
                ),
                patch.object(
                    launcher.subprocess,
                    "run",
                    side_effect=AssertionError(
                        "dry-run must not invoke sbatch"
                    ),
                ),
                contextlib.redirect_stdout(output),
            ):
                manifest = launcher.launch(args)

            campaign_root = (
                repo / "src/results/exact_cg_profiles/profile_dry_run"
            )
            self.assertFalse(campaign_root.exists())
            self.assertTrue(args.plan_out.is_file())
            approval = json.loads(args.plan_out.read_text())
            self.assertEqual(
                launcher._approval_sha256(approval),
                manifest["approval_sha256"],
            )
            self.assertEqual(
                hashlib.sha256(args.plan_out.read_bytes()).hexdigest(),
                manifest["approval_sha256"],
            )
            self.assertFalse(manifest["submitted"])
            self.assertEqual(len(manifest["jobs"]), 5)
            outputs = [job["output"] for job in manifest["jobs"]]
            self.assertEqual(len(outputs), len(set(outputs)))
            self.assertEqual(
                [job["label"] for job in manifest["jobs"]],
                ["historical", "ca", "cs", "pa", "ps"],
            )
            expected_prefixes = {
                "historical": "PFhist-",
                "ca": "PFca-",
                "cs": "PFcs-",
                "pa": "PFpa-",
                "ps": "PFps-",
            }
            for job in manifest["jobs"]:
                self.assertTrue(
                    job["job_name"].startswith(
                        expected_prefixes[job["label"]]
                    )
                )
                self.assertLessEqual(len(job["job_name"]), 15)
                spec = job["job_spec"]
                self.assertEqual(
                    spec["source_hashes"], job["source_hashes"]
                )
                self.assertEqual(
                    spec["staged_journal_sha256"],
                    job["source_hashes"]["journal"],
                )
                command = job["command"]
                self.assertIn("--no-requeue", command)
                self.assertIn("--cpus-per-task=1", command)
                self.assertIn("--mem=64G", command)
                export = next(
                    item for item in command if item.startswith("--export=")
                )
                self.assertFalse(export.startswith("--export=ALL"))
                self.assertIn("EVSP_EXPECTED_COMMIT=" + "a" * 40, export)
                self.assertIn("EVSP_PROFILE_PYTHON=", export)
                self.assertIn("EVSP_PROFILE_ENV_SHA256=" + "f" * 64, export)
            historical = manifest["jobs"][0]
            staged_historical = json.loads(
                historical["staged_result_bytes"]
            )
            self.assertEqual(staged_historical["master_sense"], "cover")
            self.assertEqual(
                staged_historical["initial_pool"], "artificial"
            )
            self.assertIn("[dry-run]", output.getvalue())

    def test_python_environment_failure_is_fatal(self):
        with self.assertRaisesRegex(SystemExit, "validation failed"):
            launcher.validated_python(Path("/bin/false"))

    def test_launcher_rejects_duplicate_or_mislabeled_factorial_pools(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, snapshots = self._campaign_fixture(Path(tmp))
            args = self._args(snapshots)
            identity = {
                "expected_commit": "a" * 40,
                "observed_commit": "a" * 40,
                "detached": True,
                "tracked_clean": True,
                "profile_core_commit": launcher.PROFILE_CORE_COMMIT,
            }

            def run_with_patches():
                stack = contextlib.ExitStack()
                stack.enter_context(patch.object(launcher, "REPO_ROOT", repo))
                stack.enter_context(patch.object(
                    launcher,
                    "reviewed_checkout_identity",
                    return_value=identity,
                ))
                stack.enter_context(patch.object(
                    launcher,
                    "validated_python",
                    return_value=self._python_identity(),
                ))
                stack.enter_context(patch.object(
                    launcher,
                    "reviewed_worker_bytes",
                    return_value=b"#!/bin/bash\n",
                ))
                return stack

            args.ps = args.pa
            with (
                run_with_patches(),
                self.assertRaisesRegex(SystemExit, "must be distinct"),
            ):
                launcher.launch(args)

            args.ps = snapshots["ps"]
            pa_status = json.loads(snapshots["pa"].read_text())
            ps_status = json.loads(snapshots["ps"].read_text())
            pa_journal = Path(pa_status["columns_journal"])
            ps_journal = Path(ps_status["columns_journal"])
            ps_journal.write_bytes(pa_journal.read_bytes())
            with (
                run_with_patches(),
                self.assertRaisesRegex(SystemExit, "journal bytes must be distinct"),
            ):
                launcher.launch(args)

            ps_journal.write_text(
                json.dumps({"trips": [1], "cost": 100004.0}) + "\n"
            )
            historical_status = json.loads(snapshots["historical"].read_text())
            historical_status["master_sense"] = "partition"
            historical_status["snapshot_mark_minutes"] = 5.0
            snapshots["historical"].write_text(json.dumps(historical_status))
            with (
                run_with_patches(),
                self.assertRaisesRegex(SystemExit, "legacy covering"),
            ):
                launcher.launch(args)
            historical_status["master_sense"] = None
            historical_status["snapshot_mark_minutes"] = None
            snapshots["historical"].write_text(json.dumps(historical_status))

            ca_status = json.loads(snapshots["ca"].read_text())
            ca_status["initial_pool"] = "singletons"
            snapshots["ca"].write_text(json.dumps(ca_status))
            with (
                run_with_patches(),
                self.assertRaisesRegex(SystemExit, "expected initial_pool"),
            ):
                launcher.launch(args)

    def test_submit_requires_exact_reviewed_plan_and_retry_is_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, snapshots = self._campaign_fixture(Path(tmp))
            args = self._args(snapshots)
            args.campaign = "approved_submit"
            identity = {
                "expected_commit": "a" * 40,
                "observed_commit": "a" * 40,
                "detached": True,
                "tracked_clean": True,
                "profile_core_commit": launcher.PROFILE_CORE_COMMIT,
            }

            def common_patches():
                return (
                    patch.object(launcher, "REPO_ROOT", repo),
                    patch.object(
                        launcher,
                        "reviewed_checkout_identity",
                        return_value=identity,
                    ),
                    patch.object(
                        launcher,
                        "validated_python",
                        return_value=self._python_identity(),
                    ),
                    patch.object(
                        launcher,
                        "reviewed_worker_bytes",
                        return_value=b"#!/bin/bash\n",
                    ),
                )

            with contextlib.ExitStack() as stack:
                for manager in common_patches():
                    stack.enter_context(manager)
                stack.enter_context(patch.object(
                    launcher.subprocess,
                    "run",
                    side_effect=AssertionError("dry-run invoked sbatch"),
                ))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                planned = launcher.launch(args)

            args.submit = True
            args.approved_plan_sha256 = "0" * 64
            with contextlib.ExitStack() as stack:
                for manager in common_patches():
                    stack.enter_context(manager)
                stack.enter_context(patch.object(
                    launcher.subprocess,
                    "run",
                    side_effect=AssertionError("wrong plan invoked sbatch"),
                ))
                stack.enter_context(self.assertRaisesRegex(
                    SystemExit, "differs from approved"
                ))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                launcher.launch(args)

            args.approved_plan_sha256 = planned["approval_sha256"]
            ca_status = json.loads(snapshots["ca"].read_text())
            ca_journal = Path(ca_status["columns_journal"])
            original_ca_journal = ca_journal.read_bytes()
            ca_journal.write_text(
                json.dumps({"trips": [1], "cost": 123456.0}) + "\n"
            )
            with contextlib.ExitStack() as stack:
                for manager in common_patches():
                    stack.enter_context(manager)
                stack.enter_context(patch.object(
                    launcher.subprocess,
                    "run",
                    side_effect=AssertionError(
                        "changed source invoked sbatch"
                    ),
                ))
                stack.enter_context(self.assertRaisesRegex(
                    SystemExit, "differs from approved"
                ))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                launcher.launch(args)
            ca_journal.write_bytes(original_ca_journal)

            submit_calls = 0

            def accept_sbatch(*_args, **_kwargs):
                nonlocal submit_calls
                manifest_path = (
                    repo / "src/results/exact_cg_profiles/"
                    "approved_submit/campaign.json"
                )
                persisted_attempt = json.loads(manifest_path.read_text())
                current = persisted_attempt["jobs"][submit_calls]
                self.assertEqual(current["submission_state"], "attempting")
                self.assertIsNone(current["job_id"])
                result = __import__("subprocess").CompletedProcess(
                    ["sbatch"], 0,
                    stdout=f"{1000 + submit_calls}\n", stderr="",
                )
                submit_calls += 1
                return result

            with contextlib.ExitStack() as stack:
                for manager in common_patches():
                    stack.enter_context(manager)
                stack.enter_context(patch.object(
                    launcher.subprocess,
                    "run",
                    side_effect=accept_sbatch,
                ))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                submitted = launcher.launch(args)

            self.assertTrue(submitted["submitted"])
            self.assertEqual(
                [job["job_id"] for job in submitted["jobs"]],
                ["1000", "1001", "1002", "1003", "1004"],
            )
            campaign_root = (
                repo / "src/results/exact_cg_profiles/approved_submit"
            )
            persisted = json.loads(
                (campaign_root / "campaign.json").read_text()
            )
            self.assertTrue(persisted["submitted"])

            with contextlib.ExitStack() as stack:
                for manager in common_patches():
                    stack.enter_context(manager)
                stack.enter_context(self.assertRaisesRegex(
                    SystemExit, "campaign already exists"
                ))
                launcher.launch(args)

    def test_worker_declares_nonresumable_single_thread_invariants(self):
        text = (
            REPO_ROOT / "src/submit_exact_cg_profile.sub"
        ).read_text()
        self.assertIn("set -euo pipefail", text)
        self.assertIn("#SBATCH --no-requeue", text)
        self.assertNotIn("#SBATCH --requeue", text)
        self.assertIn("#SBATCH --cpus-per-task=1", text)
        self.assertIn("OMP_NUM_THREADS=1", text)
        self.assertIn("OPENBLAS_NUM_THREADS=1", text)
        self.assertIn(launcher.PROFILE_CORE_COMMIT, text)
        self.assertIn("install_exact_cg_profile_input.py", text)
        self.assertIn("exact_cg_profile_environment.py", text)
        self.assertIn("EVSP_PROFILE_ENV_SHA256", text)
        self.assertNotIn("--phase-telemetry", text)

    def test_input_installer_rejects_symlink_escape_and_hash_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            source.write_text("expected\n")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            data = root / "data"
            outside = root / "outside"
            outside.mkdir()
            data.mkdir()
            (data / "nested").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "unsafe data parent"):
                input_installer.install(
                    source, data, Path("nested/input.csv"), expected
                )

            (data / "nested").unlink()
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=5) as executor:
                installed_paths = list(executor.map(
                    lambda _index: input_installer.install(
                        source,
                        data,
                        Path("nested/input.csv"),
                        expected,
                    ),
                    range(5),
                ))
            installed = installed_paths[0]
            self.assertTrue(all(path == installed for path in installed_paths))
            self.assertEqual(installed.read_text(), "expected\n")

            original_stat = input_installer.os.stat
            swapped_existing_parent = False

            def swap_existing_parent(path, *args, **kwargs):
                nonlocal swapped_existing_parent
                if (not swapped_existing_parent
                        and Path(path) == data / "nested"):
                    swapped_existing_parent = True
                    (data / "nested").rename(data / "nested-existing-safe")
                    (data / "nested").symlink_to(
                        outside, target_is_directory=True
                    )
                return original_stat(path, *args, **kwargs)

            with (
                patch.object(
                    input_installer.os,
                    "stat",
                    side_effect=swap_existing_parent,
                ),
                self.assertRaisesRegex(
                    ValueError, "data parent changed"
                ),
            ):
                input_installer.install(
                    source, data, Path("nested/input.csv"), expected
                )
            self.assertFalse((outside / "input.csv").exists())
            (data / "nested").unlink()
            (data / "nested-existing-safe").rename(data / "nested")

            changed = root / "changed.csv"
            changed.write_text("different\n")
            changed_sha = hashlib.sha256(changed.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                ValueError, "existing data hash mismatch"
            ):
                input_installer.install(
                    changed, data, Path("nested/input.csv"), changed_sha
                )

            race_source = root / "race.csv"
            race_source.write_text("race\n")
            race_sha = hashlib.sha256(race_source.read_bytes()).hexdigest()
            (data / "race").mkdir()
            original_link = input_installer.os.link

            def replace_parent_then_link(*args, **kwargs):
                (data / "race").rename(data / "race-safe")
                (data / "race").symlink_to(
                    outside, target_is_directory=True
                )
                return original_link(*args, **kwargs)

            with (
                patch.object(
                    input_installer.os,
                    "link",
                    side_effect=replace_parent_then_link,
                ),
                self.assertRaisesRegex(
                    ValueError, "data parent changed"
                ),
            ):
                input_installer.install(
                    race_source,
                    data,
                    Path("race/input.csv"),
                    race_sha,
                )
            self.assertFalse((outside / "input.csv").exists())
            self.assertTrue((data / "race-safe/input.csv").is_file())

    def test_summary_and_monitor_parse_profile_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            root.mkdir()
            commit = "a" * 40
            jobs = []
            for label in ("historical", "ca", "cs", "pa", "ps"):
                output = root / f"{label}.profile.json"
                job = {
                    "label": label,
                    "job_id": str(123 + len(jobs)),
                    "job_name": f"PF{label}-abcdef",
                    "slurm_comment": f"PROFILE:{label}:abcdef",
                    "submission_state": "submitted",
                    "output": str(output),
                    "job_spec": {
                        "staged_result_sha256": f"{label}-result",
                        "staged_journal_sha256": f"{label}-journal",
                        "staged_instance_sha256": "instance",
                        "staged_prices_sha256": "prices",
                        "repeat": 3,
                        "time_limit_s": 120.0,
                    },
                }
                output.write_text(json.dumps(self._valid_profile(
                    job,
                    commit,
                    inject_failure=(label == "historical"),
                )))
                jobs.append(job)
            manifest = {
                "schema": "evsp-dr-exact-cg-profile-campaign-v1",
                "campaign": "test",
                "jobs": jobs,
                "checkout_identity": {"expected_commit": commit},
                "profile_core_commit": launcher.PROFILE_CORE_COMMIT,
            }
            (root / "campaign.json").write_text(json.dumps(manifest))

            rows = summarizer.summarize(root)
            self.assertEqual(len(rows), 75)
            row = next(
                row for row in rows
                if row["label"] == "historical"
                and row["prefix"] == 5000
                and row["method"] == "highs-ds"
            )
            self.assertEqual(row["prefix"], 5000)
            self.assertEqual(row["method"], "highs-ds")
            self.assertEqual(row["median_total_s"], 1.2)
            self.assertEqual(row["objective"], 2900000.0)
            self.assertEqual(row["route_weight"], 29.0)
            self.assertEqual(row["artificials"], 0.0)
            self.assertEqual(row["peak_rss_bytes"], 130)
            self.assertEqual(row["failure_count"], 1)
            self.assertIn("timeout", row["failures"])

            monitored = monitor.monitor(root, query_slurm=False)
            self.assertTrue(monitored[0]["output_exists"])
            self.assertTrue(monitored[0]["artifact"]["valid_json"])
            self.assertTrue(monitored[0]["artifact"]["valid_profile"])

            original_output = jobs[0]["output"]
            jobs[0]["output"] = str(root / "not-yet-created.json")
            (root / "campaign.json").write_text(json.dumps(manifest))
            with (
                patch.object(
                    monitor,
                    "_live_queue",
                    return_value={
                        "123": {
                            "job_name": jobs[0]["job_name"],
                            "state": "PENDING",
                            "elapsed": "0:00",
                            "reason_or_node": "Resources",
                            "comment": jobs[0]["slurm_comment"],
                        }
                    },
                ),
                patch.object(
                    monitor,
                    "_accounting",
                    return_value={
                        "123": {
                            "job_name": jobs[0]["job_name"],
                            "state": "COMPLETED",
                            "elapsed": "1:00",
                            "exit_code": "0:0",
                            "max_rss": "1G",
                            "comment": jobs[0]["slurm_comment"],
                        }
                    },
                ),
            ):
                precedence = monitor.monitor(root, query_slurm=True)[0]
                jobs[0]["output"] = original_output
                (root / "campaign.json").write_text(json.dumps(manifest))
                precedence_with_output = monitor.monitor(
                    root, query_slurm=True
                )[0]
            self.assertEqual(precedence["slurm"]["state"], "PENDING")
            self.assertEqual(
                precedence["slurm"]["state_source"], "squeue"
            )
            self.assertEqual(
                precedence["slurm"]["accounting_state"], "COMPLETED"
            )
            self.assertTrue(precedence["state_disagreement"])
            self.assertTrue(
                precedence["possible_stale_or_recycled_job_id"]
            )
            self.assertTrue(
                precedence_with_output[
                    "possible_stale_or_recycled_job_id"
                ]
            )
            jobs[0]["output"] = original_output
            (root / "campaign.json").write_text(json.dumps(manifest))

            jobs[0]["output"] = str(root / "invalid.json")
            Path(jobs[0]["output"]).write_text("{}")
            (root / "campaign.json").write_text(json.dumps(manifest))
            invalid_rows = summarizer.summarize(root)
            invalid = next(
                row for row in invalid_rows
                if row["label"] == "historical"
            )
            self.assertEqual(invalid["outcome"], "invalid_profile")
            self.assertGreater(invalid["failure_count"], 0)

            Path(jobs[0]["output"]).write_text("[]")
            list_rows = summarizer.summarize(root)
            listed = next(
                row for row in list_rows if row["label"] == "historical"
            )
            self.assertEqual(listed["outcome"], "invalid_profile")

            (root / "campaign.json").write_text("[]")
            manifest_rows = summarizer.summarize(root)
            self.assertEqual(manifest_rows[0]["outcome"], "invalid_manifest")
            monitored = monitor.monitor(root, query_slurm=False)
            self.assertEqual(
                monitored[0]["submission_state"], "invalid_manifest"
            )

            (root / "campaign.json").write_text(json.dumps({
                "schema": "evsp-dr-exact-cg-profile-campaign-v1",
                "jobs": [{"label": label} for label in (
                    "historical", "ca", "cs", "pa", "ps"
                )],
                "checkout_identity": {"expected_commit": "a" * 40},
            }))
            monitored = monitor.monitor(root, query_slurm=False)
            self.assertEqual(
                monitored[0]["submission_state"], "invalid_manifest"
            )

    def test_artifact_validation_rejects_duplicates_but_allows_unavailable(self):
        commit = "a" * 40
        job = {
            "label": "historical",
            "job_spec": {
                "staged_result_sha256": "result",
                "staged_journal_sha256": "journal",
                "staged_instance_sha256": "instance",
                "staged_prices_sha256": "prices",
                "repeat": 3,
                "time_limit_s": 120.0,
            },
        }
        manifest = {
            "checkout_identity": {"expected_commit": commit},
        }
        payload = self._valid_profile(job, commit)
        payload["profiles"][0] = {
            "prefix_columns": 1000,
            "available": False,
            "reason": "pool has fewer unique incidences",
        }
        self.assertEqual(
            result_validation.validate_profile_payload(
                payload, job, manifest
            ),
            [],
        )

        payload["profiles"].append(dict(payload["profiles"][1]))
        errors = result_validation.validate_profile_payload(
            payload, job, manifest
        )
        self.assertTrue(any("prefix" in error for error in errors))

        payload = self._valid_profile(job, commit)
        payload["profiles"][0]["methods"].append(
            dict(payload["profiles"][0]["methods"][0])
        )
        errors = result_validation.validate_profile_payload(
            payload, job, manifest
        )
        self.assertTrue(any("method coverage" in error for error in errors))

        payload = self._valid_profile(job, commit)
        payload["requested_prefixes"] = 123
        errors = result_validation.validate_profile_payload(
            payload, job, manifest
        )
        self.assertTrue(any("prefix set" in error for error in errors))

        payload = self._valid_profile(job, commit)
        method = payload["profiles"][0]["methods"][0]
        method["repetitions"][1]["repetition"] = 1
        method["successful_repetitions"] = 999
        errors = result_validation.validate_profile_payload(
            payload, job, manifest
        )
        self.assertTrue(any("repetition IDs" in error for error in errors))
        self.assertTrue(any("success count" in error for error in errors))

        payload = self._valid_profile(job, commit)
        payload["profiles"][0]["methods"][0]["timing"] = None
        payload["profiles"][0]["methods"][0]["solution"] = None
        errors = result_validation.validate_profile_payload(
            payload, job, manifest
        )
        self.assertTrue(any("timing is incomplete" in error for error in errors))
        self.assertTrue(any("solution is incomplete" in error for error in errors))

        payload = self._valid_profile(job, commit)
        payload["profiles"][0]["methods"][0]["repetitions"][0][
            "total_s"
        ] = 10 ** 10000
        errors = result_validation.validate_profile_payload(
            payload, job, manifest
        )
        self.assertTrue(any("repetition IDs" in error for error in errors))

        payload = self._valid_profile(job, commit)
        timing = payload["profiles"][0]["methods"][0]["timing"]
        timing["total_min_s"] = 10.0
        timing["total_median_s"] = 1.0
        timing["total_max_s"] = 0.0
        errors = result_validation.validate_profile_payload(
            payload, job, manifest
        )
        self.assertTrue(any("timing summary" in error for error in errors))
        self.assertTrue(any("timing order" in error for error in errors))

        payload = self._valid_profile(job, commit)
        method = payload["profiles"][0]["methods"][0]
        for repetition in method["repetitions"]:
            repetition["outcome"] = "error"
            repetition["error"] = "synthetic"
            repetition.pop("backend_s", None)
            for field in (
                "objective", "route_weight", "artificial_total",
                "max_row_violation", "max_bound_violation",
            ):
                repetition.pop(field, None)
        method["successful_repetitions"] = 0
        method["outcome"] = "error"
        errors = result_validation.validate_profile_payload(
            payload, job, manifest
        )
        self.assertTrue(any("all-error timing" in error for error in errors))
        self.assertTrue(any("all-error solution" in error for error in errors))

        payload = self._valid_profile(job, commit)
        divergent = payload["profiles"][0]["methods"][1]
        divergent["solution"]["objective"] += 10.0
        for repetition in divergent["repetitions"]:
            repetition["objective"] += 10.0
        errors = result_validation.validate_profile_payload(
            payload, job, manifest
        )
        self.assertTrue(any("method solutions disagree" in error for error in errors))

        payload = self._valid_profile(job, commit)
        payload["provenance"] = []
        errors = result_validation.validate_profile_payload(
            payload, job, manifest
        )
        self.assertTrue(any("provenance is not an object" in error for error in errors))

        payload = self._valid_profile(job, commit)
        repetition = payload["profiles"][0]["methods"][0]["repetitions"][0]
        repetition["repetition"] = "1"
        repetition["outcome"] = "mystery"
        payload["time_limit_s"] = 999.0
        errors = result_validation.validate_profile_payload(
            payload, job, manifest
        )
        self.assertTrue(any("repetition IDs" in error for error in errors))
        self.assertTrue(any("time limit" in error for error in errors))

        payload = self._valid_profile(job, commit)
        method = payload["profiles"][0]["methods"][0]
        method["repetitions"][0]["total_s"] = -5
        method["repetitions"][0]["backend_s"] = -2
        method["repetitions"][0]["peak_rss_bytes"] = -1
        method["timing"] = {
            key: True for key in method["timing"]
        }
        method["solution"] = {
            key: True for key in method["solution"]
        }
        errors = result_validation.validate_profile_payload(
            payload, job, manifest
        )
        self.assertTrue(any("repetition IDs" in error for error in errors))
        self.assertTrue(any("timing is incomplete" in error for error in errors))
        self.assertTrue(any("solution is incomplete" in error for error in errors))

        malformed_manifest = {
            "schema": "evsp-dr-exact-cg-profile-campaign-v1",
            "jobs": [{"label": []} for _ in range(5)],
            "checkout_identity": [],
        }
        errors = result_validation.validate_campaign_manifest(
            malformed_manifest
        )
        self.assertTrue(errors)

        malformed_job = {
            **job,
            "job_spec": {**job["job_spec"], "repeat": 2.5},
        }
        payload = self._valid_profile(job, commit)
        payload["profiles"][0]["methods"][0]["repetitions"][0][
            "total_s"
        ] = "not-a-number"
        errors = result_validation.validate_profile_payload(
            payload, malformed_job, manifest
        )
        self.assertTrue(any("repeat" in error for error in errors))
        self.assertTrue(any("repetition IDs" in error for error in errors))

    def test_archive_records_commit_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            root.mkdir()
            logs = Path(tmp) / "logs"
            logs.mkdir()
            commit = "b" * 40
            worker = root / "input/submit_exact_cg_profile.sub"
            worker.parent.mkdir()
            worker.write_text("#!/bin/bash\n")
            worker_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
            jobs = []
            for label in ("historical", "ca", "cs", "pa", "ps"):
                output_path = root / f"{label}.json"
                cell = root / "input" / label
                cell.mkdir()
                staged_files = {}
                for kind in ("result", "journal", "instance", "prices"):
                    path = cell / f"{kind}.dat"
                    path.write_text(f"{label}-{kind}\n")
                    staged_files[kind] = (
                        path,
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                job_spec = {
                    "staged_result": str(staged_files["result"][0]),
                    "staged_result_sha256": staged_files["result"][1],
                    "staged_journal": str(staged_files["journal"][0]),
                    "staged_journal_sha256": staged_files["journal"][1],
                    "staged_instance": str(staged_files["instance"][0]),
                    "staged_instance_sha256": staged_files["instance"][1],
                    "staged_prices": str(staged_files["prices"][0]),
                    "staged_prices_sha256": staged_files["prices"][1],
                    "repeat": 3,
                    "time_limit_s": 120.0,
                }
                job_spec_path = cell / "job.json"
                job_spec_path.write_text(json.dumps(job_spec))
                job = {
                    "label": label,
                    "job_id": str(len(jobs) + 1),
                    "job_name": f"PF{label}-test-aa",
                    "submission_state": "submitted",
                    "output": str(output_path),
                    "job_spec": job_spec,
                    "job_spec_path": str(job_spec_path),
                    "job_spec_sha256": hashlib.sha256(
                        job_spec_path.read_bytes()
                    ).hexdigest(),
                }
                output_path.write_text(json.dumps(
                    self._valid_profile(job, commit)
                ))
                jobs.append(job)
                (logs / f"{job['job_name']}_{job['job_id']}.out").write_text(
                    f"[PROFILE] DONE label={label} output={output_path}\n"
                )
                (logs / f"{job['job_name']}_{job['job_id']}.err").write_text(
                    ""
                )
            manifest = {
                "schema": "evsp-dr-exact-cg-profile-campaign-v1",
                "campaign": "archive-test",
                "submitted": False,
                "jobs": jobs,
                "checkout_identity": {
                    "expected_commit": commit,
                    "observed_commit": commit,
                    "detached": True,
                    "tracked_clean": True,
                    "profile_core_commit": launcher.PROFILE_CORE_COMMIT,
                },
                "profile_core_commit": launcher.PROFILE_CORE_COMMIT,
                "log_root": str(logs),
                "worker": str(worker),
                "worker_sha256": worker_sha,
                "python": self._python_identity(),
                "runtime_environment": {
                    "HOME": "/home/test",
                    "USER": "test",
                    "PATH": "/usr/bin:/bin",
                },
                "resources": {
                    "partition": "default_partition",
                    "cpus": 1,
                    "mem_gb": 64,
                    "job_hours": 24,
                    "blas_openmp_threads": 1,
                    "requeue": False,
                },
                "profiler": {
                    "prefixes": launcher.PREFIXES,
                    "methods": launcher.METHODS,
                    "repeat": 3,
                    "per_solve_time_limit_s": 120.0,
                    "phase_telemetry": False,
                },
            }
            manifest["approval_sha256"] = launcher._approval_sha256(
                launcher._approval_payload(manifest)
            )
            manifest["jobs"][0]["submission_state"] = "submitted_reconciled"
            manifest["jobs"][0]["reconciled_slurm_state"] = "COMPLETED"
            manifest["jobs"][0]["submission_error"] = "historical stale note"
            (root / "campaign.json").write_text(json.dumps(manifest))
            output = Path(tmp) / "archive.tar.gz"

            with self.assertRaisesRegex(ValueError, "not fully submitted"):
                archiver.archive(root, output)

            manifest["submitted"] = True
            (root / "campaign.json").write_text(json.dumps(manifest))
            missing_stderr = logs / f"{jobs[0]['job_name']}_{jobs[0]['job_id']}.err"
            missing_stderr.unlink()
            with self.assertRaisesRegex(ValueError, "logs are incomplete"):
                archiver.archive(root, output)
            missing_stderr.write_text("")

            original_sha256 = archiver.sha256_file
            mutated = False

            def add_late_file(path):
                nonlocal mutated
                digest = original_sha256(path)
                if not mutated:
                    mutated = True
                    (logs / "late.out").write_text("late\n")
                return digest

            with (
                patch.object(
                    archiver, "sha256_file", side_effect=add_late_file
                ),
                self.assertRaisesRegex(RuntimeError, "changed"),
            ):
                archiver.archive(root, output)
            (logs / "late.out").unlink()

            record = archiver.archive(root, output)

            self.assertTrue(output.is_file())
            self.assertEqual(record["expected_commit"], commit)
            self.assertIn("campaign/campaign.json", record["files"])
            self.assertIn(
                "logs/PFhistorical-test-aa_1.out", record["files"]
            )
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(),
                record["archive_sha256"],
            )

    def test_reconcile_recovers_one_accepted_unrecorded_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "campaign": "reconcile-test",
                "created_at": "2026-08-14T12:00:00+00:00",
                "submitted": False,
                "jobs": [{
                    "label": "ca",
                    "job_name": "PFca-abcd-aa",
                    "slurm_comment": "EVSPPF:test:ca:abcdef",
                    "job_id": None,
                    "submission_state": "attempting",
                }],
            }
            path = root / "campaign.json"
            path.write_text(json.dumps(manifest))
            match = {
                "job_id": "12345",
                "job_name": "PFca-abcd-aa",
                "state": "RUNNING",
                "submit": "time",
                "start": "time",
                "elapsed": "00:01",
                "exit_code": "0:0",
                "comment": "EVSPPF:test:ca:abcdef",
            }
            with patch.object(
                reconciler, "_query", return_value=[match]
            ):
                preview = reconciler.reconcile(root, apply=False)
                applied = reconciler.reconcile(root, apply=True)

            self.assertEqual(preview["recovered"][0]["job_id"], "12345")
            self.assertEqual(applied["recovered"][0]["state"], "RUNNING")
            persisted = json.loads(path.read_text())
            self.assertEqual(persisted["jobs"][0]["job_id"], "12345")
            self.assertEqual(
                persisted["jobs"][0]["submission_state"],
                "submitted_reconciled",
            )
            self.assertTrue(persisted["submitted"])

    def test_reconcile_never_marks_partial_campaign_safe_to_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = [{
                "label": "historical",
                "job_name": "PFhist-abcd-aa",
                "slurm_comment": "comment",
                "job_id": "123",
                "submission_state": "submitted",
            }]
            jobs.extend({
                "label": label,
                "job_name": f"PF{label}-abcd-aa",
                "slurm_comment": f"comment-{label}",
                "job_id": None,
                "submission_state": "planned",
            } for label in ("ca", "cs", "pa", "ps"))
            (root / "campaign.json").write_text(json.dumps({
                "campaign": "partial",
                "created_at": "2026-08-14T12:00:00+00:00",
                "jobs": jobs,
            }))

            result = reconciler.reconcile(root, apply=False)

            self.assertFalse(result["safe_to_retry"])
            self.assertEqual(result["recorded"][0]["job_id"], "123")


if __name__ == "__main__":
    unittest.main()

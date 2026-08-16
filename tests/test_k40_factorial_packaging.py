import contextlib
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

import archive_k40_factorial as archiver  # noqa: E402
import launch_k40_factorial_mip_screen as mip_launcher  # noqa: E402
import summarize_k40_factorial as summarizer  # noqa: E402
import k40_factorial_artifacts as artifacts  # noqa: E402
import monitor_k40_factorial_mip_screen as mip_monitor  # noqa: E402
import prepare_k40_factorial_giro_start as seed_preparer  # noqa: E402
import reconcile_k40_factorial_mip_screen as mip_reconcile  # noqa: E402
from k40_factorial_artifacts import (  # noqa: E402
    ARMS,
    FACTORIAL_COMMIT,
    HISTORICAL_COMMIT,
    HISTORICAL_WEIGHT,
    INSTANCE_SHA256,
    MARKS,
    PRICES_SHA256,
)


class K40FactorialPackagingTests(unittest.TestCase):
    def setUp(self):
        self.instance_bytes = b"synthetic-k40-instance"
        self.prices_bytes = b"synthetic-flat-prices"
        self.instance_sha = hashlib.sha256(self.instance_bytes).hexdigest()
        self.prices_sha = hashlib.sha256(self.prices_bytes).hexdigest()
        self.enterContext(
            patch.object(artifacts, "INSTANCE_SHA256", self.instance_sha)
        )
        self.enterContext(
            patch.object(artifacts, "PRICES_SHA256", self.prices_sha)
        )

    def _status(
        self,
        *,
        journal,
        arm,
        mark,
        terminal=False,
        replicate_offset=0.0,
        csv_name="Practice_Custom_DutyUnion_k40_r1.csv",
        journal_cost=100000.0,
    ):
        sense, initial = ARMS[arm]
        artificials = 5.0 if arm == "PA" and mark == 1320 else 0.0
        weight = (
            11.58 if artificials else 40.0 + MARKS.index(mark) / 10
            + replicate_offset
        )
        objective = weight * journal_cost + artificials * 500000.0
        return {
            "csv": f"duty_unions_big/{csv_name}",
            "prices_csv": "hourly_prices_flat.csv",
            "soc_step": 15.0,
            "block_min": 10,
            "g_kwh": 300.0,
            "charge_kw": 300.0,
            "min_soc_frac": 0.0,
            "master_sense": sense,
            "initial_pool": initial,
            "trip_ids": list(range(1, 948)),
            "iterations": mark,
            "columns": 1,
            "wall_s": mark * 60.0 + replicate_offset * 60.0,
            "stop_reason": "wall_limit" if terminal else f"snapshot_m{mark}",
            "snapshot_mark_minutes": None if terminal else float(mark),
            "certified_rc_optimal": (
                arm == "CA" and mark == 1440 and replicate_offset == 0
            ),
            "final": {
                "lp_obj": objective,
                "route_weight": weight,
                "artificials": artificials,
                "min_rc": -100.0 + mark / 100.0,
            },
            "final_lp": {
                "objective": objective,
                "route_weight": weight,
                "artificial_total": artificials,
                "positive_routes": [{
                    "trips": [1],
                    "value": weight,
                    "cost": journal_cost,
                }],
            },
            "columns_journal": str(journal),
            "provenance": {
                "git_commit": FACTORIAL_COMMIT,
                "git_branch": "",
                "git_dirty": False,
                "instance_sha256": self.instance_sha,
                "prices_sha256": self.prices_sha,
                "rc_eps": 0.0001,
                "args": {
                    "csv": f"duty_unions_big/{csv_name}",
                    "prices_csv": "hourly_prices_flat.csv",
                    "soc_step": 15.0,
                    "block_min": 10,
                    "g_kwh": 300.0,
                    "charge_kw": 300.0,
                    "min_soc_frac": 0.0,
                    "master_sense": sense,
                    "initial_pool": initial,
                    "columns_per_iter": 30,
                    "rc_eps": 0.0001,
                    "max_iters": 200000,
                    "wall_limit_s": 90000.0,
                    "checkpoint_every": 25,
                    "snapshot_at_minutes": "60,180,360,720,1320,1440",
                    "resume": True,
                },
            },
        }

    def _campaign(
        self,
        repo: Path,
        name: str,
        *,
        replicate_offset=0.0,
        job_base=100,
        csv_name="Practice_Custom_DutyUnion_k40_r1.csv",
    ):
        root = repo / "src/results/k40_factorial" / name
        root.mkdir(parents=True)
        launch = [
            "role\tjob_id\tjob_name\tmaster_sense\tinitial_pool",
            f"prep\t{job_base}\tK40-PREP\t-\t-",
        ]
        for index, (arm, (sense, initial)) in enumerate(ARMS.items(), start=2):
            launch.append(
                f"arm\t{job_base + index - 1}\tK40-{arm}24\t"
                f"{sense}\t{initial}"
            )
        (root / "launch.tsv").write_text("\n".join(launch) + "\n")
        (root / "prep_attestation.tsv").write_text(
            f"git_commit\t{FACTORIAL_COMMIT}\n"
            f"instance_sha256\t{self.instance_sha}\n"
            f"prices_sha256\t{self.prices_sha}\n"
            "python\t3.12.test\n"
        )
        (root / "input_manifest.json").write_text(json.dumps({
            "seed": 20260803,
            "source": "Par_VehicleDetails_Updated.csv",
            "unions": [{
                "csv": csv_name,
                "k": 40,
                "duties": [f"duty-{index}" for index in range(40)],
                "trips": 947,
                "peak_concurrency_lb": 40,
                "sha256": self.instance_sha,
            }],
        }))
        stem_prefix = (
            "k40r1_flat" if "_r1.csv" in csv_name else "k40r2_flat"
        )
        for arm in ARMS:
            for mark in MARKS:
                status_path = root / (
                    f"{stem_prefix}_{arm}.m{mark}.snapshot.json"
                )
                journal = Path(str(status_path) + ".columns.jsonl")
                journal_cost = 100000.0 + mark
                journal.write_text(json.dumps({
                    "trips": [1], "cost": journal_cost,
                }) + "\n")
                status_path.write_text(json.dumps(self._status(
                    journal=journal,
                    arm=arm,
                    mark=mark,
                    replicate_offset=replicate_offset,
                    csv_name=csv_name,
                    journal_cost=journal_cost,
                )))
            terminal = root / f"{stem_prefix}_{arm}.json"
            journal = Path(str(terminal) + ".columns.jsonl")
            journal_cost = 200000.0
            journal.write_text(json.dumps({
                "trips": [1], "cost": journal_cost,
            }) + "\n")
            terminal.write_text(json.dumps(self._status(
                journal=journal,
                arm=arm,
                mark=1440,
                terminal=True,
                replicate_offset=replicate_offset,
                csv_name=csv_name,
                journal_cost=journal_cost,
            )))
            terminal_status = json.loads(terminal.read_text())
            final = terminal_status["final"]
            Path(str(terminal) + ".iters.csv").write_text(
                "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,"
                f"pool_columns\n1,1440,{final['lp_obj']:.6f},"
                f"{final['route_weight']:.9f},{final['artificials']:.6f},"
                f"{final['min_rc']:.6f},1\n"
            )
            (root / f"{stem_prefix}_{arm}.allocations.tsv").write_text(
                "utc\tjob_id\trestart\thost\tcpu_model\t"
                "instance_sha256\tprices_sha256\n"
                f"now\t{job_base + list(ARMS).index(arm) + 1}\t0\t"
                f"compute01\tcpu\t{self.instance_sha}\t{self.prices_sha}\n"
            )
        logs = repo / "src/cluster_logs/k40_factorial" / name
        logs.mkdir(parents=True)
        (logs / f"K40-PREP_{job_base}.out").write_text(
            "[K40-PREP] READY\n"
        )
        (logs / f"K40-PREP_{job_base}.err").write_text("")
        for arm in ARMS:
            job_id = job_base + list(ARMS).index(arm) + 1
            (logs / f"K40-{arm}24_{job_id}.out").write_text(
                "[K40-FACTORIAL] DONE\n"
            )
            (logs / f"K40-{arm}24_{job_id}.err").write_text("")
        return root

    def _historical(self, repo: Path):
        root = repo / "historical"
        root.mkdir()
        status_path = root / "k40r2_flat_final.json"
        journal = Path(str(status_path) + ".columns.jsonl")
        journal.write_text(json.dumps({
            "trips": [1], "cost": 100000.0,
        }) + "\n")
        objective = HISTORICAL_WEIGHT * 100000.0
        status = {
            "csv": "duty_unions_big/Practice_Custom_DutyUnion_k40_r2.csv",
            "prices_csv": "hourly_prices_flat.csv",
            "soc_step": 15.0,
            "block_min": 10,
            "g_kwh": 300.0,
            "charge_kw": 300.0,
            "min_soc_frac": 0.0,
            "trip_ids": list(range(1, 948)),
            "iterations": 1428,
            "columns": 1,
            "wall_s": 79348.0,
            "stop_reason": "wall_limit",
            "certified_rc_optimal": False,
            "final": {
                "lp_obj": objective,
                "route_weight": HISTORICAL_WEIGHT,
                "artificials": 0.0,
                "min_rc": -1.0,
            },
            "final_lp": {
                "objective": objective,
                "route_weight": HISTORICAL_WEIGHT,
                "artificial_total": 0.0,
                "positive_routes": [{
                    "trips": [1],
                    "value": HISTORICAL_WEIGHT,
                    "cost": 100000.0,
                }],
            },
            "columns_journal": str(journal),
            "provenance": {
                "git_commit": HISTORICAL_COMMIT,
                "instance_sha256": self.instance_sha,
                "prices_sha256": self.prices_sha,
                "rc_eps": 0.0001,
                "args": {
                    "csv": "duty_unions_big/Practice_Custom_DutyUnion_k40_r2.csv",
                    "prices_csv": "hourly_prices_flat.csv",
                    "soc_step": 15.0,
                    "block_min": 10,
                    "g_kwh": 300.0,
                    "charge_kw": 300.0,
                    "min_soc_frac": 0.0,
                    "columns_per_iter": 30,
                    "rc_eps": 0.0001,
                    "max_iters": 200000,
                    "wall_limit_s": 90000.0,
                    "checkpoint_every": 25,
                    "snapshot_at_minutes": "60,180,360,720,1320,1440",
                    "resume": True,
                },
            },
        }
        status_path.write_text(json.dumps(status))
        Path(str(status_path) + ".iters.csv").write_text(
            "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,"
            f"pool_columns\n1,1428,{objective:.6f},"
            f"{HISTORICAL_WEIGHT:.9f},0.000000,-1.000000,1\n"
        )
        return status_path

    def _repo_fixture(self, root: Path):
        repo = root / "repo"
        (repo / "data/duty_unions_big").mkdir(parents=True)
        instance = repo / "data/duty_unions_big/Practice_Custom_DutyUnion_k40_r2.csv"
        instance_r1 = repo / "data/duty_unions_big/Practice_Custom_DutyUnion_k40_r1.csv"
        prices = repo / "data/hourly_prices_flat.csv"
        instance.write_bytes(self.instance_bytes)
        instance_r1.write_bytes(self.instance_bytes)
        prices.write_bytes(self.prices_bytes)
        # Launcher tests patch source hash validation through campaign statuses;
        # source files are located but frozen task hashes remain constants.
        r1 = self._campaign(
            repo,
            "k40fx_20260814T140232Z_eb85ca0c",
            replicate_offset=0.0,
            job_base=100,
            csv_name="Practice_Custom_DutyUnion_k40_r1.csv",
        )
        r2 = self._campaign(
            repo,
            "k40fx_20260814T191933Z_eb85ca0c",
            replicate_offset=0.5,
            job_base=200,
            csv_name="Practice_Custom_DutyUnion_k40_r2.csv",
        )
        historical = self._historical(repo)
        return repo, r1, r2, historical

    def test_summarizer_keeps_artificials_and_feasibility_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _repo, r1, r2, historical = self._repo_fixture(Path(tmp))
            payload = summarizer.summarize([r1, r2], historical)
            self.assertEqual(len(payload["replicate_rows"]), 56)
            self.assertEqual(len(payload["aggregate_rows"]), 28)
            pa = next(
                row for row in payload["replicate_rows"]
                if row["replicate"] == "R1"
                and row["arm"] == "PA"
                and row["checkpoint"] == "m1320"
            )
            self.assertEqual(pa["route_weight"], 11.58)
            self.assertEqual(pa["artificials"], 5.0)
            self.assertFalse(pa["real_lp_feasible"])
            self.assertIsNone(pa["feasible_route_weight"])
            self.assertIsNone(pa["historical_delta"])
            self.assertNotEqual(pa["actual_wall_s"], 1320 * 60 + 30)
            self.assertTrue(any(
                "not presented as feasible" in conclusion
                for conclusion in payload["conclusions"]
            ))

            prefix = Path(tmp) / "summary/factorial"
            outputs = summarizer.publish(payload, prefix)
            for path in outputs.values():
                self.assertTrue(Path(path).is_file())
            with self.assertRaises(FileExistsError):
                summarizer.publish(payload, prefix)

    def test_archiver_is_compute_only_and_verifies_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            _repo, r1, r2, historical = self._repo_fixture(Path(tmp))
            accounting = Path(tmp) / "sacct.txt"
            accounting.write_text(
                "JobIDRaw|JobName|State|Elapsed|ExitCode|MaxRSS\n"
                + "\n".join(
                    f"{job_id}|job|COMPLETED|01:00:00|0:0|1G"
                    for job_id in [
                        *range(100, 105), *range(200, 205)
                    ]
                )
                + "\n"
            )
            output = Path(tmp) / "factorial.tar.gz"
            with self.assertRaisesRegex(RuntimeError, "Slurm allocation"):
                archiver.archive(
                    [r1, r2], historical, accounting, output,
                    require_compute=True,
                )
            record = archiver.archive(
                [r1, r2], historical, accounting, output,
                require_compute=False,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(),
                record["archive_sha256"],
            )
            self.assertIn("slurm/accounting.txt", record["members"])

    def _validated_start(
        self,
        root: Path,
        *,
        instance_sha=None,
        prices_sha=None,
    ):
        instance_sha = instance_sha or self.instance_sha
        prices_sha = prices_sha or self.prices_sha
        buckets = [[] for _ in range(40)]
        for index, trip in enumerate(range(1, 948)):
            buckets[index % 40].append(trip)
        routes = [{
            "route": ["PARX_0", *trips, "PARX_0"],
            "charging_stops": {
                "stations": [], "cst": [], "cet": [], "kwh": [],
            },
        } for trips in buckets]
        root.mkdir(parents=True, exist_ok=True)
        path = root / "validated_start.json"
        path.write_text(json.dumps({
            "routes": routes,
            "source": "rerealized",
            "infeasible": [],
            "physics": {
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "reserve_frac": 0.0,
            },
            "prices_csv": "hourly_prices_flat.csv",
            "_factorial_start_provenance": {
                "schema": "evsp-dr-k40-factorial-giro-start-v1",
                "reviewed_checkout_commit": "a" * 40,
                "mip_core_commit": mip_launcher.MIP_CORE_COMMIT,
                "runner_sha256": "d" * 64,
                "snapshot_sha256": "b" * 64,
                "journal_sha256": "c" * 64,
                "instance_sha256": instance_sha,
                "prices_sha256": prices_sha,
                "bus_count": 40,
                "trip_count": 947,
            },
        }))
        return path

    def test_mip_launcher_packages_12_cells_and_escalation(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, r1, r2, _historical = self._repo_fixture(Path(tmp))
            start = self._validated_start(
                Path(tmp),
                instance_sha=self.instance_sha,
                prices_sha=self.prices_sha,
            )
            args = Namespace(
                replicate=[f"R1={r1}", f"R2={r2}"],
                validated_start=start,
                mode="screen",
                cell=None,
                campaign="mip_screen",
                python=Path(sys.executable),
                plan_out=Path(tmp) / "screen.plan.json",
                approved_plan_sha256=None,
                submit=False,
            )
            identity = {
                "expected_commit": "a" * 40,
                "observed_commit": "a" * 40,
                "detached": True,
                "tracked_clean": True,
                "profile_core_commit": "702491e2b9fa548b75a8b140ba5a4213c06df24f",
                "mip_core_commit": mip_launcher.MIP_CORE_COMMIT,
                "run_exact_pool_mip_sha256": "d" * 64,
            }
            python_identity = {
                "python_executable": str(Path(sys.executable).resolve()),
                "identity_sha256": "e" * 64,
                "gurobi_version": "test",
            }
            output = io.StringIO()
            with (
                patch.object(mip_launcher, "REPO_ROOT", repo),
                patch.object(
                    mip_launcher, "_mip_identity", return_value=identity
                ),
                patch.object(
                    mip_launcher, "_mip_python",
                    return_value=python_identity,
                ),
                patch.object(
                    mip_launcher, "_worker_bytes",
                    return_value=b"#!/bin/bash\n",
                ),
                patch.object(
                    mip_launcher,
                    "INSTANCE_SHA256",
                    self.instance_sha,
                ),
                patch.object(
                    mip_launcher,
                    "PRICES_SHA256",
                    self.prices_sha,
                ),
                contextlib.redirect_stdout(output),
            ):
                manifest = mip_launcher.launch(args)
            self.assertEqual(len(manifest["jobs"]), 12)
            names = [job["job_name"] for job in manifest["jobs"]]
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(len(name) <= 15 for name in names))
            self.assertEqual(
                {job["snapshot_mark_minutes"] for job in manifest["jobs"]},
                {360, 720, 1440},
            )
            self.assertEqual(
                {job["treatment"] for job in manifest["jobs"]},
                {"CA", "CS"},
            )
            self.assertTrue(args.plan_out.is_file())
            self.assertIn("[dry-run]", output.getvalue())

            args.submit = True
            args.approved_plan_sha256 = manifest["approval_sha256"]
            changed_journal = Path(manifest["jobs"][0]["source_journal"])
            original_journal = changed_journal.read_bytes()
            changed_journal.write_text(
                json.dumps({"trips": [1], "cost": 999999.0}) + "\n"
            )
            with (
                patch.object(mip_launcher, "REPO_ROOT", repo),
                patch.object(
                    mip_launcher, "_mip_identity", return_value=identity
                ),
                patch.object(
                    mip_launcher, "_mip_python",
                    return_value=python_identity,
                ),
                patch.object(
                    mip_launcher, "_worker_bytes",
                    return_value=b"#!/bin/bash\n",
                ),
                patch.object(
                    mip_launcher,
                    "INSTANCE_SHA256",
                    self.instance_sha,
                ),
                patch.object(
                    mip_launcher,
                    "PRICES_SHA256",
                    self.prices_sha,
                ),
                patch.object(
                    mip_launcher.subprocess,
                    "run",
                    side_effect=AssertionError(
                        "changed plan must not invoke sbatch"
                    ),
                ),
                self.assertRaisesRegex(
                    (SystemExit, ValueError),
                    "differs from approved|cost disagrees with journal",
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                mip_launcher.launch(args)
            changed_journal.write_bytes(original_journal)

            variants = [
                (b"#!/bin/bash\n# changed worker\n", python_identity),
                (
                    b"#!/bin/bash\n",
                    {
                        **python_identity,
                        "identity_sha256": "0" * 64,
                    },
                ),
            ]
            for worker_variant, environment_variant in variants:
                with (
                    patch.object(mip_launcher, "REPO_ROOT", repo),
                    patch.object(
                        mip_launcher, "_mip_identity",
                        return_value=identity,
                    ),
                    patch.object(
                        mip_launcher, "_mip_python",
                        return_value=environment_variant,
                    ),
                    patch.object(
                        mip_launcher, "_worker_bytes",
                        return_value=worker_variant,
                    ),
                    patch.object(
                        mip_launcher,
                        "INSTANCE_SHA256",
                        self.instance_sha,
                    ),
                    patch.object(
                        mip_launcher,
                        "PRICES_SHA256",
                        self.prices_sha,
                    ),
                    self.assertRaisesRegex(
                        SystemExit, "differs from approved"
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    mip_launcher.launch(args)

            args.submit = False
            args.approved_plan_sha256 = None
            args.mode = "escalation"
            args.cell = ["R1:CA:M360", "R2:CS:M1440"]
            args.campaign = "mip_escalation"
            args.plan_out = Path(tmp) / "escalation.plan.json"
            with (
                patch.object(mip_launcher, "REPO_ROOT", repo),
                patch.object(
                    mip_launcher, "_mip_identity", return_value=identity
                ),
                patch.object(
                    mip_launcher, "_mip_python",
                    return_value=python_identity,
                ),
                patch.object(
                    mip_launcher, "_worker_bytes",
                    return_value=b"#!/bin/bash\n",
                ),
                patch.object(
                    mip_launcher,
                    "INSTANCE_SHA256",
                    self.instance_sha,
                ),
                patch.object(
                    mip_launcher,
                    "PRICES_SHA256",
                    self.prices_sha,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                escalation = mip_launcher.launch(args)
            self.assertEqual(len(escalation["jobs"]), 2)
            self.assertEqual(escalation["budget_seconds"], 7200)
            self.assertTrue(all(
                len(job["job_name"]) <= 15
                and job["job_name"].endswith("H02")
                for job in escalation["jobs"]
            ))

    def test_partial_giro_start_and_changed_plan_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, r1, r2, _historical = self._repo_fixture(Path(tmp))
            partial = self._validated_start(Path(tmp))
            payload = json.loads(partial.read_text())
            payload["routes"].pop()
            partial.write_text(json.dumps(payload))
            with self.assertRaisesRegex(SystemExit, "not an exact partition"):
                mip_launcher._validate_start(
                    partial, list(range(1, 948))
                )
            missing_marker = self._validated_start(
                Path(tmp) / "missing-marker"
            )
            missing_payload = json.loads(missing_marker.read_text())
            del missing_payload["infeasible"]
            missing_marker.write_text(json.dumps(missing_payload))
            with self.assertRaisesRegex(SystemExit, "infeasible/partial"):
                mip_launcher._validate_start(
                    missing_marker, list(range(1, 948))
                )

            ca_snapshot = (
                r1 / "k40r1_flat_CA.m360.snapshot.json"
            )
            ca_status = json.loads(ca_snapshot.read_text())
            ca_journal = Path(ca_status["columns_journal"])
            original_journal = ca_journal.read_bytes()
            ca_journal.unlink()
            with self.assertRaisesRegex(ValueError, "journal"):
                artifacts.validate_campaign(r1, replicate="R1")
            ca_journal.write_bytes(original_journal)

            ca_status["g_kwh"] = 240.0
            ca_snapshot.write_text(json.dumps(ca_status))
            with self.assertRaisesRegex(ValueError, "g_kwh mismatch"):
                artifacts.validate_campaign(r1, replicate="R1")

    def test_journal_replacements_are_valid_but_boolean_trips_are_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            _repo, r1, _r2, _historical = self._repo_fixture(Path(tmp))
            snapshot = r1 / "k40r1_flat_CA.m360.snapshot.json"
            status = json.loads(snapshot.read_text())
            journal = Path(status["columns_journal"])
            original = json.loads(journal.read_text())
            journal.write_text(
                json.dumps({
                    "trips": [1],
                    "cost": original["cost"] + 1000.0,
                })
                + "\n"
                + json.dumps(original)
                + "\n"
            )
            artifacts.validate_campaign(r1, replicate="R1")
            journal.write_text(
                json.dumps({"trips": [True], "cost": original["cost"]})
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "invalid journal trips"):
                artifacts.validate_campaign(r1, replicate="R1")

    def test_mixed_feasibility_does_not_create_one_replicate_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            _repo, r1, r2, historical = self._repo_fixture(Path(tmp))
            snapshot = r2 / "k40r2_flat_PA.m1320.snapshot.json"
            status = json.loads(snapshot.read_text())
            status["final"]["artificials"] = 0.0
            status["final_lp"]["artificial_total"] = 0.0
            status["final_lp"]["objective"] = (
                status["final_lp"]["route_weight"]
                * status["final_lp"]["positive_routes"][0]["cost"]
            )
            status["final"]["lp_obj"] = status["final_lp"]["objective"]
            snapshot.write_text(json.dumps(status))
            payload = summarizer.summarize([r1, r2], historical)
            aggregate = next(
                row for row in payload["aggregate_rows"]
                if row["arm"] == "PA" and row["checkpoint"] == "m1320"
            )
            self.assertEqual(aggregate["feasible_route_weight_n"], 1)
            self.assertIsNone(aggregate["feasible_route_weight"])
            self.assertIsNone(aggregate["historical_delta"])

    def test_atomic_summary_bundle_is_never_partially_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            _repo, r1, r2, historical = self._repo_fixture(Path(tmp))
            payload = summarizer.summarize([r1, r2], historical)
            prefix = Path(tmp) / "faulted-summary"
            original = summarizer._write_new
            calls = 0

            def fail_second(path, data):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected publication failure")
                return original(path, data)

            with (
                patch.object(
                    summarizer, "_write_new", side_effect=fail_second
                ),
                self.assertRaisesRegex(OSError, "injected"),
            ):
                summarizer.publish(payload, prefix)
            self.assertFalse(Path(str(prefix) + ".bundle").exists())
            self.assertTrue(Path(str(prefix) + ".bundle.lock").exists())

    def test_monitor_rejects_unattested_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "cell.json"
            output.write_text("{}\n")
            (root / "campaign.json").write_text(json.dumps({
                "schema": "evsp-dr-k40-factorial-mip-campaign-v1",
                "budget_seconds": 1800,
                "jobs": [{
                    "label": "R1_CA_m360",
                    "replicate": "R1",
                    "treatment": "CA",
                    "snapshot_mark_minutes": 360,
                    "job_id": "123",
                    "output": str(output),
                    "spec_sha256": "a" * 64,
                }],
            }))
            row = mip_monitor.rows(root, query_slurm=False)[0]
            self.assertTrue(row["output_exists"])
            self.assertFalse(row["output_valid"])
            self.assertIsNotNone(row["validation_error"])
            self.assertIsNone(row["buses"])

    def test_reconciliation_rejects_empty_and_ambiguous_campaigns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "campaign.json").write_text(json.dumps({
                "schema": "evsp-dr-k40-factorial-mip-campaign-v1",
                "mode": "screen",
                "jobs": [],
            }))
            with self.assertRaisesRegex(ValueError, "no jobs"):
                mip_reconcile.reconcile(root)
            jobs = []
            for index in range(12):
                jobs.append({
                    "label": f"cell{index}",
                    "job_name": f"Kabc{index:02d}",
                    "slurm_comment": f"comment{index}",
                    "submission_state": (
                        "attempting" if index == 0 else "planned"
                    ),
                    "job_id": None,
                })
            (root / "campaign.json").write_text(json.dumps({
                "schema": "evsp-dr-k40-factorial-mip-campaign-v1",
                "campaign": "test",
                "mode": "screen",
                "created_at": "2026-08-16T00:00:00+00:00",
                "jobs": jobs,
            }))
            with patch.object(
                mip_reconcile,
                "_query",
                return_value=[
                    {"job_id": "1"}, {"job_id": "2"}
                ],
            ):
                record = mip_reconcile.reconcile(root)
            self.assertEqual(len(record["unresolved"]), 1)
            self.assertFalse(record["safe_to_submit_pending"])

    def test_pending_submission_resume_skips_recorded_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / "worker.sub"
            worker.write_bytes(b"worker")
            start = root / "start.json"
            start.write_bytes(b"start")
            shared = {
                "staged_result": root / "snapshot.json",
                "staged_journal": root / "journal.jsonl",
                "staged_instance": root / "instance.csv",
                "staged_prices": root / "prices.csv",
            }
            for index, path in enumerate(shared.values()):
                path.write_bytes(f"source-{index}".encode())
            source_hashes = {
                key.removeprefix("staged_"): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for key, path in shared.items()
            }
            source_hashes["result"] = source_hashes.pop("result")
            jobs = []
            for index in range(2):
                spec = {
                    "staged_result_sha256": source_hashes["result"],
                    "staged_start": str(start),
                }
                spec_path = root / f"spec-{index}.json"
                spec_path.write_text(json.dumps(spec))
                jobs.append({
                    "label": f"cell{index}",
                    **{key: str(value) for key, value in shared.items()},
                    "source_hashes": source_hashes,
                    "spec": spec,
                    "spec_path": str(spec_path),
                    "spec_sha256": hashlib.sha256(
                        spec_path.read_bytes()
                    ).hexdigest(),
                    "job_name": f"Kabc0{index}",
                    "slurm_comment": f"comment{index}",
                    "command": ["sbatch", f"cell{index}"],
                    "job_id": "100" if index == 0 else None,
                    "submission_state": (
                        "submitted" if index == 0 else "planned"
                    ),
                })
            identity = {
                "expected_commit": "a" * 40,
                "observed_commit": "a" * 40,
                "run_exact_pool_mip_sha256": "b" * 64,
            }
            manifest = {
                "checkout_identity": identity,
                "worker": str(worker),
                "worker_sha256": hashlib.sha256(
                    worker.read_bytes()
                ).hexdigest(),
                "validated_start": {
                    "sha256": hashlib.sha256(
                        start.read_bytes()
                    ).hexdigest(),
                },
                "jobs": jobs,
                "submitted": False,
            }
            manifest_path = root / "campaign.json"
            manifest_path.write_text(json.dumps(manifest))
            completed = __import__("subprocess").CompletedProcess(
                ["sbatch"], 0, stdout="101\n", stderr=""
            )
            with (
                patch.object(
                    mip_launcher, "_mip_identity", return_value=identity
                ),
                patch.object(
                    mip_launcher.subprocess,
                    "run",
                    return_value=completed,
                ) as submit,
            ):
                mip_launcher._submit_pending(
                    root, manifest, manifest_path
                )
            self.assertEqual(submit.call_count, 1)
            self.assertEqual(jobs[0]["job_id"], "100")
            self.assertEqual(jobs[1]["job_id"], "101")
            self.assertTrue(manifest["submitted"])

    def test_workers_are_nonrequeue_and_strict(self):
        for name in (
            "submit_k40_factorial_mip_screen.sub",
            "submit_k40_factorial_archive.sub",
        ):
            text = (REPO_ROOT / "src" / name).read_text()
            self.assertIn("set -euo pipefail", text)
            self.assertIn("#SBATCH --no-requeue", text)
            self.assertNotIn("#SBATCH --requeue", text)
        mip_text = (
            REPO_ROOT / "src/submit_k40_factorial_mip_screen.sub"
        ).read_text()
        self.assertIn("--two-stage", mip_text)
        self.assertIn("--initial-partition-routes", mip_text)
        self.assertNotIn("--cover", mip_text)

    def test_shared_seed_preparer_is_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, r1, _r2, _historical = self._repo_fixture(Path(tmp))
            snapshot = r1 / "k40r1_flat_CA.m360.snapshot.json"
            output = Path(tmp) / "shared-start.json"
            buckets = [[] for _ in range(40)]
            for index, trip in enumerate(range(1, 948)):
                buckets[index % 40].append(trip)
            routes = [{
                "route": ["PARX_0", *trips, "PARX_0"],
                "charging_stops": {
                    "stations": [], "cst": [], "cet": [], "kwh": [],
                },
            } for trips in buckets]

            def fake_run(command, *, environment=None):
                if "make_giro_seed_routes.py" in " ".join(command):
                    Path(command[command.index("--out") + 1]).write_text(
                        json.dumps({"routes": routes})
                    )
                elif "rerealize_routes.py" in " ".join(command):
                    Path(command[command.index("--out") + 1]).write_text(
                        json.dumps({
                            "routes": routes,
                            "source": "rerealized",
                            "infeasible": [],
                            "physics": {
                                "g_kwh": 300.0,
                                "charge_kw": 300.0,
                                "reserve_frac": 0.0,
                            },
                            "prices_csv": "hourly_prices_flat.csv",
                        })
                    )
                else:
                    self.assertIn("--validate-only", command)
                    self.assertIsNotNone(environment)

            identity = {
                "expected_commit": "a" * 40,
                "mip_core_commit": mip_launcher.MIP_CORE_COMMIT,
                "run_exact_pool_mip_sha256": "d" * 64,
            }
            instance_sha = self.instance_sha
            prices_sha = self.prices_sha
            with (
                patch.object(seed_preparer, "REPO_ROOT", repo),
                patch.object(
                    seed_preparer, "_mip_identity", return_value=identity
                ),
                patch.object(seed_preparer, "_run", side_effect=fake_run),
                patch.object(
                    seed_preparer,
                    "_verify_snapshot_problem_inputs",
                    return_value={
                        "instance_sha256": instance_sha,
                        "prices_sha256": prices_sha,
                    },
                ),
                patch.object(
                    seed_preparer, "INSTANCE_SHA256", instance_sha
                ),
                patch.object(
                    seed_preparer, "PRICES_SHA256", prices_sha
                ),
            ):
                result = seed_preparer.prepare(
                    snapshot, output, Path(sys.executable)
                )
                self.assertEqual(result["bus_count"], 40)
                prepared = json.loads(output.read_text())
                self.assertEqual(
                    prepared["_factorial_start_provenance"][
                        "snapshot_sha256"
                    ],
                    hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                )
                with self.assertRaises(FileExistsError):
                    seed_preparer.prepare(
                        snapshot, output, Path(sys.executable)
                    )


if __name__ == "__main__":
    unittest.main()

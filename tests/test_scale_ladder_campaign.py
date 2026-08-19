import csv
import copy
import ctypes
import errno
import hashlib
import json
import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import launch_scale_ladder as ladder  # noqa: E402
from tariff_response_environment import (  # noqa: E402
    PORTABLE_FIELDS,
    compare_portable,
)
from build_scale_ladder_inputs import build as build_inputs  # noqa: E402
from build_tariff_response_manifest import sha256_file  # noqa: E402
from scale_ladder_trip_identity import (  # noqa: E402
    classify_legacy_trip_hash,
    identity,
    require_compatible,
)
from summarize_scale_ladder import (  # noqa: E402
    CG_FIELDS,
    PROGRESS_FIELDS,
    _validate_completion,
    summarize,
    target_gap_interpretation,
    _rename_noreplace,
)
from audit_scale_ladder_known_membership import audit  # noqa: E402
from recover_scale_ladder_mip_progress import recover  # noqa: E402


INSTANCE_MANIFEST = (
    REPO_ROOT
    / "data/scale_ladder/instances/scale_ladder_instance_manifest.csv"
)


class ScaleLadderCampaignTests(unittest.TestCase):
    def _portable_identity(self):
        portable = {
            field: f"value-{field}" for field in PORTABLE_FIELDS
        }
        encoded = json.dumps(
            portable, sort_keys=True, separators=(",", ":")
        ).encode()
        return {
            "schema": "evsp-dr-portable-environment-v1",
            "portable": portable,
            "portable_identity_sha256":
                hashlib.sha256(encoded).hexdigest(),
            "node_metadata": {
                "platform": "login",
                "hostname": "login01",
                "kernel_release": "login-kernel",
            },
        }

    def test_portable_environment_ignores_node_metadata(self):
        planned = self._portable_identity()
        observed = copy.deepcopy(planned)
        observed["node_metadata"] = {
            "platform": "compute",
            "hostname": "worker99",
            "kernel_release": "different-kernel",
        }
        self.assertEqual(compare_portable(planned, observed), [])

    def test_portable_environment_reports_exact_required_mismatches(self):
        planned = self._portable_identity()
        for field in (
            "executable_sha256",
            "numpy",
            "scipy_distribution_sha256",
        ):
            observed = copy.deepcopy(planned)
            observed["portable"][field] = "changed"
            differences = compare_portable(planned, observed)
            self.assertEqual(
                [item["field"] for item in differences],
                [f"portable.{field}"],
            )
            self.assertEqual(differences[0]["planned"],
                             planned["portable"][field])
            self.assertEqual(differences[0]["observed"], "changed")
        observed = copy.deepcopy(planned)
        del observed["portable"]["pandas_distribution_sha256"]
        differences = compare_portable(planned, observed)
        self.assertEqual(
            differences[0]["field"],
            "portable.pandas_distribution_sha256",
        )
        self.assertEqual(
            differences[0]["reason"], "missing_required_field"
        )

    def test_failed_probe_never_releases_gate(self):
        compatible = {
            "default_partition": {"compatible": True},
            "scaglione": {"compatible": True},
        }
        for failed in ("default_partition", "scaglione"):
            results = copy.deepcopy(compatible)
            results[failed]["compatible"] = False
            runner = Mock()
            completed = ladder._release_gate_after_probes(
                {"scontrol": {"path": "/approved/scontrol"}},
                "12345",
                results,
                runner=runner,
            )
            self.assertNotEqual(completed.returncode, 0)
            runner.assert_not_called()

    def test_atomic_summary_publication_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "value.txt").write_text("new")
            _rename_noreplace(source, target)
            self.assertFalse(source.exists())
            self.assertEqual((target / "value.txt").read_text(), "new")

            second = root / "second"
            second.mkdir()
            with self.assertRaises(FileExistsError):
                _rename_noreplace(second, target)
            self.assertTrue(second.exists())
            self.assertEqual((target / "value.txt").read_text(), "new")

    def test_darwin_renamex_dispatch_and_unsupported_platform(self):
        class FakeFunction:
            argtypes = None
            restype = None

            def __call__(self, source, target, flags):
                self.flags = flags
                source_path = Path(os.fsdecode(source))
                target_path = Path(os.fsdecode(target))
                try:
                    os.link(source_path, target_path)
                except FileExistsError:
                    ctypes.set_errno(errno.EEXIST)
                    return -1
                source_path.unlink()
                return 0

        class FakeLib:
            renamex_np = FakeFunction()

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            target = Path(tmp) / "target"
            source.write_text("value")
            fake = FakeLib()
            _rename_noreplace(
                source, target, platform="darwin", libc=fake
            )
            self.assertEqual(fake.renamex_np.flags, 0x00000004)
            self.assertEqual(target.read_text(), "value")
            another = Path(tmp) / "another"
            another.write_text("other")
            with self.assertRaises(FileExistsError):
                _rename_noreplace(
                    another, target, platform="darwin", libc=fake
                )
            with self.assertRaisesRegex(OSError, "unsupported"):
                _rename_noreplace(
                    another, Path(tmp) / "x",
                    platform="freebsd", libc=fake,
                )

    def test_manifest_has_exact_cells_and_identity_domains(self):
        with INSTANCE_MANIFEST.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 22)
        counts = {}
        for row in rows:
            counts[int(row["scale"])] = counts.get(
                int(row["scale"]), 0
            ) + 1
            self.assertEqual(
                row["trip_identity_schema"], "evsp-dr-trip-identity-v1"
            )
            self.assertNotIn("trip_set_sha256", row)
            observed = identity(REPO_ROOT / row["relative_path"])
            for field in (
                "ordered_trip_id_set_sha256",
                "solver_local_trip_index_sha256",
                "ordered_trip_sequence_sha256",
                "instance_file_sha256",
            ):
                self.assertEqual(row[field], observed[field])
            duties = json.loads(row["duties_json"])
            bases = [
                "13316" if duty in {"13316m", "13316uwt"}
                else "13324" if duty in {"13324muw", "13324t"}
                else duty
                for duty in duties
            ]
            self.assertEqual(len(bases), len(set(bases)))
        self.assertEqual(
            counts, {2: 3, 3: 3, 5: 3, 8: 3, 13: 3, 20: 3, 30: 3, 40: 1}
        )

    def test_k40_hash_domains_are_explicit_and_not_cross_compared(self):
        k40 = (
            REPO_ROOT
            / "data/tariff_response/frozen_instances/"
            "Practice_Custom_DutyUnion_k40_r2.csv"
        )
        observed = identity(k40)
        self.assertEqual(
            observed["instance_file_sha256"],
            "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd",
        )
        self.assertEqual(
            observed["ordered_trip_id_set_sha256"],
            "2756baf18e81509df74a2e92e3925aefd17fdf81d95ded9161ad113fd6830e50",
        )
        self.assertEqual(
            observed["solver_local_trip_index_sha256"],
            "35604b22facf1646963e85eb98a858906f0dd7dbebd86ea0d3ac7b797de62ed0",
        )
        for field in (
            "ordered_trip_id_set_sha256",
            "solver_local_trip_index_sha256",
        ):
            classified = classify_legacy_trip_hash(
                observed[field], observed
            )
            self.assertEqual(classified["legacy_trip_hash_field"], field)
        changed = dict(observed)
        changed["ordered_trip_id_set_sha256"] = observed[
            "solver_local_trip_index_sha256"
        ]
        with self.assertRaisesRegex(ValueError, "domain"):
            require_compatible(
                observed, changed, "ordered_trip_id_set_sha256"
            )

    def test_deterministic_regeneration_matches_all_instance_hashes(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            manifest, _campaign, generated = build_inputs(Path(tmp) / "out")
            self.assertTrue(manifest.is_file())
            with INSTANCE_MANIFEST.open(newline="") as handle:
                reviewed = {
                    (int(row["scale"]), int(row["selection_replicate"])):
                        row["instance_file_sha256"]
                    for row in csv.DictReader(handle)
                }
            self.assertEqual(
                {
                    (int(row["scale"]), int(row["selection_replicate"])):
                        row["instance_file_sha256"]
                    for row in generated
                },
                reviewed,
            )

    def test_plan_has_exact_task_mapping_and_no_k40_mips(self):
        environment = {
            "schema": "evsp-dr-portable-environment-v1",
            "portable": {
                "python": "3.12.3",
                "executable": str(Path(sys.executable).resolve()),
                "executable_sha256":
                    sha256_file(Path(sys.executable).resolve()),
            },
            "portable_identity_sha256": "e" * 64,
            "node_metadata": {},
        }
        checkout = {
            "commit": "a" * 40,
            "reviewed_base": ladder.REVIEWED_BASE,
            "detached": True,
            "tracked_clean": True,
        }
        with (
            patch.object(ladder, "_environment", return_value=environment),
            patch.object(ladder, "checkout_identity", return_value=checkout),
            patch.object(ladder.shutil, "which", return_value="/usr/bin/true"),
        ):
            plan = ladder.build_plan(
                "ladder-test", Path(sys.executable), Path("/tmp/reservations")
            )
        self.assertEqual(plan["task_count"], 138)
        self.assertEqual(plan["preflight_task_count"], 22)
        self.assertEqual(plan["cg_task_count"], 23)
        self.assertEqual(plan["sensitivity_cg_task_count"], 30)
        self.assertEqual(plan["mip_task_count"], 42)
        self.assertEqual(plan["k40_mip_submission_count"], 0)
        self.assertEqual(
            {key: len(value) for key, value in plan["task_groups"].items()},
            {
                "PREFLIGHT": 22, "SEED": 21, "CG": 23,
                "CG_SENSITIVITY": 30,
                "MIP_RAW": 21, "MIP_KNOWN": 21,
            },
        )
        self.assertFalse(any(
            job["phase"] == "MIP" and job["scale"] == 40
            for job in plan["jobs"]
        ))
        self.assertTrue(all(
            len(job["job_name"]) <= 15 for job in plan["jobs"]
        ))
        self.assertEqual(
            len({job["job_name"] for job in plan["jobs"]}),
            len(plan["jobs"]),
        )
        self.assertEqual(
            plan["tariff"]["primary_tariff_sha256"],
            ladder.HISTORICAL_FLAT_SHA256,
        )
        self.assertNotIn("alpha", json.dumps(plan).lower())
        self.assertEqual(len(plan["k40_reuse_slots"]), 4)
        fallback = [
            job for job in plan["jobs"]
            if job["phase"] == "CG_SENSITIVITY"
            and job["soc_step"] == 1.0 and job["block_min"] == 5
        ]
        self.assertEqual(len(fallback), 3)
        self.assertTrue(all(
            job["scale"] == 2 and job["diagnostic_only"] is True
            for job in fallback
        ))
        self.assertTrue(all(
            job["telemetry"] is None
            for job in plan["jobs"]
            if job["phase"] == "CG" and job["scale"] >= 30
        ))
        self.assertTrue(all(
            job["telemetry"] is not None
            for job in plan["jobs"]
            if job["phase"] == "CG" and job["scale"] <= 20
        ))
        self.assertIn(1440, next(
            job["snapshot_minutes"] for job in plan["jobs"]
            if job["phase"] == "CG" and job["scale"] == 40
        ))

    def test_worker_maps_dependencies_and_resume(self):
        worker = (REPO_ROOT / "src/submit_scale_ladder.sub").read_text()
        launcher = (REPO_ROOT / "src/launch_scale_ladder.py").read_text()
        self.assertIn("--resume", worker)
        self.assertIn("--snapshot-at-minutes", worker)
        self.assertIn("KNOWN-PARTITION", worker)
        self.assertIn("aftercorr:", launcher)
        self.assertIn("JobName=\"$JOB_NAME\"", worker)
        self.assertIn("EVSP_MIP_EXPECTED_RESULT_SHA256", worker)
        local = (
            REPO_ROOT / "src/run_scale_ladder_local_diagnostics.py"
        ).read_text()
        self.assertNotIn("sbatch", local)
        self.assertNotIn('"phase": "PREFLIGHT"', local)
        self.assertIn("default=3", local)
        self.assertIn("diagnostic_only", local)

    def test_k2_r1_certified_15_vs_5_route_space_distinction(self):
        with INSTANCE_MANIFEST.open(newline="") as handle:
            row = next(
                item for item in csv.DictReader(handle)
                if item["scale"] == "2"
                and item["selection_replicate"] == "1"
            )
        observed = {}
        for soc_step in (15.0, 5.0):
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "cg.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "src/exact_pricer_expanded.py"),
                        "--csv", row["relative_path"].removeprefix("data/"),
                        "--prices_csv", "hourly_prices_flat.csv",
                        "--soc-step", str(soc_step),
                        "--block-min", "10",
                        "--max-iters", "2000",
                        "--master-sense", "partition",
                        "--initial-pool", "singletons",
                        "--wall-limit-s", "300",
                        "--checkpoint-every", "25",
                        "--resume", "--out", str(output),
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=180,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                payload = json.loads(output.read_text())
                self.assertTrue(payload["certified_rc_optimal"])
                self.assertEqual(payload["final"]["artificials"], 0.0)
                observed[soc_step] = payload["final_lp"]["route_weight"]
        self.assertGreater(observed[15.0], 2.0)
        self.assertAlmostEqual(observed[5.0], 2.0, places=8)

    def test_k2_r2_positive_gap_is_not_labeled_runtime_failure(self):
        with INSTANCE_MANIFEST.open(newline="") as handle:
            row = next(
                item for item in csv.DictReader(handle)
                if item["scale"] == "2"
                and item["selection_replicate"] == "2"
            )
        membership = audit(
            REPO_ROOT / row["relative_path"],
            row["instance_file_sha256"],
            2,
            2,
        )
        self.assertFalse(
            membership["known_partition_in_primary_expanded_space"]
        )
        self.assertEqual(
            target_gap_interpretation(
                2.153846, 2,
                membership["known_partition_in_primary_expanded_space"],
            ),
            "known_partition_outside_primary_space_not_scaling_failure",
        )
        self.assertEqual(
            target_gap_interpretation(2.0, 2, True, 1.0),
            "target_not_comparable_positive_or_missing_artificial_mass",
        )

    def test_summary_schema_and_censoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = root / "instance.csv"
            instance.write_text(
                "Ordered_Trip_ID\n10\n20\n"
            )
            identities = identity(instance)
            cg = root / "cg.json"
            journal = Path(str(cg) + ".columns.jsonl")
            journal.write_text(
                json.dumps({"trips": [0, 1], "cost": 200000.0}) + "\n"
            )
            cg.write_text(json.dumps({
                "columns_journal": str(journal),
                "wall_s": 100.0,
                "iterations": 2,
                "certified_rc_optimal": False,
                "stop_reason": "wall_limit",
                "columns": 2,
                "trip_set_sha256":
                    identities["solver_local_trip_index_sha256"],
                "soc_step": 15.0, "block_min": 10,
                "g_kwh": 300.0, "charge_kw": 300.0,
                "min_soc_frac": 0.0,
                "provenance": {
                    "instance_sha256":
                        identities["instance_file_sha256"],
                    "prices_sha256": "c" * 64,
                },
            }))
            Path(str(cg) + ".iters.csv").write_text(
                "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,pool_columns\n"
                "10,1,3,2.5,1,-1,1\n"
                "100,2,2,2,0,-0.1,2\n"
            )
            telemetry = root / "telemetry.jsonl"
            telemetry.write_text(
                json.dumps({
                    "record_type": "phase", "phase": "master_attempt",
                    "duration_s": 1.0, "iteration": 1,
                }) + "\n" + json.dumps({
                    "record_type": "phase",
                    "phase": "pricing_shortest_path",
                    "duration_s": 2.0, "iteration": 1,
                }) + "\n" + json.dumps({
                    "record_type": "phase", "phase": "master_attempt",
                    "duration_s": 1.0, "iteration": 2,
                }) + "\n" + json.dumps({
                    "record_type": "phase",
                    "phase": "pricing_shortest_path",
                    "duration_s": 2.0, "iteration": 2,
                }) + "\n"
            )
            mip = root / "mip.json"
            mip.write_text(json.dumps({
                "status_name": "TIME_LIMIT", "incumbent_found": True,
                "buses": 2, "fleet_bound": 1, "mip_gap": 0.5,
                "fleet_proven": False, "runtime_s": 50,
                "optimal_scope": "none",
                "source_result_sha256": sha256_file(cg),
                "source_journal_sha256": sha256_file(journal),
                "physics": {
                    "g_kwh": 300.0, "charge_kw": 300.0,
                    "min_soc_frac": 0.0,
                },
                "experiment_arm": "B",
                "mip_start": {
                    "kind": "none", "source": None,
                },
                "progress": {"checkpoint_schedule_s": [60.0]},
            }))
            progress = root / "progress"
            progress.mkdir()
            (progress / "checkpoint_0001m.json").write_text(json.dumps({
                "checkpoint_elapsed_s": 60,
                "incumbent": {
                    "fleet": 2, "route_vector_sha256": "a" * 64,
                },
                "latest_statistics": {
                    "statistics_incumbent_fleet": 2,
                    "fleet_bound": 1, "fleet_gap": 0.5,
                    "node_count": 3, "solution_count": 1,
                },
                "solver_ended_before_checkpoint": False,
            }))
            (progress / "final.json").write_text(json.dumps({
                "kind": "final", "final": {"status_name": "TIME_LIMIT"},
            }))
            base = {
                "cell_id": "k02_s1_c1", "scale": 2,
                "selection_replicate": 1, "cg_replicate": 1,
                "campaign_replicate": 1, "target_fleet": 2,
                "soc_step": 15.0, "block_min": 10,
                "instance": {
                    "path": str(instance), **identities,
                },
            }
            preflight_output = root / "preflight.json"
            preflight_payload = {
                "schema": "evsp-dr-scale-ladder-known-membership-v1",
                "cell_id": "k02_s1", "scale": 2,
                "selection_replicate": 1,
                "known_partition_continuously_feasible": True,
                "known_partition_in_primary_expanded_space": False,
                "fixed_sequence_pricing_certified": True,
                "first_feasible_soc_step": 5.0,
                "first_feasible_block_min": 10,
                "nonrepresentability_reason": "primary_grid_blocked",
                "duties": [],
            }
            preflight_output.write_text(json.dumps(preflight_payload))
            preflight_output.with_suffix(".csv").write_text(
                ",".join([
                    "cell_id", "scale", "selection_replicate", "duty_id"
                ]) + "\n"
            )
            preflight_job = {
                **base, "job_key": "preflight", "phase": "PREFLIGHT",
                "arm": None, "budget_s": 10,
                "output": str(preflight_output),
                "telemetry": None, "progress_dir": None,
                "snapshot_minutes": [],
            }
            cg_job = {
                **base, "job_key": "cg", "phase": "CG", "arm": None,
                "budget_s": 100, "output": str(cg),
                "telemetry": str(telemetry), "progress_dir": None,
                "snapshot_minutes": [],
            }
            cg5 = root / "cg5.json"
            cg5_journal = Path(str(cg5) + ".columns.jsonl")
            cg5_journal.write_text(journal.read_text())
            cg5_status = json.loads(cg.read_text())
            cg5_status["columns_journal"] = str(cg5_journal)
            cg5_status["soc_step"] = 5.0
            cg5.write_text(json.dumps(cg5_status))
            Path(str(cg5) + ".iters.csv").write_text(
                Path(str(cg) + ".iters.csv").read_text()
            )
            telemetry5 = root / "telemetry5.jsonl"
            telemetry5.write_text(telemetry.read_text())
            cg5_job = {
                **base, "job_key": "cg5",
                "phase": "CG_SENSITIVITY", "arm": None,
                "budget_s": 100, "output": str(cg5),
                "telemetry": str(telemetry5), "progress_dir": None,
                "snapshot_minutes": [], "soc_step": 5.0,
                "block_min": 10,
            }
            mip_job = {
                **base, "job_key": "mip", "phase": "MIP", "arm": "RAW",
                "budget_s": 60, "output": str(mip),
                "progress_dir": str(progress), "telemetry": None,
                "scientific_role": None,
                "dependency_cg": "cg",
                "snapshot_minutes": [],
            }
            plan = {
                "execution_mode": "local_diagnostic",
                "checkout_identity": {"commit": "b" * 40},
                "tariff": {"primary_tariff_sha256": "c" * 64},
                "physics": {}, "trip_identity_schema":
                    "evsp-dr-trip-identity-v1",
                "code_hashes": {}, "python_identity": {},
                "task_groups": {
                    "PREFLIGHT": ["preflight"],
                    "CG": ["cg"], "CG_SENSITIVITY": ["cg5"],
                    "MIP_RAW": ["mip"],
                },
                "jobs": [preflight_job, cg_job, cg5_job, mip_job],
                "k40_reuse_slots": [],
            }
            raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            plan_sha = hashlib.sha256(raw).hexdigest()
            (root / "approved-plan.json").write_bytes(raw)
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "execution_mode": "local_diagnostic",
                "diagnostic_only": True,
                "submitted": False,
                "gate_state": "not_applicable_local",
                "submitted_arrays": {},
            }))
            for job, artifacts in (
                (
                    preflight_job,
                    [preflight_output, preflight_output.with_suffix(".csv")],
                ),
                (
                    cg_job,
                    [cg, journal, Path(str(cg) + ".iters.csv"), telemetry],
                ),
                (
                    cg5_job,
                    [
                        cg5, cg5_journal,
                        Path(str(cg5) + ".iters.csv"), telemetry5,
                    ],
                ),
                (
                    mip_job,
                    [
                        mip, progress / "checkpoint_0001m.json",
                        progress / "final.json",
                    ],
                ),
            ):
                completion = {
                    "schema":
                        "evsp-dr-scale-ladder-worker-completion-v1",
                    "phase": job["phase"],
                    "plan_sha256": plan_sha,
                    "instance_file_sha256":
                        identities["instance_file_sha256"],
                    "job_key": job["job_key"],
                    "arm": job["arm"],
                    "artifact_sha256": {
                        str(path.resolve()): sha256_file(path)
                        for path in artifacts
                    },
                }
                Path(
                    str(job["output"]) + ".worker-completion.json"
                ).write_text(json.dumps(completion))
            out = root / "summary"
            summarize(root, out)
            required = {
                "cg_iteration_long.csv", "cg_run_summary.csv",
                "mip_checkpoint_long.csv", "mip_run_summary.csv",
                "artifact_inventory.csv", "scale_progress_summary.csv",
                "known_route_membership_long.csv",
                "cg_route_weight.png", "mip_incumbent_bound.png",
                "provenance.json",
            }
            self.assertTrue(required <= {
                path.name for path in out.iterdir()
            })
            with (out / "cg_run_summary.csv").open(newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            row = next(
                item for item in summary_rows
                if item["campaign_role"] == "primary"
            )
            sensitivity = next(
                item for item in summary_rows
                if item["campaign_role"] == "small_grid_sensitivity"
            )
            self.assertEqual(row["censored"], "True")
            self.assertEqual(sensitivity["soc_step"], "5.0")
            self.assertEqual(
                sensitivity["grid_interpretation"],
                "route_space_sensitivity_diagnostic",
            )
            self.assertEqual(row["trip_identity_schema"],
                             "evsp-dr-trip-identity-v1")
            self.assertNotIn("trip_set_sha256", CG_FIELDS)
            self.assertNotIn("trip_set_sha256", PROGRESS_FIELDS)

    def test_early_certified_cg_censors_future_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "cg.json"
            journal = Path(str(output) + ".columns.jsonl")
            iterations = Path(str(output) + ".iters.csv")
            output.write_text(json.dumps({
                "wall_s": 100.0, "stop_reason": "certified",
                "snapshot_availability": {
                    "5": "censored_solver_terminated_before_mark"
                },
            }))
            journal.write_text("{}\n")
            iterations.write_text("header\n")
            job = {
                "job_key": "cg", "phase": "CG", "arm": None,
                "output": str(output), "telemetry": None,
                "snapshot_minutes": [5],
                "instance": {"instance_file_sha256": "a" * 64},
            }
            completion = {
                "schema":
                    "evsp-dr-scale-ladder-worker-completion-v1",
                "phase": "CG", "plan_sha256": "p" * 64,
                "instance_file_sha256": "a" * 64,
                "job_key": "cg", "arm": None,
                "snapshot_availability": {
                    "5": "censored_solver_terminated_before_mark"
                },
                "artifact_sha256": {
                    str(path.resolve()): sha256_file(path)
                    for path in (output, journal, iterations)
                },
            }
            Path(str(output) + ".worker-completion.json").write_text(
                json.dumps(completion)
            )
            _validate_completion(job, "p" * 64)

    def test_pending_mip_result_recovers_only_censored_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp)
            (progress / "result_pending.json").write_text(json.dumps({
                "status": 9, "status_name": "TIME_LIMIT",
                "incumbent_found": True, "buses": 3,
                "mip_obj": 300000.0, "mip_bound": 200000.0,
                "mip_gap": 0.5, "fleet_proven": False,
                "optimal_scope": "none", "runtime_s": 100.0,
                "progress": {
                    "checkpoint_schedule_s": [0.0, 60.0, 300.0],
                    "termination_signal": "SIGUSR1",
                },
            }))
            (progress / "latest.json").write_text(json.dumps({
                "schema": "evsp-dr-mip-convergence-v1",
                "kind": "latest",
                "incumbent": {
                    "fleet": 3, "route_vector_sha256": "a" * 64,
                },
                "latest_statistics": {"fleet_bound": 2.0},
            }))
            (progress / "checkpoint_0000m.json").write_text("{}")
            recover(progress)
            self.assertTrue((progress / "final.json").is_file())
            crossed = json.loads(
                (progress / "checkpoint_0001m.json").read_text()
            )
            self.assertFalse(crossed["solver_ended_before_checkpoint"])
            self.assertIsNone(crossed["incumbent"])
            self.assertEqual(
                crossed["recovery"]["observation_availability"],
                "unavailable_interrupted_before_checkpoint_publication",
            )
            recovered = json.loads(
                (progress / "checkpoint_0005m.json").read_text()
            )
            self.assertTrue(recovered["solver_ended_before_checkpoint"])
            self.assertTrue(recovered["recovery"]["observational_only"])


if __name__ == "__main__":
    unittest.main()

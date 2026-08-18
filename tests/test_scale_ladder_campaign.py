import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import launch_scale_ladder as ladder  # noqa: E402
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
)
from recover_scale_ladder_mip_progress import recover  # noqa: E402


INSTANCE_MANIFEST = (
    REPO_ROOT
    / "data/scale_ladder/instances/scale_ladder_instance_manifest.csv"
)


class ScaleLadderCampaignTests(unittest.TestCase):
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
            "python": "3.12.3",
            "executable": str(Path(sys.executable).resolve()),
            "executable_sha256": sha256_file(Path(sys.executable).resolve()),
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
        self.assertEqual(plan["task_count"], 86)
        self.assertEqual(plan["cg_task_count"], 23)
        self.assertEqual(plan["mip_task_count"], 42)
        self.assertEqual(plan["k40_mip_submission_count"], 0)
        self.assertEqual(
            {key: len(value) for key, value in plan["task_groups"].items()},
            {"SEED": 21, "CG": 23, "MIP_RAW": 21, "MIP_KNOWN": 21},
        )
        self.assertFalse(any(
            job["phase"] == "MIP" and job["scale"] == 40
            for job in plan["jobs"]
        ))
        self.assertTrue(all(
            len(job["job_name"]) <= 15 for job in plan["jobs"]
        ))
        self.assertEqual(
            plan["tariff"]["primary_tariff_sha256"],
            ladder.HISTORICAL_FLAT_SHA256,
        )
        self.assertNotIn("alpha", json.dumps(plan).lower())
        self.assertEqual(len(plan["k40_reuse_slots"]), 4)
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
                "instance": {
                    "path": str(instance), **identities,
                },
            }
            cg_job = {
                **base, "job_key": "cg", "phase": "CG", "arm": None,
                "budget_s": 100, "output": str(cg),
                "telemetry": str(telemetry), "progress_dir": None,
                "snapshot_minutes": [],
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
                "checkout_identity": {"commit": "b" * 40},
                "tariff": {"primary_tariff_sha256": "c" * 64},
                "physics": {}, "trip_identity_schema":
                    "evsp-dr-trip-identity-v1",
                "code_hashes": {}, "python_identity": {},
                "task_groups": {"CG": ["cg"], "MIP_RAW": ["mip"]},
                "jobs": [cg_job, mip_job], "k40_reuse_slots": [],
            }
            raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            plan_sha = hashlib.sha256(raw).hexdigest()
            (root / "approved-plan.json").write_bytes(raw)
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "submitted": True,
                "gate_state": "released",
                "submitted_arrays": {
                    "SEED": "1", "CG": "2",
                    "MIP_RAW": "3", "MIP_KNOWN": "4",
                },
            }))
            for job, artifacts in (
                (
                    cg_job,
                    [cg, journal, Path(str(cg) + ".iters.csv"), telemetry],
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
                "cg_route_weight.png", "mip_incumbent_bound.png",
                "provenance.json",
            }
            self.assertTrue(required <= {
                path.name for path in out.iterdir()
            })
            with (out / "cg_run_summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["censored"], "True")
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

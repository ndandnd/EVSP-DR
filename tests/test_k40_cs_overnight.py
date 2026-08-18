import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import launch_mip_statistics_campaign as launcher  # noqa: E402
import validate_k40_cs_overnight_plan as plan_validator  # noqa: E402
from prepare_k40_giro40_partition import (  # noqa: E402
    EXCLUDED_VARIANTS,
    INCLUDED_DUTIES,
    validate_duty_selection,
)
from validate_k40_cs_overnight_result import (  # noqa: E402
    validate_result,
)
from summarize_mip_statistics import _campaign_path, summarize  # noqa: E402


class K40CSOvernightTests(unittest.TestCase):
    def _candidate(self, root: Path, label: str) -> dict:
        return {
            "candidate_id": f"candidate-{label}",
            "available": True,
            "source_family": "k40_factorial",
            "status_path": str(root / f"{label}.snapshot.json"),
            "status_sha256":
                launcher.K40_CS_FROZEN_HASHES[label]["status"],
            "journal_path": str(root / f"{label}.columns.jsonl"),
            "journal_sha256":
                launcher.K40_CS_FROZEN_HASHES[label]["journal"],
            "instance_path": str(root / "data/duty_unions_big/k40.csv"),
            "instance_sha256": launcher.RAW_K40_INSTANCE_SHA256,
            "tariff_path": str(root / "data/hourly_prices_flat.csv"),
            "tariff_sha256": launcher.RAW_K40_TARIFF_SHA256,
            "source_commit": launcher.RAW_K40_SOURCE_COMMIT,
            "scale": 40,
            "replicate": label[:2],
            "raw_k40_label": label,
            "trip_count": 947,
            "trip_set_sha256": "5" * 64,
            "age_hours": 24.0,
            "actual_wall_s": 86400.0,
            "snapshot_mark_minutes": 1440,
            "stop_reason": "snapshot_m1440",
            "physics": {
                "soc_step": 15.0,
                "block_min": 10.0,
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "min_soc_frac": 0.0,
            },
            "treatment": {
                "master_sense": "cover",
                "initial_pool": "singletons",
            },
            "csv":
                "duty_unions_big/Practice_Custom_DutyUnion_k40_r2.csv",
            "prices_csv": "hourly_prices_flat.csv",
            "certified_rc_optimal": False,
        }

    def _build_plan(self, root: Path) -> dict:
        partition = root / "giro40.json"
        partition.write_text("{}")
        candidates = {
            label: self._candidate(root, label)
            for label in launcher.K40_CS_LABELS
        }
        identity = {
            "expected_commit": launcher.K40_CS_PACKAGING_BASE_COMMIT,
            "reviewed_base_commit": launcher.REVIEWED_BASE,
            "detached": True,
            "branch": "",
            "tracked_clean": True,
            "runtime_artifacts_absent": True,
        }
        payload = {
            "candidates": [],
            "selection_rule": "explicit frozen CS inputs",
            "missing_roots": [],
            "missing_slots": [],
        }
        start = {
            "path": str(partition.resolve()),
            "sha256": launcher.GIRO40_PARTITION_FILE_SHA256,
            "route_count": 40,
            "partition_sha256": "a" * 64,
            "route_set_sha256": "b" * 64,
            "trip_set_sha256": "5" * 64,
            "physical_replay_validated": True,
            "validated_bus_count": 40,
            "expected_full_objective": 4000000.0,
        }
        with (
            patch.object(
                launcher, "_validated_start", return_value=start
            ),
            patch.object(
                launcher,
                "_python_identity",
                return_value={
                    "available": True,
                    "executable": str(Path(sys.executable).resolve()),
                    "executable_sha256": "e" * 64,
                    "version": "3.12.test",
                    "gurobi_version": "test",
                    "identity_sha256": "d" * 64,
                },
            ),
        ):
            return launcher.build_plan(
                payload,
                mode=launcher.K40_CS_OVERNIGHT_MODE,
                campaign="k40-cs-overnight-test",
                start_map={
                    "R1_CS": partition,
                    "R2_CS": partition,
                },
                identity=identity,
                explicit_raw_candidates=candidates,
            )

    def test_four_cell_plan_uses_exact_labels_budgets_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._build_plan(Path(tmp))
            self.assertEqual(len(plan["jobs"]), 4)
            self.assertNotIn("R1_CA", json.dumps(plan))
            self.assertNotIn("R2_CA", json.dumps(plan))
            observed = {
                (job["source"]["raw_k40_label"], job["arm"],
                 job["time_limit_s"])
                for job in plan["jobs"]
            }
            self.assertEqual(observed, {
                ("R1_CS", "RAW", 28800),
                ("R2_CS", "RAW", 28800),
                ("R1_CS", launcher.GIRO40_AUGMENTED, 7200),
                ("R2_CS", launcher.GIRO40_AUGMENTED, 7200),
            })
            self.assertTrue(all(
                len(job["job_name"]) <= 15 for job in plan["jobs"]
            ))
            self.assertEqual(
                len(plan_validator.validate_plan(
                    plan,
                    expected_commit=launcher.K40_CS_PACKAGING_BASE_COMMIT,
                )),
                4,
            )

            mutated = copy.deepcopy(plan)
            giro = next(
                job for job in mutated["jobs"]
                if job["arm"] == launcher.GIRO40_AUGMENTED
            )
            giro["arm"] = "GIRO"
            with self.assertRaisesRegex(
                ValueError, "scientific treatment label"
            ):
                plan_validator.validate_plan(
                    mutated,
                    expected_commit=launcher.K40_CS_PACKAGING_BASE_COMMIT,
                )

            mutated = copy.deepcopy(plan)
            raw = next(
                job for job in mutated["jobs"] if job["arm"] == "RAW"
            )
            raw["validated_start"] = {"sha256": "9" * 64}
            with self.assertRaisesRegex(ValueError, "RAW received"):
                plan_validator.validate_plan(
                    mutated,
                    expected_commit=launcher.K40_CS_PACKAGING_BASE_COMMIT,
                )

            for arm, wrong_limit in (
                ("RAW", 7200),
                (launcher.GIRO40_AUGMENTED, 28800),
            ):
                mutated = copy.deepcopy(plan)
                job = next(
                    item for item in mutated["jobs"]
                    if item["arm"] == arm
                )
                job["time_limit_s"] = wrong_limit
                with self.assertRaisesRegex(
                    ValueError, "wrong Gurobi budget"
                ):
                    plan_validator.validate_plan(
                        mutated,
                        expected_commit=(
                            launcher.K40_CS_PACKAGING_BASE_COMMIT
                        ),
                    )
            for field, message in (
                ("status_sha256", "frozen status hash"),
                ("journal_sha256", "frozen journal hash"),
                ("instance_sha256", "frozen data/tariff hash"),
                ("tariff_sha256", "frozen data/tariff hash"),
            ):
                mutated = copy.deepcopy(plan)
                mutated["jobs"][0]["source"][field] = "0" * 64
                with self.assertRaisesRegex(ValueError, message):
                    plan_validator.validate_plan(
                        mutated,
                        expected_commit=(
                            launcher.K40_CS_PACKAGING_BASE_COMMIT
                        ),
                    )

    def test_weekday_variant_selection_is_exact_and_fail_closed(self):
        literals = set(INCLUDED_DUTIES) | set(EXCLUDED_VARIANTS)
        self.assertEqual(
            validate_duty_selection(list(INCLUDED_DUTIES), literals),
            EXCLUDED_VARIANTS,
        )
        wrong = list(INCLUDED_DUTIES)
        wrong[wrong.index("13316uwt")] = "13316m"
        with self.assertRaisesRegex(ValueError, "duties differ"):
            validate_duty_selection(wrong, literals)
        duplicate = list(INCLUDED_DUTIES)
        duplicate[-1] = duplicate[-2]
        with self.assertRaisesRegex(ValueError, "40 unique"):
            validate_duty_selection(duplicate, literals)
        for count in (39, 41, 43):
            duties = list(INCLUDED_DUTIES)[:count]
            if count > 40:
                duties.extend(f"foreign-{index}" for index in range(count - 40))
            with self.assertRaisesRegex(ValueError, "exactly 40"):
                validate_duty_selection(duties, literals)

    def test_ignored_runtime_bytecode_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "src/__pycache__"
            cache.mkdir(parents=True)
            (cache / "run_exact_pool_mip.cpython-312.pyc").write_bytes(
                b"unreviewed"
            )
            (root / "src/ignored_runtime.py").write_text("raise SystemExit")
            artifacts = launcher._unsafe_runtime_artifacts(root)
            self.assertIn("src/__pycache__", artifacts)
            self.assertIn("src/ignored_runtime.py", artifacts)

    def _partition_fixture(self, root: Path) -> tuple[Path, dict]:
        status = root / "status.json"
        status.write_text(json.dumps({"trip_ids": list(range(40))}))
        routes = [{
            "route": ["PARX_0", trip, "PARX_0"],
            "charging_stops": {
                "stations": [], "cst": [], "cet": [], "kwh": [],
            },
            "continuous_realized_charging_blocks": [],
            "physical_realization": {
                "status": "validated_continuous_injection",
                "continuous_cost_pricing_certified": False,
            },
        } for trip in range(40)]
        payload = {
            "schema": "evsp-dr-k40-giro40-partition-v1",
            "routes": routes,
            "source": launcher.GIRO40_AUGMENTED,
            "route_count": 40,
            "partition_sha256": "a" * 64,
            "route_set_sha256": "b" * 64,
            "continuous_cost_pricing_certified": False,
            "pricing_certificate_scope": "none_for_augmented_routes",
            "physics": {
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "reserve_frac": 0.0,
            },
            "prices_csv": "hourly_prices_flat.csv",
        }
        partition = root / "partition.json"
        partition.write_text(json.dumps(payload))
        candidate = {
            "status_path": str(status),
            "physics": {
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "min_soc_frac": 0.0,
            },
            "prices_csv": "hourly_prices_flat.csv",
            "trip_set_sha256": "5" * 64,
        }
        return partition, candidate

    def _validate_partition_fixture(self, partition, candidate):
        raw = partition.read_bytes()
        with (
            patch.object(
                launcher,
                "GIRO40_PARTITION_FILE_SHA256",
                hashlib.sha256(raw).hexdigest(),
            ),
            patch.object(
                launcher, "GIRO40_PARTITION_SHA256", "a" * 64
            ),
            patch.object(
                launcher, "GIRO40_ROUTE_SET_SHA256", "b" * 64
            ),
            patch.object(
                launcher,
                "_physical_start_validation",
                return_value={
                    "validated_bus_count": 40,
                    "expected_full_objective": 4000000.0,
                },
            ),
        ):
            return launcher._validated_start(partition, candidate)

    def test_partition_rejects_route_count_and_trip_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partition, candidate = self._partition_fixture(root)
            self.assertEqual(
                self._validate_partition_fixture(
                    partition, candidate
                )["route_count"],
                40,
            )
            baseline = json.loads(partition.read_text())
            for count in (39, 41, 43):
                payload = copy.deepcopy(baseline)
                payload["routes"] = (
                    payload["routes"][:count]
                    if count <= 40
                    else payload["routes"] + [
                        copy.deepcopy(payload["routes"][0])
                        for _ in range(count - 40)
                    ]
                )
                payload["route_count"] = count
                partition.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "metadata"):
                    self._validate_partition_fixture(partition, candidate)
            cases = []
            missing = copy.deepcopy(baseline)
            missing["routes"][-1]["route"][1] = 38
            cases.append((missing, "exact partition"))
            repeated = copy.deepcopy(baseline)
            repeated["routes"][0]["route"].insert(2, 0)
            cases.append((repeated, "invalid route"))
            foreign = copy.deepcopy(baseline)
            foreign["routes"][-1]["route"][1] = 999
            cases.append((foreign, "unknown trip"))
            for payload, message in cases:
                partition.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, message):
                    self._validate_partition_fixture(partition, candidate)
            false_certificate = copy.deepcopy(baseline)
            false_certificate["continuous_cost_pricing_certified"] = True
            partition.write_text(json.dumps(false_certificate))
            with self.assertRaisesRegex(ValueError, "metadata"):
                self._validate_partition_fixture(partition, candidate)

    def test_partition_rejects_physically_infeasible_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            partition, candidate = self._partition_fixture(Path(tmp))
            payload = json.loads(partition.read_text())
            raw = partition.read_bytes()
            with (
                patch.object(
                    launcher,
                    "GIRO40_PARTITION_FILE_SHA256",
                    hashlib.sha256(raw).hexdigest(),
                ),
                patch.object(
                    launcher,
                    "GIRO40_PARTITION_SHA256",
                    payload["partition_sha256"],
                ),
                patch.object(
                    launcher,
                    "GIRO40_ROUTE_SET_SHA256",
                    payload["route_set_sha256"],
                ),
                patch.object(
                    launcher,
                    "_physical_start_validation",
                    side_effect=SystemExit("route violates continuous SOC"),
                ),
                self.assertRaisesRegex(SystemExit, "continuous SOC"),
            ):
                launcher._validated_start(partition, candidate)

    def _result_fixture(self, progress: Path, arm: str) -> dict:
        limit = 28800 if arm == "RAW" else 7200
        required = {60, 300, 900, 1800, 3600, 7200}
        if limit == 28800:
            required.update(range(10800, 28801, 3600))
        progress.mkdir()
        for mark in required:
            (progress / f"checkpoint_{mark // 60:04d}m.json").write_text(
                json.dumps({
                    "observed_total_elapsed_s": float(mark),
                    "solver_ended_before_checkpoint": False,
                })
            )
        (progress / "final.json").write_text(json.dumps({
            "kind": "final",
            "final": {"status_name": "TIME_LIMIT"},
        }))
        augmented = arm == launcher.GIRO40_AUGMENTED
        return {
            "source_snapshot_mark_minutes": 1440,
            "status_name": "TIME_LIMIT",
            "incumbent_found": True,
            "continuous_cost_pricing_certified": False,
            "pricing_certificate_scope": "not_certified",
            "optimal_scope": "none",
            "two_stage": {"stage2_executed": False},
            "extra_route_sources": [],
            "physical_pool_preparation_wall_s": 1.0,
            "source_hashing_wall_s": 2.0,
            "gurobi_optimize_wall_s": float(limit),
            "end_to_end_before_publication_s": float(limit + 4),
            "runtime_s": float(limit),
            "node_count": 12.0,
            "solution_count": 2,
            "mip_provenance": {
                "python": "3.12.3",
                "gurobi": "12.0.3",
                "host": "scaglione-node",
                "gurobi_parameters": {
                    "Seed": 0,
                    "seed_source": "gurobi_default",
                    "seed_explicitly_set": False,
                },
                "arguments": {
                    "cover": False,
                    "two_stage": True,
                    "timelimit": limit,
                    "threads": 8,
                    "mipgap": 1e-4,
                    "initial_partition_routes":
                        "giro40.json" if augmented else None,
                }
            },
            "mip_start": {
                "kind":
                    "validated_exact_partition" if augmented else "none",
                "source": "giro40.json" if augmented else None,
                "validated_bus_count": 40 if augmented else None,
                "assigned_mip_start_route_count": 40 if augmented else None,
                "solver_acceptance": {
                    "accepted": True if augmented else None
                },
            },
            "progress": {"termination_signal": None},
            "physical_pool_audit": {
                "total_columns": 100,
                "accepted_columns": 100,
                "rejected_columns": 0,
                "base_pool_column_count": 100,
                "base_pool_ordered_sha256": "1" * 64,
                "added_giro_route_count": 40 if augmented else 0,
                "added_giro_route_set_sha256": "2" * 64,
                "augmented_pool_column_count": 140 if augmented else 100,
                "augmented_pool_ordered_sha256": "3" * 64,
                "assigned_mip_start_route_count": 40 if augmented else 0,
            },
        }

    def test_result_rejects_timing_certification_and_pool_contamination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress = root / "progress"
            result = self._result_fixture(progress, "RAW")
            validate_result(
                result, progress_dir=progress, arm="RAW",
                time_limit_s=28800, source_label="R1_CS",
            )
            changed = copy.deepcopy(result)
            changed.pop("source_hashing_wall_s")
            with self.assertRaisesRegex(ValueError, "timing missing"):
                validate_result(
                    changed, progress_dir=progress, arm="RAW",
                    time_limit_s=28800, source_label="R1_CS",
                )
            changed = copy.deepcopy(result)
            changed["physical_pool_preparation_wall_s"] = -1.0
            with self.assertRaisesRegex(ValueError, "timing missing"):
                validate_result(
                    changed, progress_dir=progress, arm="RAW",
                    time_limit_s=28800, source_label="R1_CS",
                )
            changed = copy.deepcopy(result)
            changed["status_name"] = "INTERRUPTED"
            changed["progress"]["termination_signal"] = "SIGTERM"
            with self.assertRaisesRegex(ValueError, "terminal status"):
                validate_result(
                    changed, progress_dir=progress, arm="RAW",
                    time_limit_s=28800, source_label="R1_CS",
                )
            changed = copy.deepcopy(result)
            changed["continuous_cost_pricing_certified"] = True
            with self.assertRaisesRegex(ValueError, "certificate"):
                validate_result(
                    changed, progress_dir=progress, arm="RAW",
                    time_limit_s=28800, source_label="R1_CS",
                )
            changed = copy.deepcopy(result)
            changed["physical_pool_audit"]["added_giro_route_count"] = 40
            with self.assertRaisesRegex(ValueError, "RAW cell received"):
                validate_result(
                    changed, progress_dir=progress, arm="RAW",
                    time_limit_s=28800, source_label="R1_CS",
                )
            (progress / "checkpoint_0001m.json").unlink()
            with self.assertRaisesRegex(ValueError, "cadence"):
                validate_result(
                    result, progress_dir=progress, arm="RAW",
                    time_limit_s=28800, source_label="R1_CS",
                )

    def test_giro40_result_requires_exact_accepted_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress"
            result = self._result_fixture(
                progress, launcher.GIRO40_AUGMENTED
            )
            validate_result(
                result, progress_dir=progress,
                arm=launcher.GIRO40_AUGMENTED,
                time_limit_s=7200, source_label="R2_CS",
            )
            for field, value in (
                ("validated_bus_count", 39),
                ("validated_bus_count", 41),
                ("validated_bus_count", 43),
            ):
                changed = copy.deepcopy(result)
                changed["mip_start"][field] = value
                with self.assertRaisesRegex(ValueError, "not accepted"):
                    validate_result(
                        changed, progress_dir=progress,
                        arm=launcher.GIRO40_AUGMENTED,
                        time_limit_s=7200, source_label="R2_CS",
                    )
            changed = copy.deepcopy(result)
            changed["mip_start"]["solver_acceptance"]["accepted"] = False
            with self.assertRaisesRegex(ValueError, "not accepted"):
                validate_result(
                    changed, progress_dir=progress,
                    arm=launcher.GIRO40_AUGMENTED,
                    time_limit_s=7200, source_label="R2_CS",
                )

    def test_incomplete_overnight_campaign_cannot_be_summarized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._build_plan(root)
            campaign = root / "campaign"
            campaign.mkdir()
            plan_raw = launcher._canonical(plan)
            (campaign / "approved-plan.json").write_bytes(plan_raw)
            manifest = copy.deepcopy(plan)
            manifest["approval_sha256"] = hashlib.sha256(
                plan_raw
            ).hexdigest()
            manifest["submitted"] = True
            manifest["submission_atomicity"] = (
                "single_atomic_four_task_array_submission"
            )
            for index, job in enumerate(manifest["jobs"], start=1):
                task = index - 1
                job["job_id"] = f"1000_{task}"
                job["submission_state"] = "submitted_array"
                job["slurm_array_name"] = "K40R12RG82"
                job["slurm_array_task_id"] = task
                job["slurm_display_id"] = f"K40R12RG82_{task}"
            bad_manifest = copy.deepcopy(manifest)
            for job in bad_manifest["jobs"]:
                job["job_id"] = "1000_3"
            (campaign / "campaign.json").write_text(
                json.dumps(bad_manifest)
            )
            with self.assertRaisesRegex(
                ValueError, "approved array task"
            ):
                summarize(campaign, root / "bad-summary")
            (campaign / "campaign.json").write_text(
                json.dumps(manifest)
            )
            with self.assertRaisesRegex(
                ValueError, "no completed result"
            ):
                summarize(campaign, root / "summary")

    def test_staged_archive_relocates_only_campaign_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            declared = root / "live/campaign"
            staged = root / "bundle/campaign"
            manifest = {"campaign_root": str(declared)}
            self.assertEqual(
                _campaign_path(
                    staged, manifest, declared / "progress/cell/final.json"
                ),
                staged / "progress/cell/final.json",
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                _campaign_path(
                    staged, manifest, root / "unrelated/result.json"
                )


if __name__ == "__main__":
    unittest.main()

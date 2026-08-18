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
import validate_raw_k40_mip_plan as raw_validator  # noqa: E402
from mip_statistics_inventory import (  # noqa: E402
    inventory,
    representative_candidates,
    select_age_candidate,
    validate_candidate,
)


class MIPStatisticsCampaignTests(unittest.TestCase):
    def _raw_k40_candidate(self, label: str, root: Path):
        spec = launcher.RAW_K40_SPECS[label]
        initial_pool = spec["initial_pool"]
        return {
            "candidate_id": f"candidate-{label}",
            "available": True,
            "source_family": "k40_factorial",
            "status_path": str(
                root / spec["campaign"] / spec["filename"]
            ),
            "status_sha256": ("1" if label.endswith("CA") else "2") * 64,
            "journal_path": str(root / f"{label}.columns.jsonl"),
            "journal_sha256": ("3" if label.startswith("R1") else "4") * 64,
            "instance_path": str(root / "data/duty_unions_big/k40.csv"),
            "instance_sha256": launcher.RAW_K40_INSTANCE_SHA256,
            "tariff_path": str(root / "data/hourly_prices_flat.csv"),
            "tariff_sha256": launcher.RAW_K40_TARIFF_SHA256,
            "source_commit": launcher.RAW_K40_SOURCE_COMMIT,
            "scale": 40,
            "replicate": "r2",
            "trip_count": 947,
            "trip_set_sha256": "5" * 64,
            "age_hours": 24.0,
            "actual_wall_s": 86400.0,
            "snapshot_mark_minutes": 1440,
            "stop_reason": "snapshot_m1440",
            "physics": {
                "soc_step": 15.0,
                "block_min": 10,
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "min_soc_frac": 0.0,
            },
            "treatment": {
                "master_sense": "cover",
                "initial_pool": initial_pool,
            },
            "csv": "duty_unions_big/Practice_Custom_DutyUnion_k40_r2.csv",
            "prices_csv": "hourly_prices_flat.csv",
            "certified_rc_optimal": False,
        }

    def _candidate(
        self,
        root: Path,
        *,
        scale=8,
        replicate=1,
        trips=8,
        mark=60,
    ):
        data = root / "data"
        pool_root = root / "results/repool_small"
        instance_rel = (
            f"synthetic/Practice_Custom_k{scale}_r{replicate}.csv"
        )
        tariff_rel = "hourly_prices_flat.csv"
        instance = data / instance_rel
        tariff = data / tariff_rel
        instance.parent.mkdir(parents=True, exist_ok=True)
        pool_root.mkdir(parents=True, exist_ok=True)
        instance.write_text(f"instance-{scale}-{replicate}-{trips}\n")
        tariff.write_text("flat-prices\n")
        instance_sha = hashlib.sha256(instance.read_bytes()).hexdigest()
        tariff_sha = hashlib.sha256(tariff.read_bytes()).hexdigest()
        status_path = pool_root / (
            f"k{scale}_r{replicate}.m{mark}.snapshot.json"
        )
        journal = Path(str(status_path) + ".columns.jsonl")
        journal.write_text(json.dumps({
            "trips": list(range(trips)),
            "cost": 100000.0,
            "route_nodes": ["PARX_0", *range(trips), "PARX_0"],
            "charging_stops": {
                "stations": [], "cst": [], "cet": [], "kwh": [],
            },
        }) + "\n")
        status = {
            "csv": instance_rel,
            "prices_csv": tariff_rel,
            "soc_step": 15.0,
            "block_min": 10,
            "g_kwh": 300.0,
            "charge_kw": 300.0,
            "min_soc_frac": 0.0,
            "master_sense": "cover",
            "initial_pool": "singletons",
            "trip_ids": list(range(trips)),
            "columns": 1,
            "columns_journal": str(journal),
            "wall_s": mark * 60.0,
            "snapshot_mark_minutes": float(mark),
            "stop_reason": f"snapshot_m{mark}",
            "certified_rc_optimal": False,
            "provenance": {
                "git_commit": "a" * 40,
                "instance_sha256": instance_sha,
                "prices_sha256": tariff_sha,
            },
        }
        status_path.write_text(json.dumps(status))
        return status_path, data

    def _start(self, root: Path, trips: int):
        path = root / "giro.json"
        path.write_text(json.dumps({
            "routes": [{
                "route": ["PARX_0", trip, "PARX_0"],
                "charging_stops": {
                    "stations": [], "cst": [], "cet": [], "kwh": [],
                },
            } for trip in range(trips)],
            "infeasible": [],
            "source": "rerealized",
            "physics": {
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "reserve_frac": 0.0,
            },
            "prices_csv": "hourly_prices_flat.csv",
        }))
        return path

    def test_inventory_validates_hashes_and_marks_missing_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status, data = self._candidate(root)
            payload = inventory(
                {
                    "repool_small": status.parent,
                    "exact_big": root / "missing-exact-big",
                },
                data_roots=[data],
            )
            self.assertEqual(len(payload["candidates"]), 1)
            candidate = payload["candidates"][0]
            self.assertEqual(candidate["scale"], 8)
            self.assertEqual(candidate["trip_count"], 8)
            self.assertEqual(candidate["age_hours"], 1.0)
            self.assertTrue(payload["missing_roots"])
            Path(candidate["instance_path"]).write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "instance bytes"):
                validate_candidate(
                    status,
                    source_family="repool_small",
                    data_roots=[data],
                )

    def test_inventory_rejects_row_incomplete_pool_and_live_age_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status, data = self._candidate(root, trips=2)
            payload = json.loads(status.read_text())
            journal = Path(payload["columns_journal"])
            journal.write_text(json.dumps({
                "trips": [0], "cost": 100000.0,
            }) + "\n")
            with self.assertRaisesRegex(ValueError, "cover every"):
                validate_candidate(
                    status,
                    source_family="repool_small",
                    data_roots=[data],
                )
            live_candidate = {
                "scale": 20,
                "source_family": "fresh_preparation",
                "age_hours": 1.0,
                "status_path": str(root / "k20_r1.json"),
                "trip_count": 2,
                "instance_sha256": "a" * 64,
                "replicate": "r1",
                "candidate_id": "live",
            }
            self.assertIsNone(select_age_candidate(
                [live_candidate], scale=20, target=1
            ))

    def test_representative_selection_uses_lower_median_trip_count(self):
        payload = {
            "candidates": [{
                "scale": 8,
                "source_family": "repool_small",
                "instance_sha256": f"{index:064x}",
                "trip_set_sha256": f"{index + 10:064x}",
                "replicate": f"r{index}",
                "trip_count": count,
                "age_hours": 1.0,
                "candidate_id": f"candidate-{index}",
            } for index, count in enumerate((5, 7, 9), start=1)]
        }
        selected = representative_candidates(payload)
        self.assertEqual(selected[8]["trip_count"], 7)

    def test_pilot_keeps_raw_and_giro_column_sets_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status, data = self._candidate(root, trips=8)
            payload = inventory(
                {"repool_small": status.parent},
                data_roots=[data],
            )
            start = self._start(root, 8)
            identity = {
                "expected_commit": "b" * 40,
                "reviewed_base_commit": launcher.REVIEWED_BASE,
                "detached": True,
                "branch": "",
                "tracked_clean": True,
            }
            with (
                patch.object(launcher, "REPO_ROOT", root),
                patch.object(
                    launcher,
                    "_physical_start_validation",
                    return_value={
                        "validated_bus_count": 8,
                        "expected_full_objective": 800000.0,
                    },
                ),
                patch.object(launcher, "CODE_PATHS", ()),
                patch.object(
                    launcher,
                    "_python_identity",
                    return_value={
                        "available": True,
                        "executable": str(Path(sys.executable).resolve()),
                        "executable_sha256": "e" * 64,
                        "version": "3.12.test",
                        "gurobi_version": "test",
                    },
                ),
            ):
                plan = launcher.build_plan(
                    payload,
                    mode="pilot",
                    campaign="pilot-test",
                    start_map={"8": start},
                    identity=identity,
                )
            jobs = [job for job in plan["jobs"] if job["scale"] == 8]
            self.assertEqual({job["arm"] for job in jobs}, {"RAW", "GIRO"})
            raw = next(job for job in jobs if job["arm"] == "RAW")
            giro = next(job for job in jobs if job["arm"] == "GIRO")
            self.assertIsNone(raw["validated_start"])
            self.assertFalse(raw["augmentation_changes_column_set"])
            self.assertIsNotNone(giro["validated_start"])
            self.assertTrue(giro["augmentation_changes_column_set"])
            self.assertTrue(all(len(job["job_name"]) <= 15 for job in jobs))
            self.assertIn("scale", jobs[0])
            self.assertIn("first-N stress instance", json.dumps(
                plan["fresh_exact_cg_preparations"]
            ))

    def test_explicit_raw_k40_plan_has_four_unaugmented_eight_hour_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = {
                label: self._raw_k40_candidate(label, root)
                for label in launcher.RAW_K40_SPECS
            }
            for label, candidate in candidates.items():
                candidate["replicate"] = launcher.RAW_K40_SPECS[label][
                    "replicate"
                ]
                candidate["raw_k40_label"] = label
            payload = {
                "candidates": [],
                "selection_rule": "irrelevant",
                "missing_roots": [{"source_family": "unrelated"}],
                "missing_slots": [{"source_family": "unrelated"}],
            }
            identity = {
                "expected_commit": "b" * 40,
                "reviewed_base_commit": launcher.REVIEWED_BASE,
                "detached": True,
                "branch": "",
                "tracked_clean": True,
            }
            with (
                patch.object(launcher, "REPO_ROOT", root),
                patch.object(launcher, "CODE_PATHS", ()),
                patch.object(
                    launcher,
                    "_python_identity",
                    return_value={
                        "available": True,
                        "executable": str(Path(sys.executable).resolve()),
                        "executable_sha256": "e" * 64,
                        "version": "3.12.test",
                        "gurobi_version": "test",
                    },
                ),
            ):
                plan = launcher.build_plan(
                    payload,
                    mode="raw_k40",
                    campaign="raw-k40-test",
                    start_map={},
                    identity=identity,
                    explicit_raw_candidates=candidates,
                )
            self.assertFalse(plan["blocked"])
            self.assertEqual(len(plan["jobs"]), 4)
            self.assertEqual(plan["fresh_exact_cg_preparations"], [])
            self.assertEqual(set(plan["selected_candidates"]), {
                "R1_CA", "R1_CS", "R2_CA", "R2_CS",
            })
            self.assertFalse(plan["raw_k40_guards"]["giro_columns_allowed"])
            for job in plan["jobs"]:
                self.assertEqual(job["arm"], "RAW")
                self.assertEqual(job["matrix"], "raw_k40")
                self.assertEqual(job["time_limit_s"], 28800)
                self.assertIsNone(job["validated_start"])
                self.assertIsNone(job["staged_start"])
                self.assertEqual(job["execution"]["arm"], "RAW")
                self.assertEqual(job["execution"]["matrix"], "raw_k40")
                self.assertEqual(job["execution"]["time_limit_s"], 28800)
                self.assertLessEqual(len(job["job_name"]), 15)
                self.assertIn("K40", job["job_name"])
            summary = raw_validator.validate_plan(
                plan, expected_commit="b" * 40
            )
            self.assertEqual(
                {row["label"] for row in summary},
                {"R1_CA", "R1_CS", "R2_CA", "R2_CS"},
            )

            plan["jobs"][0]["validated_start"] = {"path": "giro.json"}
            with self.assertRaisesRegex(ValueError, "external partition start"):
                raw_validator.validate_plan(
                    plan, expected_commit="b" * 40
                )

    def test_completed_physical_smoke_refuses_changed_runtime_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = {
                label: self._raw_k40_candidate(label, root)
                for label in launcher.RAW_K40_SPECS
            }
            for label, candidate in candidates.items():
                candidate["replicate"] = launcher.RAW_K40_SPECS[label][
                    "replicate"
                ]
                candidate["raw_k40_label"] = label
            payload = {
                "candidates": [],
                "selection_rule": "irrelevant",
                "missing_roots": [],
                "missing_slots": [],
            }
            identity = {
                "expected_commit": "b" * 40,
                "reviewed_base_commit": launcher.REVIEWED_BASE,
                "detached": True,
                "branch": "",
                "tracked_clean": True,
            }
            with (
                patch.object(
                    launcher,
                    "_python_identity",
                    return_value={
                        "available": True,
                        "executable": str(Path(sys.executable).resolve()),
                        "executable_sha256": "e" * 64,
                        "version": "3.12.test",
                        "gurobi_version": "test",
                    },
                ),
                self.assertRaisesRegex(
                    ValueError, "runtime differs from reviewed commit"
                ),
            ):
                launcher.build_plan(
                    payload,
                    mode="raw_k40_smoke",
                    campaign="raw-k40-smoke-test",
                    start_map={},
                    identity=identity,
                    explicit_raw_candidates=candidates,
                )

    def test_raw_k40_candidate_resolution_preserves_campaign_and_initializer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assignments = {}
            candidates = {}
            for label, spec in launcher.RAW_K40_SPECS.items():
                path = root / spec["campaign"] / spec["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                assignments[label] = path
                candidates[str(path.resolve())] = self._raw_k40_candidate(
                    label, root
                )

            def validated(path, **_kwargs):
                return candidates[str(Path(path).resolve())]

            with patch.object(
                launcher, "validate_candidate", side_effect=validated
            ):
                selected = launcher.resolve_raw_k40_candidates(
                    assignments, data_roots=[root / "data"]
                )
            self.assertEqual(set(selected), set(launcher.RAW_K40_SPECS))
            self.assertEqual(selected["R1_CA"]["replicate"], "R1")
            self.assertEqual(selected["R2_CS"]["replicate"], "R2")
            self.assertEqual(
                selected["R1_CA"]["treatment"]["initial_pool"],
                "artificial",
            )
            self.assertEqual(
                selected["R2_CS"]["treatment"]["initial_pool"],
                "singletons",
            )

            bad = dict(assignments)
            bad["R1_CA"] = assignments["R1_CS"]
            with self.assertRaisesRegex(ValueError, "distinct"):
                launcher.resolve_raw_k40_candidates(
                    bad, data_roots=[root / "data"]
                )

    def test_smoke_candidate_resolution_binds_explicit_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assignments = {}
            journals = {}
            candidates = {}
            status_hashes = {}
            journal_hashes = {}
            for label in launcher.RAW_K40_SPECS:
                status = root / f"{label}.snapshot.json"
                journal = root / f"{label}.columns.jsonl"
                status.touch()
                journal.touch()
                candidate = self._raw_k40_candidate(label, root)
                candidate.update({
                    "status_path": str(status),
                    "journal_path": str(journal),
                })
                assignments[label] = status
                journals[label] = journal
                candidates[str(status.resolve())] = candidate
                status_hashes[label] = candidate["status_sha256"]
                journal_hashes[label] = candidate["journal_sha256"]

            def validated(path, **_kwargs):
                return candidates[str(path.resolve())]

            with patch.object(
                launcher, "validate_candidate", side_effect=validated
            ):
                selected = launcher.resolve_raw_k40_candidates(
                    assignments,
                    data_roots=[root / "data"],
                    enforce_frozen_path=False,
                    expected_status_sha256=status_hashes,
                    journal_assignments=journals,
                    expected_journal_sha256=journal_hashes,
                )
            self.assertEqual(set(selected), set(launcher.RAW_K40_SPECS))
            bad_hashes = dict(status_hashes)
            bad_hashes["R1_CA"] = "0" * 64
            with (
                patch.object(
                    launcher, "validate_candidate", side_effect=validated
                ),
                self.assertRaisesRegex(ValueError, "status SHA-256"),
            ):
                launcher.resolve_raw_k40_candidates(
                    assignments,
                    data_roots=[root / "data"],
                    enforce_frozen_path=False,
                    expected_status_sha256=bad_hashes,
                    journal_assignments=journals,
                    expected_journal_sha256=journal_hashes,
                )

    def test_missing_or_partial_start_blocks_giro(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status, data = self._candidate(root, trips=8)
            payload = inventory(
                {"repool_small": status.parent}, data_roots=[data]
            )
            identity = {
                "expected_commit": "b" * 40,
                "reviewed_base_commit": launcher.REVIEWED_BASE,
                "detached": True,
                "branch": "",
                "tracked_clean": True,
            }
            with (
                patch.object(launcher, "REPO_ROOT", root),
                patch.object(launcher, "CODE_PATHS", ()),
                patch.object(
                    launcher,
                    "_python_identity",
                    return_value={
                        "available": True,
                        "executable": str(Path(sys.executable).resolve()),
                        "executable_sha256": "e" * 64,
                        "version": "3.12.test",
                        "gurobi_version": "test",
                    },
                ),
            ):
                plan = launcher.build_plan(
                    payload,
                    mode="pilot",
                    campaign="blocked",
                    start_map={},
                    identity=identity,
                )
            giro = next(job for job in plan["jobs"] if job["arm"] == "GIRO")
            self.assertIn(
                "validated_giro_start_missing", giro["blocked_reasons"]
            )
            self.assertTrue(plan["blocked"])

    def test_duplicate_campaign_and_blocked_plan_refuse_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_root = root / "campaign"
            campaign_root.mkdir()
            plan = {
                "jobs": [],
                "blocked": False,
                "campaign_root": str(campaign_root),
                "log_root": str(root / "logs"),
            }
            with self.assertRaisesRegex(SystemExit, "no runnable jobs"):
                launcher._stage_and_submit(plan, "a" * 64)
            plan["jobs"] = [{"blocked_reasons": ["missing"]}]
            plan["blocked"] = True
            with self.assertRaisesRegex(SystemExit, "blocked"):
                launcher._stage_and_submit(plan, "a" * 64)
            plan["blocked"] = False
            with self.assertRaisesRegex(SystemExit, "already exists"):
                launcher._stage_and_submit(plan, "a" * 64)

    def test_worker_is_strict_partition_and_whitelisted(self):
        text = (
            REPO_ROOT / "src/submit_mip_statistics.sub"
        ).read_text()
        self.assertIn("#SBATCH --partition=scaglione", text)
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertIn("#SBATCH --no-requeue", text)
        self.assertIn("#SBATCH --signal=B:USR1@180", text)
        self.assertIn("--two-stage", text)
        self.assertNotIn("--cover", text)
        self.assertIn("APPROVED_PLAN", text)
        self.assertIn("REQUESTED_CELL", text)
        self.assertNotIn("JOB_SPEC", text)
        self.assertNotIn("--extra-routes", text)
        self.assertIn("raw_k40 source initializer/label mismatch", text)
        self.assertIn('matrix in {"raw_k40", "raw_k40_smoke"}', text)
        self.assertIn("raw_k40 physical pool gate rejected columns", text)
        self.assertIn("selected route block hash mismatch", text)
        self.assertIn("raw_k40 preprocessing time is missing", text)
        self.assertIn("raw_k40 Gurobi time is missing", text)
        self.assertIn(
            'unset EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256', text
        )
        raw_launcher = (
            REPO_ROOT / "src/submit_raw_k40_mip_campaign.sub"
        ).read_text()
        self.assertIn("--mode raw_k40", raw_launcher)
        self.assertIn("--no-requeue", raw_launcher)
        self.assertNotIn("--giro-start", raw_launcher)
        self.assertIn("validate_raw_k40_mip_plan.py", raw_launcher)
        self.assertIn("Slurm-provided PATH is required", raw_launcher)
        self.assertIn("git sha256sum squeue sacct sbatch", raw_launcher)
        self.assertNotIn(
            "export PATH=/usr/local/bin:/usr/bin:/bin", raw_launcher
        )
        launcher_text = (
            REPO_ROOT / "src/launch_mip_statistics_campaign.py"
        ).read_text()
        self.assertNotIn("--export=ALL", launcher_text)
        self.assertLess(
            launcher_text.index("Phase 1: stage"),
            launcher_text.index("Phase 2: only now"),
        )
        self.assertIn('"--hold"', launcher_text)
        self.assertIn('"--array=0-3"', launcher_text)
        self.assertIn('"__ARRAY__"', launcher_text)
        self.assertIn('["scontrol", "release"', launcher_text)
        self.assertIn("K40R12RG82", launcher_text)
        self.assertIn("single_atomic_four_task_array_submission",
                      launcher_text)

    def test_campaign_name_escape_and_export_injection_are_rejected(self):
        payload = {
            "candidates": [],
            "selection_rule": "median",
            "missing_roots": [],
            "missing_slots": [],
        }
        identity = {
            "expected_commit": "b" * 40,
            "reviewed_base_commit": launcher.REVIEWED_BASE,
            "detached": True,
            "branch": "",
            "tracked_clean": True,
        }
        with self.assertRaisesRegex(ValueError, "safe relative"):
            launcher.build_plan(
                payload,
                mode="pilot",
                campaign="/tmp/escape",
                start_map={},
                identity=identity,
            )
        with (
            patch.object(launcher, "CODE_PATHS", ()),
            patch.object(
                launcher,
                "_python_identity",
                return_value={
                    "available": True,
                    "executable": "/safe/python",
                    "executable_sha256": "e" * 64,
                    "version": "3.12",
                    "gurobi_version": "test",
                },
            ),
            patch.dict(launcher.os.environ, {"HOME": "/tmp/x,EVIL=1"}),
            self.assertRaisesRegex(ValueError, "unsafe"),
        ):
            launcher.build_plan(
                payload,
                mode="pilot",
                campaign="safe-campaign",
                start_map={},
                identity=identity,
            )


if __name__ == "__main__":
    unittest.main()

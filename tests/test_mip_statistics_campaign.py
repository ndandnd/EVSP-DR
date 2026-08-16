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
from mip_statistics_inventory import (  # noqa: E402
    inventory,
    representative_candidates,
    select_age_candidate,
    validate_candidate,
)


class MIPStatisticsCampaignTests(unittest.TestCase):
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
        launcher_text = (
            REPO_ROOT / "src/launch_mip_statistics_campaign.py"
        ).read_text()
        self.assertNotIn("--export=ALL", launcher_text)
        self.assertLess(
            launcher_text.index("Phase 1: stage"),
            launcher_text.index("Phase 2: only now"),
        )

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

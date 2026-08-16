import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from summarize_mip_statistics import summarize  # noqa: E402
from portable_bundle import inspect_bundle  # noqa: E402


class MIPStatisticsSummaryTests(unittest.TestCase):
    def _campaign(self, root: Path):
        campaign = root / "campaign"
        campaign.mkdir()
        jobs = []
        for arm in ("RAW", "GIRO"):
            cell = f"k40_r1_{arm.lower()}"
            progress = campaign / "progress" / cell
            progress.mkdir(parents=True)
            output = campaign / f"{cell}.json"
            status_path = campaign / f"{cell}.source.json"
            status_path.write_text(json.dumps({
                "trip_ids": list(range(40)),
            }))
            source = {
                "status_sha256": hashlib.sha256(
                    status_path.read_bytes()
                ).hexdigest(),
                "journal_sha256": "b" * 64,
            }
            checkpoint = {
                "schema": "evsp-dr-mip-convergence-v1",
                "stage": "fleet",
                "checkpoint_elapsed_s": 300.0,
                "observed_total_elapsed_s": 300.0,
                "solver_ended_before_checkpoint": False,
                "incumbent_state":
                    "reused_most_recent_earlier_incumbent",
                "incumbent": {
                    "fleet": 40 if arm == "GIRO" else 41,
                    "objective": 4000000.0,
                    "route_vector_sha256": "c" * 64,
                },
                "first_feasible_incumbent_s": 0.0,
                "latest_statistics": {
                    "statistics_incumbent_fleet": (
                        40 if arm == "GIRO" else 41
                    ),
                    "fleet_bound": 40.0,
                    "objective_bound": None,
                    "fleet_gap": 0.0 if arm == "GIRO" else 1 / 41,
                    "node_count": 10,
                    "solution_count": 1,
                },
                "latest_statistics_observed_s": 300.0,
                "incumbent_improvements": [{
                    "stage": "fleet",
                    "total_elapsed_s": 0.0 if arm == "GIRO" else 200.0,
                    "stage_elapsed_s": 0.0 if arm == "GIRO" else 200.0,
                    "fleet": 40 if arm == "GIRO" else 41,
                    "objective": 4000000.0,
                    "selected_route_indices": list(range(40)),
                    "route_vector_sha256": "c" * 64,
                    "event": "first_feasible",
                }],
                "metadata": {
                    "source_result_sha256": source["status_sha256"],
                    "source_journal_sha256": source["journal_sha256"],
                    "source_initial_partition_sha256": (
                        "d" * 64 if arm == "GIRO" else None
                    ),
                    "experiment_arm": "D" if arm == "GIRO" else "B",
                    "git_commit": "e" * 40,
                    "parameters": {
                        "two_stage": True,
                        "cover": False,
                        "threads": 8,
                    },
                },
            }
            (progress / "checkpoint_0005m.json").write_text(
                json.dumps(checkpoint)
            )
            result = {
                "partitioning": True,
                "experiment_arm": "D" if arm == "GIRO" else "B",
                "incumbent_found": True,
                "status_name": "OPTIMAL",
                "buses": 40,
                "mip_obj": 4000000.0,
                "mip_bound": 4000000.0,
                "mip_gap": 0.0,
                "fleet_bound": 40.0,
                "fleet_proven": arm == "GIRO",
                "runtime_s": 300.0,
                "optimal_scope": (
                    "fleet_only" if arm == "GIRO" else "none"
                ),
                "source_result_sha256": source["status_sha256"],
                "source_journal_sha256": source["journal_sha256"],
                "mip_start": (
                    {
                        "kind": "validated_exact_partition",
                        "source_sha256": "d" * 64,
                    }
                    if arm == "GIRO" else {
                        "kind": "greedy_pool_partition",
                        "source_sha256": None,
                    }
                ),
                "mip_provenance": {
                    "expected_git_commit": "e" * 40,
                    "observed_git_commit": "e" * 40,
                    "arguments": {
                        "two_stage": True,
                        "cover": False,
                        "threads": 8,
                    },
                },
                "selected_routes": [{
                    "trips": [trip],
                    "charging_stops": {},
                } for trip in range(40)],
            }
            output.write_text(json.dumps(result))
            (progress / "final.json").write_text(json.dumps({
                "final": {
                    "incumbent_found": True,
                    "buses": result["buses"],
                    "fleet_proven": result["fleet_proven"],
                }
            }))
            job = {
                "cell_id": cell,
                "scale": 40,
                "replicate": "r1",
                "arm": arm,
                "augmentation_changes_column_set": arm == "GIRO",
                "age_hours": 6.0,
                "budget_hours": 2,
                "source": source,
                "validated_start": (
                    {"route_count": 40, "sha256": "d" * 64}
                    if arm == "GIRO" else None
                ),
                "execution": {
                    "cell_id": cell,
                    "status": str(status_path),
                    "status_sha256": source["status_sha256"],
                },
                "progress_dir": str(progress),
                "output": str(output),
            }
            jobs.append(job)
        approved = {
            "campaign": "summary-test",
            "checkout_identity": {"expected_commit": "e" * 40},
            "jobs": jobs,
        }
        plan_raw = json.dumps(
            approved, sort_keys=True, separators=(",", ":")
        ).encode()
        (campaign / "approved-plan.json").write_bytes(plan_raw)
        (campaign / "campaign.json").write_text(json.dumps({
            "campaign": "summary-test",
            "approval_sha256": hashlib.sha256(plan_raw).hexdigest(),
            "checkout_identity": {"expected_commit": "e" * 40},
            "jobs": jobs,
        }))
        return campaign

    def test_summary_emits_curves_and_separate_raw_giro_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = self._campaign(root)
            output = root / "summary"
            result = summarize(campaign, output)
            self.assertEqual(result["jobs"], 2)
            self.assertEqual(
                inspect_bundle(output)["state"], "complete_valid"
            )
            with (output / "job_final.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            with (output / "checkpoint_long.csv").open(
                    newline="") as handle:
                checkpoint_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {
                    row["arm"]: row["statistics_incumbent_fleet"]
                    for row in checkpoint_rows
                },
                {"RAW": "41", "GIRO": "40"},
            )
            self.assertEqual({row["arm"] for row in rows}, {"RAW", "GIRO"})
            raw = next(row for row in rows if row["arm"] == "RAW")
            giro = next(row for row in rows if row["arm"] == "GIRO")
            self.assertEqual(raw["route_space_scope"], "finite_raw_cg_pool")
            self.assertEqual(
                giro["route_space_scope"], "finite_augmented_pool"
            )
            self.assertEqual(giro["giro_target_buses"], "40")
            for stem in (
                "buses_vs_mip_time",
                "incumbent_fleet_bound_curves",
                "cg_age_final_buses_heatmap",
            ):
                self.assertTrue((output / f"{stem}.png").is_file())
                self.assertTrue((output / f"{stem}.pdf").is_file())
            with self.assertRaises(FileExistsError):
                summarize(campaign, output)

    def test_legacy_checkpoint_without_statistics_incumbent_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = self._campaign(root)
            for checkpoint_path in campaign.glob(
                    "progress/*/checkpoint_*.json"):
                checkpoint = json.loads(checkpoint_path.read_text())
                checkpoint["latest_statistics"].pop(
                    "statistics_incumbent_fleet"
                )
                checkpoint_path.write_text(json.dumps(checkpoint))
            output = root / "legacy-summary"
            summarize(campaign, output)
            with (output / "checkpoint_long.csv").open(
                    newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(
                row["statistics_incumbent_fleet"] == "" for row in rows
            ))

    def test_statistics_incumbent_gap_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = self._campaign(root)
            checkpoint_path = next(
                campaign.glob("progress/*raw/checkpoint_*.json")
            )
            checkpoint = json.loads(checkpoint_path.read_text())
            checkpoint["latest_statistics"][
                "statistics_incumbent_fleet"
            ] = 42
            checkpoint_path.write_text(json.dumps(checkpoint))
            with self.assertRaisesRegex(
                ValueError, "statistics fleet gap mismatch"
            ):
                summarize(campaign, root / "summary")

    def test_covering_result_is_rejected_as_integer_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = self._campaign(root)
            manifest = json.loads(
                (campaign / "campaign.json").read_text()
            )
            result_path = Path(manifest["jobs"][0]["output"])
            result = json.loads(result_path.read_text())
            result["partitioning"] = False
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "covering"):
                summarize(campaign, root / "summary")

    def test_swapped_arm_and_inconsistent_proof_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = self._campaign(root)
            manifest_path = campaign / "campaign.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["jobs"][0]["output"], manifest["jobs"][1]["output"] = (
                manifest["jobs"][1]["output"],
                manifest["jobs"][0]["output"],
            )
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "approved plan"):
                summarize(campaign, root / "swapped")

            (root / "second").mkdir()
            campaign = self._campaign(root / "second")
            manifest = json.loads(
                (campaign / "campaign.json").read_text()
            )
            raw_path = Path(manifest["jobs"][0]["output"])
            result = json.loads(raw_path.read_text())
            result.update({
                "incumbent_found": False,
                "buses": None,
                "fleet_proven": True,
                "fleet_bound": 40.0,
                "optimal_scope": "none",
            })
            raw_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(
                ValueError, "finite-pool proof|incumbent/bus"
            ):
                summarize(campaign, root / "bad-proof")

            (root / "third").mkdir()
            campaign = self._campaign(root / "third")
            manifest = json.loads(
                (campaign / "campaign.json").read_text()
            )
            result_path = Path(manifest["jobs"][1]["output"])
            result = json.loads(result_path.read_text())
            result.update({
                "fleet_proven": True,
                "optimal_scope": "full_pool_lexicographic",
                "status_name": "OPTIMAL",
                "mip_gap": 0.0,
                "absolute_cost_gap": 0.0,
                "mip_bound": result["mip_obj"],
                "two_stage": {
                    "stage2_executed": True,
                    "stage2_status": 2,
                    "stage1_buses": result["buses"],
                    "stage2_variable_obj": 1.0,
                    "stage2_variable_bound": 1.0,
                    "stage2_absolute_gap": 0.0,
                },
            })
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "cost-stage closure"):
                summarize(campaign, root / "bad-cost-proof")


if __name__ == "__main__":
    unittest.main()

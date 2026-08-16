import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import recover_k40_factorial_mip_campaign as recovery  # noqa: E402
import monitor_k40_factorial_mip_screen as k40_monitor  # noqa: E402
from portable_bundle import inspect_bundle  # noqa: E402


class K40RecoveryTests(unittest.TestCase):
    @staticmethod
    def _fake_merge(_routes, trips, *_args, **_kwargs):
        variable = 1746.666836618
        merged = [{
            "trips": [trip],
            "cost": 100000.0 + variable / 40.0,
            "route_nodes": ["PARX_0", trip, "PARX_0"],
            "charging_stops": {},
        } for trip in trips]
        return merged, list(range(40)), {
            "actual_start_column_hashes": [
                f"{trip:064x}" for trip in range(40)
            ]
        }

    def setUp(self):
        self.enterContext(patch(
            "k40_factorial_mip_result.merge_validated_partition_start",
            side_effect=self._fake_merge,
        ))
        self.enterContext(patch(
            "k40_factorial_mip_result.validate_final_selected_routes"
        ))

    @staticmethod
    def _source_sha(campaign):
        return hashlib.sha256(
            (campaign / "campaign.json").read_bytes()
        ).hexdigest()

    def _fixture(self, root: Path, *, malformed_raw=False):
        repo = root / "repo"
        campaign = (
            repo / "src/results/k40_factorial_mip"
            / "k40fx_mip2h_20260816T035618Z"
        )
        campaign.mkdir(parents=True)
        (repo / "src").mkdir(exist_ok=True)
        runner = repo / "src/run_exact_pool_mip.py"
        runner.write_text("reviewed runner\n")
        runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
        worker = campaign / "input/worker.sub"
        worker.parent.mkdir()
        worker.write_text("reviewed worker\n")
        worker_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
        common = campaign / "input/common"
        common.mkdir(parents=True)
        start = common / "start.json"
        start.write_text(json.dumps({
            "routes": [{
                "route": ["PARX_0", trip, "PARX_0"],
                "charging_stops": {},
            } for trip in range(40)],
            "infeasible": [],
        }))
        start_sha = hashlib.sha256(start.read_bytes()).hexdigest()
        jobs = []
        cells = [
            (replicate, treatment, mark)
            for replicate in ("R1", "R2")
            for treatment in ("CA", "CS")
            for mark in (360, 720, 1440)
        ]
        for index, (replicate, treatment, mark) in enumerate(cells):
            label = f"{replicate}_{treatment}_m{mark}"
            cell = campaign / "input" / label
            cell.mkdir()
            status = cell / "pool.snapshot.json"
            journal = Path(str(status) + ".columns.jsonl")
            instance = cell / "instance.csv"
            prices = cell / "prices.csv"
            source_status = {
                "csv": "instance.csv",
                "prices_csv": "prices.csv",
                "trip_ids": list(range(40)),
                "columns": 40,
                "columns_journal": str(journal),
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "min_soc_frac": 0.0,
                "provenance": {},
            }
            status.write_text(json.dumps(source_status))
            journal.write_text("".join(json.dumps({
                "trips": [trip],
                "cost": 100000.0,
            }) + "\n" for trip in range(40)))
            instance.write_text("instance\n")
            prices.write_text("prices\n")
            hashes = {
                "status": hashlib.sha256(status.read_bytes()).hexdigest(),
                "journal": hashlib.sha256(journal.read_bytes()).hexdigest(),
                "instance": hashlib.sha256(instance.read_bytes()).hexdigest(),
                "prices": hashlib.sha256(prices.read_bytes()).hexdigest(),
            }
            source_status["provenance"] = {
                "instance_sha256": hashes["instance"],
                "prices_sha256": hashes["prices"],
            }
            status.write_text(json.dumps(source_status))
            hashes["status"] = hashlib.sha256(
                status.read_bytes()
            ).hexdigest()
            output = campaign / "outputs" / f"{label}.mip.bundle"
            output.parent.mkdir(exist_ok=True)
            job_id = str(1000 + index)
            spec = {
                "label": label,
                "replicate": replicate,
                "treatment": treatment,
                "snapshot_mark_minutes": mark,
                "time_limit_s": 7200,
                "threads": 8,
                "mip_gap": 0.0001,
                "staged_result": str(status),
                "staged_result_sha256": hashes["status"],
                "staged_journal": str(journal),
                "staged_journal_sha256": hashes["journal"],
                "staged_instance": str(instance),
                "staged_instance_sha256": hashes["instance"],
                "staged_prices": str(prices),
                "staged_prices_sha256": hashes["prices"],
                "staged_start": str(start),
                "staged_start_sha256": start_sha,
                "runner_sha256": runner_sha,
                "expected_commit": recovery.SOURCE_CAMPAIGN_COMMIT,
                "csv": "instance.csv",
                "prices_csv": "prices.csv",
                "output": str(output),
            }
            spec_path = cell / "job.json"
            spec_raw = (json.dumps(spec, indent=2) + "\n").encode()
            spec_path.write_bytes(spec_raw)
            spec_sha = hashlib.sha256(spec_raw).hexdigest()
            variable = 1746.666836618
            selected = [{
                "trips": [trip],
                "cost": 100000.0 + variable / 40.0,
                "route_nodes": ["PARX_0", trip, "PARX_0"],
                "charging_stops": {},
            } for trip in range(40)]
            raw = {
                "partitioning": True,
                "source_result_sha256": hashes["status"],
                "source_journal_sha256": hashes["journal"],
                "mip_start": {
                    "kind": "validated_exact_partition",
                    "source_sha256": start_sha,
                    "solver_acceptance": {"accepted": True},
                    "actual_start_columns": [{
                        "index": trip,
                        "sha256": f"{trip:064x}",
                    } for trip in range(40)],
                },
                "selected_routes": selected,
                "buses": 40,
                "fleet_proven": True,
                "fleet_bound": 40.0,
                "optimal_scope": "full_pool_lexicographic",
                "two_stage": {
                    "stage1_buses": 40,
                    "stage2_executed": True,
                    "stage2_status": 2,
                    "stage2_variable_obj": variable,
                    "stage2_variable_bound": variable,
                },
                "mip_obj": 4000000.0 + variable,
                "mip_bound": 4000000.0 + variable,
                "mip_gap": 0.0,
                "status": 2,
                "status_name": "OPTIMAL",
                "runtime_s": 110.0,
                "mip_provenance": {
                    "expected_git_commit": recovery.SOURCE_CAMPAIGN_COMMIT,
                    "observed_git_commit": recovery.SOURCE_CAMPAIGN_COMMIT,
                    "final_observed_git_commit": recovery.SOURCE_CAMPAIGN_COMMIT,
                    "tracked_clean_at_end": True,
                    "gurobi": "12.0.0",
                    "arguments": {
                        "two_stage": True,
                        "cover": False,
                        "threads": 8,
                        "timelimit": 7200,
                    },
                },
            }
            raw_path = Path(str(output) + f".raw.{job_id}")
            raw_path.write_text(
                "not json" if malformed_raw and index == 0
                else json.dumps(raw)
            )
            jobs.append({
                "label": label,
                "replicate": replicate,
                "treatment": treatment,
                "snapshot_mark_minutes": mark,
                "job_id": job_id,
                "output": str(output),
                "spec": spec,
                "spec_path": str(spec_path),
                "spec_sha256": spec_sha,
                "submission_state": "failed",
            })
        manifest = {
            "schema": "evsp-dr-k40-factorial-mip-campaign-v1",
            "campaign": campaign.name,
            "checkout_identity": {
                "expected_commit": recovery.SOURCE_CAMPAIGN_COMMIT
            },
            "worker": str(worker),
            "worker_sha256": worker_sha,
            "jobs": jobs,
        }
        manifest["approval_sha256"] = hashlib.sha256(
            recovery._canonical(recovery._approval_payload(manifest))
        ).hexdigest()
        (campaign / "campaign.json").write_text(json.dumps(manifest))
        return campaign

    def test_dry_run_and_approved_recovery_preserve_raw_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._fixture(Path(tmp))
            source_sha = self._source_sha(campaign)
            with patch.object(
                recovery, "_git_commit", return_value="e" * 40
            ), patch(
                "k40_factorial_mip_result.validate_final_selected_routes"
            ):
                plan, _prepared = recovery.build_recovery_plan(
                    campaign, source_campaign_sha256=source_sha
                )
                self.assertEqual(plan["recoverable_count"], 12)
                self.assertEqual(plan["invalid_or_missing_count"], 0)
                plan_sha = hashlib.sha256(
                    recovery._canonical(recovery._recovery_intent(plan))
                ).hexdigest()
                monitored = k40_monitor.monitor(
                    campaign, source_campaign_sha256=source_sha
                )
                self.assertTrue(all(
                    row["outcome"] == "recoverable_validated_raw"
                    for row in monitored
                ))
                raw_paths = sorted(
                    (campaign / "outputs").glob("*.raw.*")
                )
                record = recovery.apply_recovery(
                    campaign,
                    approved_plan_sha256=plan_sha,
                    source_campaign_sha256=source_sha,
                )
                repeated = recovery.apply_recovery(
                    campaign,
                    approved_plan_sha256=plan_sha,
                    source_campaign_sha256=source_sha,
                )
            self.assertEqual(len(record["recovered"]), 12)
            self.assertEqual(repeated, record)
            self.assertTrue(all(path.is_file() for path in raw_paths))
            for job in plan["jobs"]:
                self.assertEqual(
                    inspect_bundle(Path(job["destination"]))["state"],
                    "complete_valid",
                )
            with patch.object(
                recovery, "_git_commit", return_value="e" * 40
            ), patch(
                "k40_factorial_mip_result.validate_final_selected_routes"
            ):
                monitored = k40_monitor.monitor(
                    campaign, source_campaign_sha256=source_sha
                )
            self.assertTrue(all(
                row["outcome"] == "complete_valid_output"
                for row in monitored
            ))

    def test_malformed_raw_is_not_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._fixture(Path(tmp), malformed_raw=True)
            source_sha = self._source_sha(campaign)
            with patch.object(
                recovery, "_git_commit", return_value="e" * 40
            ), patch(
                "k40_factorial_mip_result.validate_final_selected_routes"
            ):
                plan, _prepared = recovery.build_recovery_plan(
                    campaign, source_campaign_sha256=source_sha
                )
            self.assertEqual(plan["recoverable_count"], 11)
            self.assertEqual(plan["invalid_or_missing_count"], 1)
            bad = next(
                row for row in plan["jobs"]
                if row["label"] == "R1_CA_m360"
            )
            self.assertFalse(bad["recoverable"])
            self.assertTrue(bad["errors"])

    def test_failed_temporary_bundle_is_recoverable_without_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._fixture(Path(tmp))
            source_sha = self._source_sha(campaign)
            manifest = json.loads(
                (campaign / "campaign.json").read_text()
            )
            job = manifest["jobs"][0]
            output = Path(job["output"])
            raw = Path(str(output) + f".raw.{job['job_id']}")
            staging = output.parent / f".{output.name}.tmp.interrupted"
            staging.mkdir()
            (staging / "result.json").write_bytes(raw.read_bytes())
            raw.unlink()
            with patch.object(
                recovery, "_git_commit", return_value="e" * 40
            ), patch(
                "k40_factorial_mip_result.validate_final_selected_routes"
            ):
                plan, _prepared = recovery.build_recovery_plan(
                    campaign, source_campaign_sha256=source_sha
                )
            row = next(
                item for item in plan["jobs"]
                if item["label"] == job["label"]
            )
            self.assertTrue(row["recoverable"])
            self.assertEqual(
                row["recovery_method"], "failed_temporary_bundle"
            )
            self.assertTrue(staging.is_dir())

    def test_hash_mismatched_raw_is_not_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._fixture(Path(tmp))
            source_sha = self._source_sha(campaign)
            manifest = json.loads(
                (campaign / "campaign.json").read_text()
            )
            job = manifest["jobs"][0]
            raw = Path(str(job["output"]) + f".raw.{job['job_id']}")
            payload = json.loads(raw.read_text())
            payload["source_result_sha256"] = "0" * 64
            raw.write_text(json.dumps(payload))
            with patch.object(
                recovery, "_git_commit", return_value="e" * 40
            ), patch(
                "k40_factorial_mip_result.validate_final_selected_routes"
            ):
                plan, _prepared = recovery.build_recovery_plan(
                    campaign, source_campaign_sha256=source_sha
                )
            row = next(
                item for item in plan["jobs"]
                if item["label"] == job["label"]
            )
            self.assertFalse(row["recoverable"])
            self.assertTrue(any(
                "status hash mismatch" in error for error in row["errors"]
            ))

    def test_out_of_band_campaign_hash_rejects_forged_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._fixture(Path(tmp))
            approved_sha = self._source_sha(campaign)
            path = campaign / "campaign.json"
            manifest = json.loads(path.read_text())
            manifest["campaign"] = "forged"
            manifest["approval_sha256"] = hashlib.sha256(
                recovery._canonical(recovery._approval_payload(manifest))
            ).hexdigest()
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "out-of-band"):
                recovery.build_recovery_plan(
                    campaign, source_campaign_sha256=approved_sha
                )

    def test_interrupted_apply_resumes_with_same_approved_intent(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._fixture(Path(tmp))
            source_sha = self._source_sha(campaign)
            physical = patch(
                "k40_factorial_mip_result.validate_final_selected_routes"
            )
            with patch.object(
                recovery, "_git_commit", return_value="e" * 40
            ), physical:
                plan, _prepared = recovery.build_recovery_plan(
                    campaign, source_campaign_sha256=source_sha
                )
                plan_sha = hashlib.sha256(
                    recovery._canonical(recovery._recovery_intent(plan))
                ).hexdigest()
            original = recovery.publish_result_bundle
            calls = 0

            def interrupt_after_three(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("injected recovery interruption")
                return original(*args, **kwargs)

            with (
                patch.object(recovery, "_git_commit", return_value="e" * 40),
                patch(
                    "k40_factorial_mip_result.validate_final_selected_routes"
                ),
                patch.object(
                    recovery,
                    "publish_result_bundle",
                    side_effect=interrupt_after_three,
                ),
                self.assertRaisesRegex(RuntimeError, "interruption"),
            ):
                recovery.apply_recovery(
                    campaign,
                    approved_plan_sha256=plan_sha,
                    source_campaign_sha256=source_sha,
                )
            with (
                patch.object(recovery, "_git_commit", return_value="e" * 40),
                patch(
                    "k40_factorial_mip_result.validate_final_selected_routes"
                ),
            ):
                record = recovery.apply_recovery(
                    campaign,
                    approved_plan_sha256=plan_sha,
                    source_campaign_sha256=source_sha,
                )
            self.assertEqual(len(record["receipts"]), 12)


if __name__ == "__main__":
    unittest.main()

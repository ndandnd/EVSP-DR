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
        start.write_text("{}")
        start_sha = hashlib.sha256(start.read_bytes()).hexdigest()
        jobs = []
        for index in range(12):
            label = f"cell{index:02d}"
            cell = campaign / "input" / label
            cell.mkdir()
            status = cell / "pool.snapshot.json"
            journal = Path(str(status) + ".columns.jsonl")
            instance = cell / "instance.csv"
            prices = cell / "prices.csv"
            source_status = {"trip_ids": list(range(40))}
            status.write_text(json.dumps(source_status))
            journal.write_text("{}\n")
            instance.write_text("instance\n")
            prices.write_text("prices\n")
            hashes = {
                "status": hashlib.sha256(status.read_bytes()).hexdigest(),
                "journal": hashlib.sha256(journal.read_bytes()).hexdigest(),
                "instance": hashlib.sha256(instance.read_bytes()).hexdigest(),
                "prices": hashlib.sha256(prices.read_bytes()).hexdigest(),
            }
            output = campaign / "outputs" / f"{label}.mip.bundle"
            output.parent.mkdir(exist_ok=True)
            job_id = str(1000 + index)
            spec = {
                "label": label,
                "replicate": "R1" if index < 6 else "R2",
                "treatment": "CA" if index % 2 == 0 else "CS",
                "snapshot_mark_minutes": (360, 720, 1440)[index % 3],
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
            }
            spec_path = cell / "job.json"
            spec_raw = (json.dumps(spec, indent=2) + "\n").encode()
            spec_path.write_bytes(spec_raw)
            spec_sha = hashlib.sha256(spec_raw).hexdigest()
            selected = [{
                "trips": [trip],
                "charging_stops": {},
            } for trip in range(40)]
            variable = 1746.666836618
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
                "runtime_s": 110.0,
                "mip_provenance": {
                    "expected_git_commit": "f" * 40,
                    "observed_git_commit": "f" * 40,
                    "final_observed_git_commit": "f" * 40,
                    "tracked_clean_at_end": True,
                },
            }
            raw_path = Path(str(output) + f".raw.{job_id}")
            raw_path.write_text(
                "not json" if malformed_raw and index == 0
                else json.dumps(raw)
            )
            jobs.append({
                "label": label,
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
            "checkout_identity": {"expected_commit": "f" * 40},
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
            with patch.object(
                recovery, "_git_commit", return_value="e" * 40
            ):
                plan, _prepared = recovery.build_recovery_plan(campaign)
                self.assertEqual(plan["recoverable_count"], 12)
                self.assertEqual(plan["invalid_or_missing_count"], 0)
                plan_sha = hashlib.sha256(
                    recovery._canonical(plan)
                ).hexdigest()
                monitored = k40_monitor.monitor(campaign)
                self.assertTrue(all(
                    row["outcome"] == "recoverable_validated_raw"
                    for row in monitored
                ))
                raw_paths = sorted(
                    (campaign / "outputs").glob("*.raw.*")
                )
                record = recovery.apply_recovery(
                    campaign, approved_plan_sha256=plan_sha
                )
                repeated = recovery.apply_recovery(
                    campaign, approved_plan_sha256=plan_sha
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
            ):
                monitored = k40_monitor.monitor(campaign)
            self.assertTrue(all(
                row["outcome"] == "complete_valid_output"
                for row in monitored
            ))

    def test_malformed_raw_is_not_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._fixture(Path(tmp), malformed_raw=True)
            with patch.object(
                recovery, "_git_commit", return_value="e" * 40
            ):
                plan, _prepared = recovery.build_recovery_plan(campaign)
            self.assertEqual(plan["recoverable_count"], 11)
            self.assertEqual(plan["invalid_or_missing_count"], 1)
            bad = next(row for row in plan["jobs"] if row["label"] == "cell00")
            self.assertFalse(bad["recoverable"])
            self.assertTrue(bad["errors"])

    def test_failed_temporary_bundle_is_recoverable_without_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._fixture(Path(tmp))
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
            ):
                plan, _prepared = recovery.build_recovery_plan(campaign)
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
            ):
                plan, _prepared = recovery.build_recovery_plan(campaign)
            row = next(
                item for item in plan["jobs"]
                if item["label"] == job["label"]
            )
            self.assertFalse(row["recoverable"])
            self.assertTrue(any(
                "status hash mismatch" in error for error in row["errors"]
            ))


if __name__ == "__main__":
    unittest.main()

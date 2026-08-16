import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import build_convergence_evidence as evidence  # noqa: E402


class ConvergenceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.instance = b"synthetic-k40\n"
        self.tariff = b"synthetic-flat\n"
        self.instance_sha = hashlib.sha256(self.instance).hexdigest()
        self.tariff_sha = hashlib.sha256(self.tariff).hexdigest()
        self.enterContext(patch.object(
            evidence, "INSTANCE_SHA256", self.instance_sha
        ))
        self.enterContext(patch.object(
            evidence, "TARIFF_SHA256", self.tariff_sha
        ))
        self.enterContext(patch.object(
            evidence, "EXPECTED_K40_TRIPS", 1
        ))

    def _status(
        self,
        path,
        *,
        arm,
        mark,
        terminal=False,
        historical=False,
    ):
        sense, initial = evidence.ARMS.get(
            arm, ("cover", "artificial")
        )
        journal = Path(str(path) + ".columns.jsonl")
        journal.write_text(json.dumps({
            "trips": [0], "cost": 100000.0,
        }) + "\n")
        artificials = 5.0 if arm == "PA" else 0.0
        route_weight = (
            evidence.HISTORICAL_ROUTE_WEIGHT
            if historical
            else (
                11.58 if arm == "PA"
                else (1.0 if sense == "partition" else 40.0)
            )
        )
        wall_s = (
            79348.0 if historical
            else ((1440 if terminal else mark) * 60.0)
        )
        status = {
            "csv": "synthetic/k40.csv",
            "prices_csv": "flat.csv",
            "soc_step": 15.0,
            "block_min": 10,
            "g_kwh": 300.0,
            "charge_kw": 300.0,
            "min_soc_frac": 0.0,
            "master_sense": sense,
            "initial_pool": initial,
            "trip_ids": [0],
            "columns": 1,
            "columns_journal": str(journal),
            "wall_s": wall_s,
            "snapshot_mark_minutes": (
                None if terminal or historical else float(mark)
            ),
            "stop_reason": (
                "wall_limit"
                if terminal or historical else f"snapshot_m{mark}"
            ),
            "certified_rc_optimal": False,
            "final": {
                "lp_obj": route_weight * 100000 + artificials * 500000,
                "route_weight": route_weight,
                "artificials": artificials,
                "min_rc": -1.0,
            },
            "final_lp": {
                "objective": route_weight * 100000
                + artificials * 500000,
                "route_weight": route_weight,
                "artificial_total": artificials,
                "positive_routes": [{
                    "trips": [0],
                    "value": route_weight,
                    "cost": 100000.0,
                }],
            },
            "provenance": {
                "git_commit": (
                    evidence.HISTORICAL_COMMIT
                    if historical else evidence.FACTORIAL_COMMIT
                ),
                "instance_sha256": self.instance_sha,
                "prices_sha256": self.tariff_sha,
                "git_dirty": False,
                "git_branch": "",
                "args": {
                    "csv": "synthetic/k40.csv",
                    "prices_csv": "flat.csv",
                    "g_kwh": 300.0,
                    "charge_kw": 300.0,
                    "min_soc_frac": 0.0,
                    "master_sense": sense,
                    "initial_pool": initial,
                    "columns_per_iter": 30,
                    "rc_eps": 0.0001,
                },
            },
        }
        path.write_text(json.dumps(status))

    def _fixture(self, root):
        repo = root / "repo"
        data = repo / "data"
        (data / "synthetic").mkdir(parents=True)
        (data / "synthetic/k40.csv").write_bytes(self.instance)
        (data / "flat.csv").write_bytes(self.tariff)
        campaigns = []
        for replicate in (1, 2):
            campaign = repo / "results" / (
                "k40fx_20260814T140232Z_eb85ca0c"
                if replicate == 1
                else "k40fx_20260814T191933Z_eb85ca0c"
            )
            campaign.mkdir(parents=True)
            prefix = "k40r1" if replicate == 1 else "k40r2"
            launch_rows = [
                "role\tjob_id\tjob_name\tmaster_sense\tinitial_pool",
                f"prep\t{replicate}00\tK40-PREP\t-\t-",
            ]
            for index, (arm, (sense, initial)) in enumerate(
                    evidence.ARMS.items(), start=1):
                launch_rows.append(
                    f"arm\t{replicate}0{index}\tK40-{arm}24\t"
                    f"{sense}\t{initial}"
                )
            (campaign / "launch.tsv").write_text(
                "\n".join(launch_rows) + "\n"
            )
            (campaign / "prep_attestation.tsv").write_text(
                f"git_commit\t{evidence.FACTORIAL_COMMIT}\n"
                f"instance_sha256\t{self.instance_sha}\n"
                f"prices_sha256\t{self.tariff_sha}\n"
            )
            for arm in evidence.ARMS:
                job_id = f"{replicate}0{list(evidence.ARMS).index(arm) + 1}"
                (campaign / f"{prefix}_flat_{arm}.allocations.tsv").write_text(
                    "utc\tjob_id\trestart\thost\tcpu_model\t"
                    "instance_sha256\tprices_sha256\n"
                    f"now\t{job_id}\t0\tnode\tcpu\t"
                    f"{self.instance_sha}\t{self.tariff_sha}\n"
                )
                for mark in evidence.CHECKPOINTS.values():
                    self._status(
                        campaign / (
                            f"{prefix}_flat_{arm}.m{mark}.snapshot.json"
                        ),
                        arm=arm,
                        mark=mark,
                    )
                self._status(
                    campaign / f"{prefix}_flat_{arm}.json",
                    arm=arm,
                    mark=1440,
                    terminal=True,
                )
            campaigns.append(campaign)
        historical = repo / "results/historical.json"
        self._status(
            historical,
            arm="CA",
            mark=1320,
            terminal=True,
            historical=True,
        )
        legacy = repo / "analysis/legacy"
        legacy.mkdir(parents=True)
        (legacy / "README.md").write_text("legacy derived evidence\n")
        return campaigns, historical, legacy

    def _build(self, root, output):
        campaigns, historical, legacy = self._fixture(root)
        result = evidence.build(
            factorial_campaigns=campaigns,
            historical_path=historical,
            legacy_analysis=legacy,
            release_archives=[],
            verified_artifacts=[],
            output_dir=output,
            generation_command="deterministic-test-command",
        )
        return result, campaigns, historical, legacy

    def test_evidence_outputs_keep_pa_artificials_and_labels_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "evidence"
            result, _campaigns, _historical, _legacy = self._build(
                root, output
            )
            self.assertEqual(result["trajectory_rows"], 56)
            required = (
                "k40_factorial_trajectory.csv",
                "small_instance_certification.csv",
                "historical_endpoint_comparison.csv",
                "k40_convergence_cover.png",
                "k40_convergence_cover.pdf",
                "partition_failure_diagnostic.png",
                "partition_failure_diagnostic.pdf",
                "provenance.json",
                "README.md",
            )
            self.assertTrue(all((output / name).is_file() for name in required))
            with (output / "k40_factorial_trajectory.csv").open(
                    newline="") as handle:
                rows = list(csv.DictReader(handle))
            pa = next(
                row for row in rows
                if row["arm"] == "PA" and row["checkpoint"] == "h22"
            )
            self.assertEqual(pa["route_weight"], "11.58")
            self.assertEqual(pa["artificials"], "5.0")
            self.assertEqual(pa["lp_feasible"], "False")
            self.assertIn("infeasible", pa["scientific_label"])
            ca = next(row for row in rows if row["arm"] == "CA")
            self.assertIn("not_integer_schedule", ca["scientific_label"])
            with self.assertRaises(FileExistsError):
                evidence.build(
                    factorial_campaigns=_campaigns,
                    historical_path=_historical,
                    legacy_analysis=_legacy,
                    release_archives=[],
                    verified_artifacts=[],
                    output_dir=output,
                    generation_command="deterministic-test-command",
                )

    def test_missing_checkpoint_and_mixed_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaigns, historical, legacy = self._fixture(root)
            missing = campaigns[0] / "k40r1_flat_CA.m60.snapshot.json"
            Path(str(missing) + ".columns.jsonl").unlink()
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "checkpoints"):
                evidence.build(
                    factorial_campaigns=campaigns,
                    historical_path=historical,
                    legacy_analysis=legacy,
                    release_archives=[],
                    verified_artifacts=[],
                    output_dir=root / "missing-output",
                    generation_command="test",
                )

            campaigns, historical, legacy = self._fixture(root / "mixed")
            status_path = (
                campaigns[1] / "k40r2_flat_CS.m180.snapshot.json"
            )
            status = json.loads(status_path.read_text())
            status["provenance"]["instance_sha256"] = "f" * 64
            status_path.write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, "instance hash"):
                evidence.build(
                    factorial_campaigns=campaigns,
                    historical_path=historical,
                    legacy_analysis=legacy,
                    release_archives=[],
                    verified_artifacts=[],
                    output_dir=root / "mixed-output",
                    generation_command="test",
                )

    def test_false_certification_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaigns, historical, legacy = self._fixture(root)
            status_path = (
                campaigns[0] / "k40r1_flat_CA.m360.snapshot.json"
            )
            status = json.loads(status_path.read_text())
            status["certified_rc_optimal"] = True
            status_path.write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, "false pricing"):
                evidence.build(
                    factorial_campaigns=campaigns,
                    historical_path=historical,
                    legacy_analysis=legacy,
                    release_archives=[],
                    verified_artifacts=[],
                    output_dir=root / "false-cert",
                    generation_command="test",
                )

    def test_deterministic_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaigns, historical, legacy = self._fixture(root)
            outputs = []
            for name in ("one", "two"):
                output = root / name
                evidence.build(
                    factorial_campaigns=campaigns,
                    historical_path=historical,
                    legacy_analysis=legacy,
                    release_archives=[],
                    verified_artifacts=[],
                    output_dir=output,
                    generation_command="same-command",
                )
                outputs.append(output)
            names = sorted(path.name for path in outputs[0].iterdir())
            for name in names:
                self.assertEqual(
                    hashlib.sha256((outputs[0] / name).read_bytes()).hexdigest(),
                    hashlib.sha256((outputs[1] / name).read_bytes()).hexdigest(),
                    name,
                )

    def test_forged_scale_json_is_not_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forged = root / "fake_k8.json"
            forged.write_text(json.dumps({
                "csv": "fake_k8.csv",
                "trip_ids": [0],
                "soc_step": 15,
                "block_min": 10,
                "g_kwh": 300,
                "charge_kw": 300,
                "min_soc_frac": 0,
                "certified_rc_optimal": True,
                "partitioning": True,
                "fleet_proven": True,
                "provenance": {
                    "git_commit": "a" * 40,
                    "instance_sha256": "b" * 64,
                    "prices_sha256": "c" * 64,
                },
            }))
            rows = evidence._verified_scale_evidence([root])
            k8 = next(
                row for row in rows
                if row["scale_family"] == "union" and row["scale"] == 8
            )
            self.assertEqual(k8["availability"], "not available")
            self.assertIsNone(k8["finite_pool_fleet_proven"])


if __name__ == "__main__":
    unittest.main()

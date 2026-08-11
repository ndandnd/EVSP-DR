import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_snapshot_pool_mip as snapshot_mip  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SnapshotPoolMipTests(unittest.TestCase):
    def test_problem_inputs_must_match_snapshot_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            instance = data_dir / "instance.csv"
            prices = data_dir / "prices.csv"
            instance.write_text("trip\n1\n")
            prices.write_text("hour,price\n0,1\n")
            status = {
                "csv": instance.name,
                "prices_csv": prices.name,
                "provenance": {
                    "instance_sha256": sha256(instance),
                    "prices_sha256": sha256(prices),
                },
            }

            verified = snapshot_mip._verify_snapshot_problem_inputs(
                status, data_dir
            )
            self.assertEqual(
                verified["instance_sha256"], sha256(instance)
            )

            instance.write_text("trip\n2\n")
            with self.assertRaisesRegex(SystemExit, "input hash mismatch"):
                snapshot_mip._verify_snapshot_problem_inputs(status, data_dir)

    def test_cached_partial_rerealization_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "routes.json"
            expected = {"kind": "rerealized_giro_seed"}
            cache.write_text(json.dumps({
                "routes": [{"route": [1]}],
                "infeasible": [{"trips": [2]}],
                "_snapshot_worker_provenance": expected,
            }))

            self.assertIsNone(snapshot_mip._load_route_cache(cache, expected))

    def test_rerealization_exit_three_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.object(snapshot_mip, "run", return_value=3):
                with self.assertRaisesRegex(
                    SystemExit, "route-cache generation failed with rc=3"
                ):
                    snapshot_mip._generate_route_cache(
                        ["generator"],
                        folder / "temporary.json",
                        folder / "cache.json",
                        {"kind": "rerealized_giro_seed"},
                    )

    def test_existing_result_is_invalidated_when_snapshot_journal_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            snapshot = folder / "pool.snapshot.json"
            journal = folder / "pool.columns.jsonl"
            output = folder / "pool.mip.json"
            snapshot.write_text(json.dumps({
                "csv": "instance.csv",
                "prices_csv": "prices.csv",
                "columns_journal": str(journal),
            }))
            journal.write_text(json.dumps({"trips": [1], "cost": 1}) + "\n")
            output.write_text(json.dumps({
                "status_name": "OPTIMAL",
                "buses": 1,
                "source_result_sha256": sha256(snapshot),
                "source_journal_sha256": sha256(journal),
                "mip_provenance": {"git_commit": "test-commit"},
            }))

            with patch.object(
                    snapshot_mip, "git_value", return_value="test-commit"), \
                    patch.object(snapshot_mip, "run") as runner:
                self.assertEqual(snapshot_mip.main([
                    "--snapshot", str(snapshot), "--out", str(output),
                ]), 0)
                runner.assert_not_called()

            journal.write_text(
                journal.read_text()
                + json.dumps({"trips": [2], "cost": 1}) + "\n"
            )
            with patch.object(
                    snapshot_mip, "git_value", return_value="test-commit"), \
                    patch.object(
                        snapshot_mip,
                        "_verify_snapshot_problem_inputs",
                        return_value={
                            "instance_sha256": "a" * 64,
                            "prices_sha256": "b" * 64,
                        },
                    ), \
                    patch.object(
                        snapshot_mip, "_load_route_cache", return_value={}
                    ), \
                    patch.object(snapshot_mip, "run", return_value=0) as runner:
                self.assertEqual(snapshot_mip.main([
                    "--snapshot", str(snapshot), "--out", str(output),
                ]), 0)
                runner.assert_called_once()
                self.assertIn("run_exact_pool_mip.py", " ".join(
                    str(value) for value in runner.call_args.args[0]
                ))


if __name__ == "__main__":
    unittest.main()

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_exact_master_pool import audit_pool  # noqa: E402


class AuditExactMasterPoolTests(unittest.TestCase):
    def test_small_partition_pool_reports_raw_residual_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            status_path = folder / "pool.json"
            journal_path = Path(str(status_path) + ".columns.jsonl")
            journal_path.write_text(
                json.dumps({"trips": [1], "cost": 100000.0}) + "\n" +
                json.dumps({"trips": [2], "cost": 100000.0}) + "\n"
            )
            status_path.write_text(json.dumps({
                "csv": "example.csv",
                "prices_csv": "hourly_prices_flat.csv",
                "soc_step": 15.0,
                "block_min": 10,
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "min_soc_frac": 0.0,
                "trip_ids": [1, 2],
                "columns_journal": str(journal_path),
            }))

            report = audit_pool(status_path, ["highs-ds"], 1e-6)

        self.assertTrue(report["any_method_succeeded"])
        self.assertEqual(report["pool_columns"], 2)
        self.assertEqual(report["methods"][0]["route_weight"], 2.0)
        self.assertLessEqual(report["methods"][0]["max_row_violation"], 1e-12)
        self.assertEqual(len(report["source_journal_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

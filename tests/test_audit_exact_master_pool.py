import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_exact_master_pool import audit_pool  # noqa: E402
from config import BIG_M_PENALTY  # noqa: E402


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

    def test_overlapping_pool_distinguishes_cover_from_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            status_path = folder / "overlap.json"
            journal_path = Path(str(status_path) + ".columns.jsonl")
            journal_path.write_text(
                json.dumps({"trips": [1, 2], "cost": 1.0}) + "\n" +
                json.dumps({"trips": [2, 3], "cost": 1.0}) + "\n"
            )
            status_path.write_text(json.dumps({
                "csv": "overlap.csv",
                "prices_csv": "hourly_prices_flat.csv",
                "trip_ids": [1, 2, 3],
                "columns_journal": str(journal_path),
            }))

            covering = audit_pool(
                status_path, ["highs-ds"], 1e-6,
                master_sense="cover",
            )
            partitioning = audit_pool(
                status_path, ["highs-ds"], 1e-6,
                master_sense="partition",
            )

        self.assertEqual(covering["master_sense"], "cover")
        self.assertEqual(partitioning["master_sense"], "partition")
        cover_method = next(
            result for result in covering["methods"] if result["success"]
        )
        partition_method = next(
            result for result in partitioning["methods"] if result["success"]
        )
        self.assertAlmostEqual(cover_method["max_overcoverage"], 1.0)
        self.assertEqual(cover_method["overcovered_rows"], 1)
        self.assertAlmostEqual(partition_method["max_overcoverage"], 0.0)
        self.assertEqual(partition_method["overcovered_rows"], 0)
        self.assertAlmostEqual(cover_method["objective"], 2.0)
        self.assertAlmostEqual(
            partition_method["objective"], BIG_M_PENALTY + 1.0
        )
        self.assertAlmostEqual(cover_method["artificial_total"], 0.0)
        self.assertAlmostEqual(partition_method["artificial_total"], 1.0)
        self.assertLessEqual(
            cover_method["recomputed_max_row_violation"], 1e-12
        )
        self.assertLessEqual(
            partition_method["recomputed_max_row_violation"], 1e-12
        )


if __name__ == "__main__":
    unittest.main()

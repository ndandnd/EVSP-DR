import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from durable_io import DurableFileError, read_jsonl_records  # noqa: E402
from exact_pricer_expanded import load_iteration_log  # noqa: E402


class DurableIoTests(unittest.TestCase):
    def test_repairs_only_truncated_final_jsonl_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "columns.jsonl"
            path.write_bytes(
                (json.dumps({"trips": [1], "cost": 1.0}) + "\n").encode()
                + b'{"trips":[2]'
            )

            records = read_jsonl_records(path, repair_trailing=True)

            self.assertEqual(records, [{"trips": [1], "cost": 1.0}])
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_refuses_interior_jsonl_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "columns.jsonl"
            path.write_text(
                json.dumps({"trips": [1], "cost": 1.0}) + "\n"
                + "not-json\n"
                + json.dumps({"trips": [2], "cost": 1.0}) + "\n"
            )

            with self.assertRaisesRegex(DurableFileError, "before EOF"):
                read_jsonl_records(path, repair_trailing=True)

    def test_valid_final_jsonl_record_gets_newline_before_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "columns.jsonl"
            path.write_text(json.dumps({"trips": [1], "cost": 1.0}))

            read_jsonl_records(path, repair_trailing=True)

            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_iteration_log_repairs_tail_and_preserves_elapsed_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.iters.csv"
            path.write_text(
                "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,"
                "pool_columns\n"
                "21600.00,1200,200000.0,2.0,0.0,-1.0,5000\n"
                "21601.0,1201,"
            )

            rows = load_iteration_log(path, repair_trailing=True)

            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[-1][0]), 21600.0)
            self.assertEqual(int(float(rows[-1][1])), 1200)
            self.assertTrue(path.read_text().endswith("\n"))

    def test_iteration_log_refuses_interior_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.iters.csv"
            path.write_text(
                "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,"
                "pool_columns\n"
                "bad,row\n"
                "10,1,2,3,4,5,6\n"
            )
            with self.assertRaisesRegex(DurableFileError, "before EOF"):
                load_iteration_log(path, repair_trailing=True)

    def test_iteration_log_accepts_positive_infinity_only_for_min_rc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.iters.csv"
            path.write_text(
                "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,"
                "pool_columns\n"
                "10,1,200000,2,0,inf,100\n"
            )

            rows = load_iteration_log(path, repair_trailing=False)

            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0][5]), float("inf"))

    def test_iteration_log_rejects_infinity_outside_min_rc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.iters.csv"
            path.write_text(
                "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,"
                "pool_columns\n"
                "inf,1,200000,2,0,-1,100\n"
            )

            with self.assertRaisesRegex(
                    DurableFileError, "malformed final iteration row"):
                load_iteration_log(path, repair_trailing=False)

    def test_iteration_log_rejects_negative_infinity_for_min_rc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.iters.csv"
            path.write_text(
                "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,"
                "pool_columns\n"
                "10,1,200000,2,0,-inf,100\n"
            )

            with self.assertRaisesRegex(
                    DurableFileError, "malformed final iteration row"):
                load_iteration_log(path, repair_trailing=False)


if __name__ == "__main__":
    unittest.main()

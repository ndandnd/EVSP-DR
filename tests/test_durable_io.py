import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from durable_io import (  # noqa: E402
    DurableFileError,
    exclusive_output_lock,
    read_jsonl_records,
)
from exact_pricer_expanded import (  # noqa: E402
    ITERATION_LOG_HEADER,
    load_iteration_log,
)


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

    def test_recovers_concatenated_complete_records_before_partial_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "columns.jsonl"
            first = {"trips": [1], "cost": 1.0}
            second = {"trips": [2], "cost": 2.0}
            path.write_text(
                json.dumps(first) + json.dumps(second) + '{"trips":[3]'
            )

            records = read_jsonl_records(path, repair_trailing=True)

            self.assertEqual(records, [first, second])
            self.assertEqual(
                [json.loads(line) for line in path.read_text().splitlines()],
                [first, second],
            )

    def test_refuses_non_object_value_in_malformed_final_jsonl_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "columns.jsonl"
            original = b'null{"trips":[1]'
            path.write_bytes(original)

            with self.assertRaisesRegex(DurableFileError, "non-object"):
                read_jsonl_records(path, repair_trailing=True)

            self.assertEqual(path.read_bytes(), original)

    def test_archived_legacy_mode_quarantines_unparseable_final_line(self):
        for damaged_tail in (b"not-json", b'null{"trips":[2],"cost":2}'):
            with self.subTest(damaged_tail=damaged_tail):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "columns.jsonl"
                    record = {"trips": [1], "cost": 1.0}
                    expected = (json.dumps(record) + "\n").encode()
                    path.write_bytes(expected + damaged_tail)

                    records = read_jsonl_records(
                        path,
                        repair_trailing=True,
                        allow_unparseable_trailing=True,
                    )

                    self.assertEqual(records, [record])
                    self.assertEqual(path.read_bytes(), expected)

    def test_output_lock_rejects_concurrent_owner_and_allows_requeue(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run.json"
            with exclusive_output_lock(output, {"job": "first"}):
                with self.assertRaisesRegex(
                        DurableFileError, "another process holds"):
                    with exclusive_output_lock(output, {"job": "second"}):
                        self.fail("the second owner must not enter")

            with exclusive_output_lock(output, {"job": "requeue"}):
                metadata = json.loads(
                    Path(str(output) + ".lock").read_text()
                )
                self.assertEqual(metadata["job"], "requeue")

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

    def test_iteration_log_repairs_header_only_missing_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.iters.csv"
            path.write_text(ITERATION_LOG_HEADER)

            rows = load_iteration_log(path, repair_trailing=True)

            self.assertEqual(rows, [])
            self.assertEqual(path.read_text(), ITERATION_LOG_HEADER + "\n")

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

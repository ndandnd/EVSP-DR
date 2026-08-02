import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_goal1_column_pools import (  # noqa: E402
    ColumnPoolAuditError,
    audit_column_pools,
    main,
)


class Goal1ColumnPoolUnionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.price_path = self.root / "prices.csv"
        self.price_path.write_text("time_block,cost\n0,1.0\n", encoding="utf-8")
        self.price_sha256 = hashlib.sha256(self.price_path.read_bytes()).hexdigest()

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def route(*trips, charged_kwh=0.0):
        if charged_kwh:
            charging_stops = {
                "stations": ["PARX_1"],
                "cst": [0.0],
                "cet": [charged_kwh / 5.0],
                "kwh": [charged_kwh],
            }
        else:
            charging_stops = {"stations": [], "cst": [], "cet": [], "kwh": []}
        return {
            "route": ["PARX_0", *trips, "PARX_0"],
            "charging_stops": charging_stops,
            "charging_activities": int(bool(charged_kwh)),
            "deadhead_kwh": 0.0,
            "type": "truck",
        }

    def pool(self, *, routes, seed_count, commit, **overrides):
        value = {
            "num_routes": len(routes),
            "seed_route_count": seed_count,
            "dp_columns_generated": len(routes) - seed_count,
            "csv_name": "missing_instance.csv",
            "prices_csv": str(self.price_path),
            "instance_sha256": "a" * 64,
            "price_sha256": self.price_sha256,
            "trip_ids": [0, 1, 2],
            "battery_kwh": 300,
            "mode": "GREEDY",
            "queue_order": "reduced_cost_bound",
            "pricing_output_selection": "diversified",
            "dominance_mode": "resource",
            "termination_reason": "active_time_limit_reached",
            "git": {"commit": commit, "dirty": False},
            "run_arguments": {
                "G": 300,
                "max_charge2trip": 1560.0,
                "successor_charge_targets": True,
                "max_successor_charge_targets": 64,
            },
            "routes": routes,
        }
        value.update(overrides)
        return value

    def write_pool(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def make_complementary_pools(self):
        first = self.pool(
            routes=[
                self.route(0, 1, charged_kwh=10.0),
                self.route(0),
            ],
            seed_count=1,
            commit="old-commit",
        )
        second = self.pool(
            routes=[
                self.route(1, 2),
                self.route(0, 1),
                self.route(0, 2),
            ],
            seed_count=1,
            commit="new-commit",
            queue_order="time",
            dominance_mode="incidence_diverse",
        )
        return (
            self.write_pool("first.json", first),
            self.write_pool("second.json", second),
        )

    def test_union_deduplicates_by_incidence_keeps_cheapest_and_solves_lp(self):
        first_path, second_path = self.make_complementary_pools()

        report = audit_column_pools(
            [first_path, second_path],
            data_dir=self.root,
        )

        self.assertEqual(report["columns"]["total_input_columns"], 5)
        self.assertEqual(report["columns"]["unique_trip_incidences"], 4)
        self.assertEqual(report["columns"]["duplicate_incidences_removed"], 1)
        self.assertEqual(report["dp_trip_coverage"]["covered_trip_ids"], [0, 1, 2])
        self.assertEqual(report["dp_trip_coverage"]["missing_trip_ids"], [])
        master = report["restricted_master"]
        self.assertAlmostEqual(master["objective"], 150000.0)
        self.assertAlmostEqual(master["route_weight"], 1.5)
        self.assertEqual(master["artificial_total"], 0.0)
        self.assertEqual(master["artificial_count"], 0)

        active_by_trips = {
            tuple(column["trip_ids"]): column for column in report["active_columns"]
        }
        duplicate = active_by_trips[(0, 1)]
        self.assertEqual(duplicate["master_cost"], 100000.0)
        self.assertEqual(duplicate["retained_from"]["pool_id"], "pool_002")
        self.assertEqual(duplicate["retained_from"]["origin"], "dp")
        self.assertEqual(duplicate["source_pool_ids"], ["pool_001", "pool_002"])
        self.assertEqual(duplicate["source_origins"], ["dp", "seed"])
        self.assertIn("does_not", report["scope"])

    def test_rejects_identity_or_pricing_action_mismatch(self):
        first_path, second_path = self.make_complementary_pools()
        second = json.loads(second_path.read_text(encoding="utf-8"))

        second["trip_ids"] = [0, 1, 3]
        second_path.write_text(json.dumps(second), encoding="utf-8")
        with self.assertRaisesRegex(ColumnPoolAuditError, "different mathematical instances"):
            audit_column_pools([first_path, second_path], data_dir=self.root)

        second["trip_ids"] = [0, 1, 2]
        second["run_arguments"]["max_successor_charge_targets"] = 32
        second_path.write_text(json.dumps(second), encoding="utf-8")
        with self.assertRaisesRegex(ColumnPoolAuditError, "max_successor_charge_targets"):
            audit_column_pools([first_path, second_path], data_dir=self.root)

    def test_cli_prints_json_and_writes_same_optional_output(self):
        first_path, second_path = self.make_complementary_pools()
        output_path = self.root / "reports" / "union.json"
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = main([
                str(first_path),
                str(second_path),
                "--data-dir",
                str(self.root),
                "--output",
                str(output_path),
            ])

        self.assertEqual(exit_code, 0)
        stdout_report = json.loads(stdout.getvalue())
        file_report = json.loads(output_path.read_text(encoding="utf-8"))
        # Runtime is measured independently only once in this invocation, so
        # stdout and file should be byte-for-byte-equivalent JSON values.
        self.assertEqual(stdout_report, file_report)
        self.assertEqual(stdout_report["audit"], "goal1_column_pool_union")


if __name__ == "__main__":
    unittest.main()

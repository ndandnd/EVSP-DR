import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import exact_pricer_expanded as exact  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_instance(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["count_trip_id", "Ordered_Trip_ID"]
        )
        writer.writeheader()
        writer.writerows(rows)


class FakeEventNetwork:
    def __init__(self):
        self.sequences = []

    def fixed_sequence_record(self, trips):
        self.sequences.append(list(trips))
        return {
            "trips": list(trips),
            "cost": 100000.0,
            "continuous_realized_charging_blocks": [],
            "physical_realization": {
                "status": "valid_event_time_realized",
                "continuous_realized_charging_blocks_sha256":
                    exact.charging_block_schedule_sha256([]),
            },
        }


class InheritedEventPoolTests(unittest.TestCase):
    def test_translates_stable_ids_and_replays_unique_cheapest_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent.csv"
            child = root / "child.csv"
            write_instance(parent, [
                {"count_trip_id": 0, "Ordered_Trip_ID": 10},
                {"count_trip_id": 1, "Ordered_Trip_ID": 11},
            ])
            write_instance(child, [
                {"count_trip_id": 2, "Ordered_Trip_ID": 10},
                {"count_trip_id": 3, "Ordered_Trip_ID": 11},
                {"count_trip_id": 4, "Ordered_Trip_ID": 12},
            ])
            journal = root / "parent.json.columns.jsonl"
            journal.write_text(
                json.dumps({"trips": [0, 1], "cost": 11.0}) + "\n"
                + json.dumps({"trips": [1, 0], "cost": 9.0}) + "\n"
            )
            status = root / "parent.json"
            status.write_text(json.dumps({
                "csv": "parent.csv",
                "trip_ids": [0, 1],
                "columns_journal": str(journal),
                "provenance": {"instance_sha256": sha256(parent)},
            }))
            network = FakeEventNetwork()
            with (
                mock.patch.object(exact, "DATA_DIR", root),
                mock.patch.object(exact, "validate_injected_route",
                                  return_value=None),
            ):
                records, audit = exact.inherited_event_pool_records(
                    status,
                    child_csv_path=child,
                    child_problem=SimpleNamespace(trips=[2, 3, 4]),
                    child_network=network,
                    g_kwh=240.0,
                    charge_kw=240.0,
                    reserve_kwh=0.0,
                )
            self.assertEqual(network.sequences, [[3, 2]])
            self.assertEqual(records[0]["trips"], [3, 2])
            self.assertEqual(records[0]["inherited_source_ordered_trip_ids"],
                             [11, 10])
            self.assertEqual(audit["source_raw_records"], 2)
            self.assertEqual(audit["source_unique_columns"], 1)
            self.assertEqual(audit["accepted_columns"], 1)
            self.assertFalse(audit["inherited_duals"])
            self.assertFalse(audit["inherited_lp_certificate"])


if __name__ == "__main__":
    unittest.main()

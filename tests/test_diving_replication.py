"""Regression tests for incumbent loss, budget floors, and journal identity."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diving_pricing_pilot import DivePilotError, export_incumbent

spec = importlib.util.spec_from_file_location("replication", ROOT / "scripts/research/diving_pricing_20260919/run_replication.py")
replication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replication)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.journal = self.path / "columns.jsonl"
        self.out = self.path / "start.json"
        self.solution = {"buses": 2, "routes": [{"trips": [0, 1], "cost": 10}, {"trips": [1, 2], "cost": 20}]}
        self.records = [{"trips": [0, 1], "cost": 11, "route_nodes": ["DEPOT", 0, 1, "DEPOT"]},
                        {"trips": [0, 1], "cost": 10, "route_nodes": ["DEPOT", 0, "S", 1, "DEPOT"], "charging_stops": {"kwh": [2]}},
                        {"trips": [1, 2], "cost": 20, "route_nodes": ["DEPOT", 1, 2, "DEPOT"]}]

    def write(self):
        self.journal.write_text("".join(json.dumps(r) + "\n" for r in self.records))

    def test_cover_exports_full_cost_matched_existing_records_and_preserves_source(self):
        self.write()
        before = self.journal.read_bytes()
        result = export_incumbent(self.solution, self.journal, [0, 1, 2], self.out)
        payload = json.loads(self.out.read_text())
        self.assertEqual(result["buses"], 2)
        self.assertEqual(payload["routes"], self.records[1:])
        self.assertEqual(payload["source_record_ordinals"], [2, 3])
        self.assertEqual(self.journal.read_bytes(), before)
        self.assertFalse(payload["external_witness_columns_used"])

    def test_matching_incidence_with_wrong_cost_is_rejected(self):
        self.records.pop(1)
        self.write()
        with self.assertRaisesRegex(DivePilotError, "matching incidence and cost"):
            export_incumbent(self.solution, self.journal, [0, 1, 2], self.out)
        self.assertFalse(self.out.exists())

    def test_missing_physical_record_is_rejected(self):
        del self.records[1]["route_nodes"]
        self.write()
        with self.assertRaises(DivePilotError):
            export_incumbent(self.solution, self.journal, [0, 1, 2], self.out)

    def test_partial_cover_and_duplicate_incidence_rejected(self):
        self.write()
        with self.assertRaisesRegex(DivePilotError, "trip universe"):
            export_incumbent(self.solution, self.journal, [0, 1, 2, 3], self.out)
        self.solution["routes"][1] = self.solution["routes"][0]
        with self.assertRaisesRegex(DivePilotError, "duplicate"):
            export_incumbent(self.solution, self.journal, [0, 1, 2], self.out)

    def test_no_incumbent_does_not_create_a_start(self):
        self.assertIsNone(export_incumbent(None, self.journal, [0], self.out))
        self.assertFalse(self.out.exists())

    def test_budget_never_adds_floor_or_rounds_up(self):
        self.assertEqual(replication.remaining_solver_budget(3600, 2492.4), 1107)
        self.assertEqual(replication.remaining_solver_budget(3600, 3600.1), 0)
        self.assertEqual(replication.remaining_solver_budget(3600, 0), 3600)


if __name__ == "__main__":
    unittest.main()

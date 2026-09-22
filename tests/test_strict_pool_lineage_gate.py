import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/audit_strict_pool_lineage.py"
spec = importlib.util.spec_from_file_location("strict_pool_lineage_gate", PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def route(trip, *, checkpoint="old", origin="nested_previous_instance_pool"):
    return {"trips": [trip], "route_nodes": ["PARX_0", trip, "PARX_0"],
        "cg_checkpoint_id": checkpoint, "origin": origin, "found_iter": 0,
        "cost": 100000.0, "continuous_realization": {"mapping_sha256": "saved"},
        "inheritance": {"parent_route_sha256": "authenticated-parent"}}


class StrictPoolLineageGateTests(unittest.TestCase):
    def compare(self, generated, saved):
        return gate.compare_inherited(generated, saved, old_checkpoint="old", new_checkpoint="new",
                                      route_key=lambda item: gate.canonical(item["trips"]))

    def test_only_checkpoint_is_normalized_without_mutating_source(self):
        saved = [route(3), route(8)]
        original = copy.deepcopy(saved)
        generated = [{**row, "cg_checkpoint_id": "new"} for row in saved]
        self.compare(generated, saved)
        self.assertEqual(saved, original)
        for key, value in (("cost", 99999.0), ("inheritance", {}),
                           ("continuous_realization", {"mapping_sha256": "changed"})):
            changed = copy.deepcopy(generated)
            changed[0][key] = value
            with self.assertRaisesRegex(ValueError, "full-record mismatch"):
                self.compare(changed, saved)

    def test_order_duplicates_checkpoint_and_json_types_are_checked(self):
        saved = [route(3), route(8)]
        generated = [{**row, "cg_checkpoint_id": "new"} for row in saved]
        with self.assertRaisesRegex(ValueError, "full-record mismatch"):
            self.compare(list(reversed(generated)), saved)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.compare([generated[0], generated[0]], saved)
        with self.assertRaisesRegex(ValueError, "old child checkpoint"):
            self.compare(generated, [{**saved[0], "cg_checkpoint_id": "wrong"}, saved[1]])
        changed = copy.deepcopy(generated)
        changed[0]["cost"] = 100000  # equal as Python numbers, different exact JSON record
        with self.assertRaisesRegex(ValueError, "full-record mismatch"):
            self.compare(changed, saved)

    def test_singleton_suffix_is_exact_missing_trip_set(self):
        rows = []
        for trip in (3, 8):
            row = route(trip, origin="exact_event_singleton")
            row.pop("inheritance")
            rows.append(row)
        self.assertEqual(gate.singleton_trip_ids(rows, missing_trips={3, 8}, checkpoint="old"), [3, 8])
        variants = []
        changed = copy.deepcopy(rows);changed[0]["trips"] = [4];variants.append(changed)
        changed = copy.deepcopy(rows);changed[0]["found_iter"] = 1;variants.append(changed)
        changed = copy.deepcopy(rows);changed[0]["inheritance"] = {};variants.append(changed)
        changed = copy.deepcopy(rows);changed[0]["cg_checkpoint_id"] = "wrong";variants.append(changed)
        variants.append([rows[0], rows[0]])
        for changed in variants:
            with self.assertRaises(ValueError):
                gate.singleton_trip_ids(changed, missing_trips={3, 8}, checkpoint="old")
        with self.assertRaisesRegex(ValueError, "missing child trips"):
            gate.singleton_trip_ids(rows, missing_trips={3, 9}, checkpoint="old")


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "giro_zero_fee_campaign",
    ROOT / "scripts/event_uniform_envelope/giro_zero_fee_campaign.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def route(cost=100015.0, continuous=100014.0):
    stops = {"stations": ["PARX_0", "PARX_0", "2190L_0"],
             "cst": [1, 2, 3], "cet": [2, 3, 4], "kwh": [5, 5, 5]}
    return {
        "trips": [1, 2], "route_nodes": ["PARX_0", 1, 2, "PARX_0"],
        "charging_stops": stops,
        "expanded_grid_charging_stops": dict(stops),
        "cost": cost, "expanded_grid_cost": cost,
        "continuous_realized_cost": continuous,
    }


class GiroZeroFeeTests(unittest.TestCase):
    def test_reprice_preserves_fee5_cost_and_changes_both_costs(self):
        result = MODULE.reprice_route(route(), 5.0, 0.0)
        self.assertEqual(result["source_charge_start_fee"], 5.0)
        self.assertEqual(result["source_charge_start_count"], 3)
        self.assertEqual(result["source_cost"], 100015.0)
        self.assertEqual(result["cost"], 100000.0)
        self.assertEqual(result["expanded_grid_cost"], 100000.0)
        self.assertEqual(result["continuous_realized_cost"], 99999.0)

    def test_original_reprice_leaves_energy_and_removes_fee(self):
        payload = {
            "physics": {"charge_start_cost": 5.0},
            "summary": [{
                "charge_events": 52, "charge_start_fees": 260.0,
                "charging_cost_lower": 550.0, "charging_cost_upper": 551.0,
                "charging_cost_exact": None,
                "energy_cost_lower": 290.0, "energy_cost_upper": 291.0,
                "terminal_energy_normalized_charging_cost_lower": 522.0,
                "terminal_energy_normalized_charging_cost_upper": 523.0,
                "terminal_surplus_total_kwh": 280.7833253,
            }],
        }
        result = MODULE.reprice_original(payload, 5.0, 0.0)
        summary = result["summary"][0]
        self.assertEqual(summary["charge_start_fees"], 0.0)
        self.assertEqual(summary["charging_cost_lower"], 290.0)
        self.assertEqual(summary["charging_cost_upper"], 291.0)
        self.assertEqual(summary["energy_cost_lower"], 290.0)
        self.assertEqual(summary["terminal_surplus_total_kwh"], 280.7833253)
        self.assertEqual(result["physics"]["charge_start_cost"], 0.0)

    def test_source_copy_does_not_mutate_fee5_input(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            destination = Path(temp) / "fee0"
            source.mkdir()
            original = {
                "physics": {"charge_start_cost": 5.0},
                "summary": [{"charge_events": 3,
                              "charge_start_fees": 15.0,
                              "charging_cost_lower": 20.0,
                              "charging_cost_upper": 21.0,
                              "energy_cost_lower": 5.0,
                              "energy_cost_upper": 6.0,
                              "terminal_surplus_total_kwh": 280.7833253}],
            }
            snapshot = {"columns": 1, "provenance": {}, "columns_journal": "old"}
            (source / "original.json").write_text(json.dumps(original))
            (source / "snapshot.json").write_text(json.dumps(snapshot))
            (source / "snapshot.json.columns.jsonl").write_text(
                json.dumps(route()) + "\n"
            )
            source_original_hash = MODULE.digest(source / "original.json")
            manifest = MODULE.reprice_source_root(source, destination, 5.0, 0.0)
            self.assertEqual(MODULE.digest(source / "original.json"), source_original_hash)
            self.assertEqual(manifest["records_repriced"], 1)
            copied = json.loads((destination / "original.json").read_text())
            self.assertEqual(copied["physics"]["charge_start_cost"], 0.0)
            copied_route = json.loads((destination / "snapshot.json.columns.jsonl").read_text())
            self.assertEqual(copied_route["source_cost"], 100015.0)
            self.assertEqual(copied_route["cost"], 100000.0)

    def test_common_union_reprices_sibling_to_target_fee(self):
        fee5 = {"frontiers": [{"duty_id": "d", "routes": [route()]}]}
        fee0 = {"frontiers": [{"duty_id": "d", "routes": [MODULE.reprice_route(route(), 5.0, 0.0)]}]}
        union, audit = MODULE.route_union(fee0, fee5, 0.0)
        costs = sorted(r["cost"] for r in union["frontiers"][0]["routes"])
        self.assertEqual(costs, [100000.0])
        self.assertEqual(audit["repriced_routes"], 1)
        self.assertEqual(union["frontiers"][0]["routes"][0]["frontier_repriced_for_fee"], 0.0)

    def test_runtime_override_accepts_zero_and_restores_fee5(self):
        MODULE.set_runtime_fee(0.0)
        import config
        self.assertEqual(config.CHARGE_START_COST, 0.0)
        MODULE.set_runtime_fee(5.0)
        self.assertEqual(config.CHARGE_START_COST, 5.0)


if __name__ == "__main__":
    unittest.main()

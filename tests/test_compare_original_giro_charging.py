import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compare_original_giro_charging import window_cost_bounds, recorded_activity_check


class OriginalChargingTests(unittest.TestCase):
    def test_flat_price_exact_without_profile(self):
        value = window_cost_bounds(50, 70, 10, {0: 2, 1: 2}, 60)
        self.assertEqual(value["energy_cost_exact"], 20)

    def test_variable_price_capacity_bounds(self):
        value = window_cost_bounds(50, 70, 15, {0: 1, 1: 3}, 60)
        self.assertEqual(value["energy_cost_lower"], 25)
        self.assertEqual(value["energy_cost_upper"], 35)
        self.assertEqual(value["uniform_power_assumption_energy_cost"], 30)
        self.assertIsNone(value["energy_cost_exact"])

    def test_full_power_identifies_cost_across_price_boundary(self):
        value = window_cost_bounds(50, 70, 20, {0: 1, 1: 3}, 60)
        self.assertEqual(value["energy_cost_exact"], 40)

    def test_missing_tariff_never_extends_last_price(self):
        with self.assertRaisesRegex(ValueError, "hour 1"):
            window_cost_bounds(50, 70, 10, {0: 1}, 60)

    def test_infeasible_power_does_not_repair_duration(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            window_cost_bounds(0, 1, 10, {0: 1}, 60)

    def test_negative_prices_still_have_ordered_bounds(self):
        value = window_cost_bounds(50, 70, 5, {0: -2, 1: 1}, 60)
        self.assertEqual(value["energy_cost_lower"], -10)
        self.assertEqual(value["energy_cost_upper"], 5)

    def test_soc_overflow_not_silently_clipped(self):
        rows = [{"Identifier": "Recharge", "Start1": "0:00", "End1": "0:10",
                 "Recharge kWh": "5"}]
        value = recorded_activity_check(rows, g_kwh=240, charge_kw=240, reserve_kwh=0)
        self.assertFalse(value["valid"])
        self.assertEqual(value["terminal_soc_kwh"], 245)

    def test_recorded_discharge_and_recharge(self):
        rows = [{"Identifier": "Regular", "Start1": "0:00", "End1": "0:10", "Usage kWh": "50"},
                {"Identifier": "Recharge", "Start1": "0:10", "End1": "0:20", "Recharge kWh": "20"}]
        value = recorded_activity_check(rows, g_kwh=240, charge_kw=240, reserve_kwh=0)
        self.assertTrue(value["valid"])
        self.assertEqual(value["terminal_soc_kwh"], 210)


if __name__ == "__main__":
    unittest.main()

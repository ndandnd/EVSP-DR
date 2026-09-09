import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_giro_known_columns import ProblemData  # noqa: E402
from audit_giro_duty_recovery import (  # noqa: E402
    fixed_sequence_recovery,
    unrestricted_pricing_recovery,
)
from giro_partille_physics import (  # noqa: E402
    PARTILLE_PROFILES,
    charge_soc_after_minutes,
    charge_window,
    profile_for_duty,
)


class PartillePhysicsTests(unittest.TestCase):
    def test_partille_profile_mapping_and_reserve_transform(self):
        e2 = profile_for_duty("13316uwt")
        e1 = profile_for_duty("13411")
        self.assertEqual(e2.name, "18E2")
        self.assertEqual(e1.name, "18E1")
        self.assertAlmostEqual(e2.reserve_kwh, 0.15 * 239.01)
        self.assertAlmostEqual(e1.transformed_usable_kwh, 0.85 * 236.44)

    def test_depot_and_tapered_opportunity_charging_are_distinct(self):
        profile = PARTILLE_PROFILES["18E2"]
        start = 0.15 * profile.usable_capacity_kwh
        depot = charge_soc_after_minutes(profile, "PARX_1", start, 10.0)
        opportunity = charge_soc_after_minutes(profile, "3127L_0", start, 10.0)
        self.assertAlmostEqual(depot - start, 10.0)
        self.assertGreater(opportunity - start, 50.0)

    def test_setup_and_minimum_charge_window_are_explicit(self):
        profile = PARTILLE_PROFILES["18E2"]
        start = profile.reserve_kwh + 1.0
        self.assertIsNone(charge_window(profile, "3127L_0", start, 3.74))
        result = charge_window(profile, "3127L_0", start, 3.75)
        self.assertIsNotNone(result)
        self.assertEqual(result["setup_min"], 0.75)
        self.assertEqual(result["power_delivery_min"], 3.0)
        self.assertEqual(result["connected_min"], 3.0)
        self.assertGreater(result["setup_idle_kwh"], 0.0)

    def test_3127_e2_80_percent_exception_is_preserved(self):
        profile = PARTILLE_PROFILES["18E2"]
        start = 0.70 * profile.usable_capacity_kwh
        at_3127 = charge_soc_after_minutes(profile, "3127L_0", start, 1.0)
        at_other = charge_soc_after_minutes(profile, "JON_A_0", start, 1.0)
        self.assertAlmostEqual(at_3127 - at_other, 1.0 / 60.0)

    def test_source_recharge_curve_fixture_matches_recorded_soc(self):
        """Par_VehicleDetails row10: 10.25min power plus0.75 setup."""

        profile = PARTILLE_PROFILES["18E2"]
        start = 0.66367096 * profile.usable_capacity_kwh
        end = charge_soc_after_minutes(profile, "7880C", start, 10.249998)
        self.assertAlmostEqual(
            end / profile.usable_capacity_kwh, 0.83250336, places=7
        )


def _toy_problem():
    return ProblemData(
        frame=None,
        trips=(0, 1),
        adjacency={
            "PARX_0": [(0, 0.0, 0.0, "depot_trip")],
            0: [
                (1, 0.0, 0.0, "trip_trip"),
                ("3127L_0", 0.0, 0.0, "trip_station"),
            ],
            "3127L_0": [
                (1, 0.0, 0.0, "station_trip"),
                ("PARX_0", 0.0, 0.0, "station_depot"),
            ],
            1: [("PARX_0", 0.0, 0.0, "trip_depot")],
        },
        start_min={0: 0.0, 1: 120.0},
        end_min={0: 60.0, 1: 180.0},
        trip_energy={0: 100.0, 1: 100.0},
    )


class DutyRecoveryTests(unittest.TestCase):
    def test_fixed_and_unrestricted_recovery_use_new_physics(self):
        problem = _toy_problem()
        profile = PARTILLE_PROFILES["18E2"]
        fixed = fixed_sequence_recovery(
            problem, (0, 1), profile, horizon_min=240.0
        )
        unrestricted = unrestricted_pricing_recovery(
            problem, profile, horizon_min=240.0
        )
        self.assertTrue(fixed["feasible"])
        self.assertEqual(fixed["charging_stops"], 1)
        self.assertTrue(unrestricted["recovered_all_trips"])
        self.assertEqual(unrestricted["trips"], (0, 1))

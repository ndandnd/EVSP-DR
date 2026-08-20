import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_giro_known_columns import DEPOT, STATIONS  # noqa: E402
from fixed_duty_continuous_optimizer import (  # noqa: E402
    optimize_fixed_duty_continuous,
    validate_lattice_reproduction,
)


STATION = STATIONS[0]


def prices(default=1.0, overrides=None):
    overrides = overrides or {}
    return {
        station.rsplit("_", 1)[0]: {
            hour: float(overrides.get(hour, default))
            for hour in range(27)
        }
        for station in STATIONS
    }


def two_trip_problem(first_energy=190.0, second_energy=100.0):
    return SimpleNamespace(
        trips=[0, 1],
        trip_energy={0: first_energy, 1: second_energy},
        start_min={0: 0.0, 1: 180.0},
        end_min={0: 10.0, 1: 190.0},
        adjacency={
            DEPOT: [(0, 0.0, 0.0, "depot_trip")],
            0: [(STATION, 0.0, 0.0, "trip_station")],
            STATION: [
                (1, 0.0, 0.0, "station_trip"),
                (DEPOT, 0.0, 0.0, "station_depot"),
            ],
            1: [(DEPOT, 0.0, 0.0, "trip_depot")],
        },
    )


class ContinuousFixedDutyTests(unittest.TestCase):
    def test_delays_charge_into_cheap_hour_and_replays_cost(self):
        result = optimize_fixed_duty_continuous(
            two_trip_problem(),
            [0, 1],
            prices(overrides={0: 10.0}),
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["charge_events"], 1)
        self.assertEqual(result["delayed_starts"], 1)
        self.assertGreaterEqual(result["charging_events"][0]["start_min"], 60)
        self.assertAlmostEqual(result["objective"], 100055.0)
        self.assertAlmostEqual(
            result["objective"],
            result["physical_replay"]["replayed_objective"],
            places=6,
        )
        self.assertEqual(result["peak_kw"], 240.0)
        self.assertEqual(result["charger_concurrency_max"], 1)

    def test_fixed_cost_counts_two_events_around_expensive_hour(self):
        result = optimize_fixed_duty_continuous(
            two_trip_problem(first_energy=180.0, second_energy=160.0),
            [0, 1],
            prices(default=1.0, overrides={1: 100.0}),
            charge_kw=60.0,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["charge_events"], 2)
        self.assertEqual(
            [event["start_min"] for event in result["charging_events"]],
            [10.0, 120.0],
        )
        self.assertAlmostEqual(result["charging_cost"], 110.0, places=6)
        self.assertAlmostEqual(
            sum(event["delivered_kwh"] for event in result["charging_events"]),
            100.0,
            places=6,
        )

    def test_terminal_policies_are_explicit_and_switchable(self):
        problem = SimpleNamespace(
            trips=[0],
            trip_energy={0: 100.0},
            start_min={0: 0.0},
            end_min={0: 10.0},
            adjacency={
                DEPOT: [(0, 0.0, 0.0, "depot_trip")],
                0: [
                    (DEPOT, 0.0, 0.0, "trip_depot"),
                    (STATION, 0.0, 0.0, "trip_station"),
                ],
                STATION: [(DEPOT, 0.0, 0.0, "station_depot")],
            },
        )
        free = optimize_fixed_duty_continuous(
            problem, [0], prices(), terminal_soc_policy="free"
        )
        restored = optimize_fixed_duty_continuous(
            problem, [0], prices(), terminal_soc_policy=">= start"
        )
        priced = optimize_fixed_duty_continuous(
            problem,
            [0],
            prices(),
            terminal_soc_policy="priced terminal energy",
            terminal_energy_price=2.0,
        )
        self.assertAlmostEqual(free["terminal_soc_kwh"], 140.0)
        self.assertEqual(free["charge_events"], 0)
        self.assertAlmostEqual(restored["terminal_soc_kwh"], 240.0)
        self.assertEqual(restored["charge_events"], 1)
        self.assertAlmostEqual(priced["terminal_soc_kwh"], 240.0)
        self.assertLess(priced["objective"], free["objective"])

    def test_flat_tariff_has_no_delayed_timing_advantage(self):
        problem = two_trip_problem()
        optimized = optimize_fixed_duty_continuous(
            problem, [0, 1], prices(), timing_mode="optimized"
        )
        arrival = optimize_fixed_duty_continuous(
            problem, [0, 1], prices(), timing_mode="arrival"
        )
        self.assertTrue(optimized["feasible"] and arrival["feasible"])
        self.assertTrue(math.isclose(
            optimized["objective"], arrival["objective"], abs_tol=1e-6
        ))
        self.assertEqual(
            optimized["charge_events"], arrival["charge_events"]
        )

    def test_legacy_lattice_milp_reproduces_dynamic_program(self):
        problem = two_trip_problem(first_energy=250.0, second_energy=40.0)
        result = validate_lattice_reproduction(
            problem, [0, 1], prices(default=1.0)
        )
        self.assertTrue(result["feasible"])
        self.assertTrue(result["matches"])
        self.assertAlmostEqual(result["difference"], 0.0, places=7)


if __name__ == "__main__":
    unittest.main()

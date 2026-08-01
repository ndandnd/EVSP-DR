import csv
import inspect
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pricing_dp_og import Label, _is_dominated, make_dp_pricer, solve_pricing_dp  # noqa: E402
from utils_v2 import (  # noqa: E402
    calculate_truck_route_cost_accurate,
    load_station_hourly_prices,
    route_column_key,
    select_unique_station_copies,
)


class DominanceTests(unittest.TestCase):
    def make_label(self, *, trips, charges=0, rc=0.0, time=100.0, soc=100.0):
        return Label(
            rc=rc,
            time=time,
            soc=soc,
            node=9,
            path=tuple(trips),
            trips_visited=frozenset(trips),
            charging_stops=tuple((f"S{i}", 0, 1, 1) for i in range(charges)),
        )

    def test_short_route_cannot_kill_minimum_trip_ready_route(self):
        short = self.make_label(trips=[9], rc=-10, time=90, soc=200)
        ready = self.make_label(trips=[1, 2, 9], rc=-5, time=100, soc=100)
        self.assertFalse(_is_dominated(ready, [short]))

    def test_more_complete_route_can_dominate_shorter_route(self):
        ready = self.make_label(trips=[1, 2, 9])
        short = self.make_label(trips=[9])
        self.assertTrue(_is_dominated(short, [ready]))

    def test_more_recharges_cannot_dominate_fewer_recharges(self):
        more_charges = self.make_label(trips=[1, 2, 9], charges=2, rc=-10)
        fewer_charges = self.make_label(trips=[1, 2, 9], charges=1, rc=-5)
        self.assertFalse(_is_dominated(fewer_charges, [more_charges]))

    def test_earlier_station_label_cannot_kill_later_feasible_window(self):
        early = self.make_label(trips=[1], charges=1, rc=-10, time=30, soc=300)
        late = self.make_label(trips=[1], charges=1, rc=-5, time=130, soc=300)
        early.node = late.node = "S_0"

        # For a successor at minute 300 and max_charge2trip=220, the early
        # label has already expired (270-minute gap) while the late one has not
        # (170-minute gap). They must remain incomparable at a station.
        self.assertTrue(_is_dominated(late, [early]))
        self.assertFalse(_is_dominated(late, [early], station_pool=True))

    def test_public_pricer_defaults_allow_one_trip_routes(self):
        for function in (solve_pricing_dp, make_dp_pricer):
            default = inspect.signature(function).parameters["MIN_TRIPS_PER_ROUTE"].default
            self.assertEqual(default, 1)


class UtilityTests(unittest.TestCase):
    def test_depot_charger_uses_distinct_copy(self):
        selected = select_unique_station_copies(
            ["2190L_0", "2190L_1", "PARX_0", "PARX_1", "PARX_2"],
            "PARX_0",
        )
        self.assertEqual(selected, ["2190L_0", "PARX_1"])
        self.assertNotIn("PARX_0", selected)

    def test_depot_charger_requires_distinct_graph_node(self):
        with self.assertRaises(ValueError):
            select_unique_station_copies(["PARX_0"], "PARX_0")

    def test_temporal_price_curve_is_replicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flat.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time_block", "cost"])
                writer.writerows([[0, 0.1], [1, 0.2]])
            curves = load_station_hourly_prices(path, ["PARX", "2190L_0"])
        self.assertEqual(curves["PARX"], {0: 0.1, 1: 0.2})
        self.assertEqual(curves["2190L"], curves["PARX"])

    def test_station_price_curve_rejects_missing_station(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spatial.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time_block", "station", "cost"])
                writer.writerows([[0, "PARX", 0.1], [1, "PARX", 0.1]])
            with self.assertRaises(ValueError):
                load_station_hourly_prices(path, ["PARX", "2190L"])

    def test_price_curve_rejects_missing_internal_hour(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hole.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time_block", "cost"])
                writer.writerows([[0, 0.1], [2, 0.2]])
            with self.assertRaises(ValueError):
                load_station_hourly_prices(path, ["PARX"])

    def test_price_curve_rejects_empty_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            path.write_text("time_block,cost\n")
            with self.assertRaises(ValueError):
                load_station_hourly_prices(path, ["PARX"])

    def test_route_key_includes_charging_realization(self):
        base = {
            "route": ["PARX_0", 1, "2190L_0", 2, "PARX_0"],
            "charging_stops": {
                "stations": ["2190L_0"], "cst": [100], "cet": [110], "kwh": [50]
            },
        }
        changed = {
            **base,
            "charging_stops": {
                "stations": ["2190L_0"], "cst": [100], "cet": [120], "kwh": [100]
            },
        }
        self.assertNotEqual(route_column_key(base), route_column_key(changed))

    def test_partial_kwh_metadata_is_rejected(self):
        route = {
            "route": ["PARX_0", "A_0", "B_0", "PARX_0"],
            "charging_stops": {
                "stations": ["A_0", "B_0"],
                "cst": [0, 60],
                "cet": [30, 90],
                "kwh": [150],
            },
        }
        with self.assertRaises(ValueError):
            calculate_truck_route_cost_accurate(
                route,
                100000,
                {0: 0.1, 1: 0.1},
                charge_rate_kw=300,
            )

    def test_partial_charge_time_metadata_is_rejected(self):
        route = {
            "route": ["PARX_0", "A_0", "PARX_0"],
            "charging_stops": {
                "stations": ["A_0"], "cst": [], "cet": [30], "kwh": [150]
            },
        }
        with self.assertRaises(ValueError):
            calculate_truck_route_cost_accurate(
                route,
                100000,
                {0: 0.1},
                charge_rate_kw=300,
            )


if __name__ == "__main__":
    unittest.main()

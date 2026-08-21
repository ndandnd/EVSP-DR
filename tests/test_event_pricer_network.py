import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from audit_giro_known_columns import DEPOT, STATIONS  # noqa: E402
from event_pricer_network import EventExpandedNetwork  # noqa: E402
from exact_pricer_expanded import ExpandedNetwork  # noqa: E402


STATION = STATIONS[0]


def two_trip_problem():
    return SimpleNamespace(
        trips=(0, 1),
        start_min={0: 0.0, 1: 180.0},
        end_min={0: 10.0, 1: 190.0},
        trip_energy={0: 190.0, 1: 100.0},
        adjacency={
            DEPOT: [(0, 0.0, 0.0, "depot_trip")],
            0: [
                (STATION, 0.0, 0.0, "trip_station"),
                (DEPOT, 0.0, 0.0, "trip_depot"),
            ],
            STATION: [
                (1, 0.0, 0.0, "station_trip"),
                (DEPOT, 0.0, 0.0, "station_depot"),
            ],
            1: [(DEPOT, 0.0, 0.0, "trip_depot")],
        },
    )


def prices():
    return {
        station.rsplit("_", 1)[0]: {
            hour: 0.1 for hour in range(27)
        }
        for station in STATIONS
    }


class EventPricerNetworkTests(unittest.TestCase):
    def test_event_route_uses_exact_window_and_replays(self):
        network = EventExpandedNetwork(
            two_trip_problem(),
            prices(),
            soc_step=2.5,
            block_min=5,
            g_kwh=240.0,
            charge_kw=240.0,
            reserve_kwh=0.0,
        )
        route = network.min_reduced_cost_route({0: 100000.0, 1: 100000.0})
        self.assertEqual(route["trips"], [0, 1])
        record = route["_event_record"]
        self.assertEqual(record["physical_realization"]["time_model"], "event")
        self.assertGreater(len(record["continuous_realized_charging_blocks"]), 0)
        self.assertLess(route["rc"], 0.0)

    def test_event_times_include_exact_and_reachable_uniform_breakpoints(self):
        network = EventExpandedNetwork(
            two_trip_problem(),
            prices(),
            soc_step=2.5,
            block_min=5,
            g_kwh=240.0,
            charge_kw=240.0,
            reserve_kwh=0.0,
        )
        events = set(network.events[STATION])
        self.assertIn(10.0, events)
        self.assertIn(180.0, events)
        self.assertTrue(set(range(10, 181, 5)) <= events)

    def test_historical_tariff_uses_uniform_model_last_hour_policy(self):
        historical = {
            station.rsplit("_", 1)[0]: {
                hour: 0.1 for hour in range(25)
            }
            for station in STATIONS
        }
        network = EventExpandedNetwork(
            two_trip_problem(),
            historical,
            soc_step=2.5,
            block_min=5,
            g_kwh=240.0,
            charge_kw=240.0,
            reserve_kwh=0.0,
        )
        self.assertEqual(network.prices[STATION.rsplit("_",1)[0]][25],0.1)

    def test_event_dag_is_smaller_than_uniform_one_minute_grid(self):
        problem = two_trip_problem()
        event = EventExpandedNetwork(
            problem, prices(), soc_step=2.5, block_min=5,
            g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0,
        )
        uniform = ExpandedNetwork(
            problem, prices(), soc_step=2.5, block_min=1,
            g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0,
        )
        self.assertLess(len(event.node_meta), len(uniform.node_meta))
        self.assertLess(event.n_arcs, uniform.n_arcs)


if __name__ == "__main__":
    unittest.main()

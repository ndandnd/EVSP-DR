import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from audit_giro_known_columns import DEPOT, STATIONS  # noqa: E402
from event_pricer_network import EventExpandedNetwork  # noqa: E402
from exact_pricer_expanded import (  # noqa: E402
    ExpandedNetwork,
    _load_event_network_cache,
    _write_event_network_cache,
)


STATION = STATIONS[0]


def two_trip_problem(first_energy=190.0):
    return SimpleNamespace(
        trips=(0, 1),
        start_min={0: 0.0, 1: 180.0},
        end_min={0: 10.0, 1: 190.0},
        trip_energy={0: first_energy, 1: 100.0},
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


def four_trip_chain_problem():
    trips = tuple(range(4))
    adjacency = {
        DEPOT: [
            (trip, 0.0, 0.0, "depot_trip") for trip in trips
        ],
    }
    for trip in trips:
        adjacency[trip] = [
            (successor, 0.0, 0.0, "trip_trip")
            for successor in trips[trip + 1:]
        ] + [(DEPOT, 0.0, 0.0, "trip_depot")]
    return SimpleNamespace(
        trips=trips,
        start_min={trip: float(60 * trip) for trip in trips},
        end_min={trip: float(60 * trip + 10) for trip in trips},
        trip_energy={trip: 1.0 for trip in trips},
        adjacency=adjacency,
    )


class EventPricerNetworkTests(unittest.TestCase):
    def test_complementary_batch_keeps_exact_best_and_adds_novel_incidence(self):
        network = EventExpandedNetwork(
            four_trip_chain_problem(), prices(), soc_step=15,
            block_min=10, g_kwh=240.0, charge_kw=240.0,
            reserve_kwh=0.0,
        )
        duals = {trip: 110000.0 for trip in range(4)}
        reduced = network.sink_predecessor_route_batch(
            duals, limit=3, selection_mode="reduced_cost",
        )
        complementary = network.sink_predecessor_route_batch(
            duals, limit=3, selection_mode="complementary",
            diversity_weight=1.0, candidate_multiplier=4,
        )
        self.assertEqual(complementary[0]["trips"], reduced[0]["trips"])
        self.assertAlmostEqual(complementary[0]["rc"], reduced[0]["rc"])
        self.assertEqual(len(complementary), 3)
        self.assertEqual(
            len({frozenset(route["trips"]) for route in complementary}), 3,
        )
        self.assertTrue(all(route["rc"] < 0 for route in complementary))
        self.assertEqual(complementary[1]["trips"], [0])

    def test_completed_event_network_cache_is_hash_validated(self):
        network = EventExpandedNetwork(
            two_trip_problem(), prices(), soc_step=15, block_min=10,
            g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0,
        )
        identity = {"schema": "test", "instance_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "network.pkl"
            manifest = _write_event_network_cache(
                cache, network, identity, 12.5
            )
            loaded, observed = _load_event_network_cache(cache, identity)
            self.assertEqual(observed, manifest)
            self.assertEqual(loaded.metrics(), network.metrics())
            self.assertEqual(loaded._window_cache, {})
            self.assertEqual(
                loaded.min_reduced_cost_route({0: 100000.0, 1: 100000.0})[
                    "trips"
                ],
                network.min_reduced_cost_route({0: 100000.0, 1: 100000.0})[
                    "trips"
                ],
            )
            self.assertTrue(loaded._window_cache)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                _load_event_network_cache(cache, {"schema": "wrong"})

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

    def test_event_replay_preserves_residual_soc_without_overfill(self):
        network = EventExpandedNetwork(
            two_trip_problem(first_energy=191.2),
            prices(),
            soc_step=2.5,
            block_min=5,
            g_kwh=240.0,
            charge_kw=240.0,
            reserve_kwh=0.0,
        )
        route = network.min_reduced_cost_route({0:100000.0,1:100000.0})
        record = route["_event_record"]
        self.assertLessEqual(
            record["charging_stops"]["kwh"][0],
            record["expanded_grid_charging_stops"]["kwh"][0],
        )
        self.assertEqual(
            record["physical_realization"]["status"],
            "valid_event_time_realized",
        )

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

    def test_strict_tariff_rejects_missing_intermediate_hour(self):
        incomplete = prices()
        for curve in incomplete.values():
            del curve[1]
        with self.assertRaisesRegex(ValueError,"coverage"):
            EventExpandedNetwork(
                two_trip_problem(),incomplete,
                soc_step=2.5,block_min=5,g_kwh=240.0,
                charge_kw=240.0,reserve_kwh=0.0,
                strict_tariff_coverage=True,
            )

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

    def test_parallel_event_options_are_deduplicated_by_future_state(self):
        network = EventExpandedNetwork(
            two_trip_problem(), prices(), soc_step=2.5, block_min=5,
            g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0,
            arc_mode="explicit",
        )
        for arcs in network.out:
            keys=[(target,dual) for target,_cost,dual,_action in arcs]
            self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(
            len(network.sink_arcs),
            len({source for source,_cost,_action in network.sink_arcs}),
        )

    def test_explicit_and_lazy_shortest_path_oracles_match(self):
        problem = two_trip_problem()
        explicit = EventExpandedNetwork(
            problem, prices(), soc_step=2.5, block_min=5,
            g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0,
            arc_mode="explicit",
        )
        lazy = EventExpandedNetwork(
            problem, prices(), soc_step=2.5, block_min=5,
            g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0,
            arc_mode="lazy",
        )
        self.assertEqual(lazy.n_arcs, explicit.n_arcs)
        self.assertEqual(
            lazy.metrics()["materialized_python_arc_objects"], 0,
        )
        self.assertGreater(lazy.metrics()["packed_arc_bytes"], 0)
        for duals in (
            {0: 100000.0, 1: 100000.0},
            {0: 100100.0, 1: 99900.0},
            {0: 0.0, 1: 0.0},
        ):
            with self.subTest(duals=duals):
                expected = explicit.min_reduced_cost_route(duals)
                observed = lazy.min_reduced_cost_route(duals)
                if expected is None:
                    self.assertIsNone(observed)
                    continue
                self.assertAlmostEqual(observed["rc"], expected["rc"])
                self.assertEqual(observed["trips"], expected["trips"])
                self.assertEqual(
                    observed["route_nodes"], expected["route_nodes"],
                )
                self.assertEqual(
                    observed["_event_record"],
                    expected["_event_record"],
                )

    def test_explicit_and_lazy_fixed_sequence_oracles_match(self):
        problem = two_trip_problem(first_energy=191.2)
        explicit = EventExpandedNetwork(
            problem, prices(), soc_step=2.5, block_min=5,
            g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0,
            arc_mode="explicit",
        )
        lazy = EventExpandedNetwork(
            problem, prices(), soc_step=2.5, block_min=5,
            g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0,
            arc_mode="lazy",
        )
        self.assertEqual(
            lazy.fixed_sequence_record((0, 1)),
            explicit.fixed_sequence_record((0, 1)),
        )

    def test_explicit_and_lazy_lexicographic_objectives_match(self):
        problem = two_trip_problem()
        explicit = EventExpandedNetwork(
            problem, prices(), soc_step=2.5, block_min=5,
            g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0,
            arc_mode="explicit",
        )
        lazy = EventExpandedNetwork(
            problem, prices(), soc_step=2.5, block_min=5,
            g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0,
            arc_mode="lazy",
        )
        duals = {0: 0.4, 1: 0.4}
        for objective, route_dual in (
            ("artificial-elimination", 0.0),
            ("fleet-only", 0.0),
            ("charging-cost", 0.25),
        ):
            with self.subTest(objective=objective):
                expected = explicit.min_reduced_cost_route(
                    duals, objective=objective, route_dual=route_dual,
                )
                observed = lazy.min_reduced_cost_route(
                    duals, objective=objective, route_dual=route_dual,
                )
                self.assertAlmostEqual(observed["rc"], expected["rc"])
                self.assertEqual(observed["trips"], expected["trips"])
                self.assertEqual(
                    observed["_event_record"],
                    expected["_event_record"],
                )
        fleet = lazy.min_reduced_cost_route(
            duals, objective="fleet-only",
        )
        self.assertAlmostEqual(
            fleet["rc"],
            1.0 - sum(duals[trip] for trip in fleet["trips"]),
        )


if __name__ == "__main__":
    unittest.main()

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from audit_giro_known_columns import (  # noqa: E402
    DEPOT,
    HORIZON_MIN,
    STATIONS,
    build_problem,
)
from config import CHARGING_STATIONS  # noqa: E402
import event_pricer_network  # noqa: E402
from event_pricer_network import EventExpandedNetwork  # noqa: E402
from utils_v2 import load_station_hourly_prices  # noqa: E402


class EagerTieKeyNetwork(EventExpandedNetwork):
    """Pre-optimization arc retention used as a bit-exact oracle."""

    def _add(self, source, target, cost, trip, action):
        dual = self.trip_position[trip] if trip is not None else -1
        row = (target, float(cost), dual, action)
        key = (target, dual)
        candidate = (row[1], json.dumps(action, sort_keys=True))
        retained = self._building_arcs.setdefault(source, {})
        current = retained.get(key)
        if current is None or candidate < current[0]:
            retained[key] = (candidate, row)

    def _finalize_source(self, source):
        retained = self._building_arcs.pop(source, {})
        rows = sorted(
            (value[1] for value in retained.values()),
            key=lambda row: (
                row[0], row[1], json.dumps(row[3], sort_keys=True)
            ),
        )
        if self.arc_mode == "explicit":
            self.out[source] = rows
        else:
            start = len(self._arc_targets)
            self._arc_targets.extend(row[0] for row in rows)
            self._arc_costs.extend(row[1] for row in rows)
            self._arc_recipes.extend(
                self._action_recipe(row[3]) for row in rows
            )
            self._arc_slices[source] = (start, len(self._arc_targets))
        self.sink_arcs.extend(
            (
                source,
                cost,
                action if self.arc_mode == "explicit" else None,
            )
            for target, cost, _dual, action in rows
            if target == self.SINK
        )


def restricted_problem(problem, trips):
    trip_set = set(trips)
    station_set = set(STATIONS)
    adjacency = {}
    for source, arcs in problem.adjacency.items():
        if source != DEPOT and source not in trip_set and source not in station_set:
            continue
        retained = [
            arc for arc in arcs
            if arc[0] == DEPOT or arc[0] in trip_set or arc[0] in station_set
        ]
        if retained:
            adjacency[source] = retained
    return replace(
        problem,
        trips=tuple(trips),
        adjacency=adjacency,
        start_min={trip: problem.start_min[trip] for trip in trips},
        end_min={trip: problem.end_min[trip] for trip in trips},
        trip_energy={trip: problem.trip_energy[trip] for trip in trips},
    )


def arc_builder():
    network = object.__new__(EventExpandedNetwork)
    network.trip_position = {10: 0, 20: 1}
    network._building_arcs = {}
    network.arc_mode = "explicit"
    network.out = [[] for _node in range(10)]
    network.sink_arcs = []
    network.SINK = 1
    return network


class EventPricerDeferredTieKeyTests(unittest.TestCase):
    def test_adversarial_cost_and_action_ties_match_eager_rule(self):
        network = arc_builder()
        actions = (
            (8.0, {"choice": "middle"}),
            (9.0, {"choice": "dearer"}),
            (7.0, {"choice": "z-last"}),
            (7.0, {"choice": "a-first"}),
            (7.0, {"choice": "m-middle"}),
        )
        for cost, action in actions:
            network._add(2, 5, cost, 10, action)
        network._finalize_source(2)
        self.assertEqual(
            network.out[2],
            [(5, 7.0, 0, {"choice": "a-first"})],
        )

    def test_json_is_deferred_until_a_tie_can_change_the_result(self):
        network = arc_builder()
        real_dumps = json.dumps
        calls = []

        def counted_dumps(*args, **kwargs):
            calls.append(args[0])
            return real_dumps(*args, **kwargs)

        with mock.patch.object(
            event_pricer_network.json, "dumps", side_effect=counted_dumps
        ):
            network._add(2, 5, 8.0, 10, {"choice": "initial"})
            network._add(2, 5, 9.0, 10, {"choice": "dearer"})
            network._add(2, 5, 7.0, 10, {"choice": "cheaper"})
            self.assertEqual(calls, [])
            network._add(2, 5, 7.0, 10, {"choice": "tie"})
            self.assertEqual(len(calls), 2)
            network._finalize_source(2)
            self.assertEqual(len(calls), 2)

    def test_final_order_uses_action_json_only_for_exact_target_cost_ties(self):
        network = arc_builder()
        network._add(2, 7, 3.0, 10, {"choice": "z-last"})
        network._add(2, 7, 3.0, 20, {"choice": "a-first"})
        network._add(2, 6, 4.0, 10, {"choice": "unrelated"})
        network._finalize_source(2)
        self.assertEqual(
            network.out[2],
            [
                (6, 4.0, 0, {"choice": "unrelated"}),
                (7, 3.0, 1, {"choice": "a-first"}),
                (7, 3.0, 0, {"choice": "z-last"}),
            ],
        )

    def test_small_real_graph_is_identical_to_eager_implementation(self):
        data = REPO / "data"
        problem = build_problem(
            data / "overnight_decomposition_20260912",
            "d00_g3.csv",
            max_station_to_trip_wait_min=HORIZON_MIN,
            reference_data_dir=data,
        )
        problem = restricted_problem(problem, problem.trips[:12])
        prices = load_station_hourly_prices(
            data / "hourly_prices_flat.csv", CHARGING_STATIONS
        )
        kwargs = {
            "soc_step": 2.5,
            "block_min": 5,
            "g_kwh": 240.0,
            "charge_kw": 240.0,
            "reserve_kwh": 0.0,
        }
        for arc_mode in ("explicit", "lazy"):
            with self.subTest(arc_mode=arc_mode):
                expected = EagerTieKeyNetwork(
                    problem, prices, arc_mode=arc_mode, **kwargs
                )
                observed = EventExpandedNetwork(
                    problem, prices, arc_mode=arc_mode, **kwargs
                )
                self.assertEqual(observed.node_meta, expected.node_meta)
                self.assertEqual(observed.topo, expected.topo)
                self.assertEqual(observed.sink_arcs, expected.sink_arcs)
                self.assertEqual(observed.metrics(), expected.metrics())
                if arc_mode == "explicit":
                    self.assertEqual(observed.out, expected.out)
                else:
                    self.assertEqual(
                        observed._arc_targets, expected._arc_targets
                    )
                    self.assertEqual(observed._arc_costs, expected._arc_costs)
                    self.assertEqual(
                        observed._arc_recipes, expected._arc_recipes
                    )
                    self.assertEqual(observed._arc_slices, expected._arc_slices)
                duals = {trip: 100000.0 for trip in problem.trips}
                expected_route = expected.min_reduced_cost_route(duals)
                observed_route = observed.min_reduced_cost_route(duals)
                for key in (
                    "rc", "trips", "charging_stops", "route_nodes",
                    "charges_started", "_event_record",
                ):
                    self.assertEqual(observed_route[key], expected_route[key])


if __name__ == "__main__":
    unittest.main()

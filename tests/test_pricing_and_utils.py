import csv
import heapq
import inspect
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pricing_dp_og import (  # noqa: E402
    Label,
    PricingRunStats,
    _build_remaining_dual_bound,
    _cap_label_pool,
    _is_dominated,
    _make_label_priority_key,
    _make_label_queue_entry,
    _select_negative_labels,
    _successor_boundary_soc_levels,
    make_dp_pricer,
    solve_pricing_dp,
)
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

    def make_tiny_pricer(self, queue_order):
        return make_dp_pricer(
            T=[0],
            S_use=[],
            DEPOT="D",
            tau={("D", 0): 0, (0, "D"): 0},
            d={("D", 0): 0.0, (0, "D"): 0.0},
            st={0: 1},
            et={0: 2},
            sl={0: "A"},
            el={0: "B"},
            epsilon={0: 0.0},
            G=300.0,
            TB_MIN=1,
            bar_t=10,
            bus_cost=100.0,
            charge_rate_kw=300.0,
            hourly_prices={},
            charge_cost_premium=1.0,
            travel_cost_factor=0.0,
            RC_EPSILON=0.1,
            K_BEST=5,
            st_min={0: 0.0},
            et_min={0: 1.0},
            tau_min={("D", 0): 0.0, (0, "D"): 0.0},
            queue_order=queue_order,
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
        self.assertTrue(
            _is_dominated(
                late,
                [early],
                station_pool=True,
                station_waiting_unrestricted=True,
            )
        )

    def test_public_pricer_defaults_allow_one_trip_routes(self):
        for function in (solve_pricing_dp, make_dp_pricer):
            default = inspect.signature(function).parameters["MIN_TRIPS_PER_ROUTE"].default
            self.assertEqual(default, 1)

    def test_public_pricer_defaults_preserve_time_queue_order(self):
        for function in (solve_pricing_dp, make_dp_pricer):
            default = inspect.signature(function).parameters["queue_order"].default
            self.assertEqual(default, "time")

    def test_public_pricer_defaults_preserve_reduced_cost_output(self):
        for function in (solve_pricing_dp, make_dp_pricer):
            default = inspect.signature(function).parameters[
                "output_selection"
            ].default
            self.assertEqual(default, "reduced_cost")

    def test_public_pricer_defaults_preserve_fixed_soc_grid(self):
        for function in (solve_pricing_dp, make_dp_pricer):
            default = inspect.signature(function).parameters[
                "successor_charge_targets"
            ].default
            self.assertFalse(default)

    def test_queue_order_changes_which_label_pops_first(self):
        early_expensive = self.make_label(trips=[1], rc=10.0, time=100.0)
        late_negative = self.make_label(trips=[2], rc=-10.0, time=200.0)

        time_entry = _make_label_queue_entry("time")
        time_heap = [time_entry(early_expensive, 0), time_entry(late_negative, 1)]
        heapq.heapify(time_heap)
        self.assertIs(heapq.heappop(time_heap)[-1], early_expensive)

        reduced_cost_entry = _make_label_queue_entry("reduced_cost")
        reduced_cost_heap = [
            reduced_cost_entry(early_expensive, 0),
            reduced_cost_entry(late_negative, 1),
        ]
        heapq.heapify(reduced_cost_heap)
        self.assertIs(heapq.heappop(reduced_cost_heap)[-1], late_negative)

    def test_future_dual_bound_changes_which_label_pops_first(self):
        bound = _build_remaining_dual_bound(
            alpha={0: 100.0},
            T=[0],
            trip_start_min={0: 150.0},
            trip_end_min={0: 160.0},
        )
        early_promising = self.make_label(trips=[1], rc=-1.0, time=100.0)
        late_negative = self.make_label(trips=[2], rc=-10.0, time=200.0)
        entry = _make_label_queue_entry("reduced_cost_bound", bound)
        queue = [entry(early_promising, 0), entry(late_negative, 1)]

        heapq.heapify(queue)

        self.assertIs(heapq.heappop(queue)[-1], early_promising)

    def test_remaining_dual_bound_uses_nonoverlapping_future_trips(self):
        bound = _build_remaining_dual_bound(
            alpha={0: 5.0, 1: 100.0, 2: 6.0, 3: 4.0, 4: -500.0},
            T=[0, 1, 2, 3, 4],
            trip_start_min={0: 0.0, 1: 5.0, 2: 10.0, 3: 7.0, 4: 20.0},
            trip_end_min={0: 10.0, 1: 15.0, 2: 20.0, 3: 7.0, 4: 30.0},
        )

        # The positive-duration WIS chooses trip 1 (100) over trips 0+2
        # (11). The zero-duration trip is conservatively added separately.
        self.assertAlmostEqual(bound(0.0), 104.0)
        self.assertAlmostEqual(bound(10.0), 6.0)
        self.assertAlmostEqual(bound(21.0), 0.0)

    def test_bound_queue_requires_a_bound_function(self):
        with self.assertRaisesRegex(ValueError, "remaining-dual bound"):
            _make_label_queue_entry("reduced_cost_bound")

    def test_label_cap_retention_uses_configured_bound_priority(self):
        labels = [
            self.make_label(trips=[1], rc=-10.0, time=200.0),
            self.make_label(trips=[2], rc=-1.0, time=100.0),
        ]
        bound = lambda time_min: 100.0 if time_min < 150.0 else 0.0
        priority = _make_label_priority_key("reduced_cost_bound", bound)

        # Hysteresis triggers only above max_labels + 50.
        padded = labels + [
            self.make_label(trips=[index + 3], rc=1000.0 + index, time=300.0)
            for index in range(50)
        ]
        kept, evicted = _cap_label_pool(
            padded,
            max_labels=1,
            priority_key=priority,
        )

        self.assertEqual(kept, [labels[1]])
        self.assertEqual(evicted, 51)
        self.assertTrue(labels[1].alive)
        self.assertFalse(labels[0].alive)

    def test_invalid_queue_order_is_rejected_by_public_solver(self):
        with self.assertRaisesRegex(ValueError, "queue_order"):
            solve_pricing_dp(
                alpha={},
                T=[],
                S_use=[],
                queue_order="not-an-order",
            )
        with self.assertRaisesRegex(ValueError, "queue_order"):
            self.make_tiny_pricer("not-an-order")

    def test_factory_threads_queue_order_and_exposes_last_stats(self):
        pricer = self.make_tiny_pricer("reduced_cost")

        routes, best_rc, timed_out = pricer({0: 200.0})

        self.assertEqual(pricer.queue_order, "reduced_cost")
        self.assertEqual(len(routes), 1)
        self.assertAlmostEqual(best_rc, -100.0)
        self.assertFalse(timed_out)
        self.assertIsInstance(pricer.last_stats, PricingRunStats)
        self.assertEqual(pricer.last_stats.queue_order, "reduced_cost")

    def test_diversified_output_selection_has_deterministic_quotas(self):
        best_rc = self.make_label(trips=[1], rc=-100.0)
        longest = self.make_label(trips=[1, 2, 3, 4], rc=-90.0)
        rare = self.make_label(trips=[9], rc=-80.0)
        common_a = self.make_label(trips=[1, 2], rc=-70.0)
        common_b = self.make_label(trips=[3, 4], rc=-60.0)
        labels = [best_rc, longest, rare, common_a, common_b]

        selected = _select_negative_labels(
            list(reversed(labels)),
            k_best=3,
            output_selection="diversified",
        )

        # One slot each goes to best RC, longest route, and rarest incidence.
        self.assertEqual(selected, [best_rc, longest, rare])
        self.assertEqual(
            selected,
            _select_negative_labels(
                labels,
                k_best=3,
                output_selection="diversified",
            ),
        )

    def test_output_selection_k_edges(self):
        labels = [
            self.make_label(trips=[1], rc=-3.0),
            self.make_label(trips=[2], rc=-2.0),
        ]

        self.assertEqual(
            _select_negative_labels(
                labels,
                k_best=0,
                output_selection="diversified",
            ),
            [],
        )
        self.assertEqual(
            _select_negative_labels(
                labels,
                k_best=10,
                output_selection="diversified",
            ),
            labels,
        )
        self.assertEqual(
            _select_negative_labels(
                list(reversed(labels)),
                k_best=1,
                output_selection="diversified",
            ),
            [labels[0]],
        )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _select_negative_labels(
                labels,
                k_best=-1,
                output_selection="diversified",
            )
        with self.assertRaisesRegex(ValueError, "output_selection"):
            _select_negative_labels(
                labels,
                k_best=1,
                output_selection="unknown",
            )

    def test_factory_threads_optional_output_selection(self):
        pricer = make_dp_pricer(
            T=[0],
            S_use=[],
            DEPOT="D",
            tau={("D", 0): 0, (0, "D"): 0},
            d={("D", 0): 0.0, (0, "D"): 0.0},
            st={0: 1},
            et={0: 2},
            sl={0: "A"},
            el={0: "B"},
            epsilon={0: 0.0},
            G=300.0,
            TB_MIN=1,
            bar_t=10,
            bus_cost=100.0,
            charge_rate_kw=300.0,
            hourly_prices={},
            charge_cost_premium=1.0,
            travel_cost_factor=0.0,
            RC_EPSILON=0.1,
            K_BEST=5,
            st_min={0: 0.0},
            et_min={0: 1.0},
            tau_min={("D", 0): 0.0, (0, "D"): 0.0},
            output_selection="diversified",
        )

        routes, _, _ = pricer({0: 200.0})

        self.assertEqual(len(routes), 1)
        self.assertEqual(pricer.output_selection, "diversified")
        self.assertEqual(pricer.last_stats.output_selection, "diversified")

    def test_factory_reuses_a_prebuilt_adjacency(self):
        adjacency = {
            "D": [(0, 0.0, 0.0, "depot_trip")],
            0: [("D", 0.0, 0.0, "trip_depot")],
        }
        kwargs = dict(
            T=[0],
            S_use=[],
            DEPOT="D",
            tau={},
            d={},
            st={0: 1},
            et={0: 2},
            sl={0: "A"},
            el={0: "B"},
            epsilon={0: 0.0},
            G=300.0,
            TB_MIN=1,
            bar_t=10,
            bus_cost=100.0,
            charge_rate_kw=300.0,
            hourly_prices={},
            charge_cost_premium=1.0,
            travel_cost_factor=0.0,
            RC_EPSILON=0.1,
            K_BEST=5,
            st_min={0: 0.0},
            et_min={0: 1.0},
            tau_min={},
            adj=adjacency,
        )

        with mock.patch("pricing_dp_og.build_dag", side_effect=AssertionError):
            pricer = make_dp_pricer(**kwargs)

        self.assertIs(pricer.adjacency, adjacency)

    def test_existing_pattern_is_filtered_before_kbest_cutoff(self):
        incumbent_costs = {
            frozenset({0}): 100.0,
            frozenset({1}): 110.0,
        }
        routes, timed_out, stats = solve_pricing_dp(
            alpha={0: 300.0, 1: 200.0},
            T=[0, 1],
            S_use=[],
            DEPOT="D",
            adj={
                "D": [
                    (0, 0.0, 0.0, "depot_trip"),
                    (1, 0.0, 0.0, "depot_trip"),
                ],
                0: [("D", 0.0, 0.0, "trip_depot")],
                1: [("D", 0.0, 0.0, "trip_depot")],
            },
            tau={},
            d={},
            st={0: 1, 1: 1},
            et={0: 2, 1: 2},
            sl={0: "A", 1: "B"},
            el={0: "A", 1: "B"},
            epsilon={0: 0.0, 1: 0.0},
            st_min={0: 0.0, 1: 0.0},
            et_min={0: 1.0, 1: 1.0},
            G=300.0,
            TB_MIN=1,
            bar_t=10,
            bus_cost=100.0,
            hourly_prices={},
            travel_cost_factor=0.0,
            RC_EPSILON=0.1,
            K_BEST=1,
            output_selection="diversified",
            existing_trip_set_costs=incumbent_costs,
            return_stats=True,
        )

        self.assertFalse(timed_out)
        self.assertTrue(stats.exhaustive)
        self.assertEqual(stats.completed_routes, 2)
        self.assertEqual(stats.negative_completed, 2)
        self.assertEqual(stats.output_selection, "diversified")
        self.assertEqual(stats.eligible_negative_incidences, 1)
        self.assertAlmostEqual(stats.best_reduced_cost, -200.0)
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["route"], ["D", 1, "D"])
        self.assertLess(routes[0]["_rc"], -0.1)
        returned_incidence = frozenset({1})
        returned_master_cost = routes[0]["_rc"] + 200.0
        self.assertLess(returned_master_cost, incumbent_costs[returned_incidence])

    def test_cheaper_realization_of_existing_pattern_is_still_returned(self):
        routes, _ = solve_pricing_dp(
            alpha={0: 300.0},
            T=[0],
            S_use=[],
            DEPOT="D",
            adj={
                "D": [(0, 0.0, 0.0, "depot_trip")],
                0: [("D", 0.0, 0.0, "trip_depot")],
            },
            tau={},
            d={},
            st={0: 1},
            et={0: 2},
            sl={0: "A"},
            el={0: "A"},
            epsilon={0: 0.0},
            st_min={0: 0.0},
            et_min={0: 1.0},
            G=300.0,
            TB_MIN=1,
            bar_t=10,
            bus_cost=100.0,
            hourly_prices={},
            travel_cost_factor=0.0,
            RC_EPSILON=0.1,
            K_BEST=1,
            existing_trip_set_costs={frozenset({0}): 150.0},
        )

        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["route"], ["D", 0, "D"])

    def test_successor_boundary_soc_target_repairs_partial_charge_gap(self):
        levels = _successor_boundary_soc_levels(
            base_levels=[30.0 * index for index in range(1, 11)],
            successor_latest_departures=[413.0],
            arrival_soc=216.1329999,
            arrival_time_min=409.0,
            G=300.0,
            charge_rate_kw=300.0,
            max_successor_targets=64,
        )

        self.assertIn(236.1329999, levels)

    def test_actual_dp_needs_successor_boundary_target_for_known_shape(self):
        kwargs = dict(
            alpha={0: 500.0, 1: 500.0},
            T=[0, 1],
            S_use=["S"],
            DEPOT="D",
            adj={
                "D": [(0, 0.0, 0.0, "depot_trip")],
                0: [("S", 408.0, 0.0, "trip_station")],
                "S": [(1, 0.0, 0.0, "station_trip")],
                1: [("D", 0.0, 0.0, "trip_depot")],
            },
            tau={},
            d={},
            st={0: 1, 1: 414},
            et={0: 2, 1: 415},
            sl={0: "A", 1: "B"},
            el={0: "A", 1: "B"},
            epsilon={0: 83.8670001, 1: 236.0},
            st_min={0: 0.0, 1: 413.0},
            et_min={0: 1.0, 1: 414.0},
            G=300.0,
            TB_MIN=1,
            bar_t=500,
            bus_cost=100.0,
            charge_rate_kw=300.0,
            hourly_prices={},
            charge_cost_premium=0.0,
            travel_cost_factor=0.0,
            RC_EPSILON=0.1,
            K_BEST=5,
            MAX_LABELS_PER_NODE=100,
            soc_charge_levels=[30.0 * index for index in range(1, 11)],
            MIN_TRIPS_PER_ROUTE=2,
            max_charge2trip=220,
        )

        grid_only, _ = solve_pricing_dp(
            **kwargs,
            successor_charge_targets=False,
        )
        boundary, _ = solve_pricing_dp(
            **kwargs,
            successor_charge_targets=True,
        )

        self.assertEqual(grid_only, [])
        self.assertEqual(len(boundary), 1)
        self.assertEqual(
            [node for node in boundary[0]["route"] if isinstance(node, int)],
            [0, 1],
        )

    def test_higher_soc_station_pass_through_preserves_dominance(self):
        # Labels 0->2 and 1->2 meet at trip 2. The former has lower reduced
        # cost and one extra kWh, so it correctly dominates the latter. At a
        # full battery it can continue through S only if a zero-energy station
        # pass-through is available, including at the recharge-count limit.
        routes, timed_out, stats = solve_pricing_dp(
            alpha={0: 20.0, 1: 10.0, 2: 20.0, 3: 200.0},
            T=[0, 1, 2, 3],
            S_use=["S"],
            DEPOT="D",
            adj={
                "D": [
                    (0, 0.0, 0.0, "depot_trip"),
                    (1, 0.0, 0.0, "depot_trip"),
                ],
                0: [(2, 0.0, 0.0, "trip_trip")],
                1: [(2, 0.0, 0.0, "trip_trip")],
                2: [("S", 0.0, 0.0, "trip_station")],
                "S": [(3, 0.0, 0.0, "station_trip")],
                3: [("D", 0.0, 0.0, "trip_depot")],
            },
            tau={},
            d={},
            st={trip: 1 for trip in range(4)},
            et={trip: 2 for trip in range(4)},
            sl={},
            el={},
            epsilon={0: 0.0, 1: 1.0, 2: 0.0, 3: 100.0},
            st_min={0: 0.0, 1: 0.0, 2: 10.0, 3: 20.0},
            et_min={0: 1.0, 1: 1.0, 2: 11.0, 3: 21.0},
            G=100.0,
            TB_MIN=1,
            bar_t=30,
            bus_cost=100.0,
            charge_rate_kw=300.0,
            hourly_prices={0: 0.0},
            charge_cost_premium=0.0,
            travel_cost_factor=0.0,
            RC_EPSILON=0.1,
            K_BEST=10,
            MAX_LABELS_PER_NODE=100,
            soc_charge_levels=[100.0],
            successor_charge_targets=True,
            MIN_TRIPS_PER_ROUTE=3,
            MAX_DAILY_RECHARGES=0,
            max_charge2trip=30,
            return_stats=True,
        )

        self.assertFalse(timed_out)
        self.assertTrue(stats.exhaustive)
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["route"], ["D", 0, 2, "S", 3, "D"])
        self.assertEqual(routes[0]["charging_activities"], 0)
        self.assertEqual(routes[0]["charging_stops"]["kwh"], [])

    def test_dominance_tolerance_cannot_prune_boundary_feasible_soc(self):
        # At trip 2, path 0->2 is 0.00005 kWh short of path 1->2. The old
        # 1e-4 dominance tolerance treated the lower-SOC path as superior even
        # though it cannot cover the last 10 kWh trip under the solver's 1e-6
        # feasibility tolerance. The exactly feasible path must survive.
        routes, timed_out, stats = solve_pricing_dp(
            alpha={0: 21.0, 1: 20.0, 2: 20.0, 3: 100.0},
            T=[0, 1, 2, 3],
            S_use=[],
            DEPOT="D",
            adj={
                "D": [
                    (0, 0.0, 0.0, "depot_trip"),
                    (1, 0.0, 0.0, "depot_trip"),
                ],
                0: [(2, 0.0, 0.0, "trip_trip")],
                1: [(2, 0.0, 0.0, "trip_trip")],
                2: [(3, 0.0, 0.0, "trip_trip")],
                3: [("D", 0.0, 0.0, "trip_depot")],
            },
            tau={},
            d={},
            st={trip: 1 for trip in range(4)},
            et={trip: 2 for trip in range(4)},
            sl={},
            el={},
            epsilon={0: 0.00005, 1: 0.0, 2: 0.0, 3: 10.0},
            st_min={0: 0.0, 1: 0.0, 2: 10.0, 3: 20.0},
            et_min={0: 1.0, 1: 1.0, 2: 11.0, 3: 21.0},
            G=10.0,
            TB_MIN=1,
            bar_t=30,
            bus_cost=100.0,
            hourly_prices={},
            travel_cost_factor=0.0,
            RC_EPSILON=0.1,
            K_BEST=10,
            MAX_LABELS_PER_NODE=100,
            MIN_TRIPS_PER_ROUTE=3,
            return_stats=True,
        )

        self.assertFalse(timed_out)
        self.assertTrue(stats.exhaustive)
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["route"], ["D", 1, 2, 3, "D"])
        self.assertAlmostEqual(routes[0]["_rc"], -40.0)

    def test_restricted_wait_requires_equal_soc_for_dominance(self):
        # At trip 2 the cheaper path through trip 0 has 90 kWh, while the path
        # through trip 1 has 0 kWh.  With a five-minute station-to-trip wait,
        # charging 90->100 ends too early and cannot reach trip 3; charging
        # 0->100 takes long enough to enter the window.  Higher SOC is thus not
        # a monotone resource until explicit station waiting is modeled.
        routes, timed_out, stats = solve_pricing_dp(
            alpha={0: 30.0, 1: 20.0, 2: 20.0, 3: 200.0},
            T=[0, 1, 2, 3],
            S_use=["S"],
            DEPOT="D",
            adj={
                "D": [
                    (0, 0.0, 0.0, "depot_trip"),
                    (1, 0.0, 0.0, "depot_trip"),
                ],
                0: [(2, 0.0, 0.0, "trip_trip")],
                1: [(2, 0.0, 0.0, "trip_trip")],
                2: [("S", 0.0, 0.0, "trip_station")],
                "S": [(3, 0.0, 0.0, "station_trip")],
                3: [("D", 0.0, 0.0, "trip_depot")],
            },
            tau={},
            d={},
            st={trip: 1 for trip in range(4)},
            et={trip: 2 for trip in range(4)},
            sl={},
            el={},
            epsilon={0: 10.0, 1: 100.0, 2: 0.0, 3: 100.0},
            st_min={0: 0.0, 1: 0.0, 2: 9.0, 3: 30.0},
            et_min={0: 1.0, 1: 1.0, 2: 10.0, 3: 31.0},
            G=100.0,
            TB_MIN=1,
            bar_t=40,
            bus_cost=100.0,
            charge_rate_kw=300.0,
            hourly_prices={0: 0.0},
            charge_cost_premium=0.0,
            travel_cost_factor=0.0,
            RC_EPSILON=0.1,
            K_BEST=10,
            MAX_LABELS_PER_NODE=100,
            soc_charge_levels=[100.0],
            successor_charge_targets=True,
            MIN_TRIPS_PER_ROUTE=3,
            MAX_DAILY_RECHARGES=1,
            max_charge2trip=5,
            return_stats=True,
        )

        self.assertFalse(timed_out)
        self.assertTrue(stats.exhaustive)
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["route"], ["D", 1, 2, "S", 3, "D"])
        self.assertEqual(routes[0]["charging_activities"], 1)
        self.assertEqual(routes[0]["charging_stops"]["kwh"], [100.0])

    def test_relaxed_station_wait_makes_split_shift_representable(self):
        kwargs = dict(
            alpha={0: 500.0, 1: 500.0},
            T=[0, 1],
            S_use=["S"],
            DEPOT="D",
            adj={
                "D": [(0, 0.0, 0.0, "depot_trip")],
                0: [("S", 0.0, 0.0, "trip_station")],
                "S": [(1, 0.0, 0.0, "station_trip")],
                1: [("D", 0.0, 0.0, "trip_depot")],
            },
            tau={},
            d={},
            st={0: 1, 1: 401},
            et={0: 11, 1: 411},
            sl={0: "A", 1: "B"},
            el={0: "A", 1: "B"},
            epsilon={0: 100.0, 1: 10.0},
            st_min={0: 0.0, 1: 400.0},
            et_min={0: 10.0, 1: 410.0},
            G=100.0,
            TB_MIN=1,
            bar_t=500,
            bus_cost=100.0,
            charge_rate_kw=300.0,
            hourly_prices={},
            charge_cost_premium=0.0,
            travel_cost_factor=0.0,
            RC_EPSILON=0.1,
            K_BEST=5,
            MAX_LABELS_PER_NODE=100,
            soc_charge_levels=[100.0],
            MIN_TRIPS_PER_ROUTE=2,
        )

        restricted, _ = solve_pricing_dp(**kwargs, max_charge2trip=220)
        horizon_wait, _ = solve_pricing_dp(**kwargs, max_charge2trip=500)

        self.assertEqual(restricted, [])
        self.assertEqual(len(horizon_wait), 1)

    def test_pricing_stats_report_actual_search_counts(self):
        routes, timed_out, stats = solve_pricing_dp(
            alpha={0: 200.0},
            T=[0],
            S_use=[],
            DEPOT="D",
            adj={
                "D": [(0, 0.0, 0.0, "depot_trip")],
                0: [("D", 0.0, 0.0, "trip_depot")],
            },
            tau={},
            d={},
            st={0: 1},
            et={0: 2},
            sl={0: "A"},
            el={0: "B"},
            epsilon={0: 0.0},
            st_min={0: 0.0},
            et_min={0: 1.0},
            G=300.0,
            TB_MIN=1,
            bar_t=10,
            bus_cost=100.0,
            hourly_prices={},
            travel_cost_factor=0.0,
            RC_EPSILON=0.1,
            K_BEST=5,
            queue_order="reduced_cost",
            return_stats=True,
        )

        self.assertFalse(timed_out)
        self.assertEqual(len(routes), 1)
        self.assertIsInstance(stats, PricingRunStats)
        self.assertEqual(stats.queue_order, "reduced_cost")
        self.assertEqual(stats.labels_expanded, 2)
        self.assertEqual(stats.completed_routes, 1)
        self.assertEqual(stats.negative_completed, 1)
        self.assertEqual(stats.output_selection, "reduced_cost")
        self.assertEqual(stats.eligible_negative_incidences, 1)
        self.assertEqual(stats.returned_trip_count_min, 1)
        self.assertEqual(stats.returned_trip_count_mean, 1.0)
        self.assertEqual(stats.returned_trip_count_max, 1)
        self.assertEqual(stats.label_cap_evictions, 0)
        self.assertTrue(stats.exhaustive)
        self.assertFalse(stats.timed_out)
        self.assertGreaterEqual(stats.elapsed_s, 0.0)

    def test_actual_dp_marks_label_cap_truncation_nonexhaustive(self):
        predecessors = list(range(60))
        common = 60
        trips = predecessors + [common]
        adjacency = {
            "D": [(trip, 0.0, 0.0, "depot_trip") for trip in predecessors],
            common: [("D", 0.0, 0.0, "trip_depot")],
        }
        for trip in predecessors:
            adjacency[trip] = [(common, 0.0, 0.0, "trip_trip")]

        _, timed_out, stats = solve_pricing_dp(
            alpha={**{trip: float(trip) for trip in predecessors}, common: 0.0},
            T=trips,
            S_use=[],
            DEPOT="D",
            adj=adjacency,
            tau={},
            d={},
            st={trip: 1 for trip in trips},
            et={trip: 2 for trip in trips},
            sl={trip: "A" for trip in trips},
            el={trip: "B" for trip in trips},
            epsilon={**{trip: float(trip) for trip in predecessors}, common: 0.0},
            st_min={**{trip: 0.0 for trip in predecessors}, common: 2.0},
            et_min={**{trip: 1.0 for trip in predecessors}, common: 3.0},
            G=100.0,
            TB_MIN=1,
            bar_t=10,
            bus_cost=1000.0,
            hourly_prices={},
            travel_cost_factor=0.0,
            RC_EPSILON=0.1,
            K_BEST=5,
            MAX_LABELS_PER_NODE=1,
            MIN_TRIPS_PER_ROUTE=2,
            queue_order="time",
            return_stats=True,
        )

        self.assertFalse(timed_out)
        self.assertGreater(stats.label_cap_evictions, 0)
        self.assertFalse(stats.exhaustive)


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

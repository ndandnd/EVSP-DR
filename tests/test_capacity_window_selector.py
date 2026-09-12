"""Reference differential checks plus an independent returned-column oracle."""
import math
import random
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import event_pricer_network as ep
from capacity_window_selector import CapacityWindowSelector
from test_capacity_accounting_regression import fixture, prices
from utils_v2 import base_station_name


class CapacitySelectorTests(unittest.TestCase):
    def blank(self):
        net = ep.EventExpandedNetwork.__new__(ep.EventExpandedNetwork)
        net.prices = prices()
        net.events = {s: tuple(range(0, 1621, 5)) for s in ep.STATIONS}
        net.charge_kw = 240.
        net.station_charge_kw = {ep.STATIONS[0]: 60.}
        return net

    def test_differential_breakpoints_sparse_duals_and_metadata(self):
        net = self.blank()
        rng = random.Random(20260912)
        base = base_station_name(ep.STATIONS[0])
        vectors = [{}, {(base, 60): 0.}, {(base, 60): -123.4567},
            {(base, i): -rng.uniform(.01, 100) for i in range(500) if rng.random() < .1},
            {(base, i): -100. for i in range(50, 100)}, {(base, 10**9): -1., (base, 60): -4.}]
        actions = []
        for i in range(120):
            station = ep.STATIONS[i % len(ep.STATIONS)]
            arrival = rng.choice([0., 59., 60., 120.-5e-10, 60.+5e-10, rng.uniform(1, 300)])
            duration = rng.choice([2e-9, .25, 1., 15.5, 60.-5e-10, 60., rng.uniform(.2, 90)])
            end = arrival + duration + rng.choice([0., .3, 7., 65.])
            energy = duration * net._charge_power(station) / 60.
            cost, start, finish = min(ep._charge_window_options(station, arrival, end, energy,
                event_times=net.events, station_prices=net.prices, charge_kw=net._charge_power(station)))
            actions.append((ep.CHARGE_START_COST + cost, dict(kind='charge', station=station,
                arrival_min=arrival, deadline_min=end, kwh=energy, cst=start, cet=finish, metadata=i)))
        for grid in (1, 5):
            for sites in ({base}, set(), None):
                for duals in vectors:
                    selector = CapacityWindowSelector(net, duals, sites, grid)
                    for cost, action in actions:
                        old = net._capacity_adjusted_arc(cost, action, duals, sites, grid)
                        new = selector.adjust(cost, action, duals, sites, grid)
                        self.assertEqual(old, new)
                        self.assertEqual(new, selector.adjust(cost, action, duals, sites, grid))

    def test_whole_route_replay_and_changed_inputs(self):
        base = base_station_name(ep.STATIONS[0])
        for power in (60., 240.):
            for flat in (False, True):
                net = ep.EventExpandedNetwork(fixture(), prices(flat), soc_step=15, block_min=5,
                    g_kwh=240., charge_kw=240., reserve_kwh=0., arc_mode='explicit',
                    station_charge_kw={base: power})
                duals = {(base, 60): -123.4567}
                for changed in (False, True):
                    if changed:
                        duals.clear()
                        duals.update({(base, i): -100. for i in range(50, 100)})
                    net.capacity_selector = 'reference'
                    reference = net.min_reduced_cost_route({0: 100000., 1: 100000.}, capacity_duals=duals)
                    net.capacity_selector = 'prefix-memo'
                    accelerated = net.min_reduced_cost_route({0: 100000., 1: 100000.}, capacity_duals=duals)
                    self.assertEqual(reference, accelerated)
                    occupied = set()
                    node = net.SINK
                    while node:
                        node, action = accelerated['_parent'][node]
                        occupied.update(ep.conservative_capacity_rows(action))
                    rec = accelerated['_event_record']
                    replay = rec['cost'] - 100000. * len(rec['trips']) - sum(duals.get(r, 0.) for r in occupied)
                    self.assertAlmostEqual(replay, accelerated['rc'], places=8)

    def test_configuration_and_in_place_dual_invalidation(self):
        net = self.blank()
        base = base_station_name(ep.STATIONS[0])
        duals = {(base, 60): -1.}
        for change in ('initial', 'power', 'tariff', 'events', 'dual'):
            if change == 'power':
                net.station_charge_kw = {base: 120.}
            elif change == 'tariff':
                net.prices[base][1] = .031
            elif change == 'events':
                net.events[ep.STATIONS[0]] = tuple(range(0, 1621, 60))
            elif change == 'dual':
                duals[(base, 60)] = -314.159
            action = dict(kind='charge', station=ep.STATIONS[0], arrival_min=59.25,
                          deadline_min=155., kwh=15.125, cst=60., cet=76.)
            selector = CapacityWindowSelector(net, duals, None, 1)
            self.assertEqual(net._capacity_adjusted_arc(0., action, duals, None, 1),
                             selector.adjust(0., action, duals, None, 1))

    def test_infeasible_window_preserves_reference_rejection(self):
        net = self.blank()
        base = base_station_name(ep.STATIONS[0])
        duals = {(base, 60): -1.}
        action = dict(kind='charge', station=ep.STATIONS[0], arrival_min=10.,
                      deadline_min=11. - 1.5 * ep.TOL, kwh=1., cst=10., cet=11.)
        selector = CapacityWindowSelector(net, duals, None, 1)
        for adjust in (net._capacity_adjusted_arc, selector.adjust):
            with self.assertRaisesRegex(RuntimeError, 'no feasible window'):
                adjust(0., action, duals, None, 1)

    def test_deadlines_and_bounded_storage(self):
        net = self.blank()
        base = base_station_name(ep.STATIONS[0])
        duals = {(base, 60): -1.}
        with self.assertRaises(ep.PricingDeadlineExceeded):
            CapacityWindowSelector(net, duals, None, 1, deadline=0., clock=lambda: 1.)
        selector = CapacityWindowSelector(net, duals, None, 1)
        selector.MAX_OPTIONS = 0
        selector.MAX_MEMO = 1
        for i in range(3):
            action = dict(kind='charge', station=ep.STATIONS[0], arrival_min=10.+i,
                          deadline_min=100., kwh=15., cst=10.+i, cet=25.+i)
            selector.adjust(0., action, duals, None, 1)
            with self.assertRaises(ep.PricingDeadlineExceeded):
                selector.adjust(0., action, duals, None, 1, deadline=0., clock=lambda: 1.)
        self.assertEqual(len(selector._capacity_memo), 1)
        self.assertEqual(len(selector._capacity_options), 0)

if __name__ == '__main__':
    unittest.main()

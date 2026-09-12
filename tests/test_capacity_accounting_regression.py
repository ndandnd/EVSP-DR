"""Independent master-row replay for station-specific event tariff accounting."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import event_pricer_network as ep
from expanded_path_realization import realized_costs, validate_continuous_charging_blocks
from utils_v2 import base_station_name


def fixture():
    s = ep.STATIONS[0]
    return SimpleNamespace(trips=(0, 1), start_min={0: 0., 1: 180.},
        end_min={0: 10., 1: 190.}, trip_energy={0: 190., 1: 100.},
        adjacency={ep.DEPOT: [(0, 0., 0., 'depot_trip')],
        0: [(s, 0., 0., 'trip_station'), (ep.DEPOT, 0., 0., 'trip_depot')],
        s: [(1, 0., 0., 'station_trip'), (ep.DEPOT, 0., 0., 'station_depot')],
        1: [(ep.DEPOT, 0., 0., 'trip_depot')]})


def prices(flat=False):
    return {base_station_name(s): {h: .1 if flat else (.11, .17, .29)[h % 3]
            for h in range(27)} for s in ep.STATIONS}


class CapacityAccountingTests(unittest.TestCase):
    def test_independent_whole_route_reduced_cost(self):
        station = ep.STATIONS[0]
        base = base_station_name(station)
        for power in (60., 240.):
            for flat in (False, True):
                for rows in ({(base, 60): -123.4567},
                             {(base, i): -100. for i in range(50, 100)}):
                    with self.subTest(power=power, flat=flat, rows=len(rows)):
                        net = ep.EventExpandedNetwork(fixture(), prices(flat), soc_step=15,
                            block_min=5, g_kwh=240., charge_kw=240., reserve_kwh=0.,
                            arc_mode='explicit', station_charge_kw={station: power})
                        route = net.min_reduced_cost_route({0: 100000., 1: 100000.},
                            capacity_duals=rows, capacity_sites={base})
                        record = route['_event_record']
                        occupied = set()
                        node = net.SINK
                        while node:
                            node, action = route['_parent'][node]
                            occupied.update(ep.conservative_capacity_rows(action, sites={base}))
                        replay = record['cost'] - 100000. * len(record['trips']) - sum(rows.get(r, 0.) for r in occupied)
                        self.assertAlmostEqual(route['rc'], replay, places=8)
                        if power == 60. and not flat:
                            expected = 59 * .17 + .29 if len(rows) == 1 else 20 * .17 + 40 * .29
                            self.assertAlmostEqual(sum(b['expanded_grid_kwh'] * b['price_per_kwh']
                                for b in record['continuous_realized_charging_blocks']), expected)

    def test_fractional_alias_multiple_stations_and_grid_vs_realized(self):
        record = {'cost': 0., 'charging_stops': {'stations': ['4808_0', '4809_0'],
            'cst': [59.5, 120.25], 'cet': [61.5, 121.25], 'kwh': [1., 1.]},
            'expanded_grid_charging_stops': {'stations': ['4808_0', '4809_0'],
            'cst': [59.5, 120.25], 'cet': [61.5, 121.25], 'kwh': [2., 2.]}}
        mapping = {'time_model': 'event', 'charge_kw': 240.,
                   'station_charge_kw': {'4808': 60., '4809': 30., '4809_0': 120.}}
        tariff = {'4808': {0: .1, 1: .3}, '4809': {2: .7}}
        result = realized_costs(record, mapping, station_prices=tariff)
        blocks = result['continuous_realized_charging_blocks']
        self.assertEqual([(b['start_min'], b['end_min']) for b in blocks],
                         [(59.5, 60.), (60., 61.5), (120.25, 121.25)])
        self.assertEqual([b['realized_kwh'] for b in blocks], [.5, .5, 1.])
        self.assertEqual([b['expanded_grid_kwh'] for b in blocks], [.5, 1.5, 2.])
        self.assertAlmostEqual(result['expanded_minus_realized_cost'], 1.)
        with self.assertRaisesRegex(ValueError, 'power'):
            validate_continuous_charging_blocks(record, blocks, station_prices=tariff,
                charge_kw=240., station_charge_kw={'4808': 30., '4809': 120.})

if __name__ == '__main__':
    unittest.main()

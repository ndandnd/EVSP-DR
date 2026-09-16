import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_terminal_energy_cg import audit_energy, run
from terminal_energy_pricer import TerminalEnergyNetwork
from event_pricer_network import EventExpandedNetwork
from test_event_pricer_network import two_trip_problem, prices
from config import BUS_COST_KX


def network(mode='lazy'):
    return TerminalEnergyNetwork(two_trip_problem(), prices(), soc_step=30,
                                 block_min=30, g_kwh=240, charge_kw=240,
                                 reserve_kwh=0, arc_mode=mode)


def enumerate_routes(net):
    records = []
    def visit(node, actions):
        for cost, energy, terminal in net.terminal_options.get(node, []):
            record = net._record(actions + [terminal])
            record['terminal_energy_kwh'] = energy
            records.append(audit_energy(net, record))
        for target, cost in net._iter_arcs(node):
            if target != net.SINK:
                visit(target, actions + [net._edge_action(node, target)])
    visit(0, [])
    return records


class TerminalFullCGTests(unittest.TestCase):
    def test_exact_pricing_against_all_paths_and_physical_energy(self):
        for mode in ['lazy', 'explicit']:
            net = network(mode)
            routes = enumerate_routes(net)
            for objective in ['combined-cost', 'charging-cost', 'fleet-only', 'artificial-elimination']:
                for beta in [0, .1, 2, 1000]:
                    alpha = {0: 100001, 1: 99999}
                    gamma = -5
                    def rc(r):
                        cost = (r['cost'] if objective == 'combined-cost' else
                                r['cost']-BUS_COST_KX if objective == 'charging-cost' else
                                1 if objective == 'fleet-only' else 0)
                        return cost-sum(alpha[t] for t in r['trips'])-beta*r['terminal_energy_kwh']-gamma
                    got = net.terminal_batch(alpha, terminal_dual=beta,
                                             route_dual=gamma, objective=objective)
                    self.assertAlmostEqual(got[0]['rc'], min(map(rc, routes)), places=6)
                    audit_energy(net, got[0]['record'])
            self.assertTrue(any(len(options)>1 for options in net.terminal_options.values()))
            old = net.min_reduced_cost_route({0: 100000, 1: 100000})
            new = net.terminal_batch({0: 100000, 1: 100000})[0]
            self.assertAlmostEqual(old['rc'], new['rc'])

    def test_full_lp_and_integer_solution_against_enumeration(self):
        net = network()
        routes = enumerate_routes(net)
        matrix = np.array([[int(t in r['trips']) for r in routes] for t in net.problem.trips]
                          + [[r['terminal_energy_kwh'] for r in routes]])
        full = linprog([r['cost'] for r in routes],
                       A_ub=np.vstack([-matrix, np.ones(len(routes))]),
                       b_ub=[-1, -1, -100, 1], bounds=(0, None), method='highs')
        self.assertTrue(full.success)
        with tempfile.TemporaryDirectory() as temp:
            result = run(net, Path(temp)/'run', target=100, fleet_cap=1,
                         cg_seconds=60, mip_seconds=10)
        self.assertTrue(result['cg_pricing_certified'])
        self.assertAlmostEqual(result['last_iteration']['rmp_objective'], full.fun, places=5)
        self.assertEqual(result['fleet_stage']['buses'], 1)
        self.assertGreaterEqual(result['selected_terminal_kwh'], 100-1e-6)
        feasible = [r for r in routes if set(r['trips']) == {0, 1} and r['terminal_energy_kwh'] >= 100-1e-6]
        self.assertAlmostEqual(result['charging_stage']['cost'], min(r['cost']-BUS_COST_KX for r in feasible), places=5)


if __name__ == '__main__':
    unittest.main()

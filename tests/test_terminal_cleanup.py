import unittest
from test_terminal_full_cg import network,enumerate_routes
from terminal_duplicate_cleanup import frontier

class CleanupTests(unittest.TestCase):
    def test_fixed_sequence_energy_frontier_against_enumeration(self):
        net=network();routes=enumerate_routes(net)
        for sequence in [(0,),(0,1)]:
            expected=[r for r in routes if tuple(r['trips'])==sequence]
            actual=frontier(net,sequence)
            for beta in [0,.1,1,100]:
                score=lambda r:r['cost']-beta*r['terminal_energy_kwh']
                self.assertAlmostEqual(min(map(score,expected)),min(map(score,actual)))

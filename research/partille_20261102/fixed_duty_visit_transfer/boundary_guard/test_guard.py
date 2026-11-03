"""Counterexamples for depot endpoints, first-visit opening, and native visit assignment."""
import unittest
from types import SimpleNamespace
from check_guard import check_route

class WindowBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.p=SimpleNamespace(DEPOT='PARX_0',start_min={0:150},end_min={0:180})
        self.nodes=['PARX_0','JON_A_0',0,'PARX_0']; self.arcs={('PARX_0','JON_A_0'):(5,0),('JON_A_0',0):(3,0),(0,'PARX_0'):(5,0)}
    def run_case(self,start,end,node=None):
        nodes=list(self.nodes);nodes[0]=node or self.p.DEPOT
        return check_route(self.p,nodes,[0],[{'station':'JON_A','start':start,'end':end,'kwh':1}],self.arcs,1560,self.p.DEPOT)
    def test_legal(self): self.assertEqual(self.run_case(95,147),(1,1,1))
    def test_before_first_visit_opens(self):
        with self.assertRaisesRegex(ValueError,'matching native windows'):self.run_case(94,100)
    def test_after_first_visit_closes(self):
        with self.assertRaisesRegex(ValueError,'matching native windows'):self.run_case(100,148)
    def test_charger_alias_as_origin(self):
        with self.assertRaisesRegex(ValueError,'literal depot'):self.run_case(95,100,'PARX_1')
    def test_wrong_trip_map(self):
        with self.assertRaisesRegex(ValueError,'trip list'):check_route(self.p,self.nodes,[2],[],self.arcs,1560,self.p.DEPOT)

if __name__=='__main__':unittest.main()

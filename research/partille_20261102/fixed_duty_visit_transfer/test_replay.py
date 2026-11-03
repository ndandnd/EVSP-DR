from types import SimpleNamespace
import unittest
from check_transfers import replay

def fixture(two=False):
    n = 2 if two else 1
    arc = {"PARX_0":[(j,0,0,None) for j in range(n)], "JON_A_0":[("PARX_0",1,1,None)]}
    for j in range(n):
        arc[j]=[("JON_A_0",1,1,None)]
    problem=SimpleNamespace(adjacency=arc,trips=list(range(n)),start_min={j:15 for j in range(n)},
                   end_min={j:30 for j in range(n)},trip_energy={j:20 for j in range(n)})
    physics={"initial_kwh":100,"battery_kwh":100,"reserve_kwh":0,"charge_kw":60,
             "station_kw":{"PARX":60.},"charge_start_fee":0}
    routes=[{"duty_id":str(j),"route_nodes":["PARX_0",j,"JON_A_0","PARX_0"],"trips":[j]} for j in range(n)]
    sched=[{"route":j,"station":"JON_A","start":56,"end":66,"kwh":10} for j in range(n)]
    return [problem,{"JON_A":{0:1,1:3}},routes,sched,physics,88*n,100]

class TestTransfer(unittest.TestCase):
    def test_cross_hour_energy(self):
        result=replay(*fixture())
        self.assertEqual(result["feasible_cost_upper_bound"],22.)
        self.assertEqual(result["terminal_kwh"],88.)
    def test_overload(self):
        with self.assertRaisesRegex(ValueError,"plug overload"):
            replay(*fixture(two=True))
    def test_overpower(self):
        args=fixture(); args[3][0]["kwh"]=11
        with self.assertRaisesRegex(ValueError,"station power"):
            replay(*args)
    def test_terminal(self):
        args=fixture(); args[5]=89
        with self.assertRaisesRegex(ValueError,"terminal floor"):
            replay(*args)
    def test_exact_cover(self):
        args=fixture(); args[0].trips.append(1)
        with self.assertRaisesRegex(ValueError,"exactly-once"):
            replay(*args)

if __name__=="__main__":
    unittest.main()

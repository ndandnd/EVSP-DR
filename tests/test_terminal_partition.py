import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from validate_terminal_pool_partition import solve

class PartitionTests(unittest.TestCase):
    def test_overlap_is_not_an_executable_partition(self):
        routes=[{'trips':[0,1],'cost':100001,'terminal_energy_kwh':10},
                {'trips':[1,2],'cost':100001,'terminal_energy_kwh':10},
                {'trips':[2],'cost':100010,'terminal_energy_kwh':10}]
        with tempfile.TemporaryDirectory() as d:
            result,selected=solve(routes,[0,1,2],target=20,cap=2,seconds=10,log_dir=d)
        self.assertEqual(result['fleet'],2)
        self.assertEqual(result['charging_cost'],11)
        self.assertTrue(result['exact_once_verified'])
    def test_terminal_energy_cannot_be_ignored(self):
        routes=[{'trips':[0],'cost':100001,'terminal_energy_kwh':0}]
        with tempfile.TemporaryDirectory() as d:
            result,selected=solve(routes,[0],target=10,cap=1,seconds=10,log_dir=d)
        self.assertFalse(selected)
        self.assertIsNone(result['fleet'])

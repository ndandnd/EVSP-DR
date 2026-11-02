import unittest
from pathlib import Path
from campaign import cells, read, HERE, solver_argv
from gate import valid_at_floor


class Protocol(unittest.TestCase):
    def test_equal_search_budget_with_fresh_inputs(self):
        rows=read(HERE/'design.json')['cases']
        for case,row in rows.items():
            args=[solver_argv(row,a,Path('solver'),Path('data'),Path('price'),Path('out')) for a in ('repair','simple')]
            for argv in args:
                val=lambda k: int(argv[argv.index(k)+1]) if k in argv else 0
                self.assertEqual(val('--fleet-cap'),row['time_only_floor']+4)
                self.assertEqual(val('--cg-seconds')+val('--mip-seconds')+val('--dive-seconds')+val('--integer-pricing-seconds'),19800)
                self.assertNotIn('--seed-routes',argv)
                self.assertIn('PARX=60',argv)
        self.assertEqual(len(cells(8)) + len(cells(8,True)) + 1,12)
        self.assertEqual(len(set(cells(10))),12)

    def test_gate_does_not_promote_a_cover_or_wrong_instance(self):
        case=read(HERE/'design.json')['cases']['c2_k08']
        a=dict(schema='partille-selection-replay-v1',arm='blind',instance_sha256=case['instance_sha256'],
               validated=True, pricing_executed=True, exact_once=True, cover_valid=True, duplicate_service=0,grid_plug_overload_minutes=0,
               shared_fleet_cap=12, fleet=8, trip_count=case['n_trips'], objective_nonfleet=330)
        self.assertTrue(valid_at_floor(a,case,'blind'))
        for change in (dict(duplicate_service=1),dict(pricing_executed=False),dict(validated=False),dict(fleet=9),dict(grid_plug_overload_minutes=1),
                       dict(instance_sha256='different'),dict(arm='augmented'),dict(objective_nonfleet=float('nan'))):
            self.assertFalse(valid_at_floor(dict(a,**change),case,'blind'))


if __name__=='__main__':
    unittest.main()

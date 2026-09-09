import hashlib,json,sys,unittest
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.event_uniform_envelope import build_nested_replication_inputs as g
class ReplicationInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=g.read(g.DEFAULT_OUTPUT/'selection_manifest.csv');cls.order_rows=g.read(g.DEFAULT_OUTPUT/'chain_order.csv')
        cls.plan=json.loads((g.DEFAULT_OUTPUT/'input_plan.json').read_text())
        cls.certificates=g.read(g.SOURCE/'known_duty_continuous_240_240.csv')
    def test_exact_grid_counts_and_input_hashes(self):
        self.assertEqual(Counter(int(r['scale']) for r in self.rows),{3:14,5:14,6:14})
        self.assertEqual({(int(r['scale']),int(r['family_replicate'])) for r in self.rows},{(k,p) for k in (3,5,6) for p in range(7,21)})
        for r in self.rows:self.assertEqual(g.base.sha256(ROOT/r['relative_path']),r['instance_file_sha256'])
        self.assertEqual(len({r['instance_file_sha256'] for r in self.rows}),42)
    def test_all_full_orders_and_base_ids(self):
        self.assertEqual(len(self.order_rows),300)
        eligible={r['duty_id'] for r in self.certificates}
        for p in range(1,21):
            rows=[r for r in self.order_rows if int(r['family_replicate'])==p]
            self.assertEqual([int(r['addition_rank']) for r in rows],list(range(1,16)))
            self.assertEqual(len({g.base._base_task(r['duty_id']) for r in rows}),15)
            self.assertTrue({r['duty_id'] for r in rows}<=eligible)
    def test_rng_replays_original_and_extended_orders(self):
        excluded=set().union(*(g.base.existing_duty_sets(g.SOURCE/n) for n in ['excluded_existing_scale_ladder_manifest.csv','excluded_small_threshold_manifest.csv']))
        first,attempts=g.draw_orders(sorted(r['duty_id'] for r in self.certificates),excluded)
        second,_=g.draw_orders(sorted(r['duty_id'] for r in self.certificates),excluded)
        self.assertEqual(first,second)
        old=g.read(g.ORIGINAL/'chain_order.csv')
        for p in range(1,21):
            self.assertEqual(first[p-1],[r['duty_id'] for r in self.order_rows if int(r['family_replicate'])==p])
            if p<=6:self.assertEqual(first[p-1],[r['duty_id'] for r in old if int(r['family_replicate'])==p])
    def test_nested_duty_and_trip_prefixes(self):
        for p in range(7,21):
            rows={int(r['scale']):r for r in self.rows if int(r['family_replicate'])==p}
            duty={k:set(json.loads(r['duties_json'])) for k,r in rows.items()}
            trip={k:{r['Ordered_Trip_ID'] for r in g.read(ROOT/row['relative_path'])} for k,row in rows.items()}
            self.assertTrue(duty[3]<duty[5]<duty[6]);self.assertTrue(trip[3]<trip[5]<trip[6])
    def test_no_existing_instance_byte_duplicates(self):
        new={r['instance_file_sha256'] for r in self.rows}
        old={g.base.sha256(p) for p in (ROOT/'data/scale_ladder/instances').rglob('Practice*.csv') if not p.is_relative_to(g.DEFAULT_OUTPUT)}
        self.assertFalse(new&old)
        self.assertEqual(self.plan['lower_prefix_duplicate_audit'],[])
        self.assertFalse(self.plan['selection_uses_solver_outcomes'])
if __name__=='__main__':unittest.main()

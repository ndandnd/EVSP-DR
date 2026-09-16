import csv,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from inherit_capacity_pool import sha,inherit_pool
class InheritanceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.p=self.root/'p.csv';self.c=self.root/'c.csv';self.pool=self.root/'pool.jsonl';self.status=self.root/'status.json'
  self.rows=[{'Ordered_Trip_ID':10,'From1':'A','To1':'B','Start1':'05:00','End1':'05:10','Usage kWh':5},{'Ordered_Trip_ID':20,'From1':'B','To1':'C','Start1':'06:00','End1':'06:10','Usage kWh':7}]
  self.write(self.p,self.rows);self.write(self.c,[dict(self.rows[0],Ordered_Trip_ID=5)]+self.rows)
  r={'trips':[0,1],'route_nodes':['PARX_0',0,1,'PARX_0'],'cost':100000,'found_iter':8000,'cg_checkpoint_id':'parent'}
  self.pool.write_text(json.dumps(r)+'\n');self.prov={k:'same' for k in ['git_commit','prices_sha256','reference_sha256','deadhead_sha256']}
  self.doc={'pool_sha256':sha(self.pool),'provenance':dict(self.prov,instance_sha256=sha(self.p)),'physics':{'battery_kwh':236.44},'checkpoint':{'id':'parent'}};self.status.write_text(json.dumps(self.doc))
 def tearDown(self):self.tmp.cleanup()
 def write(self,p,rows):
  with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 def load(self,**kw):
  args=dict(expected_physics={'battery_kwh':236.44},child_provenance=self.prov,new_checkpoint_id='child',route_validator=lambda r:None);args.update(kw)
  return inherit_pool(self.status,self.pool,self.p,self.c,**args)
 def test_remaps_and_resets_iteration_without_mutating_source(self):
  rr,meta=self.load();self.assertEqual(rr[0]['trips'],[1,2]);self.assertEqual(rr[0]['route_nodes'],['PARX_0',1,2,'PARX_0']);self.assertEqual(rr[0]['found_iter'],0);self.assertEqual(rr[0]['inheritance']['parent_found_iter'],8000);self.assertTrue(meta['every_inherited_route_replayed']);self.assertEqual(json.loads(self.pool.read_text())['trips'],[0,1])
 def test_rejects_physics_change(self):
  with self.assertRaisesRegex(ValueError,'physics'):self.load(expected_physics={'battery_kwh':240})
 def test_rejects_changed_trip(self):
  self.write(self.c,[dict(self.rows[0],**{'Usage kWh':6}),self.rows[1]])
  with self.assertRaisesRegex(ValueError,'changed trip'):self.load()
 def test_rejects_corrupt_pool(self):
  self.pool.write_text(self.pool.read_text()+' ')
  with self.assertRaisesRegex(ValueError,'hash mismatch'):self.load()
 def test_rejects_replay_failure(self):
  with self.assertRaisesRegex(ValueError,'physical replay'):self.load(route_validator=lambda r:'lowSOC')
 def test_rejects_tariff_change(self):
  with self.assertRaisesRegex(ValueError,'provenance'):self.load(child_provenance=dict(self.prov,prices_sha256='different'))
if __name__=='__main__':unittest.main()

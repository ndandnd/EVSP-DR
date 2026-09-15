"""Scientific identity and native no-augmentation gate regressions."""
import copy,json,sqlite3,tempfile,unittest
from pathlib import Path
import pool_logic as p
from worker import result_gate
class DesignTests(unittest.TestCase):
 def test_native_dedup_preserves_cost_and_first_tie(self):
  with tempfile.TemporaryDirectory() as d:
   rows=[{'trips':[2,1],'route_nodes':['D',2,1,'D'],'charging_stops':{},'cost':200}, {'trips':[1,2],'route_nodes':['D',1,2,'D'],'charging_stops':{},'cost':200}, {'trips':[1,2],'route_nodes':['D',1,2,'D'],'charging_stops':{},'cost':199}]
   path=Path(d)/'p';path.write_text(''.join(json.dumps(r)+'\n' for r in rows));db=sqlite3.connect(':memory:')
   self.assertEqual(p.stream_source(db,'s',path,[1,2]),3);result=json.loads(db.execute('SELECT payload FROM s').fetchone()[0]);self.assertEqual(result,rows[2]);self.assertEqual(p.table_hashes(db,'s')['native_unique_columns'],1)
 def test_unknown_trip_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'p';path.write_text(json.dumps({'trips':[3],'cost':10})+'\n')
   with self.assertRaisesRegex(ValueError,'invalid column'):p.stream_source(sqlite3.connect(':memory:'),'s',path,[1,2])
 def test_physics_and_producer_must_match_within_pair(self):
  a={**p.BASELINE,'csv':'a.csv','trip_ids':[1],'provenance':{k:'a'*64 for k in p.HASHES}};a['provenance']['git_commit']=next(iter(p.PRODUCERS));b=copy.deepcopy(a);p.identities([a,b]);b['charge_kw']=60
  with self.assertRaises(ValueError):p.identities([a,b])
  b=copy.deepcopy(a);b['provenance']['git_commit']=next(x for x in p.PRODUCERS if x!=a['provenance']['git_commit'])
  with self.assertRaises(ValueError):p.identities([a,b])
 def test_donor_source_exact_membership_and_union_dominance(self):
  db=sqlite3.connect(':memory:');r={'trips':[1,2],'route_nodes':['D',1,2,'D'],'charging_stops':{},'cost':200};key=p.incidence(r);rh=p.route_hash(r)
  for table,cost in [('own',200),('u',190)]:db.execute(f'CREATE TABLE {table}(key TEXT,cost REAL,rh TEXT)');db.execute(f'INSERT INTO {table} VALUES(?,?,?)',(key,cost,rh))
  m={'buses':1,'selected_routes':[{**r,'master_cost_semantics':'expanded_grid_cost','expanded_grid_cost':200,'physical_realization':{'recorded_route_sha256':rh}}]};s={'arm':'core','mip_sha256':'abc'}
  result=p.donor_witness(db,'own','u',s,m,[1,2]);self.assertEqual(result['union_incidence_substitution_objective'],190)
  db.execute('UPDATE u SET cost=201')
  with self.assertRaisesRegex(ValueError,'lost or made more expensive'):p.donor_witness(db,'own','u',s,m,[1,2])
 def test_native_gate_refuses_augmented_seed_columns(self):
  v={'physical_replay_validated':True,'physical_pool_audit':{'rejected_columns':0,'deterministically_repaired':0,'added_giro_route_count':16,'base_pool_column_count':100,'post_augmentation_columns':116,'base_pool_ordered_sha256':'a','augmented_pool_ordered_sha256':'b'},'mip_provenance':{'arguments':{}},'source_result_sha256':'s','source_journal_sha256':'j'}
  with self.assertRaisesRegex(ValueError,'unexpected native pool augmentation'):result_gate(v,{},dict(result_sha256='s',journal_sha256='j'),{})
if __name__=='__main__':unittest.main()

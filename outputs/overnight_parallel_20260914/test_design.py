import copy,importlib.util,json,unittest
from pathlib import Path
import seed_logic
B=Path(__file__).resolve().parent
class Design(unittest.TestCase):
 def test_count_ties_and_certificate_scope(self):
  p={'csv':'parent.csv','trip_ids':[1,2,3],'provenance':{'instance_sha256':'a'*64},'final_lp':{'artificial_total':0,'positive_routes':[{'trips':[2],'cost':4,'value':.5},{'trips':[1],'cost':5,'value':.5},{'trips':[3],'cost':6,'value':.2}]}}
  a=seed_logic.choose(p,{'physical_replay_validated':True,'buses':1,'selected_routes':[{'trips':[3],'cost':6}]})
  self.assertEqual(a['lpweight'][0]['trips'],[1]);s=seed_logic.status(p,'/journal','lpweight',{'selected_sequence_count':1})
  self.assertFalse(s['optimization_run']);self.assertFalse(s['certified_rc_optimal']);self.assertNotIn('final_lp',s)
  with self.assertRaises(AssertionError):seed_logic.choose(p,{'physical_replay_validated':False})
 def test_pair_identity_and_independence(self):
  m=json.loads((B/'manifest.json').read_text());self.assertEqual(len(m['cases']),72)
  for pair in m['pairs']:
   a=m['cases'][pair['pair_id']+'_integer'];b=m['cases'][pair['pair_id']+'_lpweight']
   for key in ('input_sha256','execution_commit','solver_budget_s','cache_manifest_sha256','resources'):self.assertEqual(a[key],b[key])
   self.assertNotIn('source_case',a)
   for c in (a,b):
    self.assertEqual(m['cases'][c['id']+'_mip']['source_case'],c['id'])
    self.assertEqual(c['seed']['parent_k'],c['target_k']-1)
    self.assertNotEqual(a['seed_status_sha256'],b['seed_status_sha256'])
 def test_normalizer_case_hash_and_publication_gates(self):
  path=B.parent/'research_register/build_register.py';spec=importlib.util.spec_from_file_location('register',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  m=json.loads((B/'manifest.json').read_text());cid='c1_k08_integer';c=m['cases'][cid];name='overnight_parallel_20260914';root='/home/nc437/ladder-lite/'+name
  snap={'timestamp_utc':'2026-09-14T00:00:00Z','campaigns':{name:{'workflow':{'manifest.json':m,'case_jobs.json':{cid:'123'}}}}}
  def register():return mod.Register(snap,B/'fixture.json',B,[])
  p={'csv':c['csv'],'provenance':{'instance_sha256':c['input_sha256'],'git_commit':c['execution_commit']},'completion_marker_matches':True,'certified_rc_optimal':False,'final':{'iter':1,'artificials':0},'inherited_event_pool_audit':{'source_status_sha256':c['seed_status_sha256'],'source_journal_sha256':c['seed_journal_sha256'],'accepted_columns':7,'added_to_child_pool':7}}
  def add(p):return register().add(name,root,'diagnostic','cg',p,source_path=root+'/cases/'+cid+'/attempts/123_r0/cg.json',source_sha256='b'*64)
  row=add(p);self.assertEqual(row['case_id'],cid);self.assertEqual(row['arm'],'previous_k_integer');self.assertEqual(row['job_ids'],['123']);self.assertFalse(row['full_model_lp_certified']);self.assertEqual(row['details']['inherited_event_pool_audit']['source_status_sha256'],c['seed_status_sha256'])
  for bad in ('input','commit','publication','case'):
   q=copy.deepcopy(p)
   if bad=='input':q['provenance']['instance_sha256']='0'*64
   elif bad=='commit':q['provenance']['git_commit']='0'*40
   elif bad=='publication':q['completion_marker_matches']=False
   else:q['csv']='other.csv'
   with self.assertRaises(ValueError):add(q)
if __name__=='__main__':unittest.main()

import importlib.util,json,pathlib,tempfile,unittest,copy
spec=importlib.util.spec_from_file_location('logic',pathlib.Path(__file__).with_name('union_logic.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class UnionTests(unittest.TestCase):
 def test_streaming_native_ties_preserve_full_witness_and_order(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);a={'trips':[2,1],'cost':10,'witness':'first'};b={'trips':[1,2],'cost':9,'witness':'cheaper'};tie={**b,'witness':'tie'};last={'trips':[3],'cost':4}
   paths=[]
   for i,rows in enumerate([[a,last],[b],[tie]]):
    q=p/str(i);q.write_text(''.join(json.dumps(r)+'\n' for r in rows));paths.append(q)
   n,counts=m.merge(paths,[1,2,3],p/'db',p/'out')
   self.assertEqual((n,counts),(2,[2,1,1]));self.assertEqual([json.loads(x) for x in (p/'out').read_text().splitlines()],[b,last])
 def test_constructor_no_fabricated_certificate_and_bound_source_evidence(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);sources=[]
   for i in range(3):
    j=p/f'journal{i}';j.write_text(json.dumps({'trips':[1],'cost':100000,'marker':i})+'\n')
    status={**m.BASELINE,'csv':'input.csv','trip_ids':[1], 'columns_journal':str(j),
      'provenance':{k:('a'*64) for k in m.HASHES},'final_lp':{'objective':100000},'certified_rc_optimal':True}
    status['provenance']['git_commit']=m.PRODUCER
    s=p/f'status{i}';s.write_text(json.dumps(status));e=p/f'mip{i}'
    e.write_text(json.dumps({'source_result_sha256':m.sha(s),'source_journal_sha256':m.sha(j),'physical_replay_validated':True,
      'physical_pool_audit':{'rejected_columns':0,'deterministically_repaired':0,'input_hashes':{'instance_sha256':'a'*64}}}))
    sources.append(dict(status_path=str(s),status_sha256=m.sha(s),journal_path=str(j),journal_sha256=m.sha(j),mip_evidence_path=str(e),mip_evidence_sha256=m.sha(e)))
   c={'sources':sources,'static_hashes':{},'input_sha256':'a'*64,'baseline_scope':{'shared_station_capacity':False,'terminal_soc_floor':False,'bus_coefficient':100000,'charge_start_fee':5,'configuration_bound_by':m.PRODUCER}}
   out=p/'out.json';m.construct(c,out);v=m.read(out)
   self.assertFalse(v['certified_rc_optimal']);self.assertFalse(v['optimization_run']);self.assertEqual(v['final']['iter'],0)
   self.assertNotIn('final_lp',v);self.assertNotIn('pricing_certificate_scope',v['provenance'])
   evidence=pathlib.Path(sources[0]['mip_evidence_path']);e=m.read(evidence);e['physical_pool_audit']['rejected_columns']=1;evidence.write_text(json.dumps(e))
   sources[0]['mip_evidence_sha256']=m.sha(evidence)
   with self.assertRaises(AssertionError):m.construct(c,p/'bad.json')
 def test_model_identity_rejects_capacity_and_objective_producer_change(self):
  v={**m.BASELINE,'trip_ids':[1],'csv':'a','provenance':{k:'bound' for k in m.HASHES}};v['provenance']['git_commit']=m.PRODUCER
  m.identities([v,v])
  for extra in [{'capacity_enforced':True},{'charge_kw':60},{'soc_step':5}]:
   with self.assertRaises(ValueError):m.identities([v,{**v,**extra}])
  bad=copy.deepcopy(v);bad['provenance']['git_commit']='unreviewed'
  with self.assertRaises(ValueError):m.identities([bad,bad])

if __name__=='__main__':unittest.main()

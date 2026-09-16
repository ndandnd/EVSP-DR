import copy,unittest
from four_numbers import evaluate,FRESH,LONG,VARIANCE,F6

def fixture():
 rows=[];checks=[];registry=[]
 for n,cid in enumerate(sorted(FRESH|LONG|VARIANCE),100):
  seed=int(cid.split('seed')[1].split('_')[0]);chain=int(cid.split('_')[1][1:]);fleet=15 if cid in FRESH else 31 if cid in LONG else 32+seed
  rows.append(dict(case_id=cid,seed=seed,chain=chain,buses=fleet,physical_route_replay=True,duplicate_cleanup=False,result_path='/'+cid,result_sha256=cid,seed_observed=seed,ordered_pool_sha256=str(chain),fleet_proven_in_pool=False,pool_fleet_bound=fleet-1))
  checks.append(dict(case_id=cid,result_sha256=cid,tests={'pool':True,'physical':True},observed_stage1_limit_s=43200 if cid in LONG else 10800,observed_stage2_reserved_s=1800))
  registry.append(dict(campaign='p1',case_id=cid,job_id=str(n)))
 fr=[]
 for n,cid in enumerate(sorted(F6),200):
  peak,arm=cid.split('_');fr.append(dict(case_id=cid,peak=peak,arm=arm,buses=5,continuous_charging_cost=100 if arm=='cg' else 110,ending_kwh=281,matched_five_bus_comparison_eligible=True,result_path='/'+cid,result_sha256=cid));registry.append(dict(campaign='f6_k5',case_id=cid,job_id=str(n)))
 for peak in ['peak08','peak12','peak18']:fr.append(dict(peak=peak,arm='original_GIRO',cost_lower=120,cost_upper=125))
 return dict(campaigns={'p1':dict(collection_ok=True,data=dict(manifest_sha256='p',rows=rows,endpoint_audit=dict(manifest_sha256='p',checks=checks))),'f6_k5':dict(collection_ok=True,data=dict(manifest_sha256='f',rows=fr))},registered_jobs=registry,scheduler={'squeue':dict(ok=True,text=''),'sacct':dict(ok=True,text='JobIDRaw|State|Start|ExitCode\n')})
class GateTests(unittest.TestCase):
 def test_complete_has_exactly_four_items(self):
  r=evaluate(fixture(),'p','f');self.assertTrue(r['ready']);self.assertEqual(len(r['items']),4);self.assertEqual(r['items'][0]['hits'],18);self.assertEqual(r['items'][3]['chains'][0]['population_variance_buses2'],2/3);self.assertEqual(r['items'][3]['chains'][0]['sample_variance_buses2'],1)
 def test_one_missing_suppresses_all_four(self):
  s=fixture();s['campaigns']['p1']['data']['rows'][0].pop('result_sha256');r=evaluate(s,'p','f');self.assertFalse(r['ready']);self.assertIsNone(r['items'])
 def test_terminal_failure_censored_not_miss(self):
  s=fixture();r0=s['campaigns']['p1']['data']['rows'][0];r0.pop('result_sha256');s['scheduler']['sacct']['text']+='100|TIMEOUT|2026-09-16|0:0\n';r=evaluate(s,'p','f');self.assertTrue(r['ready']);self.assertEqual(r['items'][0]['hits'],17);self.assertEqual(r['items'][0]['censored_trials'],1);self.assertIsNone(r['items'][0]['hit_fraction_of18'])
 def test_requeued_live_is_not_terminal(self):
  s=fixture();s['campaigns']['p1']['data']['rows'][0].pop('result_sha256');s['scheduler']['sacct']['text']+='100|PREEMPTED|2026-09-16|0:0\n';s['scheduler']['squeue']['text']='100|PENDING\n';self.assertFalse(evaluate(s,'p','f')['ready'])
 def test_reused_budget_mismatch_blocks(self):
  s=fixture();s['campaigns']['p1']['data']['endpoint_audit']['checks'][-1]['observed_stage1_limit_s']=1800;r=evaluate(s,'p','f');self.assertFalse(r['ready']);self.assertTrue(r['integrity_errors'])
 def test_source_hash_mismatch_blocks(self):self.assertFalse(evaluate(fixture(),'wrong','f')['ready'])
 def test_missing_variance_seed_suppresses_variance(self):
  s=fixture();rows=s['campaigns']['p1']['data']['rows'];row=next(r for r in rows if r['case_id']=='warm_w1_k32_seed2');row.pop('result_sha256');jid=next(r['job_id'] for r in s['registered_jobs'] if r['case_id']==row['case_id']);s['scheduler']['sacct']['text']+=f'{jid}|FAILED|2026-09-16|1:0\n';r=evaluate(s,'p','f');self.assertTrue(r['ready']);self.assertIsNone(r['items'][3]['chains'][0]['population_variance_buses2'])
 def test_unmatched_f6_no_cost_difference(self):
  s=fixture();r=next(r for r in s['campaigns']['f6_k5']['data']['rows'] if r.get('case_id')=='peak08_cg');r['buses']=4;r['matched_five_bus_comparison_eligible']=False;o=evaluate(s,'p','f');self.assertTrue(o['ready']);self.assertIsNone(o['items'][2]['tariffs'][0]['joint_minus_fixed_cost'])
 def test_physical_audit_failure_blocks(self):
  s=fixture();s['campaigns']['p1']['data']['endpoint_audit']['checks'][0]['tests']['physical']=False;self.assertFalse(evaluate(s,'p','f')['ready'])
if __name__=='__main__':unittest.main()

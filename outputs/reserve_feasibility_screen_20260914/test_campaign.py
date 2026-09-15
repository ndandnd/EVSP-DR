import importlib.util,json,os,tempfile,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).resolve().parent
CODE=Path(os.environ.get('CAPACITY_CODE_ROOT',HERE.parents[2])).resolve()
spec=importlib.util.spec_from_file_location('reserve_campaign',HERE/'campaign.py'); C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
class Tests(unittest.TestCase):
 def setUp(self): self.m=C.load_manifest()
 def test_manifest_and_inputs(self):
  v=C.validate_manifest(self.m,CODE);self.assertTrue(v['valid'],v['errors']);self.assertEqual(v['case_count'],10);self.assertTrue(all(x['valid'] for x in v['input_checks'].values()))
 def test_exact_matrix_and_uniform_settings(self):
  cases=self.m['cases']; self.assertEqual([x['index'] for x in cases],list(range(10)))
  for duty in range(13405,13409):
   cells=[x for x in cases if x['instance']==f'k1_duty{duty}']
   arms={x['arm'] for x in cells}; expected={'baseline','parx60'}|({'capacity','combined'} if duty==13408 else set())
   self.assertEqual(arms,expected)
  for x in cases:
   self.assertEqual((x['battery_kwh'],x['reserve_kwh']),(236.44,35.466));self.assertEqual(x['capacity_selector'],'prefix-memo')
   self.assertEqual((x['cg_wall_s'],x['mip_wall_s'],x['slurm_time_s']),(13200,600,15300))
  self.assertAlmostEqual(35.466,.15*236.44);self.assertFalse(self.m['common_model']['terminal_65_percent_floor'])
 def test_parameter_propagation_and_dedicated_driver(self):
  with tempfile.TemporaryDirectory() as d:
   for x in self.m['cases']:
    cmds=C.build_commands(self.m,CODE,Path(d)/x['case_id'],x,python='/verified/python')
    for cmd in cmds.values():
     joined=' '.join(cmd);self.assertIn('run_capacity_speed_event_cg.py',joined);self.assertIn('--battery-kwh 236.44',joined);self.assertIn('--reserve-kwh 35.466',joined);self.assertIn('--non-parx-kw 240.0',joined);self.assertIn(C.DRIVER_COMMIT,joined);self.assertNotIn('run_exact_pool_mip.py',joined)
    self.assertIn('--capacity-selector prefix-memo',' '.join(cmds['cg']))
 def test_all_twenty_actual_commands_parse(self):
  v=C.validate_driver_commands(self.m,CODE);self.assertTrue(v['valid'],v['failures']);self.assertEqual(v['parsed_command_count'],20)
 def test_policy_and_clean_environment(self):
  p=self.m['resource_policy'];self.assertTrue(p['submission_authorized']);self.assertEqual(p['array_concurrency'],10);self.assertIn('scaglione-compute-01',p['exclude']);self.assertFalse(p['requeue'])
  with mock.patch.dict(C.os.environ,{'PYTHONPATH':'bad','LD_LIBRARY_PATH':'bad'}): e=C.execution_environment()
  self.assertNotIn('PYTHONPATH',e);self.assertEqual(e['GRB_LICENSE_FILE'],C.GUROBI_LICENSE);self.assertEqual(e['OMP_NUM_THREADS'],'1')
 def test_tooling_hash_gate(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);(r/'x').write_text('ok');m={'tooling_sha256':{'x':C.sha256_file(r/'x')}};C.validate_tooling(m,r);(r/'x').write_text('changed')
   with self.assertRaises(RuntimeError):C.validate_tooling(m,r)
 def test_completed_timecap_is_endpoint_not_certificate(self):
  case=self.m['cases'][0]
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'results'/case['case_id']/'j_r0';a.mkdir(parents=True);(a/'pool.jsonl').write_text('{}\n')
   (a/'worker_status.json').write_text(json.dumps({'returncode':0,'manifest_sha256':'m','stages':{'cg':{'returncode':0}}}))
   (a/'cg.json').write_text(json.dumps({'status':'incomplete','stop_reason':'pricing_deadline','certified_rc_optimal':False,'terminal_exact_min_reduced_cost':None,'iterations':[],'pool_sha256':'p'}))
   v=C.collect_campaign(self.m,Path(d),manifest_path=HERE/'manifest.json')
  self.assertEqual(len(v['cg']),1);self.assertFalse(v['cg'][0]['provisional']);self.assertFalse(v['cg'][0]['result']['certified_rc_optimal'])
if __name__=='__main__':unittest.main()

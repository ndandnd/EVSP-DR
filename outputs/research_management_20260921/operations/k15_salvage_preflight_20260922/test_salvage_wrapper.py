import contextlib, importlib.util, io, json, os, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).with_name('run_salvage.py')
spec=importlib.util.spec_from_file_location('salvage',P); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class Checks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name);self.out=self.root/'out';self.job=self.root/'failed';(self.job/'dive').mkdir(parents=True)
  self.source=self.job/'dive/cg.json';self.source.write_text('{}');(self.job/'execution.json').write_text(json.dumps({'dive_wall_s':5347.861750387121}))
  self.cell={'unset_environment':[],'environment':{},'failed_job_id':'704511','source_hashes':{'augmented_result':{'path':str(self.source),'actual':mod.sha(self.source),'expected':mod.sha(self.source)}},'failed_dive_wall_s':5347.861750387121,'solver_limit_s':1852,'output_directory_template':'OUTPUT','native_worker_argv_template':['python',str(self.root/'code/src/run_exact_pool_mip.py'),'--out','OUTPUT/result.json']}
  self.plan=self.root/'plan.json';self.writeplan();self.calls=[]
  self.audit={'rejected_columns':0,'accepted_columns':1,'deterministically_repaired':0,'mip_ordered_pool_sha256':'HASH'}
  self.native=types.SimpleNamespace(verified_mip_code_identity=lambda:{'git_commit':'PIN'},load_pool=lambda *a,**k:({},[{}],[1]),prepare_strict_partition_pool=lambda *a,**k:([{}],self.audit.copy()))
 def writeplan(self):self.plan.write_text(json.dumps({'execution_commit':'PIN','cases':{'c1_k15':self.cell}}))
 def solve(self,argv,**kwargs):
  self.calls.append(argv);f=Path(argv[-1]);f.write_text(json.dumps({'physical_pool_audit':{'rejected_columns':0,'base_pool_ordered_sha256':'HASH'},'runtime_s':100}));return types.SimpleNamespace(returncode=0)
 def invoke(self,extra_env=None):
  env={'SLURM_JOB_ID':'999999','SLURM_RESTART_COUNT':'0',**(extra_env or {})}
  oldpath=sys.path[:]
  try:
   with patch.dict(os.environ,env),patch.dict(sys.modules,{'run_exact_pool_mip':self.native}),patch.object(sys,'argv',['run_salvage.py','--case','c1_k15','--plan',str(self.plan),'--out-root',str(self.out)]),patch.object(mod.subprocess,'run',side_effect=self.solve):mod.main()
  finally:sys.path[:]=oldpath
 def test_success_preserves_budget_and_records_gate(self):
  self.invoke();r=json.loads((self.out/'c1_k15/job_999999_r0/receipt.json').read_text());self.assertEqual(r['status'],'finished');self.assertEqual(r['solver_limit_s'],1852);self.assertEqual(len(self.calls),1);self.assertEqual(r['physical_gate']['rejected_columns'],0)
 def test_rejected_route_never_launches(self):
  self.audit['rejected_columns']=1
  with self.assertRaises(RuntimeError):self.invoke()
  self.assertEqual(self.calls,[]);r=json.loads((self.out/'c1_k15/job_999999_r0/receipt.json').read_text());self.assertEqual(r['status'],'failed')
 def test_changed_source_never_launches(self):
  self.source.write_text('changed')
  with self.assertRaises(RuntimeError):self.invoke()
  self.assertEqual(self.calls,[])
 def test_changed_dive_time_never_launches(self):
  (self.job/'execution.json').write_text(json.dumps({'dive_wall_s':5400}))
  with self.assertRaises(RuntimeError):self.invoke()
  self.assertEqual(self.calls,[])
 def test_previous_attempt_never_launches(self):
  (self.out/'c1_k15/job_old_r0').mkdir(parents=True)
  with self.assertRaises(RuntimeError):self.invoke()
  self.assertEqual(self.calls,[])
 def test_requeue_refused(self):
  with self.assertRaises(SystemExit):self.invoke({'SLURM_RESTART_COUNT':'1'})
  self.assertEqual(self.calls,[])
 def test_native_gate_systemexit_is_persisted(self):
  def gate(*a,**k):raise SystemExit('native gate rejected')
  self.native.prepare_strict_partition_pool=gate
  with self.assertRaises(SystemExit):self.invoke()
  self.assertEqual(self.calls,[]);r=json.loads((self.out/'c1_k15/job_999999_r0/receipt.json').read_text());self.assertEqual(r['status'],'failed')
if __name__=='__main__':unittest.main(verbosity=2)

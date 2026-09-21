import os,sys,json,subprocess,time,hashlib,math
from pathlib import Path
r=Path(sys.argv[1]);task=sys.argv[2];m=json.load(open(r/'manifest.json'));code=Path(m['code']);orig=Path(m['original_campaign']);oldcode=Path(m['original_code']);case=m['case'];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()==m['execution_commit']
for p,h in m['input_hashes'].items():assert sha(p)==h,p
attempt=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0');o=r/task/attempt;o.mkdir(parents=True,exist_ok=False)
start=time.time()
if task=='benchmark':
 results={}
 for label,c,mode in [('original_explicit',oldcode,'explicit'),('new_explicit',code,'explicit'),('new_packed',code,'lazy')]:
  cmd=[sys.executable,str(r/'benchmark_variant.py'),str(c),mode,str(o/(label+'.json'))]
  with open(o/(label+'.log'),'w') as f:subprocess.run(cmd,cwd=c,stdout=f,stderr=subprocess.STDOUT,check=True)
  results[label]=json.load(open(o/(label+'.json')))
 for label in ('new_explicit','new_packed'):
  a=results['original_explicit'];b=results[label]
  assert a['metrics']['event_lattice_sha256']==b['metrics']['event_lattice_sha256']
  for x,y in zip(a['pricing'],b['pricing']):
   assert math.isclose(x['rc'],y['rc'],rel_tol=0,abs_tol=1e-7),(label,x['rc'],y['rc'])
   assert x['physical_replay_pass'] and y['physical_replay_pass']
  if label=='new_explicit':assert a['metrics']==b['metrics']
 result={'all_five_fixed_duals_match':True,'physical_replay_pass':True,'results':results}
 (o/'result.json').write_text(json.dumps(result,indent=2));(r/'BENCHMARK_COMPLETE.json').write_text(json.dumps({'result':str(o/'result.json'),'sha256':sha(o/'result.json')}))
else:
 if task=='cg':
  gate=json.load(open(r/'BENCHMARK_COMPLETE.json'));assert sha(gate['result'])==gate['sha256'];assert json.load(open(gate['result']))['all_five_fixed_duals_match']
 cmd=[sys.executable,str(code/'src/run_capacity_speed_event_cg.py'),'--mode',task,'--arm','parx60','--instance',str(orig/case['input']),'--prices',str(code/'data/hourly_prices_flat.csv'),'--reference-data-dir',str(code/'data'),'--out',str(o/'result.json'),'--battery-kwh',str(case['battery_kwh']),'--reserve-kwh',str(case['reserve_kwh']),'--non-parx-kw','240','--soc-step','2.5','--block-min','5','--max-station-wait-min','1560','--threads','8','--expected-commit',m['execution_commit'],'--require-clean','--arc-mode','lazy']
 if task=='cg':
  parent=orig/'cases'/case['previous_group_case'];cmd+=['--pool-out',str(o/'pool.jsonl'),'--cg-wall-s','14400','--max-iters','100000','--inherit-status',str(parent/'cg.json'),'--inherit-pool',str(parent/'pool.jsonl'),'--inherit-instance',str(orig/'inputs'/f"{case['previous_group_case']}.csv"),'--inherit-compatible-commit',m['base_commit']]
 else:
  receipt=json.load(open(r/'CG_COMPLETE.json'));cmd+=['--pool',receipt['pool'],'--cg-status',receipt['result'],'--mip-wall-s','3600']
 (o/'command.json').write_text(json.dumps(cmd,indent=2))
 with open(o/'solver.log','w') as f:subprocess.run(cmd,cwd=code,stdout=f,stderr=subprocess.STDOUT,check=True)
 result=json.load(open(o/'result.json'))
 if task=='cg':
  assert result['pool_sha256']==sha(o/'pool.jsonl');(r/'CG_COMPLETE.json').write_text(json.dumps({'pool':str(o/'pool.jsonl'),'result':str(o/'result.json'),'pool_sha256':result['pool_sha256']}))
(o/'COMPLETE.json').write_text(json.dumps({'task':task,'attempt':attempt,'wall_s':time.time()-start,'execution_commit':m['execution_commit'],'result_sha256':sha(o/'result.json'),'manifest_sha256':sha(r/'manifest.json')},indent=2))

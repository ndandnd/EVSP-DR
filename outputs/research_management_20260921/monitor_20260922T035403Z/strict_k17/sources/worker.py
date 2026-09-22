"""Pinned solver successor execution; no production solver source mutation."""
import os, sys, json, subprocess, time, hashlib, shutil
from pathlib import Path
R=Path(sys.argv[1]); task=sys.argv[2]; case_id=sys.argv[3] if len(sys.argv)>3 else None
m=json.loads((R/'manifest.json').read_text()); code=Path(m['code']); orig=Path(m['original_campaign'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,p)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()==m['execution_commit']
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=code,text=True).strip()
for p,h in m['input_hashes'].items():assert sha(p)==h,p
attempt=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0')
o=R/(case_id or 'validation')/task/attempt;o.mkdir(parents=True,exist_ok=False)
started=time.time()
if task=='validation':
 sys.path[:0]=[str(code/'src')]
 from inherit_capacity_pool import inherit_pool,trip_mapping
 from run_capacity_speed_event_cg import build_problem,validate_injected_route,HORIZON_MIN,station_power
 receipt=json.load(open(m['parent_receipt']));status=json.load(open(receipt['result']))
 assert sha(receipt['pool'])==receipt['pool_sha256']==status['pool_sha256']
 case=m['cases'][0];child=orig/case['input'];parent=orig/'inputs'/f"{case['previous_group_case']}.csv"
 problem=build_problem(child.parent,child.name,reference_data_dir=code/'data',max_station_to_trip_wait_min=1560)
 routes,meta=inherit_pool(receipt['result'],receipt['pool'],parent,child,expected_physics=status['physics'],child_provenance=status['provenance'],new_checkpoint_id='successor-preflight',route_validator=lambda route:validate_injected_route(problem,route,case['battery_kwh'],240,case['reserve_kwh'],HORIZON_MIN,arrival_grace_min=0.0,station_charge_kw=station_power('parx60')))
 checks={case['id']:len(routes)}
 for c in m['cases']:checks[c['id']+'_mapped_trips']=len(trip_mapping(orig/'inputs'/f"{c['previous_group_case']}.csv",orig/c['input']))
 with open(o/'tests.log','w') as f:subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_inherit_capacity_pool.py','-v'],cwd=code,stdout=f,stderr=subprocess.STDOUT,check=True)
 write(o/'result.json',{'passed':True,'checks':checks,'inheritance':meta,'solver_source_changed':False,'compatibility_exception_needed':False})
 write(R/'VALIDATION_COMPLETE.json',{'result':str(o/'result.json'),'sha256':sha(o/'result.json')})
else:
 gate=json.load(open(R/'VALIDATION_COMPLETE.json'));assert sha(gate['result'])==gate['sha256'];assert json.load(open(gate['result']))['passed']
 c=next(c for c in m['cases'] if c['id']==case_id)
 cmd=[sys.executable,str(code/'src/run_capacity_speed_event_cg.py'),'--mode',task,'--arm','parx60','--instance',str(orig/c['input']),'--prices',str(code/'data/hourly_prices_flat.csv'),'--reference-data-dir',str(code/'data'),'--out',str(o/'result.json'),'--battery-kwh',str(c['battery_kwh']),'--reserve-kwh',str(c['reserve_kwh']),'--non-parx-kw','240','--soc-step','2.5','--block-min','5','--max-station-wait-min','1560','--threads','8','--expected-commit',m['execution_commit'],'--require-clean','--arc-mode','lazy']
 if task=='cg':
  parent_receipt=m['parent_receipt'] if c==m['cases'][0] else str(R/c['previous_group_case']/'CG_COMPLETE.json')
  parent=json.load(open(parent_receipt));assert sha(parent['result'])==parent.get('result_sha256',sha(parent['result']));assert sha(parent['pool'])==parent['pool_sha256']
  cmd+=['--pool-out',str(o/'pool.jsonl'),'--cg-wall-s','14400','--max-iters','100000']
  previous=sorted((R/case_id/'cg').glob('*/pool.jsonl'),key=lambda p:p.stat().st_mtime)
  previous=[p for p in previous if p.parent!=o]
  if previous:
   shutil.copy2(previous[-1],o/'pool.jsonl');cmd+=['--resume'];write(o/'resume_source.json',{'pool':str(previous[-1]),'sha256':sha(previous[-1])})
  else:cmd+=['--inherit-status',parent['result'],'--inherit-pool',parent['pool'],'--inherit-instance',str(orig/'inputs'/f"{c['previous_group_case']}.csv")]
 else:
  p=json.load(open(R/case_id/'CG_COMPLETE.json'));assert sha(p['pool'])==p['pool_sha256'];assert sha(p['result'])==p['result_sha256'];cmd+=['--pool',p['pool'],'--cg-status',p['result'],'--mip-wall-s','3600']
 write(o/'command.json',cmd)
 with open(o/'solver.log','w') as f:subprocess.run(cmd,cwd=code,stdout=f,stderr=subprocess.STDOUT,check=True)
 result=json.load(open(o/'result.json'))
 if task=='cg':
  assert result['pool_sha256']==sha(o/'pool.jsonl')
  write(R/case_id/'CG_COMPLETE.json',{'pool':str(o/'pool.jsonl'),'pool_sha256':sha(o/'pool.jsonl'),'result':str(o/'result.json'),'result_sha256':sha(o/'result.json')})
write(o/'COMPLETE.json',{'task':task,'attempt':attempt,'wall_s':time.time()-started,'execution_commit':m['execution_commit'],'manifest_sha256':sha(R/'manifest.json'),'result_sha256':sha(o/'result.json')})

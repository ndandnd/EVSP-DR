"""One full-arm graph, assembly, CG, or finite-pool MIP stage."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['cache','assemble','cg','mip'],required=True);p.add_argument('--case',required=True);a=p.parse_args()
root=a.root.resolve();m=json.loads((root/'manifest.json').read_text());cm=json.loads((root/'continuation_manifest.json').read_text());spec=cm['cases'][a.case];arm=spec['arm'];code=Path(m['code']);prepared=root/'prepared'
from replay_journal import sha
from atomic_pool_copy import atomic_pool_copy
assert sha(root/'manifest.json')==cm['campaign_manifest_sha256']
for name,h in cm['tooling_sha256'].items():assert sha(root/name)==h
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()==m['execution_commit']
base=root/'continuation'/a.case;canonical=(root/'assembly_status'/f'{arm}.json') if a.stage=='assemble' else base/f'{a.stage}.json'
canonical.parent.mkdir(parents=True,exist_ok=True)
if canonical.exists():
 previous=json.loads(canonical.read_text())
 if a.stage=='cache':assert sha(previous['cache_path'])==previous['cache_manifest']['pickle_sha256']
 elif a.stage=='cg':assert sha(previous['pool'])==previous['pool_sha256']
 elif a.stage=='assemble':assert sha(Path(previous['assembly'])/'assembly.json')==previous['assembly_sha256']
 else:assert previous['mode']=='mip'
 print(json.dumps({'already_completed':str(canonical),'validated':True}));sys.exit(0)
attempt=f"{os.environ.get('SLURM_JOB_ID','local')}_r{os.environ.get('SLURM_RESTART_COUNT','0')}";work=base/a.stage/attempt;work.mkdir(parents=True,exist_ok=False)
started=time.monotonic();record={'stage':a.stage,'case':a.case,'arm':arm,'attempt':attempt,'started_epoch':time.time(),'status':'running','manifest_sha256':sha(root/'manifest.json'),'continuation_manifest_sha256':sha(root/'continuation_manifest.json')}
record.update(solver_budget_scope='per attempt; resumed totals may exceed nominal allowance; retain all attempts for cumulative comparisons',cg_budget_s_per_attempt=spec['cg_wall_s'],mip_budget_s_per_attempt=spec['mip_wall_s'])
def save(): (work/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
save();sys.path.insert(0,str(code/'src'));sys.path.insert(0,str(prepared));os.chdir(code)
from sequence_replay import ARMS
from run_capacity_speed_event_cg import parser,build_network,provenance,atomic_json
from audit_giro_known_columns import build_problem,STATIONS
from utils_v2 import load_station_hourly_prices,base_station_name
phys=ARMS[arm];instance=prepared/spec['input'];assert sha(instance)==spec['input_sha256']
common=['--arm','parx60' if phys['parx_kw']==60 else 'baseline','--instance',str(instance),'--prices',str(code/'data/hourly_prices_flat.csv'),
 '--reference-data-dir',str(code/'data'),'--battery-kwh',str(phys['battery_kwh']),'--reserve-kwh',str(phys['reserve_kwh']),
 '--non-parx-kw','240','--max-station-wait-min','1560','--network-arc-mode','lazy','--checkpoint-interval-s','300',
 '--threads','8','--expected-commit',m['execution_commit'],'--require-clean']
def run(cmd):
 record['command']=cmd;save()
 with open(work/'solver.log','w') as f:subprocess.run(cmd,cwd=code,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 if a.stage=='cache':
  from action3_network_cache import cache_identity,write_cache,load_cache
  args=parser().parse_args(['--mode','cg',*common,'--out',str(work/'unused.json')]);prov=provenance(args,instance,args.prices,args.reference_data_dir)
  problem=build_problem(instance.parent,instance.name,reference_data_dir=code/'data',max_station_to_trip_wait_min=1560)
  prices=load_station_hourly_prices(args.prices,sorted({base_station_name(s) for s in STATIONS}))
  t=time.monotonic();net=build_network(args,problem,prices);identity=cache_identity(args,prov);cache=work/'network.pkl';meta=write_cache(cache,net,identity,time.monotonic()-t)
  # Hash + metadata validated at write; the CG consumer repeats full validation.
  # Atomic cache.json is the sole ready pointer to both completed attempt files.
  output={'cache_path':str(cache),'cache_manifest':meta,'provenance':prov};atomic_json(canonical,output)
 elif a.stage=='assemble':
  out=work/'assembled'
  run([sys.executable,str(prepared/'assemble.py'),'--root',str(prepared),'--code',str(code),'--sequences',m['source_sequences'],
       '--chunks',str(root/'cases'/arm),'--arm',arm,'--out',str(out)])
  result=json.loads((out/'assembly.json').read_text());assert result['cg_seed_ready'];atomic_json(canonical,{'assembly':str(out),'assembly_sha256':sha(out/'assembly.json'),'result':result})
 elif a.stage=='cg':
  seed=Path(json.loads((root/'assembly_status'/f'{arm}.json').read_text())['assembly']);assembly=json.loads((seed/'assembly.json').read_text());assert assembly['cg_seed_ready']
  cache_path=json.loads((base/'cache.json').read_text())['cache_path']
  prior=[q for q in (base/'cg').glob('*/solve/pool.jsonl') if work not in q.parents]
  if prior:
   latest=max(prior,key=lambda q:q.stat().st_mtime);solve=work/'solve';solve.mkdir();atomic_pool_copy(latest,solve/'pool.jsonl')
   record['resumed_pool']={'path':str(latest),'sha256':sha(latest)}
   cmd=[sys.executable,str(code/'src/run_capacity_speed_event_cg.py'),'--mode','cg',*common,'--out',str(solve/'cg.json'),'--pool-out',str(solve/'pool.jsonl'),
     '--cg-wall-s',str(spec['cg_wall_s']),'--max-iters','100000','--network-cache',cache_path,'--resume']
  else:
   solve=work/'solve';cmd=[sys.executable,str(prepared/'continue_cg.py'),'--root',str(prepared),'--code',str(code),'--seed',str(seed),'--out',str(solve),
     '--network-cache',cache_path,'--run-cg','--cg-seconds',str(spec['cg_wall_s']),'--threads','8']
   if spec.get('group'):cmd+=['--group',spec['group']]
   if assembly['unknown_sequences']:
    cmd+=['--allow-unresolved-sequences'];record['unresolved_seed_qualification']=assembly['unknown_sequences']
  run(cmd);result=json.loads((solve/'cg.json').read_text());assert result['pool_sha256']==sha(solve/'pool.jsonl')
  atomic_json(canonical,result)
 else:
  run([sys.executable,str(code/'src/run_capacity_speed_event_cg.py'),'--mode','mip',*common,'--out',str(work/'mip.json'),
   '--pool',json.loads((base/'cg.json').read_text())['pool'],'--cg-status',str(base/'cg.json'),'--mip-wall-s',str(spec['mip_wall_s'])])
  atomic_json(canonical,json.loads((work/'mip.json').read_text()))
 record.update(status='completed',ended_epoch=time.time(),elapsed_s=time.monotonic()-started,canonical_sha256=sha(canonical));save()
except BaseException as e:
 record.update(status='failed',error=f'{type(e).__name__}: {e}',ended_epoch=time.time(),elapsed_s=time.monotonic()-started);save();raise

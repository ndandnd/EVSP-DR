#!/usr/bin/env python3
"""Execute one frozen strict-prefix stage; no submission logic."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--mode',choices=['cg','mip'],required=True);a=p.parse_args()
root=a.root.resolve();code=a.code.resolve();manifest=json.load(open(root/'manifest.json'));cases={c['id']:c for c in manifest['cases']};case=cases[a.case]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
instance=root/case['input'];assert sha(instance)==case['input_sha256']
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip();assert commit==manifest['execution_commit']
base=root/'cases'/a.case;base.mkdir(parents=True,exist_ok=True)
if (base/f'{a.mode}.json').exists():raise SystemExit('Completed canonical output exists; refusing duplicate execution')
attempt=f"{os.environ.get('SLURM_JOB_ID','local')}_r{os.environ.get('SLURM_RESTART_COUNT','0')}";work=base/a.mode/attempt;work.mkdir(parents=True,exist_ok=False)
args=[sys.executable,str(code/'src/run_capacity_speed_event_cg.py'),'--mode',a.mode,'--arm','parx60','--instance',str(instance),'--prices',str(code/'data/hourly_prices_flat.csv'),'--reference-data-dir',str(code/'data'),'--out',str(work/'result.json'),'--battery-kwh',str(case['battery_kwh']),'--reserve-kwh',str(case['reserve_kwh']),'--non-parx-kw','240','--soc-step','2.5','--block-min','5','--max-station-wait-min','1560','--threads','8','--expected-commit',commit,'--require-clean']
if a.mode=='cg':
 args+=['--pool-out',str(work/'pool.jsonl'),'--cg-wall-s',str(case['cg_wall_s']),'--max-iters',str(case['max_iters'])]
 parent=case['previous_group_case']
 if parent:
  pb=root/'cases'/parent;assert (pb/'cg.json').is_file() and (pb/'pool.jsonl').is_file()
  args+=['--inherit-status',str(pb/'cg.json'),'--inherit-pool',str(pb/'pool.jsonl'),'--inherit-instance',str(root/cases[parent]['input'])]
else:args+=['--pool',str(base/'pool.jsonl'),'--cg-status',str(base/'cg.json'),'--mip-wall-s',str(case['mip_wall_s'])]
record={'manifest_sha256':sha(root/'manifest.json'),'case':case,'mode':a.mode,'attempt':attempt,'command':args,'execution_commit':commit,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'started_unix':time.time()}
(root/'logs').mkdir(exist_ok=True);(work/'command.json').write_text(json.dumps(record,indent=2))
with open(work/'solver.log','w') as log:subprocess.run(args,cwd=code,stdout=log,stderr=subprocess.STDOUT,check=True)
result=json.load(open(work/'result.json'))
if a.mode=='cg':
 assert result['pool_sha256']==sha(work/'pool.jsonl')
 shutil.copyfile(work/'pool.jsonl',base/'pool.jsonl')
else:
 pool=[json.loads(line) for line in (base/'pool.jsonl').read_text().splitlines() if line.strip()]
 selected=[pool[i] for i in result['result']['selected_indices']]
 (work/'selected_routes.json').write_text(json.dumps(selected,indent=2))
shutil.copyfile(work/'result.json',base/f'{a.mode}.json')
record.update(finished_unix=time.time(),result_sha256=sha(work/'result.json'));(work/'COMPLETE.json').write_text(json.dumps(record,indent=2))

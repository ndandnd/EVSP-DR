"""One resumable full-coverage shard. A timeout is unknown, never infeasible."""
import collections,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();m=json.loads((root/'manifest.json').read_text());task=int(os.environ['SLURM_ARRAY_TASK_ID'])
arm=m['arms'][task%6];shard=task//6;prepared=root/'prepared';code=Path(m['code']);sys.path.insert(0,str(prepared));sys.path.insert(0,str(code/'src'))
from sequence_replay import replay,ARMS,sha
from replay_journal import atomic_json,canonical,recover,completed
from audit_giro_known_columns import build_problem,STATIONS
from event_pricer_network import _event_times,normalize_event_station_prices
from utils_v2 import load_station_hourly_prices,base_station_name
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()==m['execution_commit']
assert sha(prepared/'manifest.json')==m['prepared_manifest_sha256'];pm=json.loads((prepared/'manifest.json').read_text())
for name,h in pm['tooling_sha256'].items():assert sha(prepared/name)==h
for name,h in pm['static_hashes'].items():assert sha(code/'data'/name)==h
assert sha(prepared/pm['group_map'])==pm['group_map_sha256']
sm=json.loads((root/'shards.json').read_text());assert sm['manifest_sha256']==sha(root/'manifest.json');spec=sm['shards'][shard];path=Path(spec['path']);assert sha(path)==spec['sha256'];source=[json.loads(l) for l in open(path)]
assert len(source)==spec['count'];case=root/'cases'/arm/f'{shard:03d}';case.mkdir(parents=True,exist_ok=True)
attempt=f"{os.environ['SLURM_ARRAY_JOB_ID']}_{task}_r{os.environ.get('SLURM_RESTART_COUNT','0')}";audit=case/'attempts'/attempt;audit.mkdir(parents=True,exist_ok=False)
start=time.monotonic();record={'arm':arm,'shard':shard,'task':task,'attempt':attempt,'source_shard_sha256':sha(path),'started_epoch':time.time(),'status':'running','manifest_sha256':sha(root/'manifest.json')}
def save(): (audit/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
save()
with open(case/'lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if completed(case,{'arm':arm,'full_campaign_manifest_sha256':sha(root/'manifest.json'),'source_shard_sha256':sha(path),'sequence_count':len(source)}):record.update(status='already_completed',elapsed_s=time.monotonic()-start);save();sys.exit(0)
 journal=case/'records.jsonl';done=[]
 done=recover(journal,source,arm,audit)
 instance=prepared/pm['instance'];assert sha(instance)==pm['instance_sha256']
 problem=build_problem(instance.parent,instance.name,reference_data_dir=code/'data',max_station_to_trip_wait_min=1560)
 prices=load_station_hourly_prices(code/'data/hourly_prices_flat.csv',sorted({base_station_name(s) for s in STATIONS}))
 events=_event_times(problem,normalize_event_station_prices(prices,horizon_min=1560,strict_tariff_coverage=False),5)
 groups={int(k):v['group'] for k,v in json.loads((prepared/pm['group_map']).read_text()).items()}
 with open(journal,'a') as f:
  for src in source[len(done):]:
   if time.monotonic()-start>m['replay_attempt_budget_s']:
    record.update(status='checkpointed_for_requeue',finished_sequences=len(done),elapsed_s=time.monotonic()-start,ended_epoch=time.time());save()
    subprocess.run(['/usr/local/slurm/slurm-25.05.5/bin/scontrol','requeue',f"{os.environ['SLURM_ARRAY_JOB_ID']}_{task}"],check=True);sys.exit(0)
   outcome,route=replay(problem,prices,events,src['trip_sequence'],ARMS[arm],groups,seconds=m['sequence_timeout_s'])
   outcome.update(arm=arm,source_sequence=src)
   if route:route.update(origin='original_c5_k31_sequence_charging_reoptimized',found_iter=0,single_factor_source=src,single_factor_arm=arm)
   row={'outcome':outcome,'route':route};row['record_sha256']=canonical(row);f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno());done.append(row)
 assert len(done)==len(source)
 with open(case/'outcomes.jsonl','w') as out,open(case/'survivors.jsonl','w') as survivors:
  for row in done:
   out.write(json.dumps(row['outcome'])+'\n')
   if row['route']:survivors.write(json.dumps(row['route'])+'\n')
  out.flush();os.fsync(out.fileno());survivors.flush();os.fsync(survivors.fileno())
 receipt={'arm':arm,'manifest_sha256':m['prepared_manifest_sha256'],'full_campaign_manifest_sha256':sha(root/'manifest.json'),
 'sequences_sha256':m['source_sequences_sha256'],'shard':shard,'sequence_count':len(done),'source_shard_sha256':sha(path),
 'outcomes_sha256':sha(case/'outcomes.jsonl'),'survivors_sha256':sha(case/'survivors.jsonl'),'records_sha256':sha(journal),
 'outcome_counts':dict(collections.Counter(r['outcome']['status'] for r in done)),'completed_epoch':time.time()}
 atomic_json(case/'COMPLETE.json',receipt);record.update(status='completed',finished_sequences=len(done),elapsed_s=time.monotonic()-start,ended_epoch=time.time());save()

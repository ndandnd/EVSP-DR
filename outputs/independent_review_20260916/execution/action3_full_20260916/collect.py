"""Read-only, campaign-scoped status; never submits, resumes, or changes files remotely."""
import collections,datetime,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REMOTE=r'''
import collections,datetime,json,subprocess
from pathlib import Path
r=Path('/home/nc437/ladder-lite/action3_full_20260916');b='/usr/local/slurm/slurm-25.05.5/bin/'
m=json.loads((r/'manifest.json').read_text());out={'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest':m,'replay':{},'stages':{},'attempts':[],'errors':[]}
jobs=[]
for name in ['replay_jobs.json','continuation_jobs.json']:
 d=json.loads((r/name).read_text());out[name]=d
 jobs += [v['job_id'] for v in d['jobs'].values()]
ids=','.join(jobs)
for key,args in [('squeue',['squeue','--jobs='+ids,'--noheader','--format=%i|%T|%R|%M|%N']),('sacct',['sacct','-u','nc437','--starttime=2026-09-16','--duplicates','--jobs='+ids,'-P','--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,NodeList,Start,End'])]:
 p=subprocess.run([b+args[0],*args[1:]],capture_output=True,text=True);out[key]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for arm in m['arms']:
 counts=collections.Counter();receipts=[]
 for p in sorted((r/'cases'/arm).glob('*/COMPLETE.json')):
  try:
   d=json.loads(p.read_text());counts.update(d['outcome_counts']);receipts.append(str(p))
  except Exception as e:out['errors'].append({'path':str(p),'error':str(e)})
 out['replay'][arm]={'completed_shards':len(receipts),'expected_shards':125,'completed_sequence_counts':dict(counts),'expected_sequences':254068,'partial_shard_work_not_counted':True}
 if arm=='baseline':out['replay'][arm]['unexpected_nonfeasible_counts']={k:v for k,v in counts.items() if k!='feasible'}
for p in sorted((r/'continuation').glob('*/*.json')):
 try:
  d=json.loads(p.read_text());out['stages'][str(p.relative_to(r))]={k:v for k,v in d.items() if not isinstance(v,(dict,list))}
 except Exception as e:out['errors'].append({'path':str(p),'error':str(e)})
for p in sorted((r/'assembly_status').glob('*.json')):
 d=json.loads(p.read_text());result=d['result'];out['stages'][str(p.relative_to(r))]={'assembly':d['assembly'],**{k:v for k,v in result.items() if k not in ['source_receipts','singleton_fallbacks']}}
for pattern in ['continuation/*/*/*/execution.json','cases/*/*/attempts/*/execution.json']:
 for p in sorted(r.glob(pattern)):
  try:
   d=json.loads(p.read_text());out['attempts'].append({k:v for k,v in d.items() if k!='command'})
  except Exception as e:out['errors'].append({'path':str(p),'error':str(e)})
for p in sorted((r/'logs').glob('*.err')):
 if p.stat().st_size:out['errors'].append({'stderr_path':str(p),'tail':p.read_text(errors='replace')[-2500:]})
out['interpretation']='Scheduler success, source replay coverage, CG pricing certificate, finite-pool MIP proof, physical feasibility and GIRO target attainment are separate. Missing stage files mean pending or interrupted, not scientific failure. This read-only snapshot does not hash every large solver output.'
print(json.dumps(out))
'''
p=subprocess.run(['ssh','-S','/Users/nadan/.ssh/evsp-unicorn.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','nc437@unicorn-login-01.coecis.cornell.edu','/home/nc437/evsp_env/bin/python','-'],input=REMOTE,text=True,capture_output=True)
if p.returncode:sys.exit('Unicorn collection failed: '+p.stderr)
d=json.loads(p.stdout);folder=HERE/'snapshots'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');folder.mkdir(parents=True);(folder/'status.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'snapshot':str(folder/'status.json'),'replay':d['replay'],'errors':d['errors'],'stages':list(d['stages'])},indent=2))

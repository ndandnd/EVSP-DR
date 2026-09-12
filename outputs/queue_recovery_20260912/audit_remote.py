import subprocess,json,datetime,re,concurrent.futures
from pathlib import Path
S='/usr/local/slurm/slurm-25.05.5/bin/'
def run(args):
 r=subprocess.run(args,capture_output=True,text=True);return {'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
out={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
out['queue']=run([S+'squeue','-u','nc437','-r','-h','-o','%i|%j|%T|%M|%R'])
ids=[x.split('|')[0] for x in out['queue']['stdout'].splitlines()]
def ctl(j):return j,run([S+'scontrol','show','job',j,'-o'])
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as p:out['jobs']=dict(p.map(ctl,ids))
parents=set()
for v in out['jobs'].values():
 m=re.search(r'Dependency=(\S+)',v['stdout'])
 if m:parents.update(re.findall(r'(?:after[a-z]*:|:)(\d+(?:_\d+)?)',m.group(1)))
out['parent_ids']=sorted(parents)
out['accounting']=run([S+'sacct','-S','2026-09-08','-E','2026-09-14','-u','nc437','-X','-P','-n','-j',','.join(sorted(set(ids)|parents)),'-o','JobID,JobName,State,ExitCode,Elapsed,Start,End,MaxRSS,ReqMem,AllocCPUS,Timelimit'])
out['partitions']=run([S+'sinfo','-p','default_partition','-h','-o','%P|%a|%D|%C|%m|%l'])
out['roots']={}
for key in ['overnight_extension_20260912','parallel_research_20260911','w2_chain_recovery_retry2_20260912']:
 base=Path('/home/nc437/ladder-lite')/key
 out['roots'][key]={}
 for name in ['manifest.json','jobs.json','worker.sh','run_case.py','worker.py']:
  p=base/name
  if p.exists() and p.stat().st_size<300000:out['roots'][key][name]=p.read_text()
print(json.dumps(out))

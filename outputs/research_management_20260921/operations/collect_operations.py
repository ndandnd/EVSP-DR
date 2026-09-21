import subprocess,json,datetime,pathlib
out=pathlib.Path(__file__).parent / ('snapshot_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
remote=r'''
import subprocess,json,pathlib,datetime,hashlib
B=pathlib.Path('/usr/local/slurm/slurm-25.05.5/bin');R=pathlib.Path('/home/nc437/ladder-lite/chain_extension_33_40_20260921');S=pathlib.Path('/home/nc437/ladder-lite/strict_packed_successors_20260921')
def run(args):
 p=subprocess.run(list(map(str,args)),capture_output=True,text=True);return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
benchmark=pathlib.Path('/home/nc437/ladder-lite/strict_representation_benchmarks_20260921')
ids=[x['job_id'] for x in json.load(open(R/'jobs.json'))]+['646674','646675','646676']+[x['job'] for x in json.load(open(S/'jobs.json')).values()]
ids += [x['job'] for x in json.load(open(benchmark/'jobs.json')).values()]
d={'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'squeue':run([B/'squeue','-r','-u','nc437','-h','-o','%i|%j|%T|%M|%R|%E']),'sacct':run([B/'sacct','-D','--starttime=2026-09-21','-j',','.join(ids),'-n','-P','--format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem,ExitCode,NodeList,Start,End,AllocCPUS']),'scontrol':run([B/'scontrol','show','job','--oneliner'])}
d['scontrol']['stdout']='\n'.join(line for line in d['scontrol']['stdout'].splitlines() if 'UserId=nc437(' in line and any(name in line for name in ['JobName=drX_','JobName=pack','JobId=537227','JobName=rvS']))
d['graph_files']=[]
for case in sorted((R/'cases').glob('*')):
 for p in (case/'cache').rglob('*'):
  if p.is_file():
   entry={'path':str(p),'bytes':p.stat().st_size,'modified_unix':p.stat().st_mtime}
   if p.suffix=='.json' and p.stat().st_size<500000:entry.update(sha256=sha(p),contents=p.read_text())
   if p.suffix=='.jsonl':
    with open(p,'rb') as f:f.seek(max(0,p.stat().st_size-1500));entry['tail']=f.read().decode(errors='replace')
   d['graph_files'].append(entry)
d['graph_logs']=[]
for p in sorted((R/'logs').glob('*')):
 if p.is_file() and p.stat().st_size:
  with open(p,'rb') as f:f.seek(max(0,p.stat().st_size-2500));tail=f.read().decode(errors='replace')
  d['graph_logs'].append({'path':str(p),'bytes':p.stat().st_size,'tail':tail})
print(json.dumps(d))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','nc437@unicorn-login-01.coecis.cornell.edu','python3 -'],input=remote,text=True,capture_output=True);assert p.returncode==0,p.stderr
d=json.loads(p.stdout);(out/'snapshot.json').write_text(json.dumps(d,indent=2))
for k in ('squeue','sacct','scontrol'):(out/(k+'.txt')).write_text(d[k]['stdout'])
rows=[s.split('|') for s in d['squeue']['stdout'].splitlines()];scope=[x for x in rows if x[1].startswith(('drX_','pack21_','packS_','packB_'))]
from collections import Counter
summary={'collected_utc':d['collected_utc'],'scoped_running':sum(x[2]=='RUNNING' for x in scope),'scoped_states':dict(Counter(x[2] for x in scope)),'scoped_pending_reasons':dict(Counter(x[4] for x in scope if x[2]=='PENDING')),'graph_running':sum(x[1]=='drX_graphs' and x[2]=='RUNNING' for x in scope),'graph_file_count':len(d['graph_files'])}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(out);print(json.dumps(summary))

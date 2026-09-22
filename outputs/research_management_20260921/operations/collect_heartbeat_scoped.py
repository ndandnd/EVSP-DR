"""One SSH collection, limited to graph40 and strict successor/benchmark roots."""
import subprocess,json,datetime,pathlib
out=pathlib.Path(__file__).parent/('heartbeat_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
remote=r'''
import subprocess,json,pathlib,datetime,hashlib
B=pathlib.Path('/usr/local/slurm/slurm-25.05.5/bin');root=pathlib.Path('/home/nc437/ladder-lite');R=root/'chain_extension_33_40_20260921';S=root/'strict_packed_successors_20260921';T=root/'strict_representation_benchmarks_20260921'
def run(a):
 p=subprocess.run(list(map(str,a)),capture_output=True,text=True);return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def entry(p,full=False):
 x={'path':str(p),'bytes':p.stat().st_size,'modified_unix':p.stat().st_mtime}
 if full:x.update(sha256=sha(p),contents=p.read_text())
 else:
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-8000));x['tail']=f.read().decode(errors='replace')
 return x
ids=[v['job_id'] for v in json.load(open(R/'jobs.json'))]+[v['job'] for p in [S,T] for v in json.load(open(p/'jobs.json')).values()]
d={'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope_roots':[str(x) for x in [R,S,T]],'sacct':run([B/'sacct','-D','--starttime=2026-09-21','-j',','.join(ids),'-n','-P','--format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem,ExitCode,NodeList,Start,End,AllocCPUS']),'squeue':run([B/'squeue','-r','-u','nc437','-h','-o','%i|%j|%T|%M|%R|%E|%x'])}
d['squeue']['stdout']='\n'.join(x for x in d['squeue']['stdout'].splitlines() if x.split('|')[1].startswith(('drX_','packS_','packB_')))
d['graphs']=[]
for c in sorted((R/'cases').glob('*')):
 for p in (c/'cache').glob('*/*'):
  if p.is_file() and p.name in ['execution.json','COMPLETE.json','complete.json','result.json','progress.jsonl','network.pkl.manifest.json']:
   d['graphs'].append(entry(p,full=p.suffix=='.json' and p.stat().st_size<1000000))
d['artifacts']=[]
for campaign in [S,T]:
 for p in sorted(campaign.rglob('*')):
  if not p.is_file() or '__pycache__' in str(p):continue
  if p.suffix=='.json' and p.stat().st_size<6000000:d['artifacts'].append(entry(p,full=True))
  elif p.suffix in ['.log','.err'] and p.stat().st_size:
   e=entry(p);e['sha256']=sha(p);d['artifacts'].append(e)
 for p in campaign.glob('*/cg/*/result.json'):
  j=json.load(open(p));pool=pathlib.Path(j['pool']);d['artifacts'].append({'path':str(pool),'bytes':pool.stat().st_size,'sha256':sha(pool),'declared_sha256':j['pool_sha256']})
print(json.dumps(d))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','nc437@unicorn-login-01.coecis.cornell.edu','python3 -'],input=remote,text=True,capture_output=True)
(out/'ssh_receipt.json').write_text(json.dumps({'returncode':p.returncode,'stderr':p.stderr},indent=2));assert p.returncode==0,p.stderr
d=json.loads(p.stdout);(out/'snapshot.json').write_text(json.dumps(d,indent=2));print(str(out));print(d['collected_utc'])

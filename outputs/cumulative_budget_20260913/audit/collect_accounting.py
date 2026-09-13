from pathlib import Path
import json,csv,io,hashlib,subprocess
H=Path('/home/nc437/ladder-lite');sources={};jobs={}
def data(rel):
 p=H/rel;b=p.read_bytes();sources[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()};return b.decode()
for rel,chain in [('nested_warm_multichain_p1246_k2_10_20260910_ecb60c1/cg_jobs.tsv',None),('nested_warm_chain_p3_k2_10_20260909_8830a34/jobs_parallel8.tsv',3),('nested_warm_chain_p5_k2_10_20260910_ecb60c1/cg_jobs.tsv',5)]:
 for r in csv.DictReader(io.StringIO(data(rel)),delimiter='\t'):
  c=chain or int(r['replicate']);jobs[f'c{c}_k{int(r["scale"]):02d}']={'job_id':r['job_id'],'submission':r,'mapping_source':rel}
for cid,v in json.loads(data('full_pool_recovery_20260912/case_jobs.json')).items():
 jobs[cid.replace('w','c',1)]={'job_id':v['cg'],'mapping_source':'full_pool_recovery_20260912/case_jobs.json'}
fields=['JobIDRaw','JobID','State','ElapsedRaw','AllocCPUS','CPUTimeRAW','TotalCPU','UserCPU','SystemCPU','Start','End','Restarts','ExitCode']
args=['/usr/local/slurm/slurm-25.05.5/bin/sacct','-D','-P','-S','2026-09-08','-E','2026-09-14','-j',','.join(sorted({r['job_id'] for r in jobs.values()})),'--format='+','.join(fields)]
r=subprocess.run(args,capture_output=True,text=True)
print(json.dumps({'sources':sources,'job_mapping':jobs,'sacct_argv':args,'sacct_returncode':r.returncode,'sacct_stdout':r.stdout,'sacct_stderr':r.stderr},indent=2))

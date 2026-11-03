"""One immutable rematch case attempt; no retries or downstream submissions."""
import datetime,hashlib,json,os,pathlib,subprocess,sys,time
P=pathlib.Path;root=P(os.environ['REMATCH_ROOT']);freeze=json.loads((root/'FREEZE.json').read_text())
for rel,h in freeze['artifact_sha256'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
index=int(os.environ['SLURM_ARRAY_TASK_ID']);case=freeze['task_map'][index];assert index in [0,1,2]
attempt='job'+os.environ['SLURM_ARRAY_JOB_ID']+'_'+str(index)+'_r'+os.environ.get('SLURM_RESTART_COUNT','0')
out=root/'results'/case/attempt;out.mkdir(parents=True,exist_ok=False)
r={'case':case,'task_index':index,'array_job_id':os.environ['SLURM_ARRAY_JOB_ID'],'slurm_job_id':os.environ['SLURM_JOB_ID'],'restart_count':int(os.environ.get('SLURM_RESTART_COUNT','0')),'freeze_sha256':hashlib.sha256((root/'FREEZE.json').read_bytes()).hexdigest(),'freeze_utc':freeze['frozen_utc'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running'}
(root/'results'/case).chmod(0o700)
(out/'wrapper.private.json').write_text(json.dumps(r,indent=2)+'\n');tick=time.monotonic()
with (out/'stdout.private.log').open('w') as log:
 code=subprocess.call([sys.executable,'-u',str(root/'code/rematch.py'),'--manifest',str(root/'manifests'/(case+'.json')),'--output',str(out/'solve'),'--start-fee','5'],stdout=log,stderr=subprocess.STDOUT,cwd=root/'code')
r.update(status='terminal',exit_code=code,wall_seconds=time.monotonic()-tick,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(out/'wrapper.private.json').write_text(json.dumps(r,indent=2)+'\n')
raise SystemExit(code)

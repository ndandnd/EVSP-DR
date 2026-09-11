import subprocess,json,hashlib,os,datetime,shutil
from pathlib import Path
commit='253588e9b22d68fcbc67cb56bc3eb30cbb0e16b6'
base=Path('/home/nc437/ladder-lite');old=base/'capacity_speed_pilot_20260910_timeout6_rerun_7d38ef';oldcode=base/'capacity_speed_pilot_20260910_v2_7d38efd/code';root=base/'capacity_deadline5_retry_20260911_253588e';code=root/'code';bin='/usr/local/slurm/slurm-25.05.5/bin/';indices=[5,9,11,13,15]
def run(args,**kw):
 r=subprocess.run(args,capture_output=True,text=True,**kw)
 if r.returncode:raise RuntimeError({'args':args,'stdout':r.stdout[-2000:],'stderr':r.stderr[-2000:]})
 return r.stdout
assert not root.exists() or (set(p.name for p in root.iterdir()) <= {'logs'} and not any((root/'logs').iterdir())), 'retry root has progress; inspect before restart'
policy=(base/'SCAGLIONE_RESOURCE_POLICY.md').read_bytes()
account=run([bin+'sacct','-S','2026-09-10T23:00:00','-u','nc437','-X','-n','-P','-j','811181','-o','JobID,State,ExitCode'])
for i in indices:assert f'811181_{i}|TIMEOUT|' in account
run(['git','-C',str(oldcode),'fetch',str(base/'capacity_fix_253588e.bundle'),'codex/capacity-timeout-checkpoint-20260911'])
root.mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True);run(['git','-C',str(oldcode),'worktree','add','--detach',str(code),commit])
run(['rsync','-a','--ignore-existing',str(oldcode/'data')+'/',str(code/'data')+'/'])
launch=json.loads((old/'launch.json').read_text());miplaunch=json.loads((old/'launch_manifest.json').read_text())
for t in launch['tasks']:
 t['commit']=commit
for i in indices:
 t=launch['tasks'][i];src=oldcode/t['instance'];dst=code/t['instance'];dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():shutil.copy2(src,dst)
 assert hashlib.sha256(dst.read_bytes()).hexdigest()==t['instance_sha256']
 assert hashlib.sha256((code/t['prices']).read_bytes()).hexdigest()==launch['prices_sha256']
 assert not (old/'results'/t['task_id']/'pool.jsonl').exists()
launch['commit']=commit
(root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');(root/'launch_manifest.json').write_text(json.dumps(miplaunch,indent=2)+'\n')
for name in ['cg_runner.py','cg_mip_runner.py','cg_worker.sh','mip_worker.sh']:
 s=(old/name).read_text().replace(str(old),str(root))
 if name=='cg_runner.py':s=s.replace(str(oldcode),str(code))
 (root/name).write_text(s);(root/name).chmod(0o755)
assert run(['git','-C',str(code),'rev-parse','HEAD']).strip()==commit
assert not run(['git','-C',str(code),'status','--porcelain','--untracked-files=no']).strip()
env=dict(os.environ,GRB_LICENSE_FILE='/share/apps/software/gurobi/gurobi.lic',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
test=subprocess.run(['/home/nc437/evsp_env/bin/python','-m','unittest','tests.test_capacity_speed_event_cg','tests.test_event_pricer_network'],cwd=code,env=env,capture_output=True,text=True,timeout=180)
(root/'tests.txt').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stderr[-3000:]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'campaign_root':str(root),'cg_code_commit':commit,'mip_code_commit':'9bf3f752a5d2786bf5a1e9c613ff6440415569a9','indices':indices,'original_timeout_accounting':account,'source_campaign':str(old),'policy_sha256':hashlib.sha256(policy).hexdigest(),'fresh_not_resume':True,'scientific_settings':'same original inputs, physics, objective, capacity arms,8hCG,25minMIP; code adds deadline/checkpoint only','wrapper_sha256':{n:sha(root/n) for n in ['cg_runner.py','cg_mip_runner.py','cg_worker.sh','mip_worker.sh']},'launch_sha256':sha(root/'launch.json'),'tests_sha256':sha(root/'tests.txt'),'jobs':[]}
def save():(root/'retry_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
save()
common=['--parsable','--no-requeue','--exclude=scaglione-compute-01','--chdir='+str(root)]
cg=run([bin+'sbatch',*common,'--array=5,9,11,13,15%5','--partition=default_partition','--cpus-per-task=1','--mem=24G','--time=09:00:00','--job-name=cap5dCG','--output='+str(root/'logs/cg_%A_%a.out'),'--error='+str(root/'logs/cg_%A_%a.err'),str(root/'cg_worker.sh')]).strip().split(';')[0]
record['jobs'].append({'stage':'cg','job_id':cg,'array':indices,'concurrency':5,'cpus':1,'mem':'24G','time':'09:00:00','dependency':None});save()
for i in indices:
 dep=f'afterok:{cg}_{i}'
 job=run([bin+'sbatch',*common,'--array='+str(i),'--partition=scaglione','--cpus-per-task=8','--mem=16G','--time=00:30:00','--job-name=cap5dM','--dependency='+dep,'--output='+str(root/'logs/mip_%A_%a.out'),'--error='+str(root/'logs/mip_%A_%a.err'),str(root/'mip_worker.sh')]).strip().split(';')[0]
 record['jobs'].append({'stage':'mip','job_id':job,'array':[i],'cpus':8,'mem':'16G','time':'00:30:00','dependency':dep});save()
record['scheduler_verification']={}
for j in record['jobs']:
 s=run([bin+'scontrol','show','job',j['job_id']]);assert 'scaglione-compute-01' in s;record['scheduler_verification'][j['job_id']]=s
save();print(json.dumps({'root':str(root),'jobs':record['jobs'],'tests':test.stderr[-300:]}))

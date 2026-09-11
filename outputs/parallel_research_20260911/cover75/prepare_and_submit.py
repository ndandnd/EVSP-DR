import pathlib,subprocess,json,hashlib
local=pathlib.Path('outputs/parallel_research_20260911/cover75');local.mkdir(parents=True,exist_ok=True)
root='/home/nc437/ladder-lite/covering_complement75_20260911_21fbecb'
worker=pathlib.Path('outputs/meeting_20260910/covering_rerun9/worker.sub').read_text().replace('cov9cg','cov75cg').replace('0-8%9','0-74%50').replace('--mem=96G','--mem=32G').replace('/home/nc437/ladder-lite/covering_rerun9_20260909_21fbecb',root).replace('#SBATCH --requeue','#SBATCH --exclude=scaglione-compute-01,scaglione-cpu-[01-05]\n#SBATCH --requeue')
(local/'worker.sub').write_text(worker)
remote=r'''
import pathlib,json,hashlib,subprocess
root=pathlib.Path(ROOT)
assert not root.exists(), 'Campaign already exists; inspect rather than duplicate'
source=pathlib.Path('/home/nc437/ladder-lite/nested_probability_k2_15_fresh84_20260908_21fbecb')
policy=pathlib.Path('/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md').read_text()
rows=[x.split('\t') for x in (source/'matrix.tsv').read_text().splitlines() if x.strip()]
rows=[r for r in rows if not (int(r[2]) in (5,8,10) and int(r[3]) in (1,3,5))]
rows.sort(key=lambda r:(0 if int(r[2]) in (5,8,10) and int(r[3]) in (2,4,6) else 1 if int(r[2])<=10 else 2,int(r[2]),int(r[3])))
assert len(rows)==75 and len({r[1] for r in rows})==75
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for idx,r in enumerate(rows):
 r[0]=str(idx)
 cache=source/'network_cache'/('M__'+r[1]+'__'+r[7]+'.pkl')
 manifest=pathlib.Path(str(cache)+'.manifest.json'); d=json.loads(manifest.read_text())
 assert sha(pathlib.Path(r[5]))==r[6]
 assert d['identity']['instance_sha256']==r[6]
 assert cache.stat().st_size==d['pickle_bytes']
 r.extend([str(cache),sha(manifest),d['pickle_sha256'],str(d['pickle_bytes'])])
for d in ('cg','logs/cg','logs/freeze','logs/mip','snapshots','records','mip','progress','locks'): (root/d).mkdir(parents=True,exist_ok=True)
(root/'matrix.tsv').write_text('\n'.join('\t'.join(r) for r in rows)+'\n')
(root/'worker.sub').write_text(WORKER)
commit='21fbecba826824c44f897feef038fcf51c532582'; repo='/home/nc437/ladder-lite/execution/'+commit
plan={'campaign':root.name,'purpose':'Complete all six fresh set-covering chains k2..15 without repeating existing nine covering cells','cases':75,'execution_commit':commit,'source_root':str(source),'source_plan_sha256':sha(source/'execution_plan.json'),'matrix_sha256':sha(root/'matrix.tsv'),'worker_sha256':sha(root/'worker.sub'),'physics':{'battery_kwh':240,'charge_kw':240,'reserve_kwh':0,'soc_step_kwh':2.5,'time_step_min':5},'objective':'100000 per route + electricity + 5 per charge start','master_sense':'cover','initialization':'singletons','cg_wall_seconds':28800,'rc_epsilon':0.0001,'resources':{'cpus_per_case':1,'memory_gb':32,'concurrency':50,'partition':'default_partition','exclude':'scaglione-compute-01,scaglione-cpu-[01-05]','rationale':'Prior k15 peak RSS about 11 GiB; 32 GiB provides headroom. Reserve Scaglione CPU-node RAM for dependent MIPs.'},'dependencies':'CG cases independent; downstream freeze and MIP must depend only on corresponding case','proof_reporting':'RMP objective is not full-model lower bound absent valid pricing certificate; finite-pool proof and physical validation separate','policy_sha256':sha(pathlib.Path('/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md'))}
(root/'execution_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
subprocess.run(['bash','-n',str(root/'worker.sub')],check=True)
env={'EVSP_EXECUTION_REPO':repo,'EVSP_EXPECTED_COMMIT':commit,'EVSP_CAMPAIGN_ROOT':str(root),'EVSP_PLAN_SHA256':sha(root/'execution_plan.json'),'EVSP_MATRIX_SHA256':plan['matrix_sha256'],'EVSP_WORKER_SHA256':plan['worker_sha256'],'EVSP_SOURCE_PLAN_SHA256':plan['source_plan_sha256']}
cmd=['/usr/local/slurm/slurm-25.05.5/bin/sbatch','--parsable','--export=ALL,'+','.join(k+'='+v for k,v in env.items()),str(root/'worker.sub')]
r=subprocess.run(cmd,capture_output=True,text=True,check=True)
submission={'cg_job_id':r.stdout.strip(),'command':cmd,'root':str(root),'plan':plan}
(root/'submission.json').write_text(json.dumps(submission,indent=2)+'\n')
print(json.dumps(submission))
'''.replace('pathlib.Path(ROOT)', 'pathlib.Path('+repr(root)+')').replace('write_text(WORKER)', 'write_text('+repr(worker)+')')
r=subprocess.run(['ssh','-S',str(pathlib.Path.home()/'.ssh/evsp-unicorn.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=8','nc437@unicorn-login-01.coecis.cornell.edu','python3 -'],input=remote,text=True,capture_output=True,timeout=120)
print(r.stdout);print(r.stderr);r.check_returncode()
(local/'submission.json').write_text(r.stdout)

import pathlib,json,subprocess,hashlib,time,fcntl,os
R=pathlib.Path('/home/nc437/ladder-lite/strict_packed_20260921');P=pathlib.Path('/home/nc437/ladder-lite/review_strict_c5_20260916');B=pathlib.Path('/usr/local/slurm/slurm-25.05.5/bin');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(a,**kw):return subprocess.check_output(list(map(str,a)),text=True,**kw)
M=json.load(open(P/'manifest.json'));C=next(c for c in M['cases'] if c['id']=='w5_k16_18E2');base=M['execution_commit'];commit=run(['git','-C',R/'code','rev-parse','HEAD']).strip();assert commit=='35770aae2c08e7d5a356cc3b673e67608e5b1036'
audit=json.load(open(R/'model_compatibility_audit.json'))
for f in audit['unchanged_physics_and_validation_files']:
 assert sha(R/'code'/f['file'])==f['sha256'];assert sha(P/'code'/f['file'])==f['sha256']
parent=P/'cases'/C['previous_group_case'];pj=json.load(open(parent/'cg.json'));assert sha(parent/'pool.jsonl')==pj['pool_sha256'];assert sha(P/C['input'])==C['input_sha256'];assert not (P/'cases'/C['id']/'cg.json').exists()
inputs=[P/'manifest.json',P/C['input'],P/'inputs'/f"{C['previous_group_case']}.csv",P/'inputs/w5_k03_18E2.csv',parent/'cg.json',parent/'pool.jsonl']+[R/'code'/f for f in ['data/Ref_dict.csv','data/par_ref_dhd.csv','data/hourly_prices_flat.csv']]
m={'created_unix':time.time(),'original_campaign':str(P),'original_code':str((P/'code').resolve()),'code':str((R/'code').resolve()),'execution_commit':commit,'base_commit':base,'case':C,'input_hashes':{str(p):sha(p) for p in inputs},'objective':'CG combined cost 100000*fleet+charging; final MIP lexicographic fleet/charging','master':'set covering','physics':'unchanged original 18E2 strict parx60 single-factor; not full GIRO constraints','initialization':'exact original k15 inherited pool with trip remapping and full physical replay, plus new trip singletons','algorithm_changes':['packed lazy no-capacity graph','exact deferred JSON tie-key construction'],'compatibility_audit_sha256':sha(R/'model_compatibility_audit.json'),'resource_reason':'old k15:260658942 Python arc objects,153.8GiB; old k16:211.9GiB+6h timeout. Packed arcs are16bytes each and no per-arc Python action dicts;64G leaves substantial headroom above roughly4-6GiB packed-edge storage estimate. This is an estimate pending measured benchmark; graph budget/physics unchanged.','resources':{'benchmark':'4CPU/32G/2h','cg':'8CPU/64G/6h allocation;4h CG including graph','mip':'8CPU/24G/75m allocation;1h MIP'},'dependencies':'benchmark exact fixed-dual agreement -> k16 CG -> saved-pool MIP. True inherited k15 input is complete; no arbitrary throttle. Original blocked descendants preserved and held pending migration.','excluded_nodes':['scaglione-compute-01'],'tooling_sha256':{p.name:sha(p) for p in [R/'worker.py',R/'worker.sh',R/'benchmark_variant.py']}}
with open(R/'launch.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 ledger=R/'jobs.json';jobs=json.load(open(ledger)) if ledger.exists() else {}
 if not jobs:(R/'manifest.json').write_text(json.dumps(m,indent=2))
 else:assert json.load(open(R/'manifest.json'))['execution_commit']==commit
 os.chmod(R/'worker.sh',0o755)
 for task,cpus,mem,wall in [('benchmark',4,'32G','02:00:00'),('cg',8,'64G','06:00:00'),('mip',8,'24G','01:15:00')]:
  if task in jobs:continue
  assert not (R/(task+'_SUBMIT_INTENT.json')).exists(),'inspect uncertain submission'
  cmd=[B/'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--requeue',f'--cpus-per-task={cpus}',f'--mem={mem}',f'--time={wall}',f'--job-name=pack21_{task}','--output='+str(R/'%j.out'),'--error='+str(R/'%j.err')]
  if task!='benchmark':cmd+=['--dependency=afterok:'+jobs['benchmark' if task=='cg' else 'cg']['job']]
  cmd+=[R/'worker.sh',R,task];(R/(task+'_SUBMIT_INTENT.json')).write_text(json.dumps(list(map(str,cmd))));j=run(cmd).strip().split(';')[0];assert j.isdigit();jobs[task]={'job':j,'command':list(map(str,cmd))};ledger.write_text(json.dumps(jobs,indent=2));info=run([B/'scontrol','show','job',j,'--oneliner']);assert 'ExcNodeList=scaglione-compute-01' in info;jobs[task]['verified']=info;ledger.write_text(json.dumps(jobs,indent=2))
 changes=[];previous=json.load(open(P/'jobs.json'))['jobs'];live=set(run([B/'squeue','-h','-u','nc437','-o','%i']).split())
 for key,v in previous.items():
  if '18E2' not in key or int(key.split('_k')[1].split('_')[0])<16:continue
  j=str(v['job_id'])
  if j not in live:continue
  before=run([B/'scontrol','show','job',j,'--oneliner'])
  if 'JobState=PENDING' not in before:continue
  run([B/'scontrol','update','JobId='+j,'Comment=SUPERSEDED_PENDING_AUDIT_strict_packed_20260921']);run([B/'scontrol','hold',j]);after=run([B/'scontrol','show','job',j,'--oneliner']);assert 'JobHeldUser' in after;changes.append({'job':j,'case':key,'before':before,'after':after})
 (R/'original_descendant_holds.json').write_text(json.dumps(changes,indent=2));print(json.dumps({'jobs':jobs,'held_original_descendants':len(changes)},indent=2))

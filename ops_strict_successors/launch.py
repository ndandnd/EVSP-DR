import pathlib,json,subprocess,hashlib,time,fcntl,os
R=pathlib.Path('/home/nc437/ladder-lite/strict_packed_successors_20260921');O=pathlib.Path('/home/nc437/ladder-lite/review_strict_c5_20260916');P=pathlib.Path('/home/nc437/ladder-lite/strict_packed_20260921');B=pathlib.Path('/usr/local/slurm/slurm-25.05.5/bin')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(a):return subprocess.check_output(list(map(str,a)),text=True)
assert 'Exclude' in pathlib.Path('/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md').read_text() or 'exclude' in pathlib.Path('/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md').read_text()
original=json.load(open(O/'manifest.json'));cases=[next(c for c in original['cases'] if c['id']==x) for x in ['w5_k17_18E2','w5_k19_18E2']]
receipt=json.load(open(P/'CG_COMPLETE.json'));status=json.load(open(receipt['result']));assert status['final']['artificial_total']==0;assert sha(receipt['pool'])==receipt['pool_sha256']==status['pool_sha256']
paths=[O/'manifest.json',P/'manifest.json',P/'CG_COMPLETE.json',pathlib.Path(receipt['result']),pathlib.Path(receipt['pool'])]+[O/c['input'] for c in cases]+[O/'inputs/w5_k16_18E2.csv']+[P/'code'/x for x in ['data/Ref_dict.csv','data/par_ref_dhd.csv','data/hourly_prices_flat.csv']]
for c in cases:assert sha(O/c['input'])==c['input_sha256']
m={'created_unix':time.time(),'code':str(P/'code'),'execution_commit':'35770aae2c08e7d5a356cc3b673e67608e5b1036','original_campaign':str(O),'parent_receipt':str(P/'CG_COMPLETE.json'),'cases':cases,'input_hashes':{str(p):sha(p) for p in paths},'tooling_sha256':{f:sha(R/f) for f in ['worker.py','worker.sh','launch.py']},'physics':status['physics'],'master':'set covering','objective':'CG100000*fractional_route_weight+charging_related_cost; MIP fleet then charging','initialization':'entire immediate previous-group CG pool with authenticated bytes/trip remapping and every route physical replay; no GIRO columns','resources':{'validation':'2CPU/8G/15m','cg':'8CPU/16G/6h allocation; original4h including graph construction perattempt','mip':'8CPU/24G/75m allocation; original1h solve perattempt'},'resource_reason':'k16 batchMaxRSS4124768K=3.9337GiB,256trips. Quadratic331trip projection6.576GiB;16GiB leaves2.43x headroom. Keep8threads unchanged. Graph115.38min included in4h budget. k20notqueued:367trip quadratic graphprojection237min nearly consumes4h; awaitk17/k19timing. No compatible strict graph cache exists; baseline caches have different physics and cannot be reused.','dependencies':'validation->k17CG->k19CG; each MIP waits only its ownCG; k18adds no18E2 duty, original previous_group_case is k17->k19','excluded_nodes':['scaglione-compute-01'],'partition':'default_partition','old_held_jobs_mutated':False,'restart_policy':'distinct job/restart outputs;CGcopies latest available atomic pool checkpoint and physically validates onresume;MIPrestarts tree. Everyattempt records4hCG/1hMIP exposure separately.'}
with open(R/'launch.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 ledger=R/'jobs.json';jobs=json.load(open(ledger)) if ledger.exists() else {}
 if not jobs:(R/'manifest.json').write_text(json.dumps(m,indent=2))
 tasks=[('validation',None,2,'8G','00:15:00',None)]
 for i,c in enumerate(cases):
  key=c['id'];dep='validation' if i==0 else cases[i-1]['id']+'__cg';tasks +=[(key+'__cg',key,8,'16G','06:00:00',dep),(key+'__mip',key,8,'24G','01:15:00',key+'__cg')]
 for key,case,cpus,mem,wall,dep in tasks:
  if key in jobs:continue
  intent=R/(key+'_SUBMIT_INTENT.json');assert not intent.exists(),'inspect uncertain submission'
  task='validation' if case is None else key.split('__')[-1]
  cmd=[B/'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--requeue',f'--cpus-per-task={cpus}',f'--mem={mem}',f'--time={wall}',f'--job-name=packS_{key}','--output='+str(R/'%j.out'),'--error='+str(R/'%j.err')]
  if dep:cmd+=['--dependency=afterok:'+jobs[dep]['job'],'--kill-on-invalid-dep=yes']
  cmd +=[R/'worker.sh',R,task]+([case] if case else [])
  intent.write_text(json.dumps(list(map(str,cmd)),indent=2));j=run(cmd).strip().split(';')[0];assert j.isdigit();jobs[key]={'job':j,'command':list(map(str,cmd))};ledger.write_text(json.dumps(jobs,indent=2));s=run([B/'scontrol','show','job',j,'--oneliner']);assert 'ExcNodeList=scaglione-compute-01' in s and 'Partition=default_partition' in s;jobs[key]['verified']=s;ledger.write_text(json.dumps(jobs,indent=2))
 print(json.dumps(jobs,indent=2))

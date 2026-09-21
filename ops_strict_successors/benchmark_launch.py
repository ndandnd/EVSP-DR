import json,subprocess,pathlib,time,hashlib,fcntl
R=pathlib.Path('/home/nc437/ladder-lite/strict_representation_benchmarks_20260921');O=pathlib.Path('/home/nc437/ladder-lite/review_strict_c5_20260916');N=pathlib.Path('/home/nc437/ladder-lite/strict_packed_20260921/code');B=pathlib.Path('/usr/local/slurm/slurm-25.05.5/bin')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(a):return subprocess.check_output(list(map(str,a)),text=True)
old=(O/'code').resolve();cases=[dict(c,input=str(O/c['input'])) for c in json.load(open(O/'manifest.json'))['cases'] if c['id'] in ['w5_k04_18E2','w5_k06_18E2']]
files=[pathlib.Path(c['input']) for c in cases]+[code/f for code in [old,N] for f in ['data/Ref_dict.csv','data/par_ref_dhd.csv','data/hourly_prices_flat.csv']]
for c in cases:assert sha(c['input'])==c['input_sha256']
m={'created_unix':time.time(),'cases':cases,'old_code':str(old),'new_code':str(N),'code_commits':{str(old):'50ceb6c095a580f79f87b53bef536cac31f81963',str(N):'35770aae2c08e7d5a356cc3b673e67608e5b1036'},'input_hashes':{str(p):sha(p) for p in files},'tooling_sha256':{p.name:sha(p) for p in R.glob('*.py')},'physics':'18E2 battery239.01kWh/reserve35.8515kWh,nonPARX240kW,PARX60kW,no sharedcapacity,2.5kWhSOC/5min event,1560min maxstationwait','objective':'five deterministic combined-cost reducedcost queries; noCGcertificate orMIP result','comparison':'original explicit/newexplicit/packed runsequentially on samenode percase, separateprocesses; sameeventlattice/inputhash and5reducedcosts required; all15routes physicallyreplayed','initialization':'noinheritedpool orGIROcolumns; fixeddualvectors with RNGseed20260921','resources':{'w5_k04_18E2':'4CPU/16G/2h','w5_k06_18E2':'4CPU/48G/4h'},'resource_reason':'26tripcontrol measured2.063GiB and9m16s allvariants. Quadratic53tripmemory8.57GiB,90tripmemory24.72GiB;16G/48Gprovide1.87x/1.94xheadroom. Runtimeallocations exceedquadraticallvariant extrapolation. Bothindependenteligibleconcurrently.','excluded_nodes':['scaglione-compute-01'],'partition':'default_partition'}
with open(R/'launch.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ledger=R/'jobs.json';jobs=json.load(open(ledger)) if ledger.exists() else {}
 if not jobs:(R/'manifest.json').write_text(json.dumps(m,indent=2))
 for c in cases:
  key=c['id']
  if key in jobs:continue
  intent=R/(key+'_SUBMIT_INTENT.json');assert not intent.exists()
  mem,wall=('16G','02:00:00') if c['trip_count']==53 else ('48G','04:00:00')
  cmd=[B/'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--requeue','--cpus-per-task=4',f'--mem={mem}',f'--time={wall}',f'--job-name=packB_{key}','--output='+str(R/'%j.out'),'--error='+str(R/'%j.err'),R/'benchmark.sh',R,key]
  intent.write_text(json.dumps(list(map(str,cmd))));j=run(cmd).strip().split(';')[0];jobs[key]={'job':j,'command':list(map(str,cmd))};ledger.write_text(json.dumps(jobs,indent=2));info=run([B/'scontrol','show','job',j,'--oneliner']);assert 'ExcNodeList=scaglione-compute-01' in info;jobs[key]['verified']=info;ledger.write_text(json.dumps(jobs,indent=2))
 print(json.dumps({k:v['job'] for k,v in jobs.items()}))

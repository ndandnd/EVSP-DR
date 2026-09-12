from pathlib import Path
import json,subprocess,hashlib,sys,datetime
BASE=Path('/home/nc437/ladder-lite/overnight_extension_20260912');CODE=BASE/'code';SBATCH='/usr/local/slurm/slurm-25.05.5/bin/sbatch';SCONTROL='/usr/local/slurm/slurm-25.05.5/bin/scontrol'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if '--prepare' in sys.argv:
 plan=json.loads((CODE/'data/overnight_decomposition_20260912/manifest.json').read_text());cases={};groups=[]
 for split in plan['partitions']:
  children=[]
  for g in split['groups']:
   cases[g['id']]={'id':g['id'],'csv':g['csv'],'input_sha256':g['sha256'],'trip_count':g['trip_count'],'target_duties':8,'kind':'decomposition_component_fresh_cover','cg_seconds':14400,'partition_id':split['partition']};groups.append(g['id']);children.append(g['id'])
  key=f"join{split['partition']:02d}";cases[key]={'id':key,'csv':plan['parent']['csv'],'input_sha256':plan['parent']['sha256'],'trip_count':plan['parent']['trip_count'],'target_duties':32,'kind':'decomposition_recombine_and_cross_group_cg','cg_seconds':7200,'children':children}
 starts={1:7,2:9,3:11,4:10,5:11,6:11};multi='/home/nc437/ladder-lite/nested_warm_multichain_p1246_k2_10_20260910_ecb60c1';parents={p:f'{multi}/p{p}' for p in [1,2,4,6]};parents[3]='/home/nc437/ladder-lite/nested_warm_chain_p3_k2_10_20260909_8830a34';parents[5]='/home/nc437/ladder-lite/nested_warm_chain_p5_k2_10_20260910_ecb60c1'
 chains={}
 for p,start in starts.items():
  chain=[]
  for k in range(start,16):
   key=f'w{p}_k{k:02d}';rel=f'scale_ladder/instances/nested_probability_k2_15_20260908/Practice_Custom_DutyUnion_k{k:02d}_p{p:02d}_20260908.csv';parent=f'{parents[p]}/cg/M__k{k-1:02d}_p{p}__warm_cover__event_2p5_event5.json' if k==start else str(BASE/'cases'/f'w{p}_k{k-1:02d}'/'cg.json')
   if k==start:
    assert Path(parent).exists() and Path(parent+'.columns.jsonl').stat().st_size>0
   cases[key]={'id':key,'csv':rel,'input_sha256':sha(CODE/'data'/rel),'kind':'bounded_inherited_cover','target_duties':k,'chain':p,'parent_status':parent,'cg_seconds':14400,'source_cache':f'/home/nc437/ladder-lite/nested_probability_k2_15_fresh84_20260908_21fbecb/network_cache/M__k{k:02d}_p{p}__event_2p5_event5.pkl'};chain.append(key)
  chains[str(p)]=chain
 manifest={'schema':'evsp-dr-overnight-20260912-v1','execution_commit':subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip(),'event_network_sha256':sha(CODE/'src/event_pricer_network.py'),'group_cases':groups,'warm_chains':chains,'cases':cases,'resource_policy':{'partition':'default_partition','exclude':'scaglione-compute-01','independent_array_concurrency':50,'mip_solver_seconds':3600,'mip_stage1_seconds':1800,'mip_requeue':True},'inheritance_policy':{'max_columns':512,'budget_seconds':900,'selection':'longest then cheapest cost per trip then stable trip IDs','partial_import':'validated completed child replays only; no inherited certificate'},'decomposition_input_manifest_sha256':sha(CODE/'data/overnight_decomposition_20260912/manifest.json')}
 assert not (BASE/'manifest.json').exists();(BASE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'cases':len(cases),'groups':len(groups),'warm_cases':sum(map(len,chains.values())),'join_cases':10}));sys.exit()
manifest=json.loads((BASE/'manifest.json').read_text());records=[]
assert '--submit' in sys.argv
assert not (BASE/'jobs.json').exists(),'Refusing duplicate submission'
(BASE/'logs').mkdir(exist_ok=True)
def submit(label,mode,key='',deps=None,array=None,mem='32G',cpus=2,wall='04:45:00',requeue=False):
 args=[SBATCH,'--parsable','--partition=default_partition','--exclude=scaglione-compute-01',f'--cpus-per-task={cpus}',f'--mem={mem}',f'--time={wall}','--requeue' if requeue else '--no-requeue',f'--job-name={label}',f'--output={BASE}/logs/%x_%A_%a.out',f'--error={BASE}/logs/%x_%A_%a.err']
 if deps:args+=['--dependency='+deps]
 if array:args+=['--array='+array]
 args += [str(CODE/'scripts/overnight_worker.sub'),mode]+([key] if key else [])
 job=subprocess.check_output(args,text=True).strip().split(';')[0];assert job.isdigit()
 records.append({'label':label,'mode':mode,'case':key,'array':array,'job_id':job,'dependency':deps,'argv':args,'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});(BASE/'jobs.json').write_text(json.dumps(records,indent=2)+'\n');print(label,job,flush=True);return job
cg=submit('dec40CG','group',array='0-39%50')
mip=submit('dec40MIP','group_mip',array='0-39%50',deps=f'aftercorr:{cg}',mem='16G',cpus=8,wall='02:00:00',requeue=True)
for p,chain in manifest['warm_chains'].items():
 prev=None
 for key in chain:
  job=submit(key,'cg',key,deps=f'afterok:{prev}' if prev else None,mem='96G',cpus=8)
  submit('M'+key,'mip',key,deps=f'afterok:{job}',mem='24G',cpus=8,wall='02:00:00',requeue=True);prev=job
for split in range(10):
 key=f'join{split:02d}';deps='afterok:'+':'.join(f'{mip}_{i}' for i in range(split*4,split*4+4))
 j=submit(key,'join',key,deps=deps,mem='128G',cpus=8,wall='03:00:00')
 submit('M'+key,'mip',key,deps=f'afterok:{j}',mem='64G',cpus=8,wall='02:00:00',requeue=True)
print('SUBMITTED',len(records),'scheduler records')
for job in [cg,mip,records[2]['job_id']]:
 r=subprocess.run([SCONTROL,'show','job',job],capture_output=True,text=True);(BASE/f'scontrol_{job}.txt').write_text(r.stdout)

from pathlib import Path
import subprocess,json,datetime
B=Path('/home/nc437/ladder-lite/graph_recovery_20260912');O=Path('/home/nc437/ladder-lite/overnight_extension_20260912');Q=Path('/home/nc437/ladder-lite/queue_recovery_20260912');S='/usr/local/slurm/slurm-25.05.5/bin/';v=json.loads((B/'manifest.json').read_text());assert not(B/'jobs.json').exists();jobs=[]
recovered=json.loads((Q/'ready_mips_submission.json').read_text())['replacement_map']
def submit(mode,cid,mem,wall,deps=()):
 a=[S+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task='+('2' if mode=='cache' else '8'),'--mem='+mem,'--time='+wall,'--no-requeue','--kill-on-invalid-dep=yes','--job-name=drG_'+cid+'_'+mode,'--output='+str(B/'logs/%x_%j.out'),'--error='+str(B/'logs/%x_%j.err')]
 if deps:a+=['--dependency=afterok:'+':'.join(deps)]
 a+=[str(B/'worker.sub'),mode,cid];r=subprocess.run(a,capture_output=True,text=True,check=True);j=r.stdout.strip().split(';')[0];assert j.isdigit()
 row={'job_id':j,'mode':mode,'case_id':cid,'dependencies':list(deps),'argv':a,'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};jobs.append(row);(B/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
 st=subprocess.check_output([S+'scontrol','show','job',j,'-o'],text=True);assert 'ExcNodeList=scaglione-compute-01' in st and 'Partition=default_partition' in st;row['scontrol']=st;(B/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n');return j
cache={key:submit('cache',key,c['mem'],'12:30:00') for key,c in v['cache_builds'].items()}
cg0=submit('cg','d00_g3','32G','04:45:00',[cache['d00_g3']]);mi0=submit('mip','d00_g3','24G','02:00:00',[cg0]);mapping={'d00_g3':{'cg':cg0,'mip':mi0}}
original=json.loads((O/'manifest.json').read_text())
for cid,c in v['cases'].items():
 if not cid.startswith('join'):continue
 deps=[cache['parent32']]
 for child in c['children']:
  if child=='d00_g3':deps.append(mi0);continue
  mp=O/'cases'/child/'mip_result.json'
  if mp.exists():assert json.loads(mp.read_text()).get('physical_replay_validated') is True;continue
  index=original['group_cases'].index(child);deps.append(recovered[f'949624_{index}'])
 cj=submit('cg',cid,'128G','02:45:00',deps);mj=submit('mip',cid,'64G','02:00:00',[cj]);mapping[cid]={'cg':cj,'mip':mj}
(B/'case_jobs.json').write_text(json.dumps(mapping,indent=2)+'\n');(B/'cache_jobs.json').write_text(json.dumps(cache,indent=2)+'\n')
p=Path('/home/nc437/ladder-lite/mip_preemption_study_20260911/registry.json');reg=json.loads(p.read_text())
for r in jobs:
 if r['mode']=='mip':reg['cases'].append({'job_id':r['job_id'],'case_id':r['case_id'],'cohort':'default_shared_graph_recovery_3600','solver_budget_s':3600,'result_path':str(B/'cases'/r['case_id']/'mip_result.json'),'attempt_tag':'shared_graph_recovery_a01','requeue':False})
t=p.with_suffix('.tmp');t.write_text(json.dumps(reg,indent=2)+'\n');t.replace(p)
print(json.dumps({'cache_jobs':cache,'case_jobs':mapping,'total_jobs':len(jobs)}))

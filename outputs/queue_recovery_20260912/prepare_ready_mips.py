from pathlib import Path
import json,hashlib,subprocess,datetime
B=Path('/home/nc437/ladder-lite/queue_recovery_20260912');B.mkdir(exist_ok=True);(B/'logs').mkdir(exist_ok=True)
O=Path('/home/nc437/ladder-lite/overnight_extension_20260912');D=O/'code/data';M=json.loads((O/'manifest.json').read_text());S='/usr/local/slurm/slurm-25.05.5/bin/'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
indices=[17,18,21,23,25,26,27,28,29];cases=[]
for i in indices:
 case=M['group_cases'][i];r=O/'cases'/case;p=r/'cg.json';v=json.loads(p.read_text());j=Path(v['columns_journal'])
 assert v['csv']==M['cases'][case]['csv']
 assert sha(D/v['csv'])==M['cases'][case]['input_sha256']
 assert v['provenance']['git_commit']==M['execution_commit']
 assert v['final']['artificials']==0 and v['final']['iter']>0
 assert j.stat().st_size>0 and not (r/'mip_result.json').exists()
 acct=subprocess.check_output([S+'sacct','-S','2026-09-12','-j',f'949623_{i}','-X','-P','-n','-o','JobID,State,ExitCode'],text=True)
 assert f'949623_{i}|COMPLETED|0:0' in acct
 ctl=subprocess.check_output([S+'scontrol','show','job',f'949624_{i}','-o'],text=True)
 assert 'JobState=PENDING' in ctl and 'DependencyNeverSatisfied' in ctl
 cases.append({'index':i,'case_id':case,'parent_job_id':f'949623_{i}','replaces_pending_job':f'949624_{i}','cg_stop_reason':v['stop_reason'],'cg_certified':v.get('certificate'),'required_files':[{'path':str(p),'sha256':sha(p)},{'path':str(j),'sha256':sha(j)},{'path':str(D/v['csv']),'sha256':sha(D/v['csv'])}],'accounting':acct,'original_scontrol':ctl,'result_path':str(r/'mip_result.json')})
common=[O/'manifest.json',O/'code/scripts/overnight_worker.py',O/'code/scripts/event_uniform_envelope/gurobi_worker_preflight.sh',D/'hourly_prices_flat.csv',D/'Ref_dict.csv',D/'par_ref_dhd.csv']
v={'schema':'evsp-ready-pool-dependency-recovery-v1','prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':cases,'cg_execution_commit':M['execution_commit'],'mip_execution_commit':'871d057e1067411f09581e37d78f7c1ca43f68bb','scientific_settings':'Unchanged original covering 240/240 event2.5/5 flat, 3600s two-stage MIP, stage1<=1800s, stage2 fleet<=validated incumbent','resources':{'partition':'default_partition','cpus':8,'memory':'24G','slurm_time':'02:00:00','array_concurrency':50,'exclude':'scaglione-compute-01','requeue':True},'common_files':[{'path':str(p),'sha256':sha(p)} for p in common]}
(B/'ready_mips_manifest.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({'ready':[{k:c[k] for k in ['index','case_id','cg_stop_reason']} for c in cases]}))

from pathlib import Path
import subprocess,json
P=Path(__file__).resolve().parent;jobs=json.loads((P/'jobs.json').read_text());trials={k:v['job_id'] for k,v in jobs.items() if not k.endswith('__prepare')}
script='trials='+repr(trials)+r'''
from pathlib import Path
import json,subprocess,time,re
root=Path('/home/nc437/ladder-lite/mip_structure_20260922');receipt=root/'loader_failure_stop.json'
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
rows=[]
for key,jid in trials.items():
 case,arm=key.split('__');dirs=list((root/'results'/case/arm).glob(jid+'_r*'));logs=[str(p/'gurobi.log') for p in dirs if (p/'gurobi.log').exists()];r=subprocess.run(['scontrol','show','job','-o',jid],text=True,capture_output=True);control=r.stdout; match=re.search(r'JobState=(\S+)',control);state=match.group(1) if match else 'NOT_ACTIVE'
 rows.append({'key':key,'job':jid,'state_before':state,'scontrol_before':control,'attempts':[str(p) for p in dirs],'gurobi_logs':logs})
cancelled=[];protected=[]
for row in rows:
 logs=[str(Path(p)/'gurobi.log') for p in row['attempts'] if (Path(p)/'gurobi.log').exists()]
 if logs or row['state_before'] not in ['RUNNING','PENDING']:
  protected.append({'key':row['key'],'job':row['job'],'logs':logs,'reason':'Already reached model build/solve or no longer active; left untouched'});continue
 subprocess.run(['scancel',row['job']],check=True);cancelled.append(row)
cmd=['scancel','(individually rechecked)',*[r['job'] for r in cancelled]];d={'reason':'NPZ indices were decompressed once per column; every canceled trial had no Gurobi log and had not reached model construction/optimize. Necessary implementation recovery only.','created_unix':time.time(),'cancel_command':cmd,'jobs':rows,'cancelled':cancelled,'protected':protected,'running_loading_count':sum(x['state_before']=='RUNNING' for x in cancelled),'pending_count':sum(x['state_before']=='PENDING' for x in cancelled),'scientific_solver_exposure_seconds':0,'scope':'Preserve all attempts; relaunch same cells/settings/pools with single-load decoder'};receipt.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
''';r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','unicorn',"bash -lc 'python3 -'"],input=script,text=True,capture_output=True);print(r.stderr) if r.returncode else None;r.check_returncode();(P/'loader_failure_stop.json').write_text(r.stdout);d=json.loads(r.stdout);print(json.dumps({'stopped_loading':d['running_loading_count'],'replaced_pending':d['pending_count'],'solver_exposure':d['scientific_solver_exposure_seconds']}))

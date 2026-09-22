from pathlib import Path
import subprocess,json,hashlib,base64,datetime
P=Path(__file__).resolve().parent
remote=r'''
from pathlib import Path
import json,hashlib,base64
R=Path('/home/nc437/ladder-lite/strict_packed_successors_20260921'); D=R/'w5_k17_18E2/mip/668432_r0'; C=Path('/home/nc437/ladder-lite/strict_packed_20260921/code'); result=json.loads((D/'result.json').read_text())
files={name:D/name for name in ['result.json','result.json.gurobi.log','solver.log','command.json','COMPLETE.json']}
files.update({'manifest.json':R/'manifest.json','worker.py':R/'worker.py','worker.sh':R/'worker.sh','CG_COMPLETE.json':R/'w5_k17_18E2/CG_COMPLETE.json','run_capacity_speed_event_cg.py':C/'src/run_capacity_speed_event_cg.py','instance.csv':Path(result['provenance']['instance'])})
out={'files':{k:{'remote':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':base64.b64encode(p.read_bytes()).decode()} for k,p in files.items()}}
paths=[Path(result['pool']),Path(result['cg_status']),C/'data/Ref_dict.csv',C/'data/par_ref_dhd.csv',C/'data/hourly_prices_flat.csv']
out['remote_source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
cg=json.loads(Path(result['cg_status']).read_text());out['cg_compact']={k:v for k,v in cg.items() if k not in ['iterations','history','iteration_history','routes','selected_routes'] and len(json.dumps(v))<20000}
sel=set(result['result']['selected_indices']);stage1=set(result['result']['stage1']['selected_indices']);out['selected_pool_rows']=[{'index':i,'route':json.loads(s)} for i,s in enumerate(Path(result['pool']).open()) if i in sel|stage1]
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','unicorn','python3 -'],input=remote,text=True,capture_output=True,check=True)
d=json.loads(r.stdout);records=[]
for name,f in d.pop('files').items():
 b=base64.b64decode(f.pop('data'));assert hashlib.sha256(b).hexdigest()==f['sha256'];(P/'sources'/name).write_bytes(b);records.append({'local':'sources/'+name,**f,'bytes':len(b)})
d['files']=records;d['collected_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(P/'collection.json').write_text(json.dumps(d,indent=2)+'\n')
a=subprocess.run(['ssh','-o','BatchMode=yes','unicorn',"bash -lc 'sacct -S 2026-09-21 -j 668432 --format=JobID,State,ExitCode,Elapsed,MaxRSS,Start,End,AllocCPUS,ReqMem,NodeList -P'"],capture_output=True,text=True,check=True);(P/'scheduler.txt').write_text(a.stdout)
print(json.dumps({'downloaded_files':len(records),'downloaded_bytes':sum(x['bytes'] for x in records),'selected_rows':len(d['selected_pool_rows']),'cg_fields':list(d['cg_compact'])}))

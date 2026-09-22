"""Collect completed small receipts/logs only; retain every attempt, never pools."""
from pathlib import Path
import subprocess,json,base64,hashlib,datetime,csv
P=Path(__file__).resolve().parent;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=P/'collections'/stamp;out.mkdir(parents=True,exist_ok=False)
script=r'''
from pathlib import Path
import json,hashlib,base64
root=Path('/home/nc437/ladder-lite/mip_structure_20260922');artifacts=[];summaries=[]
for attempt in sorted(root.glob('results/*/*/*')):
 if not (attempt/'COMPLETE.json').exists():
  if not (attempt/'result.json').exists() or json.loads((attempt/'result.json').read_text()).get('status')!='BLOCKED':continue
 names=['execution.json','COMPLETE.json']
 if attempt.parent.name=='prepare':
  names+=['native_smoke.json','structure.json','screening.json','fleet_lp.gurobi.log','charging_lp.gurobi.log','fleet_dual_certificate.json.gz','charging_dual_certificate.json.gz']
  meta=json.loads((attempt/'matrix.json').read_text());structure=json.loads((attempt/'structure.json').read_text());screen=json.loads((attempt/'screening.json').read_text());summaries.append({'kind':'prepare','attempt':str(attempt),'case':meta['case'],'matrix_metadata_sha256':hashlib.sha256((attempt/'matrix.json').read_bytes()).hexdigest(),'matrix_file_sha256':meta['matrix_file_sha256'],'matrix_identity_sha256':meta['matrix_identity_sha256'],'rows':meta['rows'],'columns':meta['columns'],'nonzeros':meta['nonzeros'],'strong_start_available':meta['strong_start_available'],'preparation_s':meta['preparation_s'],'dominance_complete':structure['dominance_complete'],'identical_incidence_count':len(structure['identical_incidence_pairs']),'redundant_row_count':len(structure['redundant_row_witnesses']),'safe_dominated_column_count':len(structure['safe_cost_respecting_column_witnesses']),'components':structure['components'],'screening':screen})
 else:
  names+=['result.json','gurobi.log','phase_loading.json','phase_loaded.json','phase_model_ready.json'];result=json.loads((attempt/'result.json').read_text());summaries.append({'kind':'trial','attempt':str(attempt),**{k:result.get(k) for k in ['case','arm','status','reason','optimizer_run','fleet','bound','finite_pool_fleet_proven','target_attained','actual_optimize_wall_s','gurobi_status','matrix_identity_sha256','model_dimensions','effective_parameters','start_buses','first_target_time_s','model_build_wall_s','artifact_loading_s']}})
 for name in names:
  f=attempt/name
  if f.is_file():
   b=f.read_bytes();artifacts.append({'relative':str(f.relative_to(root)),'remote':str(f),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'data':base64.b64encode(b).decode()})
print(json.dumps({'artifacts':artifacts,'summaries':summaries}))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','unicorn','python3 -'],input=script,text=True,capture_output=True,check=True);d=json.loads(r.stdout)
for a in d['artifacts']:
 b=base64.b64decode(a.pop('data'));assert hashlib.sha256(b).hexdigest()==a['sha256'];p=out/a['relative'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
(out/'receipt.json').write_text(json.dumps(d,indent=2)+'\n');flat=[{k:v for k,v in x.items() if not isinstance(v,(dict,list))} for x in d['summaries']];fields=sorted(set(k for x in flat for k in x))
if fields:
 with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
print(json.dumps({'output':str(out),'completed_preparations':sum(s['kind']=='prepare' for s in d['summaries']),'completed_trials':sum(s['kind']=='trial' for s in d['summaries']),'files':len(d['artifacts'])}))

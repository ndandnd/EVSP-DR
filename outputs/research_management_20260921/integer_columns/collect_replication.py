"""Read-only collector; keeps scheduler, proof, target and validation separate."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
w=Path(sys.argv[1]);m=json.loads((w/'manifest.json').read_text())
rows=[]
ledger=list(csv.DictReader((w/'jobs.tsv').open(),delimiter='\t'))
ids=','.join(r['job_id'] for r in ledger)
sched=subprocess.run(['/usr/local/slurm/current/bin/sacct','-j',ids,'-n','-P','--format=JobID,State,Elapsed,MaxRSS,ReqMem'],capture_output=True,text=True,check=True).stdout
(w/'scheduler_latest.txt').write_text(sched)
states={line.split('|')[0]:line.split('|')[1] for line in sched.splitlines() if '|' in line}
for cell in ledger:
 base=w/'results'/cell['case']/f"{cell['arm']}_s{cell['seed']}"
 attempts=sorted(base.glob(cell['job_id']+'_r*'))
 if not attempts: attempts=[None]
 for out in attempts:
  row=dict(cell,scheduler_state=states.get(cell['job_id'],'unknown'),attempt=None if out is None else out.name)
  if out:
   r=json.loads((out/'execution.json').read_text()) if (out/'execution.json').exists() else {}
   row.update({k:r.get(k) for k in ['status','dive_integer_buses','columns_generated','mip_solver_budget_s','actual_charged_dive_plus_solver_s','actual_end_to_end_wall_s','mip_external_overhead_s','source_immutable']})
   if (out/'result.json').exists():
    d=json.loads((out/'result.json').read_text()); t=d.get('two_stage',{})
    for k in ['physical_replay_validated','duplicate_trip_removal_validated','cross_route_charger_capacity_validated']:row[k]=d.get(k)
    row.update(buses=d.get('buses'),fleet_bound=t.get('stage1_bound'),finite_pool_fleet_proven=t.get('fleet_proven'),target_attained=d.get('buses')==m['cases'][cell['case']]['target_k'],global_certificate=None)
    row['result_sha256']=hashlib.sha256((out/'result.json').read_bytes()).hexdigest()
    start=d.get('mip_start',{});row['handoff_validated']=None if not r.get('incumbent_export') else (start.get('validated') and start.get('pool_columns_added')==0 and start.get('pool_columns_replaced')==0)
    row['dive_plus_solver_excess_s']=max(0,r.get('actual_charged_dive_plus_solver_s',0)-m['budget']['shared_dive_wall_plus_solver_s'])
  rows.append(row)
(w/'evidence_latest.json').write_text(json.dumps({'observed_utc':datetime.now(timezone.utc).isoformat(),'rows':rows},indent=1))
fields=list(dict.fromkeys(k for row in rows for k in row))
with (w/'evidence_latest.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
print(json.dumps({'states':{s:sum(r['scheduler_state']==s for r in rows) for s in sorted(set(r['scheduler_state'] for r in rows))},'finished_results':sum('buses' in r for r in rows),'validated_handoffs':sum(r.get('handoff_validated') is True for r in rows)},indent=1))

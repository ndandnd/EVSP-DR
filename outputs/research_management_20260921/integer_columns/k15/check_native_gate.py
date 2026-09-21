"""Exit0 only after a completed native k8 handoff validates in the final MIP."""
import csv,json,math,subprocess,sys
from pathlib import Path
base=Path('/home/nc437/ladder-lite/integer_columns_20260921')
ledger=list(csv.DictReader((base/'jobs.tsv').open(),delimiter='\t'))
ids=[r['job_id'] for r in ledger]
raw=subprocess.run(['/usr/local/slurm/current/bin/sacct','-j',','.join(ids),'-n','-P','--format=JobID,State'],check=True,capture_output=True,text=True).stdout
bad=[line for line in raw.splitlines() if line.split('|')[0] in ids and any(state in line for state in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE'])]
if bad:
 print(json.dumps({'eligible':False,'investigate_k8_failures_first':bad},indent=1));sys.exit(3)
passed=[]
for p in base.glob('results/*/treatment_*/*/execution.json'):
 r=json.loads(p.read_text())
 if r.get('status')!='finished' or not r.get('incumbent_export'):continue
 d=json.loads((p.parent/'result.json').read_text());s=d.get('mip_start',{})
 checks={'own_dive_incumbent':r.get('external_witness_columns_used') is False,
 'validated_handoff':s.get('validated') is True,
 'fleet_matches':s.get('validated_bus_count')==r.get('dive_integer_buses')==r['incumbent_export']['buses'],
 'zero_added_columns':s.get('pool_columns_added')==0,
 'zero_replaced_columns':s.get('pool_columns_replaced')==0,
 'physical_replay':d.get('physical_replay_validated') is True,
 'immutable_source':r.get('source_immutable') is True,
 'no_budget_floor':r.get('mip_solver_budget_s')==max(0,math.floor(3600-r['charged_dive_wall_s'])),
 'charged_accounting':abs(r.get('actual_charged_dive_plus_solver_s',-1)-r['charged_dive_wall_s']-d['runtime_s'])<1e-6,
 'end_to_end_recorded':r.get('actual_end_to_end_wall_s',0)>=r.get('actual_charged_dive_plus_solver_s',1),
 'not_claiming_hard_wallcap':r.get('strict_end_to_end_budget_claim') is False}
 if all(checks.values()):passed.append({'receipt':str(p),'checks':checks,'charged_s':r['actual_charged_dive_plus_solver_s'],'actual_end_to_end_s':r['actual_end_to_end_wall_s']})
print(json.dumps({'eligible':bool(passed),'passed_native_smokes':passed},indent=1))
sys.exit(0 if passed else 2)

"""Reproduce advisor headline counts; keep original and longer budgets separate."""
from pathlib import Path
import csv,json,hashlib,collections
HERE=Path(__file__).resolve().parent
E=HERE.parents[1]
ROOT=E.parents[2]
p=E/'audited_chain_results.csv'
long=ROOT/'outputs/overnight_next_20260914/status_20260916T194843Z/longer_gap_results.csv'
replay=E/'f1/per_case.csv'
read=lambda p:list(csv.DictReader(p.open()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=read(p);assert len(rows)==102
lookup={(int(r['chain']),int(r['target_buses'])):r for r in rows}
audit={r['source_sha256']:r for r in read(replay)}
out=[]
for key,r in lookup.items():
 k=key[1];lb=int(r['integer_fleet_lower_bound']);fleet=int(r['integer_buses']);a=audit[r['mip_sha256']]
 assert a['input_sha256']==r['input_sha256'] and int(a['buses'])==fleet and a['continuous_replay_valid']=='True'
 assert lb in (k,k-1) and abs(float(r['fractional_route_weight'])-lb)<1e-7
 out.append(dict(case=r['case_id'],chain=key[0],target=k,route_weight=float(r['fractional_route_weight']),numerical_integer_fleet_lb=lb,original_buses=fleet,original_numerical_bound_matched=fleet==lb,best_audited_buses=fleet,original_mip_sha256=r['mip_sha256'],best_mip_sha256=r['mip_sha256']))
index={(r['chain'],r['target']):r for r in out}
for r in read(long):
 key=int(r['chain']),int(r['target']);a=audit[r['mip_sha256']];assert a['cohort']=='longer_3h_fleet' and a['input_sha256']==lookup[key]['input_sha256'] and int(a['buses'])==int(r['buses']) and a['continuous_replay_valid']=='True'
 if int(r['buses'])<index[key]['best_audited_buses']:index[key].update(best_audited_buses=int(r['buses']),best_mip_sha256=r['mip_sha256'])
for r in out:r['best_audited_numerical_bound_matched']=r['best_audited_buses']==r['numerical_integer_fleet_lb']
summary={'sources':{str(x):sha(x) for x in [p,long,replay]},'total':len(out),'bound_equals_k':sum(r['numerical_integer_fleet_lb']==r['target'] for r in out),'bound_equals_k_minus_one':sum(r['numerical_integer_fleet_lb']==r['target']-1 for r in out),'original':{'matched_target':sum(r['original_buses']==r['target'] for r in out),'matched_numerical_bound':sum(r['original_numerical_bound_matched'] for r in out),'open_fleet_gaps':sum(not r['original_numerical_bound_matched'] for r in out),'open_among_bound_k':sum(r['numerical_integer_fleet_lb']==r['target'] and not r['original_numerical_bound_matched'] for r in out)},'including_audited_longer':{'matched_target':sum(r['best_audited_buses']==r['target'] for r in out),'matched_numerical_bound':sum(r['best_audited_numerical_bound_matched'] for r in out),'open_fleet_gaps':sum(not r['best_audited_numerical_bound_matched'] for r in out),'k_minus_one_cases_reaching_GIRO_k':sum(r['numerical_integer_fleet_lb']==r['target']-1 and r['best_audited_buses']==r['target'] for r in out)},'numerical_not_exact_arithmetic':True,'scope':'Frozen102original endpoints and26already-audited longer searches; does not include currently running P1 repeats. Event-route model; no shared-capacity proof.'}
assert summary['bound_equals_k']==93 and summary['original']['matched_numerical_bound']==51 and summary['original']['open_fleet_gaps']==51
with (HERE/'per_case.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(HERE/'counts.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2))

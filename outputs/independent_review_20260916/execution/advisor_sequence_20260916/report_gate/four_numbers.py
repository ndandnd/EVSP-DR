"""Read-only, deterministic gate. No network, Slurm commands, or submissions."""
import argparse,csv,hashlib,io,json,math,statistics
from pathlib import Path

FRESH={f'fresh_c{c}_k15_seed{s}' for c in range(1,7) for s in range(3)}
LONG={'warm_w5_k31_seed0_12h'}
VARIANCE={f'warm_w{c}_k32_seed{s}' for c in [1,3,4,5] for s in range(3)}
F6={f'peak{p}_{arm}' for p in ['08','12','18'] for arm in ['cg','fixed']}
FAILURES={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED','REVOKED','COMPLETED'}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def indexed(rows,key):
 result={}
 for row in rows:
  value=row[key]
  if value in result:raise ValueError('Duplicate cell '+str(value))
  result[value]=row
 return result

def evaluate(snapshot,p1_manifest_sha256,f6_manifest_sha256):
 campaigns=snapshot['campaigns'];errors=[]
 for name in ['p1','f6_k5']:
  if not campaigns.get(name,{}).get('collection_ok'):errors.append(name+' collector failed or absent')
 if errors:return {'ready':False,'integrity_errors':errors,'items':None}
 p=campaigns['p1']['data'];f=campaigns['f6_k5']['data']
 if p.get('manifest_sha256')!=p1_manifest_sha256:errors.append('P1 frozen manifest hash mismatch')
 if f.get('manifest_sha256')!=f6_manifest_sha256:errors.append('F6 frozen manifest hash mismatch')
 rows=indexed(p['rows'],'case_id');frows=indexed([r for r in f['rows'] if r['arm']!='original_GIRO'],'case_id')
 if set(rows)!=FRESH|LONG|VARIANCE:errors.append('P1 expected 31 cell identities not present exactly')
 if set(frows)!=F6:errors.append('F6 expected six cell identities not present exactly')
 original=indexed([r for r in f['rows'] if r['arm']=='original_GIRO'],'peak')
 if set(original)!={'peak08','peak12','peak18'}:errors.append('F6 original three tariff rows absent or unexpected')
 checks=indexed(p.get('endpoint_audit',{}).get('checks',[]),'case_id')
 if p.get('endpoint_audit',{}).get('manifest_sha256')!=p1_manifest_sha256:errors.append('P1 endpoint audit absent or manifest mismatch')
 jobs={(r['campaign'],r['case_id']):r['job_id'] for r in snapshot['registered_jobs']}
 queued=set()
 if snapshot['scheduler'].get('squeue',{}).get('ok'):
  queued={line.split('|')[0] for line in snapshot['scheduler']['squeue']['text'].splitlines() if line}
 else:errors.append('squeue unavailable')
 accounts={}
 if snapshot['scheduler'].get('sacct',{}).get('ok'):
  for r in csv.DictReader(io.StringIO(snapshot['scheduler']['sacct']['text']),delimiter='|'):
   jid=r['JobIDRaw']
   if '.' not in jid and (jid not in accounts or r.get('Start','')>=accounts[jid].get('Start','')):accounts[jid]=r
 else:errors.append('sacct unavailable')
 status={}
 def classify(cid,campaign,row):
  jid=jobs.get((campaign,cid));base={'job_id':jid,'case_id':cid,'campaign':campaign}
  endpoint=bool(row.get('result_sha256'))
  if endpoint:
   if not row.get('result_path'):errors.append(cid+' result path missing')
   if campaign=='p1':
    a=checks.get(cid,{})
    if a.get('result_sha256')!=row['result_sha256'] or not a.get('tests') or not all(v is True for v in a['tests'].values()):errors.append(cid+' control/physical audit failed or absent')
    budget=43200 if cid in LONG else 10800
    if a.get('observed_stage1_limit_s')!=budget or a.get('observed_stage2_reserved_s')!=1800:errors.append(cid+' observed stage allowances mismatch')
    if row.get('seed_observed')!=row['seed']:errors.append(cid+' observed seed mismatch')
    if row.get('physical_route_replay') is not True:errors.append(cid+' route physical replay failed or missing')
    if not isinstance(row.get('buses'),(int,float)) or not math.isfinite(row['buses']) or row['buses']<1 or row['buses']!=int(row['buses']):errors.append(cid+' fleet missing/noninteger')
   return dict(base,state='endpoint',result_sha256=row['result_sha256'])
  account=accounts.get(jid,{})
  scheduler_state=account.get('State','').split()[0].rstrip('+') if account else ''
  if jid and jid not in queued and scheduler_state in FAILURES:
   return dict(base,state='terminal_without_scientific_endpoint',scheduler_state=scheduler_state,exit_code=account.get('ExitCode'),interpretation='Censored; not a target miss or proof of infeasibility. No automatic retry.')
  return dict(base,state='pending',scheduler_state=scheduler_state or 'unobserved')
 for cid in sorted(FRESH|LONG|VARIANCE):status[cid]=classify(cid,'p1',rows.get(cid,{}))
 for cid in sorted(F6):status[cid]=classify(cid,'f6_k5',frows.get(cid,{}))
 pending=[x for x in status.values() if x['state']=='pending'];failed=[x for x in status.values() if x['state']=='terminal_without_scientific_endpoint']
 out={'ready':not pending and not errors,'required_solver_cells':37,'endpoint_cells':sum(x['state']=='endpoint' for x in status.values()),'terminal_without_endpoint':failed,'pending_cells':pending,'integrity_errors':errors,'items':None,'submission_policy':'No new submissions, retries, requeues, cancellations or partition changes. A ready report does not authorize new runs.'}
 if not out['ready']:return out
 def record(cid):
  if status[cid]['state']!='endpoint':return status[cid]
  row=rows[cid];a=checks[cid]
  return {k:row.get(k) for k in ['case_id','chain','seed','buses','pool_fleet_bound','fleet_proven_in_pool','physical_route_replay','duplicate_cleanup','result_path','result_sha256','ordered_pool_sha256']}|{'observed_fleet_budget_s':a['observed_stage1_limit_s'],'observed_stage2_reserved_s':a['observed_stage2_reserved_s'],'control_audit_verified':True,'scope':'Finite saved column pool; physical route replay is separate from duplicate cleanup and full-model proof.'}
 fresh=[record(cid) for cid in sorted(FRESH)];valid=[r for r in fresh if 'buses' in r];hits=sum(r['buses']<=15 for r in valid)
 item1={'finding':'F5','question':'Fresh-CG saved k15 pools: target hits after three-hour MIP fleet search','hits':hits,'planned_trials':18,'valid_endpoints':len(valid),'censored_trials':18-len(valid),'hit_fraction_of18':hits/18 if len(valid)==18 else None,'trials':fresh,'interpretation':'18 seed trials over six existing pools, not 18 independent instances; no fresh CG rerun in this MIP experiment.'}
 item2={'findings':['F2','F4'],'question':'C5 k31: 12-hour fleet search plus 30-minute charging stage','outcome':record(next(iter(LONG)))}
 item3={'finding':'F6','question':'k5: 15% reserve, minimum three-minute active charging, zero start fee','tariffs':[],'scope':'240kWh battery/350kW ceiling; controllable power; common aggregate ending minimum280.7833253kWh. Continuous modeled costs, not a global continuous-cost proof. Original invoice is an interval; its uniform-power estimate is not an observed trace.'}
 for peak in ['peak08','peak12','peak18']:
  armrows={}
  for arm in ['cg','fixed']:
   cid=peak+'_'+arm;r=frows[cid]
   armrows[arm]=r if status[cid]['state']=='endpoint' else status[cid]
  matched=all(r.get('matched_five_bus_comparison_eligible') is True and r.get('buses')==5 and r.get('continuous_charging_cost') is not None for r in armrows.values())
  item3['tariffs'].append({'tariff':peak,'original':original[peak],**armrows,'same_fleet_cost_comparison_available':matched,'joint_minus_fixed_cost':armrows['cg']['continuous_charging_cost']-armrows['fixed']['continuous_charging_cost'] if matched else None,'same_achieved_ending_energy':abs(armrows['cg']['ending_kwh']-armrows['fixed']['ending_kwh'])<1e-6 if matched else None})
 item4={'findings':['F2','F5'],'question':'k32 fleet variability over Gurobi seeds0,1,2','variance_definition':'Population variance = sum((fleet-mean)^2)/3 over these three specified solver seeds, in buses squared. Sample variance using divisor2 is separately shown. These are descriptive, not generalization estimates. Missing/failed endpoints suppress both variance statistics.','chains':[]}
 for chain in [1,3,4,5]:
  trials=[record(f'warm_w{chain}_k32_seed{s}') for s in range(3)];values=[r['buses'] for r in trials if 'buses' in r];matched=len(values)==3 and len({r['ordered_pool_sha256'] for r in trials})==1
  item4['chains'].append({'chain':chain,'trials':trials,'same_pool_and_allowances_verified':matched,'population_variance_buses2':statistics.pvariance(values) if matched else None,'sample_variance_buses2':statistics.variance(values) if matched else None,'fleet_range':max(values)-min(values) if matched else None,'censored_seeds':3-len(values)})
 out['items']=[item1,item2,item3,item4];return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args();base=Path(__file__).resolve().parents[2]
 pm=base/'p1/manifest.json';fm=base/'advisor_followup_20260916/f6_k5/manifest.json';result=evaluate(load(a.snapshot),digest(pm),digest(fm))
 result['sources']={str(Path(a.snapshot).resolve()):digest(a.snapshot),str(pm):digest(pm),str(fm):digest(fm),str(Path(__file__).resolve()):digest(__file__)}
 o=Path(a.out);o.mkdir(parents=True,exist_ok=True);(o/'readiness.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'ready':result['ready'],'endpoints':result.get('endpoint_cells'),'pending':len(result.get('pending_cells',[])),'terminal_without_endpoint':len(result.get('terminal_without_endpoint',[])),'integrity_errors':result['integrity_errors'],'output':str(o/'readiness.json')}))
if __name__=='__main__':main()

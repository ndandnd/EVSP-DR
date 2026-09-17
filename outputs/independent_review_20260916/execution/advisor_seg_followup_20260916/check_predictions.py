"""Compare immutable reviewer predictions against read-only six-arm snapshots.

No scheduler mutations, solvers, or pool reads. An agreement is not a causal proof.
"""
import argparse,datetime,hashlib,json,math,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parent/'action3_full_20260916'
def finite(x):return isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x)
def close(x,y,tol):return finite(x) and abs(x-y)<=tol

def classify_lp(prediction,components,tol):
    out={'status':'PENDING','route_weight':None,'components':{},'scope':'fractional route weight at weighted-objective optimum, not a fleet-only bound'}
    for name,d in components.items():
        if not d:out['components'][name]={'state':'PENDING'};continue
        final=d.get('final',{});rc=d.get('terminal_exact_min_reduced_cost')
        certified=(d.get('certified_rc_optimal') is True and d.get('stop_reason')=='exact_nonnegative_reduced_cost' and finite(final.get('artificial_total')) and final['artificial_total']<=1e-7 and finite(rc) and rc>=-1e-5)
        out['components'][name]={'state':'CERTIFIED' if certified else 'UNCERTIFIED','route_weight':final.get('route_weight'),'weighted_objective':final.get('objective'),'stop_reason':d.get('stop_reason'),'certificate_claim':d.get('certified_rc_optimal'),'certificate_consistent':certified,'terminal_exact_min_reduced_cost':rc,'artificial_total':final.get('artificial_total'),'endpoint_sha256':d.get('_endpoint_sha256')}
    if any(x['state']=='PENDING' for x in out['components'].values()):return out
    weights=[d.get('route_weight') for d in out['components'].values()]
    if not all(finite(x) for x in weights):out['status']='UNRESOLVED_MISSING_WEIGHT';return out
    out['route_weight']=sum(weights)
    totalmatch=any(close(out['route_weight'],x,tol) for x in prediction['predicted_lp_route_weight'])
    groupmatch=all(name in out['components'] and close(out['components'][name].get('route_weight'),weight,tol) for name,weight in prediction.get('predicted_group_weights',{}).items())
    out['numeric_total_agreement']=totalmatch;out['numeric_group_agreement']=groupmatch
    if not all(d['state']=='CERTIFIED' for d in out['components'].values()):out['status']='UNRESOLVED_UNCERTIFIED';return out
    out['status']='VERIFIED_NUMERIC_PREDICTION' if totalmatch and groupmatch else 'REFUTED_CERTIFIED_NUMERIC_PREDICTION'
    return out

def classify_mip(prediction,components):
    out={'status':'PENDING','fleet':None,'components':{},'proof_scope':'finite saved column pools only; neither certified weighted CG nor pool MIP alone proves global fleet optimality','independent_physical_replay':'NOT_REPORTED_BY_THIS_DRIVER'}
    for name,d in components.items():
        if not d:out['components'][name]={'state':'PENDING'};continue
        result=d.get('result',{});s1=result.get('stage1',{});covered=d.get('duplicate_service_audit',{}).get('all_trips_covered') is True
        cap=d.get('physical_station_capacity_audit',{}).get('valid');physical=result.get('has_solution') is True and covered and (d.get('capacity_enforced_in_mip') is False or cap is True)
        fleet=result.get('fleet');proved=physical and s1.get('fleet_proven') is True and s1.get('validated_incumbent') is True and s1.get('incumbent_fleet')==fleet and finite(s1.get('fleet_integer_lower_bound')) and s1['fleet_integer_lower_bound']>=fleet
        out['components'][name]={'state':'POOL_PROVED' if proved else 'INCUMBENT_ONLY' if physical else 'UNRESOLVED_PHYSICAL_OR_NO_SOLUTION','fleet':fleet,'finite_pool_proved':proved,'integer_pool_bound':s1.get('fleet_integer_lower_bound'),'stage1_status':s1.get('status'),'stage2_status':result.get('stage2',{}).get('status'),'coverage_valid':covered,'capacity_enforced':d.get('capacity_enforced_in_mip'),'capacity_sweep_valid':cap,'physical_scope':'coverage plus capacity when imposed; individual routes feasible by event-model construction, not independent full-route replay','model_incumbent_valid':physical,'endpoint_sha256':d.get('_endpoint_sha256')}
    if any(x['state']=='PENDING' for x in out['components'].values()):return out
    if not all(x.get('model_incumbent_valid') and finite(x.get('fleet')) for x in out['components'].values()):out['status']='UNRESOLVED_PHYSICAL_OR_NO_SOLUTION';return out
    out['fleet']=sum(x['fleet'] for x in out['components'].values());out['finite_pool_proved']=all(x['finite_pool_proved'] for x in out['components'].values())
    target=prediction.get('predicted_integer_fleet')
    if target is None:out['status']='NO_INTEGER_PREDICTION';return out
    match=(out['fleet']>=int(target[2:])) if target.startswith('>=') else out['fleet']==int(target)
    out['numeric_incumbent_agreement']=match
    if out['finite_pool_proved']:out['status']='VERIFIED_FINITE_POOL_PREDICTION' if match else 'REFUTED_FINITE_POOL_PREDICTION'
    else:out['status']='UNRESOLVED_INCUMBENT_NOT_PROOF'
    return out

def check(snapshot,predictions):
    cm=json.loads((CAMPAIGN/'continuation_manifest.json').read_text());m=json.loads((CAMPAIGN/'manifest.json').read_text())
    rows=[]
    for pred in predictions['predictions']:
        arm=pred['arm'];specs={name:spec for name,spec in cm['cases'].items() if spec['arm']==arm};cg={};mip={};settings=[]
        for case,spec in specs.items():
            label=spec['group'] or arm
            c=snapshot.get('stages',{}).get('continuation/'+case+'/cg.json');v=snapshot.get('stages',{}).get('continuation/'+case+'/mip.json')
            cg[label]=c;mip[label]=v
            for endpoint in [c,v]:
                if not endpoint:continue
                prov=endpoint.get('provenance',{})
                if prov.get('git_commit')!=m['execution_commit'] or prov.get('instance_sha256')!=spec['input_sha256']:settings.append(case+': execution/input provenance mismatch or missing')
            if c:
                ph=c.get('physics',{})
                for key in ['battery_kwh','reserve_kwh','parx_kw']:
                    if ph.get(key)!=m['physics'][arm][key]:settings.append(case+': '+key+' mismatch or missing')
                for key,expected in [('non_parx_kw',240),('soc_step_kwh',2.5),('capacity_enforced',False),('terminal_soc_constraint','reserve_only')]:
                    if ph.get(key)!=expected:settings.append(case+': '+key+' mismatch or missing')
        lp=classify_lp(pred,cg,predictions['tolerance_route_weight']);ip=classify_mip(pred,mip)
        if settings:lp['status']='UNRESOLVED_SETTINGS_OR_PROVENANCE';ip['status']='UNRESOLVED_SETTINGS_OR_PROVENANCE'
        rows.append({'arm':arm,'prediction':pred,'replay':snapshot.get('replay',{}).get(arm,{}),'settings_issues':settings,'lp':lp,'mip':ip})
    return {'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshot_collected_utc':snapshot.get('collected_utc'),'predictions_sha256':hashlib.sha256((HERE/'predictions.json').read_bytes()).hexdigest(),'source_readme_sha256':predictions['source_readme_sha256'],'arms':rows,'qualification':'Numeric agreement is not causal proof. Certified route weight belongs to one weighted-objective LP optimum and need not be unique. Finite-pool integer proof does not establish the full-model integer minimum. Replay progress from completed shards excludes partial work.','scheduler':snapshot.get('squeue'),'collection_errors':snapshot.get('errors',[])}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path);parser.add_argument('--refresh',action='store_true');parser.add_argument('--out',type=Path);args=parser.parse_args()
    predictions=json.loads((HERE/'predictions.json').read_text())
    assert hashlib.sha256((HERE/'reviewer_source/README.md').read_bytes()).hexdigest()==predictions['source_readme_sha256'],'frozen reviewer source changed'
    if args.refresh:subprocess.run([sys.executable,str(CAMPAIGN/'collect.py')],check=True,stdout=subprocess.DEVNULL)
    source=args.snapshot or max((CAMPAIGN/'snapshots').glob('*/status.json'));report=check(json.loads(source.read_text()),predictions);report['snapshot_path']=str(source);report['snapshot_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    out=args.out or HERE/'comparisons'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Reviewer prediction check', '', 'Predictions remain unchanged. Current endpoints are compared separately from replay progress.','', '| Arm | Replay shards / 125 | CG weight | CG prediction | MIP fleet | MIP prediction |','|---|---:|---:|---|---:|---|']
    for row in report['arms']:
        lines.append('| '+ ' | '.join(map(str,[row['arm'],row['replay'].get('completed_shards','?'),row['lp']['route_weight'],row['lp']['status'],row['mip']['fleet'],row['mip']['status']]))+' |')
    lines+=['',report['qualification'],'','See comparison.json for exact stop reasons, certificates, group splits, physical-validation scope and source hashes.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n');print(str(out/'comparison.md'))
if __name__=='__main__':main()

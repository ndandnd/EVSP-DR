"""Classify archived LP support only. No solver imports/calls or remote writes."""
import collections,csv,gzip,hashlib,json,math,statistics
from pathlib import Path
B=Path(__file__).resolve().parent;ROOT=B.parents[3];TABLE=B.parent/'audited_chain_results.csv';MASTER=ROOT/'outputs/chain_extension_20260913/inputs/sources/Par_VehicleDetails_Updated.csv'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def csvrows(p):return list(csv.DictReader(Path(p).open()))
def writecsv(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def classify(routes,mapping,trip_duty):
 rows=[]
 for ordinal,r in enumerate(routes):
  value=float(r['value']);assert math.isfinite(value) and value>0
  tids=r['trips'];assert len(tids)==len(set(tids)) and tids
  duties=sorted({trip_duty[mapping[int(t)]] for t in tids})
  groups=sorted({'18E1' if d.startswith('134') else '18E2' for d in duties})
  rows.append(dict(route_ordinal=ordinal,lambda_value=value,route_cost=r['cost'],trip_count=len(tids),giro_duty_count=len(duties),groups='/'.join(groups),mixed_group=len(groups)==2,original_duties=';'.join(duties)))
 return rows

def main():
 table=csvrows(TABLE);assert len(table)==102;byid={r['case_id']:r for r in table};assert len(byid)==102
 saved=json.loads(gzip.decompress((B/'saved_final_supports.json.gz').read_bytes()));assert len(saved)==102
 master=csvrows(MASTER);assert sha(MASTER)=='6b46acce8b0870aff967c73aac372b90873ed32a6e424e55b851e4b8676ab57f'
 regular=[r for r in master if r['Identifier']=='Regular'];trip_duty={int(r['Ordered_Trip_ID']):r['VehicleTask'] for r in regular};assert len(trip_duty)==len(regular)==987
 assert all(r['VehicleTask'].startswith(('133','134')) for r in regular)
 assert all((int(float(r['Route']))==5021)==r['VehicleTask'].startswith('134') for r in regular)
 action=B.parent/'advisor_sequence_20260916/single_factor_pilot/prepared/inputs/groups.json';actionmap=json.loads(action.read_text())
 assert all(trip_duty[r['ordered_trip_id']]==r['original_duty'] and ('18E1' if r['original_duty'].startswith('134') else '18E2')==r['group'] for r in actionmap.values())
 cases=[];allroutes=[];inputhashes={}
 for d in saved:
  t=byid[d['case_id']];cid=d['case_id'];k=int(t['target_buses']);mass=float(t['fractional_route_weight']);cohort='k_minus_1' if abs(mass-(k-1))<1e-6 else 'k' if abs(mass-k)<1e-6 else 'other';assert cohort!='other'
  base=dict(case_id=cid,chain=int(t['chain']),target_k=k,cohort=cohort,trip_count=int(t['trip_count']),recorded_fractional_route_weight=mass,weighted_lp_objective=float(t['weighted_lp_objective']),cg_certificate=t['cg_pricing_certificate']=='True',cg_stop_reason=t['cg_stop_reason'],source_path=d['path'],source_file_sha256=d.get('file_sha256'),input_sha256=t['input_sha256'])
  try:
   assert d['file_sha256']==t['bound_source_sha256'],'CG byte hash changed since reviewed endpoint'
   assert d['instance_sha256']==t['input_sha256'] and d['execution_commit']==t['recorded_cg_commit'],'input/execution mismatch'
   fl=d['final_lp'];final=d['final'];assert fl and isinstance(fl.get('positive_routes'),list),'missing saved lambdas'
   assert fl['iteration']==final['iter']==int(t['priced_iteration']),'saved support not matching declared final priced iteration'
   assert fl['pool_columns']==int(t['lp_endpoint_pool_columns']),'pool count mismatch'
   assert abs(fl['route_weight']-mass)<1e-9 and abs(fl['objective']-float(t['weighted_lp_objective']))<1e-6,'support endpoint scalar mismatch'
   pricing_aligned=abs(fl['objective']-final['lp_obj'])<1e-6 and abs(fl['route_weight']-final['route_weight'])<1e-9
   if fl['source']=='last_good_iterate' or d['certified_rc_optimal']:assert pricing_aligned,'saved support does not match its claimed priced/certified iterate'
   else:assert fl['source']=='final_pool_resolve' and fl['pool_columns']==d['columns'],'unrecognized endpoint fallback'
   inp=ROOT/'outputs'/t['campaign']/'inputs'/f'{cid}.csv';assert sha(inp)==t['input_sha256'],'input bytes mismatch';inputhashes[str(inp.relative_to(ROOT))]=sha(inp)
   rr=csvrows(inp);mapping={int(r['count_trip_id']):int(r['Ordered_Trip_ID']) for r in rr};assert len(mapping)==len(rr)==int(t['trip_count']);assert set(mapping)==set(d['trip_ids'])
   rout=classify(fl['positive_routes'],mapping,trip_duty);positive_mass=math.fsum(r['lambda_value'] for r in rout);objective=math.fsum(r['lambda_value']*r['route_cost'] for r in rout)
   assert abs(positive_mass-mass)<=2e-6 and abs(objective-fl['objective'])<=.21,'saved positive support does not reconstruct LP scalars within declared numerical allowance'
   assert abs(fl['artificial_total'])<=1e-8,'artificial coverage present'
   cover={t:[] for t in mapping}
   for r in fl['positive_routes']:
    for tid in r['trips']:cover[tid].append(r['value'])
   mincover=min(math.fsum(v) for v in cover.values());assert mincover>=1-2e-6,'positive support fails covering check'
   mixed=[r for r in rout if r['mixed_group']];mixedmass=math.fsum(r['lambda_value'] for r in mixed);material=[r for r in rout if r['lambda_value']>1e-7];materialmixed=[r for r in material if r['mixed_group']]
   final_pool=fl['source']=='final_pool_resolve' and fl['pool_columns']==d['columns']
   base.update(status='verified_saved_endpoint',matches_last_priced_rmp=pricing_aligned,last_priced_rmp_objective=final['lp_obj'],endpoint_minus_last_priced_objective=fl['objective']-final['lp_obj'],endpoint_source=fl['source'],endpoint_is_final_pool=final_pool,endpoint_iteration=fl['iteration'],endpoint_pool_columns=fl['pool_columns'],later_unsolved_columns=d['columns']-fl['pool_columns'],final_lp_sha256=canonical_sha(fl),positive_support_sha256=canonical_sha(fl['positive_routes']),saved_positive_route_count=len(rout),mixed_positive_route_count=len(mixed),positive_route_count_above_1e_7=len(material),mixed_route_count_above_1e_7=len(materialmixed),total_saved_positive_lambda=positive_mass,mixed_group_lambda=mixedmass,pure_group_lambda=positive_mass-mixedmass,mixed_lambda_fraction=mixedmass/positive_mass,mixed_group_lambda_above_1e_7=math.fsum(r['lambda_value'] for r in materialmixed),positive_minus_reported_signed_mass=positive_mass-mass,positive_objective_minus_recorded=objective-fl['objective'],minimum_trip_coverage_from_positive_support=mincover,max_recorded_bound_violation=fl.get('max_bound_violation'),end_to_end_group_assignment_verified=True)
   allroutes.extend(dict(case_id=cid,cohort=cohort,**r) for r in rout)
  except Exception as exc:base.update(status='missing_or_unaligned',reason=type(exc).__name__+': '+str(exc))
  cases.append(base)
 cases.sort(key=lambda r:(r['target_k'],r['chain']));writecsv(B/'per_case.csv',cases)
 tmp=B/'per_route.csv';writecsv(tmp,allroutes);(B/'per_route.csv.gz').write_bytes(gzip.compress(tmp.read_bytes(),mtime=0));tmp.unlink()
 summaries=[]
 for endpoint_filter in ['all_table_endpoints','final_pool_resolve_only','pricing_certified_only','size_matched_k27_to32']+[f'exact_k_{k}' for k in range(27,33)]:
  for cohort in ['k_minus_1','k']:
   members=[r for r in cases if r['cohort']==cohort]
   # Explicit filtering keeps endpoint quality separate from the scientific cohort.
   valid=[r for r in members if r['status']=='verified_saved_endpoint']
   if endpoint_filter=='final_pool_resolve_only':valid=[r for r in valid if r['endpoint_is_final_pool']]
   if endpoint_filter=='pricing_certified_only':valid=[r for r in valid if r['cg_certificate']]
   if endpoint_filter=='size_matched_k27_to32':valid=[r for r in valid if 27<=r['target_k']<=32]
   if endpoint_filter.startswith('exact_k_'):valid=[r for r in valid if r['target_k']==int(endpoint_filter.rsplit('_',1)[1])]
   s=dict(endpoint_filter=endpoint_filter,cohort=cohort,cohort_cases=len(members),included_cases=len(valid),missing_or_unaligned_cases=sum(r['status']!='verified_saved_endpoint' for r in members),cases_with_any_material_mixing=sum(r['mixed_group_lambda_above_1e_7']>0 for r in valid))
   for key in ['mixed_group_lambda','total_saved_positive_lambda','mixed_lambda_fraction','saved_positive_route_count','mixed_positive_route_count']:
    vals=[r[key] for r in valid];s.update({key+'_sum':math.fsum(vals),key+'_mean':statistics.mean(vals) if vals else None,key+'_min':min(vals) if vals else None,key+'_max':max(vals) if vals else None})
   s['pooled_mixed_lambda_fraction']=math.fsum(r['mixed_group_lambda'] for r in valid)/math.fsum(r['total_saved_positive_lambda'] for r in valid) if valid else None;summaries.append(s)
 writecsv(B/'group_summary.csv',summaries)
 provenance=dict(finding='F4',secondary_finding='F2',method='Posthoc observational classification of saved final_lp.positive_routes; no optimization or reconstruction of missing lambdas.',solver_calls=0,source_table_sha256=sha(TABLE),master_sha256=sha(MASTER),action3_map_sha256=sha(action),archive_sha256=sha(B/'saved_final_supports.json.gz'),script_sha256=sha(__file__),input_hashes=inputhashes,case_count=len(cases),verified_saved_endpoints=sum(r['status']=='verified_saved_endpoint' for r in cases),missing_or_unaligned=[r for r in cases if r['status']!='verified_saved_endpoint'],source_mapping='PDF assigns route21 to18E1 and local routes to18E2; all134 regular source duties are Route5021 and all133 are local55xx. Prefix mapping inferred from source rows, independently matches action3 map.',positive_support_numerics='Serializer saves value>0 only. Tiny nonpositive solver values are omitted. All positive weights retained for mass totals; material route counts also use1e-7. Signed route mass and reconstruction discrepancy are reported, never silently rounded.',causal_scope='Mixed-group support demonstrates use of the homogeneous compatibility relaxation in this saved solution. It does not prove mixing caused a k−1 bound, is necessary, or excludes an unmixed alternate optimum. Chains are nested and cohorts differ in k/composition; no independence or significance claim.',summaries=summaries)
 (B/'summary.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps({'verified':provenance['verified_saved_endpoints'],'missing':provenance['missing_or_unaligned'],'summary':summaries[:2]},indent=2))
if __name__=='__main__':main()

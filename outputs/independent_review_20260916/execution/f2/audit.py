#!/usr/bin/env python3
"""Read-only selected-route nearest-GIRO-duty audit, frozen 20260916T194843Z cohort.
No optimizer is called. Remote artifacts are read only; output stays beside this script.
"""
import csv,json,hashlib,subprocess,datetime,statistics,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
TABLES=ROOT/'outputs/overnight_next_20260914/status_20260916T194843Z'
MASTER=ROOT/'outputs/chain_extension_20260913/inputs/sources/Par_VehicleDetails_Updated.csv'
def sha(b):return hashlib.sha256(b).hexdigest()
def readcsv(p):return list(csv.DictReader(p.open()))
def writecsv(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
primary=readcsv(TABLES/'all_chain_extension_results.csv');longer=readcsv(TABLES/'longer_gap_results.csv')
assert len(primary)==102 and len(longer)==26
primary_by_id={r['case_id']:r for r in primary}
requests=[]
for cohort,rows in [('original_1h',primary),('longer_3h_fleet',longer)]:
 for r in rows:
  requests.append(dict(cohort=cohort,case_id=r['case_id'],path=r['mip_path'],sha256=r['mip_sha256']))
remote='''import json,hashlib,sys,datetime
from pathlib import Path
req=json.loads(sys.stdin.read());out=[]
for q in req:
 b=Path(q['path']).read_bytes();h=hashlib.sha256(b).hexdigest();assert h==q['sha256'],(q['case_id'],h,q['sha256'])
 r=json.loads(b)
 out.append(dict(**q,buses=r['buses'],instance=r['instance'],input_sha256=r['physical_pool_audit']['input_hashes']['instance_sha256'],selected_route_set_sha256=r['selected_route_set_sha256'],source_journal_sha256=r['source_journal_sha256'],routes=[dict(index=i,trips=v['trips'],origin=v.get('origin'),inherited_source_ordered_trip_ids=v.get('inherited_source_ordered_trip_ids')) for i,v in enumerate(r['selected_routes'])]))
print(json.dumps(dict(collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=out)))
'''
import shlex
if not (OUT/'selected_trip_sets.json').exists():
 p=subprocess.run(['ssh','-S','/Users/nadan/.ssh/evsp-unicorn.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','nc437@unicorn-login-01.coecis.cornell.edu','python3 -c '+shlex.quote(remote)],input=json.dumps(requests),text=True,capture_output=True,check=True)
 (OUT/'selected_trip_sets.json').write_text(p.stdout)
collection=json.loads((OUT/'selected_trip_sets.json').read_text())
master=readcsv(MASTER);trip_duty={int(r['Ordered_Trip_ID']):r['VehicleTask'] for r in master if r['Identifier']=='Regular'}
assert len(trip_duty)==987
full_duties=collections.defaultdict(set)
for t,d in trip_duty.items():full_duties[d].add(t)
route_rows=[];case_rows=[];inputs={}
for res in collection['results']:
 cid=res['case_id'].replace('_longmip','');p=primary_by_id[cid];k=int(p['target_buses']);lp=float(p['fractional_route_weight']);group='LP_weight_k' if abs(lp-k)<1e-6 else 'LP_weight_k_minus_1' if abs(lp-k+1)<1e-6 else 'other'
 assert group!='other'
 inp=ROOT/'outputs/chain_extension_20260913/inputs'/f'{cid}.csv';b=inp.read_bytes();h=sha(b);assert h==p['input_sha256']==res['input_sha256'];inputs[str(inp.relative_to(ROOT))]=h
 rr=readcsv(inp);mapping={int(r['count_trip_id']):int(r['Ordered_Trip_ID']) for r in rr};assert len(mapping)==len(rr)
 input_t=set(mapping.values());duties={d:s for d,s in full_duties.items() if s&input_t}
 assert len(duties)==k and set.union(*duties.values())==input_t, (cid,'partial GIRO duty or wrong k')
 counts=collections.Counter();scores=[];weighted=0;total=0;matches=0
 for route in res['routes']:
  tids=route['trips'];assert len(tids)==len(set(tids));ts={mapping[t] for t in tids};assert ts
  counts.update(ts)
  overlaps=sorted([(len(ts&s)/len(ts|s),d,len(ts&s),len(s)) for d,s in duties.items()],key=lambda v:(-v[0],v[1]))
  score,d,inter,ds=overlaps[0]; scores.append(score);total+=len(ts);weighted+=score*len(ts);matches+=score==1
  route_rows.append(dict(cohort=res['cohort'],case_id=cid,chain=int(p['chain']),target=k,lp_weight_group=group,selected_buses=res['buses'],route_index=route['index'],route_trip_count=len(ts),nearest_giro_duty=d,giro_duty_trip_count=ds,intersection_trip_count=inter,union_trip_count=len(ts)+ds-inter,jaccard=score,route_share_in_nearest_duty=inter/len(ts),nearest_duty_share_on_route=inter/ds,distinct_giro_duties_on_route=len({trip_duty[t] for t in ts}),exact_giro_tripset=score==1,origin=route['origin'],ordered_trip_ids=json.dumps(sorted(ts)),mip_sha256=res['sha256'],input_sha256=h))
 assert set(counts)==input_t and len(scores)==res['buses']
 case_rows.append(dict(cohort=res['cohort'],case_id=cid,chain=int(p['chain']),target=k,lp_weight=lp,lp_weight_group=group,cg_certified=p['cg_pricing_certificate'],selected_buses=res['buses'],excess_buses_over_target=res['buses']-k,mean_route_jaccard=statistics.mean(scores),median_route_jaccard=statistics.median(scores),trip_occurrence_weighted_jaccard=weighted/total,min_route_jaccard=min(scores),max_route_jaccard=max(scores),exact_giro_tripsets=matches,routes_at_least_09=sum(s>=.9 for s in scores),routes_at_least_08=sum(s>=.8 for s in scores),selected_routes=len(scores),input_trips=len(input_t),trip_occurrences=total,duplicated_trip_ids=sum(n>1 for n in counts.values()),extra_trip_occurrences=total-len(input_t),mip_path=res['path'],mip_sha256=res['sha256'],input_sha256=h,selected_route_set_sha256=res['selected_route_set_sha256'],source_journal_sha256=res['source_journal_sha256']))
writecsv('per_route.csv',route_rows);writecsv('per_case.csv',case_rows)
summaries=[]
for cohort in ['original_1h','longer_3h_fleet']:
 for group in ['LP_weight_k','LP_weight_k_minus_1']:
  cases=[r for r in case_rows if r['cohort']==cohort and r['lp_weight_group']==group];routes=[r for r in route_rows if r['cohort']==cohort and r['lp_weight_group']==group]
  summaries.append(dict(cohort=cohort,lp_weight_group=group,cases=len(cases),selected_routes=len(routes),mean_case_jaccard=statistics.mean(r['mean_route_jaccard'] for r in cases),median_case_jaccard=statistics.median(r['mean_route_jaccard'] for r in cases),min_case_jaccard=min(r['mean_route_jaccard'] for r in cases),max_case_jaccard=max(r['mean_route_jaccard'] for r in cases),mean_route_jaccard=statistics.mean(r['jaccard'] for r in routes),median_route_jaccard=statistics.median(r['jaccard'] for r in routes),exact_giro_tripsets=sum(r['exact_giro_tripset'] for r in routes),routes_at_least_09=sum(r['jaccard']>=.9 for r in routes),routes_at_least_08=sum(r['jaccard']>=.8 for r in routes),mean_distinct_duties_per_route=statistics.mean(r['distinct_giro_duties_on_route'] for r in routes)))
writecsv('group_summary.csv',summaries)
provenance=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),review_sha256=sha((ROOT/'outputs/independent_review_20260916/REVIEW.md').read_bytes()),master_path=str(MASTER),master_sha256=sha(MASTER.read_bytes()),source_tables={str((TABLES/name).relative_to(ROOT)):sha((TABLES/name).read_bytes()) for name in ['all_chain_extension_results.csv','longer_gap_results.csv']},input_files_sha256=inputs,outputs_sha256={name:sha((OUT/name).read_bytes()) for name in ['selected_trip_sets.json','per_route.csv','per_case.csv','group_summary.csv']},audit_script_sha256=sha(Path(__file__).read_bytes()),validation='All 128 remote MIP file hashes match frozen source CSV. All 102 instance hashes match remote physical_pool_audit and frozen source CSV. Each mapped trip belongs to exactly one original GIRO duty; every instance is exactly a union of k full duties; every selected collection covers its full instance. Distinct MIP routes and duplicate trip occurrences are retained.')
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(summaries,indent=2))

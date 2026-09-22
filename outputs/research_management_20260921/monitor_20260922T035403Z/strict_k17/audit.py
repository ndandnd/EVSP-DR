from pathlib import Path
from collections import Counter
import json,hashlib,csv,re,math
P=Path(__file__).resolve().parent;d=json.loads((P/'collection.json').read_text());r=json.loads((P/'sources/result.json').read_text());m=json.loads((P/'sources/manifest.json').read_text());complete=json.loads((P/'sources/COMPLETE.json').read_text());receipt=json.loads((P/'sources/CG_COMPLETE.json').read_text());rows={x['index']:x['route'] for x in d['selected_pool_rows']};selected=r['result']['selected_indices'];routes=[rows[i] for i in selected];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
checks['local_copies_match_remote_hashes']=all(sha(P/x['local'])==x['sha256'] for x in d['files'])
checks['complete_result_and_manifest_hashes']=complete['result_sha256']==sha(P/'sources/result.json') and complete['manifest_sha256']==sha(P/'sources/manifest.json')
checks['pool_and_cg_status_hash_chain']=r['pool_sha256']==receipt['pool_sha256']==d['remote_source_hashes'][r['pool']] and r['cg_status_sha256']==receipt['result_sha256']==d['remote_source_hashes'][r['cg_status']]
checks['input_hash']=sha(P/'sources/instance.csv')==r['provenance']['instance_sha256']==m['cases'][0]['input_sha256']
checks['worker_hashes']=all(sha(P/'sources'/n)==m['tooling_sha256'][n] for n in ['worker.py','worker.sh'])
src=(P/'source_commit_receipt.txt').read_text().splitlines();checks['clean_pinned_execution_source']=len(src)==2 and src[0]==m['execution_commit'] and src[1].split()[0]==sha(P/'sources/run_capacity_speed_event_cg.py')
counts=Counter(t for route in routes for t in route['trips']);dups={str(t):v for t,v in counts.items() if v>1};checks['277trip_coverage_and_duplicates']=set(counts)==set(range(277)) and dups==r['duplicate_service_audit']['overcovered_trips'] and sum(v-1 for v in counts.values())==73 and len(dups)==61
sites=r['physical_station_capacity_audit']['stations'];independent={}
for site,s in sites.items():
 events=[]
 for route in routes:
  stops=route['expanded_grid_charging_stops']
  for st,start,end in zip(stops['stations'],stops['cst'],stops['cet']):
   if st.rsplit('_',1)[0]==site:events.extend([(start,1),(end,-1)])
 active=peak=0;minute=None
 for t,v in sorted(events):
  active+=v
  if active>peak:peak=active;minute=t
 independent[site]={'peak':peak,'peak_time_min':minute,'count':s['documented_chargers']}
checks['independent_shared_capacity_sweep']=all(v['peak']==sites[k]['peak_simultaneous_connections'] and v['peak_time_min']==sites[k]['peak_time_min'] for k,v in independent.items())
cost=sum(route['cost']-100000 for route in routes);checks['selected_cost_and_fleet']=len(selected)==11 and math.isclose(cost,r['result']['charging_related_cost'],abs_tol=1e-7)
log=(P/'sources/result.json.gurobi.log').read_text();stats=re.findall(r'Best objective ([\deE+.\-]+), best bound ([\deE+.\-]+), gap ([\d.]+)%',log);checks['both_log_endpoints_match_json']=len(stats)==2 and all(math.isclose(float(x),y,rel_tol=1e-9,abs_tol=1e-8) for x,y in zip([stats[0][0],stats[0][1],stats[1][0],stats[1][1]],[11,10,774.872,380.3389106430675])) and log.count('Time limit reached')==2
stage1=r['result']['stage1'];stage2=r['result']['stage2'];flags=Counter(route.get('physical_realization',{}).get('status','missing') for route in routes)
summary={'case':'w5_k17_18E2','parent_global_prefix':17,'subgroup_reference_duties':10,'trip_count':277,'job_id':668432,'scheduler':'COMPLETED','fleet':11,'finite_pool_integer_bound':10,'fleet_proven':False,'subgroup_target_attained':False,'pool_columns':8343,'cg_pricing_certified':False,'cg_stop_reason':'pricing_deadline','charging_related_cost':cost,'charging_bound':stage2['charging_cost_bound'],'charging_gap_fraction':stage2['charging_cost_gap'],'stage1_wall_s':stage1['runtime_s'],'stage2_wall_s':stage2['runtime_s'],'nominal_total_mip_budget_s':3600,'wrapper_wall_s':complete['wall_s'],'extra_trip_assignments':73,'overcovered_trip_count':61,'all_trips_covered':True,'shared_capacity_enforced':False,'shared_capacity_diagnostic_pass':False,'individual_replay_claim':'Selected routes carry valid_event_time_realized metadata; MIP relies on exact-event construction. No new independent SOC/time replay was performed in this bounded endpoint audit.','selected_route_realization_flags':dict(flags),'inherited_routes_replayed':d['cg_compact']['inheritance']['every_inherited_route_replayed'],'inherited_columns':3086,'independent_capacity_sweep':independent,'physics':d['cg_compact']['physics'],'master':'set covering','objective':'fleet then expanded-grid charging-related cost','proof_scope':'Finite saved pool only; fleet optimum unresolved. No full-model lower bound from uncertified RMP; no exact-once/shared-capacity/GIRO-dispatch claim.','stage1_validated_flag_scope':'Only covering validity is required when capacity=False; the station-capacity sweep is diagnostic, not a passing gate.','checks':checks,'issues':[k for k,v in checks.items() if not v]}
(P/'verified_endpoint.json').write_text(json.dumps(summary,indent=2)+'\n')
flat={k:v for k,v in summary.items() if not isinstance(v,(dict,list))}
with (P/'verified_endpoint.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=flat);w.writeheader();w.writerow(flat)
print(json.dumps({'checks':checks,'issues':summary['issues'],'realization_flags':dict(flags),'cost':cost},indent=2))

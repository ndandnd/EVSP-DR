"""Independent post-run replay and registration audit of frozen rematch evidence.

No candidate search,retiming alternatives,cluster calls or publication occur here.
Requires the privately retained package/root prefixes,source snapshot and input copies.
"""
import argparse,collections,csv,datetime,gzip,hashlib,json,math,re,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--validator',type=Path,required=True);p.add_argument('--start-fee',type=float,required=True,choices=[5]);a=p.parse_args();root=a.package;raw=root/'private/final_native/raw';audit=root/'private/independent_audit';audit.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def close(x,y):return math.isclose(x,y,abs_tol=1e-6,rel_tol=1e-12)
def minutes(t):h,m=map(int,t.split(':'));return h*60+m
freeze=read(raw/'FREEZE.json');protocol=read(raw/'protocol.json');inventory=read(root/'private/final_native/native_inventory.private.json')
assert sha(raw/'FREEZE.json')==inventory['freeze_sha256'] and inventory['artifacts_unchanged']
for rel,h in freeze['artifact_sha256'].items():assert sha(raw/rel)==h,rel
assert sha(a.validator)=='5a72fb549251e9889cc7d1376481ab65d1c2d630f706bc9df0cbacd63692538a'
solver=root/'private/frozen_solver';data=root/'private/frozen_inputs/data';sys.path.insert(0,str(solver/'src'))
from utils_v2 import base_station_name

def occupancy(routes,key,plugs):
 load=collections.Counter()
 for r in routes:
  s=r[key]
  for station,x,y in zip(s['stations'],s['cst'],s['cet']):
   station=base_station_name(station);assert station in plugs and y>=x
   for minute in range(math.floor(x+1e-9),math.ceil(y-1e-9)):load[station,minute]+=1
 return sum(plugs[s] is not None and n>plugs[s] for (s,_),n in load.items())
def path_nogood_lhs(path,selected):
 return sum((u,v) in selected for u,v in zip(path,path[1:]))-sum(v==path[0] for u,v in selected)-sum(u==path[-1] for u,v in selected)
results=[];begin=time.monotonic()
for case in freeze['task_map']:
 tick=time.monotonic();m=read(raw/'manifests'/(case+'.json'));source_map={str(p.relative_to(solver)):sha(p) for p in (solver/'src').rglob('*.py')};assert source_map==m['solver_source_sha256']
 assert sha(root/'private/frozen_inputs/inventory.private.json')==m['native_evidence'][0]['sha256']
 for role,path in m['inputs'].items():assert sha(data/Path(path).relative_to(m['data_root']))==m['input_sha256'][role]
 native=next(x for x in inventory['cases'] if x['case']==case);attempt=Path(native['attempt_path']).name;folder=raw/'results'/case/attempt;solve=folder/'solve';receipt=read(solve/'receipt.json');wrapper=read(folder/'wrapper.private.json')
 for rel,h in native['native_evidence_hashes'].items():assert sha(folder/rel)==h,rel
 assert wrapper['exit_code']==receipt['child_exit_code']==0 and receipt['validated'] and not receipt['parent_timeout']
 assert datetime.datetime.fromisoformat(freeze['frozen_utc'])<datetime.datetime.fromisoformat(wrapper['started_utc'])
 assert wrapper['restart_count']==0 and receipt['candidates_attempted']<=32 and receipt['parent_wall_seconds']<180
 assert receipt['elapsed_seconds']<=receipt['parent_wall_seconds']<=wrapper['wall_seconds']
 assert sum(receipt['phase_seconds'].values())<=receipt['elapsed_seconds']+1e-6
 rows=list(csv.DictReader((data/Path(m['inputs']['instance']).relative_to(m['data_root'])).open()));starts=[minutes(r['Start1']) for r in rows];ends=[minutes(r['End1']) for r in rows];n=len(rows);assert all(b>a for a,b in zip(starts,ends))
 events=collections.Counter()
 for x,y in zip(starts,ends):events[x]+=1;events[y]-=1
 active=peak=0
 for t,delta in sorted(events.items()):active+=delta;peak=max(peak,active)
 assert peak==8
 counts=collections.Counter();digest=hashlib.sha256();count=0;root_file=root/'private/frozen_inputs'/(case+'_root.jsonl.gz')
 with gzip.open(root_file,'rb') as f:
  for line in f:
   digest.update(line);count+=1;r=json.loads(line);seq=r['trips'];assert len(seq)==len(set(seq)) and all(0<=t<n for t in seq)
   assert r['physical_realization']['status']=='valid_event_time_realized'
   for u,v in zip(seq,seq[1:]):assert ends[u]<=starts[v];counts[u,v]+=1
 assert count==m['root_columns'] and digest.hexdigest()==m['root_prefix_sha256']
 edges=sorted(counts,key=lambda e:(-counts[e],starts[e[0]],e[0],e[1]));ranks={e:i+1 for i,e in enumerate(edges)}
 log=(solve/'matching.private.log').read_text();best=[(float(v),float(b)) for v,b in re.findall(r'Best objective ([^,]+), best bound ([^,]+)',log)];assert len(best)==receipt['candidates_attempted']
 source_version=re.findall(r'Gurobi Optimizer version ([^\n]+)',log)[0]
 assert 'Set parameter Threads to value 1' in log and 'Set parameter Seed to value 0' in log
 previous_bad=[];visited=[];seen={};cut_count=excluded=single_calls=cache_hits=joint_calls=0;trace=[];winner=None
 for ordinal,entry in enumerate(receipt['attempts']):
  assert entry['candidate']==ordinal and entry['ranking_status']==2
  dest=solve/f'candidate_{ordinal:02d}';paths=read(dest/'paths.private.json');assert len(paths)==8 and sorted(t for q in paths for t in q)==list(range(n))
  assert paths==sorted(paths,key=lambda q:(starts[q[0]],q[0]))
  selected=set((u,v) for q in paths for u,v in zip(q,q[1:]));assert len(selected)==n-8 and selected<=counts.keys()
  assert all(path_nogood_lhs(q,selected)<=len(q)-2 for q in previous_bad)
  assert all(len(old&selected)<=n-9 for old in visited)
  rank=sum(ranks[e] for e in selected);assert close(rank,entry['rank_objective']) and close(rank,best[ordinal][0]) and close(rank,best[ordinal][1])
  for i,check in enumerate(entry['individual_checks']):
   assert check['ordinal']==i;key=tuple(paths[i]);value=check['reachable']
   if key in seen:cache_hits+=1;assert seen[key]==value
   else:
    single_calls+=1;single=dest/f'single_{i:02d}'
    if value:
     proof=read(single/'replay.private.json');rs=read(single/'routes.json');assert proof['validated'] and proof['exact_once'] and len(rs)==1 and rs[0]['trips']==paths[i]
    else:
     proof=read(single/'unreachable.private.json');assert proof['fixed_path']==paths[i] and 'no other-bus reservations' in proof['scope'] and proof['engine_sha256']==sha(raw/'code/engine.py')
    seen[key]=value
  item={'candidate_ordinal':ordinal,'rank_objective':rank,'native_rank_optimality_log_verified':True,'outcome':entry['outcome']}
  if entry['outcome']=='excluded_exact_unreachable_path':
   failed=entry['individual_checks'][-1];assert not failed['reachable'] and sum(not q['reachable'] for q in entry['individual_checks'])==1
   bad=paths[failed['ordinal']];assert path_nogood_lhs(bad,selected)==len(bad)-1;previous_bad.append(bad);cut_count+=1;item['cut_kind']='exact_whole_path_with_source_and_sink_boundaries';item['failed_path_ordinal']=failed['ordinal']
  else:
   assert len(entry['individual_checks'])==8 and all(q['reachable'] for q in entry['individual_checks']);joint_calls+=1
   if entry['outcome']=='reservation_guided_retiming_miss':
    assert entry['not_an_infeasibility_proof'] and entry['shared_retiming_reason'] in ['first_order_capacity_conflict','fixed_sequence_unreachable']
    if entry['shared_retiming_reason']=='first_order_capacity_conflict':assert occupancy(read(dest/'joint/routes.json'),'expanded_grid_charging_stops',receipt['plugs'] if 'plugs' in receipt else {'2190L':1,'4808':1,'7880C':1,'JON_A':1,'3127L':2,'PARX':None})>0
    visited.append(selected);excluded+=1;item['cut_kind']='heuristic_visited_matching_only';item['shared_reason']=entry['shared_retiming_reason']
   else:assert entry['outcome']=='validated_exploratory_candidate' and ordinal==len(receipt['attempts'])-1;winner=dest;item['cut_kind']='none_winner'
  trace.append(item)
 assert (cut_count,excluded,single_calls,cache_hits,joint_calls)==(receipt['safe_whole_path_nogoods'],receipt['heuristic_visited_matching_exclusions'],receipt['individual_retime_calls'],receipt['individual_cache_hits'],receipt['joint_retime_calls'])
 assert winner is not None
 schedules=winner/'joint/routes.json';paths=read(winner/'paths.private.json');routes=read(schedules);assert [r['trips'] for r in routes]==paths and sha(schedules)==receipt['selected_sha256']
 summary=root/'private/frozen_inputs'/(case+'_replay_summary.json');out=audit/(case+'_replay.private.json');mark=time.monotonic()
 with (audit/(case+'_replay.private.log')).open('w') as output:
  replay=subprocess.run([sys.executable,str(a.validator),'--summary',str(summary),'--selected',str(schedules),'--out',str(out),'--arm','augmented','--start-fee','5'],stdout=output,stderr=subprocess.STDOUT)
 proof=read(out);assert replay.returncode==0 and proof['validated'] and proof['exact_once'] and proof['fleet']==peak==8 and proof['duplicate_service']==0
 assert close(proof['objective_nonfleet'],receipt['grid_nonfleet_cost']) and close(proof['continuous_nonfleet'],receipt['continuous_nonfleet_cost']) and close(proof['grid_terminal_kwh'],receipt['grid_terminal_kwh'])
 physical=occupancy(routes,'charging_stops',proof['plugs']);expanded=occupancy(routes,'expanded_grid_charging_stops',proof['plugs']);assert physical==expanded==proof['grid_plug_overload_minutes']==0
 results.append({'case':case,'validated':True,'fleet':8,'fleet_only_lower_bound':peak,'fleet_minimum_scope':'Frozen trips and named v15 model;simultaneous passenger trips plus fully replayed8-bus dispatch. No cost optimality claim.','trip_count':n,'exact_once':True,'missing_trips':0,'duplicate_service':0,'physical_charging_stop_overload_minutes':physical,'expanded_grid_stop_overload_minutes':expanded,'grid_terminal_kwh':proof['grid_terminal_kwh'],'grid_nonfleet_cost':proof['objective_nonfleet'],'continuous_nonfleet_cost':proof['continuous_nonfleet'],'charging_cost_optimal':False,'source_solver_commit':m['solver_commit'],'root_pricing_certified':receipt['root_pricing_certified'],'root_columns':count,'root_prefix_sha256':digest.hexdigest(),'input_sha256':m['input_sha256'],'source_execution':{'rematch_sha256':sha(raw/'code/rematch.py'),'engine_sha256':sha(raw/'code/engine.py'),'native_matching_solver':source_version},'source_hashes':{'winner_routes':sha(schedules),'winner_paths':sha(winner/'paths.private.json'),'native_receipt':sha(solve/'receipt.json'),'native_wrapper':sha(folder/'wrapper.private.json'),'native_matching_log':sha(solve/'matching.private.log'),'postrun_independent_replay':sha(out),'root_gzip':sha(root_file)},'registration_adherence':{'frozen_before_start':True,'artifact_hashes_verified':True,'candidate_cap':32,'wall_limit_seconds':180,'candidates_attempted':len(trace),'safe_whole_path_nogoods':cut_count,'heuristic_shared_matching_exclusions':excluded,'individual_retime_calls':single_calls,'individual_cache_hits':cache_hits,'joint_retime_calls':joint_calls,'first_cover_and_all_following_rank_objectives_match_native_optimality_logs':True,'all_prior_cuts_satisfied_by_later_covers':True,'no_shared_miss_used_as_path_infeasibility_cut':True,'negative_paths_not_repriced_in_postrun_audit':True,'cut_sequence':trace},'run_clocks_seconds':{'worker':receipt['elapsed_seconds'],'parent':receipt['parent_wall_seconds'],'wrapper':wrapper['wall_seconds'],'phases':receipt['phase_seconds'],'parent_timed_out':receipt['parent_timeout']},'source_ancestor_durations':receipt['exploratory_ancestor_durations'],'source_graph_seconds':receipt['upstream'].get('graph_seconds'),'source_root_cg_seconds':receipt['upstream'].get('cg_seconds'),'postrun_audit_seconds':time.monotonic()-tick,'postrun_replay_seconds':time.monotonic()-mark})
aggregate={'schema':'independent-final-rematch-winner-audit-v1','audit_source_sha256':sha(__file__),'registered_validator_sha256':sha(a.validator),'freeze_sha256':sha(raw/'FREEZE.json'),'freeze_utc':freeze['frozen_utc'],'native_evidence_tar_sha256':sha(root/'private/final_native/native_evidence.tar'),'array_job_id':'918045','cases':results,'all_three_verified':True,'postrun_audit_total_seconds':time.monotonic()-begin,'limits':['Exploratory follow-up on3 selected first-order failures;does not replace prospective comparison or heldout scores.','Code/input/root hashes,route replay,occupancy and cut algebra were independently checked. Rank optimality uses archived native Gurobi logs;matching MIPs and negative-path pricing were not rerun.','Budget clocks are incremental saved-root worker/parent times;upstream root graph/CG and prior screening/diagnostic work remain separate.','C5 used4 heuristic visited-matching exclusions following actual shared plug conflicts;these are not infeasibility proofs.','No electricity-cost optimum,full-model lower bound,cold speed ratio,continuous-grid equivalence or operational portability claim.']}
write(root/'final_public/aggregate.json',aggregate);print(json.dumps({'all_three_verified':True,'cases':[{'case':r['case'],'fleet':r['fleet'],'trips':r['trip_count'],'grid_cost':r['grid_nonfleet_cost'],'continuous_cost':r['continuous_nonfleet_cost'],'audit_seconds':r['postrun_audit_seconds']} for r in results]},indent=2))

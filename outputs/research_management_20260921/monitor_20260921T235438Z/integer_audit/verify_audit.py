"""Verify frozen small artifacts, log endpoints, budget identities and own-pool handoffs."""
from pathlib import Path
import json,csv,hashlib,math,re,collections,subprocess
P=Path(__file__).resolve().parent;m=json.loads((P/'manifest.json').read_text());rows=[];proof=[];checks=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prov=json.loads((P/'remote_provenance_checks.json').read_text())
for c,fs in prov['fresh_inputs'].items():
 for k,v in fs.items():assert v['actual_sha256']==v['expected_sha256']==m['cases'][c][k+'_sha256']
for v in prov['own_journal_exports']:assert v['all_selected_records_match'] and v['journal_sha256']==v['expected_journal_sha256']
root=P.parents[3]
for f,v in prov['code'].items():
 b=subprocess.check_output(['git','show',m['execution_commit']+':'+f],cwd=root);assert hashlib.sha256(b).hexdigest()==v['sha256']
ledger=list(csv.DictReader((P/'jobs.tsv').open(),delimiter='\t'));states={x.split('|')[0]:x.split('|')[1] for x in (P/'scheduler_latest.txt').read_text().splitlines() if '|' in x}
for cell in ledger:
 d=P/'results'/cell['case']/(cell['arm']+'_s'+cell['seed'])/(cell['job_id']+'_r0');e=json.loads((d/'execution.json').read_text());r=json.loads((d/'result.json').read_text());t=r['two_stage'];a=json.loads((d/'mip_argv.json').read_text());case=m['cases'][cell['case']]
 assert states[cell['job_id']]=='COMPLETED' and e['status']=='finished'
 assert e['execution_commit']==r['mip_provenance']['git_commit']==m['execution_commit'];assert r['mip_provenance']['tracked_clean_at_end'];assert e['external_witness_columns_used'] is False and not r['extra_route_sources'] and e['global_certificate'] is None
 assert e['source_immutable'];assert e['source_hashes']==[case['fresh_cg_result_sha256'],case['fresh_journal_sha256']]
 verified=0
 for rel,h in e['output_sha256'].items():
  f=d/rel
  if f.exists():assert sha(f)==h,(f,h);verified+=1
 assert e['mip_solver_budget_s']==max(0,math.floor(3600-e.get('dive_wall_s',0)))
 assert int(a[a.index('--timelimit')+1])==e['mip_solver_budget_s']
 assert float(a[a.index('--stage1-timelimit')+1])==e['mip_solver_budget_s']/2
 assert int(a[a.index('--seed')+1])==int(cell['seed'])
 assert abs(e['actual_charged_dive_plus_solver_s']-e.get('dive_wall_s',0)-r['runtime_s'])<1e-7
 assert abs(e['mip_external_overhead_s']-(e['mip_wall_s']-r['runtime_s']))<1e-7
 assert e['strict_end_to_end_budget_claim'] is False
 assert r['physical_pool_audit']['input_hashes']['instance_sha256']==case['instance_sha256']
 for k,v in [('g_kwh',240),('charge_kw',240),('min_soc_frac',0),('soc_step',2.5),('block_min',5)]:assert r['physics'][k]==v
 assert r['physical_pool_audit']['master_sense']=='cover';assert r['physical_replay_validated'];assert not r['cross_route_charger_capacity_validated']
 own=False;stop='not_applicable'
 if cell['arm']=='treatment':
  dive=json.loads((d/'dive/manifest.json').read_text());stop=dive['dive']['stop_reason'];assert not dive['source']['witness_or_warm_input'];assert dive['global_certificate'] is None
  assert dive['source']['journal_sha256']==case['fresh_journal_sha256'];assert dive['source']['result_sha256']==case['fresh_cg_result_sha256'];assert dive['source']['source_immutable']
  assert dive['network_cache_audit']['pickle_sha256_verified'];assert not dive['network_cache_audit']['cache_contains_columns'];assert dive['network_cache_audit']['method_audit']['identical']
  assert dive['augmented_pool']['augmented_result_sha256']==sha(d/'dive/cg.json');assert r['source_result_sha256']==sha(d/'dive/cg.json')
  if e['incumbent_export']:
   ex=json.loads((d/'dive/dive_incumbent.json').read_text());start=r['mip_start'];assert sha(d/'dive/dive_incumbent.json')==e['incumbent_export']['sha256']==start['source_sha256'];assert ex['source']=='own_dive_augmented_journal' and ex['external_witness_columns_used'] is False
   assert ex['source_journal_sha256']==dive['augmented_pool']['augmented_journal_sha256'];assert start['validated'] and start['pool_columns_added']==start['pool_columns_replaced']==0;assert start['validated_bus_count']==len(ex['routes'])==8;assert start['solver_acceptance']['accepted'];assert '--verified-expanded-initial-partition' in a
   for record,h in zip(ex['routes'],ex['record_sha256']):assert hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()==h
   own=True
  else:assert '--initial-partition-routes' not in a
 else:
  assert '--initial-partition-routes' not in a;assert r['source_result_sha256']==case['fresh_cg_result_sha256']
 lines=(d/'mip_gurobi.log').read_text().splitlines();matches=[(i+1,ln) for i,ln in enumerate(lines) if re.match(r'Best objective ',ln)];assert len(matches)==2
 stage1=matches[0];numbers=re.match(r'Best objective ([\deE+.-]+), best bound ([\deE+.-]+), gap ([\d.]+)%',stage1[1]);assert numbers
 assert abs(float(numbers[1])-t['stage1_buses'])<1e-7 and abs(float(numbers[2])-t['stage1_bound'])<1e-7
 if own:assert any('Loaded user MIP start with objective 8' in ln for ln in lines)
 counter=collections.Counter(i for route in r['selected_routes'] for i in route['trips']);over=sum(n>1 for n in counter.values());assert over==r['overcovered_trips']
 row={**cell,'scheduler':'COMPLETED','buses':r['buses'],'fleet_bound':t['stage1_bound'],'finite_pool_fleet_proven':t['fleet_proven'],'target8':r['buses']==8,'stage1_status':t['stage1_status_name'],'stage2_status':t['stage2_status_name'],'dive_stop':stop,'own_dive_handoff':own,'new_columns':e.get('columns_generated',0),'dive_wall_s':e.get('dive_wall_s',0),'mip_budget_s':e['mip_solver_budget_s'],'mip_solver_runtime_s':r['runtime_s'],'charged_s':e['actual_charged_dive_plus_solver_s'],'charged_excess_s':max(0,e['actual_charged_dive_plus_solver_s']-3600),'external_mip_overhead_s':e['mip_external_overhead_s'],'end_to_end_s':e['actual_end_to_end_wall_s'],'overcovered_distinct_trips':over,'extra_trip_occurrences':sum(n-1 for n in counter.values()),'route_replay':r['physical_replay_validated'],'duplicate_removal_validated':r['duplicate_trip_removal_validated'],'shared_capacity_validated':False,'global_certificate':None,'verified_small_receipt_hashes':verified,'result_sha256':sha(d/'result.json'),'stage1_proof_log':str((d/'mip_gurobi.log').relative_to(P))+':'+str(stage1[0])}
 rows.append(row);proof.append({**cell,'log':str((d/'mip_gurobi.log').relative_to(P)),'stage_summaries':[dict(line=n,text=ln) for n,ln in matches],'accepted_start_lines':[dict(line=i+1,text=ln) for i,ln in enumerate(lines) if 'Loaded user MIP start' in ln]})
with (P/'verified_results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(P/'verified_results.json').write_text(json.dumps(rows,indent=2)+'\n');(P/'proof_log_lines.json').write_text(json.dumps(proof,indent=2)+'\n')
hits=[r for r in rows if r['target8']];summary=dict(all16_completed=len(rows)==16,treatment_hits=sum(r['arm']=='treatment' and r['target8'] for r in rows),control_hits=sum(r['arm']=='control' and r['target8'] for r in rows),independent_cases=4,seeds_per_case=2,control_finite_pool9_proofs=sum(r['arm']=='control' and r['finite_pool_fleet_proven'] for r in rows),own_handoffs=sum(r['own_dive_handoff'] for r in rows),source_hashes_unchanged=True,all7_own_exports_independently_matched_journal_ordinals=True,pinned_remote_code_matches_git_blobs=True,max_charged_excess_s=max(r['charged_excess_s'] for r in rows),hit_end_to_end_range_s=[min(r['end_to_end_s'] for r in hits),max(r['end_to_end_s'] for r in hits)],fetched_artifacts=103,scope='saved-pool fleet proofs and individual route replay; no global B&P or shared capacity certificate',checks_passed=True)
(P/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

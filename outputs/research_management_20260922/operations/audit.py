from pathlib import Path
import json,hashlib,re,csv,collections
P=Path(__file__).resolve().parent
plan=json.loads((P.parents[1]/'research_management_20260921/operations/k15_salvage_preflight_20260922/salvage_plan.json').read_text())
collection=json.loads((P/'collection.json').read_text());checks=[]
def check(label,test):
 checks.append({'label':label,'passed':bool(test)})
 assert test,label
for f in collection['files']:check('copied hash '+f['local'],hashlib.sha256((P/f['local']).read_bytes()).hexdigest()==f['sha256'])
rows=[];proof=[]
for case,j,control in [('c1_k15','728184',19),('c5_k15','728185',16)]:
 p=P/'salvage'/case;r=json.loads((p/'result.json').read_text());receipt=json.loads((p/'receipt.json').read_text());gate=json.loads((p/'physical_gate.json').read_text());cell=plan['cases'][case];t=r['two_stage'];native=r['physical_pool_audit'];args=r['mip_provenance']['arguments']
 check(case+' receipt finished/immutable',receipt['status']=='finished' and receipt['source_immutable'])
 check(case+' result hash',receipt['result_sha256']==hashlib.sha256((p/'result.json').read_bytes()).hexdigest())
 check(case+' input hashes',r['source_result_sha256']==cell['source_hashes']['augmented_result']['actual'] and r['source_journal_sha256']==cell['source_hashes']['augmented_journal']['actual'])
 check(case+' native zero rejects/repairs',gate['rejected_columns']==native['rejected_columns']==gate['deterministically_repaired']==native['deterministically_repaired']==0)
 check(case+' pool identity',gate['mip_ordered_pool_sha256']==native['base_pool_ordered_sha256']==native['mip_ordered_pool_sha256'])
 check(case+' original settings',args['cover'] and args['two_stage'] and args['seed']==20260921 and args['threads']==8 and args['mipgap']==0.0001 and args['initial_partition_routes'] is None and args['timelimit']==cell['solver_limit_s'])
 check(case+' pinned code',r['mip_provenance']['git_commit']==plan['execution_commit'] and not r['mip_provenance']['git_dirty'])
 check(case+' budget accounting',abs(receipt['charged_dive_plus_solver_s']-receipt['failed_dive_wall_s']-r['runtime_s'])<1e-6)
 log=(p/'mip_gurobi.log').read_text().splitlines();summaries=[(i,s) for i,s in enumerate(log,1) if s.startswith('Best objective')]
 check(case+' two full log endpoints',len(summaries)==2)
 nums=[float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?',summaries[0][1],re.I)]
 check(case+' fleet log matches',abs(nums[0]-r['buses'])<1e-6 and abs(nums[1]-t['stage1_bound'])<1e-6)
 proof.append({'case':case,'log':str((p/'mip_gurobi.log').relative_to(P)),'summaries':[{'line':i,'text':s} for i,s in summaries]})
 rows.append({'case':case,'job_id':j,'scheduler':'COMPLETED','control_buses':control,'salvage_buses':r['buses'],'fleet_bound':t['stage1_bound'],'fleet_proven':t['fleet_proven'],'target15_attained':r['buses']==15,'pool_columns':r['pool_columns'],'gate_rejected':gate['rejected_columns'],'gate_repaired':gate['deterministically_repaired'],'charging_cost':r['charging_cost'],'charging_bound':t['stage2_variable_bound'],'reported_overcovered_trips':r['overcovered_trips'],'physical_replay_validated':r['physical_replay_validated'],'duplicate_trip_removal_validated':r['duplicate_trip_removal_validated'],'shared_capacity_validated':r['cross_route_charger_capacity_validated'],'solver_limit_s':cell['solver_limit_s'],'actual_solver_runtime_s':r['runtime_s'],'charged_dive_plus_solver_s':receipt['charged_dive_plus_solver_s'],'gate_wall_s':receipt['gate_wall_s'],'external_mip_overhead_s':receipt['external_mip_overhead_s']})
strict=json.loads((P/'strict_k19/cg_result.json').read_text());complete=json.loads((P/'strict_k19/CG_COMPLETE.json').read_text())
check('strict published result hash',hashlib.sha256((P/'strict_k19/cg_result.json').read_bytes()).hexdigest()==complete['result_sha256'])
check('strict declared pool hash',strict['pool_sha256']==complete['pool_sha256'])
check('strict zero iterations / budget',not strict['iterations'] and strict['network_build_s']>14400 and strict['stop_reason']=='cg_wall_limit' and not strict['certified_rc_optimal'])
check('strict initial pool accounting',strict['final']['pool_columns']==strict['checkpoint']['initial_pool_columns']==strict['inheritance']['inherited_columns']+54)
lines=(P/'initial_policy_queue.txt').read_text().splitlines();q=[l.split('|') for l in lines if len(l.split('|'))==5];alloc=[l.split('|') for l in lines if l.startswith('661616_') and len(l.split('|'))>5]
summary={'observed_utc':collection['observed_utc'],'checks_passed':len(checks),'checks':checks,'queue_display_rows':dict(collections.Counter(x[2] for x in q)),'baseline_allocations':dict(collections.Counter(x[2] for x in alloc)),'baseline_completed_ids':[x[0] for x in alloc if x[2]=='COMPLETED'],'salvage_results':rows,'strict_k19':{'parent_prefix':19,'subgroup_duties':11,'trips':331,'pricing_iterations':0,'network_build_s':strict['network_build_s'],'total_cg_runtime_s':strict['runtime_s'],'final':strict['final'],'certified':False,'full_pool_independently_replayed_this_collection':False,'inheritance_replay_recorded':strict['inheritance']['every_inherited_route_replayed']}}
(P/'verified_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(P/'proof_lines.json').write_text(json.dumps(proof,indent=2)+'\n')
with (P/'salvage_endpoints.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'checks_passed':len(checks),'baseline':summary['baseline_allocations'],'completed_graphs':summary['baseline_completed_ids'],'salvage':[(r['case'],r['salvage_buses'],r['fleet_proven']) for r in rows]}))

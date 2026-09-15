"""Auditable branch of frozen native main, after its complete model preparation."""
from pathlib import Path
from collections import Counter
import hashlib,json,os,sys,time
NATIVE_SHA='bcb5a6b76040ff6ddfa932433d296a1f0f72207b28cbba738b1b4dd39f1eaac7'
MODEL={'objective':'constant_zero','master_sense':'cover','fleet_constraint':'sum_binary_x_at_most_target','augmentation':False,'shared_capacity':False,'duplicate_removal_validated':False,'fleet_minimum_proven':False,'charging_optimality_proven':False}
def model_digest():return hashlib.sha256(json.dumps(MODEL,sort_keys=True).encode()).hexdigest()
def identity_gate(v,c):
 p=v['physical_pool_audit']
 for key in ('source_result_sha256','source_journal_sha256'):
  if v[key]!=c[key]:raise ValueError('source identity changed: '+key)
 if p['base_pool_ordered_sha256']!=c['native_pool_ordered_sha256'] or p['base_pool_column_count']!=c['native_pool_columns']:raise ValueError('native pool identity changed')
 if p['post_augmentation_columns']!=p['base_pool_column_count'] or p['augmented_pool_ordered_sha256']!=p['base_pool_ordered_sha256'] or p['added_giro_route_count']:raise ValueError('augmentation forbidden')
 if p['rejected_columns'] or p['deterministically_repaired']:raise ValueError('physical admission changed')
 if v['model_sha256']!=c['model_sha256'] or model_digest()!=c['model_sha256']:raise ValueError('model identity changed')
def solve(ns,g):
 args=ns['args'];m=ns['m'];a=ns['a'];routes=ns['routes'];trips=ns['trips'];pa=ns['physical_pool_audit']
 c=json.loads(Path(os.environ['EVSP_TARGET_CASE']).read_text())
 if not args.cover or args.extra_routes or args.initial_partition_routes or args.target_cap!=c['target_cap'] or args.threads!=8 or args.timelimit!=c['solver_budget_s']:raise ValueError('target experiment arguments changed')
 v={'schema':'evsp-dr-target-feasibility-v1','case_id':c['id'],'is_validation':c['is_validation'],'target_cap':args.target_cap,'model':MODEL,'model_sha256':model_digest(),'physical_pool_audit':pa,'source_result_sha256':ns['source_result_sha256'],'source_journal_sha256':ns['source_journal_sha256'],'mip_start':ns['initial_partition_start'],'source_union_pool_set':c['source_union_pool_set'],'native_execution_identity':ns['code_identity'],'tooling_sha256':json.loads(os.environ['EVSP_TARGET_TOOLING']),'parameters':{'TimeLimit':args.timelimit,'MIPGap':args.mipgap,'Threads':args.threads,'Seed':m.Params.Seed},'independent_donor_bound_is_solver_incumbent':False}
 identity_gate(v,c)
 m.addConstr(ns['gp'].quicksum(a[i] for i in range(len(routes)))<=args.target_cap,name='target_fleet_cap')
 m.setObjective(0.0,ns['GRB'].MINIMIZE)
 t=time.perf_counter()
 acceptance=g['optimize_with_start_audit'](m,ns['GRB'],start_supplied=bool(ns['mip_start']))
 v.update(solver_status=ns['status_names'].get(int(m.Status),str(m.Status)),solver_status_code=int(m.Status),solution_count=int(m.SolCount),solver_runtime_s=float(m.Runtime),optimize_wall_s=time.perf_counter()-t,node_count=float(m.NodeCount),mip_start_acceptance=acceptance,numerically_rejected_covers=0,physical_replay_validated=False,buses=None,selected_route_indices=[],selected_routes=[],coverage_counts={},proof_classification='unresolved')
 if m.SolCount:
  chosen=[i for i in range(len(routes)) if a[i].X>0.5]
  ok,reason=g['validate_incumbent_assignment'](chosen,routes,trips,cover=True)
  if len(chosen)>args.target_cap:ok,reason=False,'integer_fleet_cap_violation'
  if not ok:v.update(numerically_rejected_covers=1,rejection_reason=reason)
  else:
   selected=[routes[i] for i in chosen]
   g['validate_final_selected_routes'](ns['status'],trips,selected,data_dir=args.data_dir,reference_data_dir=args.reference_data_dir,physical_pool_audit=pa,cover=True)
   counts=Counter(t for r in selected for t in r['trips'])
   v.update(buses=len(chosen),selected_route_indices=chosen,selected_routes=selected,coverage_counts={str(t):counts[t] for t in trips},exact_cover_counts_all_one=all(counts[t]==1 for t in trips),physical_replay_validated=True,proof_classification='target_feasible_in_validated_finite_pool')
 elif m.Status==ns['GRB'].INFEASIBLE:v['proof_classification']='target_infeasible_in_validated_finite_pool'
 if g['file_sha256'](args.result)!=v['source_result_sha256'] or g['file_sha256'](ns['source_journal'])!=v['source_journal_sha256']:raise ValueError('source changed during solve')
 v['final_native_execution_identity']=g['verified_mip_code_identity']()
 v['end_to_end_wall_s']=time.perf_counter()-ns['end_to_end_started']
 identity_gate(v,c);g['write_new_json'](ns['out'],v);return 0

def patched_source(source):
 if hashlib.sha256(source.encode()).hexdigest()!=NATIVE_SHA:raise ValueError('native code SHA changed')
 needle='    args = parser.parse_args(argv)'
 assert source.count(needle)==1
 source=source.replace(needle,'    parser.add_argument("--target-cap", type=int, required=True)\n'+needle)
 needle='    def progress_observer(\n'
 assert source.count(needle)==1
 return source.replace(needle,'    return target_solve(locals(), globals())\n\n'+needle)
if __name__=='__main__':
 native=Path(os.environ['EVSP_NATIVE_SOURCE'])
 sys.path.insert(0,str(native.parent))
 source=patched_source(native.read_text())
 scope={'__file__':str(native),'__name__':'__main__','target_solve':solve}
 exec(compile(source,str(native), 'exec'),scope)

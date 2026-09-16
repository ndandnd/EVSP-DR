"""Read-only verification of completed review P1 cells against frozen controls."""
from pathlib import Path
import json,argparse,hashlib,datetime,math

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main(root,out):
 b=Path(root);o=Path(out);m=read(b/'manifest.json');audit=[]
 cells=[]
 for cid,c in m['cases'].items():cells.append((cid,c,b/'cases'/cid/'completion.json',False))
 for c in m['reused_cells']:cells.append((c['case_id'],c,Path(c['campaign'])/'cases'/c['case']/'completion.json',True))
 for cid,c,cp,reused in cells:
  if not cp.exists():continue
  done=read(cp);p=Path(done['result_path']);assert sha(p)==done['result_sha256'];r=read(p)
  original=c['comparison'] if reused else c['comparator'];pool=r['physical_pool_audit'];prov=r['mip_provenance'];two=r['two_stage']
  tests={
   'ordered_pool':pool['mip_ordered_pool_sha256']==original['ordered_pool_sha256'],
   'pool_column_count':r['pool_columns']==original['pool_columns'],
   'mip_start':{k:v for k,v in r['mip_start'].items() if k!='solver_acceptance'}==original['mip_start'],
   'journal_identity':r['source_journal_sha256']==c['source_journal_sha256'],
   'seed':prov['gurobi_parameters']['Seed']==c['seed'],
   'threads':prov['gurobi_parameters']['Threads']==8,
   'fleet_stage_seconds':two['stage1_time_limit_s']==c.get('stage1_budget_s',10800),
   'total_seconds':prov['gurobi_parameters']['TimeLimit_s']==c.get('solver_budget_s',12600),
   'stage2_fleet_constraint':two['stage2_fleet_constraint']=='at_most',
   'covering':prov['arguments']['cover'] is True,
   'physics':r['physics']==original['physics'],
   'clean_code':prov['tracked_clean_at_end'] and prov['git_commit']==prov['final_observed_git_commit'],
   'execution_commit':prov['git_commit']==('871d057e1067411f09581e37d78f7c1ca43f68bb' if reused else m['execution_commit']),
   'route_replay':r['physical_replay_validated'] is True,
   'no_pool_edit':pool['rejected_columns']==pool['deterministically_repaired']==pool['added_giro_route_count']==0,
  }
  assert all(tests.values()),(cid,tests)
  bound=r['fleet_bound'];fleet=r['buses'];claimed=r['fleet_proven']
  if claimed:assert bound>fleet-1-1e-6,(cid,bound,fleet)
  assert two['fleet_proven']==claimed
  audit.append(dict(case_id=cid,review_item=c['review_item'],findings=c['findings'],status='verified_control_identity',result_path=str(p),result_sha256=sha(p),reused=reused,tests=tests,buses=fleet,pool_fleet_bound=bound,fleet_proven_in_pool=claimed,target=c.get('target_k',c.get('k')),physical_route_replay=r['physical_replay_validated'],duplicate_trip_removal_validated=r.get('duplicate_trip_removal_validated'),cross_route_capacity_validated=r.get('cross_route_charger_capacity_validated'),optimal_scope=r.get('optimal_scope'),two_stage=two))
 report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(b/'manifest.json'),completed_cells=len(audit),verified_cells=len(audit),new_completed=sum(not x['reused'] for x in audit),checks=audit)
 o.mkdir(parents=True,exist_ok=True);(o/'endpoint_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.root,a.out)

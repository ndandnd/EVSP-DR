"""Read this campaign only; retain target-feasibility proof scope and every restart."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,re,subprocess
import common as w
from target_solver import identity_gate,MODEL
ROOT=Path(__file__).resolve().parent
S='/usr/local/slurm/slurm-25.05.5/bin/'
def snapshot(p):
 if not p.is_file():return {'path':str(p),'exists':False}
 raw=p.read_bytes();return {'path':str(p),'exists':True,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def json_snapshot(p):
 raw=p.read_bytes();return json.loads(raw),{'path':str(p),'exists':True,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def proof(v,c):
 w.require_hash(c['source_result'],c['source_result_sha256'])
 identity_gate(v,c)
 assert v['schema']=='evsp-dr-target-feasibility-v1' and v['case_id']==c['id'] and v['is_validation']==c['is_validation'] and v['target_cap']==c['target_cap'] and v['model']==MODEL
 assert v['parameters']=={'TimeLimit':c['solver_budget_s'],'MIPGap':0.0001,'Threads':8,'Seed':0}
 for key in ['native_execution_identity','final_native_execution_identity']:
  assert v[key]['observed_commit']==c['execution_commit'] and v[key]['tracked_clean'] and v[key]['detached']
 classification='unresolved'
 if v['physical_replay_validated']:
  assert v['solution_count']>0 and v['buses']==len(v['selected_routes'])==len(v['selected_route_indices']) and 0<v['buses']<=c['target_cap']
  counts=Counter(str(t) for r in v['selected_routes'] for t in r['trips'])
  expected={str(t) for t in w.read(c['source_result'])['trip_ids']}
  assert set(counts)==expected and dict(counts)==v['coverage_counts'] and min(counts.values())>=1
  assert v['numerically_rejected_covers']==0
  classification='target_feasible_in_validated_finite_pool'
 elif v['solver_status']=='INFEASIBLE':
  assert v['solution_count']==0 and v['buses'] is None
  classification='target_infeasible_in_validated_finite_pool'
 assert classification==v['proof_classification']
 return classification

def collect(root=ROOT):
 root=Path(root);m,manifest= json_snapshot(root/'manifest.json');mh=manifest['sha256']
 for n,h in m['tooling_sha256'].items():w.require_hash(root/n,h)
 groups={'production':[],'validation':[]};alljobs=[]
 for group in groups:
  jf=root/(group+'_jobs.json')
  if not jf.exists():continue
  jobs=w.read(jf);alljobs.extend(jobs)
  for j in jobs:
   cid=j['case_id'];c=m['cases'][cid];assert c['is_validation']==(group=='validation')
   base=root/'cases'/cid;completion=None;completion_artifact=snapshot(base/'completion.json')
   if completion_artifact['exists']:
    completion,completion_artifact=json_snapshot(base/'completion.json')
    assert completion['case_id']==cid and completion['manifest_sha256']==mh and completion['status']=='finished'
   sharedlog=root/'logs'/('utf_'+cid+'_'+j['job_id']+'.out');log=sharedlog.read_text(errors='replace') if sharedlog.exists() else ''
   startup={'full_size_license_probe_pass_marker': 'Native Gurobi license check passed' in log,'full_size_probe_pass_count_job_log':log.count('Native Gurobi license check passed'),'restricted_license_message_observed':'Restricted license' in log,'license_error_observed':bool(re.search(r'(license expired|No Gurobi license|Unable to open Gurobi license|Model too large for size-limited)',log,re.I)),'log':snapshot(sharedlog),'job_log_can_span_restarts':True}
   row={'case_id':cid,'job_id':str(j['job_id']),'is_validation':c['is_validation'],'target_cap':c['target_cap'],'solver_budget_s':c['solver_budget_s'],'resources':c['resources'],'source_result_path':c['source_result'],'source_result_sha256':c['source_result_sha256'],'source_journal_path':c['source_journal'],'source_journal_sha256':c['source_journal_sha256'],'native_pool_ordered_sha256':c['native_pool_ordered_sha256'],'source_union_pool_set':c['source_union_pool_set'],'completion':completion_artifact,'startup':startup,'attempts':[],'proof_classification':'unresolved','proof_scope':'Only target-cap feasibility/infeasibility in the exact validated finite pool; no fleet-minimum or charging-optimality claim','scheduler_status_is_proof':False,'completion_identity_verified':False}
   attempts=sorted((base/'attempts').glob(str(j['job_id'])+'_r*'))
   if not attempts:attempts=[base/'attempts'/(str(j['job_id'])+'_r0')]
   for attempt in attempts:
    assert re.fullmatch(re.escape(str(j['job_id']))+r'_r\d+',attempt.name)
    sp=attempt/'state.json';state,sa=json_snapshot(sp) if sp.exists() else ({},{'path':str(sp),'exists':False})
    if state:assert state['manifest_sha256']==mh and state['case_id']==cid and state['attempt']==attempt.name and str(state['job_id'])==str(j['job_id'])
    rp=attempt/'result.json';a={'attempt_tag':attempt.name,'restart_count':int(attempt.name.rsplit('_r',1)[1]),'attempt_path':str(attempt),'attempt_exists':attempt.exists(),'state':sa,'worker_status':state.get('status','not_started'),'result':snapshot(rp),'execution':snapshot(attempt/'execution.json'),'stdout':snapshot(attempt/'stdout.log'),'stderr':snapshot(attempt/'stderr.log'),'gurobi_log':snapshot(attempt/'gurobi.log'),'full_size_license_probe_pass_inferred_from_worker_entry':bool(state),'license_inference_basis':'worker.sub is set -e and runs worker only after full-size 2001-variable probe succeeds','proof_classification':'unresolved','result_identity_verified':False}
    if rp.exists():
     v,ra=json_snapshot(rp);a['result']=ra
     assert v['tooling_sha256']==m['tooling_sha256']
     a.update(proof_classification=proof(v,c),result_identity_verified=True,solver_status=v['solver_status'],solution_count=v['solution_count'],buses=v['buses'],physical_replay_validated=v['physical_replay_validated'],numerically_rejected_covers=v['numerically_rejected_covers'],solver_runtime_s=v['solver_runtime_s'],end_to_end_wall_s=v['end_to_end_wall_s'])
     if completion and Path(completion['result_path'])==rp:
      assert completion['result_sha256']==ra['sha256'] and completion['attempt']==attempt.name and completion['proof_classification']==a['proof_classification']
      row.update(completion_identity_verified=True,proof_classification=a['proof_classification'],published_result_path=str(rp),published_result_sha256=ra['sha256'])
    row['attempts'].append(a)
   if completion:assert row['completion_identity_verified'],'completion lacks exact verified attempt result'
   groups[group].append(row)
 ids=','.join(str(j['job_id']) for j in alljobs)
 scheduler={}
 if ids:
  for name,args in [('accounting',['sacct','-j',ids,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TotalCPU,MaxRSS,NodeList']),('queue',['squeue','-j',ids,'-h','-o','%i|%T|%M|%R'])]:
   r=subprocess.run([S+args[0],*args[1:]],capture_output=True,text=True);scheduler[name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 return {'schema':'union-target-feasibility-collection-v1','utc':w.now(),'manifest':manifest,'helper_sha256':w.sha(__file__),'production':groups['production'],'validation':groups['validation'],'scheduler':scheduler,'mutable_log_hashes_scope':'point-in-time bytes only; active logs may grow','no_job_mutations':True}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'collection_status.json');p.add_argument('--json',action='store_true');a=p.parse_args();v=collect();w.save(a.out,v);print(json.dumps(v if a.json else {'production':len(v['production']),'validation':len(v['validation']),'out':str(a.out)}))

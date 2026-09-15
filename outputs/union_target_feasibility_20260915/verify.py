from pathlib import Path
import subprocess
import common as w
from target_solver import identity_gate
ROOT=Path(__file__).resolve().parent;S='/usr/local/slurm/slurm-25.05.5/bin/'
def main():
 m=w.read(ROOT/'manifest.json');mh=w.sha(ROOT/'manifest.json');rows=[];missing=[];jobs=w.read(ROOT/'validation_jobs.json')
 for j in jobs:
  c=m['cases'][j['case_id']];p=ROOT/'cases'/c['id']/'completion.json'
  if not p.exists():missing.append(c['id']);continue
  done=w.read(p);assert done['status']=='finished' and done['manifest_sha256']==mh
  w.require_hash(done['result_path'],done['result_sha256']);v=w.read(done['result_path']);identity_gate(v,c)
  assert v['proof_classification']==c['validation_expected']
  assert v['is_validation'] and v['model']['fleet_minimum_proven'] is False and v['model']['charging_optimality_proven'] is False
  assert v['parameters']=={'TimeLimit':60,'MIPGap':0.0001,'Threads':8,'Seed':0}
  if v['buses'] is not None:assert v['physical_replay_validated'] and v['buses']<=c['target_cap'] and min(v['coverage_counts'].values())>=1
  else:assert v['solver_status']=='INFEASIBLE' and v['solution_count']==0
  assert v['native_execution_identity']['observed_commit']==c['execution_commit'] and v['final_native_execution_identity']['observed_commit']==c['execution_commit']
  log=(ROOT/'logs'/('utf_'+c['id']+'_'+j['job_id']+'.out')).read_text();assert 'Native Gurobi license check passed' in log
  x=dict(t.split('=',1) for t in j['effective_scontrol_at_submission'].split() if '=' in t)
  assert x['Partition']=='default_partition' and x['ExcNodeList']=='scaglione-compute-01' and x['NumCPUs']=='8' and x['MinMemoryNode']=='24G' and x['Requeue']=='1' and x['TimeLimit']=='00:30:00'
  row={k:v[k] for k in ['case_id','target_cap','solver_status','solution_count','proof_classification','buses','physical_replay_validated','numerically_rejected_covers','solver_runtime_s','end_to_end_wall_s','parameters','source_result_sha256','source_journal_sha256','source_union_pool_set','mip_start_acceptance']};row.update(job_id=j['job_id'],result_path=done['result_path'],result_sha256=done['result_sha256'],native_pool_ordered_sha256=v['physical_pool_audit']['base_pool_ordered_sha256'],native_pool_columns=v['physical_pool_audit']['base_pool_column_count']);rows.append(row)
 raw=subprocess.check_output([S+'sacct','-j',','.join(j['job_id'] for j in jobs),'-P','-n','-o','JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TotalCPU,MaxRSS,NodeList'],text=True);(ROOT/'validation_accounting.txt').write_text(raw)
 neg=w.read(ROOT/'negative_validation.json');assert neg['status']=='passed' and neg['manifest_sha256']==mh
 report={'status':'pending' if missing else 'passed','missing':missing,'utc':w.now(),'manifest_sha256':mh,'rows':rows,'negative_validation_sha256':w.sha(ROOT/'negative_validation.json'),'accounting_sha256':w.sha(ROOT/'validation_accounting.txt'),'production_submitted':False,'limitations':'Easy classification fixtures on two full production pools; no evidence of target recovery or speedup, no fleet or charging optimum proof.'}
 w.save(ROOT/('native_validation_pending.json' if missing else 'native_validation.json'),report);print(report)
if __name__=='__main__':main()

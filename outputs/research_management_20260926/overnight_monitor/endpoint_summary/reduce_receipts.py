"""Read-only reduction of the frozen 04:00 endpoint receipts; no network or solver."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'terminal_receipts_0400.json'
S = json.loads(SOURCE.read_text())
ROOT = S['root2']
checks = []
missing_attempt_output_hashes = set()

def check(condition, label):
    assert condition, label
    checks.append(label)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def emit_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def status(value):
    return {None:'missing',2:'OPTIMAL',3:'INFEASIBLE',9:'TIME_LIMIT'}.get(value,str(value))

def counts(rows, key):
    return dict(sorted(Counter(str(r.get(key,'missing')) for r in rows).items()))

summaries = {r['path']:r for r in S['summaries']}
attempts = {r['path']:r for r in S['attempts']}
check(len(summaries)==len(S['summaries']), 'unique summary paths')
check(len(attempts)==len(S['attempts']), 'unique attempt paths')

def bound_summary(attempt_dir):
    record = summaries.get(attempt_dir + '/out/summary.json')
    if record:
        attempt = attempts[attempt_dir + '/attempt.json']['data']
        recorded_hash=attempt.get('output_hashes',{}).get('summary.json')
        if recorded_hash is None:
            check(attempt.get('stage')=='fresh_cg' and attempt.get('returncode')==3,
                  'absent output hash limited to rc3 fresh attempt '+record['path'])
            missing_attempt_output_hashes.add(record['path'])
        else:
            check(recorded_hash==record['sha256'],'attempt output hash matches '+record['path'])
    return record

fixed = defaultdict(list)
for record in S['attempts']:
    a = record['data']
    if a['stage']=='fixed_duty' and a.get('state')=='finished' and a.get('returncode')==0:
        fixed[a['cell']].append(record)

rows = []
post_sorted = sorted(S['post_records'], key=lambda r:r['data']['idx'])
check(len(post_sorted)==80 and len({p['data']['cell'] for p in post_sorted})==80,'80 unique post cells')
for post in post_sorted:
    p = post['data']; cell = p['cell']
    original = bound_summary(p['fresh_cg_source'])
    check(original is not None,'original fresh summary present '+cell)
    cg = original['data']
    check(not cg.get('fallback',False),'original source is not fallback '+cell)
    original_attempt = attempts[p['fresh_cg_source']+'/attempt.json']['data']
    check(len(fixed[cell])==1,'one terminal fixed-duty attempt '+cell)
    fa = fixed[cell][0]
    fix_record = bound_summary(fa['path'].rsplit('/',1)[0]); fix = fix_record['data']
    for k in ['instance_sha256','tariff_sha256','target_kwh','fleet_cap']:
        check(original_attempt[k]==fa['data'][k]==p[k], 'matched '+k+' '+cell)
    check(cg['inputs']['instance']==fix['inputs']['instance']==p['instance'],'matched instance path '+cell)
    check({k:v for k,v in cg['physics'].items() if k!='coverage'}==
          {k:v for k,v in fix['physics'].items() if k!='coverage'},'matched charging physics '+cell)
    check(cg['physics']['coverage']=='cover' and fix['physics']['coverage']=='exact_once_fixed_duties',
          'explicit different cover/exact-once scope '+cell)
    is_fallback = p['branch']=='fallback_then_cleanup'
    pool_record = bound_summary(p['fallback_attempt']) if is_fallback else original
    pool = pool_record['data']
    if is_fallback:
        check(pool['fallback'] is True,'fallback marker '+cell)
        check(pool['source_fresh_cg']['summary_sha256']==original['sha256'],'fallback original binding '+cell)
        check(pool['source_fixed_duty']['summary_sha256']==fix_record['sha256'],'fallback fixed binding '+cell)
    cleanup_record = bound_summary(p['cleanup_attempt'])
    clean = cleanup_record['data'] if cleanup_record else {}
    if cleanup_record:
        check(clean['source_sha256']==pool_record['sha256'],'cleanup source binding '+cell)
        check(clean['source']==pool_record['path'],'cleanup source path '+cell)
    fs,cs = cg.get('fleet_stage') or {}, cg.get('charging_stage') or {}
    ps,pc = pool.get('fleet_stage') or {}, pool.get('charging_stage') or {}
    origins = pool.get('selected_origin_counts') or {}
    cell_attempts = [r['data'] for r in S['attempts'] if r['data']['cell']==cell]
    row = dict(root_label='expansion',cell=cell,idx=p['idx'],cohort=p['cohort'],
      execution_commit=p['execution_commit'],instance_sha256=p['instance_sha256'],
      tariff_sha256=p['tariff_sha256'],aggregate_terminal_target_kwh=p['target_kwh'],
      fleet_cap=p['fleet_cap'],original_cg_pricing_certified=cg['cg_pricing_certified'],
      original_cg_stop=cg['cg_stop'],original_cg_seconds=cg['cg_seconds'],
      original_cg_weighted_full_graph_lp_bound=cg.get('weighted_full_graph_lp_lower_bound'),
      original_cg_fractional_route_weight=(cg.get('last_iteration') or {}).get('route_weight'),
      original_cg_selection_state=cg.get('selection_state'),
      original_fleet_status=status(fs.get('status')),original_fleet_buses=fs.get('buses'),
      original_finite_pool_fleet_bound=fs.get('bound'),
      original_charging_status=status(cs.get('status')),original_charging_cost=cs.get('cost'),
      original_charging_bound=cs.get('bound'),
      fixed_fleet_status=status(fix.get('fleet_status')),fixed_fleet=fix.get('fleet'),
      fixed_finite_frontier_fleet_bound=fix.get('fleet_bound'),
      fixed_charging_status=status(fix.get('charging_status')),fixed_charging_grid_cost=fix.get('charging_cost'),
      fixed_charging_grid_bound=fix.get('charging_bound'),fixed_continuous_cost=fix.get('continuous_charging_cost'),
      fixed_exact_once_verified=fix.get('exact_once_verified'),fixed_proof_scope=fix.get('proof_scope'),
      post_branch=p['branch'],post_state=p['state'],post_job_id=p['slurm_job_id'],
      source_pool_type='fresh CG + GIRO fixed-duty frontier columns' if is_fallback else 'pure fresh CG',
      fallback_added=is_fallback,fallback_pool_fresh_columns=(pool.get('pool') or {}).get('fresh_input') if is_fallback else None,
      fallback_pool_fixed_frontier_columns=(pool.get('pool') or {}).get('fixed_duty_frontier_input') if is_fallback else None,
      fallback_selected_fresh_only=origins.get('fresh_cg'),fallback_selected_fixed_only=origins.get('fixed_duty_frontier'),
      fallback_selected_both=origins.get('fresh_cg+fixed_duty_frontier'),
      fallback_all_selected_giro_sequences=pool.get('all_selected_are_giro_sequences') if is_fallback else None,
      source_pool_fleet_status=status(ps.get('status')),source_pool_fleet=ps.get('buses'),
      source_pool_charging_status=status(pc.get('status')),source_pool_charging_grid_cost=pc.get('cost'),
      source_pool_charging_grid_bound=pc.get('bound'),
      cleanup_returncode=p.get('cleanup_returncode'),cleanup_summary_present=bool(cleanup_record),
      cleanup_fleet_status=status(clean.get('fleet_status')),cleanup_fleet=clean.get('fleet'),
      cleanup_finite_pool_fleet_bound=clean.get('fleet_bound'),
      cleanup_charging_status=status(clean.get('charging_status')),cleanup_charging_grid_cost=clean.get('charging_cost'),
      cleanup_charging_grid_bound=clean.get('charging_bound'),cleanup_continuous_cost=clean.get('continuous_charging_cost'),
      cleanup_exact_once_verified=clean.get('exact_once_verified'),
      cleanup_individual_replay_verified=clean.get('individual_replay_verified'),
      cleanup_shared_capacity_validated=clean.get('shared_capacity_validated'),cleanup_proof_scope=clean.get('proof_scope'),
      cleanup_input_duplicate_trip_count=clean.get('duplicate_trip_count'),cleanup_generated_sequences=clean.get('generated_sequences'),
      attempt_receipt_count=len(cell_attempts),stale_running_attempt_receipts=sum(a.get('state')=='running' for a in cell_attempts),
      fresh_cg_max_restart=max(int(a.get('slurm_restart',0)) for a in cell_attempts if a['stage']=='fresh_cg'),
      fixed_max_restart=max(int(a.get('slurm_restart',0)) for a in cell_attempts if a['stage']=='fixed_duty'),
      original_summary_path=original['path'],original_summary_sha256=original['sha256'],
      fixed_summary_path=fix_record['path'],fixed_summary_sha256=fix_record['sha256'],
      source_pool_summary_path=pool_record['path'],source_pool_summary_sha256=pool_record['sha256'],
      cleanup_summary_path=cleanup_record['path'] if cleanup_record else None,
      cleanup_summary_sha256=cleanup_record['sha256'] if cleanup_record else None,
      post_receipt_path=post['path'],post_receipt_sha256=post['sha256'])
    rows.append(row)
emit_csv('expansion_cells.csv',rows)

attempt_rows=[]
for record in S['attempts']:
    a=record['data']
    attempt_rows.append(dict(cell=a['cell'],stage=a['stage'],job_id=a['slurm_job_id'],
      array_job_id=a.get('slurm_array_job_id'),array_task_id=a.get('slurm_array_task_id'),
      restart=int(a.get('slurm_restart',0)),state=a.get('state'),returncode=a.get('returncode'),
      started_utc=a.get('started_utc'),finished_utc=a.get('finished_utc'),
      receipt_path=record['path'],receipt_sha256=record['sha256']))
emit_csv('attempt_receipts.csv',attempt_rows)
for a in attempt_rows:
    if a['state']=='running':
        check(a['receipt_path'].replace('/attempt.json','/out/summary.json') not in summaries,
              'stale running attempt has no summary '+a['receipt_path'])
        later=[b for b in attempt_rows if b['cell']==a['cell'] and b['stage']==a['stage']
               and b['restart']>a['restart'] and b['state'] in ('finished','failed')]
        check(bool(later),'stale running attempt has later terminal restart '+a['receipt_path'])

scheduler=[]
for line in S['sacct']:
    fields=line.split('|')
    check(len(fields)==8,'eight accounting fields '+fields[0])
    jid,name,state,exitcode,elapsed,start,end,restarts=fields
    stage='post' if name.startswith('sx_po_') else 'fresh_cg' if name.startswith('sx_cg_') else 'fixed_duty' if name.startswith('sx_fx') else 'original_recovery'
    scheduler.append(dict(root_label='expansion' if name.startswith('sx_') else 'original_recovery',
      job_id=jid,job_name=name,stage=stage,state=state,exit_code=exitcode,elapsed=elapsed,
      start_scheduler_local=start,end_scheduler_local=end,restarts=int(restarts)))
emit_csv('scheduler_jobs.csv',scheduler)
for a in attempt_rows:
    if a['state']=='running': continue
    scheduler_id=f"{a['array_job_id']}_{a['array_task_id']}" if a['array_job_id'] else a['job_id']
    job=next(r for r in scheduler if r['job_id']==scheduler_id)
    check(a['restart']==job['restarts'],'terminal attempt restart matches accounting '+a['receipt_path'])

original_records={r['path']:r for r in S['original_recovery']}
recovery=[]
for r in S['original_recovery']:
    if not r['path'].endswith('/attempt.json'): continue
    a=r['data']; sr=original_records.get(r['path'].replace('/attempt.json','/out/summary.json'))
    d=sr['data'] if sr else {}
    check(sr is not None and a['output_hashes']['summary.json']==sr['sha256'],'original recovery output binding '+a['slurm_job_id'])
    f=d.get('fleet_stage') or {}; c=d.get('charging_stage') or {}
    recovery.append(dict(root_label='original',cell=a['cell'],job_id=a['slurm_job_id'],stage=a['stage'],
      state=a['state'],returncode=a['returncode'],fallback=d.get('fallback',False),
      fleet_status=status(d.get('fleet_status',f.get('status'))),fleet=d.get('fleet',f.get('buses')),
      finite_pool_fleet_bound=d.get('fleet_bound',f.get('bound')),
      charging_status=status(d.get('charging_status',c.get('status'))),charging_grid_cost=d.get('charging_cost',c.get('cost')),
      charging_grid_bound=d.get('charging_bound',c.get('bound')),continuous_cost=d.get('continuous_charging_cost',d.get('selected_continuous_charging_cost')),
      exact_once_verified=d.get('exact_once_verified'),individual_replay_verified=d.get('individual_replay_verified'),
      shared_capacity_validated=d.get('shared_capacity_validated'),all_selected_giro_sequences=d.get('all_selected_are_giro_sequences'),
      raw_proof_scope=d.get('proof_scope'),summary_path=sr['path'],summary_sha256=sr['sha256']))
emit_csv('original_root_recoveries.csv',recovery)

exp_sched=[r for r in scheduler if r['root_label']=='expansion']
totals=dict(collected_utc=S['collected_utc'],root=ROOT,cells=len(rows),
 original_cg_certificates=counts(rows,'original_cg_pricing_certified'),original_cg_stops=counts(rows,'original_cg_stop'),
 original_fleet_status=counts(rows,'original_fleet_status'),original_charging_status=counts(rows,'original_charging_status'),
 fixed_charging_status=counts(rows,'fixed_charging_status'),post_branches=counts(rows,'post_branch'),post_state=counts(rows,'post_state'),
 successful_cleanup_by_branch=counts([r for r in rows if r['cleanup_summary_present']],'post_branch'),
 fallback_charging_status=counts([r for r in rows if r['fallback_added']],'source_pool_charging_status'),
 fallback_original_cg_certificates=counts([r for r in rows if r['fallback_added']],'original_cg_pricing_certified'),
 cleanup_fleet_status=counts(rows,'cleanup_fleet_status'),cleanup_charging_status=counts(rows,'cleanup_charging_status'),
 cleanup_exact_once=counts(rows,'cleanup_exact_once_verified'),cleanup_replay=counts(rows,'cleanup_individual_replay_verified'),
 fallback_all_giro_sequences=counts([r for r in rows if r['fallback_added']],'fallback_all_selected_giro_sequences'),
 scheduler_jobs=len(exp_sched),scheduler_states=counts(exp_sched,'state'),scheduler_restart_sum=sum(r['restarts'] for r in exp_sched),
 scheduler_restarted_jobs=sum(r['restarts']>0 for r in exp_sched),
 scheduler_by_stage={stage:dict(jobs=len(rr),states=counts(rr,'state'),restarts=sum(r['restarts'] for r in rr),restarted_jobs=sum(r['restarts']>0 for r in rr))
   for stage in sorted({r['stage'] for r in exp_sched}) if (rr:=[r for r in exp_sched if r['stage']==stage])},
 attempt_receipts=len(attempt_rows),attempt_states=counts(attempt_rows,'state'),
 attempt_stage_counts=counts(attempt_rows,'stage'),
 original_recovery_jobs=len(recovery),failed_cleanup_diagnosis=S['failed_cleanup_diagnosis'])
(OUT/'summary.json').write_text(json.dumps(totals,indent=2)+'\n')
provenance=dict(source=str(SOURCE),source_sha256=sha(SOURCE),reducer_sha256=sha(Path(__file__)),
 capture_utc=S['collected_utc'],checks_passed=len(checks),checks=checks,
 absent_attempt_output_hashes=sorted(missing_attempt_output_hashes),
 source_hash_note='Original file hashes are receipt-reported; bound against attempt output_hashes and inter-stage source hashes. Raw remote JSON bytes are not in this local receipt bundle.',
 no_network_or_solver=True,no_price_saving_comparison=True,
 outputs={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='provenance.json'})
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(totals,indent=2))

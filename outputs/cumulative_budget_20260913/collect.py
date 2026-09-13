"""Compact completed endpoints and operational state; never reads journals.

Import collect(root, scheduler_states=None), or run with an optional campaign root.
Only manifest cases are visited. Shared endpoints are represented in records, not
repeated in cg/mip. Scheduler state is separate from solver/publication state.
"""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys

STAGES = ('base', 'extra', 'mip_base', 'mip_extra', 'mip_warm')
SCALARS = (str, int, float, bool, type(None))

def scalar_fields(value, keys=None):
    return {k: v for k, v in (value or {}).items()
            if isinstance(v, SCALARS) and (keys is None or k in keys)}

def load(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: '+str(path))
    return value, hashlib.sha256(raw).hexdigest()

def latest(paths):
    for path in sorted(paths, key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True):
        try:
            value, digest = load(path)
            return path, value, digest
        except (OSError, ValueError):
            continue
    return None, {}, None

def process(value):
    return scalar_fields(value, ('status', 'started_utc', 'ended_utc', 'host',
        'wall_s', 'user_cpu_s', 'system_cpu_s', 'children_maxrss_kib',
        'returncode', 'watchdog', 'watchdog_s'))

def two_stage(value):
    if not isinstance(value, dict):
        return value if isinstance(value, SCALARS) else None
    result = scalar_fields(value)
    for key, child in value.items():
        if isinstance(child, dict) and key in ('stage1', 'stage2', 'stage_1', 'stage_2'):
            result[key] = scalar_fields(child)
    return result

def scheduler_snapshot(jobs):
    ids = sorted({str(job) for stages in jobs.values() for job in stages.values()})
    if not ids:
        return {}, {'status': 'no_submitted_jobs'}
    argv = ['/usr/local/slurm/slurm-25.05.5/bin/squeue', '-h', '-j', ','.join(ids), '-o', '%i|%T']
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        if result.returncode:
            return {}, {'status': 'unavailable', 'error': result.stderr.strip()[:500]}
        states = dict(line.split('|', 1) for line in result.stdout.splitlines() if '|' in line)
        return states, {'status': 'available', 'scope': 'squeue current state; absent jobs are not presumed completed'}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, {'status': 'unavailable', 'error': str(exc)[:500]}

def collect(root, scheduler_states=None):
    root = Path(root)
    manifest, manifest_hash = load(root/'manifest.json')
    jobs = load(root/'case_jobs.json')[0] if (root/'case_jobs.json').exists() else {}
    if scheduler_states is None:
        scheduler_states, scheduler_info = scheduler_snapshot(jobs)
    else:
        scheduler_info = {'status': 'provided'}
    output = {'schema': 'evsp-cumulative-budget-collection-v1',
        'collected_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'root': str(root), 'manifest_sha256': manifest_hash,
        'execution_commit': manifest.get('execution_commit'),
        'mip_execution_commit': manifest.get('mip_execution_commit'),
        'physics': manifest.get('settings'), 'warm_reference': manifest.get('warm_reference'), 'source_hashes': {
            'audit_sha256': manifest.get('audit_sha256'),
            'ancestry_sha256': manifest.get('ancestry_sha256'),
            'static_sha256': manifest.get('static_sha256'),
            'tooling_sha256': manifest.get('tooling_sha256')},
        'cg': [], 'mip': [], 'records': [],
        'workflow': {'scheduler': scheduler_info, 'cases': {}}, 'errors': []}
    seen = {'cg': set(), 'mip': set()}
    for cid, case in sorted(manifest['cases'].items()):
        identity = {'case_id': cid, 'chain': case['chain'], 'target_k': case['k'],
            'input_sha256': case['input_sha256'], 'csv': case['csv']}
        budgets = scalar_fields(case, ('fresh_primary_budget_s', 'fresh_graph_sensitivity_budget_s',
            'native_cumulative_wall_s', 'ancestor_external_graph_build_s', 'target_external_graph_build_s',
            'actual_cpu_s', 'allocated_cpu_s', 'distinct_stage_count'))
        record = {**identity, 'budgets': budgets,
            'cache_evidence': scalar_fields(case, ('cache', 'cache_sha256', 'cache_manifest_sha256', 'consumer_cache_manifest_sha256')),
            'warm_baseline': {'status_path': case.get('status_path'), 'status_sha256': case.get('status_sha256'),
                'source_commits': case.get('source_commits'),
                'target_certified_rc_optimal': case.get('target_certified_rc_optimal'),
                'mip': scalar_fields(case.get('warm_target_mip')),
                'mip_missing': case.get('warm_target_mip_missing')}, 'stages': {}}
        operational = {}
        for mode in STAGES:
            stage = root/'cases'/cid/mode
            job = str(jobs.get(cid, {}).get(mode, '')) or None
            scheduler = scheduler_states.get(job) if job else None
            path, state, digest = latest(stage.glob('*/state.json'))
            observed = {'job_id': job, 'scheduler_state': scheduler,
                'state_path': str(path) if path else None, 'state_sha256': digest,
                **scalar_fields(state, ('attempt', 'status', 'started_utc', 'ended_utc', 'error', 'budget_s'))}
            if path:
                execution_path = path.parent/'process/execution.json'
                if execution_path.exists():
                    try:
                        execution, eh = load(execution_path)
                        observed['latest_process'] = {'path': str(execution_path), 'sha256': eh, **process(execution)}
                    except (OSError, ValueError) as exc:
                        output['errors'].append({'path': str(execution_path), 'error': str(exc)})
            totals = {'wall_s': 0.0, 'user_cpu_s': 0.0, 'system_cpu_s': 0.0,
                'measured_attempts': 0, 'unmeasured_attempts': 0,
                'scope': 'Completed process measurement records in this stage, including interruptions; running/missing measurements excluded'}
            for ep in stage.glob('*/process/execution.json'):
                try:
                    ev, _ = load(ep)
                    if all(isinstance(ev.get(k), (int, float)) for k in ('wall_s','user_cpu_s','system_cpu_s')):
                        for k in ('wall_s','user_cpu_s','system_cpu_s'): totals[k] += ev[k]
                        totals['measured_attempts'] += 1
                    else: totals['unmeasured_attempts'] += 1
                except (OSError, ValueError): totals['unmeasured_attempts'] += 1
            observed['process_totals'] = totals
            completion_path = stage/'completion.json'
            summary = {'scheduler_state': scheduler, 'job_id': job}
            if completion_path.exists():
                try:
                    completed, ch = load(completion_path)
                    summary.update(scalar_fields(completed, ('status', 'usable', 'reason', 'optimization_run',
                        'result_path', 'result_sha256', 'journal_sha256', 'budget_s', 'native_wall_s',
                        'budget_overshoot_s', 'certified', 'stop_reason', 'shared_completion')))
                    summary.update(completion_path=str(completion_path), completion_sha256=ch,
                                   publication='completed', authority='published')
                    result_path = completed.get('result_path')
                    if completed.get('status') == 'finished' and completed.get('optimization_run') is True and result_path:
                        result, rh = load(result_path)
                        if rh != completed['result_sha256']:
                            raise ValueError('Published result hash mismatch: '+result_path)
                        kind = 'mip' if mode.startswith('mip_') else 'cg'
                        resolved = str(Path(result_path).resolve())
                        key = (resolved, rh)
                        if key not in seen[kind]:
                            seen[kind].add(key)
                            arm = mode.removeprefix('mip_')
                            row = {**identity, 'budget_arm': arm,
                                'budget_s': ((manifest.get('settings') or {}).get('mip_s', 3600) if kind=='mip' else case['fresh_primary_budget_s'] if arm=='base' else case['fresh_graph_sensitivity_budget_s']),
                                'path': result_path, 'resolved_source_path': resolved, 'sha256': rh,
                                'completion_sha256': ch, 'authority': 'published',
                                'execution_commit': manifest.get('mip_execution_commit' if kind=='mip' else 'execution_commit'),
                                'process': process(completed.get('execution')), 'stage_process_totals': totals}
                            if kind == 'cg':
                                row.update(scalar_fields(result, ('wall_s','attempt_wall_s','iterations','attempt_iterations',
                                    'certified_rc_optimal','stop_reason','master_sense','g_kwh','charge_kw',
                                    'soc_step','block_min','min_soc_frac','time_model','initial_pool')))
                                row['final'] = scalar_fields(result.get('final'))
                                row['usable'] = completed.get('usable')
                                row['journal_sha256'] = completed.get('journal_sha256')
                                row['provenance'] = scalar_fields(result.get('provenance'), ('git_commit',
                                    'instance_sha256','prices_sha256','reference_sha256','deadhead_sha256',
                                    'pricing_certificate_scope','rc_eps'))
                            else:
                                row.update(scalar_fields(result, ('buses','fleet_bound','fleet_proven','status','status_name',
                                    'mip_gap','optimal_scope','physical_replay_validated','physical_replay_scope',
                                    'duplicate_trip_removal_validated','cross_route_charger_capacity_validated',
                                    'pool_columns','runtime_s','gurobi_optimize_wall_s','charging_cost',
                                    'continuous_realized_charging_cost','source_cg_iterations','source_cg_wall_s')))
                                row['cg_budget_s'] = case.get('native_cumulative_wall_s') if arm=='warm' else case['fresh_primary_budget_s'] if arm=='base' else case['fresh_graph_sensitivity_budget_s']
                                row['two_stage'] = two_stage(result.get('two_stage'))
                                row['source_status_sha256'] = completed.get('source_status_sha256')
                                row['source_journal_sha256'] = completed.get('source_journal_sha256')
                            output[kind].append(row)
                            if mode == 'mip_warm':
                                record['target_reference_mip'] = row
                    elif completed.get('status', '').startswith('shared_primary'):
                        summary['alias_of_budget_arm'] = 'base'
                except (OSError, ValueError, KeyError) as exc:
                    summary.update(publication='invalid', authority='unverified', error=str(exc))
                    output['errors'].append({'path': str(completion_path), 'error': str(exc)})
            else:
                # No output checkpoint is treated as a scientific endpoint.
                if scheduler == 'PENDING': status = 'pending'
                elif scheduler in ('RUNNING','COMPLETING','CONFIGURING'): status = 'running'
                elif scheduler: status = 'scheduler_'+scheduler.lower()
                elif state.get('status') == 'execution_failed': status = 'failed'
                elif path: status = 'unpublished_scheduler_unknown'
                elif job: status = 'not_started_scheduler_unknown'
                else: status = 'not_submitted'
                summary.update(status=status, publication='absent', authority='operational')
            record['stages'][mode] = summary
            operational[mode] = observed
        output['records'].append(record)
        output['workflow']['cases'][cid] = operational
    return output

if __name__ == '__main__':
    root = Path(sys.argv[1]) if len(sys.argv)>1 else Path('/home/nc437/ladder-lite/cumulative_budget_20260913')
    print(json.dumps(collect(root), separators=(',', ':')))

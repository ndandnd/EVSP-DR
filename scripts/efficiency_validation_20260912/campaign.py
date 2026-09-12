#!/usr/bin/env python3
"""Authenticated, paired efficiency validation. Only launch --submit calls sbatch."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

PEAK12_SHA = '8b231a2574fd4e3b4dc94873ad2d6515bfaba09e07afefbb1df43d5f775a8381'

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def read(path):
    return json.loads(Path(path).read_text())

def authenticate(path, expected):
    path = Path(path).resolve()
    observed = digest(path)
    if observed != expected:
        raise ValueError(f'hash mismatch {path}: expected {expected}, got {observed}')
    return {'path': str(path), 'sha256': observed, 'bytes': path.stat().st_size}

def code_check(path, expected):
    path = Path(path).resolve()
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    pin = subprocess.check_output(['git', '-C', str(path), 'rev-parse', expected + '^{commit}'], text=True).strip()
    if actual != pin:
        raise ValueError(f'checkout {path}: {actual} != {pin}')
    if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError(f'tracked dirty checkout {path}')
    return actual

def freeze(source, target, expected):
    identity = authenticate(source, expected)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        authenticate(target, expected)
    else:
        tmp = target.with_name(target.name + '.copying')
        shutil.copyfile(source, tmp)
        authenticate(tmp, expected)
        os.replace(tmp, target)
    return {**authenticate(target, expected), 'source': identity}

def prepare(args):
    root = args.root.resolve()
    if (root / 'manifest.json').exists():
        raise RuntimeError('manifest already exists; preserve it and use a distinct campaign root')
    baseline, capacity = args.baseline_code.resolve(), args.capacity_code.resolve()
    bp, cp = code_check(baseline, args.baseline_commit), code_check(capacity, args.capacity_commit)
    ev = read(args.audit / 'source_evidence.json')
    ce = read(args.audit / 'capacity_evidence.json')
    frozen = []
    policy_path = root.parent / 'SCAGLIONE_RESOURCE_POLICY.md'
    policy = authenticate(policy_path, digest(policy_path))
    policy['text'] = policy_path.read_text()
    frozen.append(policy)
    for path in [args.audit / 'source_evidence.json', args.audit / 'capacity_evidence.json']:
        frozen.append(authenticate(path, digest(path)))
    # These source identities come from audited completed runs, not current code guesses.
    for code, prov in [(baseline, ev['w1_k08']['provenance']),
                       (capacity, ce['k1_duty13406__capacity']['provenance'])]:
        for name, key in [('Ref_dict.csv', 'reference_sha256'), ('par_ref_dhd.csv', 'deadhead_sha256')]:
            frozen.append(authenticate(code / 'data' / name, prov[key]))
    flat = freeze(baseline / 'data/hourly_prices_flat.csv', root / 'inputs/flat.csv',
                  ev['w1_k08']['provenance']['prices_sha256'])
    frozen.append(flat)
    frozen.append(authenticate(capacity / 'data/hourly_prices_flat.csv', flat['sha256']))
    peak = freeze(capacity / 'data/tariff_response/peak12_h26.csv', root / 'inputs/peak12.csv', PEAK12_SHA)
    frozen.append(peak)
    parents = {}
    for parent in ['w1_k07', 'w4_k10', 'w6_k11']:
        e = ev[parent]
        original_status = Path(e['columns_journal']).parent / 'cg.json'
        raw = freeze(original_status, root / 'inputs' / parent / 'raw_status.json', e['status_sha256'])
        journal = freeze(e['columns_journal'], root / 'inputs' / parent / 'columns.jsonl', e['pool_sha256'])
        frozen.extend([raw, journal])
        # Importer intentionally retains the descriptor's original CSV identity.
        frozen.append(authenticate(baseline / 'data' / e['csv'], e['provenance']['instance_sha256']))
        desc = read(raw['path'])
        if desc.get('certified_rc_optimal') is not True:
            raise ValueError(f'parent {parent} is not certified')
        desc['columns_journal'] = journal['path']
        dest = root / 'inputs' / parent / 'descriptor.json'
        write(dest, desc)
        frozen.append(authenticate(dest, digest(dest)))
        parents[parent] = {'descriptor': str(dest), 'descriptor_sha256': digest(dest),
                           'raw_status': raw, 'journal': journal}
    cases = []
    selections = [('d00_g0', None), ('d00_g1', None), ('w1_k08', 'w1_k07'),
                  ('w4_k11', 'w4_k10'), ('w6_k12', 'w6_k11'), ('w1_k08_repeat', 'w1_k07')]
    for name, parent in selections:
        source = name.removesuffix('_repeat')
        e = ev[source]
        instance = freeze(baseline / 'data' / e['csv'], root / 'inputs' / (source + '.csv'),
                          e['provenance']['instance_sha256'])
        frozen.append(instance)
        warm = parent is not None
        cases.append(dict(id=name, kind='warm' if warm else 'fresh', code=str(baseline), commit=bp,
                          instance=instance['path'], prices=flat['path'], source_args=e['args'],
                          parent=parents[parent] if warm else None, cpus=8 if warm else 2,
                          mem='96G' if warm else '32G', slurm_time='08:00:00' if warm else '04:30:00',
                          arm_seconds=7200, preparation_seconds=10800 if warm else 0))
    for name, source, arm, tariff in [
        ('cap_k1', 'k1_duty13406__capacity', 'capacity', flat),
        ('cap_k2', 'e1_short_k2__capacity', 'capacity', flat),
        ('cap_k1_combined_peak12', 'k1_duty13406__capacity', 'combined', peak)]:
        prov = ce[source]['provenance']
        instance = freeze(prov['instance'], root / 'inputs' / (source + '.csv'), prov['instance_sha256'])
        frozen.append(instance)
        cases.append(dict(id=name, kind='capacity', code=str(capacity), commit=cp,
                          instance=instance['path'], prices=tariff['path'], arm=arm,
                          cpus=1, mem='24G', slurm_time='06:30:00', arm_seconds=10800,
                          preparation_seconds=0, source_provenance=prov))
    for i, case in enumerate(cases):
        case['order'] = ['reference', 'optimized'] if i % 2 == 0 else ['optimized', 'reference']
    manifest = dict(schema='evsp-efficiency-validation-v1', prepared_utc=now(), root=str(root),
                    python=str(args.python), baseline_commit=bp, capacity_commit=cp, cases=cases,
                    frozen=frozen, parents=parents, policy=dict(partition='default_partition',
                    exclude='scaglione-compute-01', independent_cases=len(cases), concurrency=len(cases),
                    requeue=False, old_jobs_untouched=True),
                    interpretation='Exit status is scheduler/process evidence, never a CG certificate. '
                    'Both capacity selectors use corrected accounting. Warm worker completion order is not canonicalized. '
                    'One cache preparation per warm allocation is outside paired timing; runtime index setup remains inside arm timing.',
                    resources_reason='Same per-case resources as audited successful campaigns; warm parent RSS excludes worker aggregate. '
                    '7200s baseline arms cover historical 1663–3877s; capacity10800s includes historical8598s pricing call. '
                    'All nine independent cases eligible; no artificial throttle.')
    write(root / 'manifest.json', manifest)
    print(json.dumps({'manifest': str(root / 'manifest.json'), 'sha256': digest(root / 'manifest.json'), 'cases': len(cases)}))

def baseline_command(case, output, cache=None, prepare_only=False):
    a = case['source_args']
    cmd = ['src/exact_pricer_expanded.py', '--csv', case['instance'], '--prices_csv', case['prices']]
    keys = [('soc_step','--soc-step'), ('block_min','--block-min'), ('g_kwh','--g-kwh'),
            ('charge_kw','--charge-kw'), ('min_soc_frac','--min-soc-frac'),
            ('max_iters','--max-iters'), ('columns_per_iter','--columns_per_iter'),
            ('column_selection','--column-selection'), ('column_diversity_weight','--column-diversity-weight'),
            ('column_candidate_multiplier','--column-candidate-multiplier'), ('rc_eps','--rc-eps'),
            ('master_sense','--master-sense'), ('master_backend','--master-backend'),
            ('initial_pool','--initial-pool'), ('time_model','--time-model'),
            ('event_arc_mode','--event-arc-mode'), ('checkpoint_every','--checkpoint-every')]
    for key, flag in keys:
        cmd += [flag, str(a[key])]
    cmd += ['--wall-limit-s', str(case['preparation_seconds'] if prepare_only else case['arm_seconds']),
            '--gurobi-log', str(output / 'gurobi.log'), '--phase-telemetry', str(output / 'phases.jsonl')]
    # Cache-only CLI rejects --out; its durable product is the authenticated
    # cache/manifest, with process and phase evidence kept by the wrapper.
    if not prepare_only:
        cmd += ['--out', str(output / 'cg.json')]
    if a.get('strict_tariff_coverage'):
        cmd += ['--strict-tariff-coverage']
    if cache:
        cmd += ['--event-network-cache', str(cache), '--event-network-cache-mode',
                'build-or-load' if prepare_only else 'require']
    if prepare_only:
        cmd += ['--event-network-cache-only']
    elif case['kind'] == 'warm':
        cmd += ['--inherit-event-pool-from', case['parent']['descriptor'],
                '--inherit-max-columns', '512', '--inherit-time-limit-s', '900', '--inherit-event-pool-workers', '8']
    return cmd

def command(case, mode, output, cache=None):
    if case['kind'] != 'capacity':
        cmd = baseline_command(case, output, cache)
        if mode == 'optimized':
            cmd += ['--skip-gurobi-incidence']
            if case['kind'] == 'warm':
                cmd += ['--fixed-sequence-index']
        return cmd
    return ['src/run_capacity_speed_event_cg.py', '--mode', 'cg', '--arm', case['arm'],
            '--instance', case['instance'], '--prices', case['prices'], '--reference-data-dir', str(Path(case['code']) / 'data'),
            '--out', str(output / 'cg.json'), '--pool-out', str(output / 'pool.jsonl'),
            '--expected-commit', case['commit'], '--require-clean', '--battery-kwh', '240',
            '--non-parx-kw', '240', '--reserve-kwh', '0', '--soc-step', '2.5', '--block-min', '5',
            '--rc-eps', '1e-5', '--max-iters', '10000', '--threads', '1',
            '--cg-wall-s', str(case['arm_seconds']), '--capacity-selector',
            'reference' if mode == 'reference' else 'prefix-memo']

def run_process(cmd, cwd, output, limit, env):
    output.mkdir(parents=True, exist_ok=False)
    record = dict(command=cmd, cwd=str(cwd), started_utc=now(), watchdog_seconds=limit,
                  host=socket.getfqdn(), env={k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED']})
    write(output / 'execution.json', record)
    started = time.monotonic()
    with (output / 'stdout.log').open('wb') as stdout, (output / 'stderr.log').open('wb') as stderr:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        try:
            proc.wait(timeout=limit)
            record['watchdog_triggered'] = False
        except subprocess.TimeoutExpired:
            record['watchdog_triggered'] = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    record.update(ended_utc=now(), wall_seconds=time.monotonic()-started, returncode=proc.returncode)
    record['artifacts'] = [dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)
                           for p in output.rglob('*') if p.is_file() and p.name!='execution.json']
    write(output / 'execution.json', record)
    return record

def worker(args):
    manifest = read(args.root / 'manifest.json')
    case = manifest['cases'][args.index]
    code_check(case['code'], case['commit'])
    for f in manifest['frozen']:
        authenticate(f['path'], f['sha256'])
    token = os.environ.get('SLURM_JOB_ID', 'local') + '_r' + os.environ.get('SLURM_RESTART_COUNT', '0')
    attempt = args.root / 'cases' / case['id'] / token
    attempt.mkdir(parents=True, exist_ok=False)
    allocation = dict(case=case, started_utc=now(), host=socket.getfqdn(), manifest_sha256=digest(args.root / 'manifest.json'),
                      slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')})
    write(attempt / 'allocation.json', allocation)
    env = os.environ.copy()
    for key in ['PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LM_LICENSE_FILE']:
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
               GRB_LICENSE_FILE='/share/apps/software/gurobi/gurobi.lic')
    cache = attempt / 'network.pkl' if case['kind']=='warm' else None
    python = manifest['python']
    if cache:
        prep = attempt / 'prepare'
        result = run_process([python, '-u'] + baseline_command(case, prep, cache, True),
                             case['code'], prep, case['preparation_seconds'], env)
        if result['returncode'] != 0 or not cache.exists():
            write(attempt / 'pair_status.json', dict(status='preparation_failed', preparation=result))
            return 1
        write(attempt / 'cache_identity.json', {'files': [dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)
              for p in attempt.glob('network.pkl*') if p.is_file()], 'source_commit':case['commit'],
              'preparation_separate_from_pair':True})
    results = {}
    # Reserve complete arm budget plus watchdog grace; never silently truncate one arm.
    end_epoch = int(os.environ.get('SLURM_JOB_END_TIME', '0') or '0')
    for mode in case['order']:
        if end_epoch and end_epoch - time.time() < case['arm_seconds'] + 180:
            results[mode] = {'status':'not_started_insufficient_allocation_time'}
            continue
        out = attempt / mode
        try:
            results[mode] = run_process([python, '-u'] + command(case, mode, out, cache),
                                        case['code'], out, case['arm_seconds'] + 60, env)
        except Exception as exc:
            results[mode] = {'status':'execution_error', 'error':repr(exc), 'ended_utc':now()}
        write(attempt / 'pair_status.json', dict(status='running', order=case['order'], results=results))
    write(attempt / 'pair_status.json', dict(status='finished', ended_utc=now(), order=case['order'], results=results,
          note='Read cg.json certification fields independently; process completion does not imply optimality.'))
    return 0 if all(r.get('returncode') == 0 for r in results.values()) else 1

def launch(args):
    root = args.root.resolve()
    manifest = read(root / 'manifest.json')
    script = Path(__file__).resolve()
    jobs_path = root / 'jobs.json'
    commands = []
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(manifest['cases']):
        commands.append(['sbatch', '--parsable', '--partition=default_partition', '--exclude=scaglione-compute-01',
                         '--cpus-per-task='+str(c['cpus']), '--mem='+c['mem'], '--time='+c['slurm_time'],
                         '--no-requeue', '--job-name=eff_'+c['id'], '--output='+str(root/'logs/%x_%j.out'),
                         '--error='+str(root/'logs/%x_%j.err'), '--chdir='+str(root),
                         '--wrap='+shlex.join([manifest['python'], str(script), 'worker', '--root', str(root), '--index', str(i)])])
    if not args.submit:
        print(json.dumps({'dry_run': True, 'commands':commands}, indent=2))
        return
    # Atomic reservation prevents duplicate or concurrent submissions. Partial failures stay recorded.
    fd = os.open(jobs_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.close(fd)
    ledger = dict(started_utc=now(), manifest_sha256=digest(root/'manifest.json'), jobs=[])
    write(jobs_path, ledger)
    for case, cmd in zip(manifest['cases'], commands):
        entry = dict(case=case['id'], command=cmd, submission_started_utc=now())
        ledger['jobs'].append(entry)
        write(jobs_path, ledger)
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, check=True)
            entry.update(submit_stdout=result.stdout, submit_stderr=result.stderr,
                         job_id=result.stdout.strip().split(';')[0], submitted_utc=now())
            write(jobs_path, ledger)  # Record ID before any further operation.
            check = subprocess.check_output(['scontrol','show','job',entry['job_id']], text=True)
            entry['scontrol'] = check
            entry['exclusion_verified'] = 'ExcNodeList=scaglione-compute-01' in check
            entry['partition_verified'] = 'Partition=default_partition' in check
            if not (entry['exclusion_verified'] and entry['partition_verified']):
                raise RuntimeError('effective physical node exclusion not verified; stopping further submissions')
        except BaseException as exc:
            entry['error'] = repr(exc)
            write(jobs_path, ledger)
            raise
        write(jobs_path, ledger)
    ledger['finished_utc'] = now()
    write(jobs_path, ledger)
    print(json.dumps(ledger, indent=2))

def observe(path, hash_content=True):
    """Hash an available artifact without implying that a running file is frozen."""
    path = Path(path)
    result = {'path':str(path), 'exists':path.is_file()}
    if not result['exists']:
        return result
    try:
        before = path.stat()
        result.update(bytes=before.st_size, mtime_ns=before.st_mtime_ns, content_hash_recomputed=hash_content)
        if hash_content:
            result['sha256']=digest(path)
        after = path.stat()
        result['changed_during_hash'] = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
    except OSError as exc:
        result['read_error'] = repr(exc)
    return result


def read_available(path):
    if not Path(path).is_file():
        return {}
    try:
        return read(path)
    except (OSError, ValueError) as exc:
        return {'read_error':repr(exc)}


def command_flags(command):
    flags = {}
    for i, value in enumerate(command or []):
        if str(value).startswith('--'):
            flags[value] = command[i+1] if i+1 < len(command) and not str(command[i+1]).startswith('--') else True
    return flags


def execution_summary(value):
    if not value:
        return None
    keys = ['status','started_utc','ended_utc','wall_seconds','watchdog_seconds',
            'watchdog_triggered','returncode','host','cwd','env','error','read_error']
    result = {k:value[k] for k in keys if k in value}
    result['flags'] = command_flags(value.get('command'))
    return result


def collect(args):
    root = args.root.resolve()
    manifest = read_available(root/'manifest.json')
    manifest_cases = {c['id']:c for c in manifest.get('cases',[])}
    rows, attempts = [], []
    seen_cases = set()
    for allocation_path in sorted((root/'cases').glob('*/*/allocation.json')):
        directory = allocation_path.parent
        allocation = read_available(allocation_path)
        case_name = directory.parent.name
        case = allocation.get('case') or manifest_cases.get(case_name,{})
        seen_cases.add(case_name)
        pair = read_available(directory/'pair_status.json')
        preparation = read_available(directory/'prepare/execution.json')
        cache_identity = read_available(directory/'cache_identity.json')
        current_cache = [observe(p,hash_content=False) for p in sorted(directory.glob('network.pkl*')) if p.is_file()]
        recorded_cache_hashes = {f['path']:f.get('sha256') for f in cache_identity.get('files',[])}
        for item in current_cache:
            item['recorded_preparation_sha256']=recorded_cache_hashes.get(item['path'])
        slurm = allocation.get('slurm',{})
        attempt = dict(case=case_name, attempt=directory.name,
                       pair_status=pair.get('status','not_written'), order=pair.get('order',case.get('order')),
                       allocation=observe(allocation_path), started_utc=allocation.get('started_utc'),
                       host=allocation.get('host'), manifest_sha256=allocation.get('manifest_sha256'),
                       resources={k:case.get(k) for k in ['cpus','mem','slurm_time','arm_seconds','preparation_seconds']},
                       slurm={k:v for k,v in slurm.items() if k in ['SLURM_JOB_ID','SLURM_JOB_NAME',
                              'SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_RESTART_COUNT',
                              'SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE','SLURM_JOB_END_TIME']},
                       code=case.get('code'), commit=case.get('commit'), kind=case.get('kind'),
                       pair_status_artifact=observe(directory/'pair_status.json'),
                       preparation=execution_summary(preparation),
                       preparation_artifacts={k:observe(directory/'prepare'/v,hash_content=k not in {'stdout','stderr'}) for k,v in
                          [('execution','execution.json'),('status','cg.json'),('phases','phases.jsonl'),
                           ('stdout','stdout.log'),('stderr','stderr.log')]},
                       cache_identity=cache_identity, cache_identity_artifact=observe(directory/'cache_identity.json'),
                       current_cache=current_cache,
                       input_paths={'instance':case.get('instance'),'prices':case.get('prices'),
                                    'parent_descriptor':(case.get('parent') or {}).get('descriptor')})
        if allocation.get('read_error'):
            attempt['read_error']=allocation['read_error']
        attempts.append(attempt)
        for mode in ['reference','optimized']:
            arm = directory/mode
            live_execution = read_available(arm/'execution.json')
            saved_execution = pair.get('results',{}).get(mode)
            execution = saved_execution or live_execution
            status_file = arm/'cg.json'
            status = read_available(status_file)
            prov = status.get('provenance') or {}
            final = status.get('final') or {}
            final_lp = status.get('final_lp') or {}
            pool_path = status.get('columns_journal') or status.get('pool') or str(arm/('pool.jsonl' if case.get('kind')=='capacity' else 'cg.json.columns.jsonl'))
            pool_path = Path(pool_path)
            if not pool_path.is_absolute():
                pool_path = arm/pool_path
            if execution and execution.get('status'):
                state = execution['status']
            elif execution and 'returncode' in execution:
                state = 'process_finished'
            elif live_execution:
                state = 'started_no_completion_record'
            else:
                state = 'not_started'
            record = dict(case=case_name, attempt=directory.name, mode=mode,
                          pair_status=pair.get('status','not_written'), process_state=state,
                          execution_source='pair_status' if saved_execution else ('execution.json' if live_execution else None),
                          execution=execution_summary(execution), flags=command_flags((execution or {}).get('command')),
                          artifacts={k:observe(p,hash_content=k not in {'stdout','stderr'}) for k,p in [('execution',arm/'execution.json'),
                            ('status',status_file),('pool',pool_path),('phases',arm/'phases.jsonl'),
                            ('stdout',arm/'stdout.log'),('stderr',arm/'stderr.log')]},
                          certified_rc_optimal=status.get('certified_rc_optimal'),
                          stop_reason=status.get('stop_reason'), solver_status=status.get('status'),
                          status_schema=status.get('schema'),
                          pricing_certificate_scope=status.get('pricing_certificate_scope',prov.get('pricing_certificate_scope')),
                          continuous_cost_pricing_certified=status.get('continuous_cost_pricing_certified',prov.get('continuous_cost_pricing_certified')),
                          terminal_exact_min_reduced_cost=status.get('terminal_exact_min_reduced_cost'),
                          rc_tolerance=(prov.get('rc_eps') if prov.get('rc_eps') is not None else (1e-5 if case.get('kind')=='capacity' else None)),
                          final={k:v for k,v in final.items() if not isinstance(v,(list,dict))},
                          final_lp={k:final_lp.get(k) for k in ['objective','route_weight','artificial_total',
                             'source','iteration','pool_columns','max_row_violation','max_bound_violation'] if k in final_lp},
                          inherited_event_pool_audit=status.get('inherited_event_pool_audit'),
                          physics=status.get('physics'), capacity_selector=status.get('capacity_selector'),
                          runtime_s=status.get('runtime_s',status.get('wall_s')),
                          network_build_s=status.get('network_build_s'),
                          provenance={k:prov.get(k) for k in ['git_commit','git_dirty','git_tracked_dirty',
                             'instance_sha256','prices_sha256','reference_sha256','deadhead_sha256',
                             'inherited_event_pool_status_sha256','pricing_cost_semantics'] if k in prov},
                          finite_pool_mip_proof=None, physical_validation=None, giro_target_attainment=None)
            history = status.get('iterations')
            record['iterations'] = len(history) if isinstance(history,list) else history
            if isinstance(history,list):
                record['pricing_seconds_completed_iterations'] = sum(i.get('pricing_s',0) for i in history)
                record['master_seconds_completed_iterations'] = sum(i.get('lp_solve_s',0) for i in history)
                record['iterations_with_capacity_duals'] = sum(bool(i.get('nonzero_capacity_duals')) for i in history)
            if status.get('read_error'):
                record['read_error']=status['read_error']
            if case.get('kind') in {'warm','fresh','capacity'}:
                record['planned_flags'] = command_flags(command(case,mode,arm,directory/'network.pkl' if case['kind']=='warm' else None))
            rows.append(record)
    pending = [{'case':name,'state':'no_allocation_record','kind':case.get('kind'),'order':case.get('order')}
               for name,case in manifest_cases.items() if name not in seen_cases]
    inputs = []
    for item in manifest.get('frozen',[]):
        observation = observe(item['path'])
        observation['expected_sha256']=item['sha256']
        observation['matches_expected']=observation.get('sha256')==item['sha256'] if observation.get('sha256') else None
        inputs.append(observation)
    print(json.dumps(dict(schema='evsp-efficiency-collection-v2',collected_utc=now(),
                         manifest=observe(root/'manifest.json'), baseline_commit=manifest.get('baseline_commit'),
                         capacity_commit=manifest.get('capacity_commit'), inputs=inputs,
                         pending_cases=pending, attempts=attempts, rows=rows,
                         interpretation='No scheduler query. Started without completion is not proof of current liveness. '
                         'Missing certificate fields are unknown. Artifact hashes describe this read; changing files are flagged.'),indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action', required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--baseline-code', type=Path, required=True)
    p.add_argument('--capacity-code', type=Path, required=True)
    p.add_argument('--baseline-commit', required=True)
    p.add_argument('--capacity-commit', required=True)
    p.add_argument('--python', type=Path, default=Path('/home/nc437/evsp_env/bin/python'))
    for name in ['worker','launch','collect']:
        p=sub.add_parser(name); p.add_argument('--root',type=Path,required=True)
        if name=='worker':p.add_argument('--index',type=int,required=True)
        if name=='launch':p.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    return {'prepare':prepare,'worker':worker,'launch':launch,'collect':collect}[args.action](args)

if __name__=='__main__':
    sys.exit(main())

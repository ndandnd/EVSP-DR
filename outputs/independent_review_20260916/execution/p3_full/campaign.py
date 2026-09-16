"""Frozen k16–25 continuation: independent graphs, sequential CG, per-case MIPs."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

B = Path('/home/nc437/ladder-lite/review_full40_20260916')
D = Path('/share/scaglione/nc437/evsp-dr/review_full40_20260916')
PARENT = Path('/home/nc437/ladder-lite/full_pool_recovery_20260912')
SOURCE = Path('/home/nc437/ladder-lite/graph_recovery_20260912/code')
COMMIT = 'a0e0bb7681c8451e3cbbbfa06aef390026d9af4b'
MIP_COMMIT = '871d057e1067411f09581e37d78f7c1ca43f68bb'
MIP = Path('/home/nc437/ladder-lite/execution')/MIP_COMMIT
PY = '/home/nc437/evsp_env/bin/python'
SLURM = '/usr/local/slurm/slurm-25.05.5/bin/'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def git(*args, root=None):
    return subprocess.check_output(['git', '-C', str(root or B/'code'), *args], text=True).strip()


def read(path):
    return json.loads(Path(path).read_text())


def historical_cg_payload_sha(path):
    """Match the dated collector's compact payload hash, not file bytes."""
    value = read(path)
    row = {k: x for k, x in value.items() if k not in
           ['routes', 'columns', 'selected_routes', 'iterations', 'history', 'iteration_log']
           and not isinstance(x, list)}
    row['path'] = str(path)
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare():
    assert not (B/'manifest.json').exists(), 'Already prepared; do not overwrite'
    D.mkdir(parents=True, exist_ok=True); (D/'cases').mkdir(exist_ok=True)
    (B/'logs').mkdir(exist_ok=True)
    (B/'cases').symlink_to(D/'cases', target_is_directory=True)
    subprocess.run(['git', 'clone', '--shared', '--no-checkout', str(SOURCE), str(B/'code')], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(B/'code'), 'checkout', '--detach', COMMIT], check=True, capture_output=True)
    assert git('rev-parse', 'HEAD') == COMMIT
    assert git('diff', '--name-only', 'e091a4d', COMMIT, '--', 'src') == 'src/event_pricer_network.py'
    data = B/'code/data'
    dependencies = {}
    for name in ['hourly_prices_flat.csv', 'Ref_dict.csv', 'par_ref_dhd.csv']:
        shutil.copy2(PARENT/'code/data'/name, data/name)
        dependencies[name] = sha(data/name)
    source = read(B/'inputs/manifest.json')
    parents = {}
    for chain in range(1, 7):
        status = PARENT/'cases'/f'w{chain}_k15'/'cg.json'
        value = read(status)
        assert value['certified_rc_optimal'] and value['final']['artificials'] == 0
        previous = PARENT/'code/data'/value['csv']; target = data/value['csv']
        frozen_parent = source['chains'][str(chain)]
        assert str(previous) == frozen_parent['parent_remote_csv']
        assert str(status) == frozen_parent['parent_status']
        assert sha(previous) == frozen_parent['parent_sha256']
        assert historical_cg_payload_sha(status) == frozen_parent['parent_cg_payload_sha256']
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(previous, target)
        assert sha(target) == value['provenance']['instance_sha256']
        parents[str(chain)] = {'status': str(status), 'status_sha256': sha(status),
            'journal': value['columns_journal'], 'journal_sha256': sha(value['columns_journal']),
            'csv': value['csv'], 'csv_sha256': sha(target)}
    cases = {}
    for chain, ids in source['warm_chains'].items():
        for pos, cid in enumerate(ids):
            spec = source['cases'][cid]
            rel = 'scale_ladder/instances/chain_extension_20260913/'+spec['csv']
            target = data/rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(B/'inputs'/spec['csv'], target)
            assert sha(target) == spec['input_sha256']
            cases[cid] = {**spec, 'id': cid, 'csv': rel,
                'parent_status': parents[chain]['status'] if pos == 0 else str(B/'cases'/ids[pos-1]/'cg.json'),
                'cg_seconds': 14400, 'cache': str(B/'cases'/cid/'network.pkl')}
    settings = {'battery_kwh': 240, 'initial_soc_kwh': 240, 'charge_kw': 240,
        'reserve_kwh': 0, 'terminal_floor': None, 'shared_station_capacity': False,
        'master_sense': 'cover', 'objective': '100000 + electricity + 5 per charge start',
        'tariff': 'flat', 'soc_step_kwh': 2.5, 'block_min': 5, 'columns_per_iter': 30,
        'rc_epsilon': 0.0001, 'inherit_max_columns': 0, 'inherit_time_limit_s': 0,
        'inherit_workers': 8, 'fixed_sequence_index': True, 'skip_gurobi_incidence': False,
        'mip_seconds': 3600, 'stage1_seconds': 1800, 'stage2_fleet_constraint': '<= validated incumbent'}
    v = {'schema': 'evsp-chain-extension-launch-v1', 'prepared_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'execution_commit': COMMIT, 'mip_execution_commit': MIP_COMMIT,
        'input_manifest_sha256': sha(B/'inputs/manifest.json'), 'input_generation': source,
        'initial_parents': parents, 'cases': cases, 'warm_chains': source['warm_chains'],
        'scientific_settings': settings, 'data_sha256': dependencies,
        'tooling_sha256': {p: sha(B/p) for p in ['campaign.py', 'graph_entry.py', 'worker.sub']},
        'graph_seconds': 43200, 'resources': {
            'cache': {'cpus': 2, 'mem': '64G', 'wall': '12:30:00'},
            'cg': {'cpus': 8, 'mem': '96G', 'wall': '05:00:00'},
            'mip': {'cpus': 8, 'mem': '24G', 'wall': '02:00:00'}},
        'cache_array_concurrency': 50, 'storage_root': str(D),
        'resource_basis': 'Parent750-trip graph used25.4GiB peak; graphs throughk25 have551–624 trips.64GiB requests allow headroom. CG96GiB and MIP24GiB retain successful k15 requests. Slurm admits jobs by real resources; six CG chains are true data dependencies.',
        'graph_change': 'Only source diff from baseline e091a4d is validated deferred JSON tie-key evaluation; original graph winner/order rule unchanged. Coarse synchronous progress wrapper is separate tooling.',
        'requeue_policy': 'Automatic scheduler preemption requeue uses unique stage/job/restart directories. CG resumes a copied identity-validated checkpoint; graph starts fresh if unpublished; MIP starts a fresh tree. No blind retry of algorithmic/time-limit failure.',
        'borrowed_object_store': str(SOURCE/'.git/objects')}
    save(B/'manifest.json', v)
    print(json.dumps({'prepared_cases': len(cases), 'manifest_sha256': sha(B/'manifest.json')}))


def cg_argv(v, case, out, cache_only=False):
    args = ['--csv', case['csv'], '--prices_csv', 'hourly_prices_flat.csv', '--time-model', 'event',
        '--event-arc-mode', 'lazy', '--event-network-cache', case['cache'],
        '--event-network-cache-mode', 'build-or-load' if cache_only else 'require',
        '--fixed-sequence-index', '--soc-step', '2.5', '--block-min', '5', '--max-iters', '100000',
        '--columns_per_iter', '30', '--column-selection', 'reduced_cost', '--column-diversity-weight', '0.0',
        '--column-candidate-multiplier', '4', '--rc-eps', '0.0001', '--master-sense', 'cover',
        '--master-backend', 'gurobi', '--initial-pool', 'singletons',
        '--wall-limit-s', str(v['graph_seconds'] if cache_only else case['cg_seconds']),
        '--checkpoint-every', '25', '--g-kwh', '240', '--charge-kw', '240', '--min-soc-frac', '0']
    if cache_only:
        args += ['--event-network-cache-only']
    else:
        args += ['--phase-telemetry', str(out)+'.phase-telemetry.jsonl',
            '--gurobi-log', str(out)+'.gurobi.log', '--out', str(out)]
    return args


def usable(value):
    final = value.get('final') or {}
    return final.get('artificials') == 0 and final.get('iter', 0) > 0


def publish_link(target, source):
    target = Path(target); source = Path(source).resolve()
    assert source.is_file()
    temporary = target.with_name(target.name+'.link.'+str(os.getpid()))
    temporary.symlink_to(source); temporary.replace(target)


def worker(mode, cid):
    v = read(B/'manifest.json')
    if cid == 'array':
        cid = sorted(v['cases'])[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    c = dict(v['cases'][cid]); root = B/'cases'/cid
    attempt = os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT', '0')
    a = root/mode/attempt; a.mkdir(parents=True, exist_ok=False)
    assert git('rev-parse', 'HEAD') == v['execution_commit']
    assert not git('status', '--porcelain', '--untracked-files=no')
    for name, digest in v['tooling_sha256'].items():
        assert sha(B/name) == digest
    assert sha(B/'code/data'/c['csv']) == c['input_sha256']
    for name, digest in v['data_sha256'].items():
        assert sha(B/'code/data'/name) == digest
    record = {'case_id': cid, 'mode': mode, 'attempt': attempt, 'started_epoch': time.time(),
        'execution_commit': v['execution_commit'], 'manifest_sha256': sha(B/'manifest.json'),
        'input_sha256': c['input_sha256'], 'resource_request': v['resources'][mode]}
    if mode == 'cache':
        if (root/'cache_result.json').exists():
            done = read(root/'cache_result.json'); assert sha(c['cache']) == done['cache_sha256']
            save(a/'execution.json', {**record, 'status': 'already_complete'}); return
        c['cache'] = str(a/'network.pkl')
        args = [PY, str(B/'graph_entry.py'), str(B/'code'), str(a/'progress.jsonl'), *cg_argv(v, c, None, True)]
        limit = v['graph_seconds']
    elif mode == 'cg':
        # Fresh full-instance CG: no previous-k or GIRO columns imported.
        cache = read(root/'cache_result.json'); assert sha(c['cache']) == cache['cache_sha256']
        out = a/'cg.json'; args = [PY, str(B/'code/src/exact_pricer_expanded.py'), *cg_argv(v, c, out)]
        previous = sorted((p for p in (root/'cg').glob('*/cg.json') if p.parent != a), key=lambda p: p.stat().st_mtime)
        for previous_status in reversed(previous):
            previous_value = read(previous_status)
            journal = Path(str(previous_status)+'.columns.jsonl')
            if not journal.is_file():
                continue
            assert previous_value['provenance']['git_commit'] == v['execution_commit']
            assert previous_value['provenance']['instance_sha256'] == c['input_sha256']
            record['resume_from'] = {'status': str(previous_status), 'status_sha256': sha(previous_status), 'journal_sha256': sha(journal)}
            for suffix in ['', '.columns.jsonl', '.iters.csv']:
                old = Path(str(previous_status)+suffix)
                if old.exists(): shutil.copy2(old, Path(str(out)+suffix))
            args += ['--resume']; break
        limit = 176400
    elif mode == 'mip':
        status = root/'cg.json'; value = read(status); assert usable(value)
        os.environ.update(EVSP_EXPECTED_COMMIT=v['mip_execution_commit'], EVSP_REQUIRE_DETACHED='1',
            EVSP_MIP_EXPECTED_RESULT_SHA256=sha(status), EVSP_MIP_EXPECTED_JOURNAL_SHA256=sha(value['columns_journal']))
        out = a/'result.json'
        record.update(source_status=str(status), source_status_sha256=sha(status), source_journal_sha256=sha(value['columns_journal']))
        args = [PY, str(MIP/'src/run_exact_pool_mip.py'), '--result', str(status), '--data-dir', str(B/'code/data'),
            '--reference-data-dir', str(B/'code/data'), '--cover', '--two-stage', '--timelimit', '12600',
            '--stage1-timelimit', '10800', '--threads', '8', '--mipgap', '0.0001',
            '--gurobi-log', str(a/'gurobi.log'), '--out', str(out)]
        limit = 15300
    else:
        raise ValueError(mode)
    record.update(argv=args, status='running'); save(a/'execution.json', record)
    received = []; proc = None
    def forward(signum, _frame):
        received.append(signum)
        if proc is not None and proc.poll() is None:
            try: os.killpg(proc.pid, signum)
            except ProcessLookupError: pass
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1]: signal.signal(sig, forward)
    print('RUN '+json.dumps(args), flush=True)
    proc = subprocess.Popen(args, cwd=B/'code', start_new_session=True)
    timed_out = False
    try:
        rc = proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        timed_out = True; os.killpg(proc.pid, signal.SIGTERM)
        try: rc = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); rc = proc.wait()
    record.update(returncode=rc, ended_epoch=time.time(), signals=received,
                  watchdog_time_limit=timed_out, status='interrupted' if received else 'time_limit' if timed_out else 'finished' if rc == 0 else 'failed')
    save(a/'execution.json', record)
    if received or timed_out or rc:
        raise SystemExit(128+received[-1] if received else 124 if timed_out else rc)
    if mode == 'cache':
        cache = Path(c['cache']); meta = read(str(cache)+'.manifest.json')
        assert meta['identity']['instance_sha256'] == c['input_sha256']
        assert meta['identity']['git_commit'] == v['execution_commit']
        assert sha(cache) == meta['pickle_sha256']
        publish_link(root/'network.pkl', cache); publish_link(root/'network.pkl.manifest.json', str(cache)+'.manifest.json')
        save(root/'cache_result.json', {'case_id': cid, 'attempt': attempt, 'cache_sha256': sha(cache),
            'cache_manifest_sha256': sha(str(cache)+'.manifest.json'), 'runtime_s': record['ended_epoch']-record['started_epoch'],
            'source_commit': v['execution_commit'], 'input_sha256': c['input_sha256']})
    elif mode == 'cg':
        value = read(out); assert usable(value), 'CG stopped without a usable terminal LP'
        publish_link(root/'cg.json', out)
        save(root/'cg_provenance.json', {'result_path': str(out), 'result_sha256': sha(out),
            'journal_sha256': sha(value['columns_journal']), 'certified': value.get('certified_rc_optimal'),
            'stop_reason': value.get('stop_reason'), 'attempt': attempt})
    else:
        value = read(out); assert value.get('physical_replay_validated') is True
        save(root/'mip_result.json', value)
        save(root/'mip_provenance.json', {'result_path': str(out), 'result_sha256': sha(out),
            'source_status_sha256': record['source_status_sha256'], 'execution_commit': v['mip_execution_commit']})


def submit():
    assert (B/'validation.json').exists(), 'Complete validation before submission'
    assert not (B/'jobs.json').exists(), 'Do not duplicate this campaign'
    v = read(B/'manifest.json'); jobs = []; mapping = {}; ids = sorted(v['cases'])
    def launch(mode, cid, dependencies=(), array=None):
        r = v['resources'][mode]
        args = [SLURM+'sbatch', '--parsable', '--partition=default_partition', '--exclude=scaglione-compute-01',
            '--cpus-per-task='+str(r['cpus']), '--mem='+r['mem'], '--time='+r['wall'], '--requeue',
            '--kill-on-invalid-dep=yes', '--job-name=drX_'+('graphs' if array else cid+'_'+mode),
            '--output='+str(B/'logs/%x_%A_%a_%j.out'), '--error='+str(B/'logs/%x_%A_%a_%j.err')]
        if dependencies: args += ['--dependency=afterok:'+':'.join(dependencies)]
        if array: args += ['--array='+array]
        args += [str(B/'worker.sub'), mode, cid]
        response = subprocess.run(args, capture_output=True, text=True, check=True)
        job = response.stdout.strip().split(';')[0]; assert job.isdigit()
        row = {'job_id': job, 'mode': mode, 'case_id': cid, 'dependencies': list(dependencies),
            'argv': args, 'submitted_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        jobs.append(row); save(B/'jobs.json', jobs)
        state = subprocess.check_output([SLURM+'scontrol', 'show', 'job', job, '-o'], text=True)
        assert 'ExcNodeList=scaglione-compute-01' in state and 'Partition=default_partition' in state
        if array: assert 'ArrayTaskThrottle=1' in state
        row['scontrol'] = state; save(B/'jobs.json', jobs)
        return job
    graph = launch('cache', 'array', array=f'0-{len(ids)-1}%1')
    index = {cid: str(i) for i, cid in enumerate(ids)}
    # Each k depends only on its graph and the previous CG, never the previous MIP.
    for chain, chain_ids in v['warm_chains'].items():
        previous = None
        for cid in chain_ids:
            deps = [graph+'_'+index[cid]]+([previous] if previous else [])
            cg = launch('cg', cid, deps); mip = launch('mip', cid, [cg])
            mapping[cid] = {'cache': graph+'_'+index[cid], 'cg': cg, 'mip': mip}
            previous = cg; save(B/'case_jobs.json', mapping)
    registry = Path('/home/nc437/ladder-lite/mip_preemption_study_20260911/registry.json')
    with registry.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); r = read(registry)
        existing = {x['job_id'] for x in r['cases']}
        for cid, row in mapping.items():
            if row['mip'] not in existing:
                r['cases'].append({'job_id': row['mip'], 'case_id': cid, 'cohort': 'review_full40_20260916',
                    'solver_budget_s': 12600, 'result_path': str(B/'cases'/cid/'mip_result.json'),
                    'attempt_tag': 'review_full40_20260916', 'requeue': True})
        save(registry, r)
    print(json.dumps({'graph_array': graph, 'independent_graph_tasks': len(ids), 'graph_concurrency': 1,
        'cg_jobs': len(mapping), 'mip_jobs': len(mapping), 'sequential_chains': len(v['warm_chains'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'submit', 'worker'])
    parser.add_argument('mode', nargs='?'); parser.add_argument('case', nargs='?'); args = parser.parse_args()
    if args.command == 'prepare': raise RuntimeError('Use audited prepare_remote.py; template preparation disabled')
    elif args.command == 'submit': submit()
    else: worker(args.mode, args.case)

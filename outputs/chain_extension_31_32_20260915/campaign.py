"""Frozen k31–32 continuation: independent graphs, sequential CG, per-case MIPs."""
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

B = Path('/home/nc437/ladder-lite/chain_extension_31_32_20260915')
D = Path('/share/scaglione/nc437/evsp-dr/chain_extension_31_32_20260915')
PARENT = Path('/home/nc437/ladder-lite/chain_extension_20260915')
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


def authenticate_parent(status, input_sha):
    status = Path(status); value = read(status); gate = read(status.parent/'cg_provenance.json')
    assert usable(value) and value['provenance']['git_commit'] == COMMIT
    assert value['provenance']['instance_sha256'] == input_sha
    assert status.resolve() == Path(gate['result_path']).resolve()
    assert sha(status) == gate['result_sha256']
    assert sha(value['columns_journal']) == gate['journal_sha256']
    return value


def source_provenance(root, commit):
    """Record the execution checkout and its complete borrowed object-store chain."""
    root = Path(root)
    assert git('rev-parse', 'HEAD', root=root) == commit
    assert not git('status', '--porcelain', '--untracked-files=no', root=root)
    assert subprocess.run(['git', '-C', str(root), 'symbolic-ref', '-q', 'HEAD'],
                          capture_output=True).returncode == 1
    stores = []; seen = set()
    def visit(path):
        path = Path(path).resolve()
        if str(path) in seen: return
        seen.add(str(path)); assert path.is_dir()
        alternate = path/'info/alternates'
        row = {'object_store': str(path), 'alternates': []}
        if alternate.exists():
            row['alternates_sha256'] = sha(alternate)
            for line in alternate.read_text().splitlines():
                target = (path/line).resolve()
                row['alternates'].append(str(target))
        stores.append(row)
        for target in row['alternates']: visit(target)
    visit(root/'.git/objects')
    return {'root': str(root), 'commit': commit,
            'tree': git('rev-parse', 'HEAD^{tree}', root=root),
            'object_stores': stores,
            'source_sha256': {str(p.relative_to(root)): sha(p) for p in (root/'src').glob('*.py')}}


def initial_parent_gate(v, require_job=False):
    """Authenticate frozen parent identity independently before any production launch."""
    assert sha(PARENT/'manifest.json') == v['parent_manifest_sha256']
    assert sha(PARENT/'case_jobs.json') == v['parent_case_jobs_sha256']
    old = read(PARENT/'manifest.json'); old_jobs = read(PARENT/'case_jobs.json')
    rows = {}
    for chain, p in v['initial_parents'].items():
        cid = p['parent_case_id']; case = old['cases'][cid]
        child = v['cases'][v['warm_chains'][chain][0]]
        assert child['previous_case'] == cid
        assert child['previous_input_sha256'] == p['csv_sha256'] == case['input_sha256']
        assert sha(PARENT/'code/data'/case['csv']) == p['csv_sha256']
        assert sha(B/'code/data'/p['csv']) == p['csv_sha256']
        assert p['parent_job_id'] == old_jobs[cid]['cg']
        row = {'case_id': cid, 'parent_job_id': p['parent_job_id'], 'input_sha256': p['csv_sha256']}
        if Path(p['status']).exists():
            value = authenticate_parent(p['status'], p['csv_sha256'])
            assert value['csv'] == p['csv']
            row.update(status='published_authenticated', result_sha256=sha(p['status']),
                       journal_sha256=sha(value['columns_journal']))
            if p['published_at_prepare']:
                assert row['result_sha256'] == p['status_sha256']
                assert row['journal_sha256'] == p['journal_sha256']
        else:
            assert not p['published_at_prepare'], 'Frozen published parent disappeared'
            row['status'] = 'pending_published_artifact'
            if require_job:
                state = subprocess.check_output([SLURM+'scontrol', 'show', 'job', p['parent_job_id'], '-o'], text=True, timeout=45)
                assert 'JobState=PENDING' in state or 'JobState=RUNNING' in state
                assert 'Command='+str(PARENT/'worker.sub')+' ' in state
                assert 'JobName=drX_'+cid+'_cg ' in state
                assert 'ExcNodeList=scaglione-compute-01' in state
                row['scontrol'] = state
        rows[chain] = row
    return rows


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
    old = read(PARENT/'manifest.json'); old_jobs = read(PARENT/'case_jobs.json')
    parents = {}; cases = {}; chains = {}
    assert old['execution_commit'] == COMMIT
    for chain in range(1, 7):
        key = str(chain); pid = f'w{chain}_k30'; pc = old['cases'][pid]
        previous = PARENT/'code/data'/pc['csv']; target = data/pc['csv']
        assert sha(previous) == pc['input_sha256'] == source['cases'][pid]['input_sha256']
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(previous, target)
        status = PARENT/'cases'/pid/'cg.json'
        parents[key] = {'status': str(status), 'csv': pc['csv'], 'csv_sha256': sha(target),
            'parent_manifest_sha256': sha(PARENT/'manifest.json'), 'parent_job_id': old_jobs[pid]['cg'],
            'parent_case_id': pid, 'published_at_prepare': status.exists()}
        if status.exists():
            authenticate_parent(status, pc['input_sha256'])
            parents[key].update(status_sha256=sha(status), journal_sha256=sha(read(status)['columns_journal']))
        chains[key] = [f'w{chain}_k{k}' for k in range(31,33)]
        for pos,cid in enumerate(chains[key]):
            spec = source['cases'][cid]
            rel = 'scale_ladder/instances/chain_extension_31_32_20260915/'+spec['csv']
            target = data/rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(B/'inputs'/spec['csv'], target); assert sha(target) == spec['input_sha256']
            cases[cid] = {**spec, 'stage_enabled': True, 'id': cid, 'csv': rel,
                'parent_status': str(status) if pos == 0 else str(B/'cases'/chains[key][pos-1]/'cg.json'),
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
        'initial_parents': parents, 'cases': cases, 'warm_chains': chains,
        'scientific_settings': settings, 'data_sha256': dependencies,
        'source_sha256': {str(p.relative_to(B/'code')): sha(p) for p in (B/'code/src').glob('*.py')},
        'parent_manifest_sha256': sha(PARENT/'manifest.json'), 'parent_case_jobs_sha256': sha(PARENT/'case_jobs.json'),
        'tooling_sha256': {p: sha(B/p) for p in ['campaign.py', 'graph_entry.py', 'worker.sub', 'duplicate_gate.py']},
        'source_provenance': {'cg': source_provenance(B/'code', COMMIT),
                              'mip': source_provenance(MIP, MIP_COMMIT)},
        'graph_seconds': 86400, 'resources': {
            'cache': {'cpus': 2, 'mem': '64G', 'wall': '25:00:00'},
            'cg': {'cpus': 8, 'mem': '96G', 'wall': '05:00:00'},
            'mip': {'cpus': 8, 'mem': '24G', 'wall': '02:00:00'}},
        'cache_array_concurrency': 12, 'storage_root': str(D),
        'resource_basis': 'Campaign14 C1 k28 graph187967_2 failed124 after12:01:11 at its12h native watchdog, within its12:30 Slurm allocation (sacct); recovery job189917 uses24h native watchdog and24:30 allocation. Larger k31–32 graphs therefore use a 24h native watchdog and 25h Slurm allocation, with graph preparation timed separately. Recovery189917 completed11:55:46 with37861336KiB MaxRSS;64GiB retains headroom. CG96GiB and MIP24GiB retain existing requests. Successful CG187984 and188002 completed with batch MaxRSS142762740KiB and179778832KiB despite96GiB requests; accounting may include fork/shared-memory effects. Running187972 MaxRSS190137176KiB is not evidence of physical OOM; resource interpretation remains unresolved. All12 independent graphs are eligible; six CG chains retain true previous-k data dependencies.',
        'graph_change': 'Only source diff from baseline e091a4d is validated deferred JSON tie-key evaluation; original graph winner/order rule unchanged. Coarse synchronous progress wrapper is separate tooling.',
        'requeue_policy': 'Automatic scheduler preemption requeue uses unique stage/job/restart directories. CG resumes a copied identity-validated checkpoint; graph starts fresh if unpublished; MIP starts a fresh tree. No blind retry of algorithmic/time-limit failure.',
        'borrowed_object_store': str(SOURCE/'.git/objects')}
    v['parent_gate_at_prepare'] = initial_parent_gate(v, require_job=True)
    save(B/'manifest.json', v)
    print(json.dumps({'prepared_cases': len(cases), 'manifest_sha256': sha(B/'manifest.json')}))


def cg_argv(v, case, out, cache_only=False):
    args = ['--csv', case['csv'], '--prices_csv', 'hourly_prices_flat.csv', '--time-model', 'event',
        '--event-arc-mode', 'lazy', '--event-network-cache', case['cache'],
        '--event-network-cache-mode', 'build-or-load' if cache_only else 'require',
        '--fixed-sequence-index', '--soc-step', '2.5', '--block-min', '5', '--max-iters', '50000',
        '--columns_per_iter', '30', '--column-selection', 'reduced_cost', '--column-diversity-weight', '0.0',
        '--column-candidate-multiplier', '4', '--rc-eps', '0.0001', '--master-sense', 'cover',
        '--master-backend', 'gurobi', '--initial-pool', 'singletons',
        '--wall-limit-s', str(v['graph_seconds'] if cache_only else case['cg_seconds']),
        '--checkpoint-every', '25', '--g-kwh', '240', '--charge-kw', '240', '--min-soc-frac', '0']
    if cache_only:
        args += ['--event-network-cache-only']
    else:
        args += ['--inherit-event-pool-from', case['parent_status'], '--inherit-event-pool-workers', '8',
            '--inherit-max-columns', '0', '--inherit-time-limit-s', '0',
            '--phase-telemetry', str(out)+'.phase-telemetry.jsonl',
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
    for name,digest in v['source_sha256'].items(): assert sha(B/'code'/name) == digest
    assert sha(B/'code/data'/c['csv']) == c['input_sha256']
    for name, digest in v['data_sha256'].items():
        assert sha(B/'code/data'/name) == digest
    record = {'case_id': cid, 'mode': mode, 'attempt': attempt, 'started_epoch': time.time(),
        'execution_commit': v['execution_commit'], 'manifest_sha256': sha(B/'manifest.json'),
        'input_sha256': c['input_sha256'], 'resource_request': v['resources'][mode]}
    if mode == 'cg' and (root/'cg_provenance.json').exists():
        authenticate_parent(root/'cg.json', c['input_sha256'])
        save(a/'execution.json', {**record, 'status': 'already_complete'}); return
    if mode == 'mip' and (root/'mip_provenance.json').exists():
        done = read(root/'mip_provenance.json')
        assert sha(done['result_path']) == done['result_sha256']
        assert read(done['result_path']).get('physical_replay_validated') is True
        save(a/'execution.json', {**record, 'status': 'already_complete'}); return
    if mode == 'cache':
        if (root/'cache_result.json').exists():
            done = read(root/'cache_result.json'); assert sha(c['cache']) == done['cache_sha256']
            save(a/'execution.json', {**record, 'status': 'already_complete'}); return
        c['cache'] = str(a/'network.pkl')
        args = [PY, str(B/'graph_entry.py'), str(B/'code'), str(a/'progress.jsonl'), *cg_argv(v, c, None, True)]
        limit = v['graph_seconds']
    elif mode == 'cg':
        parent = Path(c['parent_status']); value = read(parent); assert usable(value)
        record.update(parent_status=str(parent), parent_status_sha256=sha(parent),
                      parent_journal_sha256=sha(value['columns_journal']))
        initial = v['initial_parents'][str(c['chain'])]
        if str(parent) == initial['status']:
            assert c['previous_case'] == initial['parent_case_id']
            assert c['previous_input_sha256'] == initial['csv_sha256']
            assert sha(PARENT/'manifest.json') == initial['parent_manifest_sha256']
            authenticate_parent(parent, initial['csv_sha256'])
            if initial['published_at_prepare']:
                assert record['parent_status_sha256'] == initial['status_sha256']
                assert record['parent_journal_sha256'] == initial['journal_sha256']
        else:
            authenticate_parent(parent, c['previous_input_sha256'])
        assert sha(B/'code/data'/value['csv']) == value['provenance']['instance_sha256']
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
        limit = 16200
    elif mode == 'mip':
        status = root/'cg.json'; value = authenticate_parent(status, c['input_sha256'])
        assert source_provenance(MIP, v['mip_execution_commit']) == v['source_provenance']['mip']
        os.environ.update(EVSP_EXPECTED_COMMIT=v['mip_execution_commit'], EVSP_REQUIRE_DETACHED='1',
            EVSP_MIP_EXPECTED_RESULT_SHA256=sha(status), EVSP_MIP_EXPECTED_JOURNAL_SHA256=sha(value['columns_journal']))
        out = a/'result.json'
        record.update(source_status=str(status), source_status_sha256=sha(status), source_journal_sha256=sha(value['columns_journal']))
        args = [PY, str(MIP/'src/run_exact_pool_mip.py'), '--result', str(status), '--data-dir', str(B/'code/data'),
            '--reference-data-dir', str(B/'code/data'), '--cover', '--two-stage', '--timelimit', '3600',
            '--stage1-timelimit', '1800', '--threads', '8', '--mipgap', '0.0001',
            '--gurobi-log', str(a/'gurobi.log'), '--out', str(out)]
        limit = 6300
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


def submission_args(v, mode, cid, dependencies=(), array=None):
    r = v['resources'][mode]
    args = [SLURM+'sbatch', '--parsable', '--partition=default_partition', '--exclude=scaglione-compute-01',
        '--cpus-per-task='+str(r['cpus']), '--mem='+r['mem'], '--time='+r['wall'], '--requeue',
        '--kill-on-invalid-dep=yes', '--job-name=drX_'+('graphs' if array else cid+'_'+mode),
        '--output='+str(B/'logs/%x_%A_%a_%j.out'), '--error='+str(B/'logs/%x_%A_%a_%j.err')]
    if dependencies: args += ['--dependency=afterok:'+':'.join(dependencies)]
    if array: args += ['--array='+array]
    return args+[str(B/'worker.sub'), mode, cid]


def prelaunch():
    assert (B/'validation.json').exists(), 'Complete validation before submission'
    assert not (B/'jobs.json').exists(), 'Do not duplicate this campaign'
    v = read(B/'manifest.json'); validation = read(B/'validation.json')
    assert validation['status'] == 'passed'
    assert validation['tooling_sha256'] == v['tooling_sha256']
    assert validation['manifest_sha256'] == sha(B/'manifest.json')
    assert validation['validator_sha256'] == sha(B/'validate.py')
    assert validation['input_validation_sha256'] == sha(B/'input_validation.json')
    for name, digest in v['tooling_sha256'].items(): assert sha(B/name) == digest
    for kind, root in [('cg', B/'code'), ('mip', MIP)]:
        assert source_provenance(root, v['source_provenance'][kind]['commit']) == v['source_provenance'][kind]
    subprocess.run([PY, str(B/'duplicate_gate.py')], check=True)
    return initial_parent_gate(v, require_job=True)


def plan():
    gate = prelaunch(); v = read(B/'manifest.json'); ids = sorted(v['cases']); rows = []
    array = f'0-{len(ids)-1}%{v["cache_array_concurrency"]}'
    rows.append({'mode': 'cache', 'case_id': 'array', 'job_symbol': 'GRAPH_ARRAY',
                 'argv': submission_args(v, 'cache', 'array', array=array)})
    for chain, chain_ids in v['warm_chains'].items():
        p = v['initial_parents'][chain]
        previous = None if gate[chain]['status'] == 'published_authenticated' else p['parent_job_id']
        for cid in chain_ids:
            cg = 'CG_'+cid; deps = ['GRAPH_ARRAY_'+str(ids.index(cid))]+([previous] if previous else [])
            rows += [{'mode': 'cg', 'case_id': cid, 'job_symbol': cg, 'dependencies': deps,
                      'argv': submission_args(v, 'cg', cid, deps)},
                     {'mode': 'mip', 'case_id': cid, 'job_symbol': 'MIP_'+cid, 'dependencies': [cg],
                      'argv': submission_args(v, 'mip', cid, [cg])}]
            previous = cg
    save(B/'submission_plan.json', {'status': 'review_only_not_submitted', 'manifest_sha256': sha(B/'manifest.json'),
        'parent_gate': gate, 'jobs': rows, 'individual_tasks': 36, 'graph_concurrency': 12,
        'production_command': [PY, str(B/'campaign.py'), 'submit']})
    print(json.dumps({'plan': str(B/'submission_plan.json'), 'submission_calls': len(rows), 'individual_tasks': 36}))


def submit():
    prelaunch()
    v = read(B/'manifest.json'); jobs = []; mapping = {}; ids = sorted(v['cases'])
    def launch(mode, cid, dependencies=(), array=None):
        args = submission_args(v, mode, cid, dependencies, array)
        response = subprocess.run(args, capture_output=True, text=True, check=True)
        job = response.stdout.strip().split(';')[0]; assert job.isdigit()
        row = {'job_id': job, 'mode': mode, 'case_id': cid, 'dependencies': list(dependencies),
            'argv': args, 'submitted_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        jobs.append(row); save(B/'jobs.json', jobs)
        state = subprocess.check_output([SLURM+'scontrol', 'show', 'job', job, '-o'], text=True)
        assert 'ExcNodeList=scaglione-compute-01' in state and 'Partition=default_partition' in state
        if array: assert 'ArrayTaskThrottle=12' in state
        row['scontrol'] = state; save(B/'jobs.json', jobs)
        return job
    graph = launch('cache', 'array', array=f'0-{len(ids)-1}%12')
    index = {cid: str(i) for i, cid in enumerate(ids)}
    # Each k depends only on its graph and the previous CG, never the previous MIP.
    for chain, chain_ids in v['warm_chains'].items():
        p = v['initial_parents'][chain]
        assert sha(PARENT/'manifest.json') == p['parent_manifest_sha256']
        if Path(p['status']).exists():
            authenticate_parent(p['status'], p['csv_sha256']); previous = None
        else:
            previous = p['parent_job_id']
        for cid in chain_ids:
            deps = [graph+'_'+index[cid]]+([previous] if previous else [])
            cg = launch('cg', cid, deps); mip = launch('mip', cid, [cg])
            mapping[cid] = {'cache': graph+'_'+index[cid], 'cg': cg, 'mip': mip}
            previous = cg; save(B/'case_jobs.json', mapping)
    print(json.dumps({'graph_array': graph, 'independent_graph_tasks': len(ids), 'graph_concurrency': 12,
        'cg_jobs': len(mapping), 'mip_jobs': len(mapping), 'sequential_chains': len(v['warm_chains'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'plan', 'submit', 'worker'])
    parser.add_argument('mode', nargs='?'); parser.add_argument('case', nargs='?'); args = parser.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'plan': plan()
    elif args.command == 'submit': submit()
    else: worker(args.mode, args.case)

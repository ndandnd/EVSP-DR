"""Random trip-group control: stage2–15, independent graphs, sequential full-pool CG."""
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

B = Path('/home/nc437/ladder-lite/random_trip_groups_c1_20260916')
D = Path('/share/scaglione/nc437/evsp-dr/random_trip_groups_c1_20260916')
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
    assert not (B/'manifest.json').exists(), 'Already prepared; preserve the frozen campaign'
    source = read(B/'inputs/manifest.json')
    D.mkdir(parents=True, exist_ok=True); (D/'cases').mkdir(exist_ok=True)
    (B/'logs').mkdir(exist_ok=True); (B/'cases').symlink_to(D/'cases', target_is_directory=True)
    subprocess.run(['git','clone','--shared','--no-checkout',str(SOURCE),str(B/'code')],check=True,capture_output=True)
    subprocess.run(['git','-C',str(B/'code'),'checkout','--detach',COMMIT],check=True,capture_output=True)
    assert git('rev-parse','HEAD')==COMMIT
    data=B/'code/data'; dependencies={}
    for name in ['hourly_prices_flat.csv','Ref_dict.csv','par_ref_dhd.csv']:
        shutil.copy2(PARENT/'code/data'/name,data/name);dependencies[name]=sha(data/name)
    cases={}
    for cid,spec in source['cases'].items():
        rel='scale_ladder/instances/random_trip_groups_c1_20260916/'+spec['csv'];target=data/rel
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(B/'inputs'/spec['csv'],target)
        assert sha(target)==spec['input_sha256']
        cases[cid]={**spec,'csv':rel,'parent_status':str(B/'cases'/spec['previous_case']/'cg.json') if spec['previous_case'] else None,'cg_seconds':14400,'cache':str(B/'cases'/cid/'network.pkl')}
    settings={'battery_kwh':240,'initial_soc_kwh':240,'charge_kw':240,'depot_charge_kw':240,'reserve_kwh':0,'terminal_floor':None,'shared_station_capacity':False,'master_sense':'cover','objective':'100000 + electricity + 5 per charge start','charge_start_cost':5,'tariff':'flat','soc_step_kwh':2.5,'block_min':5,'columns_per_iter':30,'rc_epsilon':0.0001,'inherit_max_columns':0,'inherit_time_limit_s':0,'inherit_workers':8,'fixed_sequence_index':True,'initial_pool':'singletons; later stages also inherit all previous-stage columns','giro_columns_added':False,'mip_seconds':3600,'stage1_seconds':1800,'stage2_fleet_constraint':'<= validated incumbent','mip_seed':'solver default 0'}
    v={'schema':'evsp-random-trip-group-launch-v1','prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_commit':COMMIT,'mip_execution_commit':MIP_COMMIT,'input_manifest_sha256':sha(B/'inputs/manifest.json'),'input_generation':source,'initial_parents':{},'cases':cases,'warm_chains':source['warm_chains'],'scientific_settings':settings,'data_sha256':dependencies,'tooling_sha256':{p:sha(B/p) for p in ['campaign.py','graph_entry.py','worker.sub']},'graph_seconds':43200,'resources':{'cache':{'cpus':2,'mem':'64G','wall':'12:30:00'},'cg':{'cpus':8,'mem':'96G','wall':'05:00:00'},'mip':{'cpus':8,'mem':'24G','wall':'02:00:00'}},'cache_array_concurrency':14,'storage_root':str(D),'resource_basis':'All14 independent graph cases may run simultaneously. Requests retain previously successful baseline campaign envelopes. Later CGs depend on actual previous-stage pools, never previous MIPs.','requeue_policy':'CG resumes identity-validated checkpoint; graph rebuilds unpublished cache; MIP restarts a new tree. Each attempt is retained.','borrowed_object_store':str(SOURCE/'.git/objects'),'stage_index_is_fleet_target':False,'comparison':'Same final364 trips as original C1k15; randomized intermediate trip groups match original cumulative trip counts. Report cumulative CPU/wall/graph/init/CG/MIP time separately; no speedup claim from mixed code/runtime controls.'}
    save(B/'manifest.json',v)
    save(B/'validation.json',{'input_hashes_verified':True,'case_count':len(cases),'parent_edges':13,'independent_graphs':14,'stage2_parent':None,'source_final_input_hash':source['final_instance_sha256']})
    print(json.dumps({'prepared_cases':len(cases),'manifest_sha256':sha(B/'manifest.json')}))


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
        if case['parent_status']:
            args += ['--inherit-event-pool-from', case['parent_status'], '--inherit-event-pool-workers', '8', '--inherit-max-columns', '0', '--inherit-time-limit-s', '0']
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
        if c['parent_status']:
            parent=Path(c['parent_status']);value=read(parent);assert usable(value)
            record.update(parent_status=str(parent),parent_status_sha256=sha(parent),parent_journal_sha256=sha(value['columns_journal']))
            assert sha(B/'code/data'/value['csv'])==value['provenance']['instance_sha256']
        else:
            record['initialization']='fresh singletons, no GIRO columns'
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
        status = root/'cg.json'; value = read(status); assert usable(value)
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


def submit():
    assert (B/'validation.json').exists(), 'Complete validation before submission'
    assert not (B/'jobs.json').exists(), 'Do not duplicate this campaign'
    v = read(B/'manifest.json'); jobs = []; mapping = {}; ids = sorted(v['cases'])
    policy=Path('/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md')
    print(policy.read_text());save(B/'policy_receipt.json',{'path':str(policy),'sha256':sha(policy),'read_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
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
        if array: assert 'ArrayTaskThrottle=14' in state
        row['scontrol'] = state; save(B/'jobs.json', jobs)
        return job
    graph = launch('cache', 'array', array=f'0-{len(ids)-1}%14')
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
                r['cases'].append({'job_id': row['mip'], 'case_id': cid, 'cohort': 'default_random_trip_groups_3600',
                    'solver_budget_s': 3600, 'result_path': str(B/'cases'/cid/'mip_result.json'),
                    'attempt_tag': 'random_trip_groups_c1_20260916', 'requeue': True})
        save(registry, r)
    print(json.dumps({'graph_array': graph, 'independent_graph_tasks': len(ids), 'graph_concurrency': 14,
        'cg_jobs': len(mapping), 'mip_jobs': len(mapping), 'sequential_chains': len(v['warm_chains'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'submit', 'worker'])
    parser.add_argument('mode', nargs='?'); parser.add_argument('case', nargs='?'); args = parser.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'submit': submit()
    else: worker(args.mode, args.case)

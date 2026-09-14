"""Prepare and submit independent diagnostics without changing production chains."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import subprocess

B = Path('/home/nc437/ladder-lite/overnight_diagnostics_20260914')
D = Path('/share/scaglione/nc437/evsp-dr/overnight_diagnostics_20260914')
C = Path('/home/nc437/ladder-lite/cumulative_budget_20260913')
E = Path('/home/nc437/ladder-lite/chain_extension_20260913')
S = '/usr/local/slurm/slurm-25.05.5/bin/'
PY = '/home/nc437/evsp_env/bin/python'
MIP_COMMIT = '871d057e1067411f09581e37d78f7c1ca43f68bb'
MIP = Path('/home/nc437/ladder-lite/execution') / MIP_COMMIT


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(p, value):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def common(cid, kind, chain, k, treatment, code, commit, data, csv, digest):
    path = Path(data) / csv
    assert sha(path) == digest
    return dict(id=cid, kind=kind, chain=chain, target_k=k, target_duties=k,
        treatment=treatment, source_code=str(code), execution_commit=commit,
        data_dir=str(data), csv=csv, input_path=str(path), input_sha256=digest,
        static_hashes={str(Path(data)/n): sha(Path(data)/n)
            for n in ['Ref_dict.csv', 'par_ref_dhd.csv', 'hourly_prices_flat.csv']})


def mip_case(cid, chain, k, data, csv, digest, treatment, total, fleet):
    c = common(cid, 'mip', chain, k, treatment, MIP, MIP_COMMIT, data, csv, digest)
    c.update(solver_budget_s=total, stage1_budget_s=fleet, watchdog_s=total+2700,
        resources={'cpus': 8, 'mem': '24G', 'allocation_s': total+3600},
        argv=[PY, str(MIP/'src/run_exact_pool_mip.py'), '--result', '{source_status}',
            '--data-dir', str(data), '--reference-data-dir', str(data), '--cover',
            '--two-stage', '--timelimit', str(total), '--stage1-timelimit', str(fleet),
            '--threads', '8', '--mipgap', '0.0001', '--gurobi-log', '{attempt_dir}/gurobi.log',
            '--out', '{out}'])
    return c


def prepare():
    assert not (B/'manifest.json').exists(), 'Prepared campaign already exists'
    selection = read(B/'selection.json')
    cm = module(C/'campaign.py', 'cumulative_reference')
    em = module(E/'campaign.py', 'extension_reference')
    cm.code_check()
    ev = read(E/'manifest.json')
    cv = read(C/'manifest.json')
    assert subprocess.check_output(['git', '-C', str(E/'code'), 'rev-parse', 'HEAD'], text=True).strip() == ev['execution_commit']
    assert not subprocess.check_output(['git', '-C', str(E/'code'), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    assert subprocess.run(['git', '-C', str(E/'code'), 'symbolic-ref', '-q', 'HEAD'], capture_output=True).returncode == 1
    D.mkdir(parents=True, exist_ok=True)
    (D/'cases').mkdir(exist_ok=True)
    if (B/'cases').is_symlink():
        assert (B/'cases').resolve() == D/'cases'
    else:
        (B/'cases').symlink_to(D/'cases', target_is_directory=True)
    (B/'logs').mkdir(exist_ok=True)
    cases = {}
    for original in selection['fresh_misses']:
        src = cv['cases'][original]
        for suffix in ['c200', 'complementary']:
            cid = original+'_'+suffix
            c = common(cid, 'cg', src['chain'], src['k'], suffix,
                cm.CODE, cm.COMMIT, cm.CODE/'data', src['csv'], src['input_sha256'])
            budget = src['fresh_primary_budget_s']
            args = cm.cg_args(src, '{out}', budget)
            if suffix == 'c200':
                args[args.index('--columns_per_iter')+1] = '200'
                change = 'columns_per_iter:30 -> 200; reduced-cost selection retained'
            else:
                args[args.index('--column-selection')+1] = 'complementary'
                args[args.index('--column-diversity-weight')+1] = '0.5'
                change = 'selection:reduced_cost -> complementary (weight0.5,candidate multiplier4);30columns retained'
            c.update(argv=args, solver_budget_s=budget, watchdog_s=budget+1800,
                resources={'cpus': 8, 'mem': '96G', 'allocation_s': budget+3600},
                cache_manifest_path=src['cache']+'.manifest.json',
                cache_manifest_sha256=sha(src['cache']+'.manifest.json'),
                original_case=original, changed_factor=change,
                comparator=str(C/'cases'/original/'base/completion.json'),
                interpretation='Fresh singleton start; original cumulative CG allowance retained. Selected difficult cases, not an unbiased success-rate sample.')
            cases[cid] = c
            child = mip_case(cid+'_mip', src['chain'], src['k'], cm.CODE/'data',
                src['csv'], src['input_sha256'], suffix, 3600, 1800)
            child.update(source_case=cid, original_case=original)
            cases[child['id']] = child
    for selected in selection['long_mips']:
        original = selected['case_id']
        assert sha(selected['prior_result']) == selected['prior_sha256']
        old_mip = read(selected['prior_result'])
        assert old_mip['physical_replay_validated'] and not old_mip['fleet_proven']
        if selected['origin'] == 'fresh':
            src = cv['cases'][original]
            source = Path(read(C/'cases'/original/'base/completion.json')['result_path'])
            data = cm.CODE/'data'
        else:
            src = ev['cases'][original]
            source = E/'cases'/original/'cg.json'
            data = E/'code/data'
        status = read(source)
        journal = Path(status['columns_journal'])
        state_paths = [Path(selected['prior_result']).parent/n for n in ['state.json', 'execution.json']]
        states = [read(p) for p in state_paths if p.exists()]
        assert any(x.get('source_status_sha256') == sha(source) and
                   x.get('source_journal_sha256') == sha(journal) for x in states), original
        cid = original+'_'+selected['origin']+'_longmip'
        c = mip_case(cid, selected['chain'], selected['target_k'], data,
            src['csv'], src['input_sha256'], 'longer_fleet_search', 12600, 10800)
        c.update(source_status=str(source.resolve()), source_status_sha256=sha(source),
            source_journal_sha256=sha(journal), original_case=original,
            source_cg_commit=status['provenance']['git_commit'], comparator=selected,
            changed_factor='Fleet-stage allowance1800 -> 10800s; total3600 ->12600s preserves1800s cost allowance',
            interpretation='New Gurobi tree on exactly the same saved pool; not a resumed tree or proof of full-model optimality.')
        cases[cid] = c
    for original in selection['resume_cases']:
        src = dict(ev['cases'][original])
        source = E/'cases'/original/'cg.json'
        prior = read(source)
        assert not prior['certified_rc_optimal'] and prior['stop_reason'] == 'wall_limit'
        assert prior['final']['artificials'] == 0
        src['cg_seconds'] = 28800
        cid = original+'_resume8h'
        c = common(cid, 'cg', src['chain'], src['k'], 'same_k_resume_to8h',
            E/'code', ev['execution_commit'], E/'code/data', src['csv'], src['input_sha256'])
        remaining = math.ceil(28800-prior['wall_s'])
        c.update(argv=[PY, str(E/'code/src/exact_pricer_expanded.py'), *em.cg_argv(ev, src, '{out}')],
            solver_budget_s=28800, watchdog_s=remaining+1800,
            resources={'cpus': 8, 'mem': '96G', 'allocation_s': remaining+3600},
            resume_from=str(source.resolve()), resume_from_sha256=sha(source),
            resume_journal_sha256=sha(prior['columns_journal']),
            prior_native_wall_s=prior['wall_s'], cache_manifest_path=src['cache']+'.manifest.json',
            cache_manifest_sha256=sha(src['cache']+'.manifest.json'), original_case=original,
            changed_factor='Cumulative CG time limit14400 ->28800s on an isolated copy of the saved checkpoint',
            interpretation='Same-k continuation; previous compute is charged. Production chain and its successor remain unchanged.')
        cases[cid] = c
        child = mip_case(cid+'_mip', src['chain'], src['k'], E/'code/data',
            src['csv'], src['input_sha256'], 'same_k_resume_to8h', 3600, 1800)
        child.update(source_case=cid, original_case=original)
        cases[child['id']] = child
    assert len(cases) == 99
    assert sum(not x.get('source_case') for x in cases.values()) == 61
    v = dict(schema='evsp-overnight-diagnostics-v1', prepared_utc=now(), cases=cases,
        selection=selection, selection_sha256=sha(B/'selection.json'),
        policy_sha256=sha(B.parent/'SCAGLIONE_RESOURCE_POLICY.md'),
        storage_root=str(D), physics=cv['settings'],
        resources_note='38independent CGs +23independent MIPs. No artificial throttle; all cases may start. Default partition reported9651idleCPUslots,70idle nodes at initial audit. Slurm enforces CPU/RAM admission. EveryCPUjob excludes scaglione-compute-01.',
        tooling_sha256={n:sha(B/n) for n in ['worker.py','worker.sub']})
    save(B/'manifest.json', v)
    print(json.dumps({'cases':99, 'ready_cg':38, 'ready_mip':23, 'dependent_mip':38, 'manifest_sha256':sha(B/'manifest.json')}))


def submit():
    with (B/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        submit_locked()


def submit_locked():
    v = read(B/'manifest.json')
    validation = read(B/'validation.json')
    assert validation['status'] in ('passed', 'partially_passed') and validation['manifest_sha256'] == sha(B/'manifest.json')
    jobs = read(B/'jobs.json') if (B/'jobs.json').exists() else []
    mapping = {j['case_id']:j['job_id'] for j in jobs}
    assert len(mapping) == len(jobs), 'Duplicate recorded submissions'
    if (B/'case_jobs.json').exists():
        assert mapping == read(B/'case_jobs.json'), 'Submission records disagree'
    allowed = validation.get('validated_treatments')
    eligible = [cid for cid,c in v['cases'].items()
                if validation['status'] == 'passed' or c['treatment'] in allowed]
    ordered = sorted(eligible, key=lambda cid: (bool(v['cases'][cid].get('source_case')), cid))
    newly_submitted = 0
    for cid in ordered:
        if cid in mapping:
            continue
        c = v['cases'][cid]
        deps = [mapping[c['source_case']]] if c.get('source_case') else []
        resource = c['resources']
        minutes = math.ceil(resource['allocation_s']/60)
        args = [S+'sbatch', '--parsable', '--partition=default_partition',
            '--exclude=scaglione-compute-01', '--cpus-per-task='+str(resource['cpus']),
            '--mem='+resource['mem'], '--time='+f'{minutes//60}:{minutes%60:02d}:00',
            '--requeue', '--kill-on-invalid-dep=yes', '--job-name=drD_'+cid,
            '--output='+str(B/'logs/%x_%j.out'), '--error='+str(B/'logs/%x_%j.err')]
        if deps:
            args += ['--dependency=afterok:'+':'.join(deps)]
        args += [str(B/'worker.sub'), str(B), cid]
        job = subprocess.check_output(args, text=True).strip().split(';')[0]
        assert job.isdigit()
        mapping[cid] = job
        newly_submitted += 1
        jobs.append(dict(job_id=job, case_id=cid, kind=c['kind'], treatment=c['treatment'],
            dependencies=deps, argv=args, submitted_utc=now()))
        save(B/'jobs.json', jobs)
        save(B/'case_jobs.json', mapping)
        control = subprocess.check_output([S+'scontrol', 'show', 'job', job, '-o'], text=True)
        assert 'ExcNodeList=scaglione-compute-01' in control and 'Partition=default_partition' in control
        jobs[-1]['scontrol'] = control
        save(B/'jobs.json', jobs)
    print(json.dumps({'submitted':len(jobs), 'newly_submitted':newly_submitted,
                     'independent':sum(not j['dependencies'] for j in jobs)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare','submit'])
    args = p.parse_args()
    prepare() if args.action == 'prepare' else submit()

#!/usr/bin/env python3
"""Execute one immutable overnight diagnostic; publish only verified results."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import socket
import subprocess
import time

DEFAULT_ROOT = Path('/home/nc437/ladder-lite/overnight_diagnostics_20260914')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def require_hash(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected):
        raise ValueError(f'Missing or malformed frozen SHA-256 for {path}')
    observed = sha(path)
    if observed != expected:
        raise ValueError(f'Frozen hash mismatch for {path}: {observed} != {expected}')
    return observed


def check_code(path, expected):
    def git(*args):
        return subprocess.run(['git', '-C', str(path), *args], text=True,
                              capture_output=True, check=False)
    head = git('rev-parse', '--verify', 'HEAD')
    tracked = git('status', '--porcelain', '--untracked-files=no')
    branch = git('symbolic-ref', '-q', 'HEAD')
    if head.returncode or head.stdout.strip() != expected:
        raise ValueError(f'Execution commit mismatch in {path}')
    if tracked.returncode or tracked.stdout.strip():
        raise ValueError(f'Execution checkout has tracked modifications: {path}')
    if branch.returncode != 1:
        raise ValueError(f'Execution checkout must have detached HEAD: {path}')


def journal_path(status_path, value):
    path = Path(value['columns_journal'])
    return path if path.is_absolute() else Path(status_path).parent / path


def usable_cg(value):
    final = value.get('final') or {}
    return final.get('artificials') == 0 and final.get('iter', 0) > 0


def bound_source(path, status_hash, journal_hash, expected_commit=None):
    path = Path(path)
    require_hash(path, status_hash)
    value = read(path)
    journal = journal_path(path, value)
    require_hash(journal, journal_hash)
    if expected_commit and (value.get('provenance') or {}).get('git_commit') != expected_commit:
        raise ValueError(f'Source CG commit mismatch: {path}')
    return dict(path=str(path), status_sha256=status_hash,
                journal_path=str(journal), journal_sha256=journal_hash,
                value=value)


def resolve_source(root, case):
    if case.get('source_case'):
        cid = case['source_case']
        safe_id(cid)
        completion = read(root / 'cases' / cid / 'completion.json')
        if completion.get('kind') != 'cg' or not completion.get('usable'):
            raise ValueError(f'Dependent CG source is not usable: {cid}')
        return bound_source(completion['result_path'], completion['result_sha256'],
                            completion['journal_sha256'], case.get('source_cg_commit'))
    if case.get('source_status'):
        return bound_source(case['source_status'], case.get('source_status_sha256'),
                            case.get('source_journal_sha256'), case.get('source_cg_commit'))
    if case['kind'] == 'mip':
        raise ValueError('MIP case must identify an immutable source pool')
    return None


def preflight(root, manifest, case):
    if case['kind'] not in ('cg', 'mip'):
        raise ValueError('Unsupported worker kind')
    for name in ('worker.py', 'worker.sub'):
        require_hash(root / name, manifest['tooling_sha256'].get(name))
    check_code(case['source_code'], case['execution_commit'])
    require_hash(case['input_path'], case['input_sha256'])
    for path, digest in case['static_hashes'].items():
        require_hash(path, digest)
    if case.get('cache_manifest_path'):
        require_hash(case['cache_manifest_path'], case.get('cache_manifest_sha256'))
    if float(case['solver_budget_s']) <= 0 or float(case['watchdog_s']) <= 0:
        raise ValueError('Solver and watchdog budgets must be positive')
    source = resolve_source(root, case)
    if case['kind'] == 'mip' and not usable_cg(source['value']):
        raise ValueError('MIP source has no usable artificial-free CG endpoint')
    if case['kind'] == 'mip':
        source_input = (source['value'].get('provenance') or {}).get('instance_sha256')
        if source_input != case['input_sha256']:
            raise ValueError('MIP source and frozen input identities differ')
    return source


def expand_argv(template, out, attempt_dir, source_status=None):
    substitutions = {'out': str(out), 'attempt_dir': str(attempt_dir)}
    if source_status is not None:
        substitutions['source_status'] = str(source_status)
    if not isinstance(template, list) or not template or any(not isinstance(x, str) for x in template):
        raise ValueError('argv must be a nonempty list of strings')
    def replace(match):
        key = match.group(1)
        if key not in substitutions:
            raise ValueError(f'Unresolved argv placeholder: {{{key}}}')
        return substitutions[key]
    return [re.sub(r'\{([^{}]+)\}', replace, value) for value in template]


def copy_resume(case, out):
    source = case['resume_from']
    if case['kind'] != 'cg':
        raise ValueError('Only CG can resume')
    status_hash = (case.get('resume_from_sha256') or case.get('resume_status_sha256')
                   or case.get('source_status_sha256'))
    frozen = bound_source(source, status_hash, case.get('resume_journal_sha256'),
                          case['execution_commit'])
    shutil.copy2(source, out)
    shutil.copy2(frozen['journal_path'], str(out) + '.columns.jsonl')
    # The solver resolves its append journal from --out, not from the copied
    # status's historical columns_journal field. Preserve the original bytes.
    iters = Path(str(source) + '.iters.csv')
    if iters.exists():
        if case.get('resume_iters_sha256'):
            require_hash(iters, case['resume_iters_sha256'])
        shutil.copy2(iters, str(out) + '.iters.csv')
    return dict(source=str(source), status_sha256=status_hash,
                journal_sha256=frozen['journal_sha256'],
                native_elapsed_s=frozen['value'].get('wall_s'),
                copied_iters_sha256=sha(iters) if iters.exists() else None)


def run_process(argv, attempt, cwd, watchdog_s, env):
    record = dict(argv=argv, cwd=str(cwd), started_utc=now(), host=socket.getfqdn(),
                  watchdog_s=watchdog_s, status='running')
    save(attempt / 'execution.json', record)
    signals = []
    proc = None
    received_at = None
    timed_out = False
    forced_kill = False
    termination_started = None
    error = None
    rc = None
    def kill_group(sig):
        if proc is not None:
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
    def forward(sig, _frame):
        nonlocal received_at
        signals.append(sig)
        received_at = received_at or time.monotonic()
        kill_group(sig)
    previous = {sig: signal.signal(sig, forward)
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        with (attempt / 'stdout.log').open('xb') as stdout, (attempt / 'stderr.log').open('xb') as stderr:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout,
                                    stderr=stderr, start_new_session=True)
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed >= watchdog_s and not timed_out:
                    timed_out = True
                    termination_started = time.monotonic()
                    kill_group(signal.SIGTERM)
                deadline_start = received_at or termination_started
                if deadline_start and time.monotonic() - deadline_start >= 30:
                    forced_kill = True
                    kill_group(signal.SIGKILL)
                try:
                    proc.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    pass
            rc = proc.returncode
            if signals or timed_out or rc:
                # Also remove descendants if the group leader exited first.
                kill_group(signal.SIGKILL)
    except BaseException as exc:
        error = repr(exc)
        kill_group(signal.SIGKILL)
        if proc is not None:
            rc = proc.wait()
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        status = ('interrupted' if signals else 'watchdog' if timed_out else
                  'failed' if error or rc != 0 else 'finished')
        record.update(returncode=rc, ended_utc=now(), wall_s=time.monotonic() - start,
                      user_cpu_s=after.ru_utime - before.ru_utime,
                      system_cpu_s=after.ru_stime - before.ru_stime,
                      children_maxrss_native=after.ru_maxrss,
                      signals=signals, watchdog=timed_out, forced_kill=forced_kill,
                      status=status, error=error)
        save(attempt / 'execution.json', record)
    if status != 'finished':
        raise RuntimeError(f'Solver execution {status}, return code {rc}')
    return record


def register_mip(root, case_id, case, attempt_token, out):
    path = root.parent / 'mip_preemption_study_20260911' / 'registry.json'
    # This is the established registry; never silently create a replacement.
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = read(path)
        job = os.environ['SLURM_JOB_ID']
        key = (job, str(out))
        if key not in {(str(x['job_id']), x.get('result_path')) for x in value['cases']}:
            value['cases'].append(dict(job_id=job, case_id=case_id,
                cohort=('validation_overnight_diagnostics_20260914' if case.get('is_validation')
                        else 'review_p1_mip_20260916'),
                is_validation=bool(case.get('is_validation')),
                solver_budget_s=case['solver_budget_s'], result_path=str(out),
                attempt_tag=attempt_token, restart_count=os.environ.get('SLURM_RESTART_COUNT', '0'),
                requeue=True, registered_utc=now()))
            save(path, value)


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
        raise ValueError(f'Unsafe case or attempt identifier: {value!r}')


def run_case(root, case_id):
    root = Path(root).resolve()
    safe_id(case_id)
    manifest_path = root / 'manifest.json'
    manifest = read(manifest_path)
    case = manifest['cases'][case_id]
    token = os.environ['SLURM_JOB_ID'] + '_r' + os.environ.get('SLURM_RESTART_COUNT', '0')
    safe_id(token)
    case_root = root / 'cases' / case_id
    attempt = case_root / 'attempts' / token
    attempt.mkdir(parents=True, exist_ok=False)
    out = attempt / ('cg.json' if case['kind'] == 'cg' else 'result.json')
    state = dict(case_id=case_id, kind=case['kind'], attempt=token,
                 started_utc=now(), manifest_sha256=sha(manifest_path),
                 status='preflight', execution_commit=case['execution_commit'])
    save(attempt / 'state.json', state)
    # Serialize duplicate launches of this case; no lock crosses case boundaries.
    with (case_root / '.worker.lock').open('a') as lock:
        try:
            if case['kind'] == 'mip':
                register_mip(root, case_id, case, token, out)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            source = preflight(root, manifest, case)
            if (case_root / 'completion.json').exists():
                previous = read(case_root / 'completion.json')
                if previous['manifest_sha256'] != state['manifest_sha256']:
                    raise ValueError('Existing completion belongs to another manifest')
                require_hash(previous['result_path'], previous['result_sha256'])
                if previous.get('journal_sha256'):
                    require_hash(previous['journal_path'], previous['journal_sha256'])
                state.update(status='already_complete', optimization_run=False,
                             prior_completion=previous, ended_utc=now())
                save(attempt / 'state.json', state)
                return state
            argv = expand_argv(case['argv'], out, attempt, source['path'] if source else None)
            if '--out' not in argv or argv[argv.index('--out') + 1] != str(out):
                raise ValueError('argv must target this attempt via --out {out}')
            if case.get('resume_from'):
                state['resume'] = copy_resume(case, out)
                if '--resume' not in argv:
                    argv.append('--resume')
            elif '--resume' in argv:
                raise ValueError('Unbound resume flag is prohibited')
            env = os.environ.copy()
            if case['kind'] == 'mip':
                env.update(EVSP_EXPECTED_COMMIT=case['execution_commit'], EVSP_REQUIRE_DETACHED='1',
                    EVSP_MIP_EXPECTED_RESULT_SHA256=source['status_sha256'],
                    EVSP_MIP_EXPECTED_JOURNAL_SHA256=source['journal_sha256'])
            state.update(status='running', optimization_run=True,
                         source_status_sha256=source['status_sha256'] if source else None,
                         source_journal_sha256=source['journal_sha256'] if source else None)
            save(attempt / 'state.json', state)
            execution = run_process(argv, attempt, case['source_code'], float(case['watchdog_s']), env)
            value = read(out)
            final = dict(state, status='finished', ended_utc=now(), usable=True,
                         result_path=str(out), result_sha256=sha(out), execution=execution,
                         solver_budget_s=case['solver_budget_s'])
            if case['kind'] == 'cg':
                if not usable_cg(value):
                    raise ValueError('CG result is not a usable artificial-free endpoint')
                if (value.get('provenance') or {}).get('git_commit') != case['execution_commit']:
                    raise ValueError('CG output has wrong execution commit')
                if (value.get('provenance') or {}).get('instance_sha256') != case['input_sha256']:
                    raise ValueError('CG output has wrong instance identity')
                journal = journal_path(out, value)
                # An attempt must never publish another attempt's mutable journal.
                if journal.resolve() != Path(str(out) + '.columns.jsonl').resolve():
                    raise ValueError('CG output journal is outside its private attempt')
                final.update(journal_path=str(journal), journal_sha256=sha(journal),
                             certified=bool(value.get('certified_rc_optimal')),
                             stop_reason=value.get('stop_reason'), native_wall_s=value.get('wall_s'))
                published = case_root / 'cg.json'
            else:
                if value.get('physical_replay_validated') is not True:
                    raise ValueError('MIP result has no successful physical replay')
                comparator = case['comparator']
                audit = value['physical_pool_audit']
                if audit['mip_ordered_pool_sha256'] != comparator['ordered_pool_sha256']:
                    raise ValueError('Controlled comparison has a different ordered pool')
                start = {k: v for k, v in value['mip_start'].items() if k != 'solver_acceptance'}
                if start != comparator['mip_start']:
                    raise ValueError('Controlled comparison has a different MIP start')
                parameters = value['mip_provenance']['gurobi_parameters']
                if parameters['Seed'] != case['seed'] or parameters['Threads'] != 8:
                    raise ValueError('Controlled comparison Seed/Threads mismatch')
                detail = value.get('two_stage') or {}
                if detail['stage1_time_limit_s'] != case['stage1_budget_s']:
                    raise ValueError('Controlled comparison fleet budget mismatch')
                if detail['stage2_fleet_constraint'] != 'at_most':
                    raise ValueError('Unexpected second-stage fleet constraint')
                final['controlled_comparison_validated'] = True
                final.update(physical_replay_validated=True,
                    fleet_proven=detail.get('fleet_proven', value.get('fleet_proven')),
                    fleet_bound=value.get('fleet_bound'), buses=value.get('buses'),
                    target_matched=(value.get('buses') <= case['target_k']
                                    if value.get('buses') is not None and case.get('target_k') is not None
                                    else None),
                    target_k=case.get('target_k'))
                published = case_root / 'mip_result.json'
            link = published.with_name(published.name + '.tmp.' + str(os.getpid()))
            link.symlink_to(out.resolve())
            link.replace(published)
            save(case_root / 'completion.json', final)
            save(attempt / 'state.json', final)
            return final
        except BaseException as exc:
            state.update(status='execution_failed', error=repr(exc), ended_utc=now())
            save(attempt / 'state.json', state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    run_case(args.root, args.case)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""One-case cleanup recovery; keeps standard collector attempt layout."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

from cleanup_case11 import (CELL, CODE, COMMIT, MODULE_SHA, ROOT, SOURCE, SOURCE_SHA,
                            patched_source, sha, validate_source)

HERE = Path(__file__).resolve().parent
CONFIG_SHA = '040f5567d4e68dacc03243711a9dbe46d7405a965fde48791ca433e72d9f2bcb'
RUN_CELLS_SHA = 'f6d4d9bf9caf4e9a82652e1a05656d8a0821b7d2a700c2baa3dc618067f8ffb2'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def git(*args):
    return subprocess.check_output(['git', '-C', str(CODE), *args], text=True).strip()


def check_checkout():
    if git('rev-parse', 'HEAD') != COMMIT:
        raise ValueError('execution commit mismatch')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('frozen checkout has tracked changes')
    if subprocess.run(['git', '-C', str(CODE), 'symbolic-ref', '-q', 'HEAD'],
                      capture_output=True).returncode != 1:
        raise ValueError('frozen checkout must be detached')
    if sha(ROOT / 'config.json') != CONFIG_SHA or sha(ROOT / 'run_cells.tsv') != RUN_CELLS_SHA:
        raise ValueError('campaign config or row table changed')


def main():
    job = os.environ.get('SLURM_JOB_ID', '')
    restart = os.environ.get('SLURM_RESTART_COUNT', '0')
    if not job.isdigit() or not restart.isdigit():
        raise ValueError('a Slurm job/restart is required; local solver execution is disabled')
    if os.environ.get('SLURM_CPUS_PER_TASK') != '8':
        raise ValueError('recovery requires the original 8 CPUs')
    lock = open(HERE / 'mix1_two_price_split.lock', 'a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('another case11 recovery already holds the case lock')
    check_checkout()
    summary, receipt = validate_source(SOURCE, SOURCE_SHA)
    cleanup_dir = ROOT / 'results' / CELL / 'cleanup'
    for previous in cleanup_dir.glob('job*_r*/attempt.json'):
        if json.loads(previous.read_text()).get('state') == 'finished':
            raise ValueError('a finished cleanup already exists: ' + str(previous))
    attempt = cleanup_dir / f'job{job}_r{restart}'
    attempt.mkdir(parents=True, exist_ok=False)
    out = attempt / 'out'
    patched = patched_source((CODE / 'src/terminal_duplicate_cleanup.py').read_bytes())
    (attempt / 'patched_terminal_duplicate_cleanup.py').write_bytes(patched)
    receipt['patched_module_sha256'] = hashlib.sha256(patched).hexdigest()
    write(attempt / 'recovery_design.json', receipt)
    argv = [sys.executable, '-u', str(HERE / 'cleanup_case11.py'), '--source', str(SOURCE),
            '--source-sha256', SOURCE_SHA, '--out', str(out), '--seconds', '3600.0', '--threads', '8']
    inputs = summary['inputs']
    meta = dict(stage='cleanup', cell=CELL, cohort='mix1', idx=43, argv=argv,
                execution_commit=COMMIT, code=str(CODE), worker_sha256=sha(__file__),
                wrapper_sha256=sha(HERE / 'cleanup_case11.py'), original_module_sha256=MODULE_SHA,
                patched_module_sha256=receipt['patched_module_sha256'],
                config_sha256=CONFIG_SHA, run_cells_sha256=RUN_CELLS_SHA,
                source_attempt=str(SOURCE.parent.parent), source_is_fallback=False,
                source_sha256=SOURCE_SHA, selected_source_sha256=receipt['selected_source_sha256'],
                tariff=inputs['tariff'], tariff_sha256=summary['input_hashes'][inputs['tariff']],
                instance=inputs['instance'], instance_sha256=summary['input_hashes'][inputs['instance']],
                target_kwh=summary['terminal_target_kwh'], fleet_cap=summary['fleet_cap'],
                input_hashes=summary['input_hashes'], physics=summary['physics'],
                slurm_job_id=job, slurm_restart=restart, slurm_array_job_id=None,
                slurm_array_task_id=None, node=socket.gethostname(), cpus='8',
                recovery_of_post='498962', recovery_guard=11, recovery_design=str(attempt / 'recovery_design.json'),
                resources=dict(partition='default_partition', cpus=8, memory='48G', walltime='05:00:00',
                               exclude='scaglione-compute-01', requeue=True),
                started_utc=now(), state='running')
    write(attempt / 'attempt.json', meta)
    rc = 1
    try:
        # No solver or native-module import occurs before all source/selection guards.
        rc = subprocess.run(argv, cwd=CODE).returncode
        if rc == 0:
            for name in ('summary.json', 'selected_routes.json', 'repair_routes.json'):
                if not (out / name).is_file():
                    raise ValueError('cleanup exited without expected output ' + name)
            meta['output_hashes'] = {p.name: sha(p) for p in sorted(out.glob('*.json'))}
            final = json.loads((out / 'summary.json').read_text())
            meta['has_selection'] = bool(json.loads((out / 'selected_routes.json').read_text()))
            meta['exact_once_verified'] = final.get('exact_once_verified', False)
            meta['shared_capacity_validated'] = final.get('shared_capacity_validated', False)
    except Exception:
        rc = 1
        meta['worker_error'] = traceback.format_exc()
        print(meta['worker_error'], file=sys.stderr, flush=True)
    meta.update(state='finished' if rc == 0 else 'failed', returncode=rc, finished_utc=now())
    write(attempt / 'attempt.json', meta)
    return rc


if __name__ == '__main__':
    sys.exit(main())

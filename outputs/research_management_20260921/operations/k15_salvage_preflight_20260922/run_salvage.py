#!/usr/bin/env python3
"""Compute-worker-only, fail-closed salvage of an immutable failed dive pool.

No submissions; no production source edits. Automatic requeue is deliberately
refused: a new attempt requires explicit accounting of earlier solver runtime.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def assert_sources(cell):
    for label, item in cell['source_hashes'].items():
        if sha(item['path']) != item['actual'] or item['actual'] != item['expected']:
            raise RuntimeError('Source hash changed: ' + label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=['c1_k15', 'c5_k15'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    args = parser.parse_args()
    job = os.environ.get('SLURM_JOB_ID')
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    if not job or not job.isdigit() or restart:
        raise SystemExit('Run in a fresh allocation only; account prior attempts before retry.')
    plan = json.loads(args.plan.read_text())
    cell = plan['cases'][args.case]
    for name in cell['unset_environment']:
        os.environ.pop(name, None)
    os.environ.update(cell['environment'])
    case_root = args.out_root / args.case
    case_root.mkdir(parents=True, exist_ok=True)
    with (case_root / 'case.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if list(case_root.glob('job_*')):
            raise RuntimeError('Existing salvage attempt: debit its solver work before any retry.')
        out = case_root / f'job_{job}_r0'
        out.mkdir(exist_ok=False)
        receipt = {'status': 'preflight', 'case': args.case, 'job_id': job,
                   'failed_job_id': cell['failed_job_id'],
                   'plan_sha256': sha(args.plan), 'wrapper_sha256': sha(__file__),
                   'execution_commit': plan['execution_commit'],
                   'scope': 'supplemental finite-pool salvage; original failed trial preserved',
                   'global_certificate': None, 'incumbent_imported': False}
        save(out / 'receipt.json', receipt)
        try:
            assert_sources(cell)
            augmented = Path(cell['source_hashes']['augmented_result']['path'])
            prior = json.loads((augmented.parent.parent / 'execution.json').read_text())
            residual = math.floor(7200 - prior['dive_wall_s'])
            if prior['dive_wall_s'] != cell['failed_dive_wall_s'] or residual != cell['solver_limit_s']:
                raise RuntimeError('Failed-dive time or remaining solver budget changed')
            argv = [a.replace(cell['output_directory_template'], str(out))
                    for a in cell['native_worker_argv_template']]
            source_dir = Path(argv[1]).parent
            sys.path.insert(0, str(source_dir))
            from run_exact_pool_mip import (verified_mip_code_identity, load_pool,
                                           prepare_strict_partition_pool)
            receipt['code_identity'] = verified_mip_code_identity()
            started = time.monotonic()
            status, routes, trips = load_pool(augmented, deduplicate=True)
            prepared, audit = prepare_strict_partition_pool(
                status, routes, data_dir=source_dir.parent / 'data',
                reference_data_dir=source_dir.parent / 'data')
            receipt['gate_wall_s'] = time.monotonic() - started
            receipt['physical_gate'] = audit
            save(out / 'physical_gate.json', audit)
            if audit['rejected_columns'] != 0 or not prepared:
                raise RuntimeError('Native physical gate rejects columns or yields an empty pool')
            # Preserve the native repair counts/ordered hash; no feasibility tolerance change.
            del prepared, routes, status, trips
            gc.collect()
            assert_sources(cell)
            receipt.update(status='solver_running', solver_limit_s=residual,
                           failed_dive_wall_s=prior['dive_wall_s'], argv=argv)
            save(out / 'receipt.json', receipt)
            started = time.monotonic()
            with (out / 'mip_stdout.log').open('w') as stdout, (out / 'mip_stderr.log').open('w') as stderr:
                solved = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False)
            receipt.update(mip_subprocess_wall_s=time.monotonic() - started,
                           mip_returncode=solved.returncode)
            if solved.returncode:
                raise RuntimeError('Pinned MIP runner failed; preserve outputs and account this attempt')
            result = json.loads((out / 'result.json').read_text())
            native = result['physical_pool_audit']
            if native['rejected_columns'] != 0:
                raise RuntimeError('Solver physical preparation disagrees with zero-rejection gate')
            # The native runner adds metadata but uses the same ordered pool identity.
            if native['base_pool_ordered_sha256'] != audit['mip_ordered_pool_sha256']:
                raise RuntimeError('Physical pool identity changed between gate and final MIP')
            assert_sources(cell)
            receipt.update(status='finished', source_immutable=True,
                           actual_solver_runtime_s=result['runtime_s'],
                           charged_dive_plus_solver_s=prior['dive_wall_s'] + result['runtime_s'],
                           external_mip_overhead_s=receipt['mip_subprocess_wall_s'] - result['runtime_s'],
                           result_sha256=sha(out / 'result.json'))
        except (Exception, SystemExit) as exc:
            receipt.update(status='failed', error=repr(exc))
            save(out / 'receipt.json', receipt)
            raise
        save(out / 'receipt.json', receipt)


if __name__ == '__main__':
    main()

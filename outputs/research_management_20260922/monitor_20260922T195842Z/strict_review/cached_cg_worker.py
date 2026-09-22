#!/usr/bin/env python3
"""One hash-gated cached strict k19 CG; no graph build, MIP or successor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MODEL = 'fedf421461f94727e6b1292a0e7789ab76ed8587'
PARENT = '35770aae2c08e7d5a356cc3b673e67608e5b1036'
WRAPPER = '3bb32c1a84af73c97689dfc9f5ad43136cfcead9'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def graph_gate(root, plan):
    """Read-only proof verification; cache bytes are separately checked natively."""
    for name, digest in plan['graph_receipt_sha256'].items():
        require(sha(root / name) == digest, 'graph receipt hash mismatch: ' + name)
    cold = read(root / 'artifacts/cold.json')
    reload = read(root / 'attempts/772820_r0/reload.json')
    result = read(root / 'attempts/772820_r0/result.json')
    complete = read(root / 'attempts/772820_r0/COMPLETE.json')
    seal = read(root / 'artifacts/COLD_COMPLETE.json')
    require(complete['result_sha256'] == sha(root / 'attempts/772820_r0/result.json'), 'result seal mismatch')
    require(seal['cold_sha256'] == result['cold_sha256'] == sha(root / 'artifacts/cold.json'), 'cold seal mismatch')
    require(result['reload_sha256'] == sha(root / 'attempts/772820_r0/reload.json'), 'reload seal mismatch')
    require(result['status'] == 'passed' and result['model_commit'] == MODEL and result['wrapper_commit'] == WRAPPER, 'wrong completed graph identity')
    require(result['job_id'] == '772820' and result['restart'] == '0', 'wrong graph attempt')
    require(complete['cache_sha256'] == result['full_graph_sha256'] == plan['cache_sha256'], 'cache seal mismatch')
    require(result['cache_bytes'] == plan['cache_bytes'], 'cache byte count mismatch')
    require(result['cold_and_reload_separate_processes'] is True, 'process separation missing')
    for record in (cold, reload, result, complete, seal):
        require(record['manifest_sha256'] == plan['graph_manifest_sha256'], 'graph manifest identity mismatch')
    for name, record in (('cold', cold), ('reload', reload)):
        require(record['phase'] == name and record['status'] == 'passed', 'phase did not pass')
        require(record['model_provenance']['git_commit'] == MODEL and record['wrapper_commit'] == WRAPPER, 'phase source mismatch')
        require(record['fresh_initial_pool_equality_proved'] is True, 'fresh initializer proof absent')
        require(all(record[x] is False for x in ('solver_started', 'cg_started', 'mip_started')), 'unexpected solver in graph gate')
        initial = record['initial_pool']
        require((initial['inherited_columns'], initial['new_singleton_columns'], initial['initial_pool_columns'], initial['singleton_queries']) == (8343, 54, 8397, 331), 'initializer counts mismatch')
        require(initial['full_record_equality_to_saved_initial_pool'] is True and initial['normalized_fields'] == ['cg_checkpoint_id'], 'initializer equality missing')
        require(initial['new_checkpoint_id'] == plan['checkpoint_id'], 'new checkpoint mismatch')
        require(len(initial['singleton_checks']) == 331 and sum(r['novel'] for r in initial['singleton_checks']) == 54, 'singleton evidence incomplete')
        require(record['identity_sha256'] == plan['cache_identity_sha256'] and record['payload_sha256'] == plan['cache_payload_sha256'], 'cache identity mismatch')
        require(record['network'] == plan['expected_network'], 'network metrics mismatch')
        require(len(record['queries']) == 2 and [r['name'] for r in record['queries']] == ['zero', 'saved_pool_frequency'], 'wrong dual queries')
        for query in record['queries']:
            require(query['physical_replay'] == 'passed' and query['reduced_cost'] == query['independent_reduced_cost'], 'query replay/RC failure')
    require(reload['cold_reload_parity'] is True and cold['graph_fingerprint'] == reload['graph_fingerprint'], 'cold/reload graph parity absent')
    require(cold['initial_pool']['ordered_record_sha256'] == reload['initial_pool']['ordered_record_sha256'], 'initializer parity mismatch')
    require([r['record_sha256'] for r in cold['initial_pool']['singleton_checks']] == [r['record_sha256'] for r in reload['initial_pool']['singleton_checks']], 'singleton parity mismatch')
    for a, b in zip(cold['queries'], reload['queries']):
        require(all(a[k] == b[k] for k in ('duals_sha256', 'reduced_cost', 'route_sha256', 'route')), 'query parity mismatch')
    return result


def command(plan, attempt):
    original = list(plan['original_command'])
    require('--resume' not in original and original[original.index('--mode') + 1] == 'cg', 'unsafe original mode/resume')
    graph = Path(plan['graph_root'])
    inputs = graph / 'input_artifacts'
    original[:2] = [plan['python'], str(graph / 'model_code/src/run_capacity_speed_event_cg.py')]
    replacements = {'--instance': inputs / 'k19_instance.csv', '--prices': inputs / 'hourly_prices_flat.csv',
        '--reference-data-dir': inputs, '--out': attempt / 'result.json', '--pool-out': attempt / 'pool.jsonl',
        '--expected-commit': MODEL, '--inherit-status': inputs / 'k17_status.json',
        '--inherit-pool': inputs / 'k17_pool.jsonl', '--inherit-instance': inputs / 'k17_instance.csv'}
    for flag, value in replacements.items():
        original[original.index(flag) + 1] = str(value)
    require(original[original.index('--threads') + 1] == '8' and original[original.index('--cg-wall-s') + 1] == '14400', 'changed computational budget')
    return original + ['--graph-cache', str(graph / 'artifacts/k19_fedf4214.graph.cache'), '--inherit-compatible-commit', PARENT]


def worker(root, plan):
    job, restart = os.environ['SLURM_JOB_ID'], os.environ.get('SLURM_RESTART_COUNT', '0')
    attempt = root / 'attempts' / (job + '_r' + restart)
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    execution = {'job_id': job, 'restart': restart, 'host': os.uname().nodename, 'model_commit': MODEL,
        'plan_sha256': sha(root / 'manifest.json'), 'worker_sha256': sha(Path(__file__)), 'started_epoch': time.time(),
        'cg_started': False, 'mip_started': False, 'graph_built': False}
    write(attempt / 'execution.json', execution)
    try:
        require(restart == '0', 'requeue requires explicit same-commit checkpoint recovery review; no fresh duplicate solve')
        require(sha(Path(__file__)) == plan['worker_sha256'], 'worker hash mismatch')
        write(root / 'STARTED.json', execution)  # Exclusive campaign start; any second job fails closed.
        graph = Path(plan['graph_root'])
        require(sha(graph / 'manifest.json') == plan['graph_manifest_sha256'], 'graph manifest changed')
        gm = read(graph / 'manifest.json')
        graph_gate(graph, plan)
        code = graph / 'model_code'
        require(subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == MODEL, 'wrong model HEAD')
        require(not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'], text=True).strip(), 'dirty model source')
        for relative, expected in gm['model_source_sha256'].items():
            require(sha(code / relative) == expected, 'model source hash mismatch: ' + relative)
        for item in gm['input_files'].values():
            require(sha(graph / 'input_artifacts' / item['local_name']) == item['sha256'], 'input snapshot changed')
        for name, expected in gm['native_lineage_sha256'].items():
            require(sha(Path(gm['native_lineage_attempt']) / name) == expected, 'native lineage proof changed')
        cache = graph / 'artifacts/k19_fedf4214.graph.cache'
        require(cache.stat().st_size == plan['cache_bytes'] and sha(cache) == plan['cache_sha256'], 'cache bytes changed')
        require(read(graph / 'input_artifacts/k19_command.json') == plan['original_command'], 'original command drift')
        argv = command(plan, attempt)
        environment = dict(os.environ, GRB_LICENSE_FILE='/share/apps/software/gurobi/gurobi.lic',
            OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        for key in ('LM_LICENSE_FILE', 'PYTHONPATH', 'PYTHONHOME'):
            environment.pop(key, None)
        license_command = [plan['python'], str(code / 'src/gurobi_preflight.py')]
        write(attempt / 'command.json', argv)
        write(attempt / 'license_command.json', license_command)
        with (attempt / 'license.log').open('x') as log:
            license_process = subprocess.run(license_command, cwd=code, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        require(license_process.returncode == 0, 'native license preflight failed; CG not started')
        write(attempt / 'CG_STARTED.json', {**execution, 'cg_started': True, 'license_log_sha256': sha(attempt / 'license.log')})
        with (attempt / 'cg.log').open('x') as log:
            process = subprocess.run(argv, cwd=code, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=plan['cg_process_timeout_s'])
        require(process.returncode == 0, 'CG process failed; preserve checkpoint for explicit recovery review')
        result = read(attempt / 'result.json')
        require(result['provenance']['git_commit'] == MODEL and result['checkpoint']['id'] == plan['checkpoint_id'], 'CG result identity mismatch')
        require(result['checkpoint']['initial_pool_columns'] == 8397 and result['checkpoint']['resumed'] is False, 'wrong CG initialization')
        require(result['cg_allowance_scope'] == 'solver_after_verified_graph_load' and result['network_build_s'] == 0.0, 'graph rebuilt or wrong allowance')
        require(result['graph_cache']['identity_sha256'] == plan['cache_identity_sha256'] and result['graph_cache']['payload_sha256'] == plan['cache_payload_sha256'], 'wrong CG graph')
        require(result['physics'] == plan['physics'] and result['pool_sha256'] == sha(attempt / 'pool.jsonl'), 'CG physics/pool mismatch')
        summary = {**execution, 'cg_started': True, 'status': 'process_completed', 'cg_status': result['status'],
            'cg_certified': result['certified_rc_optimal'], 'result_sha256': sha(attempt / 'result.json'),
            'pool_sha256': result['pool_sha256'], 'runtime_s': time.perf_counter() - started,
            'mip_queued': False, 'successor_queued': False}
        write(attempt / 'worker_result.json', summary)
        write(attempt / 'COMPLETE.json', {'worker_result_sha256': sha(attempt / 'worker_result.json'), 'plan_sha256': execution['plan_sha256']})
    except BaseException as error:
        write(attempt / 'FAILED.json', {**execution, 'cg_started': (attempt / 'CG_STARTED.json').exists(), 'error': str(error), 'runtime_s': time.perf_counter() - started,
            'checkpoint_exists': (attempt / 'pool.jsonl').exists(), 'automatic_retry': False})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--manifest-sha256', required=True)
    args = p.parse_args()
    root = args.root.resolve()
    require(sha(root / 'manifest.json') == args.manifest_sha256, 'unapproved manifest')
    worker(root, read(root / 'manifest.json'))


if __name__ == '__main__':
    main()

"""Execute the one authorized bounded diagnostic; never submit other jobs."""
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
HELPER = Path('/home/nc437/ladder-lite/efficiency_validation_20260912/code-baseline/scripts/efficiency_validation_20260912/campaign.py')
spec = importlib.util.spec_from_file_location('campaign', HELPER)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def main():
    manifest = c.read(ROOT / 'execution_manifest.json')
    for entry in manifest['files']:
        c.authenticate(ROOT / entry['name'], entry['sha256'])
    c.code_check(HELPER.parents[2], manifest['watchdog_source_commit'])
    plan = c.read(ROOT / 'recovery_plan.json')
    code = Path(plan['command'][plan['command'].index('--solver') + 1]).parents[1]
    c.code_check(code, plan['original_source_commit'])
    for name, expected in plan['input_hashes'].items():
        c.authenticate(code / 'data' / name, expected)
    token = os.environ['SLURM_JOB_ID'] + '_r' + os.environ.get('SLURM_RESTART_COUNT', '0')
    attempt = ROOT / 'attempts' / token
    attempt.mkdir(parents=True, exist_ok=False)
    c.write(attempt / 'identity.json', {'manifest_sha256': c.digest(ROOT / 'execution_manifest.json'),
            'job': token, 'source_commit': plan['original_source_commit'],
            'scope': 'bounded graph diagnostic; no CG or MIP result',
            'slurm': {k:v for k,v in os.environ.items() if k.startswith('SLURM_')}})
    cmd = list(plan['command'])
    for flag, name in [('--progress','graph_progress.jsonl'),('--stacks','graph_stacks.txt'),
                       ('--event-network-cache','network.pkl'),('--phase-telemetry','graph_phases.jsonl')]:
        cmd[cmd.index(flag)+1] = str(attempt / name)
    env = os.environ.copy()
    for key in ['PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LM_LICENSE_FILE']:
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
               GRB_LICENSE_FILE='/share/apps/software/gurobi/gurobi.lic')
    result = c.run_process(cmd, code, attempt / 'run', plan['external_process_watchdog_seconds'], env)
    state = ('diagnostic_budget_exhausted' if result.get('watchdog_triggered')
             else 'cache_build_completed' if result['returncode'] == 0 else 'diagnostic_error')
    c.write(attempt / 'diagnostic_result.json', {'state': state, 'execution': result,
            'CG_certificate': None, 'MIP_proof': None,
            'files': [{'path':str(p), 'sha256':c.digest(p), 'bytes':p.stat().st_size}
                      for p in attempt.rglob('*') if p.is_file()]})
    return 1 if state == 'diagnostic_error' else 0


if __name__ == '__main__':
    sys.exit(main())

"""Run a prospectively paired, fresh Partille search using the pinned capacity solver.

The public manifest contains hashes and algorithm settings. Trip records, route columns
and solver logs stay under the operator's research directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PIN = '3a0ab6a8ab50e4af7d5a7be7688d5ffa1fbbf78c'
HERE = Path(__file__).resolve().parent
PHASE_SECONDS = dict(cg=10800, heuristic=5400, mip=3600)
ARMS = ('simple', 'repair')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def cells(k, pilot_control=False):
    if pilot_control:
        return [(2, 8, 'simple')]
    return [(c, k, arm) for c in range(1, 7) for arm in ARMS
            if k != 8 or c != 2]


def audit_inputs(row, data, price):
    paths = {'instance': data / row['csv'], 'prices': price,
             'locations': data / 'Ref_dict.csv', 'deadheads': data / 'par_ref_dhd.csv'}
    expected = dict(read(HERE / 'design.json')['shared_hashes'], instance=row['instance_sha256'])
    if {key:sha(value) for key,value in paths.items()} != expected:
        raise ValueError('Input files differ from the frozen campaign')
    return paths


def solver_argv(row, arm, solver, data, price, out):
    args = [sys.executable, '-u', str(solver / 'src/run_capacity_terminal_cg.py'),
            '--data-dir', str(data), '--csv', row['csv'], '--prices', str(price),
            '--target', '0', '--fleet-cap', str(row['time_only_floor'] + 4),
            '--cg-seconds', str(PHASE_SECONDS['cg']), '--mip-seconds', str(PHASE_SECONDS['mip']),
            '--dive-seconds', str(PHASE_SECONDS['heuristic'] if arm == 'simple' else 0),
            '--charge-kw','240','--battery-kwh','240','--start-fee','5','--non-strict-tariff',
            '--station-kw','PARX=60','--out',str(out)]
    if arm == 'repair':
        args += ['--integer-pricing-seconds', str(PHASE_SECONDS['heuristic'])]
    return args


def run_case(opts):
    chain,k,arm = cells(opts.k, opts.pilot_control)[opts.index]
    cell = f'c{chain}_k{k:02d}'
    design = read(HERE / 'design.json')
    row = design['cases'][cell]
    root = Path(os.environ['CAMPAIGN_ROOT'])
    solver = Path(os.environ['CAPACITY_SOLVER'])
    data = Path(os.environ['CAPACITY_DATA'])
    price = data / 'hourly_prices_flat.csv'
    pin = subprocess.check_output(['git','-C',str(solver),'rev-parse','HEAD'],text=True).strip()
    if pin != PIN or subprocess.check_output(['git','-C',str(solver),'status','--porcelain','--untracked-files=no']):
        raise ValueError('Pinned solver revision or tracked source differs')
    paths = audit_inputs(row, data, price)
    attempt = 'job' + os.environ['SLURM_JOB_ID'] + '_r' + os.environ.get('SLURM_RESTART_COUNT','0')
    dest = root / 'paired' / cell / arm / attempt
    dest.mkdir(parents=True, exist_ok=False)
    argv = solver_argv(row, arm, solver, data, price, dest/'solve')
    ncpu = int(os.environ['SLURM_CPUS_PER_TASK'])
    receipt = dict(case=cell, arm=arm, attempt=attempt, solver_commit=pin, argv=argv,
                   hashes={n:sha(p) for n,p in paths.items()}, phase_limits_s=PHASE_SECONDS,
                   source=sha(__file__), design=sha(HERE/'design.json'), cpus=ncpu,
                   initialization='fresh', ancestor_wall_seconds=0, allocation_limit_hours=20,
                   job_family=os.environ.get('SLURM_ARRAY_JOB_ID',os.environ['SLURM_JOB_ID']),
                   started_unix=time.time())
    write(dest/'receipt.json',receipt)
    tick = time.monotonic()
    with (dest/'stdout.log').open('w') as log:
        p = subprocess.run(argv, cwd=solver/'src', stdout=log, stderr=subprocess.STDOUT)
    receipt.update(exit_code=p.returncode, solver_wall_seconds=time.monotonic()-tick)
    summary = dest/'solve/summary.json'
    if summary.exists():
        d = read(summary)
        receipt.update(summary_sha256=sha(summary), pricing_certified=d.get('cg_pricing_certified'),
                       charging_stage=d.get('charging_stage'), fleet_stage=d.get('fleet_stage'),
                       cg_stop=d.get('cg_stop'), graph_seconds=d.get('graph_seconds'))
    selected = dest/'solve/selected_routes.json'
    if selected.exists() and p.returncode == 0:
        validator = root/'validate_selection.py'
        if not validator.is_file():
            raise FileNotFoundError('Missing independent selection validator')
        verify = [sys.executable,str(validator),'--summary',str(summary),'--selected',str(selected),
                  '--out',str(dest/'verification.json'),'--arm',('blind' if arm == 'repair' else 'baseline'),'--start-fee','5']
        env = dict(os.environ, PYTHONPATH=str(solver/'src'))
        with (dest/'validation.log').open('w') as log:
            v = subprocess.run(verify, env=env,stdout=log,stderr=subprocess.STDOUT)
        receipt.update(validator_exit=v.returncode, validator_sha256=sha(validator),
                       selected_sha256=sha(selected))
    receipt.update(end_to_end_seconds=time.monotonic()-tick, ended_unix=time.time())
    write(dest/'receipt.json',receipt)
    if p.returncode == 3 and summary.is_file() and read(summary).get('selection_state') == 'no_selection':
        return 0  # Scientific miss, recorded separately from a crash.
    return p.returncode or int(not (selected.exists() and (dest/'verification.json').exists()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',type=int,choices=[8,10],required=True)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--pilot-control',action='store_true')
    sys.exit(run_case(parser.parse_args()))

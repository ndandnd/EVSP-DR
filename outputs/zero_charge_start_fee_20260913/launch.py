#!/usr/bin/env python3
"""Submit each recorded pair once; preserve partial launch and scheduler evidence."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/nc437/ladder-lite/zero_charge_start_fee_20260913')
SLURM = Path('/usr/local/slurm/slurm-25.05.5/bin')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, data):
    p = Path(path)
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    os.replace(temp, p)


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    deployment = json.loads((ROOT / 'deployment.json').read_text())
    for rel, expected in deployment['tooling_sha256'].items():
        assert sha(ROOT / rel) == expected, rel
    commands = []
    for index, pair in enumerate(manifest['pairs']):
        commands.append([str(SLURM/'sbatch'), '--parsable', '--partition=default_partition',
                         '--exclude=scaglione-compute-01', '--cpus-per-task=8', '--mem=96G',
                         '--time=04:00:00', '--no-requeue', '--job-name=fee_'+pair['id'],
                         '--output='+str(ROOT/'logs/%x_%j.out'), '--error='+str(ROOT/'logs/%x_%j.err'),
                         '--chdir='+str(ROOT), str(ROOT/'worker.sh'), str(index)])
    if '--submit' not in sys.argv:
        print(json.dumps({'commands': commands, 'count': len(commands)}, indent=2))
        return
    validation = json.loads((ROOT / 'validation/PASS.json').read_text())
    assert validation['commit'] == manifest['cg_commit']
    assert {r['arm'] for r in validation['checks']} == {'fee0', 'fee5'}
    assert all(r['metrics']['cost_components_reconcile'] is True for r in validation['checks'])
    (ROOT/'logs').mkdir(exist_ok=True)
    jobs_path = ROOT/'jobs.json'
    fd = os.open(jobs_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(fd)
    ledger = dict(schema='evsp-zero-charge-start-fee-submission-v1', started_utc=now(),
                  manifest_sha256=sha(ROOT/'manifest.json'), deployment_sha256=sha(ROOT/'deployment.json'), jobs=[])
    write(jobs_path, ledger)
    for pair, command in zip(manifest['pairs'], commands):
        row = dict(pair_id=pair['id'], case_id=pair['case_id'], order=pair['order'], command=command,
                   submit_started_utc=now(), cg_commit=manifest['cg_commit'], mip_commit=manifest['mip_commit'])
        ledger['jobs'].append(row)
        write(jobs_path, ledger)
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            row.update(job_id=result.stdout.strip().split(';')[0], submitted_utc=now(),
                       submission_stdout=result.stdout, submission_stderr=result.stderr)
            write(jobs_path, ledger)
            control = subprocess.check_output([str(SLURM/'scontrol'), 'show', 'job', row['job_id']], text=True)
            row.update(scontrol=control, exclusion_verified='ExcNodeList=scaglione-compute-01' in control,
                       partition_verified='Partition=default_partition' in control)
            assert row['exclusion_verified'] and row['partition_verified'], 'Unexpected scheduler resource settings'
        except BaseException as exc:
            row.update(error=repr(exc))
            write(jobs_path, ledger)
            raise
        write(jobs_path, ledger)
        print(json.dumps({'pair': pair['id'], 'job': row['job_id'], 'verified': True}), flush=True)
    ledger['finished_utc'] = now()
    write(jobs_path, ledger)


if __name__ == '__main__':
    main()

"""Install once into an isolated campaign. Record Slurm IDs under a lock as jobs are issued."""
import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path
from campaign import HERE, PIN, audit_inputs, read, write


def run(args):
    return subprocess.check_output([str(x) for x in args],text=True).strip()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--solver',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True)
    p.add_argument('--slurm',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--blind',type=int,required=True)
    p.add_argument('--witness',type=int,required=True)
    p.add_argument('--audit',type=int,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args()
    if 'scaglione-compute-01' not in a.policy.read_text() or run(['git','-C',a.solver,'rev-parse','HEAD'])!=PIN:
        raise ValueError('resource policy or execution pin missing')
    for row in read(HERE/'design.json')['cases'].values():
        audit_inputs(row,a.data,a.data/'hourly_prices_flat.csv')
    if not (a.campaign/'validate_selection.py').is_file() or not (a.baseline/'summary.json').is_file():
        raise ValueError('missing validator or primary baseline')
    (a.campaign/'logs').mkdir(exist_ok=True)
    os.environ.update(CAMPAIGN_ROOT=str(a.campaign),CAPACITY_SOLVER=str(a.solver),CAPACITY_DATA=str(a.data),
                      SOLVER_PYTHON=str(a.python),C2_BASELINE=str(a.baseline))
    receipt=a.campaign/'paired_jobs.json'
    with (a.campaign/'paired_jobs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        jobs=read(receipt) if receipt.exists() else {'blind':str(a.blind),'witness':str(a.witness),'audit':str(a.audit)}
        if any(jobs[k]!=str(getattr(a,k)) for k in ('blind','witness','audit')):
            raise ValueError('conflicting pilot job IDs')
        def submit(key,flags,script,*args):
            if key in jobs:
                return jobs[key]
            name='drN02'+key
            # A request that succeeded remotely but failed before the ID was saved must be reconciled first.
            if run([a.slurm/'sacct','-X','-n','-u',os.environ['USER'],'-S','2026-11-02','--name',name,'-o','JobID']):
                raise ValueError('existing Slurm job is missing from ledger: '+name)
            jid=run([a.slurm/'sbatch','--parsable','--job-name='+name,
                     '--output='+str(a.campaign/'logs/%x_%A_%a.out'),'--kill-on-invalid-dep=yes',
                     *flags,HERE/script,*args]).split(';')[0]
            if not jid.isdecimal():
                raise RuntimeError('unexpected Slurm response')
            jobs[key]=jid
            write(receipt,jobs)
            return jid
        ctrl=submit('control',[], 'run_pair.sbatch','8','--pilot-control')
        g8=submit('gate8',[f'--dependency=afterany:{a.audit}:{a.witness}'],'gate.sbatch','eight')
        eight=submit('eight',[f'--dependency=afterok:{g8}','--array=0-9%10'],'run_pair.sbatch','8')
        g10=submit('gate10',[f'--dependency=afterany:{eight}:{ctrl}:{g8}'],'gate.sbatch','ten')
        submit('ten',[f'--dependency=afterok:{g10}','--array=0-11%12'],'run_pair.sbatch','10')
        print(json.dumps(jobs,sort_keys=True))
if __name__=='__main__':
    main()

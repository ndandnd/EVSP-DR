"""Install once in a fresh private prefix workspace, verify the release gates, and submit both arrays."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from sparse_worker import PIN,HERE,audit_stage_input,read,require,sha,write

SSH_BASE=Path(os.environ.get('LADDER_ROOT',Path.home()/'ladder-lite'))
PY=Path(os.environ.get('SOLVER_PYTHON',Path.home()/'evsp_env/bin/python'))
SLURM=Path('/usr/local/slurm/current/bin')
DATA=SSH_BASE/'chain_extension_33_40_20260921/code/data'
OWNER=SSH_BASE/'capacity_v15_20261002/code'
PILOT=SSH_BASE/'plug_diagnostic_20261102'
ROOT=SSH_BASE/'sparse_prefix_20261102'
ARTIFACTS=('sparse_worker.py','sparse.sbatch','receiver.patch','paired_design.json','prefix_k05.json','test_sparse.py','deploy.py')


def call(*argv):
    return subprocess.check_output([str(a) for a in argv],text=True).strip()


def main():
    require(HERE==ROOT/'code','install into isolated prefix workspace only')
    policy=(SSH_BASE/'SCAGLIONE_RESOURCE_POLICY.md').read_text()
    require('scaglione-compute-01' in policy and '50' in policy,'policy unavailable')
    jobs=read(PILOT/'paired_jobs.json')
    require((jobs['gate8'],jobs['gate10'])==('905844','905846'),'gate ledger changed')
    require(sha(HERE/'paired_design.json')==sha(PILOT/'paired_code/design.json'),'fresh paired design differs')
    require(call('git','-C',OWNER,'rev-parse','HEAD')==PIN,'owner base mismatch')
    design=read(HERE/'paired_design.json')
    for row in list(design['cases'].values())+list(read(HERE/'prefix_k05.json').values()):
        audit_stage_input(row,DATA,design['shared_hashes'])
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        src=ROOT/'solver'
        evidence=ROOT/'deployment.json'
        if evidence.exists():
            meta=read(evidence)
            require(meta['files_sha256']=={n:sha(HERE/n) for n in ARTIFACTS},'deployment source changed')
        else:
            require(not src.exists(),'partial installation: reconcile before retry')
            call('git','clone','--shared','--no-checkout',OWNER,src)
            call('git','-C',src,'sparse-checkout','init','--cone')
            call('git','-C',src,'sparse-checkout','set','src','tests')
            call('git','-C',src,'checkout','--detach',PIN)
            call('git','-C',src,'apply','--check',HERE/'receiver.patch')
            call('git','-C',src,'apply','--index',HERE/'receiver.patch')
            call('git','-C',src,'-c','user.name=EVSP sparse-prefix experiment','-c','user.email=experiment@localhost',
                 'commit','-m','Rebuild authenticated sparse-prefix sequences on receiver graph')
            shutil.copyfile(PILOT/'validate_selection.py',ROOT/'validate_selection.py')
            pins=read(HERE/'approved_hashes.json')
            require(sha(ROOT/'validate_selection.py')==pins['validator'] and sha(HERE/'receiver.patch')==pins['patch'],
                    'qualified patch/validator bytes changed')
            meta=dict(base_commit=PIN,patched_commit=call('git','-C',src,'rev-parse','HEAD'),
                 initializer_sha256=sha(src/'src/prefix_initializer.py'),solver_source_sha256=sha(src/'src/run_capacity_terminal_cg.py'),
                 files_sha256={n:sha(HERE/n) for n in ARTIFACTS},validator_sha256=sha(ROOT/'validate_selection.py'),
                 policy_sha256=sha(SSH_BASE/'SCAGLIONE_RESOURCE_POLICY.md'))
            write(evidence,meta)
        require(call('git','-C',src,'rev-parse','HEAD')==meta['patched_commit'] and
                not call('git','-C',src,'status','--porcelain','--untracked-files=no'),'isolated checkout changed')
        # Check every real nesting relation on the private host before reserving work.
        sys.path.insert(0,str(src/'src'))
        from prefix_initializer import remap_sequences,row_index
        rows=dict(design['cases'],**read(HERE/'prefix_k05.json'))
        nesting={}
        for c in range(1,7):
            for k,h in ((5,8),(8,10)):
                a,b=(DATA/rows[f'c{c}_k{v:02}']['csv'] for v in (k,h))
                parent_count=len(row_index(a))
                mapped=remap_sequences(a,b,[{'trips':[t]} for t in range(parent_count)])
                nesting[f'c{c}_{k}_{h}']=dict(parent_trips=parent_count,receiver_trips=len(row_index(b)),
                    remapped_distinct=len({x[0] for x in mapped}),nested=True)
        write(ROOT/'nesting.json',nesting)
        (ROOT/'logs').mkdir(exist_ok=True)
        ledger=ROOT/'jobs.json'
        result=read(ledger) if ledger.is_file() else {}
        os.environ.update(PREFIX_ROOT=str(ROOT),CAPACITY_DATA=str(DATA),SOLVER_PYTHON=str(PY))
        for k in (8,10):
            tag=f'k{k:02}'
            gate=jobs[f'gate{k}']
            if tag in result:
                require(result[tag]['gate']==gate,'gate changed')
                continue
            name='drN02sp'+tag
            require(not call(SLURM/'sacct','-X','-n','-u',os.environ['USER'],'-S','2026-11-02','--name',name,'-o','JobID') and
                    not call(SLURM/'squeue','-h','-u',os.environ['USER'],'-n',name,'-o','%A'),
                    'existing family absent from ledger, reconcile before submitting')
            job=call(SLURM/'sbatch','--parsable','--job-name='+name,'--array=0-5%6','--dependency=afterok:'+gate,
                     '--kill-on-invalid-dep=yes','--output='+str(ROOT/'logs/%x_%A_%a.out'),HERE/'sparse.sbatch',str(k)).split(';')[0]
            require(job.isdecimal(),'unexpected Slurm job response')
            result[tag]=dict(job=job,gate=gate,array='0-5%6',dependency='afterok:'+gate)
            write(ledger,result)
        for tag,r in result.items():
            output=call(SLURM/'scontrol','show','job',r['job'])
            require('ExcNodeList=scaglione-compute-01' in output and 'TimeLimit=20:00:00' in output
                    and 'NumCPUs=8' in output and 'ArrayTaskThrottle=6' in output,'effective resources mismatch')
            (ROOT/(tag+'_scontrol.txt')).write_text(output+'\n')
        print(json.dumps(result))


if __name__=='__main__':
    main()

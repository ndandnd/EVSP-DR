"""Native smoke checks in an isolated root; never publish research outcomes."""
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import subprocess

B=Path('/home/nc437/ladder-lite/overnight_diagnostics_20260914')
T=B.with_name(B.name+'_smoke_resume_v3')
FIRST=B.with_name(B.name+'_smoke')
PY='/home/nc437/evsp_env/bin/python'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def option(argv,key,value):
    argv[argv.index(key)+1]=str(value)


def main():
    v=json.loads((B/'manifest.json').read_text())
    assert len(v['cases'])==99
    T.mkdir(exist_ok=False)
    (T/'cases').mkdir()
    for name in ['worker.py','worker.sub']:
        shutil.copy2(B/name,T/name)
    completed=['c200','complementary','c200_mip','complementary_mip']
    records={cid:json.loads((FIRST/'cases'/cid/'completion.json').read_text()) for cid in completed}
    for cid,r in records.items():
        assert r['usable'] and sha(r['result_path']) == r['result_sha256']
    for name in ['worker.py','worker.sub']:
        assert sha(FIRST/name) == sha(B/name)
    cases={}
    c=copy.deepcopy(v['cases']['w4_k19_resume8h'])
    prior=json.loads(Path(c['resume_from']).read_text())
    c.update(id='resume_probe',is_validation=True,
        solver_budget_s=int(prior['wall_s'])+2400,watchdog_s=3000)
    # Native max-iters counts NEW iterations after a resume, not lifetime iterations.
    option(c['argv'],'--max-iters',1)
    option(c['argv'],'--wall-limit-s',c['solver_budget_s'])
    cases[c['id']]=c
    smoke={'schema':v['schema'],'cases':cases,'tooling_sha256':
        {n:sha(T/n) for n in ['worker.py','worker.sub']}}
    (T/'manifest.json').write_text(json.dumps(smoke,indent=2)+'\n')
    for cid in ['resume_probe']:
        subprocess.run(['bash',str(T/'worker.sub'),str(T),cid],check=True)
    records.update({cid:json.loads((T/'cases'/cid/'completion.json').read_text()) for cid in cases})
    assert all(r['usable'] for r in records.values())
    resumed=json.loads(Path(records['resume_probe']['result_path']).read_text())
    assert resumed['final']['iter'] > prior['final']['iter'], 'No new CG iteration completed'
    assert all(records[cid]['physical_replay_validated'] for cid in ['c200_mip','complementary_mip'])
    result={'status':'passed','manifest_sha256':sha(B/'manifest.json'),
        'smoke_root':str(T),'smoke_manifest_sha256':sha(T/'manifest.json'),
        'job_id':os.environ['SLURM_JOB_ID'],
        'prior_job_id':'155072',
        'prior_probe_jobs':['155072','155894'],
        'prior_probe_issue':'The30s and600s added allowances expired during parent-pool replay, before a new CG iteration. Compatible prior final_lp was retained but final remained null and the worker correctly rejected publication. Native max-iters counts new iterations; the earlier zero-iteration explanation was incorrect. This check allows one new iteration and2400s for replay/setup/CG. Production commands remain unchanged.',
        'tests':['200-column event CG','complementary event CG','both dependent two-stage MIPs with replay','copied native same-k checkpoint'],
        'completion_files':{cid:{'path':str((FIRST if cid in completed else T)/'cases'/cid/'completion.json'),'sha256':sha((FIRST if cid in completed else T)/'cases'/cid/'completion.json')} for cid in records}}
    (B/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    subprocess.run([PY,str(B/'campaign.py'),'submit'],check=True)
    subprocess.run([PY,str(B/'verify_launch.py')],check=True)


if __name__=='__main__':
    main()

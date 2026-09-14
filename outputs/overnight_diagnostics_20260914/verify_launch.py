"""Verify submitted identities/resources/dependencies and record live admission."""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json
import subprocess

B=Path('/home/nc437/ladder-lite/overnight_diagnostics_20260914')
S='/usr/local/slurm/slurm-25.05.5/bin/'

def read(p):
    return json.loads(p.read_text())

def main():
    manifest=read(B/'manifest.json')
    jobs=read(B/'jobs.json')
    mapping=read(B/'case_jobs.json')
    assert len(mapping)==len(jobs)==len({j['job_id'] for j in jobs})
    for j in jobs:
        c=manifest['cases'][j['case_id']]
        expected=[mapping[c['source_case']]] if c.get('source_case') else []
        assert j['dependencies']==expected
        control=j['scontrol']
        assert 'Partition=default_partition' in control
        assert 'ExcNodeList=scaglione-compute-01' in control
        assert 'Requeue=1' in control and 'CPUs/Task=8' in control
        assert '--mem='+c['resources']['mem'] in j['argv']
        if expected:
            assert 'Dependency=afterok:'+expected[0] in control
        else:
            assert 'Dependency=(null)' in control
    queue=subprocess.check_output([S+'squeue','-u','nc437','-h','-o','%i|%j|%T|%R|%E'],text=True)
    submitted={j['job_id'] for j in jobs}
    rows=[]
    for line in queue.splitlines():
        values=line.split('|')
        if values[0] in submitted:
            rows.append(dict(zip(['job_id','name','state','reason_or_node','dependency'],values)))
    issues=[r for r in rows if 'NeverSatisfied' in r['reason_or_node']]
    assert not issues, issues
    result=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        manifest_sha256=hashlib.sha256((B/'manifest.json').read_bytes()).hexdigest(),
        submitted=len(jobs),independent=sum(not j['dependencies'] for j in jobs),
        dependent=sum(bool(j['dependencies']) for j in jobs),
        unsubmitted_cases=sorted(set(manifest['cases'])-set(mapping)),
        resources_and_dependencies_verified=True,
        live_states=dict(Counter(r['state'] for r in rows)),
        pending_reasons=dict(Counter(r['reason_or_node'] for r in rows if r['state']=='PENDING')),
        jobs=rows,
        checks=['Unique job for each submitted case','Frozen case settings',
                'Default partition and reserved-node exclusion','Eight CPUs and declared memory',
                'Requeue enabled','Each MIP waits only for its own new CG',
                'Independent jobs have no scheduler dependency','No invalid live dependencies'])
    (B/'scheduler_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['jobs','checks']}))

if __name__=='__main__':
    main()

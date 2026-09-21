"""Restart-safe campaign wrapper; all mathematical work stays in the frozen model."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, resource, signal, subprocess, sys, time
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(p)
def verify():
    code=read(P/'code_receipt.json')
    for rel,digest in code['files'].items():assert sha(P/rel)==digest,rel
    assert sha(P/'manifest.json')==code['manifest_sha256']
    return code
def run(index,seconds=None,smoke=False):
    args=[sys.executable,str(P/'run_case.py'),str(index)]
    if seconds is not None:args+=['--seconds',str(seconds)]
    if smoke:args+=['--smoke']
    proc=subprocess.Popen(args,start_new_session=True)
    caught=[]
    def forward(signum,_frame):
        caught.append(signum)
        if proc.poll() is None:
            try:os.killpg(proc.pid,signum)
            except ProcessLookupError:pass
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sig,forward)
    try:rc=proc.wait(timeout=1050 if not smoke else 120)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        return 124
    return 128+caught[-1] if caught else rc
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','array']);args=ap.parse_args()
    code=verify();manifest=read(P/'manifest.json')
    if args.mode=='smoke':
        import gurobipy as gp
        model=gp.Model('license_2001');model.Params.OutputFlag=0;model.Params.Threads=1
        x=model.addVars(2001,lb=0,ub=1,obj=1);model.addConstr(gp.quicksum(x.values())>=1);model.optimize()
        assert model.Status==gp.GRB.OPTIMAL;model.dispose()
        checks=[]
        for index in [0,6,17]:
            assert run(index,20,True)==0
            case=manifest['cases'][index]
            matches=sorted((P/'smoke'/case['case_id']).glob('*/receipt.json'),key=lambda p:p.stat().st_mtime)
            receipt=read(matches[-1]);validation=read(matches[-1].parent/'validation.json')
            assert receipt['slurm_job_id']==os.environ['SLURM_JOB_ID'] and receipt['physical_validation'] is True
            assert all(validation.get(k) is True for k in ['corrupt_terminal_rejected','corrupt_charge_rejected','corrupt_terminal_floor_rejected','corrupt_cost_rejected'])
            checks.append(dict(case_id=case['case_id'],receipt=str(matches[-1]),receipt_sha256=sha(matches[-1])))
        save(P/'native_validation.json',dict(status='passed',job_id=os.environ['SLURM_JOB_ID'],code_commit=code['commit'],
            manifest_sha256=sha(P/'manifest.json'),code_receipt_sha256=sha(P/'code_receipt.json'),
            unrestricted_2001_variable_license=True,checks=checks,solver_version=gp.gurobi.version()))
        return
    gate=read(P/'native_validation.json');assert gate['status']=='passed'
    assert gate['manifest_sha256']==sha(P/'manifest.json') and gate['code_receipt_sha256']==sha(P/'code_receipt.json')
    index=int(os.environ['SLURM_ARRAY_TASK_ID']);case=manifest['cases'][index]
    attempt=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0')+'_'+str(time.time_ns())
    out=P/'execution'/case['case_id']/attempt
    record=dict(case_id=case['case_id'],case_index=index,job_id=os.environ['SLURM_JOB_ID'],
        array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task_id=index,attempt=attempt,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(P/'manifest.json'),
        source_execution_commit=code['commit'],resources=manifest['resources'],status='running')
    save(out/'execution.json',record)
    rc=run(index)
    record.update(returncode=rc,status='completed' if rc==0 else 'failed_or_interrupted',
        ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),max_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    save(out/'execution.json',record)
    if rc:raise SystemExit(rc)
if __name__=='__main__':main()

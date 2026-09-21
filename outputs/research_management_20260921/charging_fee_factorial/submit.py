"""Submit only this campaign; immutable gates prevent a duplicate array."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess
P=Path(__file__).resolve().parent;S='/usr/local/slurm/slurm-25.05.5/bin/'
ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','production']);args=ap.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
code=json.loads((P/'code_receipt.json').read_text())
for rel,digest in code['files'].items():assert sha(P/rel)==digest,rel
record=P/('native_submission.json' if args.mode=='smoke' else 'submission.json')
assert not record.exists(),'Existing submission retained; reconcile instead of blindly resubmitting'
queue=subprocess.check_output([S+'squeue','-u','nc437','-h','-o','%i|%j|%T|%o'],text=True)
assert not any('worker.sub' in line and str(P) in line for line in queue.splitlines()),'Campaign already has live workers'
argv=[S+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01',
    '--cpus-per-task=2','--mem=8G','--time='+('00:05:00' if args.mode=='smoke' else '00:20:00'),
    '--requeue','--job-name=drFee_'+args.mode,'--output='+str(P/'logs/%x_%A_%a_%j.out'),
    '--error='+str(P/'logs/%x_%A_%a_%j.err')]
if args.mode=='production':
    gate=json.loads((P/'native_validation.json').read_text());assert gate['status']=='passed'
    assert gate['manifest_sha256']==sha(P/'manifest.json') and gate['code_receipt_sha256']==sha(P/'code_receipt.json')
    argv+=['--array=0-17%18']
argv+=[str(P/'worker.sub'),'smoke' if args.mode=='smoke' else 'array']
(P/'logs').mkdir(exist_ok=True)
output=subprocess.check_output(argv,text=True).strip();job=output.split(';')[0];assert job.isdigit()
receipt=dict(job_id=job,argv=argv,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    manifest_sha256=sha(P/'manifest.json'),code_commit=code['commit'],queue_before=queue)
record.write_text(json.dumps(receipt,indent=2)+'\n')
state=subprocess.check_output([S+'scontrol','show','job',job,'-o'],text=True)
assert 'ExcNodeList=scaglione-compute-01' in state and 'Partition=default_partition' in state
assert 'MinMemoryNode=8G' in state and 'CPUs/Task=2' in state
assert 'Requeue=1' in state
if args.mode=='production':assert 'ArrayTaskThrottle=18' in state
receipt['scontrol']=state;receipt['scheduler_verified']=True;record.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

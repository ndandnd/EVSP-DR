"""Submit a single locked three-cell array and record the effective Slurm configuration."""
import datetime, hashlib, json, os, re, subprocess, sys
from pathlib import Path
root=Path.home()/'ladder-lite/partille_soc_grid_20261103'; root.resolve(strict=True); os.chdir(root)
slurm=Path('/usr/local/slurm/slurm-25.05.5/bin'); expected=sys.argv[1]; actual=hashlib.sha256((root/'registration/manifest.json').read_bytes()).hexdigest()
if actual!=expected:raise ValueError('registration mismatch')
queue=subprocess.check_output([slurm/'squeue','-h','-u',os.environ['USER'],'-o','%j'],text=True).splitlines()
if 'drSG5k8' in queue:raise RuntimeError('matching array already active')
with (root/'submit_intent.json').open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'registration_sha256':actual,'idempotency_key':'C2_C3_C6_k8_SOC5_20261103'},f)
job=subprocess.check_output([slurm/'sbatch','--parsable','code/run.sbatch',expected],text=True).strip().split(';')[0]
if not job.isdigit(): raise RuntimeError('bad job receipt')
(root/'submitted_job.txt').write_text(job+'\n'); results=[]
for i in range(3):
    s=subprocess.check_output([slurm/'scontrol','-o','show','job',f'{job}_{i}'],text=True); val=dict(re.findall(r'(?:^|\s)(\w+)=([^\s]*)',s)); fields=('JobState','ArrayJobId','ArrayTaskId','ExcNodeList','Partition','Nice','TimeLimit','NumCPUs','MinMemoryNode','Reason','Dependency','Requeue','ArrayTaskThrottle');results.append({k:val.get(k) for k in fields})
    # Slurm displays grouped task ranges on admission; scontrol's %3 or throttle covers the entire array.
    if not (val['ExcNodeList']=='scaglione-compute-01' and val['Partition']=='default_partition' and val['Nice']=='0' and val['NumCPUs']=='8' and val['MinMemoryNode']=='48G' and val['TimeLimit']=='20:00:00' and val['ArrayJobId']==job):raise ValueError('effective Slurm setting mismatch: '+str(results[-1]))
q=subprocess.check_output([slurm/'squeue','-rh','-j',job,'-o','%K|%T|%r|%P|%Q'],text=True);receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':job,'array':'0-2%3','registration_sha256':actual,'scontrol':results,'squeue':q.splitlines(),'receipt_scope':'accepted jobs and admission, no scientific outcomes yet'};(root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

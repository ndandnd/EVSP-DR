"""Idempotent per-case preparation -> five independent fleet trials."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
ARMS=['default','focus1','focus2','presparsify1','strong_start']

def command(root,case,mode,dependency=None,arm=None):
    name=f"mstruct_{case}_{arm or 'prep'}"
    c=['sbatch','--parsable','--job-name='+name,'--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem=32G','--time=02:00:00','--requeue','--open-mode=append','--output='+str(root/'slurm'/'%x-%j.out'),'--error='+str(root/'slurm'/'%x-%j.err')]
    if dependency is not None:c+=['--dependency=afterok:'+str(dependency)]
    c+=[str(root/'code'/'worker.sh'),str(root),mode,case]
    if arm is not None:c+=['--arm',arm]
    return c

def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();root=args.root
    root.mkdir(exist_ok=True);(root/'slurm').mkdir(exist_ok=True);(root/'prepared').mkdir(exist_ok=True)
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads((root/'manifest.json').read_text());path=root/'jobs.json';jobs=json.loads(path.read_text()) if path.exists() else {}
        def submit(key,cmd):
            if key in jobs:return jobs[key]['job_id']
            intent=root/(key+'_SUBMIT_INTENT.json')
            if intent.exists():raise RuntimeError('Unresolved submission intent; inspect scheduler before retry: '+str(intent))
            intent.write_text(json.dumps({'argv':cmd,'created_unix':time.time()},indent=2)+'\n')
            r=subprocess.run(cmd,text=True,capture_output=True,check=True);job=r.stdout.strip().split(';')[0]
            if not job.isdigit():raise RuntimeError('Unrecognized job receipt '+r.stdout)
            jobs[key]={'job_id':job,'argv':cmd,'stdout':r.stdout,'stderr':r.stderr,'submitted_unix':time.time()};atomic(path,jobs);return job
        for c in manifest['cases']:
            prep=submit(c['id']+'__prepare',command(root,c['id'],'prepare'))
            for arm in ARMS:submit(c['id']+'__'+arm,command(root,c['id'],'trial',prep,arm))
        print(json.dumps(jobs,indent=2))
if __name__=='__main__':main()

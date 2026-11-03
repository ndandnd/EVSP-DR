"""One guarded array submission, only after explicit recorded approval; dry-run by default."""
import argparse,datetime,fcntl,json,os,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--registration-sha',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
assert re.fullmatch('[0-9a-f]{40}',a.registration_sha)
slurm=Path('/usr/local/slurm/slurm-25.05.5/bin');name='drLH1103'
lock=a.root/'submission.lock'
with lock.open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 path=a.root/'submission_state.json';state=json.loads(path.read_text())
 assert state['authorization']['status']=='approved' and state['authorization']['registration_commit']==a.registration_sha
 assert state['submission']['status']=='not_submitted' and not state['submission']['job_ids']
 argv=[str(slurm/'sbatch'),'--parsable','--job-name='+name,'--export=HOLDOUT_ROOT='+str(a.root),
       '--output='+str(a.root/'logs/%x_%A_%a.out'),'--error='+str(a.root/'logs/%x_%A_%a.err'),str(a.root/'code/holdout.sbatch')]
 if not a.execute:
  print(json.dumps({'dry_run':True,'argv':argv,'tasks':10,'concurrency':10}));raise SystemExit(0)
 queued=subprocess.check_output([str(slurm/'squeue'),'-h','-u',os.environ['USER'],'-n',name,'-o','%i'],text=True).strip()
 prior=subprocess.check_output([str(slurm/'sacct'),'-X','-n','-S','2026-11-02','-u','nc437','--name',name,'--format=JobIDRaw','-P'],text=True).strip()
 assert not queued and not prior,'Existing same-campaign scheduler record; inspect and register it, never duplicate.'
 jid=subprocess.check_output(argv,text=True).strip().split(';')[0];assert jid.isdigit()
 state['submission'].update(status='submitted',job_ids=[jid],submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
 print(json.dumps({'submitted_job_id':jid,'tasks':10,'concurrency':10}))

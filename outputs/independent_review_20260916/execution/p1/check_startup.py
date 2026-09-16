"""Read-only launch audit; accepted jobs are not resubmitted."""
from pathlib import Path
import argparse,datetime,json,subprocess
S='/usr/local/slurm/slurm-25.05.5/bin/'
def main(root):
 b=Path(root); m=json.loads((b/'manifest.json').read_text());jobs=json.loads((b/'jobs.json').read_text());rows=[]
 for j in jobs:
  cid=j['case_id'];c=m['cases'][cid]
  raw=subprocess.check_output([S+'scontrol','show','job',j['job_id'],'-o'],text=True)
  x=dict(t.split('=',1) for t in raw.split() if '=' in t)
  assert x['Partition']=='default_partition' and x['ExcNodeList']=='scaglione-compute-01' and x['NumCPUs']=='8'
  latest=[]
  for p in (b/'cases'/cid/'attempts').glob('*/state.json'):
   state=json.loads(p.read_text());latest.append(dict(path=str(p),status=state['status'],error=state.get('error')))
  rows.append(dict(case_id=cid,job_id=j['job_id'],state=x['JobState'],reason=x.get('Reason'),restart_count=x.get('Restarts'),node=x.get('NodeList'),time_limit=x.get('TimeLimit'),attempts=latest))
 report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verified_exclusion_and_resources=True,rows=rows)
 (b/'startup_verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(jobs=len(rows),states={s:sum(r['state']==s for r in rows) for s in {r['state'] for r in rows}},failed_attempts=[r for r in rows if any(a['error'] for a in r['attempts'])])))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)

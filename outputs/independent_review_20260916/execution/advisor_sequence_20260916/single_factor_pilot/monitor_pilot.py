"""Local read-only snapshot; output contains no full source pool or new jobs."""
import datetime,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
remote='/home/nc437/ladder-lite/advisor_sequence_20260916/single_factor_pilot'
cmd=['ssh','-S','/Users/nadan/.ssh/evsp-unicorn.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','nc437@unicorn-login-01.coecis.cornell.edu','/home/nc437/evsp_env/bin/python '+remote+'/collect.py']
p=subprocess.run(cmd,capture_output=True,text=True,timeout=75)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=ROOT/'snapshots'/stamp;out.mkdir(parents=True,exist_ok=False)
if p.returncode:
 (out/'failure.json').write_text(json.dumps({'returncode':p.returncode,'error':p.stderr},indent=2));raise SystemExit('Unicorn pilot access/collector failed: '+p.stderr)
d=json.loads(p.stdout);(out/'status.json').write_text(json.dumps(d,indent=2)+'\n');(ROOT/'latest_snapshot.txt').write_text(str(out)+'\n')
print(json.dumps({'snapshot':str(out),'jobs':{a:r['job_id'] for a,r in d['jobs'].items()},'conditional_pilots_authorized_to_submit':d['conditional_pilots_authorized_to_submit'],'attempt_states':[(r['arm'],r['execution']['status']) for r in d['attempts']]}))

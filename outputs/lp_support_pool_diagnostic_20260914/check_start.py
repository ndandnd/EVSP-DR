from pathlib import Path
import json,subprocess
import worker as w
B=Path(__file__).resolve().parent
jobs=w.read(B/'jobs.json');rows=[]
for j in jobs:
 raw=subprocess.check_output(['/usr/local/slurm/slurm-25.05.5/bin/scontrol','show','job',j['job_id'],'-o'],text=True,timeout=20)
 d=dict(x.split('=',1) for x in raw.split() if '=' in x)
 logs=list((B/'logs').glob('*_'+j['job_id']+'.out'));txt='\n'.join(p.read_text(errors='replace') for p in logs)
 rows.append(dict(case_id=j['case_id'],job_id=j['job_id'],state=d['JobState'],reason=d.get('Reason'),nodes=d.get('NodeList'),cpus=d['NumCPUs'],memory=d.get('MinMemoryNode'),exclusion=d['ExcNodeList'],dependency=d.get('Dependency'),license_marker_lines=[l for l in txt.splitlines() if 'license' in l.lower() or '2001' in l],attempts=[str(p) for p in (B/'cases'/j['case_id']/'attempts').glob('*')]))
registry=w.read(B.parent/'mip_preemption_study_20260911/registry.json')['cases'];entries=[x for x in registry if str(B)+'/cases/' in (x.get('result_path') or '')]
assert len({(x['job_id'],x['result_path']) for x in entries})==len(entries)
w.save(B/'launch_status.json',dict(utc=w.now(),rows=rows,registry_entries=entries,registry_unique=True,counts={s:sum(r['state']==s for r in rows) for s in {r['state'] for r in rows}}))
print(json.dumps({'counts':{s:sum(r['state']==s for r in rows) for s in {r['state'] for r in rows}},'registered_attempts':len(entries)}))

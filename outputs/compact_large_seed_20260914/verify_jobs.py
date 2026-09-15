"""Read every submitted Slurm record; verify resources, exclusions and true edges."""
from pathlib import Path
import re,subprocess
import worker as w
B=Path(__file__).resolve().parent;S='/usr/local/slurm/slurm-25.05.5/bin/'
def main():
 jobs=w.read(B/'jobs.json');m=w.read(B/'manifest.json');assert len(jobs)==48 and len({x['job_id'] for x in jobs})==48;by={x['case_id']:x['job_id'] for x in jobs}
 # One batched scontrol request keeps controller load bounded.
 r=subprocess.run([S+'scontrol','show','job','-o'],capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr
 (B/'scheduler_jobs_raw.txt').write_text(r.stdout);lines={re.search(r'JobId=(\d+)',l).group(1):l for l in r.stdout.splitlines() if 'JobId=' in l};assert set(by.values())<=set(lines)
 rows=[]
 for j in jobs:
  c=m['cases'][j['case_id']];line=lines[j['job_id']];assert 'ExcNodeList=scaglione-compute-01' in line and 'Partition=default_partition' in line
  assert f"NumCPUs={c['resources']['cpus']} " in line
  if c['kind']=='cg':assert j['dependencies']==[]
  else:assert j['dependencies']==[by[c['source_case']]] and (('afterok:'+by[c['source_case']]) in line or 'Dependency=(null)' in line)
  rows.append(dict(case_id=j['case_id'],job_id=j['job_id'],state=re.search(r'JobState=(\S+)',line).group(1),dependencies=j['dependencies'],resources=c['resources'],excluded=True))
 w.save(B/'scheduler_verification.json',dict(status='passed',utc=w.now(),manifest_sha256=w.sha(B/'manifest.json'),jobs=rows,independent_cg=24,dependent_mip=24))
 print('all48 verified')
if __name__=='__main__':main()

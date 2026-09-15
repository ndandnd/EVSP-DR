#!/usr/bin/env python3
"""Submit one independent allocation for each frozen reserve-screen case."""
from __future__ import annotations
import argparse,datetime as dt,json,subprocess
from pathlib import Path
import campaign
SBATCH='/usr/local/slurm/slurm-25.05.5/bin/sbatch'

def main(root:Path):
 root=root.resolve(); mp=root/'manifest.json'; m=json.loads(mp.read_text()); freeze=json.loads((root/'freeze.json').read_text())
 ms=campaign.sha256_file(mp)
 if freeze['manifest_sha256']!=ms: raise ValueError('manifest freeze mismatch')
 campaign.validate_tooling(m,root)
 native=json.loads((root/'native_validation.json').read_text())
 if native.get('status')!='passed' or native.get('manifest_sha256')!=ms: raise ValueError('native validation does not bind frozen manifest')
 jp=root/'jobs.json'
 if jp.exists(): raise FileExistsError('jobs.json exists; refusing duplicate submission')
 jobs={'schema':'evsp-dr-reserve-feasibility-jobs-v1','campaign_id':m['campaign_id'],'manifest_sha256':ms,
       'native_validation_sha256':campaign.sha256_file(root/'native_validation.json'),'automatic_retry':False,'allocations':[]}
 for case in m['cases']:
  cmd=[SBATCH,'--parsable','--no-requeue','--partition','default_partition','--exclude','scaglione-compute-01',
       '--cpus-per-task','1','--mem','24G','--time','04:15:00','--job-name',('rfs_'+case['case_id'])[:64],
       '--output',str(root/'logs'/f"{case['case_id']}_%j.out"),'--error',str(root/'logs'/f"{case['case_id']}_%j.err"),
       str(root/'worker.sh'),str(case['index'])]
  cp=subprocess.run(cmd,text=True,capture_output=True)
  if cp.returncode: raise RuntimeError(f"sbatch failed for {case['case_id']}: {cp.stderr}")
  jid=cp.stdout.strip().split(';',1)[0]
  if not jid.isdigit(): raise RuntimeError(f"ambiguous sbatch output: {cp.stdout!r}")
  jobs['allocations'].append({'case_id':case['case_id'],'case_index':case['index'],'job_id':jid,
    'phase_tag':'cg_plus_dedicated_mip','cg_wall_s':case['cg_wall_s'],'mip_wall_s':case['mip_wall_s'],
    'slurm_time_s':case['slurm_time_s'],'submitted_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'argv':cmd})
  campaign.atomic_json(jp,jobs); print(case['case_id'],jid,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)

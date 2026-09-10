#!/usr/bin/env python3
"""Submit corrected two-stage MIPs against an existing capacity-speed CG array."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--cg-job-id", required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    code = args.code.resolve()
    manifest = json.loads(args.manifest.read_text())
    tasks = manifest["tasks"]
    if not tasks or len(tasks) > 16:
        raise ValueError("pilot must contain 1..16 tasks")
    submission = root / "submission_twostage.json"
    if submission.exists():
        raise FileExistsError(submission)

    runner = root / "run_twostage_task.py"
    runner.write_text('''import json,os,subprocess,sys
from pathlib import Path
root=Path(''' + repr(str(root)) + ''')
code=Path(''' + repr(str(code)) + ''')
task=json.loads((root/'launch_manifest.json').read_text())['tasks'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
out=root/'results'/task['task_id'];out.mkdir(parents=True,exist_ok=True)
cmd=[sys.executable,'-u','src/run_capacity_speed_event_cg.py','--mode','mip','--arm',task['arm'],'--instance',str(code/task['instance']),'--prices',str(code/task['prices']),'--reference-data-dir',str(code/'data'),'--expected-commit',''' + repr(args.commit) + ''','--require-clean','--out',str(out/'mip_twostage.json'),'--pool',str(out/'pool.jsonl'),'--cg-status',str(out/'cg.json'),'--mip-wall-s','1500','--threads','4']
(out/'mip_twostage_command.json').write_text(json.dumps(cmd,indent=2)+'\\n')
(out/'mip_twostage_allocation.json').write_text(json.dumps({k:v for k,v in os.environ.items() if k.startswith('SLURM_')},indent=2)+'\\n')
subprocess.run(cmd,cwd=code,check=True)
''')
    worker = root / "worker_twostage.sh"
    worker.write_text('''#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
exec /home/nc437/evsp_env/bin/python ''' + shlex.quote(str(runner)) + '''
''')
    worker.chmod(0o750)
    last = len(tasks) - 1
    command = [
        "sbatch", "--parsable", "--job-name=capspd2MIP",
        "--partition=scaglione", "--exclude=scaglione-compute-01",
        f"--dependency=aftercorr:{args.cg_job_id}", f"--array=0-{last}%2",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=16G",
        "--time=00:30:00", "--no-requeue", "--kill-on-invalid-dep=yes",
        f"--output={root}/logs/mip2_%A_%a.out",
        f"--error={root}/logs/mip2_%A_%a.err", str(worker),
    ]
    result = subprocess.run(
        ["bash", "-lc", shlex.join(command)],
        check=True, capture_output=True, text=True,
    )
    record = {
        "schema": "evsp-dr-capacity-speed-twostage-submission-v1",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": args.commit,
        "reused_cg_job_id": args.cg_job_id,
        "mip_job_id": result.stdout.strip().split(";")[0],
        "command": command,
        "runner_sha256": digest(runner),
        "worker_sha256": digest(worker),
        "task_count": len(tasks),
        "mip_policy": {
            "total_seconds": 1500,
            "stage1_max_seconds": 750,
            "stage2_seconds": "remaining",
            "stage1_objective": "fleet",
            "stage2_fleet_cap": "<= best validated stage1 incumbent",
            "stage2_objective": "electricity plus 5 per charge start"
        }
    }
    submission.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Submit immutable matched capacity/speed CG and dependent MIP arrays."""

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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    code = root / "code"
    manifest = json.loads(args.manifest.read_text())
    tasks = manifest["tasks"]
    if not tasks or len(tasks) > 16:
        raise ValueError("pilot must contain 1..16 tasks")
    if (root / "submission.json").exists():
        raise FileExistsError(root / "submission.json")
    (root / "results").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")

    runner = root / "run_task.py"
    runner.write_text('''import json,os,subprocess,sys
from pathlib import Path
root=Path(''' + repr(str(root)) + ''')
code=root/'code'
mode=sys.argv[1]
task=json.loads((root/'launch.json').read_text())['tasks'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
out=root/'results'/task['task_id'];out.mkdir(parents=True,exist_ok=True)
common=[sys.executable,'-u','src/run_capacity_speed_event_cg.py','--arm',task['arm'],'--instance',str(code/task['instance']),'--prices',str(code/task['prices']),'--reference-data-dir',str(code/'data'),'--expected-commit',task['commit'],'--require-clean']
if mode=='cg':
    cmd=common+['--mode','cg','--out',str(out/'cg.json'),'--pool-out',str(out/'pool.jsonl'),'--cg-wall-s','5100','--max-iters','10000','--threads','1']
else:
    cmd=common+['--mode','mip','--out',str(out/'mip.json'),'--pool',str(out/'pool.jsonl'),'--cg-status',str(out/'cg.json'),'--mip-wall-s','1500','--threads','4']
(out/(mode+'_command.json')).write_text(json.dumps(cmd,indent=2)+'\\n')
(out/(mode+'_allocation.json')).write_text(json.dumps({k:v for k,v in os.environ.items() if k.startswith('SLURM_')},indent=2)+'\\n')
subprocess.run(cmd,cwd=code,check=True)
''')
    worker = root / "worker.sh"
    worker.write_text('''#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
exec /home/nc437/evsp_env/bin/python ''' + shlex.quote(str(runner)) + ''' "$1"
''')
    worker.chmod(0o750)
    last = len(tasks) - 1
    cg_cmd = [
        "sbatch", "--parsable", "--job-name=capspdCG",
        "--partition=default_partition", f"--array=0-{last}%2",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=24G",
        "--time=01:30:00", "--no-requeue",
        f"--output={root}/logs/cg_%A_%a.out",
        f"--error={root}/logs/cg_%A_%a.err", str(worker), "cg",
    ]
    cg = subprocess.run(
        ["bash", "-lc", shlex.join(cg_cmd)],
        check=True, capture_output=True, text=True,
    )
    cg_id = cg.stdout.strip().split(";")[0]
    mip_cmd = [
        "sbatch", "--parsable", "--job-name=capspdMIP",
        "--partition=scaglione", "--exclude=scaglione-compute-01",
        f"--dependency=aftercorr:{cg_id}", f"--array=0-{last}%2",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=16G",
        "--time=00:30:00", "--no-requeue", "--kill-on-invalid-dep=yes",
        f"--output={root}/logs/mip_%A_%a.out",
        f"--error={root}/logs/mip_%A_%a.err", str(worker), "mip",
    ]
    mip = subprocess.run(
        ["bash", "-lc", shlex.join(mip_cmd)],
        check=True, capture_output=True, text=True,
    )
    record = {
        "schema": "evsp-dr-capacity-speed-pilot-submission-v1",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": args.commit,
        "cg_job_id": cg_id,
        "mip_job_id": mip.stdout.strip().split(";")[0],
        "cg_command": cg_cmd,
        "mip_command": mip_cmd,
        "runner_sha256": digest(runner),
        "worker_sha256": digest(worker),
        "manifest_sha256": digest(root / "launch.json"),
        "task_count": len(tasks),
    }
    (root / "submission.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

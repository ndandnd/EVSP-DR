from pathlib import Path
import subprocess,shlex
p=Path(__file__).resolve().parent;cmd='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/nc437/evsp_env/bin/python -c '+shlex.quote((p/'strict_arrival_replay_remote.py').read_text())
with (p/'strict_arrival_replay_results.json').open('w') as o,(p/'strict_arrival_progress.log').open('w') as e:
 subprocess.run(['ssh','-S','/Users/nadan/.ssh/evsp-unicorn.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','nc437@unicorn-login-01.coecis.cornell.edu',cmd],input=(p/'requests.json').read_text(),text=True,stdout=o,stderr=e,check=True)

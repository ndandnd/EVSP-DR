"""Submit one already-authorized, source-pinned guard recovery; no treatments."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import subprocess

root = Path('/home/nc437/ladder-lite/spatial_tariff_expansion_20260925')
work = root / 'recovery_20260926_case11'
receipt = work / 'submission.json'
slurm = '/usr/local/slurm/slurm-25.05.5/bin/'
lock = open(work / 'submission.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if receipt.exists():
    raise SystemExit('Existing submission receipt: refusing duplicate')
queue = subprocess.check_output([slurm+'squeue', '-u', 'nc437', '-h', '-r', '-o', '%i|%j|%T'], text=True)
for line in queue.splitlines():
    if 'mix1_two_price_split' in line:
        raise SystemExit('Case already in queue: ' + line)
for attempt in (root/'results/mix1_two_price_split/cleanup').glob('job*_r*/attempt.json'):
    if json.loads(attempt.read_text()).get('state') == 'finished':
        raise SystemExit('Existing completed cleanup: ' + str(attempt))
command = [slurm+'sbatch', '--parsable', '--dependency=afterany:498917_43:498918_43', str(work/'recovery.sbatch')]
job = subprocess.check_output(command, text=True).strip()
data = {'submitted_utc': datetime.now(timezone.utc).isoformat(), 'job_id': job,
        'command': command, 'root': str(root), 'cell': 'mix1_two_price_split',
        'replaces_failed_post': '498962', 'new_treatment': False,
        'scope': 'same exhaustive duplicate removal; pinned guard10 to11 only',
        'resources': {'partition':'default_partition','cpus':8,'mem':'48G','time':'05:00:00',
                      'exclude':'scaglione-compute-01','requeue':True},
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in work.iterdir() if p.is_file() and p.suffix in ('.py','.sbatch')}}
temp = receipt.with_suffix('.tmp')
temp.write_text(json.dumps(data, indent=2)+'\n')
temp.replace(receipt)
print(json.dumps(data, indent=2))

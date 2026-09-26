"""Read-only overnight endpoint collection; run on Unicorn with Python stdin."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import subprocess

BASE = Path('/home/nc437/ladder-lite')
R2 = BASE / 'spatial_tariff_expansion_20260925'
R1 = BASE / 'spatial_tariff_k5_20260925'

def record(path):
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'data': json.loads(raw)}

out = {'collected_utc': datetime.now(timezone.utc).isoformat(),
       'root2': str(R2), 'post_records': [], 'summaries': [], 'attempts': [],
       'original_recovery': []}
for path in sorted((R2 / 'post_records').glob('*/*.json')):
    out['post_records'].append(record(path))
for pattern, key in [('*/ */*/attempt.json'.replace(' ', ''), 'attempts'),
                     ('*/*/*/out/summary.json', 'summaries')]:
    for path in sorted((R2 / 'results').glob(pattern)):
        out[key].append(record(path))
for job in range(506710, 506715):
    for path in sorted((R1 / 'results').glob(f'*/*/job{job}_r*')):
        for suffix in ('attempt.json', 'out/summary.json', 'summary.json'):
            if (path / suffix).exists():
                out['original_recovery'].append(record(path / suffix))

selected = R2 / 'results/mix1_two_price_split/fresh_cg/job499021_r0/out/selected_routes.json'
raw = selected.read_bytes()
routes = json.loads(raw)
if isinstance(routes, dict):
    routes = routes.get('routes', routes.get('selected_routes'))
counts = Counter(t for route in routes for t in route['trips'])
dup = [[t for t in route['trips'] if counts[t] > 1] for route in routes]
out['failed_cleanup_diagnosis'] = {
    'selection': str(selected), 'sha256': hashlib.sha256(raw).hexdigest(),
    'route_trip_counts': [len(r['trips']) for r in routes],
    'duplicated_trip_counts_by_route': [len(d) for d in dup],
    'duplicated_trip_ids_by_route': dup,
    'duplicated_distinct_trips': sum(n > 1 for n in counts.values()),
    'duplicate_excess': sum(max(0, n-1) for n in counts.values()),
    'subsequence_upper_count': sum(2**len(d) for d in dup),
}
slurm = '/usr/local/slurm/slurm-25.05.5/bin/'
cmd = [slurm + 'sacct', '-u', 'nc437', '-S', '2026-09-25T18:00', '-X', '-n', '-P',
       '-o', 'JobID%24,JobName%60,State,ExitCode,Elapsed,Start,End,Restarts']
p = subprocess.run(cmd, text=True, capture_output=True)
out['sacct_command'] = cmd
out['sacct_returncode'] = p.returncode
out['sacct_stderr'] = p.stderr
out['sacct'] = [line for line in p.stdout.splitlines()
                if len(line.split('|')) > 1 and line.split('|')[1].startswith(('sx_', 'st_'))]
print(json.dumps(out, indent=2))

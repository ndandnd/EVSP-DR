"""Read-only final scoped scheduler check and pinned recovery receipts."""
from pathlib import Path
from datetime import datetime, timezone
import subprocess, hashlib, json

root = Path('/home/nc437/ladder-lite/spatial_tariff_expansion_20260925')
slurm = '/usr/local/slurm/slurm-25.05.5/bin/'
def run(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return {'command':args, 'returncode':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr}
def record(p):
    raw = p.read_bytes()
    return {'path':str(p), 'sha256':hashlib.sha256(raw).hexdigest(), 'data':json.loads(raw)}

out = {'collected_utc':datetime.now(timezone.utc).isoformat(), 'root':str(root)}
out['queue'] = run([slurm+'squeue','-u','nc437','-h','-r','-o','%i|%j|%T|%M|%R|%E'])
out['queue']['stdout'] = '\n'.join(s for s in out['queue']['stdout'].splitlines()
    if len(s.split('|'))>1 and s.split('|')[1].startswith(('sx_','st_')))
out['accounting'] = run([slurm+'sacct','-u','nc437','-S','2026-09-25T18:00','-X','-n','-P',
    '-o','JobID%24,JobName%60,State,ExitCode,Elapsed,Start,End,Restarts'])
out['accounting']['stdout'] = '\n'.join(s for s in out['accounting']['stdout'].splitlines()
    if len(s.split('|'))>1 and s.split('|')[1].startswith(('sx_','st_')))
out['recovery_resources'] = run([slurm+'sacct','-j','520378','-n','-P',
    '-o','JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocCPUS'])
out['root_files'] = [{'name':p.name,'mtime':p.stat().st_mtime} for p in root.iterdir() if p.is_file()]
out['submission'] = record(root/'recovery_20260926_case11/submission.json')
out['attempts'] = []
for p in sorted((root/'results/mix1_two_price_split/cleanup').glob('job520378_r*')):
    out['attempts'].append({'dir':str(p), 'files': [record(p/s) for s in
        ('attempt.json','recovery_design.json','out/summary.json','out/selected_routes.json') if (p/s).is_file()]})
out['original_post'] = record(root/'post_records/mix1_two_price_split/job498962_r0.json')
out['source_summary'] = record(root/'results/mix1_two_price_split/fresh_cg/job499021_r0/out/summary.json')
print(json.dumps(out,indent=2))

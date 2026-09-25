"""Read-only Unicorn snapshot; stdout is JSON. No scheduler mutations/solver runs."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, subprocess

ROOT = Path('/home/nc437/ladder-lite')
SLURM = '/usr/local/slurm/slurm-25.05.5/bin/'

def run(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=90)
    return {'argv': args, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def lean(v):
    if isinstance(v, dict):
        return {k: lean(x) for k, x in v.items() if not isinstance(x, list)}
    return v

def artifact(p, full=False):
    if not p.exists():
        return {'path': str(p), 'exists': False}
    data = p.read_bytes()
    out = {'path': str(p), 'resolved_path': str(p.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'mtime_utc': datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()}
    if p.suffix == '.json':
        v = json.loads(data)
        out['content'] = v if full else lean(v)
        out['content_scope'] = 'full JSON' if full else 'JSON with list-valued fields omitted recursively; full-source SHA256 retained'
    else:
        out['text'] = data.decode()
    return out

out = {'schema': 'evsp-dr-cluster-snapshot-20260925-v1', 'captured_utc': datetime.now(timezone.utc).isoformat()}
out['policy'] = artifact(ROOT/'SCAGLIONE_RESOURCE_POLICY.md')
out['queue'] = run([SLURM+'squeue', '-u', 'nc437', '-h', '-o', '%i|%j|%P|%T|%M|%l|%R|%E'])
out['scontrol_all_user'] = run([SLURM+'scontrol', 'show', 'job', '-o'])
# Limit saved scontrol records to nc437 without exposing other users' job data.
out['scontrol_all_user']['stdout'] = '\n'.join(x for x in out['scontrol_all_user']['stdout'].splitlines() if 'UserId=nc437(' in x)
full = ROOT/'chain_extension_33_40_20260921'
out['full40_manifest'] = artifact(full/'manifest.json', True)
out['full40_case_jobs'] = artifact(full/'case_jobs.json', True)
jobs = json.loads((full/'case_jobs.json').read_text())
out['full40_artifacts'] = {}
for cid in sorted(jobs):
    out['full40_artifacts'][cid] = {name: artifact(full/'cases'/cid/name) for name in ['cg.json', 'cg_provenance.json', 'mip_result.json']}
out['full40_sacct'] = run([SLURM+'sacct', '-u', 'nc437', '-S', '2026-09-21', '-n', '-P', '-j', ','.join(str(jobs[x][mode]) for x in jobs for mode in ['cg', 'mip']), '--format=JobID,JobName,State,ExitCode,Elapsed,Start,End,ReqMem,MaxRSS,NodeList'])
dive = ROOT/'dive_cap_pilot_20260923'
out['dive_manifest'] = artifact(dive/'manifest.json', True)
out['dive_artifacts'] = []
for p in sorted((dive/'results').rglob('*.json')):
    if p.name in ['execution.json', 'dive_argv.json', 'summary.json', 'result.json', 'worker_result.json', 'handoff.json']:
        out['dive_artifacts'].append(artifact(p))
out['dive_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size} for p in sorted((dive/'results').rglob('*')) if p.is_file() and p.suffix in ['.json', '.log', '.out', '.err']]
out['dive_sacct'] = run([SLURM+'sacct', '-u', 'nc437', '-S', '2026-09-23', '-n', '-P', '-j', '889582,889583,889584,889585', '--format=JobID,JobName,State,ExitCode,Elapsed,Start,End,ReqMem,MaxRSS,NodeList'])
spatial = ROOT/'spatial_tariff_k5_20260925'
out['spatial_manifest'] = artifact(spatial/'manifest.json', True)
out['spatial_jobs'] = artifact(spatial/'jobs.tsv')
out['spatial_collected'] = [artifact(p, True) for p in sorted((spatial/'collected').glob('*')) if p.suffix in ['.json', '.csv']]
out['spatial_attempts'] = [artifact(p) for p in sorted((spatial/'results').rglob('attempt.json'))]
out['spatial_summaries'] = [artifact(p) for p in sorted((spatial/'results').rglob('summary.json'))]
sj = [row['job_id'] for row in csv.DictReader((spatial/'jobs.tsv').open(), delimiter='\t')]
sj += ['476611', '476612', '476622', '476623']
out['spatial_sacct'] = run([SLURM+'sacct', '-u', 'nc437', '-S', '2026-09-25', '-n', '-P', '-j', ','.join(sj), '--format=JobID,JobName,State,ExitCode,Elapsed,Start,End,ReqMem,MaxRSS,NodeList'])
out['finished_utc'] = datetime.now(timezone.utc).isoformat()
print(json.dumps(out, indent=2))

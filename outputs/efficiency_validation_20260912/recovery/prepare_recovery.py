"""Create a new four-case manifest after the recorded cache-only CLI failure."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--original-root', type=Path, required=True)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--code', type=Path, required=True)
p.add_argument('--commit', required=True)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('campaign', a.code / 'scripts/efficiency_validation_20260912/campaign.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
assert not (a.root / 'manifest.json').exists(), 'Do not overwrite an existing manifest'
pin = c.code_check(a.code, a.commit)
original = c.read(a.original_root / 'manifest.json')
jobs = c.read(a.original_root / 'jobs.json')
job_ids = {j['case']: j['job_id'] for j in jobs['jobs']}
manifest = copy.deepcopy(original)
manifest['prepared_utc'] = c.now()
manifest['root'] = str(a.root)
manifest['baseline_commit'] = pin
manifest['cases'] = [v for v in manifest['cases'] if v['kind'] == 'warm']
assert len(manifest['cases']) == 4
old_code = str(a.original_root / 'code-baseline')
for case in manifest['cases']:
    failed = a.original_root / 'cases' / case['id'] / (job_ids[case['id']] + '_r0')
    assert c.read(failed / 'pair_status.json')['status'] == 'preparation_failed'
    error = (failed / 'prepare/stderr.log').read_text()
    assert '--event-network-cache-only does not use --out' in error
    assert not (failed / 'reference').exists() and not (failed / 'optimized').exists()
    case['code'] = str(a.code)
    case['commit'] = pin
    case['supersedes_startup_attempt'] = str(failed)
    manifest['frozen'].append(c.authenticate(failed / 'pair_status.json', c.digest(failed / 'pair_status.json')))
for entry in manifest['frozen']:
    # Authenticate references in the new code checkout; frozen inputs stay untouched.
    if entry['path'].startswith(old_code + '/'):
        entry['path'] = str(a.code) + entry['path'][len(old_code):]
    c.authenticate(entry['path'], entry['sha256'])
manifest['frozen'].append(c.authenticate(a.original_root / 'manifest.json', c.digest(a.original_root / 'manifest.json')))
manifest['recovery'] = {
    'original_manifest': str(a.original_root / 'manifest.json'),
    'original_manifest_sha256': c.digest(a.original_root / 'manifest.json'),
    'reason': 'Cache-only preparation rejected --out before graph construction. Launcher fixed; solver code unchanged.',
    'failed_jobs': {v['id']: job_ids[v['id']] for v in manifest['cases']},
    'preserved': 'Original manifest, jobs, failed artifacts, frozen inputs, parent journals and five active pairs remain untouched.',
    'differences': ['new root', 'new baseline execution commit and checkout', 'four warm cases only', 'cache-only command omits --out'],
}
manifest['policy']['independent_cases'] = 4
manifest['policy']['concurrency'] = 4
manifest['resources_reason'] += ' Recovery contains only the four warm allocations whose preparation failed before solving; resources, treatment order and physics are unchanged.'
c.write(a.root / 'manifest.json', manifest)
print(json.dumps({'manifest': str(a.root / 'manifest.json'), 'sha256': c.digest(a.root / 'manifest.json'), 'cases': 4}))

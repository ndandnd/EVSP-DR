"""Read-only verification of staged inputs, frozen duty prefixes and launch collisions."""
import csv
import datetime
from pathlib import Path
import campaign as c


def rows(path):
    with Path(path).open() as stream:
        reader = csv.DictReader(stream); result = list(reader)
    indexed = {r['Ordered_Trip_ID']: r for r in result}
    assert len(indexed) == len(result), (path, 'duplicate stable trip ID')
    return indexed


v = c.read(c.B/'manifest.json')
frozen = c.read(c.B/'inputs/manifest.json')
parent_manifest = c.read(c.PARENT/'manifest.json')
assert v['scientific_settings'] == parent_manifest['scientific_settings']
assert v['execution_commit'] == parent_manifest['execution_commit'] == c.COMMIT
assert v['mip_execution_commit'] == parent_manifest['mip_execution_commit'] == c.MIP_COMMIT
for mode in ['cg', 'mip']: assert v['resources'][mode] == parent_manifest['resources'][mode]
assert all(case['cg_seconds'] == 14400 for case in v['cases'].values())
original = c.PARENT.parent/'chain_extension_20260913/inputs'
assert c.sha(original/'manifest.json') == v['input_manifest_sha256']
assert c.sha(c.PARENT/'inputs/manifest.json') == v['input_manifest_sha256']
assert len(frozen['cases']) == 150
checks = []
for chain in range(1, 7):
    previous_id = f'w{chain}_k28'
    p = v['initial_parents'][str(chain)]
    previous = rows(c.PARENT/'code/data'/p['csv'])
    assert c.sha(original/frozen['cases'][previous_id]['csv']) == p['csv_sha256']
    assert c.sha(c.PARENT/'inputs'/frozen['cases'][previous_id]['csv']) == p['csv_sha256']
    assert c.sha(c.B/'code/data'/p['csv']) == p['csv_sha256']
    for k in [29, 30]:
        cid = f'w{chain}_k{k}'; spec = frozen['cases'][cid]; case = v['cases'][cid]
        staged = c.B/'inputs'/spec['csv']
        assert c.sha(staged) == c.sha(original/spec['csv']) == spec['input_sha256']
        assert c.sha(c.B/'code/data'/case['csv']) == spec['input_sha256']
        assert spec['previous_case'] == previous_id
        assert spec['previous_input_sha256'] == frozen['cases'][previous_id]['input_sha256']
        old_spec = frozen['cases'][previous_id]
        assert spec['duties'][:-1] == old_spec['duties']
        assert spec['duties'][-1] == spec['added_duty'] == frozen['chains'][str(chain)]['added_duty_order'][k-16]
        assert len({''.join(x for x in d if x.isdigit()) for d in spec['duties']}) == k
        current = rows(staged)
        assert len(current) == spec['trip_count']
        assert set(previous) <= set(current)
        for trip_id, old_row in previous.items():
            assert {k: value for k, value in old_row.items() if k != 'count_trip_id'} == {
                k: value for k, value in current[trip_id].items() if k != 'count_trip_id'}
        assert len(current)-len(previous) == spec['added_trip_count']
        checks.append({'case_id': cid, 'input_sha256': c.sha(staged),
            'original_csv': str(original/spec['csv']), 'previous_case': previous_id,
            'previous_input_sha256': spec['previous_input_sha256'], 'trips': len(current),
            'added_trips': len(current)-len(previous), 'added_duty': spec['added_duty'],
            'stable_nesting': True, 'previous_attributes_unchanged_except_count_trip_id': True})
        previous_id = cid; previous = current

scanned = []; collisions = []
for path in sorted(c.B.parent.glob('*/case_jobs.json')):
    if path.parent == c.B: continue
    mapping = c.read(path); overlap = set(mapping) & set(v['cases'])
    scanned.append({'path': str(path), 'sha256': c.sha(path), 'overlapping_case_ids': sorted(overlap)})
    if not overlap: continue
    manifest_path = path.parent/'manifest.json'
    if not manifest_path.is_file():
        collisions.append({'path': str(path), 'reason': 'unresolved overlapping case IDs'}); continue
    other = c.read(manifest_path)
    for cid in overlap:
        if other.get('cases', {}).get(cid, {}).get('input_sha256') == v['cases'][cid]['input_sha256']:
            collisions.append({'path': str(path), 'case_id': cid, 'jobs': mapping[cid]})
assert not collisions, collisions
c.save(c.B/'input_validation.json', {'status': 'passed',
    'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'manifest_sha256': c.sha(c.B/'manifest.json'), 'input_manifest_sha256': v['input_manifest_sha256'],
    'frozen_original_manifest_sha256': c.sha(original/'manifest.json'), 'random_redraw': False,
    'source_generator_manifest_case_count': len(frozen['cases']),
    'staged_case_count': len(checks), 'cases': checks,
    'scientific_settings_equal_parent': True, 'execution_commits_equal_parent': True,
    'cg_mip_requests_equal_parent': True,
    'equivalent_submitted_case_collisions': collisions, 'collision_scan': scanned})
print('PASS: 12 frozen inputs, six k28 parents, stable trip nesting and launch-collision checks')

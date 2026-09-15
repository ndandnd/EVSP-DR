"""Validate newly published overnight endpoints and export compact source tables."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    source = json.loads(raw)
    out = Path(__file__).parent / ('status_' + args.snapshot.stem)
    out.mkdir(exist_ok=True)
    register = json.loads((Path(__file__).parents[1] / 'research_register/register.json').read_text())
    assert register['source_snapshot']['sha256'] == hashlib.sha256(raw).hexdigest()
    normalized = {(r['campaign_id'], r['stage'], r['source_path']): r for r in register['rows']}
    checks, reserves, gaps = [], [], []
    name = 'reserve_feasibility_screen_20260914'
    campaign = source['campaigns'][name]
    records = {(r['case_id'], r['phase']): r for r in campaign['records']}
    for case in campaign['rows']:
        cid = case['case_id']
        cg, mip = records.get((cid, 'cg')), records.get((cid, 'mip'))
        if not cg or not mip:
            continue
        for item in [cg, mip]:
            row = normalized[name, item['phase'], item['path']]
            assert item['stage_completion_verified'] and item['artifact_complete']
            assert row['source_sha256'] == item['sha256']
            checks.append(dict(campaign=name, case=cid, stage=item['phase'], path=item['path'], sha256=item['sha256']))
        c, m = cg['result'], mip['result']
        assert m['cg_status_sha256'] == cg['sha256']
        assert c['pool_sha256'] == m['pool_sha256']
        assert c['provenance']['instance_sha256'] == m['provenance']['instance_sha256']
        result = m['result']
        row = normalized[name, 'mip', mip['path']]
        assert row['mip_incumbent_fleet'] == result['fleet']
        assert row['fleet_proven'] == result['stage1']['fleet_proven']
        assert row['cross_route_capacity_validated'] == m['physical_station_capacity_audit']['valid']
        assert row['capacity_enforced'] == m['capacity_enforced_in_mip']
        reserves.append(dict(case_id=cid, duty=cid.split('_')[1], arm=m['arm'],
            cg_minutes=c['runtime_s']/60, cg_certified=c['certified_rc_optimal'],
            weighted_lp_objective=c['final']['objective'], fractional_route_weight=c['final']['route_weight'],
            buses=result['fleet'], pool_fleet_proved=result['stage1']['fleet_proven'],
            charging_related_cost=result['charging_related_cost'], capacity_enforced=m['capacity_enforced_in_mip'],
            station_count_audit=m['physical_station_capacity_audit']['valid'],
            duplicate_assignments=m['duplicate_service_audit']['extra_trip_assignments'],
            cg_path=cg['path'], cg_sha256=cg['sha256'], mip_path=mip['path'], mip_sha256=mip['sha256']))
    name = 'remaining_chain_gaps_20260914'
    campaign = source['campaigns'][name]
    manifest = campaign['workflow']['manifest.json']
    manifest_sha = hashlib.sha256((Path(__file__).parents[1] / name / 'manifest.json').read_bytes()).hexdigest()
    bindings = {r['case']: r for r in json.loads((Path(__file__).parents[1] / name / 'original_result_bindings.json').read_text())}
    original = {Path(r['path']).parts[Path(r['path']).parts.index('cases')+1]: r
                for r in source['campaigns']['chain_extension_20260913']['mip']}
    for item in campaign['mip']:
        cid = Path(item['path']).parent.name
        case = manifest['cases'][cid]
        prior = original[case['original_case']]
        assert item['completion_marker_matches']
        assert item['publication_manifest_sha256'] == manifest_sha
        assert item['source_status_sha256'] == case['source_status_sha256']
        assert item['source_journal_sha256'] == case['source_journal_sha256']
        binding = bindings[case['original_case']]
        assert binding['canonical_hash_matches'] and binding['attempt_hash_matches'] and binding['semantic_equal']
        assert prior['sha256'] == binding['attempt_expected']
        assert case['comparator']['prior_sha256'] == binding['expected']
        assert item['pool_columns'] == prior['pool_columns']
        assert item['mip_provenance']['git_commit'] == case['execution_commit']
        row = normalized[name, 'mip', item['path']]
        assert row['source_sha256'] == item['sha256']
        assert row['mip_incumbent_fleet'] == item['buses']
        two = item['two_stage']
        gaps.append(dict(case_id=case['original_case'], target=case['target_k'],
            original_buses=prior['buses'], rerun_buses=item['buses'], pool_fleet_bound=item['fleet_bound'],
            fleet_proved=item['fleet_proven'], stage1_minutes=two['stage1_runtime_s']/60,
            total_mip_minutes=item['runtime_s']/60, stage2_status=two['stage2_status_name'],
            individual_route_replay=item['physical_replay_validated'],
            duplicate_removal_validated=item['duplicate_trip_removal_validated'],
            shared_capacity_validated=item['cross_route_charger_capacity_validated'],
            original_sha256=prior['sha256'], source_status_sha256=item['source_status_sha256'],
            source_journal_sha256=item['source_journal_sha256'], path=item['path'], sha256=item['sha256']))
        checks.append(dict(campaign=name, case=cid, stage='mip', path=item['path'], sha256=item['sha256']))
    pricing = []
    for name in ['capacity_fixed_dual_20260914', 'capacity_fixed_dual_retry_20260914']:
        pricing.extend(dict(item, source_campaign=name)
                       for item in source['campaigns'][name]['pricing_calls'])
    for item in pricing:
        assert item['actual_pricing_raw_dual_vector_sha256'] == item['starting_raw_dual_vector_sha256']
        assert item['pricing_call']['raw_dual_vector_sha256'] == item['starting_raw_dual_vector_sha256']
        checks.append(dict(campaign=item['source_campaign'], case=item['case_id'],
                           raw_dual_sha256=item['starting_raw_dual_vector_sha256']))
    pairs = collections.defaultdict(dict)
    for item in pricing:
        pairs[item['pair_id']][item['selector']] = item
    pair_rows = []
    for pair_id, arms in sorted(pairs.items()):
        if set(arms) != {'reference', 'prefix-memo'}:
            continue
        reference, cached = arms['reference'], arms['prefix-memo']
        assert reference['source_pool_sha256'] == cached['source_pool_sha256']
        assert reference['starting_raw_dual_vector_sha256'] == cached['starting_raw_dual_vector_sha256']
        for key in ['rows', 'columns', 'nonzeros', 'objective', 'route_weight']:
            assert reference['rmp'][key] == cached['rmp'][key]
        a, b = reference['pricing_call'], cached['pricing_call']
        pair_rows.append(dict(pair_id=pair_id, reference_seconds=a['elapsed_s'],
            reference_status=a['status'], reference_min_reduced_cost=a.get('min_reduced_cost'),
            cached_seconds=b['elapsed_s'], cached_status=b['status'], cached_stop_reason=b.get('stop_reason'),
            cached_min_reduced_cost=b.get('min_reduced_cost'), raw_dual_sha256=a['raw_dual_vector_sha256'],
            pool_sha256=reference['source_pool_sha256'], rmp_rows=reference['rmp']['rows'],
            rmp_columns=reference['rmp']['columns'], rmp_nonzeros=reference['rmp']['nonzeros']))
    write_csv(out/'pricing_comparison.csv', pair_rows)
    write_csv(out/'reserve_results.csv', reserves)
    write_csv(out/'remaining_gap_results.csv', gaps)
    (out/'pricing_calls.json').write_text(json.dumps(pricing, indent=2)+'\n')
    counts = collections.Counter(line.split('|')[1] for line in source['squeue']['stdout'].splitlines()
                                 if 'JobHeldUser' not in line)
    result = dict(snapshot_sha256=hashlib.sha256(raw).hexdigest(), checks=checks,
                  reserve_completed=len(reserves), gap_reruns_completed=len(gaps), queue=dict(counts))
    (out/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__ == '__main__':
    main()

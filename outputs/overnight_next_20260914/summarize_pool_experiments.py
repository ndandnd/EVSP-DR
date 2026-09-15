"""Verify published large-start and LP-column-addition endpoints and write tables."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def case_id(item):
    parts = Path(item['path']).parts
    return parts[parts.index('cases') + 1]


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_mip(item, case, manifest_hash, source_hash, journal_hash):
    assert item['completion_marker_matches'] is True
    assert item['authority'] == 'published'
    assert item['publication_manifest_sha256'] == manifest_hash
    assert item['source_status_sha256'] == source_hash
    assert item['source_journal_sha256'] == journal_hash
    provenance = item['mip_provenance']
    assert provenance['git_commit'] == case['execution_commit']
    assert provenance['final_observed_git_commit'] == case['execution_commit']
    assert provenance['tracked_clean_at_end'] is True
    assert provenance['arguments']['timelimit'] == case['solver_budget_s']
    assert provenance['arguments']['cover'] is True
    assert provenance['arguments']['two_stage'] is True
    assert item['two_stage']['stage1_time_limit_s'] == case['stage1_budget_s']
    audit = item['physical_pool_audit']
    assert audit['input_hashes']['instance_sha256'] == case['input_sha256']
    assert audit['rejected_columns'] == 0
    assert audit['deterministically_repaired'] == 0
    assert audit['added_giro_route_count'] == 0
    for key in ['prices_sha256', 'reference_sha256', 'deadhead_sha256']:
        assert audit['input_hashes'][key] in case['static_hashes'].values()
    # Excluding the target does not require proving the exact integer optimum.
    # Keep a conservative margin so floating-point 20.00000000000008 does not
    # exclude a target of 20. This is a fleet-search bound, not a weighted LP.
    bound = item.get('fleet_bound')
    target_excluded = (float(bound) > case['target_k'] + 1e-5
                       if bound is not None and math.isfinite(float(bound)) else None)
    return dict(buses=item['buses'], pool_fleet_bound=item['fleet_bound'],
        target_excluded_in_saved_pool=target_excluded,
        fleet_proved=item['fleet_proven'], pool_columns=item['pool_columns'],
        total_mip_minutes=item['runtime_s']/60,
        fleet_search_minutes=item['two_stage']['stage1_runtime_s']/60,
        charging_status=item['two_stage']['stage2_status_name'],
        individual_route_replay=item['physical_replay_validated'],
        duplicate_removal_validated=item['duplicate_trip_removal_validated'],
        shared_capacity_validated=item['cross_route_charger_capacity_validated'],
        mip_path=item['path'], mip_sha256=item['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    source = json.loads(raw)
    out = args.out_dir or Path(__file__).parent / ('status_' + args.snapshot.stem)
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    large = source['campaigns']['compact_large_seed_20260914']
    cases = large['workflow']['manifest.json']['cases']
    manifest_hash = large['workflow']['validation.json']['manifest_sha256']
    cg_by_case = {case_id(item): item for item in large['cg']}
    mip_by_case = {case_id(item): item for item in large['mip']}
    rows = []
    for cid, case in cases.items():
        if case['kind'] != 'cg':
            continue
        row = dict(case_id=cid, chain=case['chain'], target=case['target_k'],
            treatment=case['treatment'],
            starting_sequences=case['seed']['selected_sequence_count'])
        if cid in cg_by_case:
            item = cg_by_case[cid]
            assert item['authority'] == 'published'
            assert item['completion_marker_matches'] is True
            assert item['publication_manifest_sha256'] == manifest_hash
            provenance = item['provenance']
            assert provenance['git_commit'] == case['execution_commit']
            assert provenance['instance_sha256'] == case['input_sha256']
            for key in ['prices_sha256', 'reference_sha256', 'deadhead_sha256']:
                assert provenance[key] in case['static_hashes'].values()
            assert provenance['args']['wall_limit_s'] == case['solver_budget_s']
            audit = item['inherited_event_pool_audit']
            assert audit['source_status_sha256'] == case['seed_status_sha256']
            assert audit['source_journal_sha256'] == case['seed_journal_sha256']
            assert audit['selected_for_replay'] == case['seed']['selected_sequence_count']
            assert audit['unprocessed_selected'] == audit['rejected_columns'] == 0
            assert audit['inherited_lp_certificate'] is False
            row.update(cg_minutes=item['wall_s']/60, cg_certificate=item['certified_rc_optimal'],
                cg_stop_reason=item['stop_reason'], weighted_lp_objective=item['final_lp']['objective'],
                fractional_route_weight=item['final_lp']['route_weight'],
                cg_pool_columns=item['final_lp']['pool_columns'], last_reduced_cost=item['final']['min_rc'],
                cg_path=item['path'], cg_sha256=item['sha256'])
            checks.append(dict(campaign='compact_large_seed_20260914', case=cid, stage='cg', passed=True))
        mip_id = cid + '_mip'
        if mip_id in mip_by_case:
            cg = cg_by_case[cid]
            row.update(verify_mip(mip_by_case[mip_id], cases[mip_id], manifest_hash,
                                 cg['sha256'], cg['columns_journal_sha256']))
            checks.append(dict(campaign='compact_large_seed_20260914', case=mip_id, stage='mip', passed=True))
        rows.append(row)
    write_csv(out/'compact_large_results.csv', rows)
    lp = source['campaigns']['lp_support_pool_diagnostic_20260914']
    cases = lp['workflow']['manifest.json']['cases']
    manifest_hash = lp['workflow']['validation.json']['manifest_sha256']
    published = {case_id(item): item for item in lp['mip']}
    rows = []
    for cid, case in cases.items():
        row = dict(case_id=cid, pair_id=case['pair_id'], chain=case['chain'],
            target=case['target_k'], treatment=case['treatment'],
            source_pool_sha256=case['source_status_sha256'])
        if cid in published:
            row.update(verify_mip(published[cid], case, manifest_hash,
                                 case['source_status_sha256'], case['source_journal_sha256']))
            checks.append(dict(campaign='lp_support_pool_diagnostic_20260914', case=cid, stage='mip', passed=True))
        rows.append(row)
    write_csv(out/'lp_addition_results.csv', rows)
    result = dict(snapshot_sha256=hashlib.sha256(raw).hexdigest(), checks=checks, errors=[],
                  note='Collector publication and recorded provenance bindings verified; no new physical replay run.')
    (out/'pool_experiments_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(verified_endpoints=len(checks), large_cg=len(cg_by_case),
                         large_mip=len(mip_by_case), lp_addition_mip=len(published))))


if __name__ == '__main__':
    main()

"""Verify compact-pool union/control results without inventing CG certificates."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_pool_experiments import case_id, verify_mip, write_csv


def summarize(source, include_validation=False):
    name = 'compact_pool_union_20260915'
    campaign = source.get('campaigns', {}).get(name)
    if not campaign or 'manifest.json' not in campaign.get('workflow', {}):
        return [], []
    workflow = campaign['workflow']
    manifest = workflow['manifest.json']
    manifest_hash = workflow['scheduler_verification.json']['manifest_sha256']
    artifacts = list(campaign['mip'])
    if include_validation:
        artifacts += workflow.get('validation_mip_artifacts', [])
    published = {case_id(item): item for item in artifacts}
    constructions = workflow.get('pool_constructions', {})
    rows, checks = [], []
    for cid, case in manifest['cases'].items():
        if case['kind'] != 'mip' or (case.get('is_validation') and not include_validation):
            continue
        row = dict(case_id=cid, chain=case['chain'], target=case['target_k'],
                   treatment=case['treatment'], validation_only=bool(case.get('is_validation')),
                   total_budget_s=case['solver_budget_s'], fleet_budget_s=case['stage1_s'],
                   input_sha256=case['input_sha256'], execution_commit=case['execution_commit'],
                   new_cg_run=False, new_pricing_certificate=False,
                   initialization_policy='Frozen native greedy; realized start depends on pool')
        built = constructions.get(case['source_case'])
        if built:
            assert built['optimization_run'] is False
            assert built['full_model_lp_certified'] is False
            assert built['manifest_sha256'] == manifest_hash
            detail = built['construction_summary']
            row.update(known_donor_fleet_upper_bound=detail['best_independent_fleet_upper_bound'],
                       donor_bound_is_solver_incumbent=False,
                       donor_child_cg_minutes=detail['combined_child_cg_wall_s']/60,
                       construction_minutes=detail['constructor_wall_s']/60,
                       union_columns=detail['union_pool_set']['native_unique_columns'])
        if cid in published:
            assert built is not None, (cid, 'missing source construction audit')
            item = published[cid]
            if case['treatment'] == 'union':
                status_hash, journal_hash = built['result_sha256'], built['journal_sha256']
            else:
                donor = case['control_source']
                status_hash, journal_hash = donor['status_sha256'], donor['journal_sha256']
            row.update(verify_mip(item, {**case, 'stage1_budget_s': case['stage1_s']},
                                  manifest_hash, status_hash, journal_hash))
            audit = item['physical_pool_audit']
            assert audit['base_pool_column_count'] == audit['post_augmentation_columns']
            assert audit['base_pool_ordered_sha256'] == audit['augmented_pool_ordered_sha256']
            assert item['two_stage']['stage2_fleet_constraint'] == 'at_most'
            args = item['mip_provenance']['arguments']
            assert args['initial_partition_routes'] is None
            assert not args['verified_expanded_initial_partition']
            assert args['threads'] == 8
            if case['treatment'] == 'control':
                assert audit['base_pool_ordered_sha256'] == donor['native_pool_ordered_sha256']
                row['unchanged_control_ordered_pool_verified'] = True
            else:
                assert audit['base_pool_column_count'] == row['union_columns']
            row.update(target_matched_by_solver=item['buses'] <= case['target_k'],
                       native_greedy_start=json.dumps(item.get('mip_start'), sort_keys=True),
                       source_status_sha256=status_hash, source_journal_sha256=journal_hash)
            checks.append(dict(case_id=cid, publication_and_model_match=True,
                               zero_augmentation=True, validation_only=row['validation_only']))
        rows.append(row)
    return rows, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--include-validation', action='store_true')
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    rows, checks = summarize(json.loads(raw), args.include_validation)
    out = args.out_dir or Path(__file__).parent / ('status_' + args.snapshot.stem)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'compact_union_results.csv', rows)
    (out / 'compact_union_validation.json').write_text(json.dumps(dict(
        snapshot_sha256=hashlib.sha256(raw).hexdigest(), checks=checks, errors=[],
        include_validation=args.include_validation,
        note='Donor upper bound is separate from the new solver incumbent. Construction creates no CG certificate.'
    ), indent=2) + '\n')
    print(json.dumps(dict(registered_mips=len(rows), verified_results=len(checks))))


if __name__ == '__main__':
    main()

"""Audit the two separately launched longer searches on unchanged chain pools."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_pool_experiments import case_id, verify_mip, write_csv


def summarize(source):
    rows, checks = [], []
    for name in ('final_chain_gap_20260915', 'continuation_gap_20260915', 'continuation_gaps2_20260915', 'continuation_gaps3_20260915', 'continuation_gaps4_20260915'):
        campaign = source['campaigns'].get(name)
        if campaign is None:
            continue  # A collection can predate registration of a new campaign.
        workflow = campaign['workflow']
        manifest = workflow['manifest.json']
        manifest_hash = workflow['validation.json']['manifest_sha256']
        published = {case_id(item): item for item in campaign['mip']}
        for cid, case in manifest['cases'].items():
            row = dict(campaign=name, case_id=cid, chain=case['chain'],
                       target=case['target_k'], original_buses=case['latest_original_buses'],
                       total_budget_s=case['solver_budget_s'],
                       fleet_budget_s=case['stage1_budget_s'],
                       input_sha256=case['input_sha256'],
                       source_status_sha256=case['source_status_sha256'],
                       source_journal_sha256=case['source_journal_sha256'])
            if cid in published:
                item = published[cid]
                row.update(verify_mip(item, case, manifest_hash,
                                      case['source_status_sha256'],
                                      case['source_journal_sha256']))
                original = next(x for x in manifest['selection_audit']
                                if x['case_id'] == case['original_case'])['physical_pool_audit']
                actual = item['physical_pool_audit']
                for key in ('mip_ordered_pool_sha256', 'mip_unique_accepted_columns',
                            'assigned_mip_start_route_count', 'input_hashes'):
                    assert actual[key] == original[key], (name, cid, key)
                assert item['two_stage']['stage2_fleet_constraint'] == 'at_most'
                row['ordered_pool_sha256'] = actual['mip_ordered_pool_sha256']
                row['unchanged_pool_verified'] = True
                row['same_initializer_count_verified'] = True
                row['target_matched'] = item['buses'] <= case['target_k']
                row['interpretation'] = 'New tree on unchanged pool; no new CG certificate.'
                checks.append(dict(campaign=name, case_id=cid, passed=True))
            rows.append(row)
    return rows, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    rows, checks = summarize(json.loads(raw))
    out = args.out_dir or Path(__file__).parent / ('status_' + args.snapshot.stem)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'longer_gap_results.csv', rows)
    (out / 'longer_gap_validation.json').write_text(json.dumps(dict(
        snapshot_sha256=hashlib.sha256(raw).hexdigest(), checks=checks, errors=[],
        note='Source and recorded pool bindings checked; no new physical replay performed.'
    ), indent=2) + '\n')
    print(json.dumps(dict(registered_cases=len(rows), verified_results=len(checks))))


if __name__ == '__main__':
    main()

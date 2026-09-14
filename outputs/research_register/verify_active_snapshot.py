"""Check active campaign values against source evidence and retain dated checks."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--visual-review', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    directory = Path(__file__).parent
    workbook = root / 'outputs/01a07ecc-9b77-79b3-9782-e4308a80ba07'
    stamp = args.snapshot.stem
    source, register = read(args.snapshot), read(directory / 'register.json')
    assert register['source_snapshot']['sha256'] == sha(args.snapshot)
    assert register['source_snapshot']['timestamp_utc'] == source['timestamp_utc']
    lookup = {}
    for row in register['rows']:
        key = (row['campaign_id'], row['stage'], row['source_path'])
        lookup.setdefault(key, []).append(row)
    checks, errors, counts = [], [], {}
    for campaign in ('chain_extension_20260913', 'cumulative_budget_20260913',
                     'overnight_diagnostics_20260914', 'mip_repeatability_20260914'):
        counts[campaign] = {}
        for stage in ('cg', 'mip'):
            items = source['campaigns'][campaign][stage]
            counts[campaign][stage] = len(items)
            for item in items:
                path = item['path']
                matches = lookup.get((campaign, stage, path), [])
                assert len(matches) == 1, (campaign, stage, path, len(matches))
                if stage == 'cg':
                    lp, final = item.get('final_lp') or {}, item.get('final') or {}
                    expected = dict(full_model_lp_certified=item.get('certified_rc_optimal'),
                        stop_reason=item.get('stop_reason'),
                        weighted_lp_objective=lp.get('objective', final.get('lp_obj')),
                        fractional_fleet=lp.get('route_weight', final.get('route_weight')))
                else:
                    expected = dict(mip_incumbent_fleet=item.get('buses'),
                        mip_bound_fleet=item.get('fleet_bound'), fleet_proven=item.get('fleet_proven'),
                        physical_selected_validated=item.get('physical_replay_validated'))
                differences = [[key, value, matches[0].get(key)]
                               for key, value in expected.items() if matches[0].get(key) != value]
                if differences:
                    errors.append([campaign, stage, path, differences])
                checks.append(dict(campaign=campaign, stage=stage, path=path, exact_match=not differences))
    supplements = register['supplemental_sources']
    assert len(supplements) == 6
    for item in supplements:
        assert sha(directory / item['path']) == item['sha256']
    workbook_checks = read(workbook / 'workbook_checks.json')
    assert workbook_checks['sourceTime'] == source['timestamp_utc']
    assert workbook_checks['sourceRows'] == register['row_count']
    result = dict(snapshot=register['source_snapshot'], scientific_endpoints_verified=len(checks),
        counts=counts, exact_match=not errors, errors=errors, supplements_preserved=supplements,
        checks=checks, workbook_checks=workbook_checks,
        rendered_changed_views=read(workbook / f'{stamp}_verification.json')['views'],
        artifact_hashes={str(path.relative_to(root)): sha(path) for path in (
            directory / 'register.json', directory / 'register.csv', workbook / 'EVSP_DR_Experiment_Register.xlsx')},
        visual_review=args.visual_review)
    (directory / f'cumulative_validation_{stamp}.json').write_text(json.dumps(result, indent=2) + '\n')
    assert not errors, errors
    shutil.copy2(directory / 'validation.json', directory / f'validation_{stamp}.json')
    shutil.copy2(workbook / 'workbook_checks.json', workbook / f'workbook_checks_{stamp}.json')
    print(json.dumps({key: result[key] for key in ('scientific_endpoints_verified', 'counts', 'exact_match', 'errors')}))


if __name__ == '__main__':
    main()

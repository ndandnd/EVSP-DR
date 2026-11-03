#!/usr/bin/env python3
"""Bind one selected archived attempt to the frozen remote audit inventory.

Paths default to audited Unicorn locations. Optional local overrides support regression;
all override contents must still match the authenticated inventory hashes.
"""
import argparse
import json
from pathlib import Path
from run_recombination import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--attempt', required=True,
                        help='Exact native attempt directory name, e.g. job920645_r1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--solver-root', type=Path)
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    if sha(args.inventory) != args.inventory_sha256:
        raise ValueError('Inventory hash mismatch')
    native = json.loads(args.inventory.read_text())
    choices = [(index, case) for index, case in enumerate(native['case_attempts'])
               if case['case'] == args.case
               and Path(case['summary_path']).parent.name == args.attempt]
    if len(choices) != 1:
        raise ValueError('Select exactly one explicitly named audited attempt')
    index, case = choices[0]
    summary = case['summary']
    data = args.data_root or Path(summary['inputs']['data_dir'])
    owner_data = Path(summary['inputs']['data_dir'])
    instance_relative = Path(summary['inputs']['instance']).relative_to(owner_data)
    inputs = {'instance': data / instance_relative,
              'prices': data / 'hourly_prices_flat.csv',
              'locations': data / 'Ref_dict.csv',
              'deadheads': data / 'par_ref_dhd.csv'}
    owner_paths = {'instance': summary['inputs']['instance'],
                   'prices': summary['inputs']['tariff'],
                   'locations': str(owner_data / 'Ref_dict.csv'),
                   'deadheads': str(owner_data / 'par_ref_dhd.csv')}
    manifest = {
        'schema': 'frozen-root-trip-link-manifest-v1',
        'case': args.case,
        'summary': str(args.summary or Path(case['summary_path'])),
        'summary_sha256': case['summary_sha256'],
        'pool': str(args.pool or Path(case['pool_path'])),
        'root_columns': case['root_columns'],
        'root_prefix_sha256': case['root_actual_sha256'],
        'expected_cg_pricing_certified': summary['cg_pricing_certified'],
        'solver_root': str(args.solver_root or Path(native['solver']['path'])),
        'solver_commit': native['solver']['revision'],
        'solver_source_sha256': native['solver']['source_sha256'],
        'data_root': str(data),
        'inputs': {role: str(path) for role, path in inputs.items()},
        'input_sha256': {role: summary['input_hashes'][path]
                         for role, path in owner_paths.items()},
        'target_fleet': 8,
        'native_evidence': [{'path': str(args.inventory.resolve()),
                             'sha256': args.inventory_sha256}],
        'native_binding': {'evidence_index': 0,
                           'case_pointer': '/case_attempts/' + str(index),
                           'solver_pointer': '/solver'},
        'source_provenance_limit': 'Audit authenticates the current pinned source. '
            'Archived as-run owner summary does not independently record source hashes.'}
    with args.output.open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(str(args.output))


if __name__ == '__main__':
    main()

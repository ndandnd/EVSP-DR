"""Read-only comparison of the original and repeated C3 k28 MIP logs."""
from pathlib import Path
import datetime
import hashlib
import json
import re


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_case(label, path, expected):
    raw = path.read_bytes()
    assert sha(raw) == expected
    value = json.loads(raw)
    log = path.resolve().parent / 'gurobi.log'
    data = log.read_bytes()
    lines = data.decode().splitlines()
    matches = [(i + 1, line) for i, line in enumerate(lines)
               if line.startswith(('Optimize a model', 'CPU model:',
                                   'Root relaxation:', 'Explored ', 'Best objective'))]
    explored = next((i, s) for i, s in matches if s.startswith('Explored '))
    parsed = re.fullmatch(r'Explored (\d+) nodes \((\d+) simplex iterations\) in ([\d.]+) seconds \(([\d.]+) work units\)', explored[1])
    assert parsed is not None
    stage = value['two_stage']
    args = value['mip_provenance']['gurobi_parameters']
    assert args['Threads'] == 8 and args['Seed'] == 0
    return value, dict(label=label, result_path=str(path), result_sha256=expected,
        log_path=str(log), log_sha256=sha(data), host=value['mip_provenance']['host'],
        cpu_model=next(s for _, s in matches if s.startswith('CPU model:')),
        fleet_model=next(s for _, s in matches if s.startswith('Optimize a model')),
        root_relaxation=next(s for _, s in matches if s.startswith('Root relaxation:')),
        fleet_budget_s=stage['stage1_time_limit_s'],
        fleet_native_elapsed_s=stage['stage1_runtime_s'],
        fleet_optimizer_elapsed_s=float(parsed[3]), fleet_work_units=float(parsed[4]),
        fleet_nodes=int(parsed[1]), fleet_simplex_iterations=int(parsed[2]),
        buses=value['buses'], fleet_bound=value['fleet_bound'],
        fleet_proven=value['fleet_proven'], threads=args['Threads'], seed=args['Seed'],
        fleet_summary_log_line=explored[0], captured_log_lines=matches)


def main():
    home = Path('/home/nc437/ladder-lite')
    pairs = [
        ('original', home/'chain_extension_20260914/cases/w3_k28/mip/187985_r0/result.json',
         '8ef6e4eff1790817f1e73a6e38c4ba317d3a5bbc8e5f7dcadae7ba074e191ab6'),
        ('separate_longer_budget', home/'continuation_gap_20260915/cases/w3_k28_longmip/mip_result.json',
         '5471dffc6183f138c2a5bbeba4d533b967e8a0b3725724e3e24ffa62a9c766d7'),
    ]
    cases = [read_case(*pair) for pair in pairs]
    left, right = [c[0] for c in cases]
    for key in ('source_result_sha256', 'source_journal_sha256'):
        assert left[key] == right[key], key
    for key in ('base_pool_ordered_sha256', 'base_pool_column_count',
                'assigned_mip_start_route_count', 'input_hashes'):
        assert left['physical_pool_audit'][key] == right['physical_pool_audit'][key], key
    assert left['mip_provenance']['git_commit'] == right['mip_provenance']['git_commit']
    rows = [c[1] for c in cases]
    print(json.dumps(dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_and_ordered_pool_match=True, rows=rows,
        repeated_to_original_reported_work_ratio=rows[1]['fleet_work_units']/rows[0]['fleet_work_units'],
        reported_work_per_optimizer_second_ratio=(rows[1]['fleet_work_units']/rows[1]['fleet_optimizer_elapsed_s'])/(rows[0]['fleet_work_units']/rows[0]['fleet_optimizer_elapsed_s']),
        interpretation='Different hardware and time-limit settings; observational comparison, not a controlled causal hardware or algorithm benchmark. Same root LP iterations/work and faster repeated root solve support a throughput difference. Work units do not establish identical search paths across hardware.',
        work_reference='https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/model.html#work'
    ), indent=2))


if __name__ == '__main__':
    main()

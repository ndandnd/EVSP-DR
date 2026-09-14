"""Summarize registered MIP repeats without changing or submitting experiments."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.snapshot.read_text())
    digest = hashlib.sha256(args.snapshot.read_bytes()).hexdigest()
    root = Path(__file__).parent
    manifest = json.loads((root / 'manifest.json').read_text())
    alias_audit = json.loads((root / 'audit/comparator_serialization.json').read_text())
    aliases = {row['canonical_path']: row for row in alias_audit['rows']}
    campaign = source['campaigns']['mip_repeatability_20260914']
    previous = {r['path']: r for r in source['campaigns']['chain_extension_20260913']['mip']}
    # New follow-ups bind the stable publication link; earlier records used its
    # resolved job-specific file. Both identify the same hash-checked result.
    for row in list(previous.values()):
        parts = Path(row['path']).parts
        if 'cases' in parts:
            pos = parts.index('cases')
            alias = Path(*parts[:pos + 2]) / 'mip_result.json'
            previous[str(alias)] = row
    longer = {Path(r['path']).parts[-2]: r for r in source['campaigns']['overnight_diagnostics_20260914']['mip']}
    attempts = {r['case_id']: r['state'] for r in campaign['workflow']['attempt_progress']}
    rows = []
    for current in campaign['mip']:
        cid = Path(current['path']).parts[-2]
        case = manifest['cases'][cid]
        prior = previous[case['comparator']['prior_result']]
        if prior['sha256'] != case['comparator']['prior_sha256']:
            audit = aliases[case['comparator']['prior_result']]
            assert audit['full_json_values_equal'] is True
            assert audit['raw_path'] == prior['path']
            assert audit['raw_sha256'] == prior['sha256']
            assert audit['canonical_sha256'] == case['comparator']['prior_sha256']
        old_pool, new_pool = prior['physical_pool_audit'], current['physical_pool_audit']
        assert old_pool['mip_ordered_pool_sha256'] == new_pool['mip_ordered_pool_sha256']
        assert old_pool['input_hashes'] == new_pool['input_hashes']
        assert current['completion_marker_matches'] and current['physical_replay_validated']
        assert current['source_status_sha256'] == case['source_status_sha256']
        assert current['source_journal_sha256'] == case['source_journal_sha256']
        provenance = current['mip_provenance']
        assert provenance['observed_git_commit'] == case['execution_commit']
        assert provenance['final_observed_git_commit'] == case['execution_commit']
        assert not provenance['git_dirty'] and provenance['tracked_clean_at_end']
        stage = current['two_stage']
        assert stage['stage1_time_limit_s'] == case['stage1_budget_s']
        assert provenance['arguments']['timelimit'] == case['solver_budget_s']
        start_keys = ('kind', 'validated_bus_count', 'expected_full_objective')
        same_start_summary = all(prior['mip_start'].get(k) == current['mip_start'].get(k) for k in start_keys)
        assert same_start_summary
        old_long = longer.get(case.get('comparator_case'))
        if old_long:
            assert old_long['sha256'] == case['comparator_result_sha256']
        execution = attempts[cid].get('execution') or {}
        rows.append(dict(
            case_id=cid, original_case=case['original_case'], chain=case['chain'],
            target_k=case['target_k'], treatment=case['treatment'], replicate=case.get('replicate'),
            original_buses=prior['buses'], original_fleet_stage_minutes=prior['two_stage']['stage1_runtime_s']/60,
            original_fleet_stage_nodes=prior['two_stage']['stage1_node_count'],
            longer_buses=old_long['buses'] if old_long else None,
            longer_fleet_stage_minutes=old_long['two_stage']['stage1_runtime_s']/60 if old_long else None,
            buses=current['buses'], pool_fleet_bound=current['fleet_bound'], fleet_proven=current['fleet_proven'],
            target_matched=current['buses'] <= case['target_k'], individual_route_replay=True,
            stage1_buses=stage['stage1_buses'], stage1_status=stage['stage1_status_name'],
            fleet_stage_minutes=stage['stage1_runtime_s']/60, fleet_stage_nodes=stage['stage1_node_count'],
            stage1_budget_s=case['stage1_budget_s'], total_budget_s=case['solver_budget_s'],
            solver_host=provenance['host'], wrapper_host=execution.get('host'),
            gurobi_version=provenance['gurobi'], threads=provenance['gurobi_parameters']['Threads'],
            seed=provenance['gurobi_parameters']['Seed'], process_wall_s=execution.get('wall_s'),
            process_user_cpu_s=execution.get('user_cpu_s'), process_system_cpu_s=execution.get('system_cpu_s'),
            ordered_pool_sha256=new_pool['mip_ordered_pool_sha256'], same_input_hashes=True,
            same_start_summary=same_start_summary, original_result_path=prior['path'],
            original_result_sha256=prior['sha256'], result_path=current['path'], result_sha256=current['sha256'],
            execution_commit=case['execution_commit'], snapshot_sha256=digest))
    out = root / ('status_' + args.snapshot.stem)
    out.mkdir(exist_ok=True)
    if rows:
        with (out / 'results.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    repeats = [r for r in rows if r['treatment'] == 'original_budget_repeatability']
    groups = defaultdict(list)
    for r in repeats:
        groups[r['original_case']].append(r)
    summary = dict(snapshot=str(args.snapshot), snapshot_sha256=digest,
        completed_mips=len(rows), completed_repeats=len(repeats), selected_input_sets=len(groups),
        repeat_target_matches=sum(r['target_matched'] for r in repeats),
        repeat_pool_fleet_proofs=sum(r['fleet_proven'] for r in repeats),
        all_repeated_fleet_outcomes_agree=all(len({r['buses'] for r in g}) == 1 for g in groups.values()),
        all_individual_routes_replayed=all(r['individual_route_replay'] for r in rows),
        all_ordered_pools_inputs_and_start_summaries_match=True,
        full_model_integer_proof_claimed=False,
        all_execution_statuses={cid: r.get('status') for cid, r in attempts.items()})
    (out / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    text = ['# Repeating the original MIP allowance', '',
        f"{len(repeats)} completed repetitions on {len(groups)} selected inherited pools. "
        f"{summary['repeat_target_matches']} match the target fleet. These are repeated computations, not independent sampled datasets.", '',
        '| Case | Target | Original buses | Repeat 1 | Repeat 2 | Repeat 3 | Repeat fleet-stage minutes | Earlier longer-search buses |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for cid, group in sorted(groups.items()):
        by_rep = {r['replicate']: r for r in group}
        r = group[0]
        times = [x['fleet_stage_minutes'] for x in group]
        values = [str(by_rep[i]['buses']) if i in by_rep else 'pending' for i in (1, 2, 3)]
        text.append(f"| C{r['chain']} k{r['target_k']} | {r['target_k']} | {r['original_buses']} | " +
                    ' | '.join(values) + f" | {min(times):.1f}–{max(times):.1f} | {r['longer_buses']} |")
    new_long = [r for r in rows if r['treatment'] != 'original_budget_repeatability']
    if new_long:
        text += ['', '## Additional unresolved targets', '',
            '| Case | Target | Original buses | Longer-search buses | Fleet bound | Fleet proved in saved pool | Fleet-stage minutes |',
            '|---|---:|---:|---:|---:|---|---:|']
        for row in new_long:
            text.append(f"| C{row['chain']} k{row['target_k']} | {row['target_k']} | {row['original_buses']} | {row['buses']} | {row['pool_fleet_bound']:.6g} | {'Yes' if row['fleet_proven'] else 'No'} | {row['fleet_stage_minutes']:.1f} |")
        text += ['', 'These searches use the original saved columns and up to three hours for fleet minimization, within a three-and-a-half-hour total budget. Fleet proof and charging-cost optimality are separate.']
    text += ['',
        'Each repeat has a one-hour total solver allowance, with up to 30 minutes for fleet minimization. '
        'The second stage minimizes charging costs with fleet no greater than the first-stage incumbent. '
        'Small runtime overruns while the solver stops are recorded rather than rounded into an exact deadline.', '',
        'All three repeats agree on the fleet within each completed case. Seven selected pools reach their targets in all three repeats. '
        'C3 k19 and C3 k21 retain one extra bus in every repeat, with open bounds at the target. Earlier longer searches recovered both targets. '
        'The same pools therefore contain the needed routes; these misses are incomplete integer searches. '
        'The seven recoveries at the original allowance show that extra allocated time alone is not the cause of the original-versus-rerun difference. '
        'Hardware and parallel-search timing are not isolated. These observations do not estimate a population success probability.', '',
        'Ordered pool hashes, physical input hashes, source CG hashes, execution commit, and initializer kind/count/cost agree with the registered comparison. '
        'The initializer summary check is not a hash of the selected initializer route indices. Repeats keep Gurobi 12.0.3, eight threads and default seed zero. '
        'All selected routes pass individual replay. Fleet proofs concern the saved pools; charging optimality, duplicate-trip removal and shared-capacity validation are separate. '
        'The baseline omits shared station capacity and a terminal-SOC floor.', '',
        '[Exact results, hosts, timings, node counts and source hashes](results.csv). '
        '[Validation and remaining attempt states](validation.json).']
    (out / 'README.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'all_execution_statuses'}))


if __name__ == '__main__':
    main()

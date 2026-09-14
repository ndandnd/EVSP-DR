"""Report the frozen longer-MIP comparison; never submit jobs or alter outputs."""
import csv
import hashlib
import json
from pathlib import Path


def report(source, campaign, cases, endpoints, root, snapshot_sha):
    selected = {cid: row for cid, row in endpoints.items() if cid.endswith('longmip')}
    if not selected:
        return None
    # Original fresh collector projections omit the pool audit. This retained
    # extraction was read from the original files after verifying their hashes.
    audit_path = Path(__file__).parent/'status_20260914T082509Z/original_fresh_pool_audit.json'
    extra = {r['path']: r for r in json.loads(audit_path.read_text())}
    prior_rows = {r['path']: r for name in ('cumulative_budget_20260913', 'chain_extension_20260913')
                  for r in source['campaigns'][name]['mip']}
    rows = []
    for cid, current in sorted(selected.items()):
        case = cases[cid]
        comparator = case['comparator']
        prior = prior_rows[comparator['prior_result']]
        assert prior['sha256'] == comparator['prior_sha256']
        detail = prior if prior.get('physical_pool_audit') else extra[prior['path']]
        assert detail.get('sha256', prior['sha256']) == prior['sha256']
        old_pool, new_pool = detail['physical_pool_audit'], current['physical_pool_audit']
        pool_hash = old_pool['mip_ordered_pool_sha256']
        assert pool_hash and pool_hash == new_pool['mip_ordered_pool_sha256']
        assert old_pool['input_hashes'] == new_pool['input_hashes']
        assert current['completion_marker_matches'] and current['physical_replay_validated']
        k = case['target_k']
        outcome = ('target matched' if current['buses'] <= k else
                   'proved pool limit above target' if current['fleet_proven'] else 'fleet gap open')
        rows.append(dict(case_id=cid, origin=comparator['origin'], chain=case['chain'], target_k=k,
            original_buses=prior['buses'], original_bound=prior['fleet_bound'],
            original_fleet_proven=prior['fleet_proven'], longer_buses=current['buses'],
            longer_bound=current['fleet_bound'], longer_fleet_proven=current['fleet_proven'],
            outcome=outcome, individual_route_replay=True,
            original_fleet_stage_minutes=prior['two_stage']['stage1_runtime_s']/60,
            longer_fleet_stage_minutes=current['two_stage']['stage1_runtime_s']/60,
            original_stage1_budget_s=prior['two_stage']['stage1_time_limit_s'],
            longer_stage1_budget_s=current['two_stage']['stage1_time_limit_s'],
            longer_total_budget_s=case['solver_budget_s'],
            original_start_bus_count=detail['mip_start']['validated_bus_count'],
            longer_start_bus_count=current['mip_start']['validated_bus_count'],
            same_ordered_mip_pool=True, same_input_hashes=True, ordered_mip_pool_sha256=pool_hash,
            original_result_path=prior['path'], original_result_sha256=prior['sha256'],
            longer_result_path=current['path'], longer_result_sha256=current['sha256'],
            execution_commit=case['execution_commit'], snapshot_sha256=snapshot_sha))
    with (root/'longer_mip_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for origin in ('fresh', 'extension'):
        group = [r for r in rows if r['origin'] == origin]
        grouped[origin] = dict(cases=len(group),
            targets=sum(r['outcome'] == 'target matched' for r in group),
            proved_pool_limits=sum(r['outcome'] == 'proved pool limit above target' for r in group),
            open_gaps=sum(r['outcome'] == 'fleet gap open' for r in group),
            target_fleet_proved_within_30_minutes=sum(r['outcome'] == 'target matched' and
                r['longer_fleet_proven'] and r['longer_fleet_stage_minutes'] <= 30 for r in group))
    validation = dict(cases=len(rows), grouped=grouped, all_ordered_pools_and_inputs_identical=True,
        original_fresh_audit_path=str(audit_path),
        original_fresh_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        snapshot_sha256=snapshot_sha, full_model_integer_proof_claimed=False)
    (root/'longer_mip_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    text = ['# Longer MIP searches on unchanged pools', '',
        'These are the selected unresolved one-hour cases from the frozen overnight manifest. Each rerun allows up to three hours for fleet search and 3½ hours total, compared with 30 minutes and one hour originally. It starts a new Gurobi tree. All 23 ordered MIP-pool hashes and input hashes match their original runs; no new CG columns enter this comparison.', '',
        '| Pool source | Completed cases | Targets matched | Proved extra bus required in pool | Fleet gap still open |',
        '|---|---:|---:|---:|---:|']
    for origin, label in [('extension', 'Inherited chain pools'), ('fresh', 'Fresh CG pools')]:
        g = grouped[origin]
        text.append(f"| {label} | {g['cases']} | {g['targets']} | {g['proved_pool_limits']} | {g['open_gaps']} |")
    text += ['', 'All nine inherited-pool reruns find and prove the target fleet. Seven finish that fleet proof within 30 minutes in this rerun, despite the original 30-minute fleet search missing it. Therefore extra elapsed time alone does not explain all recoveries. The configured time allowance changed; hardware, parallel search and timing effects are not isolated. The result establishes that the target solutions were already present in those original pools.', '',
        'None of the fourteen fresh-pool reruns reaches its target. C3k8 proves nine necessary, C5k10 proves eleven, and C6k10 proves eleven, each within its saved pool. Their fleet stages take about 71.8, 77.8 and 46.1 minutes. Eleven other gaps remain open; these are not proofs of missing target solutions. C2k10 improves twelve to eleven and C6k15 improves twenty to eighteen, still above target.', '',
        'The populations have different sizes and were selected because they previously missed targets. These counts do not estimate general success probabilities or a causal warm-versus-fresh advantage. Individual-route replay passes. Shared station capacity and terminal-SOC constraints remain absent. Fleet proofs are finite-pool claims; charging optimality is separate.', '',
        '| Case | Target | Original buses | Longer-search buses | Pool fleet bound | Fleet proved in pool? | Fleet-stage minutes |',
        '|---|---:|---:|---:|---:|---|---:|']
    text += [f"| {r['case_id']} | {r['target_k']} | {r['original_buses']} | {r['longer_buses']} | {r['longer_bound']:.0f} | {'yes' if r['longer_fleet_proven'] else 'no'} | {r['longer_fleet_stage_minutes']:.1f} |" for r in rows]
    text += ['', '[Exact results, timings and matched pool hashes](longer_mip_comparison.csv).']
    (root/'LONGER_MIP_RESULTS.md').write_text('\n'.join(text)+'\n')
    return validation

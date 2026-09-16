"""Verify saved selections and compare like-for-like cost accounting."""
import collections
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
baseline = json.loads((root/'outputs/research_questions_20260915/evidence.json').read_text())
manifest = json.loads((root/'outputs/terminal_duplicate_cleanup_20260916/manifest.json').read_text())
rows = []
for case in manifest['cases']:
    peak = case['id'][:6]
    comparison = next(r for r in baseline['charging_comparisons'] if r['pair_id'] == peak+'_fee0')
    fixed = comparison['result']['fixed_duties_optimized']
    source = root/'outputs'/Path(case['source']).relative_to('/home/nc437/ladder-lite')
    assert sha(source) == case['sha256']
    assert sha(source.parent/'selected_routes.json') == case['selected_source_sha256']
    original = json.loads(source.read_text())
    assert comparison['instance_sha256'] in original['input_hashes'].values()
    assert comparison['tariff_sha256'] in original['input_hashes'].values()
    row = dict(peak=peak, fixed_grid=fixed['expanded_grid_charging_cost'],
               fixed_continuous=fixed['physical_charging_cost'],
               fixed_terminal_kwh=fixed['continuous_terminal_energy_kwh'],
               baseline_source=comparison['path'], baseline_sha256=comparison['sha256'],
               cleanup_status='pending')
    results = sorted((root/'outputs/terminal_duplicate_cleanup_20260916/results'/case['id']).glob('*/summary.json'))
    assert len(results) <= 1, 'Choose and report attempts explicitly, never best-of silently'
    if results:
        path = results[0]
        result = json.loads(path.read_text())
        assert result['source_sha256'] == case['sha256']
        assert result['selected_source_sha256'] == case['selected_source_sha256']
        chosen_path = path.parent/'selected_routes.json'
        chosen = json.loads(chosen_path.read_text())
        counts = collections.Counter(t for r in chosen for t in r['trips'])
        assert len(chosen) == 5 and len(counts) == 62 and set(counts.values()) == {1}
        for field, summary_field in [('cost','charging_cost'), ('continuous_realized_cost','continuous_charging_cost')]:
            # Gurobi objective uses numerical variable values; selected routes use integers.
            # Retain the discrepancy and reject anything larger than 0.0001 cost units.
            assert abs(sum(r[field]-100000 for r in chosen)-result[summary_field]) < 1e-4
        assert abs(sum(r['continuous_terminal_energy_kwh'] for r in chosen)-result['terminal_kwh']) < 1e-5
        assert result['terminal_kwh'] + 1e-5 >= comparison['result']['target_terminal_energy_kwh']
        assert result['exact_once_verified'] and result['individual_replay_verified']
        row.update(cleanup_status='validated selection', cleanup_grid=sum(r['cost']-100000 for r in chosen),
                   solver_grid_objective=result['charging_cost'],
                   selected_minus_solver_grid_cost=sum(r['cost']-100000 for r in chosen)-result['charging_cost'],
                   cleanup_continuous=result['continuous_charging_cost'],
                   cleanup_terminal_kwh=result['terminal_kwh'],
                   same_terminal_energy=abs(result['terminal_kwh']-row['fixed_terminal_kwh']) < 1e-5,
                   reduction_percent=100*(1-result['continuous_charging_cost']/row['fixed_continuous']),
                   charging_status=result['charging_status'],
                   charging_bound_grid=result.get('charging_bound'),
                   charging_relative_gap=(result['charging_cost']-result['charging_bound'])/abs(result['charging_cost']),
                   summary_path=str(path.relative_to(root)), summary_sha256=sha(path),
                   selected_sha256=sha(chosen_path))
    rows.append(row)
(out/'comparison.json').write_text(json.dumps(dict(rows=rows, scope='Fresh full CG followed by duplicate deletion and charging reoptimization; simplified 240 kWh/350 kW, no reserve or shared capacity; aggregate ending energy, zero start fee. Not global optimality.'),indent=2)+'\n')
lines=['# Zero-fee comparison: validated trip assignments', '',
       'All costs below use continuously replayed charging, in tariff cost units. Each completed cleanup covers all 62 trips exactly once using five buses. Cleanup deletes repeated trips from the selected CG duties and reoptimizes their charging; it imports no GIRO duties and is a separate postprocessing step.', '',
       '| Tariff peak | Fixed GIRO duties: charging optimized | Fresh CG + validated cleanup | Reduction | Same total ending energy? |',
       '|---|---:|---:|---:|---|']
for r in rows:
    done=r['cleanup_status']!='pending'
    lines.append('| '+r['peak'][4:]+':00 | '+f"{r['fixed_continuous']:.2f}"+' | '+(f"{r['cleanup_continuous']:.2f}" if done else 'Pending')+' | '+(f"{r['reduction_percent']:.2f}%" if done else '—')+' | '+(str(r['same_terminal_energy']) if done else '—')+' |')
lines += ['', 'These results use 240 kWh batteries, 350 kW charging, no reserve and no shared-station capacity. The common aggregate ending-energy minimum is 280.7833253 kWh; it is not a per-bus SOC requirement. Completed 08:00/12:00 selections and their fixed-duty comparators return with 281.1700005 kWh in total.', '',
          'Full CG reported convergence on its weighted event-graph objective at 15.2, 16.8 and 22.2 minutes (graph construction excluded). Its original covering selections repeat trips. Requiring exact-once coverage using only those unchanged pools rules out five buses at 08:00 and 12:00; 18:00 has a five-bus exact-once selection. This is a pool limitation: deleting duplicated trips and creating new charging variants can produce feasible schedules outside that pool, as the completed cleanups show.', '',
          'The 08:00 and 12:00 cleanup MIPs prove their grid charging objectives optimal within the cleanup pools. The 18:00 search ends at its one-hour limit: grid incumbent 90.403385, bound 90.280210, a 0.1363% gap. Its feasible improvement is verified; charging optimality remains unproved. All three fixed and cleaned schedules have matching total ending energy (281.1700005 kWh at 08:00/12:00, 282.9 kWh at 18:00).', '',
          'A charging proof in the cleanup pool is not a full-model charging proof. Continuous replayed costs are not the grid objective covered by the solver proof. The results support an improvement over the tested fixed-duty optimizer under these declared assumptions, not universal superiority or full GIRO feasibility.', '',
          'Sources: [comparison and hashes](comparison.json), [full-CG manifest](../zero_fee_full_cg_20260916/manifest.json), [cleanup manifest](../terminal_duplicate_cleanup_20260916/manifest.json), [exact-once pool check](../terminal_exact_once_20260916/manifest.json).']
(out/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))

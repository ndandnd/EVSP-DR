"""Write reviewable summaries and a small explicit publication allowlist."""
from pathlib import Path
import hashlib,json,subprocess,sys
P=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(P/'collect_and_plot.py')],check=True)
d=json.loads((P/'results_summary.json').read_text())
assert d['cases_completed']==18 and d['validated_incumbents']==18 and d['pairs_verified']==9
assert d['all_pairs_same_fixed_problem']
rows=d['rows'];pairs=d['pairs'];optimal=sum(r['solver_status']==2 for r in rows);timed=sum(r['solver_status']==9 for r in rows)
lines=['# Controlled fee results','',f'All 18 cells have validated five-bus witnesses serving the 62 trips exactly once; all nine fee-pair problem hashes match. {optimal} solver searches reached the configured 0.01% MIP gap; {timed} stopped at 600 seconds. One raw witness required the documented numerical repair below. All original solver statuses, lower bounds and revised witness gaps are retained in cell_results.csv.','',
 '| Fixed trip assignment | Tariff peak | Starts, fee 0→5 | Electricity cost, fee 0→5 | Electricity increase | Saving at common fee 5 | Fee-5 gap |',
 '|---|---:|---:|---:|---:|---:|---|']
names={'original':'Original','saved_joint_fee0':'Saved fee-0-derived','saved_joint_fee5':'Saved fee-5-derived'}
def percent(value):
    value*=100
    return (f'{value:.6f}%' if 0<value<0.0001 else f'{value:.4f}%')
for p in pairs:
    lines.append(f"| {names[p['assignment']]} | {p['tariff_peak']:02d}:00 | {p['fee0_starts']}→{p['fee5_starts']} | {p['fee0_electricity']:.3f}→{p['fee5_electricity']:.3f} | +{p['electricity_change']:.3f} | {p['fee5_objective_improvement_over_fee0_schedule']:.3f} | {percent(p['fee5_gap'])} |")
strict=sum(p['strict_reduction_for_all_restricted_model_optima'] for p in pairs)
lines += ['',f'Observed starts fell in {sum(p["starts_change"]<0 for p in pairs)}/9 paired incumbents. Lower and upper objective bounds establish strict separation of optimal start counts in {strict}/9 restricted models, with the 0.01 synthetic-unit numerical margin documented in README.md. This does not certify the displayed counts as unique.','',
 'Electricity cost E sums hourly tariff × charged kWh. The objective is F = E + fee × charging starts. “Saving at common fee 5” compares the two schedules under the same fee: (E0 + 5N0) − (E5 + 5N5). The increased electricity bill is reported separately. There is no fleet-cost term or currency conversion. Model objective gaps are not finite route-pool fleet gaps. The single-charge-per-gap, single-tariff-hour restriction and recovered station paths remain fixed.','',
 'For original/08:00/fee 5, raw replay missed one terminal floor by 0.00002816 kWh. A separate witness extends one existing charge by 6 milliseconds within its visit, tariff hour and available capacity, then recomputes downstream SOC, taper energy and costs. It passes the unchanged validator. Its feasible objective is 369.636334162, against the unchanged solver lower bound 369.636248039 (0.00002330% gap). The raw failed attempt remains untouched; this repaired witness is an upper bound, not a new solver optimality certificate. [Repair details](postprocess_repair/original_peak08_fee5/repair_report.json).','',
 '| Fixed trip assignment | Tariff peak | Fee-0 optimal starts, lower bound | Fee-5 optimal starts, upper bound | Strict reduction established |',
 '|---|---:|---:|---:|---|']
for p in pairs:
    lines.append(f"| {names[p['assignment']]} | {p['tariff_peak']:02d}:00 | {p['fee0_optimal_starts_lower_bound']} | {p['fee5_optimal_starts_upper_bound']} | {'Yes' if p['strict_reduction_for_all_restricted_model_optima'] else 'No'} |")
lines += ['',
 '[Editable paired table](paired_fee_results.csv) · [All endpoints and log locations](cell_results.csv) · [Figure caption](figure_caption.txt) · [Gurobi endpoint excerpts](log_excerpts.md)','']
(P/'RESULTS.md').write_text('\n'.join(lines))
logs=['# Gurobi endpoint evidence','', 'The complete logs remain at the absolute paths below. Excerpts are the solver endpoint lines, not a replacement for physical validation or the restricted-model scope.','']
for row in rows:
    path=Path(row['gurobi_log']);text=path.read_text()
    endpoint=[line for line in text.splitlines() if line.startswith(('Explored ','Solution count ','Optimal solution found','Time limit reached','Best objective'))]
    logs += [f"## {row['case_id']}",'',f'[{path.name}]({path})','', '```text', *endpoint[-5:], '```','']
    if row['witness_repaired']:
        logs += ['The raw incumbent failed physical replay by 0.00002816 kWh. A separately validated 6 ms charging extension supplies the reported feasible upper bound; the log above is the original search, whose bound is unchanged.','']
(P/'log_excerpts.md').write_text('\n'.join(logs))
files=['README.md','RESULTS.md','manifest.json','code_receipt.json','mathematical_model_diff.patch',
    'cell_results.csv','paired_fee_results.csv','results_summary.json','controlled_start_fee.png','controlled_start_fee.pdf',
    'controlled_start_fee.svg','figure_caption.txt','log_excerpts.md','collect_and_plot.py','finish_artifacts.py',
    'repair_numerical_witness.py','postprocess_repair/original_peak08_fee5/repair_report.json']
hashes={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in files}
allow=dict(root=str(P.resolve()),files=files,sha256=hashes,
    excludes=['bundle/ raw inputs and source deployment','native/ complete operational artifacts; retain locally for audit','smoke/ QA cases',
              '__pycache__/','deployment.tar.gz','scheduler snapshots','license contents','private real-tariff and invitation files'],
    note='Only listed aggregate findings, fixed-model provenance, plot source and endpoint excerpts are proposed for publication. Raw logs/models/results remain locally accessible via cell_results.csv.')
(P/'publish_allowlist.json').write_text(json.dumps(allow,indent=2)+'\n')
print(json.dumps(dict(validated_endpoints=18,verified_pairs=9,configured_gap_reached=optimal,time_limits=timed,
    strict_optimal_start_reduction_pairs=strict,publication_files=len(files),source_commit=d['source_commit'])))

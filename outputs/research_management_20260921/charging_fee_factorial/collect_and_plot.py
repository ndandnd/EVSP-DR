"""Verify collected endpoints, build editable paired tables, and plot incumbents."""
from pathlib import Path
import csv,hashlib,json,math
P=Path(__file__).resolve().parent;N=P/'native';M=json.loads((P/'manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def csvout(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (P/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
rows=[];missing=[];bad=[];receipts={}
for case in M['cases']:
    candidates=[]
    paths=list((N/'results'/case['case_id']).glob('*/receipt.json'))
    repair=P/'postprocess_repair'/case['case_id']/'receipt.json'
    if repair.exists():paths.append(repair)
    for path in paths:
        r=read(path)
        if 'ended_epoch' in r:candidates.append((r['ended_epoch'],path,r))
    if not candidates:missing.append(case['case_id']);continue
    _,path,r=max(candidates);out=path.parent
    assert r['case']==case and r['manifest_sha256']==sha(P/'manifest.json')
    assert r['source_execution_commit']==read(P/'code_receipt.json')['commit']
    assert all(sha(out/name)==digest for name,digest in r['artifact_sha256'].items())
    solver_out=Path(r.get('solver_artifacts_directory',out))
    if r.get('witness_repaired'):
        assert sha(Path(r['raw_receipt_path']))==r['raw_receipt_sha256']
        repair_report=read(out/'repair_report.json')
        assert sha(Path(repair_report['raw_result_path']))==repair_report['raw_result_sha256']
        assert repair_report['validation_thresholds_unchanged'] and not repair_report['solver_rerun']
    result=read(out/f"{case['arm']}.json");v=read(out/'validation.json')
    assert sha(out/f"{case['arm']}.json")==r['result_sha256'] and sha(out/'validation.json')==r['validation_sha256']
    if result['solutions']:
        assert r['physical_validation'] and v['valid']
        assert v['fleet']==5 and v['exactly_once_trips']==62
    else:bad.append(dict(case_id=case['case_id'],status=result['status'],reason=v['reason']))
    objective,bound=result['objective'],result['bound']
    if v['valid']:
        objective=v['electricity']+case['fee']*v['starts']
        assert abs(objective-result['objective'])<1e-4
        assert bound is None or bound-objective<1e-4
    row=dict(case_id=case['case_id'],fee_pair_id=case['fee_pair_id'],assignment=case['arm'],tariff_peak=case['peak'],fee=case['fee'],
        validated=v['valid'],solver_status=result['status'],solution_count=result['solutions'],runtime_seconds=result['wall_seconds'],
        witness_repaired=r.get('witness_repaired',False),raw_solver_objective=result.get('solver_objective_before_repair',result['objective']),
        electricity_cost=v.get('electricity'),charging_starts=v.get('starts'),charged_kwh=v.get('charged_kwh'),terminal_kwh=v.get('terminal_kwh'),
        objective=objective,bound=bound,relative_gap=max(0.0,(objective-bound)/abs(objective)) if objective is not None and bound is not None else None,
        numerical_bound_over_witness=max(0.0,bound-objective) if objective is not None and bound is not None else None,
        source_execution_commit=r['source_execution_commit'],fee_pair_identity_sha256=r['fee_pair_identity_sha256'],
        result_path=str((out/f"{case['arm']}.json").resolve()),gurobi_log=str((solver_out/f"{case['arm']}.gurobi.log").resolve()),
        model_path=str((solver_out/f"{case['arm']}.lp").resolve()),validation_path=str((out/'validation.json').resolve()),
        receipt_path=str(path.resolve()),result_sha256=r['result_sha256'])
    rows.append(row);receipts[case['case_id']]=r
pairs=[]
for arm in ['original','saved_joint_fee0','saved_joint_fee5']:
    for peak in [8,12,18]:
        selected=[r for r in rows if r['assignment']==arm and r['tariff_peak']==peak]
        if len(selected)!=2 or not all(r['validated'] for r in selected):continue
        selected.sort(key=lambda r:r['fee']);a,b=selected
        assert a['fee']==0 and b['fee']==5 and a['fee_pair_identity_sha256']==b['fee_pair_identity_sha256']
        # For any feasible dispatch: E>=L0 and E+5*N>=L5. Applying these
        # inequalities to the two optima separates their possible start counts.
        numeric_margin=.01
        n0_lower=math.ceil((b['bound']-a['objective']-numeric_margin)/5)
        n5_upper=math.floor((b['objective']-a['bound']+numeric_margin)/5)
        pairs.append(dict(fee_pair_id=a['fee_pair_id'],assignment=arm,tariff_peak=peak,
            fee0_starts=a['charging_starts'],fee5_starts=b['charging_starts'],starts_change=b['charging_starts']-a['charging_starts'],
            fee0_electricity=a['electricity_cost'],fee5_electricity=b['electricity_cost'],electricity_change=b['electricity_cost']-a['electricity_cost'],
            fee0_charged_kwh=a['charged_kwh'],fee5_charged_kwh=b['charged_kwh'],charged_kwh_change=b['charged_kwh']-a['charged_kwh'],
            fee0_objective=a['objective'],fee0_bound=a['bound'],fee0_gap=a['relative_gap'],
            fee5_objective=b['objective'],fee5_bound=b['bound'],fee5_gap=b['relative_gap'],
            fee0_schedule_repriced_at_fee5=a['electricity_cost']+5*a['charging_starts'],
            fee5_objective_improvement_over_fee0_schedule=a['electricity_cost']+5*a['charging_starts']-b['objective'],
            fee0_optimal_starts_lower_bound=n0_lower,fee5_optimal_starts_upper_bound=n5_upper,
            strict_reduction_for_all_restricted_model_optima=n0_lower>n5_upper,
            bound_derivation_numeric_cost_margin=numeric_margin,
            fee0_status=a['solver_status'],fee5_status=b['solver_status'],fee_pair_identity_sha256=a['fee_pair_identity_sha256']))
csvout('cell_results.csv',rows);csvout('paired_fee_results.csv',pairs)
summary=dict(cases_completed=len(rows),validated_incumbents=sum(r['validated'] for r in rows),pairs_verified=len(pairs),
    missing=missing,no_incumbents=bad,all18validated=len(rows)==18 and all(r['validated'] for r in rows),
    all_pairs_same_fixed_problem=len(pairs)==9,rows=rows,pairs=pairs,
    scope=M['scope'],limitations=M['limitations'],manifest_sha256=sha(P/'manifest.json'),source_commit=read(P/'code_receipt.json')['commit'])
(P/'results_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if pairs:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axs=plt.subplots(2,3,figsize=(10.5,6),layout='constrained',sharex=True)
    colors={8:'#156c9e',12:'#cb6727',18:'#588234'}
    markers={8:'o',12:'s',18:'^'}
    labels={'original':'Original trip assignment','saved_joint_fee0':'Fee-0-derived trip assignment','saved_joint_fee5':'Fee-5-derived trip assignment'}
    for col,arm in enumerate(labels):
        axs[0,col].set_title(labels[arm],loc='left',fontsize=11)
        for p in pairs:
            if p['assignment']!=arm:continue
            axs[0,col].plot([0,5],[p['fee0_starts'],p['fee5_starts']],color=colors[p['tariff_peak']],marker=markers[p['tariff_peak']],lw=1.8,ms=5)
            axs[1,col].plot([0,5],[p['fee0_electricity'],p['fee5_electricity']],color=colors[p['tariff_peak']],marker=markers[p['tariff_peak']],lw=1.8,ms=5)
        for ax in axs[:,col]:
            ax.grid(axis='y',alpha=.18);ax.set_xticks([0,5]);ax.set_xlim(-.5,5.5)
        axs[1,col].set_xlabel('Charge-start fee')
    axs[0,0].set_ylabel('Charging starts · five buses')
    axs[1,0].set_ylabel('Electricity cost · synthetic units')
    for row in axs:
        lo=min(a.get_ylim()[0] for a in row);hi=max(a.get_ylim()[1] for a in row)
        for a in row:a.set_ylim(lo,hi)
    fig.legend(handles=[Line2D([0],[0],color=colors[h],marker=markers[h],label=f'Peak {h:02}:00') for h in colors],
        loc='outside lower center',ncol=3,frameon=False)
    for ext in ['png','pdf','svg']:fig.savefig(P/f'controlled_start_fee.{ext}',dpi=220)
    plt.close(fig)
    available_tariffs=', '.join(f'{h:02}:00' for h in sorted({p['tariff_peak'] for p in pairs}))
    caption=(f'Controlled start-fee comparisons: {len(pairs)} of 9 planned pairs completed and physically validated, '
             f'covering tariff peaks at {available_tariffs}. '
             + ('All 18 cells are represented. ' if len(pairs)==9 else 'Partial snapshot; missing pairs are listed in results_summary.json. ')
             + 'Within each line, only the start fee changes; trip sequences, recovered station paths, charging physics, terminal floors and 600-second solver budget are identical. Points are validated feasible witnesses; cell_results.csv retains solver status, lower bounds and witness gaps. One optional charge per gap, confined to one tariff hour. This isolates the fee effect within the restricted fixed-path model; it is not fresh CG or an unrestricted routing optimum. Original means the original trip assignment with recovered station paths, not the recorded GIRO charging schedule.')
    if any(r['witness_repaired'] for r in rows):
        caption+=' One original/08:00/fee-5 witness includes a documented 6 ms charging extension after raw replay failed a terminal floor by 0.00002816 kWh; its recomputed cost is the feasible upper bound, and its original solver lower bound is retained.'
    (P/'figure_caption.txt').write_text(caption+'\n')
print(json.dumps({k:summary[k] for k in ['cases_completed','validated_incumbents','pairs_verified','missing','no_incumbents','all18validated']}))

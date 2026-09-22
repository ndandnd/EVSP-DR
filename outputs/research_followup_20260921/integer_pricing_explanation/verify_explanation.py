from pathlib import Path
import collections,csv,datetime,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
A=ROOT/'outputs/research_management_20260921/monitor_20260921T235438Z/integer_audit'
E=ROOT/'outputs/week_20260921/evidence'
W=ROOT/'.codex-work/integer-columns-20260921'
analysis=json.loads((E/'witness_analysis.json').read_text())['cases']['c1_k08']
cg=json.loads((E/'k8_sources/c1_k08/fresh_cg.json').read_text())
lp=cg['final_lp']; pi={int(k):v for k,v in lp['trip_duals'].items()}; witness=analysis['witness_routes']
cover=collections.Counter(t for r in witness for t in r['trips'])
assert cg['certified_rc_optimal'] and set(cover)==set(cg['trip_ids'])
assert analysis['physics_match_warm'] and analysis['provenance_match_warm']
rc=[r['cost']-sum(pi[t] for t in r['trips']) for r in witness]
assert all(abs(a-r['rc_fresh_duals'])<1e-7 for a,r in zip(rc,witness))
w=sum(r['cost'] for r in witness); surplus=sum(pi[t]*(n-1) for t,n in cover.items())
residual=w-lp['objective']-sum(rc)-surplus
assert abs(residual)<1e-6
rows=list(csv.DictReader((A/'verified_results.csv').open()))
treat=[r for r in rows if r['arm']=='treatment'];control=[r for r in rows if r['arm']=='control']
assert len(rows)==16 and len(treat)==8 and len(control)==8
assert sum(r['target8']=='True' for r in treat)==7 and sum(r['target8']=='True' for r in control)==0
proof=json.loads((A/'proof_log_lines.json').read_text())
for p in proof:
    lines=(A/p['log']).read_text().splitlines()
    for record in p['stage_summaries']+p['accepted_start_lines']:
        assert lines[record['line']-1]==record['text']
dp=A/'results/c1_k08/treatment_s20260921/668457_r0/dive/manifest.json'
dive=json.loads(dp.read_text())['dive']; nodes=dive['node_outcomes']
assert nodes[4]['depth']==nodes[5]['depth']==4
assert nodes[4]['fixed_routes'][:3]==nodes[5]['fixed_routes'][:3]
assert nodes[4]['fixed_routes'][3]!=nodes[5]['fixed_routes'][3]
summary={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'c1_witness':{'trips':len(pi),'pool_columns':cg['columns'],'route_weight':lp['route_weight'],'lp_weighted_objective':lp['objective'],'lp_positive_routes':len(lp['positive_routes']),'all_positive_routes_fractional':all(0<r['value']<1 for r in lp['positive_routes']),'witness_routes':len(witness),'witness_weighted_cost':w,'witness_gap':w-lp['objective'],'reduced_costs':rc,'reduced_cost_sum':sum(rc),'coverage_surplus_dual_value':surplus,'duplicate_trip_ids':[t for t,n in cover.items() if n>1],'identity_residual':residual},'replication':{'paired_case_seeds':8,'distinct_cases':4,'seeds_per_case':2,'treatment_target8':7,'control_target8':0,'proof_log_lines_checked':sum(len(p['stage_summaries'])+len(p['accepted_start_lines']) for p in proof)},'c1_dive_nodes':[{k:n[k] for k in ['node_id','depth','outcome','node_stop','artificial_total','columns_added','penalized_pricing_closed']} for n in nodes]}
(OUT/'verified_numbers.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Paired k8 results and full proof logs','', 'Fleet / bound refers to the supplied finite pool. Each link points to the fleet-stage summary in a complete log.','', '| Case | Seed | Control fleet / bound | Treatment fleet / bound | New records | Treatment charged s | Treatment elapsed min |','|---|---:|---|---|---:|---:|---:|']
for t in treat:
    c=next(r for r in control if (r['case'],r['seed'])==(t['case'],t['seed']))
    def link(r):
        p=next(p for p in proof if (p['case'],p['arm'],p['seed'])==(r['case'],r['arm'],r['seed']))
        return f"[{r['buses']} / {float(r['fleet_bound']):g}{' proved' if r['finite_pool_fleet_proven']=='True' else ' open'}]({A/p['log']}:{p['stage_summaries'][0]['line']})"
    lines.append(f"| {t['case']} | {t['seed']} | {link(c)} | {link(t)} | {t['new_columns']} | {float(t['charged_s']):.2f} | {float(t['end_to_end_s'])/60:.2f} |")
(OUT/'proof_links.md').write_text('\n'.join(lines)+'\n')
paths=[E/'witness_analysis.json',E/'k8_sources/c1_k08/fresh_cg.json',A/'manifest.json',A/'verified_results.csv',A/'audit_summary.json',A/'proof_log_lines.json',A/'remote_provenance_checks.json',dp,W/'src/diving_pricing_pilot.py',W/'src/run_exact_pool_mip.py',W/'src/config.py',W/'scripts/research/diving_pricing_20260919/run_replication.py']
paths += [A/p['log'] for p in proof]
source={'execution_commit':json.loads((A/'manifest.json').read_text())['execution_commit'],'local_worktree_head':subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip(),'sources':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
(OUT/'source_hashes.json').write_text(json.dumps(source,indent=2)+'\n')
print(json.dumps(summary['c1_witness'],indent=2));print(summary['replication']);print('Verified actual C1 backtrack at depth4, nodes5→6.')

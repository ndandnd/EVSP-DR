"""Reconstruct numerical lower bounds from paired pricing-iteration records.
Does not claim exact arithmetic certification; retains archival dual availability.
"""
from pathlib import Path
import csv,json,math,hashlib
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
original=ROOT/'outputs/overnight_next_20260914/status_20260916T194843Z/all_chain_extension_results.csv'
rows=list(csv.DictReader(original.open())); raw=json.loads((BASE/'sources.json').read_text()); sources={r['case']:r for r in raw}
M=100000.;fee=5.;horizon=1560.;numerical_guard=1.
for r in rows:
 s=sources[r['case_id']];p=s['last_pricing_iteration'];fl=s['final_lp_scalars'];n=s['n'];a=s['provenance']['args']
 assert s['input_sha256']==r['input_sha256'] and s['provenance']['instance_sha256']==r['input_sha256']
 assert s['provenance']['git_commit']=='a0e0bb7681c8451e3cbbbfa06aef390026d9af4b'
 assert s['physics']['master_sense']=='cover' and s['physics']['charge_kw']==240 and p['artificials']==0
 assert a['diversify_rounds']==0 and s['physics']['time_model']=='event'
 # Every event route occupies <=26h, charges at240kW and has <=one charge
 # between successive trips (plus a conservative extra start). Ignoring all
 # deadhead gives an upper bound on trips/route via interval scheduling.
 qmax=s['physics']['charge_kw']*horizon/60*s['max_price']+fee*(s['max_nonoverlapping_trips']+1)
 incumbent_upper=float(r['integer_buses'])*(M+qmax)
 K=incumbent_upper/M
 matched=fl['iteration']==p['iter'] and abs(fl['objective']-p['lp_obj'])<=1e-6
 # Sum of archived final duals usable ONLY when it is the priced iterate.
 # Otherwise retain paired RMP optimal objective; strong duality is invoked
 # from the successful LP solve, not from a mismatched final re-solve.
 D=s['dual_sum'] if matched else p['lp_obj']
 delta=min(0.,p['min_rc'])
 weighted_lb=D+K*delta-numerical_guard
 fleet_fraction_lb=weighted_lb/(M+qmax)
 fleet_lb=math.ceil(fleet_fraction_lb-1e-9)
 r.update(priced_iteration=p['iter'],priced_rmp_objective=p['lp_obj'],priced_min_reduced_cost=p['min_rc'],priced_duals_archived_and_matched=matched,priced_dual_objective=s['dual_sum'] if matched else '',dual_objective_basis='archived_matching_duals' if matched else 'paired_optimal_RMP_objective_strong_duality',max_nonoverlapping_trips_per_route=s['max_nonoverlapping_trips'],route_nonfleet_cost_upper_bound=qmax,incumbent_weighted_cost_upper_bound=incumbent_upper,route_mass_upper_bound=K,lagrangian_weighted_lower_bound=weighted_lb,fleet_relaxation_lower_bound_from_cost_envelope=fleet_fraction_lb,integer_fleet_lower_bound=fleet_lb,numerical_objective_guard=numerical_guard,bound_scope='pinned_event_route_space; numerical_LP_and_pricing',bound_exact_arithmetic_certificate=False,bound_source_sha256=s['cg_sha256'],bound_validity='numerical_bound_reconstructed; no_rational_certificate')
 assert fleet_lb<=int(r['integer_buses'])
with (BASE/'all_chain_extension_results_with_bounds.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
u=[r for r in rows if r['cg_pricing_certificate']=='False']
summary={'cases':len(rows),'certified_cg':len(rows)-len(u),'uncertified_cg':len(u),'uncertified_matching_archived_duals':sum(r['priced_duals_archived_and_matched'] for r in u),'uncertified_fleet_bound_equals_rounded_route_weight':sum(r['integer_fleet_lower_bound']==round(float(r['fractional_route_weight'])) for r in u),'all_fleet_bound_equals_rounded_route_weight':sum(r['integer_fleet_lower_bound']==round(float(r['fractional_route_weight'])) for r in rows),'max_qmax':max(r['route_nonfleet_cost_upper_bound'] for r in rows),'max_primal_archived_dual_difference':max(abs(s['dual_sum']-s['final_lp_scalars']['objective']) for s in raw),'k32':[{k:r[k] for k in ('case_id','integer_buses','integer_fleet_lower_bound','priced_duals_archived_and_matched','fleet_relaxation_lower_bound_from_cost_envelope')} for r in rows if r['target_buses']=='32'],'source_table_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'source_extract_sha256':hashlib.sha256((BASE/'sources.json').read_bytes()).hexdigest()}
(BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

"""Render the registered cell-level public aggregates plus a compact, human-readable report."""
import argparse, base64, csv, hashlib, json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--envelope',required=True,type=Path);p.add_argument('--public',required=True,type=Path);p.add_argument('--private',required=True,type=Path);a=p.parse_args();a.public.mkdir(parents=True,exist_ok=True);a.private.mkdir(parents=True,exist_ok=True)
payload=json.loads(a.envelope.read_bytes());aggregate=payload['public'];arms=aggregate['observed_subsets']
for name,b in payload['private_small_files_base64'].items():
    if not Path(name).is_relative_to('.') or '/' in name:raise ValueError('invalid receipt name')
    (a.private/name).write_bytes(base64.b64decode(b,validate=True))
(a.public/'aggregate.json').write_text(json.dumps(aggregate,indent=2,sort_keys=True)+'\n')
rows=[]
for s in arms:
    final=s['terminal_verified'];ac=s['accounting'];pair=s['root'];sel=s.get('selection_replay',{});fl=s.get('fleet_stage',{});ch=s.get('charging_stage') or {};cl=s.get('charging_stage_log') or {};d=s.get('dive',{});rss=ac['batch_peak_rss_kib']
    rows.append(dict(chain=s['case'].split('_')[0].upper(),grid_kwh=s['soc_step_kwh'],terminal_verified=final,trip_count=s.get('trip_count'),simultaneous_floor=s.get('peak_simultaneous_passenger_trips'),job_id=ac['job_id'],state=ac['state'],cg_certified=pair['cg_pricing_certified'],root_weighted_lb=pair['weighted_grid_lp_lower_bound'],root_weight=pair['route_weight'],root_rmp=pair['rmp_objective'],min_rc=pair['min_reduced_cost'],certificate_tolerance=pair['effective_tolerance'],graph_arcs=s.get('packed_arcs'),graph_s=s.get('graph_seconds'),root_cg_s=s['root_cg_seconds'],root_pricing_s=s['root_pricing_seconds'],root_restricted_lp_s=s['root_lp_seconds'],root_columns=s['root_columns'],dive_state=d.get('state'),total_pool_columns=sel.get('final_pool_rows'),fleet=sel.get('fleet'),pool_fleet_lb=fl.get('bound'),fleet_status=fl.get('status'),fleet_stage_s=fl.get('seconds'),grid_nonfleet=sel.get('grid_nonfleet_cost'),pool_charging_lb=ch.get('bound'),charging_status=ch.get('status'),charging_optimize_s=cl.get('optimize_seconds'),continuous_nonfleet=sel.get('continuous_nonfleet_cost'),extra_trip_visits=sel.get('extra_trip_visits'),exactly_once=sel.get('exactly_once'),minute_plug_overloads=sel.get('shared_plug_overloads'),job_elapsed_s=ac['elapsed_s'],batch_peak_rss_gib=rss/1048576 if rss is not None else None,replay_rerun_matched=s.get('replay_reproduced')))
with (a.public/'paired.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def f(v,dec=0):return '—' if v is None else f'{v:,.{dec}f}'
def t(v):return '—' if v is None else f'{v/60:.1f}'
text=['# Partille k8 SOC-grid pilot','',f"Collected {aggregate['verified_at_utc']}. The 2.5 kWh rows are the registered October v15 controls. The 5 kWh rows are November array 917989. The three chain inputs, flat tariff, 60 kW PARX depot, 240 kW opportunity chargers, 240 kWh battery, per-start fee of 5, one-minute plug limits and fresh covering search match their respective control.",'','## Search and resources','','| Chain | SOC kWh | Job state | Arcs, millions | Graph min | Root CG min | Root columns | Post-dive columns | Job min | Peak GiB |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    text.append(f"| {r['chain']} | {r['grid_kwh']} | {r['state']} | {f(r['graph_arcs']/1e6 if r['graph_arcs'] else None,2)} | {t(r['graph_s'])} | {t(r['root_cg_s'])} | {f(r['root_columns'])} | {f(r['total_pool_columns'])} | {t(r['job_elapsed_s'])} | {f(r['batch_peak_rss_gib'],2)} |")
text+=['','Jobs had eight CPUs, 48 GiB RAM, and a 20-hour limit. Each job built a fresh graph. The root and simple-dive LP master used one solver thread. Finite-pool MIPs could use eight. The registered budgets were 180 min for root CG, 90 min for the simple dive and 60 min shared by the fleet and charging MIPs. Memory is peak resident memory for the batch step. Separate per-phase memory measurements are unavailable. The runs used different CPU models. The treatment jobs also include terminal replay. Wall times describe these attempts.','', '## Pricing certificates and integer schedules','','| Chain | SOC kWh | Weighted grid LP lower bound | Route weight | Fleet / pool bound | Extra visits | Grid cost + fees | Continuous cost + fees | Finite-pool charging lower bound |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    text.append(f"| {r['chain']} | {r['grid_kwh']} | {f(r['root_weighted_lb'],6)} | {f(r['root_weight'],4)} | {f(r['fleet'])} / {f(r['pool_fleet_lb'])} | {f(r['extra_trip_visits'])} | {f(r['grid_nonfleet'],3)} | {f(r['continuous_nonfleet'],3)} | {f(r['pool_charging_lb'],3)} |")
text+=['','The root lower bound combines the 100,000 bus coefficient with grid charging cost and start fees. It uses the minimum reduced cost from full conservative pricing and the registered fleet cap. The effective reduced-cost tolerance is shown per row in `paired.csv`. Each charging-stage bound applies to a fixed pool with its corresponding fleet cap and grid. Continuous costs come from replay of the selected schedules. A covering selection can visit a passenger trip multiple times. Every terminal selection passed independent route and shared-plug replay.','', '## Paired comparisons','']
for c in ['C2','C3','C6']:
    b,h=[r for r in rows if r['chain']==c];
    if not h['terminal_verified']:
        text.append(f"- **{c}**. The 5 kWh row has a root certificate. Its terminal schedule was still pending at collection.");continue
    memdrop=100*(1-h['batch_peak_rss_gib']/b['batch_peak_rss_gib']); arcdrop=100*(1-h['graph_arcs']/b['graph_arcs']);gfac=100*(1-h['graph_s']/b['graph_s']);cfac=100*(1-h['root_cg_s']/b['root_cg_s']);msg=f"- **{c}**. The coarse graph had {arcdrop:.1f}% fewer arcs, graph construction took {gfac:.1f}% less time and root CG took {cfac:.1f}% less time. Batch peak memory fell {memdrop:.1f}%. The fleet changed from {b['fleet']} to {h['fleet']}, with {b['extra_trip_visits']} and {h['extra_trip_visits']} extra passenger-trip visits, respectively."
    if b['fleet']==h['fleet']:
        msg+=f" At that fleet the grid cost with fees changed by {h['grid_nonfleet']-b['grid_nonfleet']:+.3f}, and the continuous replay cost changed by {h['continuous_nonfleet']-b['continuous_nonfleet']:+.3f}."
    if h['exactly_once']:msg+=f" The coarse selection covered all {h['trip_count']} trips exactly once with zero plug overloads. Its fleet attains the independently counted {h['simultaneous_floor']}-trip overlap lower bound."
    text.append(msg)
text+=['','`aggregate.json` records hashes and solver status for each arm, including the original and replayed selection, root journal, post-dive pool, immutable source revision and verifier. The collector authenticates all six arms against the prelaunch manifest and requires byte-identical replay receipts for every terminal case.','']
(a.public/'README.md').write_text('\n'.join(text)); dest=a.public/'SHA256SUMS';dest.write_text(''.join(f'{hashlib.sha256(x.read_bytes()).hexdigest()}  {x.name}\n' for x in sorted(a.public.iterdir()) if x.is_file() and x!=dest))

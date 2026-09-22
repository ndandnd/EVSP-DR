from pathlib import Path
import csv,json,hashlib
P=Path(__file__).resolve().parent;rs=list(csv.DictReader((P/'mip_stage_times.csv').open()));summary=json.loads((P/'timing_summary.json').read_text());lp=json.loads((P/'lp_similarity.json').read_text());table=[]
for k in [5,8,10,15]:
 a=next(r for r in rs if r['chain']=='1' and int(r['target_k'])==k and r['arm']=='fresh');b=next(r for r in rs if r['chain']=='1' and int(r['target_k'])==k and r['arm']=='sequential')
 table.append('|'+str(k)+'|'+ '|'.join(f"{float(r[f]):.2f}" for r in [a,b] for f in ['fleet_optimize_s','charging_optimize_s','total_optimize_s'])+'|')
text='''# Chain comparisons with actual final-MIP timing

[Chain 1 charts](chain1_charts.png) · [Chain 2](chain2_charts.png) · [Chain 3](chain3_charts.png) · [Chain 4](chain4_charts.png) · [Chain 5](chain5_charts.png) · [Chain 6](chain6_charts.png) · [All chains](all_chains_charts.png)

Each chain has a **charts-only 13 × 6.2 inch PNG/PDF/SVG**, suitable for a 16:9 slide with a native editable title/caption outside the image. The six panels preserve cumulative CG work, integer excess and the precise weighted-LP comparison, then add actual fleet-search, charging-cost-search and total optimizer times. All time axes are logarithmic; 0.01 minute is 0.6 seconds. The dotted guides mark nominal fleet-search (30 min) and total MIP (60 min) allowances, not observations.

**Sequential cumulative CG includes ancestor CG work but excludes every earlier MIP. The MIP panels show only the final MIP at the displayed k.** These are not complete end-to-end ladder runtimes. Graph building, scheduler wait and physical preparation are excluded from the plotted optimizer times; graph and preparation fields remain in editable tables. Historical machines/code differ, so this is a descriptive comparison, not a controlled causal speedup estimate.

## Actual timing, not the one-hour allowance

The nominal MIP allowance was 3,600 seconds, with up to 1,800 seconds initially reserved for fleet search. If fleet search stops earlier, charging-cost search can use the unused time. It need not spend its allowance.

- **Fresh:** seven of 24 MIPs finished both stages to their configured optimality tolerance in under ten optimizer minutes: all six k5 cases (0.86–163.31 seconds) and C6 k8 (91.70 seconds). The other seventeen charging stages reached their time limit. Fleet search was optimal in10/24 and time-limited in14/24; final GIRO target attained in6/24.
- **Sequential:** fleet search proved its saved-pool fleet optimum in24/24, all at target k. Charging search was optimal in10/24 and time-limited in14/24. Seven finished in under ten total optimizer minutes; three more finished in about14–16minutes.
- Median fleet optimizer time was30.001minutes fresh versus0.0885minutes sequential (5.31seconds). Median total optimizer time was59.986 versus59.951minutes, because many sequential runs spent the remaining allowance improving charging cost after the fleet result was already settled. Across the24 endpoints per arm, total optimizer time summed to17.106hours fresh and14.781hours sequential; these sums omit all earlier sequential MIPs.

For C1, actual **optimizer-call seconds** were:

|Target k|Fresh fleet s|Fresh charging s|Fresh total s|Sequential fleet s|Sequential charging s|Sequential total s|
|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

C1 k5 therefore takes only3.46/2.92seconds of optimizer time, fresh/sequential—not an hour. At k8, the fresh pool proves nine buses in26.35optimizer minutes, while the sequential pool proves eight in27.15seconds; both then continue charging-cost search and use nearly an hour in total. At k10/k15, fresh fleet search reaches its limit with11/18buses and bounds10/15, while sequential proves10/15 within152.29/105.75optimizer seconds. Final MIP termination can be TIME_LIMIT despite a proven fleet optimum because charging search is a separate stage.

## What the recorded clocks mean

The plotted data use `gurobi_optimize_stage_wall_s[0:2]`: direct wall-clock timers surrounding the two `optimize_with_start_audit` calls. Their sum matches `gurobi_optimize_wall_s` in all48results. These include the audited solver call and its callback/log handling. They exclude pool preparation and unrelated pipeline time.

The source also retains `two_stage.stage1_runtime_s` and `runtime_s`, which are broader elapsed counters from the MIP phase start. They include inter-stage validation and postsolve work. They are **not** interchangeable with the direct optimizer timers, and `runtime_s − stage1_runtime_s` is not a pure charging optimizer time. Both sets of measurements remain in [mip_stage_times.csv](mip_stage_times.csv), alongside preparation, source hashing, stopping statuses, finite-pool bounds, nominal limits and charging gaps. Small limit overshoots are preserved. [Pinned implementation excerpts](timing_source_excerpt.txt) make this distinction auditable.

**No first-target timestamps are claimed.** All48pinned result payloads have no progress trace. The end of fleet search is proof/termination time, not necessarily the first instant a k-bus incumbent was found. `first_target_time_s` is therefore blank with an explicit reason; logs were not collected or parsed for this follow-up.

## LP, integer and physical scope

The weighted LP values remain numerically indistinguishable across the24pairs: maximum absolute difference0.0000010617synthetic cost units, maximum relative difference7.08×10⁻¹³; both sides have conservative event-grid pricing certificates at reduced-cost tolerance0.0001. The third panel subtracts the common100,000×k term solely to reveal the charging-related component. It is not a lower-bound gap, and equal LP values do not imply equal route supports or saved integer pools.

The integer panel shows buses above the GIRO target. Its faint fresh vertical segment descends to the finite-pool fleet bound; it is not an error bar. A proven extra bus is a result about that saved pool, not full-model integer impossibility. Historical physics remain240kWh/240kW, zero reserve, no shared charger-capacity or terminal-energy floor, flat tariff, start fee5, covering and a2.5kWh/5minute event grid. Duplicate service and omitted physical constraints remain the earlier comparison's qualifications.

## Reproducibility and artifacts

[48-row actual stage table](mip_stage_times.csv) · [24 paired rows retaining all earlier data](chain_comparison_with_mip.csv) · [Timing summary](timing_summary.json) · [LP precision](lp_similarity.json) · [Hashes and checks](provenance.json) · [Rebuild](build.py)

Standalone stage-table PNG/PDF/SVG files are `chain1_mip_table` through `chain6_mip_table`; native editable per-chain CSV/Markdown tables are also included. Every source JSON is checked against both the earlier paired-panel SHA256 and the pinned battery-replay source hash. Input identities, bus outcomes, allowances, stage statuses and exact optimize-time sums are checked. Source MIP commit: `871d057e1067411f09581e37d78f7c1ca43f68bb`. All six charts and the C1 table were visually inspected. Every predecessor figure and source record remains unchanged. No solver, cluster job, live Doc or Slides edit was performed.
'''
(P/'README.md').write_text(text)
caption='Sequential CG sums all ancestors through k; earlier MIPs are excluded. MIP times are measured final fleet and charging optimizer calls, not one-hour allowances or first-target times. Fleet proof and charging termination are separate. LP values coincide to numerical tolerance under the conservative event-grid model; saved integer pools can differ. Graph/queue/preparation costs and full operational feasibility are outside these plotted timings.'
(P/'figure_caption.txt').write_text(caption+'\n')
for chain in range(1,7):
 rows=[r for r in rs if int(r['chain'])==chain]
 fields=['target_k','arm','buses','fleet_bound','fleet_optimize_s','charging_optimize_s','total_optimize_s','fleet_status','charging_status']
 with (P/f'chain{chain}_mip_table.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 md=['|Target|Pool|Buses/bound|Fleet seconds|Charging seconds|Total seconds|Fleet status|Charging status|','|---:|---|---|---:|---:|---:|---|---|']
 for r in rows:md.append('|'+ '|'.join([r['target_k'],r['arm'],r['buses']+'/'+f"{float(r['fleet_bound']):.0f}",*[f"{float(r[k]):.2f}" for k in ['fleet_optimize_s','charging_optimize_s','total_optimize_s']],r['fleet_status'],r['charging_status']])+'|')
 (P/f'chain{chain}_mip_table.md').write_text('\n'.join(md)+'\n')
manifest=json.loads((P/'provenance.json').read_text());manifest['visual_qa']={'all_six_chain_pngs_inspected':True,'chain1_stage_table_inspected':True,'charts_only_dimensions_inches':[13,6.2],'long_caption_outside_image':'figure_caption.txt'};manifest['output_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.iterdir() if f.is_file() and f.name!='provenance.json'};(P/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')

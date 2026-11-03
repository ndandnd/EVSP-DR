# Does a 60 kW depot change the comparison with fixed passenger duties?

On 72 distinct scenarios where both fixed-visit schedules work at both depot rates, the joint schedule costs less on the same 38 scenarios at each rate. The median saving is 1.468 modeled cost units with a 350 kW depot and 1.194 with a 60 kW depot. Many more of the archived station-visit plans become infeasible when the depot slows. The lower overall number of wins partly reflects which schedules remain in the comparison.

## Fixed visits and comparable denominators

The October 3 retiming array 911126 contains 312 Gurobi result files, one for each of 104 campaign records × three arms. It used existing passenger sequences and station visits from the 350 kW runs, and reoptimized charging minutes and amounts. Each bus started full at 240 kWh with zero reserve and zero charging-start fee. The opportunity stations stayed at 350 kW. Finite plug counts were enforced minute by minute in the plug-constrained arm. Results count solver incumbents with optimal status. All other endpoints in this array are solver-proved infeasibility under the fixed visits.

| Feasible schedules on the original records | 350 kW depot | 60 kW depot |
|---|---:|---:|
| Fixed GIRO trip sequences, fixed visits, terminal target and plugs | 104/104 | 92/104 |
| Joint routing, fixed visits, terminal target and plugs | 103/104 | 76/104 |
| Fixed GIRO trip sequences, fixed visits, no terminal target, no plug limits | — | 92/104 |
| Joint routing, fixed visits, terminal target, no plug limits | — | 77/104 |

The same 12 fixed-visit records are infeasible after both the terminal target and shared-plug rows are removed. That result concerns their selected visits. A model that holds the passenger assignments fixed but allows the bus to choose other charging stations answers a separate question.

| Comparable population | 350 kW: joint cheaper / feasible · median saving | 60 kW: joint cheaper / feasible · median saving |
|---|---:|---:|
| Feasible at each rate, all campaign records | 64/103 · 4.941 | 38/74 · 0.534 |
| Feasible at each rate, distinct scenarios | 64/101 · 5.102 | 38/72 · 1.194 |
| Feasible at both rates, all campaign records | 38/74 · 0.715 | 38/74 · 0.534 |
| Feasible at both rates, distinct scenarios | 38/72 · 1.468 | 38/72 · 1.194 |

A saving is the retimed fixed plan’s cost minus the retimed joint plan’s cost. A win requires a difference above 0.000001. The k5 and expansion campaigns both include loc_flat and r21_flat. Each repeated record has identical passenger input, flat tariff, fixed passenger assignment, terminal target, and fixed and joint source schedule bytes. The October summary implicitly reported all campaign records for 350 kW and distinct scenarios for 60 kW. The last row compares the same 72 scenarios at both rates. All 38 winning scenarios are the same, and the other 34 stay tied or favor the fixed plan. These results do not measure the performance of a newly optimized station-selection policy. [Input identity hashes](duplicate_inputs.json) · [Per-scenario statuses and costs](aggregate.json)

## Fresh routes under the two depot rates

The separate 34-arm v15 comparison used the same code revision 3a0ab6a8 and pairwise-matched inputs in array 920164. Every run started fresh and used a trip-covering master. The five route-21 k5 spatial cases have 350 kW opportunity chargers and a 280.7833253 kWh aggregate end target. All return five buses at both depot rates. Four have the same certified conservative charging LP component. The half-price depot case rises from 151.806 to 172.825 (+13.85%) · its five-bus pool charging cost rises from 151.90 to 172.86. The LP component removes the constant 100,000 per bus at fractional route weight five.

The six chain k5 and six chain k8 pairs use 240 kW opportunity chargers, zero end target, a flat tariff and a five-unit charging start fee. All six k5 pairs return five buses. At k8, the pool covering selections in chain order C1 through C6 are 10, 9, 8, 11, 9, 8 at a uniform 240 kW, and no incumbent, 9, 8, 11, 11, 8 at PARX60. C1’s time-limited fleet stage at PARX60 provides no infeasibility proof. Only C3 and C6 return eight at both rates. The new November work on exact dispatch uses separate replay and proof. Eight of 12 conservative chain roots certified at each rate. All five conservative spatial roots certified at each rate. A selected fleet count from this October covering panel alone does not establish an exactly-once dispatch or the fleet minimum. [34 arm summaries with input and receipt hashes](aggregate.json)

## Reproduction and search scope

Run `python3 compare.py /path/to/native/campaigns` against the archived campaign root. It needs the 350 kW baseline CSV in plug_retiming_20261001, the 312 retiming receipts and task manifest in plug_retiming_parx60_20261002, and 34 completed v15 task results under capacity_v15_20261002. It checks the two repeated cases against the supplied input hash receipt, discards the incomplete earlier C4 restart, and writes aggregate.json. The aggregate excludes passenger itineraries, charging events and source account paths. [Comparison code](compare.py)

As of the start of the 3 November audit, search of the October and November campaign directories and task manifests, Slurm accounting, local research register and relevant worktrees located no completed follow-up which reran all 12 failed cells with freely reselected stations on identical fixed passenger duties. The five route-21 v15 tariff pairs allow routes and visits to change. The 62-trip November R21-A transfer study uses the same passenger input as R21-A’s October runs but mandatory recovered station paths, a different 08:00 tariff, taper and per-duty floors. Neither result resolves the 12 individually. A separately dated witness using different admissible station visits would establish feasibility for the tested cells without certifying minimum charging cost.

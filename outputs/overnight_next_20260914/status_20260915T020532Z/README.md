# Current research — 14 September, 22:13 EDT

**Chain 3 now matches 27 buses.** Fleet proof took 23.8 minutes in a pool of 156,516 columns, after 118.3 minutes of CG. The full two-stage MIP took 60.2 minutes; charging search reached its time limit. Individual routes pass replay. The covering solution has 57 duplicated trip assignments; removing those assignments has not been separately validated. Shared charger capacity and a terminal energy floor are absent from this baseline.

The compact-seed campaign now has **33 certified CGs and 28 completed MIPs, all matching target**. Both methods match every k=8 and k=10 case. At k=15, both methods are complete for chains 5 and 6; all four use 15 buses. Chain 1, 2 and 4 CGs with 512 inherited routes certified in 147.6, 159.6 and 132.3 minutes, respectively. Eight k=15 MIP results remain unpublished.

The reserve screen has seven completed cases. All seven CGs certified; five integer results recover one bus, while duty 13405 needs two buses in both tested pools. Three cases have no published endpoint yet. In the table, baseline means PARX charging at 240 kW; parx60 changes only PARX to 60 kW. Pool fleet proof means that no smaller fleet exists among the saved columns.

| Duty | Charging option | CG minutes | Fractional route weight | Integer buses | Pool fleet proved |
|---|---|---:|---:|---:|---|
| 13405 | baseline | 6.02 | 1.09 | 2 | yes |
| 13405 | parx60 | 8.64 | 1.09 | 2 | yes |
| 13406 | baseline | 8.12 | 1.00 | 1 | yes |
| 13406 | parx60 | 11.11 | 1.00 | 1 | yes |
| 13407 | baseline | 15.33 | 1.00 | 1 | yes |
| 13407 | parx60 | pending | pending | pending | pending |
| 13408 | baseline | 2.17 | 1.00 | 1 | yes |
| 13408 | parx60 | 3.05 | 1.00 | 1 | yes |
| 13408 | capacity | pending | pending | pending | pending |
| 13408 | combined | pending | pending | pending | pending |


These tests use 236.44 kWh batteries, a 35.466 kWh reserve, constant-rate charging and no assumed 65% terminal target. The seven completed cases do not enforce shared station capacity, but each selected schedule passes the documented station-count audit afterward. Individual route feasibility follows from the dedicated driver's construction, not an independent continuous replay. The two tests that enforce capacity remain unpublished.

Duty 13405's certified weighted LP objective is 109177.584 (baseline) or 109177.606545 (PARX at 60 kW), with fractional route weight 1.090909. The discrepancy is already present at the weighted LP optimum. Fractional route weight is not a separately optimized fleet lower bound. The finite-pool proof does not, by itself, prove that every possible one-bus route is infeasible; that requires a separate fleet or feasibility argument.

The original chain 5 k=25 MIP now has 26 buses with bound 25 and an open fleet gap. This was the late scheduler completion noted in the prior snapshot. Original-extension totals are 59 CG and 59 MIP endpoints, with 45 CG certificates, 34 target matches and 25 integer misses. Across the original and continuation campaigns, the largest individual one-hour matches on chains 1–6 are **24, 22, 27, 23, 24 and 26**. These maxima do not imply uninterrupted success at every smaller size.

The nine LP-support-only results are unchanged from the launch update. There is no new confirmed preemption or MIP execution failure. The queue observation has **82 running jobs and 49 waiting for required inputs**, excluding 33 held historical tasks. The preemption study contains 902 attempt records. This check submitted and requeued no jobs.

[Reserve values and source hashes](reserve_results.csv) · [Compact-seed results](../../overnight_evening_20260914/status_20260915T020532Z/README.md) · [Original chain and cumulative-budget tables](../../cumulative_budget_20260913/status_20260915T020532Z/README.md) · [Source checks](validation.json).

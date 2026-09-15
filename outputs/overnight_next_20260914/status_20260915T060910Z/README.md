# Research results — 15 September, 02:17 EDT

**The smaller-start comparison is complete: 33 of 36 integer results match target.** All24 k8/k10 cases match. At k15, nine of twelve match:

| Chain | Core buses | Expanded-start buses | Core CG minutes | Expanded CG minutes |
|---|---:|---:|---:|---:|
| 1 | 16 | 16 | 168.4 | 147.6 |
| 2 | 15 | 15 | 161.2 | 159.6 |
| 3 | 17 | 15 | 50.9 | 46.5 |
| 4 | 15 | 15 | 166.5 | 132.3 |
| 5 | 15 | 15 | 99.4 | 100.2 |
| 6 | 15 | 15 | 98.5 | 77.1 |

All36 CGs have pricing certificates. All integer fleet counts are proved within their saved pools except chain3's core: it found17, with bound15, so that gap remains open. “Core” retains previous integer routes and all positive-weight LP routes; the expanded start fills the core to512 distinct trip sequences. [Every result and proof field](../../overnight_evening_20260914/status_20260915T060910Z/compact_seed_results.csv).

**Chain1 k15 now isolates a missing-column problem.** Both compact pools prove a minimum of16 buses. The earlier full-pool run finds15. The input/reference/price/deadhead hashes, recorded CG revision, model and pricing settings match, and all three weighted LP objectives agree within0.000001 at1,500,717.373326. Thus LP convergence did not preserve an integer-optimal column set. More MIP time cannot find15 in either compact pool. Individual-route replay passes; duplicate-removal and shared-capacity validation remain separate. [Matched-model source audit](c1_k15_pool_limitation_audit.json).

**All13 column-addition pairs are complete, and none reaches target.** Each pair adds either donor routes with positive final LP weight or the same number of donor routes with zero final LP weight. Both arms find the same fleet in every pair:9 for target8 or11 for target10. Twenty-five of the26 arms prove those above-target minima. The C2k8 zero-weight arm has9 buses with bound8, so its target remains unresolved. All nine donor-support-only controls also prove above-target fleets. This is evidence against these particular selection rules on the selected difficult pools, not a universal claim about all such routes. Zero LP weight is not zero reduced cost. Constructed pools have no new CG certificate. [All35 outcomes and source hashes](lp_addition_results.csv).

**Larger compact starts have three early integer successes.**

| Case | Starting sequences | Final columns | Integer buses / pool bound | Total MIP minutes |
|---|---:|---:|---|---:|
| C1 k25 core | 417 | 30,335 | 25 /25, proved | 5.7 |
| C1 k25 expanded | 512 | 28,540 | 25 /25, proved | 6.2 |
| C4 k20 core | 356 | 68,275 | 20 /20, proved | 11.9 |

These three also prove their charging-related objective within the saved pool and pass individual-route replay. Their CGs reached the four-hour limit without pricing certificates. Only three of24 MIPs are published, so early completions are not an overall success rate. Of20 completed CGs, three certify and17 reach their limits; four remain pending. This cohort uses a0e0 CG code and different immediate parent pools, while the smaller cohort uses e091. Do not interpret their difference as a pure size effect. [All24 cases](compact_large_results.csv).

**The larger chains continue.** C4 and C5 now match26 buses in their one-hour MIPs, with pool fleet proofs. C3 at target28 finds29, with bound28; neither its integer optimum nor its CG optimum is established. The new largest individual one-hour matches across chains1–6 are25,26,27,26,26,27. These maxima do not imply uninterrupted success at every smaller k. Baseline physics remain covering,240kWh/240kW and charging-start fee5, with no shared-capacity or minimum-ending-SOC constraint.

**Useful new work:** job220545 is running a longer MIP on the original C5k25 pool, which found26 buses with bound25. Preparation verified no duplicate longer attempt, matched all non-time settings, and froze the same207,717 columns. The full-size license check passed. It uses default partition,8CPUs/24GB,12600seconds total and10800seconds maximum fleet search, excluding scaglione-compute-01. This is a new search tree, not a resumed Gurobi tree. [Submission and matched-setting audit](../../final_chain_gap_20260915/README.md). Its launch is separately dated02:13–02:14EDT and is not a new scientific result.

Queue in the full collection:29 running /17 true input dependencies;33 held tasks excluded. No new execution failure, confirmed preemption or invalid dependency. Preemption study924 attempts. The new one-case campaign is registered for the next full collection; the current snapshot was already running when it was prepared. Existing dependencies and held/V2G jobs were untouched.

Snapshot20260915T060910Z completed06:17:47UTC, SHA256 `1c5ab22bbd62596df1cdc48d2b5aadf92beb294faaa2d82cdcc31b47d163a9ef`. Register/workbook:3,205 records /70 source groups, with the exact six supplements retained. Checked169 evening endpoints and58 large/pool-diagnostic endpoints. [Changed endpoint values and hashes](new_endpoints.csv) · [Pool experiment validation](pool_experiments_validation.json). The completed reserve and fixed-state pricing studies are unchanged.

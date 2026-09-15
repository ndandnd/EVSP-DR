# Research results — 15 September, 05:19 EDT

**Chain 6 now matches the 28-bus target.** Its saved pool has 219,566 columns. Gurobi proves the fleet minimum of 28 after 25.1 minutes of fleet search; total MIP time is 60.1 minutes. CG reached its four-hour limit, so there is no full-model optimality claim. Charging optimization remains unfinished. The result passes individual-route replay, but 60 overcovered trips have no separate duplicate-removal validation.

| Chain | Largest target matched in the original one-hour MIP |
|---|---:|
| 1 | 26 |
| 2 | 27 |
| 3 | 27 |
| 4 | 26 |
| 5 | 26 |
| 6 | 28 |

This table describes the largest individual match. It does not imply that every smaller case matched within its original budget. **[All original k16–30 bus counts, CG times, stopping reasons and bounds](CHAIN_TABLES.md)** now appear in one table. Blank cells mean no verified result yet. Separate longer MIPs are excluded. The accompanying [CSV](all_chain_extension_results.csv) includes all 90 submitted cases, their inputs, job IDs and source hashes; 73 CG and 71 MIP endpoints are verified.

These are baseline covering runs with inherited columns, 240 kWh batteries, 240 kW charging and a fee of 5 per charging start. Reserve, shared charger capacity and a minimum ending SOC are absent. A fleet proof applies only to the saved pool; individual-route replay, duplicate-removal validation and charging optimality are separate.

**Larger compact starts now have 10 verified MIPs: seven target matches and three open gaps.** The new result is chain 2 at k20 with the expanded 512-sequence start: 21 buses, bound 20, after 180.0 minutes of fleet search. CG had converged in 171.4 minutes; weighted LP objective 2,000,806.3818165 and fractional route weight 20. This does not prove that its pool lacks a 20-bus solution.

| Unresolved larger-start case | Buses found | Saved-pool fleet bound | CG minutes | Fleet-search minutes |
|---|---:|---:|---:|---:|
| C2 k20, expanded | 21 | 20 | 171.4 | 180.0 |
| C3 k20, core | 24 | 20 | 110.7 | 180.0 |
| C3 k20, expanded | 23 | 20 | 102.7 | 180.0 |

All three CGs converged; their integer gaps remain open. This differs from the proved C1 k15 compact-pool limitation. The seven verified target matches are unchanged. Fourteen larger MIPs remain unverified; early completion selects the current subset, so seven of ten is not the overall success rate. All 24 larger CGs have ended: three converged and 21 hit their limits. [Every larger case and proof flag](compact_large_results.csv).

New C4 k27 CG also reached its four-hour limit: fractional route weight 27, final pool objective 2,701,152.225104, last reported reduced cost −0.003261035. Its integer result was not yet in this scientific collection. The earlier C5 k27 example still has fractional weight 26 for target 27. Its MIP completed after the campaign scan; [the scheduler's path and hash](late_scheduler_results.json) are retained for the next verification, without promoting that endpoint into this table.

**The next chain steps are active.** All 12 k29–30 graph jobs are running. The full collection shows 34 running jobs and 30 genuine input dependencies, excluding 33 held historical tasks. No new execution failure, confirmed preemption or invalid dependency was detected. The new extension is now represented in the collector and register. Longer searches 220545 (C5 k25) and 222757 (C3 k28) still have no verified endpoint here. No new jobs, cancellations or requeues were needed in this check; held and EVSPV2G work were untouched.

The existing [C1 k15 missing-column audit](../status_20260915T080947Z/README.md), stricter-physics tests and completed comparisons are unchanged. The dashboard replaces values and explanations in place and preserves both figure tabs. Morning consolidation around 09:00 EDT remains planned.

Snapshot 20260915T091033Z ran from 09:10:33 to 09:19:20 UTC (527.0 seconds); SHA256 `8b625bd4a557566490d003fd0fa83a52b2d9ca771da844d17e55dbf238ca89a4`. Preemption study: 941 attempt records. Register/workbook: 3,262 records across 73 source groups; exact six supplements retained. Checks cover 321 core endpoints, 177 evening endpoints, 69 larger/pool-diagnostic endpoints and 144 original-chain endpoints. [Three new endpoints](new_endpoints.csv) · [Chain-table checks](chain_extension_table_validation.json) · [Document verification](doc_verification.json).

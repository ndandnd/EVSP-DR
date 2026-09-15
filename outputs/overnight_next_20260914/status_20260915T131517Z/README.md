# Morning research summary — 15 September, 09:24 EDT

**The current Google Doc has been consolidated into six editable tables and shorter explanations.** It now shows actual integer buses at fixed targets 25 and 28, original versus separate MIP searches, CG minutes, fractional route weights, weighted LP objectives and precise stopping reasons. The text shrank from 1,728 to 1,268 whitespace-delimited words including Markdown tables. Existing figure and history tabs are untouched; the prior detailed dashboard remains linked. Slides were not edited. [Verified exported dashboard](doc_after.md) · [Document checks](doc_verification.json).

## What changed in this collection

| Item | Verified finding | Interpretation |
|---|---|---|
| C4, target 28 CG | 239.92 minutes; route weight 28.0000; weighted LP objective 2,801,157.317663; last reduced cost −0.0216695 | Four-hour limit, without a pricing certificate. Its restricted-pool LP is not a full-model lower bound. |
| C1, target 28 graph recovery | Completed, manifest matches; CG187972 is now running | The upstream execution delay cleared. Graph construction is not an optimization proof. |
| C1 and C5, target 27 longer MIPs | Both earlier launches are now included in the full collector and register; neither has a published result | Same source pools, 12600-second total and 10800-second fleet limits. Original outcomes remain separate. |

No new original MIP endpoint was in the campaign scan. C5 k28 job187997 finished during the later scheduler scan; its path/hash are retained in [late_scheduler_results.json](late_scheduler_results.json), and its scientific result awaits the next full verification. This timing difference is not a failure. Original extension totals are 77 CG and 75 MIP endpoints across 90 submitted cases.

## Baseline interpretation

All six baseline chains have a 25-bus solution for target25 when separate unchanged-pool searches are included. At target28, C2 and C6 match in their original one-hour searches; C3 matches in a separate search. Every target match displayed in the dashboard has a saved-pool fleet proof. Charging optimality and full-model fleet optimality are separate.

C2 k25's repeat belongs to the older parallel_pool_followup_20260914 cohort. The morning builder explicitly rechecks its completion marker, manifest, input/static hashes, source status/journal, solver commit, budgets and physical audit before placing25 in the table. It found25/proved in the pool; resultSHA87c52d4a7855a94650a481f19af8a6bf445bcd4ea800ae98555a0601a849d175. [Verification](c2_k25_repeat_verification.json). The other entries come from the source-bound current summary CSVs.

The baseline remains set covering, inherited columns,240kWh/240kW, no reserve/shared-capacity/ending-SOC floor, and cost100000 plus electricity plus5 per charging start. Individual routes were replayed; duplicate-trip removal has not been separately validated. The harder GIRO settings remain a separate small-instance study.

Two integer obstacles are established by the completed experiments: all25 original k16–25 misses were recovered without adding columns; conversely C1k15's two smaller pools prove16 buses necessary although the full pool supports15 with the same LP objective. Neither conclusion establishes a general speedup or a full-model integer proof. [Every original chain count and stopping reason](CHAIN_TABLES.md) · [Larger compact-start results](compact_large_results.csv).

## Cluster and next decisions

The collection shows28running and25true input dependencies, excluding33held historical tasks. No new execution failure, impossible dependency or confirmed preemption. All12k29–30graphs, the12union/control MIPs, two longer target27MIPs, C1k28CG and C4k28MIP were running. No jobs were launched, cancelled or requeued in this check. Held historical and EVSPV2G work are untouched.

Three sampled target29 graph builders have processed19,248/60,740,19,954/55,766 and22,539/60,811 source states after about4.9hours. [Exact progress records](graph_progress_sample.json). They are progressing; these fractions do not provide reliable linear completion estimates. Graph preparation time is separate from the four-hour CG budget. Do not describe the chain's CG minutes as its full end-to-end compute cost.

The next useful comparison is whether the eight core/expanded-pool unions repair their integer gaps, relative to four unchanged-pool controls. All12 are still running, with no verified production endpoint. Do not launch duplicates or erase real dependencies to increase the queue count. Routine hourly monitoring continues; the morning document consolidation is complete.

Snapshot20260915T131517Z ran13:15:17–13:24:26UTC,549.0seconds; SHA256324342c29de68900e2ce2cb8ade6607fb4d5020c548a44cc26fb03b26d78cf3f. Register/workbook3,288records/75source groups; exact6supplements retained. Checks321core,185evening,83pool and152original-chain endpoints; C2k25repeat checked separately. Preemption958attempts. [Changes and hashes](new_endpoints.csv).

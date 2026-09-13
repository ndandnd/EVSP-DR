# Full-pool chain results

Evidence captured 2026-09-13T02:21:20.956649+00:00.

**33 of 33 completed MIPs match their fleet targets.** **35 of 37 CG cases** have pricing certificates. Unfinished cases are not failed target matches.

| Chain | Highest target matched | Full-pool buses | Earlier bounded buses at that target | CG minutes |
|---|---:|---:|---:|---:|
| 1 | 13 | 13 | 15 | 70.6 |
| 2 | 14 | 14 | 16 | 89.4 |
| 3 | 15 | 15 | 17 | 33.1 |
| 4 | 15 | 15 | 18 | 64.4 |
| 5 | 15 | 15 | 17 | 52.3 |
| 6 | 14 | 14 | 16 | 65.1 |

The table shows the highest completed match per chain, not a computational threshold or independent statistical replications. [All results and source hashes](results.json), [editable CSV](integer_results.csv).

Earlier bounded results for chain 2 k14–15 come from the recorded shutdown-repaired retry, with the same 512-route / 900-second inheritance treatment. Its source revision differs from the original bounded campaign. These chain-level outcomes are not single-change timing comparisons; the new controlled campaign provides those.

## Settings and proof limits

Set covering; unlimited inherited sequences checked by the fixed-sequence index; 240 kWh batteries; 240 kW charging; 2.5 kWh / 5-minute event graph; flat prices; no shared-station capacity or return-SOC floor. No GIRO solution columns are injected. CG uses 100,000 per bus plus electricity and 5 per charge start. MIP stage 1 minimizes fleet for up to 1,800 seconds; stage 2 imposes fleet ≤ the validated incumbent and minimizes charging with the remainder of the 3,600-second budget. CG minutes include importing routes and use existing graph caches; original graph construction is excluded.

Read the separate certificate, pool proof and physical-check fields in results.json. A fleet proof concerns the accepted column pool. A CG pricing certificate concerns its represented graph and reduced-cost tolerance. Neither is a full-model integer proof. Individual-route replay, duplicate-trip removal and shared-capacity checks are separate. A final MIP TIME_LIMIT can refer to charging optimization after stage 1 has proved the fleet.

## Diagnosis

C4 k10 previously had a pool proved to require 11 buses; the full-pool run permits 10 at the same certified LP objective. C1 k8 also improves a proved pool minimum from 9 to 8. Other old incumbents were sometimes unproved, so MIP search difficulty remains relevant. Do not attribute every change in an inherited chain to one isolated code modification. [Initial matched-parent audit](../status_20260912T220047Z/README.md).

## Execution

This summarizer does not submit or retry jobs. Consult the dated collector snapshot and its delta for scheduler transitions. The separate bounded chain 1 k15 MIP ended with 18 buses and pool bound 15, without a fleet proof; its individual-route replay passed. It must not be mixed with the new full-pool results. Held historical jobs and V2G work remain protected.


Dated execution check: At 22:22 EDT, 30 EVSP–DR jobs were running: 24 controlled paired comparisons, four full-pool chain CG/MIP jobs and two decomposition jobs. Twenty-three jobs waited for valid inputs; none waited on an impossible dependency. Thirty-three held historical tasks were untouched. Nine parent CG jobs wait only for shared graph job 42509; join00 also needs component MIP 42511, whose CG 42510 is running. Next-k CG depends on previous-k CG, while each MIP depends on its own CG. No new execution errors or confirmed preemptions were observed.

The controlled comparison campaign has 24 active pairs and one certified first-arm CG: C4 k10, full-index contrast repetition 2, indexed full-pool arm, 545.617 seconds / 222 iterations, weighted objective 1,000,426.271076, fractional route weight 10.000000000000057. Its paired arm is unfinished; no comparative speed estimate is available. Allocation survival is tracked separately from standalone one-hour MIPs.

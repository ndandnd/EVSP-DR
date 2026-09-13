# Full-pool chain results

Evidence captured 2026-09-13T03:21:00.549714+00:00.

**35 of 35 completed MIPs match their fleet targets.** **36 of 37 CG cases** have pricing certificates. Unfinished cases are not failed target matches.

| Chain | Highest target matched | Full-pool buses | Earlier bounded buses at that target | CG minutes |
|---|---:|---:|---:|---:|
| 1 | 14 | 14 | 16 | 75.0 |
| 2 | 14 | 14 | 16 | 89.4 |
| 3 | 15 | 15 | 17 | 33.1 |
| 4 | 15 | 15 | 18 | 64.4 |
| 5 | 15 | 15 | 17 | 52.3 |
| 6 | 15 | 15 | 18 | 64.0 |

The table shows the highest completed match per chain, not a computational threshold or independent statistical replications. [All results and source hashes](results.json), [editable CSV](integer_results.csv).

Earlier bounded results for chain 2 k14–15 come from the recorded shutdown-repaired retry, with the same 512-route / 900-second inheritance treatment. Its source revision differs from the original bounded campaign. These chain-level outcomes are not single-change timing comparisons; the new controlled campaign provides those.

## Settings and proof limits

Set covering; unlimited inherited sequences checked by the fixed-sequence index; 240 kWh batteries; 240 kW charging; 2.5 kWh / 5-minute event graph; flat prices; no shared-station capacity or return-SOC floor. No GIRO solution columns are injected. CG uses 100,000 per bus plus electricity and 5 per charge start. MIP stage 1 minimizes fleet for up to 1,800 seconds; stage 2 imposes fleet ≤ the validated incumbent and minimizes charging with the remainder of the 3,600-second budget. CG minutes include importing routes and use existing graph caches; original graph construction is excluded.

Read the separate certificate, pool proof and physical-check fields in results.json. A fleet proof concerns the accepted column pool. A CG pricing certificate concerns its represented graph and reduced-cost tolerance. Neither is a full-model integer proof. Individual-route replay, duplicate-trip removal and shared-capacity checks are separate. A final MIP TIME_LIMIT can refer to charging optimization after stage 1 has proved the fleet.

## Diagnosis

C4 k10 previously had a pool proved to require 11 buses; the full-pool run permits 10 at the same certified LP objective. C1 k8 also improves a proved pool minimum from 9 to 8. Other old incumbents were sometimes unproved, so MIP search difficulty remains relevant. Do not attribute every change in an inherited chain to one isolated code modification. [Initial matched-parent audit](../status_20260912T220047Z/README.md).

## Execution

This summarizer does not submit or retry jobs. Consult the dated collector snapshot and its delta for scheduler transitions. The separate bounded chain 1 k15 MIP ended with 18 buses and pool bound 15, without a fleet proof; its individual-route replay passed. It must not be mixed with the new full-pool results. Held historical jobs and V2G work remain protected.


Dated execution check: At the 23:21 EDT collection, 28 EVSP–DR jobs were running: 24 controlled paired comparisons, two full-pool chain jobs and two decomposition jobs. Twenty-two jobs waited for valid inputs; no invalid dependencies or new execution failures were observed. Thirty-three held historical tasks were untouched. Nine parent CG jobs wait only for shared graph job 42509; join00 also needs component MIP 42511, whose CG 42510 is running. No new confirmed preemptions were observed.

All 24 controlled pairs remain active: 21 first arms have CG pricing certificates and are running MIPs; the three original-scanning/full-pool arms remain in initialization. No pair has completed both arms, so no paired speedup estimate is established. Keep these seven-hour paired allocations separate from standalone one-hour MIPs in reliability statistics.

Remaining full-pool recovery work: chain 1 k15 CG and MIP; chain 2 k15 MIP. Chain 2 k15 CG certified after 122.7 minutes.

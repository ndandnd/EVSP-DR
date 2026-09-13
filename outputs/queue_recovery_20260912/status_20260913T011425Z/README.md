# Full-pool chain results

Evidence captured 2026-09-13T01:14:26.124426+00:00.

**29 of 29 completed MIPs match their fleet targets.** **33 of 37 CG cases** have pricing certificates. Unfinished cases are not failed target matches.

| Chain | Highest target matched | Full-pool buses | Earlier bounded buses at that target | CG minutes |
|---|---:|---:|---:|---:|
| 1 | 12 | 12 | 13 | 26.8 |
| 2 | 13 | 13 | 15 | 34.2 |
| 3 | 15 | 15 | 17 | 33.1 |
| 4 | 14 | 14 | 18 | 59.2 |
| 5 | 15 | 15 | 17 | 52.3 |
| 6 | 13 | 13 | 15 | 75.0 |

The table shows the highest completed match per chain, not a computational threshold or independent statistical replications. [All results and source hashes](results.json), [editable CSV](integer_results.csv).

## Settings and proof limits

Set covering; unlimited inherited sequences checked by the fixed-sequence index; 240 kWh batteries; 240 kW charging; 2.5 kWh / 5-minute event graph; flat prices; no shared-station capacity or return-SOC floor. No GIRO solution columns are injected. CG uses 100,000 per bus plus electricity and 5 per charge start. MIP stage 1 minimizes fleet for up to 1,800 seconds; stage 2 imposes fleet ≤ the validated incumbent and minimizes charging with the remainder of the 3,600-second budget. CG minutes include importing routes and use existing graph caches; original graph construction is excluded.

Read the separate certificate, pool proof and physical-check fields in results.json. A fleet proof concerns the accepted column pool. A CG pricing certificate concerns its represented graph and reduced-cost tolerance. Neither is a full-model integer proof. Individual-route replay, duplicate-trip removal and shared-capacity checks are separate. A final MIP TIME_LIMIT can refer to charging optimization after stage 1 has proved the fleet.

## Diagnosis

C4 k10 previously had a pool proved to require 11 buses; the full-pool run permits 10 at the same certified LP objective. C1 k8 also improves a proved pool minimum from 9 to 8. Other old incumbents were sometimes unproved, so MIP search difficulty remains relevant. Do not attribute every change in an inherited chain to one isolated code modification. [Initial matched-parent audit](../status_20260912T220047Z/README.md).

## Execution

This summarizer does not submit or retry jobs. Consult the dated collector snapshot and its delta for scheduler transitions. The separate bounded chain 1 k15 MIP ended with 18 buses and pool bound 15, without a fleet proof; its individual-route replay passed. It must not be mixed with the new full-pool results. Held historical jobs and V2G work remain protected.

At 21:18 EDT, nine EVSP–DR jobs were running: seven full-pool chain CG/MIP jobs and two decomposition jobs. Twenty-six jobs waited for valid inputs; none waited on an impossible dependency. Thirty-three held historical tasks were untouched. Nine parent CG jobs wait only for shared graph job 42509; join00 also needs component MIP 42511, whose CG 42510 is running. Next-k CG depends on previous-k CG, while each MIP depends on its own CG. The separate bounded chain 2 k15 MIP finished with 17 buses and a pool fleet lower bound of 15, without a fleet proof. Its CG stopped at four hours without certification. No new execution errors or confirmed preemptions were observed. See [dated queue with exact dependencies](queue_status.json).

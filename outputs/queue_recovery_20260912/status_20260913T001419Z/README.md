# Full-pool chain results

Evidence captured 2026-09-13T00:14:19.915334+00:00.

**25 of 25 completed MIPs match their fleet targets.** **29 of 37 CG cases** have pricing certificates. Unfinished cases are not failed target matches.

| Chain | Highest target matched | Full-pool buses | Earlier bounded buses at that target | CG minutes |
|---|---:|---:|---:|---:|
| 1 | 10 | 10 | 11 | 17.6 |
| 2 | 13 | 13 | 15 | 34.2 |
| 3 | 15 | 15 | 17 | 33.1 |
| 4 | 13 | 13 | 15 | 42.7 |
| 5 | 14 | 14 | 16 | 36.6 |
| 6 | 13 | 13 | 15 | 75.0 |

The table shows the highest completed match per chain, not a computational threshold or independent statistical replications. [All results and source hashes](results.json), [editable CSV](integer_results.csv).

## Settings and proof limits

Set covering; unlimited inherited sequences checked by the fixed-sequence index; 240 kWh batteries; 240 kW charging; 2.5 kWh / 5-minute event graph; flat prices; no shared-station capacity or return-SOC floor. No GIRO solution columns are injected. CG uses 100,000 per bus plus electricity and 5 per charge start. MIP stage 1 minimizes fleet for up to 1,800 seconds; stage 2 imposes fleet ≤ the validated incumbent and minimizes charging with the remainder of the 3,600-second budget. CG minutes include importing routes and use existing graph caches; original graph construction is excluded.

Read the separate certificate, pool proof and physical-check fields in results.json. A fleet proof concerns the accepted column pool. A CG pricing certificate concerns its represented graph and reduced-cost tolerance. Neither is a full-model integer proof. Individual-route replay, duplicate-trip removal and shared-capacity checks are separate. A final MIP TIME_LIMIT can refer to charging optimization after stage 1 has proved the fleet.

## Diagnosis

C4 k10 previously had a pool proved to require 11 buses; the full-pool run permits 10 at the same certified LP objective. C1 k8 also improves a proved pool minimum from 9 to 8. Other old incumbents were sometimes unproved, so MIP search difficulty remains relevant. Do not attribute every change in an inherited chain to one isolated code modification. [Initial matched-parent audit](../status_20260912T220047Z/README.md).

## Execution

This summarizer does not submit or retry jobs. Consult the dated collector snapshot and its delta for scheduler transitions. The separate bounded chain 1 k15 MIP ended with 18 buses and pool bound 15, without a fleet proof; its individual-route replay passed. It must not be mixed with the new full-pool results. Held historical jobs and V2G work remain protected.


## Execution at 20:16 EDT

Eleven EVSP–DR jobs were running, with no invalid dependencies. The missing d00_g3 component graph is saved and its CG 42510 has started; the shared parent32 graph job 42509 remains running. See decomposition_progress.json for the cache manifest, hash and observed job states. The old bounded chain 2 k15 CG stopped at its four-hour budget without a pricing certificate; its MIP is running. No new speculative jobs or retries were submitted.

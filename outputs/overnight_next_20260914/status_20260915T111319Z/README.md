# Research results — 15 September, 07:21 EDT

**Chain 3 now matches the 28-bus target in a separate MIP search on its unchanged saved pool.** Gurobi proves 28 minimal within those 166,052 columns after 27.41 native fleet-search minutes. Total runtime is 210.06 minutes; charging optimality remains open. Individual-route replay passes. Duplicate-trip removal and shared charger capacity are not validated by this result.

| Chain | Largest target matched in original one-hour MIP | Including separate longer-budget searches |
|---|---:|---:|
| 1 | 26 | 26 |
| 2 | 27 | 27 |
| 3 | 27 | 28 |
| 4 | 27 | 27 |
| 5 | 26 | 26 |
| 6 | 28 | 28 |

These are largest individual matches, not a claim that all smaller cases matched within the original budget. The original k16–25 batch remains 35/60 target matches; all 25 original misses have separately been recovered from unchanged pools. [All 90 original extension cases with actual bus counts, CG times and stopping reasons](CHAIN_TABLES.md) · [New longer-search result and unchanged-pool checks](longer_gap_results.csv).

**Why did this repeat succeed within about 27 minutes when the original 30-minute fleet search missed?** Its hardware differs. The original ran on Xeon E5-2665 and the repeat on Xeon Gold 6348. Both use eight solver threads. The same 14,238-iteration root LP takes 4.81 seconds originally and 2.55 seconds in the repeat. The fleet search reports 2,053.56 versus 3,685.14 work units, and 206 versus 673 nodes. Machine throughput is a plausible contributor; hardware and allocated time limit both changed, so the cause is not isolated. [Side-by-side measurements and exact log references](SEARCH_WORK.md). This does not establish a pricing-algorithm improvement.

Two new CG endpoints are verified:

| Case | Trips | CG minutes | Fractional route weight | Weighted RMP objective | Last reported reduced cost | Why CG stopped |
|---|---:|---:|---:|---:|---:|---|
| C1, target27 | 637 | 239.40 | 27.0000 | 2,701,142.358847 | −0.00122474 | Four-hour limit |
| C2, target28 | 643 | 239.61 | 28.0000 | 2,801,153.403676 | −0.00759260 | Four-hour limit |

Neither has a pricing certificate, and neither has a collected MIP endpoint in this snapshot. These restricted-pool objectives are not certified full-model lower bounds. The baseline model remains covering, inherited columns, 240 kWh/240 kW, bus coefficient 100,000 plus electricity and 5 per charging start, without reserve/shared capacity/ending-SOC limits. Stricter-physics and completed compact-start results are unchanged.

**The new pool-combination comparison is active:** eight construction audits and all 12 MIP launches are now part of the full snapshot. No production MIP has a published endpoint yet; the short validation results remain separate. The previous larger compact batch stays complete: 8 targets, 8 target-excluding pools, 8 open misses. [Current union/control registration and source accounting](compact_union_results.csv) · [Complete larger-start results](compact_large_results.csv).

Queue in this collection: 29 running / 28 true input dependencies, excluding 33 held historical tasks. All 12new comparison MIPs and 12 k29–30 graphs are running. There is no new failed/unsatisfiable dependency or confirmed preemption. No new jobs, cancellations or requeues were needed. Held historical and EVSPV2G work are untouched. The new campaign's 15 MIP registrations (3 validation / 12 production) are included in 956 attempt records; registration refresh added no duplicate.

The current Doc replaces its existing dates, chain values and explanations in place. Both figure tabs remain preserved, and Slides were not edited. Morning consolidation around 09:00 EDT remains planned.

Snapshot `20260915T111319Z` ran 11:13:19–11:21:39 UTC, 499.8 seconds; SHA256 `13102de8112b628bee34dff85292547510932e0bd9f20b387e6ee39afd112895`. Register/workbook: 3,281 records across 74 source groups with the exact 6 supplements preserved. Checks cover 321 core, 181 evening, 83 pool-diagnostic and 148 original-chain endpoints. Consolidated extension table: 75 CG / 73 MIP endpoints among 90 submitted cases. [Three new endpoints](new_endpoints.csv) · [Document checks](doc_verification.json).

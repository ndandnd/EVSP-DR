# 06:33 EDT results — 11 September 2026

**Warm chain 6 matches k=10 with ten buses.** Stage 1 proved fleet ten in its 33,879-column pool in 1.74 seconds. Stage 2 kept fleet ≤10 and reached its time limit with charging-related objective 441.032, bound 420.419239 and gap 4.67%. Total MIP runtime was 60.01 minutes; source CG time was 160.21 minutes. Individual route replay passed; 12 trips are overcovered. No GIRO routes were injected.

Warm chain 4 matches k=8: fleet eight, both MIP stages optimal within the pool, CG 203.62 minutes and MIP 9.63 minutes. Three trips are overcovered. Duplicate removal and shared station capacity remain separate validation requirements for both results; these are not full-model integer proofs.

## Fresh covering fleet incumbents

| Target | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 2 | 2 | 2 | 3 | 2 |
| 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| 5 | 5 | 5 | 5 | 5 | 6 | 5 |
| 6 | 6 | 7 | 6 | 6 | 7 | 6 |
| 7 | 8 | 7 | 8 | 8 | 8 | 7 |
| 8 | 9 | 9 | 9 | 9 | 9 | 8 |
| 9 | 11 | 11 | 11 | 10 | 10 | 10 |
| 10 | 11 | 11 | 11 | 11 | 11 | 11 |
| 11 | 13 | 12 | 13 | 13 | 12 | 12 |
| 12 | 14 | 13 | 14 | 14 | 13 | 14 |
| 13 | 15 | 15 | 15 | 15 | 14 | 15 |
| 14 | 17 | 17 | pending | pending | pending | pending |
| 15 | pending | pending | pending | pending | pending | pending |

All 75 newly launched CG cases report pricing certificates for their documented conservative event-grid model. The table combines their completed MIPs with the nine earlier covering cases. A fleet entry is an incumbent: it is **not uniformly proved optimal**. Exact pool bounds, proof scopes, runtime, replay status and source hashes are in [the editable CSV](fresh_covering_results.csv).

New k=7 results for chains 1/3/5 prove eight buses within their pools. At k=9, chains 5/6 prove ten buses in their pools; chains 1–4 remain unproved. At k=11–13, the new incumbents are usually one or two buses above target, with unresolved fleet gaps. This distinguishes an integer limitation of a particular pool from an unfinished MIP search.

## Overnight reliability and remaining work

Default trial: 75 started, 65 completed, 10 running; zero recorded preemptions. Thirty-four completed solver runs used at least 3500 seconds. These heterogeneous, correlated overnight observations do not prove a general one-hour survival probability. No new FAILED, TIMEOUT, OUT_OF_MEMORY or PREEMPTED state appeared in the queried main arrays.

All fresh CG and freeze prerequisites have cleared. Ten fresh MIPs, additional sequential warm-chain work, and all six capacity-budget-extension CG cases remain active or dependent. The registered backlog is not exhausted. No rerun was needed this hour.

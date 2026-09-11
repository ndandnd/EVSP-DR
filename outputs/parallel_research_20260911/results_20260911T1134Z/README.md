# 07:34 EDT results — 11 September 2026

**The fresh covering grid is complete: all 84 chain/size cases now have MIP results.** This combines 75 new cases with nine previously completed controls. All 75 new default-partition attempts completed, with zero recorded preemptions; 44 used at least 3500 seconds of optimization. This is evidence from one overnight workload, not a general reliability guarantee.

## Actual fleet incumbents

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
| 14 | 17 | 17 | 16 | 18 | 16 | 17 |
| 15 | 18 | 17 | 18 | 19 | 16 | 19 |

[Editable results with exact pool bounds, proof scopes, times and source hashes](fresh_covering_complete84.csv). Every entry is an incumbent, not necessarily a proved fleet optimum. In particular, all k=14–15 fleet bounds remain below the incumbent. All selected-route replays passed, but duplicate removal and shared station capacity remain separate checks. CG certificates concern the conservative event-grid model; finite-pool MIP proofs do not prove full-model integer optimality.

Warm chain 1 additionally matches k=6 with six buses; both MIP stages optimal within the saved pool, MIP 53.94 seconds, six trips overcovered. Other warm chains still have genuine previous-k work outstanding.

## Capacity run status

All six capacity-budget-extension CG tasks are still RUNNING after about 8 hours 11 minutes, with no final CG or MIP artifacts. Their latest logged LP solves take 0.00–0.01 seconds on approximately 7,814–7,835 rows and 46–73 columns. Stderr is empty. These logs do not establish a pricing certificate or physical feasibility. They point away from the LP solve as the immediate bottleneck, but do not by themselves prove where the process currently spends time. The driver has a nine-hour scheduler allocation and can overrun its eight-hour loop budget inside a pricing call; live jobs were left intact. Source tails and timestamps are retained in `capacity_live_logs.json`.

## Next decision

The fresh batch is exhausted; the overall registered backlog is not. Finish the inherited-chain comparisons and capacity pilot, then prioritize a matched saved-pool comparison on the remaining misses: fresh pool versus inherited/union pool, identical MIP budget and objective. Pair that with duplicate-removal validation before treating covering solutions as executable duty schedules. Additional random fresh runs would be less diagnostic than testing why the existing pools differ.

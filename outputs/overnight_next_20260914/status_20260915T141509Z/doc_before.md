# **EVSP–DR: current results**

**Verified 15 September, 09:24 EDT.** Read this tab for current conclusions. Figures and the dated research history remain in the other tabs.

## **The main result**

**All six baseline chains have a 25-bus solution for target 25; several now reach target 28\.** Some matches require a separate MIP search on the same columns. These larger results do **not** include shared charger capacity or an ending-SOC requirement.

## **Actual buses found**

Target k means the trips came from k GIRO bus duties. “Original” is the one-hour MIP. “Separate search” uses the unchanged saved columns with a larger allowance; it starts a new search tree. A dash means no separate result is needed or available. Pending is not a failed solve.

| Chain | Target 25: original | Target 25: separate search | Target 28: original | Target 28: separate search |
| ----- | ----- | ----- | ----- | ----- |
| 1 | 25 | — | Pending | — |
| 2 | 26 | 25 | 28 | — |
| 3 | 26 | 25 | 29 | 28 |
| 4 | 26 | 25 | Pending | — |
| 5 | 26 | 25 | Pending | — |
| 6 | 26 | 25 | 28 | — |

Every target match shown has its fleet minimum proved within its own saved pool. Charging optimality is a separate question.

The original k16–25 batch matched 35 of 60 targets. Separate searches recovered all 25 misses. Those pools already contained adequate routes. This does not isolate an effect of extra time: CPU hardware and search trajectories can differ.

## **What CG achieved at target 28**

| Chain | CG minutes | Fractional route weight | Weighted LP objective | Why CG stopped |
| ----- | ----- | ----- | ----- | ----- |
| 1 | Running | Pending | Pending | Pending |
| 2 | 239.6 | 28.0000 | 2,801,153.4037 | Four-hour limit |
| 3 | 239.9 | 28.0000 | 2,801,137.0745 | Four-hour limit |
| 4 | 239.9 | 28.0000 | 2,801,157.3177 | Four-hour limit |
| 5 | 239.8 | 27.0000 | 2,701,242.9135 | Four-hour limit |
| 6 | 239.9 | 28.0000 | 2,801,156.0234 | Four-hour limit |

CG minutes include this k’s route import and CG, but exclude earlier k values, graph preparation and MIP. Fractional route weight is the sum of route variables, not the weighted objective. A time-limited result is not a certified full-model lower bound.

**Three distinct checks:** CG convergence proves that pricing found no improving route within its modeled representation and tolerance. A saved-pool fleet proof establishes the fewest buses using only the generated columns. Individual-route replay checks each route’s modeled feasibility. None alone proves full operational optimality.

## **Why some integer solutions are worse**

| Obstacle | Evidence | What can help |
| ----- | ----- | ----- |
| Search has not found the available combination | All 25 original k16–25 misses were recovered without adding columns. | More effective MIP search; record hardware and search work. |
| Useful integer routes are missing | At C1 target 15, both smaller pools prove 16 buses are necessary; the full pool supports 15\. Their LP objectives agree. | Add different columns; more MIP time alone cannot repair these pools. |

In that C1 example, 12 of the known 15-bus solution’s trip patterns are absent from the core pool, and 11 from the expanded pool. All 15 witness routes have positive reduced cost at the smaller pools’ final dual prices. They would not improve the LP, so negative-reduced-cost pricing need not generate them.

## **Does keeping fewer starting routes help?**

**Core:** keep earlier integer-solution routes and routes with positive LP weight. **Expanded:** fill that core to 512 distinct trip sequences. These are starting sequences; CG can add more. All 24 tests at targets 8 and 10 match. At target 15, 9 of 12 match.

| Chain | Target 20: core | Target 20: expanded | Target 25: core | Target 25: expanded |
| ----- | ----- | ----- | ----- | ----- |
| 1 | 21 | 21 | 25 | 25 |
| 2 | 22 | 21 | 31 | 26 |
| 3 | 24 | 23 | 26 | 25 |
| 4 | 20 | 20 | 26 | 26 |
| 5 | 21 | 21 | 26 | 26 |
| 6 | 20 | 20 | 26 | 25 |

Of these 24 larger results, eight match, eight pools prove the target impossible, and eight misses remain unresolved. Only three CG runs converge; 21 hit four hours. MIPs allow three hours for fleet search, 3.5 hours total. Smaller starts therefore do not reliably preserve the full pool’s integer quality.

In the separate accumulated-CG-time comparison, warm starts match 24 of 24 targets; fresh starts match 6 of 24\. All 24 fresh CGs converge. Extra CG time alone does not reproduce the warm results.

## **Which GIRO settings are included?**

| Model or test | Settings and result |
| ----- | ----- |
| Large baseline chains | 240 kWh battery; 240 kW charging; no reserve, shared charger capacity or ending-SOC floor. Set covering; inherited columns. Route cost \= 100,000 \+ electricity \+ 5 per charging start. |
| Reserve and selected speed/capacity tests | 236.44 kWh battery with 15% reserve. Eight of ten one-duty tests use one bus; duty 13405 uses two in both variants. All ten CGs converge. No 65% ending-SOC floor or nonlinear charging. |
| Duty 13408: reserve, shared capacity, PARX at 60 kW | One bus recovered; CG takes 77 minutes versus 2.2 minutes with reserve alone. A one-bus test does not establish multi-bus charging feasibility. |
| Harder capacity pricing | Two reference pricing calls finish in 3.58 and 3.19 hours. The cached version does not finish either within about four hours. No speedup demonstrated. |

For covering results, individual routes were replayed, but removing duplicate trip assignments has not been separately validated. The large-chain table must not be presented as a match to every GIRO constraint.

## **Work in progress**

Queue in this collection: **28 running**, 25 waiting for required inputs; held historical jobs excluded. No array throttle is blocking this batch.

| Work | Question |
| ----- | ----- |
| Six chains through targets 29–30 | Graph preparation runs in parallel. Each CG then waits for its own graph and the preceding k’s columns; its MIP follows. |
| Eight pool unions and four unchanged-pool controls | Can combining core and expanded columns restore a target fleet? No new CG or GIRO routes are added. |
| Longer MIPs: C1 and C5 at target 27 | Originals found 28/bound 27 and 29/bound 26\. Are target fleets already present in these unchanged pools? |

Three sampled target-29 graph builders processed roughly one-third of their source states after 4.9 hours. This preprocessing is separate from CG time; progress is not a reliable completion-time estimate.

The monitor continues hourly, repairs demonstrated execution problems, and reports verified changes or access loss. The next decision is whether pool combinations repair the integer gaps; further algorithm changes should be tested with matched settings.

## **Figures and sources**

[Comparison figures and explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) · [CG curves and Gantt plots](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly). Existing figures are preserved; check their dates and model settings.

[Every original chain count, exact LP objective and CG stopping reason](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260915T131517Z/CHAIN_TABLES.md) · [Verified source report](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260915T131517Z/README.md) · [Detailed dashboard before this consolidation](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260915T131517Z/doc_before.md) · [Dated research history](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow).
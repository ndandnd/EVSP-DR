# **EVSP DR Current Research**

Week of 14 September. Results checked 12 September 2026 at 09:30 EDT; overnight extensions are still running. Open [Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) for timing, charging costs and geography, or [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) for the restored figure library.

**Current finding.** With set covering and inherited columns, all six chains match the GIRO fleet at k=3–6. Chains 3, 5 and 6 also reach ten buses at k=10. The new 512-route import treatment reaches 11 buses at k=11 in chain 6; other completed extension cases need extra buses.

## **Integer fleet with full-pool inheritance**

Each cell is the number of integer bus routes found. Green matches the target. Red requires extra buses. Dashes denote interrupted CG; see the C2 k10 note below.

| Target buses | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 2 | 2 | 2 | 2 | 3 | 2 |
| 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| 7 | — | 7 | 7 | 7 | 7 | 7 |
| 8 | — | 8 | 8 | 8 | 8 | 8 |
| 9 | — | — | 9 | 9 | 9 | 9 |
| 10 | — | — | 10 | — | 10 | 10 |

**The table above retains the earlier full-pool treatment. The extension checks at most 512 inherited routes before CG; its new integer results are shown separately below. Runs continue through k=15 for all six chains.**

k=2 starts from single-trip routes; later sizes reuse validated previous-k columns. Chain 3 k=10 also includes routes from a fresh solver solution. These are pool proofs. Chain 2 k10 (71 buses) had zero pricing iterations and no CG certificate. Individual route replay passed; duplicate-trip removal and shared charger capacity are not established by this table.

## 

## **Fresh starts on the same chains**

Each size starts independently. Both tables use covering and 240-kWh batteries with 240-kW charging. Fresh fleet proof status varies by cell.

| Target buses | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 5 | 5 | 5 | 5 | 5 | 6 | 5 |
| 8 | 9 | 9 | 9 | 9 | 9 | 8 |
| 10 | 11 | 11 | 11 | 11 | 11 | 11 |
| 15 | 18 | 17 | 18 | 19 | 16 | 19 |

Inherited columns improve the available integer pool: chain 3 at k=8 improves from a proved nine-bus pool optimum to eight; chain 5 at k=5 improves from a proved six-bus pool optimum to five.

## **What is limiting progress**

**Checking saved routes took too long.** Before starting CG on a larger trip set, we check that routes saved from the previous size are still valid. In chain 1 at k=7, chain 2 at k=9 and chain 4 at k=10, this checking used up the available time. We now check at most 512 saved routes, for at most 15 minutes, then let CG search for new routes. All three cases have now finished CG. Their integer solves found 8, 10 and 11 buses, respectively; only the first and third are proved minimums within their saved pools. Using fewer saved routes may affect the final integer solution, so this is a separate experiment.

**Charger limits made the search for new routes very slow.** In one three-bus test, the LP solved in 0.006 seconds, but one search for a new route took 7.15 hours. The run ran out of time before it could prove that no improving route remained. Its 16-bus integer solution uses only the routes found so far; it does not show that 16 buses are necessary. The timing figures explain these two different delays.

| Case | Baseline | PARX 60 kW | Station capacity | Both |
| ----- | ----- | ----- | ----- | ----- |
| k=1, two duties tested | 1 each | 1 each | 1 each | 1 each |
| k=2, 23 trips | 2 | 2 | 3 | 3 |
| k=3, 35 trips | 3 | 3 | 16 | 16 |

Sixteen is the result in a small generated pool, not a proved physical requirement. Capacity-constrained schedules passed station sweeps. This pilot has no 65% return-SOC floor and is separate from the six baseline chains.

## 

## **Overnight extension results**

09:30 EDT collection: 63/87 CG certificates; 59 MIP results (29 warm, 30 component).

Warm starts check ≤512 saved routes. All CG results below are certified; times include preparation.

| Chain and target | Buses | Pool fleet bound | Fleet proved in pool | CG minutes | Weighted LP objective |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Chain 1, k=07 | 8 | 8 | Yes | 53.6 | 700313.457 |
| Chain 1, k=08 | 9 | 9 | Yes | 49.5 | 800383.688 |
| Chain 1, k=09 | 10 | 9 | No | 67.9 | 900453.496 |
| Chain 1, k=10 | 11 | 11 | Yes | 91.2 | 1000511.205 |
| Chain 1, k=11 | 13 | 11 | No | 114.1 | 1100537.061 |
| Chain 1, k=12 | 13 | 12 | No | 95.6 | 1200608.942 |
| Chain 2, k=09 | 10 | 9 | No | 64.0 | 900301.353 |
| Chain 2, k=10 | 11 | 10 | No | 55.7 | 1000371.146 |
| Chain 2, k=11 | 12 | 11 | No | 65.7 | 1100404.250 |
| Chain 2, k=12 | 13 | 12 | No | 85.6 | 1200441.362 |
| Chain 2, k=13 | 15 | 13 | No | 98.4 | 1300493.195 |
| Chain 3, k=11 | 12 | 11 | No | 27.7 | 1100375.859 |
| Chain 3, k=12 | 14 | 12 | No | 32.8 | 1200427.901 |
| Chain 3, k=13 | 14 | 13 | No | 52.6 | 1300474.559 |
| Chain 3, k=14 | 15 | 14 | No | 57.9 | 1400485.597 |
| Chain 3, k=15 | 17 | 15 | No | 59.7 | 1500507.420 |
| Chain 4, k=10 | 11 | 11 | Yes | 42.0 | 1000426.271 |
| Chain 4, k=11 | 12 | 11 | No | 59.4 | 1100462.865 |
| Chain 4, k=12 | 13 | 12 | No | 68.2 | 1200505.601 |
| Chain 4, k=13 | 15 | 13 | No | 88.6 | 1300530.332 |
| Chain 4, k=14 | 18 | 14 | No | 204.6 | 1400590.723 |
| Chain 5, k=11 | 12 | 11 | No | 70.5 | 1100522.228 |
| Chain 5, k=12 | 13 | 12 | No | 109.2 | 1200576.644 |
| Chain 5, k=13 | 14 | 13 | No | 134.6 | 1300625.367 |
| Chain 5, k=14 | 16 | 14 | No | 140.1 | 1400639.061 |
| Chain 6, k=11 | 11 | 11 | Yes | 32.5 | 1100399.029 |
| Chain 6, k=12 | 13 | 12 | No | 64.6 | 1200464.548 |
| Chain 6, k=13 | 15 | 13 | No | 117.1 | 1300532.041 |
| Chain 6, k=14 | 16 | 14 | No | 136.9 | 1400578.400 |

Pool fleet bounds concern saved columns only. A proved fleet above target requires different columns to reach target. An unproved gap does not distinguish incomplete integer search from missing useful columns.

## **Decomposition of 32 duties**

| Grouping | Four component fleets | Combined buses | Target |
| ----- | ----- | ----- | ----- |
| join01 | 9 \+ 9 \+ 9 \+ 8 | 35 | 32 |
| join02 | 9 \+ 9 \+ 9 \+ 8 | 35 | 32 |
| join03 | 9 \+ 9 \+ 8 \+ 10 | 36 | 32 |
| join08 | 9 \+ 9 \+ 10 \+ 9 | 37 | 32 |
| join09 | 8 \+ 9 \+ 9 \+ 9 | 35 | 32 |

These saved constructions combine disjoint trip groups chosen using GIRO duty membership; no GIRO route columns were supplied. Component routes passed replay. Shared charger capacity and duplicate removal are not certified. **All five joined-parent attempts timed out before producing a parent CG result.** Their saved decomposed constructions remain available, but there is no joint improvement or parent optimality proof.

## **Completed speed comparisons**

| Case | Reference minutes | Optimized minutes | Observation |
| ----- | ----- | ----- | ----- |
| d00\_g0 | 22.3 | 23.6 | 5.8% slower |
| d00\_g1 | 52.2 | 52.1 | Essentially unchanged |
| w1\_k08 | 41.1 | 34.4 | 16.2% less time |
| w1\_k08\_repeat | 31.1 | 24.1 | 22.4% less time |
| w4\_k11 | 58.8 | 49.1 | 16.5% less time |
| w6\_k12 | 67.2 | 54.6 | 18.6% less time |

All six pairs reached matching certified LP objectives. Four warm comparisons used 16–22% less time; inherited-route checking fell from 366–447 seconds to 5–7 seconds. Common graph-cache preparation is excluded. CG iteration counts differed, so the total savings are descriptive, not an isolated causal estimate. Fresh comparisons show no speed improvement. No integer improvement is claimed from these timing tests.

## **Capacity comparisons at the time limit**

| Capacity case | Iterations ref / opt | Final restricted LP objective | Fractional route weight | Pricing certified ref / opt |
| ----- | ----- | ----- | ----- | ----- |
| cap\_k1 | 13 / 12 | 100056.952000 | 1.0 | No / No |
| cap\_k1\_combined\_peak12 | 34 / 37 | 100074.693539 | 1.0 | No / No |
| cap\_k2 | 4 / 4 | 300094.976000 | 3.0 | No / No |

Every arm stopped after three hours. Both methods reached the same restricted-LP objective in each case; none proved that no improving route remained. Fractional route weight is not an integer fleet result. These tests do not demonstrate a convergence speedup.

**Default MIPs:** 75 earlier and 59 extension attempts completed with zero confirmed preemptions. One extension MIP was running at collection.

**Methods and evidence.** CG route cost \= 100,000 \+ electricity \+ 5 per charging start. MIP stage 1 minimizes buses. Stage 2 constrains buses ≤ the validated stage-1 incumbent and minimizes electricity plus start fees.

A pricing certificate establishes the LP result only in its represented graph and tolerance. A MIP proof concerns its saved columns. Fleet attainment, physical replay and shared-capacity validation are separate checks.

[Experiment register](https://github.com/ndandnd/EVSP-DR/tree/codex/parallel-research-20260911/outputs/research_register) · [Dated results and source tables](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_extension_20260912/RESULTS_20260912T133042Z.md) · [Experiment settings](https://github.com/ndandnd/EVSP-DR/tree/codex/parallel-research-20260911/outputs/overnight_extension_20260912) · [Historical document and figures](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit) · [Local prototype tests and charging-cost bug](https://github.com/ndandnd/EVSP-DR/blob/b142f360f8a9018c2f50e3419f0dddcaf87a489d/outputs/algorithm_benchmarks_20260912/README.md)

**Storage, 12 September:** 156 cold files archived losslessly; 133.74 GB freed. Active research and V2G data preserved. [Restore instructions](https://github.com/ndandnd/EVSP-DR/tree/b1129bf4/outputs/storage_cleanup_20260912).

**Blocked decomposition case d00\_g3:** the original run timed out before CG, while constructing the graph. Its MIP remains blocked. A separate diagnostic crashed after 366 seconds (SIGSEGV), before its watchdog limit. Partial samples show JSON serialization while adding charging arcs; A local timer-only test also hung, so the asynchronous sampler has been retired. The exact cluster crash remains unresolved. [Diagnostic evidence](https://github.com/ndandnd/EVSP-DR/blob/57f663eb/outputs/d00_g3_startup_review_20260912/TERMINAL_REVIEW.md). No long retry has been launched.
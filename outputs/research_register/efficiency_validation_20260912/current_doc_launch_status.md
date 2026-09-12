# **EVSP DR Current Research**

Week of 14 September. Results checked 12 September 2026 at 02:25 EDT; overnight extensions are still running. Open [Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) for timing, charging costs and geography, or [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) for the restored figure library.

**Current finding.** With set covering and inherited columns, all six chains match the GIRO fleet at k=3–6. Chains 3, 5 and 6 also reach ten buses at k=10. The new 512-route import treatment reaches 11 buses at k=11 in chain 6; other completed extension cases need extra buses.

## **Integer fleet with full-pool inheritance**

Each cell is the number of integer bus routes found. Green matches the target. Red requires extra buses. Grey means no completed result.

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

k=2 starts from single-trip routes; later sizes reuse validated previous-k columns. Chain 3 k=10 also includes routes from a fresh solver solution. Displayed warm fleet optima are proved within their saved pools. Individual route replay passed; duplicate-trip removal and shared charger capacity are not established by this table.

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

At 02:25 EDT, **30 of 87 CG cases are certified** (12 warm, 18 decomposition groups) and nine MIPs have finished. Runs continue through k=15 and across ten decompositions of a 32-duty parent.

**Warm results with at most 512 inherited routes.** All six CG runs below are certified at reduced-cost tolerance 0.0001. CG minutes include initialization. Fractional route weight equals the target in each case; the weighted objective also includes charging costs.

| Case k \= target | Buses found | Pool fleet bound | Fleet proved in pool | CG minutes | Weighted LP objective |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Chain 1, k=7 | 8 | 8 | Yes | 53.6 | 700313.457 |
| Chain 2, k=9 | 10 | 9 | No | 64.0 | 900301.353 |
| Chain 3, k=11 | 12 | 11 | No | 27.7 | 1100375.859 |
| Chain 3, k=12 | 14 | 12 | No | 32.8 | 1200427.901 |
| Chain 4, k=10 | 11 | 11 | Yes | 42.0 | 1000426.271 |
| Chain 6, k=11 | 11 | 11 | Yes | 32.5 | 1100399.029 |

The pool bound concerns only saved columns. Stage 2 reached its charging-cost time limit in all six cases. More MIP time on the same pools cannot reduce the proved eight- and eleven-bus results in chain 1 k=7 and chain 4 k=10. The other above-target cases retain an unresolved integer gap.

**Decomposition:** groups d00\_g0, d00\_g1 and d02\_g3 each match eight buses, with both MIP stages proved in their pools. No combined 32-duty result yet. All nine solutions pass individual-route replay and trip coverage; duplicate removal and shared charger capacity remain unchecked.

**Default MIPs:** the earlier 75 and the new nine completed with zero confirmed preemptions. Ten new MIPs are running; their outcomes remain unknown.

## **Methods and evidence**

CG route cost \= 100,000 \+ electricity \+ 5 per charging start. MIP stage 1 minimizes buses. Stage 2 constrains buses ≤ the validated stage-1 incumbent and minimizes electricity plus start fees.

A pricing certificate establishes the LP result only in its represented graph and tolerance. A MIP proof concerns its saved columns. Fleet attainment, physical replay and shared-capacity validation are separate checks.

[Experiment register](https://github.com/ndandnd/EVSP-DR/tree/codex/parallel-research-20260911/outputs/research_register) · [Dated results and source tables](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_extension_20260912/RESULTS_20260912T062451Z.md) · [Experiment settings](https://github.com/ndandnd/EVSP-DR/tree/codex/parallel-research-20260911/outputs/overnight_extension_20260912) · [Historical document and figures](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit) · [Local prototype tests and charging-cost bug](https://github.com/ndandnd/EVSP-DR/blob/b142f360f8a9018c2f50e3419f0dddcaf87a489d/outputs/algorithm_benchmarks_20260912/README.md) (12 September; full-CG speedup unmeasured)

**Storage, 12 September:** 156 cold files archived losslessly; 133.74 GB freed. Active research and V2G data preserved. [Verification and restore instructions](https://github.com/ndandnd/EVSP-DR/tree/b1129bf4/outputs/storage_cleanup_20260912).

**Paired speed tests launched, 12 September at 02:53 EDT.** Nine jobs compare reference and accelerated code. At 02:56, two fresh-CG and three capacity jobs are running. Four warm jobs stopped before optimization because the cache-preparation command included an incompatible output argument. Separately recorded retries are being prepared. No cluster speedup result yet. [Jobs, settings and exact error](https://github.com/ndandnd/EVSP-DR/tree/codex/parallel-research-20260911/outputs/research_register/efficiency_validation_20260912).
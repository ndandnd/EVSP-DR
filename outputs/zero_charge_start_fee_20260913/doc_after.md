# **EVSP DR Current Research**

Week of 14 September. Results collected 13 September 2026 at 01:12 EDT; queue checked at 01:14 EDT.

## **Charging-start fee: 5 versus 0**

**48 new jobs launched.** Six chains at k=5, 10 and 15 give 18 matched comparisons, with two hours of CG and one hour of MIP per run. Both fees start from the same saved trip sequences, with charging optimized again. We do not restart at k=2. Chain 1 k15 uses its frozen k14 pool; the other 17 cases use same-k pools.

Eight MIPs have already completed, all matching k=5 with pool fleet proofs and individual-route replay. Two comparisons have finished both fees:

| Chain (k=5) | Fee per start | Integer buses | Charging starts | Electricity only | End energy (kWh) |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 3 | 5 | 5 | 7 | 91.56 | 41.95 |
| 3 | 0 | 5 | 27 | 89.11 | 17.17 |
| 6 | 5 | 5 | 7 | 100.93 | 36.81 |
| 6 | 0 | 5 | 34 | 98.14 | 10.86 |

Removing the fee allows many more charging starts while retaining five buses in these two cases. These flat-price runs return less energy, so the electricity differences are not savings at equal returning energy. [Full paired table and source evidence](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/zero_charge_start_fee_20260913/early_pairs.csv).

**GIRO cost comparison:** six fresh fixed-duty charging optimizations are running; six joint MIPs wait for both fee frontiers at their price peak (08:00, 12:00 or 18:00). We compare original charging, optimized charging on fixed duties, and the common joint route pool. All optimized schedules must return at least GIRO’s aggregate 280.7833 kWh. This separate five-duty cohort retains 240 kWh / 350 kW physics; the chains use 240 kWh / 240 kW. [Settings, job IDs and launch repairs](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/zero_charge_start_fee_20260913/README.md).

## **Baseline chains with full-pool inheritance**

**All 37 CG cases have pricing certificates. All 36 completed MIPs match their fleet targets.** Chain 1 k15’s MIP remains in progress.

| Chain | Highest target matched | Integer buses | Earlier 512-route buses | CG minutes |
| ----- | ----- | ----- | ----- | ----- |
| 1 | 14 | 14 | 16 | 75.0 |
| 2 | 15 | 15 | 17 | 122.7 |
| 3 | 15 | 15 | 17 | 33.1 |
| 4 | 15 | 15 | 18 | 64.4 |
| 5 | 15 | 15 | 17 | 52.3 |
| 6 | 15 | 15 | 18 | 64.0 |

These are covering runs with 240 kWh batteries, 240 kW charging, no shared charger capacity and no return-SOC floor. Fleet proofs concern the saved route pools. Individual-route replay passed; duplicate-trip removal has not been validated. CG time includes importing routes but excludes original graph construction. Later chain pools differ, so this table does not isolate a single code change. [All 36 results, source hashes and proof limits](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/queue_recovery_20260912/status_20260913T051251Z/README.md).

The separate algorithm comparison has finished 11 of its 24 paired allocations; 13 remain active. Analysis of the completed pairs is pending. At the queue check, 50 EVSP–DR jobs were running, with no invalid dependencies. Held historical jobs remain untouched.

Figures remain in [Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) and [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly).

**Earlier results, before the 12 September queue and import repairs.** With set covering and inherited columns, all six chains match the GIRO fleet at k=3–6. Chains 3, 5 and 6 also reach ten buses at k=10. The new 512-route import treatment reaches 11 buses at k=11 in chain 6; other completed extension cases need extra buses.

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

**This table preserves the earlier full-pool runs. The separate 512-route extension is shown below as a comparison. For the repaired full-pool runs through k=15, use the latest table at the top; an old dash here does not mean the case is still blocked.**

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

## **Earlier bottlenecks and what changed**

**Saved-route checking: the earlier workaround, now superseded by indexed full inheritance.** Before starting CG on a larger trip set, we check that routes saved from the previous size are still valid. In chain 1 at k=7, chain 2 at k=9 and chain 4 at k=10, this checking used up the available time. The first workaround checked at most 512 saved routes for 15 minutes, then started CG. The current recovery instead uses an index to check all inherited sequences efficiently; its results are at the top. That earlier bounded experiment found 8, 10 and 11 buses, respectively; only the first and third were proved minimums within those saved pools. Using fewer saved routes may affect the final integer solution, so this is a separate experiment.

**Charger limits made the search for new routes very slow.** In one three-bus test, the LP solved in 0.006 seconds, but one search for a new route took 7.15 hours. The run ran out of time before it could prove that no improving route remained. Its 16-bus integer solution uses only the routes found so far; it does not show that 16 buses are necessary. The timing figures explain these two different delays.

| Case | Baseline | PARX 60 kW | Station capacity | Both |
| ----- | ----- | ----- | ----- | ----- |
| k=1, two duties tested | 1 each | 1 each | 1 each | 1 each |
| k=2, 23 trips | 2 | 2 | 3 | 3 |
| k=3, 35 trips | 3 | 3 | 16 | 16 |

Sixteen is the result in a small generated pool, not a proved physical requirement. Capacity-constrained schedules passed station sweeps. This pilot has no 65% return-SOC floor and is separate from the six baseline chains.

## 

## **Overnight extension results**

16:48 EDT collection: 66/87 original CG certificates, plus the certified chain 2 k14 retry. There are now 69 MIP results under the original extension campaign (34 warm, 35 component), plus the recovered chain 2 k14 result described below. The table below retains the earlier 34 warm results; the new k14 result is 16 buses, pool bound 14, fleet unproved, CG 159.5 minutes, weighted LP 1,400,558.356.

Warm starts check ≤512 saved routes. All CG results below except chain 4 k=15 are pricing-certified. Times include preparation. Chain 4 k=15 has only a restricted LP objective, not a certified full-model bound.

| Chain and target | Buses | Pool fleet bound | Fleet proved in pool | CG minutes | Weighted LP objective |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Chain 1, k=07 | 8 | 8 | Yes | 53.6 | 700313.457 |
| Chain 1, k=08 | 9 | 9 | Yes | 49.5 | 800383.688 |
| Chain 1, k=09 | 10 | 9 | No | 67.9 | 900453.496 |
| Chain 1, k=10 | 11 | 11 | Yes | 91.2 | 1000511.205 |
| Chain 1, k=11 | 13 | 11 | No | 114.1 | 1100537.061 |
| Chain 1, k=12 | 13 | 12 | No | 95.6 | 1200608.942 |
| Chain 1, k=13 | 15 | 13 | No | 152.2 | 1300648.369 |
| Chain 1, k=14 | 16 | 14 | No | 147.9 | 1400650.097 |
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
| Chain 4, k=15 | 18 | 15 | No | 239.1 | 1500637.979 |
| Chain 5, k=11 | 12 | 11 | No | 70.5 | 1100522.228 |
| Chain 5, k=12 | 13 | 12 | No | 109.2 | 1200576.644 |
| Chain 5, k=13 | 14 | 13 | No | 134.6 | 1300625.367 |
| Chain 5, k=14 | 16 | 14 | No | 140.1 | 1400639.061 |
| Chain 5, k=15 | 17 | 15 | No | 209.7 | 1500669.159 |
| Chain 6, k=11 | 11 | 11 | Yes | 32.5 | 1100399.029 |
| Chain 6, k=12 | 13 | 12 | No | 64.6 | 1200464.548 |
| Chain 6, k=13 | 15 | 13 | No | 117.1 | 1300532.041 |
| Chain 6, k=14 | 16 | 14 | No | 136.9 | 1400578.400 |
| Chain 6, k=15 | 18 | 15 | No | 176.3 | 1500606.338 |

Pool fleet bounds concern saved columns only. A proved fleet above target requires different columns to reach target. An unproved gap does not distinguish incomplete integer search from missing useful columns.

### **New results and remaining uncertainty**

**Chain 1 at k=14:** CG finished with a pricing certificate after 147.9 minutes and 1,301 iterations. Its weighted LP objective is 1,400,650.097. The completed MIP found 16 buses, with a saved-pool fleet bound of 14; the fleet is unproved. Individual routes passed replay; duplicate removal and shared capacity remain unverified.

**Chain 4 at k=15:** the MIP found 18 buses with a saved-pool fleet bound of 15; neither fleet optimality nor full-model LP optimality is proved. CG had stopped at its time limit after 239.1 minutes, with last reduced cost −0.1998 against tolerance 0.0001. Individual routes passed replay; duplicate removal and shared capacity remain unverified.

**Chain 2 at k=14, updated 12 September:** CG reached a pricing certificate after 159.5 minutes and 1,484 iterations, with weighted LP objective 1,400,558.356. The subsequent one-hour MIP found **16 buses** and a **pool fleet bound of 14**; it did not prove the best fleet. Individual-route replay passed, but duplicate-trip removal and shared-station capacity were not validated. This is the earlier bounded512 warm start. The new full-pool chain is a separate experiment. The k15 CG continues from the recovered k14 routes, with its MIP queued afterward.

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

**Cluster queue — 12 September**

At 16:57 EDT, 17 EVSP–DR jobs and 9 separate V2G jobs were running, with no invalid dependencies. The first new full-pool CG jobs for chain 3 k11 and chain 4 k10 completed with pricing certificates, and their successors are releasing automatically. The table below is the earlier 16:48 snapshot.

| Work | Running | Waiting for input data |
| ----- | ----- | ----- |
| Full-pool warm chains through k=15 | 6 | 68 |
| Decomposition: shared graph preparation | 2 | 22 |
| Earlier bounded warm chains | 2 | 2 |
| Recovered MIPs with ready CG outputs | 4 | 0 |
| Separate V2G project | 9 | 0 |

We removed 43 obsolete queued entries and recovered nine MIPs whose data were already ready. Five have finished: each found 8 buses for its 8-bus subset, proved the fleet optimal within its saved column pool, and passed individual-route replay.

The remaining waits are genuine: the next k needs the previous k’s routes; a MIP needs its CG output; the larger decomposition runs need their component solutions and shared graph. No dependency is marked impossible.

On Unicorn, run **\~/ladder-lite/drq** for this grouped view, or add **\--details** to see each prerequisite. [Job map and recovery evidence](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/queue_recovery_20260912/README.md).

**Methods and evidence.** CG route cost \= 100,000 \+ electricity \+ 5 per charging start. MIP stage 1 minimizes buses. Stage 2 constrains buses ≤ the validated stage-1 incumbent and minimizes electricity plus start fees.

A pricing certificate establishes the LP result only in its represented graph and tolerance. A MIP proof concerns its saved columns. Fleet attainment, physical replay and shared-capacity validation are separate checks.

[Experiment register](https://github.com/ndandnd/EVSP-DR/tree/codex/parallel-research-20260911/outputs/research_register) · [Dated results and source tables](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_extension_20260912/RESULTS_20260912T193423Z.md) · [Experiment settings](https://github.com/ndandnd/EVSP-DR/tree/codex/parallel-research-20260911/outputs/overnight_extension_20260912) · [Historical document and figures](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit) · [Local prototype tests and charging-cost bug](https://github.com/ndandnd/EVSP-DR/blob/b142f360f8a9018c2f50e3419f0dddcaf87a489d/outputs/algorithm_benchmarks_20260912/README.md)

**Storage, 12 September:** 156 cold files archived losslessly; 133.74 GB freed. Active research and V2G data preserved. [Restore instructions](https://github.com/ndandnd/EVSP-DR/tree/b1129bf4/outputs/storage_cleanup_20260912).

**Blocked decomposition case d00\_g3:** the original run timed out before CG, while constructing the graph. Its MIP remains blocked. A separate diagnostic crashed after 366 seconds (SIGSEGV), before its watchdog limit. Partial samples show JSON serialization while adding charging arcs; A local timer-only test also hung, so the asynchronous sampler has been retired. The exact cluster crash remains unresolved. [Diagnostic evidence](https://github.com/ndandnd/EVSP-DR/blob/57f663eb/outputs/d00_g3_startup_review_20260912/TERMINAL_REVIEW.md). No long retry has been launched.
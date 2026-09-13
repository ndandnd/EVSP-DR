# **EVSP DR Current Research**

Week of 14 September. Results collected 13 September 2026 at 04:36 EDT.

## **All six baseline chains reach 15 buses**

**All 37 CG cases are certified; all 37 completed MIPs match their targets.** Chain 1 k15 is now complete. Every fleet below has a finite-pool proof and individual-route replay.

| Chain | Full-pool buses (target 15\) | Earlier 512-route buses | CG minutes |
| ----- | ----- | ----- | ----- |
| 1 | 15 | 18 | 156.3 |
| 2 | 15 | 17 | 122.7 |
| 3 | 15 | 17 | 33.1 |
| 4 | 15 | 18 | 64.4 |
| 5 | 15 | 17 | 52.3 |
| 6 | 15 | 18 | 64.0 |

These are covering runs with 240 kWh batteries, 240 kW charging, no shared charger capacity and no return-SOC floor. Duplicate-trip removal has not been validated. CG time includes inherited-route import but excludes original graph construction. Earlier 512-route entries are timed incumbents; later inherited pools differ, so this table alone does not isolate a single code change. [All 37 results and exact proof scopes](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/queue_recovery_20260912/status_20260913T063519Z/README.md).

## 

## **Controlled tests: which changes help?**

**All 24 paired allocations are complete.** Three selected inputs were tested in both execution orders, changing one setting per comparison. Timings below are for CG, including loading and import, excluding the final MIP. The percentage reductions must not be added together.

| Change tested | Reduction in total CG time | What else changed? |
| ----- | ----- | ----- |
| Indexed replay, same 512 routes | 12.4–18.1% | Same imported pool and certified LP endpoint |
| Remove unused LP setup, same full pool | 9.3–14.6% | Same imported pool and certified LP endpoint |
| Inherit full pool instead of 512 routes | 40.6–64.5% | More columns; better integer fleets |

For the implementation-only comparisons, imported sequence order and pool hashes, iteration counts, final column counts and certified weighted LP objectives agree. These are descriptive results on three inputs, not a population-wide speed guarantee.

| Input | CG iterations: 512 → full | MIP buses: 512 → full |
| ----- | ----- | ----- |
| Chain 1, k=8 | 950 → 177 | 9 proved → 8 proved |
| Chain 4, k=10 | 945 → 222 | 11 proved → 10 proved |
| Chain 3, k=15 | 891 → 280 | 17 timed → 15 proved |

Both execution orders give these fleet outcomes. All proof labels concern the supplied route pool. For chain 1 k8 and chain 4 k10, the smaller pools provably require 9 and 11 buses, while the richer pools use 8 and 10 at the same certified weighted LP optimum. This directly demonstrates a restricted-column bottleneck. The 17-bus result is unproved, so we cannot rule out 15 buses in that smaller pool.

Original arc scanning did not finish importing the full pool within two hours in any of six runs; no CG iteration completed and the MIPs were intentionally skipped. Indexed full-pool CG finished in 9.1–20.8 minutes in their paired counterparts. These capped runs are not execution failures. [Comparison details, paired timings and hashes](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/controlled_comparison_20260913/status_20260913T063519Z/README.md).

## 

## **Charging-start fee: 5 versus 0**

**All 36 MIPs finished; 35 match their target fleet.** All 36 selected solutions pass individual-route physical replay; 35 have finite-pool fleet proofs. Chain 1 k15 with fee 0 found 16 buses with a pool bound of 15\. CG has 34 certificates; both chain 1 k15 arms reached their two-hour budgets with negative reduced costs remaining.

| Target k | Fee 5 matches (of 6\) | Fee 0 matches (of 6\) |
| ----- | ----- | ----- |
| 5 | 6 | 6 |
| 10 | 6 | 6 |
| 15 | 6 | 5 |

The 16-bus result does not establish a physical need for another bus: the fee-5 schedule uses 15 buses and remains feasible with its start fee set to zero. Its routes may be absent from the fee-0 pool. The fee-0 MIP reached its time limits without closing the 15–16 fleet gap. Both fees start from the same frozen trip sequences, with charging reoptimized. Chain 1 k15 uses its frozen k14 pool; the other 17 inputs use same-k pools. Each run gets two hours of CG and one hour of MIP. [All paired outcomes and stopping reasons](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/zero_charge_start_fee_20260913/status_20260913T083655Z/README.md).

**Completed k5 detail:** every row uses five buses under both fees. Arrows mean fee 5 → fee 0\.

| Chain (k=5) | Charging starts | Electricity cost | End energy (kWh) |
| ----- | ----- | ----- | ----- |
| 1 | 16 → 49 | 150.60 → 145.37 | 38.60 → 11.07 |
| 2 | 10 → 35 | 143.42 → 113.85 | 54.51 → 41.95 |
| 3 | 7 → 27 | 91.56 → 89.11 | 41.95 → 17.17 |
| 4 | 12 → 51 | 166.52 → 151.77 | 43.91 → 11.11 |
| 5 | 17 → 60 | 213.90 → 191.17 | 33.80 → 7.45 |
| 6 | 7 → 34 | 100.93 → 98.14 | 36.81 → 10.86 |

Across all 18 pairs, charging starts increased 2.0–4.9 times (median 3.0). Fee-0 CG took a median 2.9 times as long across 17 certified pairs (range 1.2–15.0); concurrent host load also affects timing. Return energy differs, so electricity differences are not savings at equal energy.

## 

## **GIRO electricity costs with no start fee**

**All six fee/tariff comparisons are complete** (checked 01:41 EDT). The table shows fee 0, electricity only. All schedules use five buses. Optimized schedules return at least the original GIRO total of 280.7833 kWh.

| Price peak | Original GIRO | Fixed duties, charging optimized | Joint route pool |
| ----- | ----- | ----- | ----- |
| 08:00 | 230.29–230.98 | 128.29 | 128.29 |
| 12:00 | 289.59–290.60 | 164.23 | 164.23 |
| 18:00 | 223.45–223.72 | 95.29 | 95.29 |

Optimizing charging on the fixed duties reduces electricity cost by about 43–57% versus the repriced original. The joint pool finds the same cost here: this experiment shows a gain from changing charging, with no additional gain from regrouping trips. GIRO has 52 charging starts; the zero-fee optimized schedules have 46, 43 and 42 starts respectively.

This separate cohort has 62 trips, 240 kWh batteries and 350 kW charging, without shared-station capacity constraints; the six chains use 240 kW. Optimized returning energy is 281.17, 281.17 and 282.90 kWh. Original costs are intervals because the exact within-window power trace is unavailable. Physical replay, trip coverage, returning-energy checks and completion-file hashes passed. The joint MIPs prove five buses within the supplied pools; charging gaps in the grid objective are at most 0.0088%. The table reports continuous replay costs, not a continuous-model optimality proof.

[Both fees: electricity, starts, total cost and returning energy](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/zero_charge_start_fee_20260913/status_20260913T053436Z/giro_costs.csv). [Experiment settings and job records](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/zero_charge_start_fee_20260913/README.md).

No new execution failures or confirmed preemptions appeared. Dependencies remained valid; held historical jobs were untouched.

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
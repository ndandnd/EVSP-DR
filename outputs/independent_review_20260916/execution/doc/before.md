# **EVSP–DR: the research questions**

Evidence checked 16 September, 15:55 EDT. This page contains conclusions and open questions; detailed experiment logs belong in the linked sources.

## **1a. Can sequential CG \+ MIP recover GIRO’s fleet?**

**Sequential CG \+ MIP matches target 32 in chains 2, 4, 5 and 6; chains 1 and 3 reach 31 under baseline covering assumptions. Each step adds one GIRO duty’s trips and retains earlier columns; this uses known duty grouping.**

| Largest target matched | C1 | C2 | C3 | C4 | C5 | C6 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Original one-hour MIP | 26 | 28 | 31 | 29 | 26 | 28 |
| Including separate longer MIPs | 31 | 32 | 31 | 32 | 32 | 32 |

**Actual buses found at the largest targets**

| Target / MIP treatment | C1 | C2 | C3 | C4 | C5 | C6 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 31 / original one-hour MIP | 34 | 32 | 31 | 40 | 36 | 32 |
| 31 / best including completed longer MIPs | 31 | 31 | 31 | 37 | 31 | 31 |
| 32 / original one-hour MIP | 35 | 34 | 33 | 34 | 37 | 33 |
| 32 / best including completed longer MIPs | 35 | 32 | 33 | 32 | 32 | 32 |

Matching the target does not prove global optimality or duplicate-free feasibility. These are covering solutions. Chain 4 matches 32 although its target-31 run still uses 37: separately generated pools and limited MIP searches need not give monotone outcomes.

Baseline: 240 kWh batteries, 240 kW charging, fee 5 per charging start, set covering; no reserve, shared charger capacity or ending-SOC floor. Largest matches are not guaranteed cutoffs. The four target-32 matches contain 114–161 trip IDs assigned to multiple buses per solution; removing duplicates still needs validation. Saved-pool optimality is not full-model optimality.

## **1b. Does a fresh start catch up with the accumulated time?**

**No in the completed comparison: warm 24/24, fresh 6/24.** Fresh CG starts with single-trip routes and receives T \= the sum of earlier import \+ CG times through k. Both pools get the same one-hour MIP allowance.

| Target k | Warm matches | Fresh matches |
| ----- | ----- | ----- |
| 5 | 6/6 | 5/6 |
| 8 | 6/6 | 1/6 |
| 10 | 6/6 | 0/6 |
| 15 | 6/6 | 0/6 |

The largest tested fresh match is k=8, chain 6\. All fresh CGs converged before their time budget expired. Four fresh pools exclude the target; 14 misses remain unresolved. This retrospective comparison has code/hardware differences. The budget excludes intermediate MIPs; graph preparation is accounted for separately.

## **2\. Which algorithm changes worked?**

**Summer:** event-based time/SOC graphs allow more charging choices, and exact shortest-path pricing can certify that no improving route remains in the modeled graph. This is stronger than the older capped label search; it is not a controlled historical speedup claim.

**Recent controlled tests:** indexed route replay reduced CG time 12.4–18.1%; skipping unused LP-matrix construction saved 9.3–14.6%. Keeping all inherited routes rather than 512 cut CG time 40.6–64.5% and improved the three tested fleets from 9→8, 11→10 and 17→15. These are selected-input results; percentages cannot be added. Capacity-pricing caching has not demonstrated a speedup.

## **3\. Without the start fee, does changing duties beat charging optimization alone?**

**Yes at all three tariff peaks on this five-duty set. Fixed GIRO duties with optimized charging versus fresh CG plus duplicate cleanup: 08:00, 128.29 versus 124.69 (2.8% lower); 12:00, 164.23 versus 160.61 (2.2% lower); 18:00, 95.29 versus 88.30 (7.3% lower). Each schedule uses five buses, assigns every trip once and matches its comparator’s total ending energy. Costs use continuous replay, in tariff cost units. The evening cleanup reached its one-hour limit with a 0.136% grid-objective gap; its improvement is verified, but charging optimality is unproved.**

Fresh CG converged in 15.2, 16.8 and 22.2 minutes, excluding graph construction; duplicate cleanup then assigned each trip once. Settings: 240 kWh/350 kW, zero start fee, no reserve or shared capacity, and matched total ending energy—not per-bus SOC. This is an improvement on one instance under simplified physics, not proof of global charging optimality or full GIRO feasibility.

## **Two additional questions that matter**

4\. Where is the bottleneck? At target 31, five CG runs hit the four-hour limit; chain 5 converged in 197.5 minutes. Chain 5 then improved from 36 to 31 buses using unchanged columns and a longer MIP search: those columns were sufficient. Its pool bound is 30, so 30 versus 31 remains unresolved. Pricing certificates concern the modeled graph; MIP proofs concern the saved columns. Duplicate removal still needs validation before calling the larger fleets executable schedules.

**5\. Do the conclusions survive realistic charging constraints?** Reserve, station power/capacity and ending SOC remain incompletely tested together. Keep those results separate from baseline scaling. Decomposition is a possible method for scaling; broader instances and tariffs test whether the findings generalize.

**Next: validate large-chain trip assignments; test charging savings on more instances and under stricter constraints. Historical code audits remain in the linked reports.**  
[Chain evidence and diagnostics](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260916T194843Z/README.md) · [Accumulated-time experiment](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/cumulative_budget_20260913/README.md) · [Algorithm comparisons](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/controlled_comparison_20260913/status_20260913T063519Z/README.md) · [Charging comparison evidence](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/zero_fee_validated_comparison_20260916/README.md)  
[Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) · [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) · [Previous detailed dashboard](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/research_questions_20260915/doc_before.md)· [Trip-assignment audit (16 September)](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/large_chain_cleanup_screen_20260916/README.md)
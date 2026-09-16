# **EVSP–DR: the research questions**

Evidence checked 15 September, 14:26 EDT. This page contains conclusions and open questions; detailed experiment logs belong in the linked sources.

## **1a. Can sequential CG \+ MIP recover GIRO’s fleet?**

**Yes under the baseline model: all six chains reach target 26; four reach 28\.** Each step adds another GIRO duty’s trips and imports earlier generated columns. These are constructed smaller problems, so this is a useful strategy that uses the known duty grouping—not a solve from an unstructured trip set.

| Largest target matched | C1 | C2 | C3 | C4 | C5 | C6 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Original one-hour MIP | 26 | 28 | 27 | 27 | 26 | 28 |
| Including separate longer MIPs | 27 | 28 | 28 | 28 | 26 | 28 |

These are largest observed matches, not guaranteed cutoffs. Tests through 32 are launched. Matching k buses establishes a feasible fleet; a proof using saved columns does not establish the full model’s optimum. Baseline: 240 kWh/240 kW, covering, no reserve, shared charger capacity or ending-SOC floor.

## **1b. Does a fresh start catch up with the accumulated time?**

**No in the completed comparison: warm 24/24, fresh 6/24.** Fresh CG starts with single-trip routes and receives T \= the sum of earlier import \+ CG times through k. Both pools get the same one-hour MIP allowance.

| Target k | Warm matches | Fresh matches |
| ----- | ----- | ----- |
| 5 | 6/6 | 5/6 |
| 8 | 6/6 | 1/6 |
| 10 | 6/6 | 0/6 |
| 15 | 6/6 | 0/6 |

The largest tested fresh match is k=8, chain 6\. All fresh CGs converge before T expires: more ordinary CG time alone would not change their stopping decision. Four fresh pools provably need more than k buses; 14 other misses remain open. This is retrospective: historical code/hardware differ. T excludes intermediate MIPs; graph construction is accounted for separately. There is no mathematical guarantee that either heuristic must dominate.

## **2\. Which algorithm changes worked?**

**Summer:** event-based time/SOC graphs allow more charging choices, and exact shortest-path pricing can certify that no improving route remains in the modeled graph. This is stronger than the older capped label search; it is not a controlled historical speedup claim.

**Recent controlled tests:** indexed route replay reduced CG time 12.4–18.1%; skipping unused LP-matrix construction saved 9.3–14.6%. Keeping all inherited routes rather than 512 cut CG time 40.6–64.5% and improved the three tested fleets from 9→8, 11→10 and 17→15. These are selected-input results; percentages cannot be added. Capacity-pricing caching has not demonstrated a speedup.

## **3\. Without the start fee, does changing duties beat charging optimization alone?**

**Not demonstrated.** On one five-duty set, optimized fixed GIRO duties and the joint saved-pool solution tie at all three tariff peaks: 128.29 (08:00), 164.23 (12:00), and 95.29 (18:00), in tariff cost units. Both use five buses and meet the same aggregate ending-energy requirement. Original GIRO charging repriced under those tariffs costs about 230–231, 290–291 and 223–224.

This is a saved-pool comparison, not a test of full joint optimization. The columns were generated under different conditions. Repricing them and adding fixed-duty alternatives cannot tell us what new routes CG would discover with zero start fee and the return-energy requirement. The reported ties therefore leave the research question unanswered. It uses 350 kW, not the chain baseline’s 240 kW. Therefore GIRO’s duties are competitive in this tested pool—not proved globally optimal. **Full-CG rerun submitted (15 September, late evening): array 275432 tests all three peaks from fresh singletons, with zero start fee, at most five buses, and aggregate return energy at least 280.7833253 kWh. Pricing now includes that energy constraint and retains higher-energy return alternatives. Each case allows four hours of CG plus one hour of two-stage MIP. Native validation passed; results are pending.**

## **Two additional questions that matter**

**4\. Why do we miss a target?** Separate unfinished pricing, missing useful integer routes, unfinished MIP search and model infeasibility. Chain 1 at target 27 improved from 28 to 27 buses with longer search on the same columns. Seven other target-only tests timed out without a solution or an infeasibility proof; those pools remain unresolved.

**5\. Do the conclusions survive realistic charging constraints?** Reserve, station power/capacity and ending SOC remain incompletely tested together. Keep those results separate from baseline scaling. Decomposition is a possible method for scaling; broader instances and tariffs test whether the findings generalize.

**Priorities:** finish chain reach and target-gap diagnoses; complete the fair zero-fee comparison. Keep detailed pool variants and scheduler counts off this page.

[Chain evidence and diagnostics](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260915T141509Z/README.md) · [Accumulated-time experiment](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/cumulative_budget_20260913/README.md) · [Algorithm comparisons](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/controlled_comparison_20260913/status_20260913T063519Z/README.md) · [Charging comparison evidence](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/research_questions_20260915/evidence.json)

[Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) · [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) · [Previous detailed dashboard](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/research_questions_20260915/doc_before.md)
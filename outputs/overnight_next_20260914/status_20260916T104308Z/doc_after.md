# **EVSP–DR: the research questions**

Evidence checked 16 September, 06:49 EDT. This page contains conclusions and open questions; detailed experiment logs belong in the linked sources.

## **1a. Can sequential CG \+ MIP recover GIRO’s fleet?**

**Under the baseline covering model, all six chains match target 26\. Chain 3 reaches 31; chains 1, 2 and 6 reach 30; chain 4 reaches 29, including the separate longer MIPs.** Each step adds another GIRO duty’s trips and imports earlier generated columns. These are constructed smaller problems, so this is a useful strategy that uses the known duty grouping—not a solve from an unstructured trip set.

| Largest target matched | C1 | C2 | C3 | C4 | C5 | C6 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Original one-hour MIP | 26 | 28 | 31 | 29 | 26 | 28 |
| Including separate longer MIPs | 30 | 30 | 31 | 29 | 26 | 30 |

These are largest observed matches, not guaranteed cutoffs. Tests through 32 are launched. Matching k buses establishes a covering solution. An executable schedule also needs each trip assigned once; duplicate removal is not yet validated for every reported selection. A proof using saved columns does not establish the full model’s optimum. Baseline: 240 kWh/240 kW, covering, no reserve, shared charger capacity or ending-SOC floor.

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

**Yes at all three tariff peaks on this five-duty set. Fixed GIRO duties with optimized charging versus fresh CG plus duplicate cleanup: 08:00, 128.29 versus 124.69 (2.8% lower); 12:00, 164.23 versus 160.61 (2.2% lower); 18:00, 95.29 versus 88.30 (7.3% lower). Each schedule uses five buses, assigns every trip once and matches its comparator’s total ending energy. Costs use continuous replay, in tariff cost units. The evening cleanup reached its one-hour limit with a 0.136% grid-objective gap; its improvement is verified, but charging optimality is unproved.**

CG converged in 15.2, 16.8 and 22.2 minutes, excluding graph construction. Its covering solutions repeated trips; the extra cleanup step was necessary to validate executable schedules. Earlier saved-pool ties did not answer this full-CG question. This experiment uses 240 kWh/350 kW, zero start fee, no reserve or shared-station capacity, and an aggregate ending-energy minimum—not a per-bus SOC floor. The improvement is under these simplified assumptions; neither global charging optimality nor full GIRO feasibility is proved.

## **Two additional questions that matter**

4\. Why do we miss a target? CG convergence and integer recovery are separate. At target 31, chain 3 stopped CG at four hours without a certificate, yet its MIP found 31 buses and proved that fleet minimal in the saved pool. Chain 5 certified weighted LP objective 3,001,338.80 (fractional route weight 30\) in 197.5 minutes, but its MIP found 36 buses with bound 30\. Whether its columns can supply 31 buses remains unresolved; a longer unchanged-pool search is running. Fractional weight is not a fleet-only lower bound. The source table now distinguishes a final-pool LP solve from the last solved iteration retained when the final solve times out; this corrects ten endpoint labels, not their values. These integer counts remain covering results pending duplicate-removal validation.

**5\. Do the conclusions survive realistic charging constraints?** Reserve, station power/capacity and ending SOC remain incompletely tested together. Keep those results separate from baseline scaling. Decomposition is a possible method for scaling; broader instances and tariffs test whether the findings generalize.

**Priorities: validate duplicate removal for chain solutions, test the zero-fee result on more instances, and audit historical code paths. The corrected fleet-dual pricing bug is not triggered by the current chain driver, which passes zero fleet dual. Other historical adapters remain to be checked.**

[Chain evidence and diagnostics](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260916T104308Z/README.md) · [Accumulated-time experiment](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/cumulative_budget_20260913/README.md) · [Algorithm comparisons](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/controlled_comparison_20260913/status_20260913T063519Z/README.md) · [Charging comparison evidence](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/zero_fee_validated_comparison_20260916/README.md)

[Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) · [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) · [Previous detailed dashboard](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/research_questions_20260915/doc_before.md)
## **14:29 EDT — warm k=10 CG complete; MIP blocked by license**

## Chain 3 reached fractional fleet 10, combined objective 1,000,316.823186, zero artificials, and minimum reduced cost −4.03×10⁻⁹ (tolerance 10⁻⁴), after 144 CG iterations. The pricing certificate applies to the represented event/SOC graph.

## Total CG-run time: 292.57 min. Importing inherited routes took 282.59 min (96.6%); the remaining CG and overhead took 9.98 min. The frozen pool contains 35,495 columns for 205 trips.

## MIP job 740390 failed before optimization with “Model too large for size-limited license.” It did not receive the correct cluster license setup; the earlier claim that this queued job would use the fix was incorrect. No warm k=10 integer result is available yet. CG need not be repeated to retry the MIP. No jobs are currently active in the retrieved queue.

## Evidence: snapshots/20260910T1829Z\_heartbeat.json and .md; stage2\_cap\_saved\_pool\_reruns/740390.out and 740390.err. Unicorn access healthy. Slides unchanged.

## **12:34 EDT diagnosis:** the saved-start importer dropped the distinction between grid-rounded charging energy and physically realized energy, but retained the grid-based cost. Example: one route was priced at 132.5 kWh while its imported blocks held 124.417 kWh. The original route records are consistent. Preserve both representations, reconstruct the saved selected vectors, and rerun validation; another optimization run should not be needed for that check. This repair has not yet been applied. Details: rejected\_20260910T1627Z/DIAGNOSIS.md.

## **12:27 EDT — reruns rejected at validation**

## Jobs 740435 (warm k=9) and 740436 (fresh covering k=10, chain 3\) finished their one-hour solver budgets, then failed the final cost check on an imported saved-start route. Their 9- and 11-bus solver incumbents are not new validated schedules. Earlier accepted results below remain the baseline.

## Gurobi charging gaps were 7.93% and 21.84%; checkpoints and rejection diagnostics are preserved. Warm k=10 CG continues, with its MIP dependency-pending. Unicorn access is healthy. Evidence: outputs/meeting\_20260910/stage2\_cap\_saved\_pool\_reruns/rejected\_20260910T1627Z/.

## **September 10: answers and presentation**

The answers now appear beside their figures and in the relevant sections below. The separate answers tab remains as reference. Use the coauthor refresher slides to present.

# **EVSP DR research progress**

Discussion for 10 September 2026

We can certify the fractional solution on the current event graph. The central unresolved problem is building a route pool that also supports a good integer solution. Controlled examples now demonstrate where useful routes are missing. Figures for Thursday contains the embedded plots, historical Gantts and mathematical proof.

## **1 What changed since spring**

The current baseline uses 240 kWh batteries and constant 240 kW charging (Compared to 300kWh, 300kW from before). The historical comparison appears once with the Gantt plots. Coverage rows, initialization, route selection and stopping rules also changed. Spring used capped partial-route labels, coarse charging targets, and immediate charging starts. The current DP uses an acyclic event/SOC graph with relevant trip/tariff times, 2.5-kWh SOC states, delayed charging, and packed arcs. Coverage is either exact-one partitioning or at-least-one covering; initial pools may use single-trip RAW routes, GIRO duties, or a greedy partition. Each k restarts with its own duals and pool. A pricing stop is certified only when the represented graph has no reduced-cost violation below tolerance; a time limit is not a certificate.

The remembered RND002 Gantts are May results on 193 trips: GIRO initialization gave 10 buses at all three tariff peaks; without it the incumbents were 12, 12 and 11\. The April 175-trip result was a separate 12-route covering incumbent. No matched current 193-trip run establishes a before-and-after improvement yet.

## **2 What we can now verify**

All six original k=2 cases have fractional fleet 2 and integer fleet 2\. Nested84 CG minimizes one weighted bus-plus-charging objective; there is no target constraint and no separate fleet-only CG phase. For original k=2–14, the feasible fractional route weight equals the independent maximum simultaneous-trip bound, so matching bounds establish the fleet-only LP value. For example, chain 1 k=7 has weighted objective 700,313.457099, fractional route weight 7, and seven simultaneous passenger trips at 07:21. Dividing the weighted objective by 100,000 is not an exact fleet calculation because charging-related cost is present.

The two remaining full-model fleet intervals are 14–15. At k=15 in chains 1 and 3, the half-open simultaneous-trip sweep gives 14 while a validated GIRO witness gives 15; both CG endpoints have zero artificials. Chain 1 stopped without a pricing certificate. Chain 3 certified the combined-cost LP at route weight 15, which does not exclude a fleet-14 solution with higher charging cost. Their frozen-pool MIPs ended with unproved incumbents of 142 and 86 and solver bounds near 15; those solver bounds apply only to the saved pools.

Fresh covering CG completed on chains 1, 3 and 5 at k=5, 8 and 10\. All nine reach fractional route weight k and the pricing certificate. Frozen-pool integer fleets are 5, 5, 6 at k=5; 9, 9, 9 at k=8; and 11, 11, 11 at k=10. Chain 5 k=5 has a separate validated five-bus full-model witness, so its six-bus pool optimum demonstrates missing useful columns. The selected routes pass individual physical replay; duplicate removal and shared charger capacity remain unvalidated where applicable.

**Implemented 10 September policy.** A 3,600-second MIP budget reserves up to 1,800 seconds for Stage 1\. Stage 2 uses the remaining time, constrains fleet to be no larger than the best validated feasible incumbent even if fleet optimality is unproved, and minimizes charging-related cost. New reruns are queued. The numerical results already summarized here predate this policy and retain their original proof scopes.

### **Dependent warm chain, chain 3 — snapshot 20260910T1426Z**

| k | Source | CG total | Inherited import | Other CG | MIP | Fleet result |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 2 | RAW | 1.7m | 0.0m | 1.7m | \<0.1m, optimal | 2, pool proved; exact service |
| 3 | warm from 2 | 5.8m | 4.9m | 1.0m | \<0.1m, optimal | 3, pool proved; exact service |
| 4 | warm from 3 | 11.0m | 9.2m | 1.8m | \<0.1m, optimal | 4, pool proved; exact service |
| 5 | warm from 4 | 32.4m | 25.9m | 6.5m | 0.5m, optimal | 5, pool proved; 3 duplicates |
| 6 | warm from 5 | 81.4m | 74.9m | 6.5m | 0.5m, optimal | 6, pool proved; 3 duplicates |
| 7 | warm from 6 | 108.5m | 102.6m | 6.0m | 4.4m, optimal | 7, pool proved; 5 duplicates |
| 8 | warm from 7 | 137.1m | 133.2m | 3.9m | 5.8m, optimal | 8, pool proved; 2 duplicates |
| 9 | warm from 8 | 215.7m | 200.4m | 15.3m | 60.0m, time limit | 9, fleet proved; charging cost unproved; 4 duplicates |

Every completed warm CG row is pricing-certified with fractional route weight k. CG total includes inherited-event-pool import; “Other CG” is the saved remainder. Each inherited trip pattern is reoptimized and replayed in the child graph. Selected routes are individually valid; shared charger capacity is unchecked, and duplicate removal is unvalidated where duplicates remain.

| Instance | RAW | Greedy | Interpretation |
| :---- | :---- | :---- | :---- |
| chain 2, k=3 | 4 (pool proved) | 4 (pool proved) | No fleet change |
| chain 2, k=5 | 8 (pool proved) | 8 (pool proved) | No fleet change |
| chain 1, k=6 | 14 incumbent; bound 6 | 8 incumbent; bound 7 | Greedy improved the incumbent; neither proved |
| easy k=10 | 12 incumbent; bound 10 | 13 incumbent; bound 10 | Greedy was one bus worse; neither proved |

These pairs match inputs, physics and MIP budgets but use different commits and independently generated pools. They show mixed behavior, not a clean one-factor causal estimate.

The easy ladder is one deterministic chain formed by sorting eligible GIRO duties by regular-trip count and then duty ID. RAW singletons currently recover the target through k=5. GIRO-seeded/augmented initialization—also called GIRO-added or CHEAT in older labels—supplies validated duties and recovers the tested targets through k=10. The only easy greedy run is k=10 and found 13 buses versus RAW’s 12, with neither fleet proved. Solver bounds are frozen-pool bounds, not unrestricted full-model bounds; matching a validated route set to the independent overlap lower bound is what proves the named model’s fleet.

## **3 Why the integer results need not be monotone**

Original chain3 restarted independently at each size. At k=4: 69 trips, 2.6 minutes CG, fractional fleet 4, saved-pool integer optimum 7\. At k=5: 105 trips, 14.2 minutes CG, fractional fleet 5, saved-pool integer optimum 6\. The fifth duty adds 36 trips. Independently replayed GIRO routes prove that the full stated models can use 4 and 5 buses.

The batch contains the exact best route plus at most 29 alternatives. The DP keeps one cheapest prefix to each ending state (time, location and SOC), completes those prefixes to the depot, and deduplicates trip sets. For example, AB may be cheapest at one state while AC and DE end at others. A second-best path to the same state is not enumerated: these are not the global 30 shortest routes. The figure-tab toy shows why this matters: a zero-reduced-cost route AD can be absent from an LP-optimal pool, yet allow a three-bus integer solution where the saved pool requires four.

## **4 Where the proof stops**

With an optimal restricted master and minimum reduced cost nonnegative over every represented route, the master dual is feasible for the full LP. Strong duality proves LP optimality. Our implemented stop uses a reduced-cost tolerance of 0.0001 and feasibility checks. A time or iteration limit is not that certificate. A pool MIP proves only the best integer combination of saved routes.

The controlled chain2 k=3 example is decisive: the same pool gives partition optimum 4 and covering optimum 3\. Adding all three validated GIRO duties restores partition optimum 3\. Their final reduced costs are approximately 0, 0 and \+3.472. Negative-only pricing has no reason to add them. Longer MIP time on the unchanged partition pool cannot fix that example.

## **5 Charging response and a paper direction**

The three-baseline chart covers 08:00, 12:00 and 18:00 for one separate cohort: 62 passenger trips and GIRO duties 13401, 13403, 13405, 13408 and 13414\. All baselines use five buses, 240-kWh batteries and the declared constant 350-kW charging scenario. Fixed-duty DP preserves each bus’s ordered trips and changes charging only; joint optimization starts from those five validated GIRO routes and may reassign trips.

Compare joint directly with fixed-duty optimized: joint minus fixed is −4.78 electricity, −20 start fees and −24.78 total at 08:00; \+9.69, −25 and −15.31 at 12:00; and \+5.52, −35 and −29.48 at 18:00. These are not differences from original repriced charging. The five-unit start fee is modeled, terminal energy differs, and shared capacity is absent, so separate energy, fees and terminal inventory before interpreting savings.

## **6 Decisions to discuss with coauthors**

Prioritize reliable k=2–10 experiments with matched inputs, physics and time budgets before increasing instance size.

Compare independent starts with the new dependent warm-start chain; retain and revalidate earlier columns as duties are added.

Use validated GIRO initialization as a practical schedule-improvement method, while reporting it separately from recovery without GIRO seeds.

Test integer-oriented pool enrichment: zero/near-zero reduced-cost routes, complementary trip coverage and small exact branch-and-price controls.

Measure demand response with fixed fleet and matched terminal energy. Vary tariff amplitude and location; report charging shifted between periods/stations, reassigned trips, savings and the cost of deviating from original duties. Threshold changes are more informative than assuming a smooth elasticity.

Keep the newly recovered nonlinear charging, SOC reserve and shared-charger constraints as a separate model-reconciliation experiment. Six sampled chains characterize these instances; they do not establish a universal maximum solvable k.

## **Dependent warm-chain status**

At the 14:26 UTC snapshot, k=2–9 CG and their MIPs had completed. Each CG endpoint is pricing-certified at fractional route weight k. Frozen-pool fleet is proved at k buses throughout; k=9’s charging-cost stage reached its one-hour limit. Most wall time at the larger sizes is inherited-route replay/import rather than subsequent CG. k=10 had not completed at that snapshot. The new two-stage-policy reruns are separate and queued.

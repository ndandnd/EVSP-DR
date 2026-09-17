# Meeting progress — 17 September 2026

Latest read-only cluster collection: **16 September, 23:44 EDT**. This is the interim meeting briefing requested by the user; the separately gated four-result report still awaits C5 k31's 12-hour MIP. No new jobs or algorithm changes were made for this briefing.

**Suggested opening:** “Sequential column generation now reaches GIRO-sized fleets on much larger instances: with longer integer searches and seed repeats, every chain has a 32-bus solution at target 32. Fresh starts remain much weaker even after comparable accumulated CG time. We have also demonstrated modest charging-cost improvements over optimized fixed GIRO duties on one small instance. The next questions are how to obtain the useful integer routes without the sequential construction, and whether these results survive the stricter operating rules.”

## 1. Sequential runs: the old k=5 headline is obsolete

| Largest target matched | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
|---|---:|---:|---:|---:|---:|---:|
| Original one-hour MIP at each k | 26 | 28 | 31 | 29 | 26 | 28 |
| Including selected longer MIPs and new seed repeats | 32 | 32 | 32 | 32 | 32 | 32 |

“Largest matched” is not a guarantee that every smaller instance succeeded. The original one-hour MIP gave about 30 minutes to fleet search; follow-ups give 3 hours to fleet and 30 minutes to charging, with multiple seeds on selected cases. Each k inherits earlier generated columns and adds another GIRO duty's trips. Earlier CG work and knowledge of duty grouping are part of the method's cost and information advantage.

Baseline: 240 kWh batteries,240 kW chargers, set covering, flat electricity price and a 5-unit charge-start fee; no SOC reserve, shared charger capacity or terminal-energy floor. Latest k32 results pass individual-route replay. Earlier 128 schedules have the separate passenger-assignment/empty-driving audit; that audit was not rerun for these new seed results. Do not call this recovery of every GIRO constraint or silently change the frozen67/35 audit counts.

New same-pool k32 seed outcomes (3hours fleet search each):

| Chain | Seed 0 | Seed 1 | Seed 2 | Saved-pool fleet bound |
|---|---:|---:|---:|---:|
| 1 | 33 | 32 | 32 | 32 |
| 3 | 33 | 32 | 32 | 32 |
| 4 | 32 | 32 | 33 | 31 |
| 5 | 32 | 32 | 35 | 31 |

Chains2/6 already had32-bus results from longer searches. Thus the best recorded result is32 in all six, but search performance is variable. Bounds31 in C4/C5 leave open a one-bus improvement; a 32-bus result there matches GIRO without proving the mixed model's fleet optimum.

## 2. Fresh starts: give the tested sizes, not a universal cutoff

Fresh CG receives the accumulated earlier import+CG time through k. Original MIP budgets are matched. All 24 fresh CGs reached their pricing stopping certificate.

| Target | Sequential target matches | Fresh target matches, original MIP budget |
|---|---:|---:|
| 5 | 6/6 | 5/6 |
| 8 | 6/6 | 1/6 |
| 10 | 6/6 | 0/6 |
| 15 | 6/6 | 0/6 |

The largest target matched by these fresh starts is8, on one chain. Four fresh pools provably exclude their targets; other misses retain open integer gaps. This is a retrospective comparison with source/hardware differences, not a clean isolated causal estimate of warm starting.

**New stronger check:** all 18 fresh-k15 MIP repeats are complete; **0/18 matched 15**, despite 3 hours of fleet search. Actual counts:

| Chain | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| 1 | 18 | 18 | 19 |
| 2 | 17 | 18 | 17 |
| 3 | 17 | 17 | 17 |
| 4 | 19 | 18 | 18 |
| 5 | 16 | 17 | 17 |
| 6 | 18 | 18 | 19 |

Every one still has a pool bound near 15, so none proves that 15 is absent from its pool. More CG tail reduction, missing integer-complementary routes and MIP search difficulty are different issues. The 18 seeds cover six fixed instances, not 18 independent instances.

## 3. Charging: the old tie was a saved-pool experiment

The earlier tie repriced existing columns and inserted fixed-duty alternatives. Actual fresh zero-fee CG followed by duplicate-trip cleanup and charging reoptimization found improvements over the tested fixed-duty schedules:

| Tariff peak | Original GIRO cost interval | Same GIRO duties, optimized charging | Fresh-CG-derived schedule | Reduction versus fixed duties |
|---|---:|---:|---:|---:|
| 08:00 | 230.287–230.981 | 128.293 | 124.692 | 2.81% |
| 12:00 | 289.594–290.597 | 164.231 | 160.605 | 2.21% |
| 18:00 | 223.447–223.723 | 95.285 | 88.296 | 7.34% |

One five-duty/62-trip input, five buses throughout,240 kWh/350kW, zero start fee; no reserve or minimum-charge rule. Both optimized arms have equal achieved aggregate ending energy. These are feasible improvements over the computed fixed-duty comparator, not global charging-optimality proofs. Original within-window power is unknown, hence invoice intervals. Three tariffs are not three independent instances.

**Stricter check, preliminary:** adding a 15% reserve and 3-minute minimum charge still gives a five-bus morning comparison of 157.965 fixed-duty versus 154.921 freshCG (**1.93% lower**). Both serve trips exactly once and pass the physical audit. They share the same minimum ending-energy requirement; freshCG actually returns with 289.44 kWh versus 281.17 kWh, so the saving does not come from returning with less total energy. Evening is 107.189 versus 104.094 (**2.89% lower**), with equal 282.9 kWh ending energy, but the fresh selection retains one duplicate trip and needs its separate dispatch-conversion check. The noon fixed-duty job timed out before its first worker record; there is no comparison. These still use 350 kW and omit shared capacity/setup/per-bus terminal targets, so they are not full-GIRO tests. Source hashes and native physical audits were rechecked for the four reported optimized outcomes.

## 4. Decomposition: operational, but no target recovery demonstrated

One 750-trip, 32-duty parent was split into four eight-duty groups in nine complete partitions. Joining component solutions gave 34–37 buses. Two selection methods then combined pools across partitions; each tested nine controls, 36 pairs and one all-nine union (92 searches on the same parent).

| Recombination | Best fleet found | Saved-pool fleet lower bound |
|---|---:|---:|
| All nine partitions, original selection | 34 | 33 |
| All nine, retaining component LP-support routes | 34 | 32 |

No pair improved on its best contributing partition. Neither all-nine search reached 32 after 4 hours. The second still leaves 32 open. These are pool recombination tests without new routes joining trips across components; they are not successful full-parent CG refinements. Earlier parent graph construction timed out before CG. This parent is not necessarily any chain's k32 input, so compare methods only on matched trip sets.

## 5. What to discuss with coauthors

- **Strong structural result:** separated-group time-only lower bounds equal GIRO's count in all 102 cases. Existing continuous-charging duty witnesses attain them under baseline physics. The nine mixed k−1 cases require mixing; integer electric feasibility remains open.
- **Established speedups:** selected controlled tests measured 12–18% from indexed replay, 9–15% from skipping unused incidence construction, and 41–65% from full-pool inheritance versus 512 routes. They are separate contrasts; do not add percentages. Full inheritance also improved selected integer fleets.
- **Next methodological priority:** generate columns useful for integer combinations, and reduce expensive graph/pool work. Fleet-bound stopping and pricing during a diving heuristic are candidates to benchmark, not demonstrated speedups. The code review's 10×/25-minute estimates are projections.
- **Next scientific priority:** finish the existing stricter-physics and larger-tariff comparisons, maintaining equal fleet and explicit ending-energy assumptions. Do not present the baseline chain reach as a fully realistic operating result.

## Sources

- [Latest P1/F6 collection and native endpoint-control audit](../monitor/20260917T034416Z/snapshot.json).
- [Original chain maxima and longer results](../../../overnight_next_20260914/status_20260916T194843Z/README.md).
- [Original accumulated-time fresh comparison](../../../cumulative_budget_20260913/status_20260916T194843Z/README.md).
- [Audited relaxed three-arm charging comparison](../f6/README.md).
- [Completed decomposition comparisons](../../../overnight_parallel_20260914/status_20260914T225805Z/CURRENT_STATUS.md).
- [Controlled implementation comparisons](../../../controlled_comparison_20260913/status_20260913T063519Z/README.md).
- [Structural bound and exact certificates](../../time_only_vsp_20260916/README.md).
- [Code-review qualifications](CODE_REVIEW_RESPONSE.md).

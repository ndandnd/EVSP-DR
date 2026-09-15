# **EVSP–DR: current results**

**Verified results through 15 September, 04:19 EDT.** Overnight results will replace these tables as they are checked. This page is the summary; the other tabs retain the figures and detailed evidence.

## **What we know**

**All six baseline chains have matched a target of 26 or 27 buses.** Reusing earlier columns helps substantially. The harder model with shared charging limits is still a small-instance research problem.

| Question | Evidence |
| ----- | ----- |
| Does a warm start help? | In the accumulated-time comparison, all 24 warm runs match their targets; 6 of 24 fresh runs do. All 24 fresh CG runs converge. Extra CG time alone therefore does not reproduce the warm results. |
| Why do integer solutions miss the target? | Two causes are now separated. Some saved pools cannot form the target fleet: see chain 1 at k15 below. Others can: separate longer MIP searches recovered 24 of the 25 original k16–25 misses. A longer search for the last case, chain 5 at k25, is now running. |
| Are the stricter GIRO settings solved? | Some one-bus tests work. Two-bus tests with shared charging limits still have unfinished pricing and poor integer solutions. We cannot yet claim that the larger-chain results meet those stricter settings. |

## **How far the baseline chains reach**

**Target k** means the selected trips came from k GIRO buses. Each number below is the largest individual target matched, not a claim that every smaller case succeeded in the original time limit.

| Chain | Largest target matched with a one-hour MIP | Largest target matched including separate longer MIPs |
| ----- | ----- | ----- |
| **1** | **26 buses** | **26 buses** |
| **2** | **27 buses** | **27 buses** |
| **3** | **27 buses** | **27 buses** |
| **4** | **26 buses** | **26 buses** |
| **5** | **26 buses** | **26 buses** |
| **6** | **27 buses** | **27 buses** |

**Model:** set covering, inherited columns, 240 kWh batteries, 240 kW charging, and a fee of 5 per charging start. Shared charger capacity and minimum ending SOC are absent. Individual routes were replayed; removing duplicate trip assignments has not been separately validated.

**Proof scope:** “fleet proved in the saved pool” means Gurobi cannot use fewer buses from those columns. It does not establish the minimum over every feasible route. A CG certificate separately establishes LP convergence for the modeled pricing problem. In the original k16–25 batch, 45 of 60 CG runs have that certificate; 15 reached their time limits.

The LP is not forced to equal the GIRO target. For chain 5 at k27, the saved fractional solution covers all 660 trips with route weights summing to 26.0000 and weighted objective 2,601,198.9161. CG stopped at four hours, so this is a feasible fractional solution, not a certified full-model lower bound. Its integer result is still pending in this collection.

## **Can a smaller inherited start work?**

**Core start:** retain earlier integer-solution routes and every route with positive weight in the earlier LP. **Expanded start:** fill that core to 512 distinct trip sequences. The number 512 is an experimental limit on starting sequences, not a fleet size or final column count.

Both starts match every k8 and k10 case: 24 of 24 results. For target 15, the complete results are:

| Chain | Core: buses used | Expanded: buses used | Core CG: minutes | Expanded CG: minutes |
| ----- | ----- | ----- | ----- | ----- |
| 1 | 16 | 16 | 168.4 | 147.6 |
| 2 | 15 | 15 | 161.2 | 159.6 |
| 3 | 17 | 15 | 50.9 | 46.5 |
| 4 | 15 | 15 | 166.5 | 132.3 |
| 5 | 15 | 15 | 99.4 | 100.2 |
| 6 | 15 | 15 | 98.5 | 77.1 |

All 12 CG runs converged. All fleet counts in this table are proved within their saved pools except chain 3’s core: it found 17, but its bound of 15 leaves the answer open.

Chain 1 pinpoints a missing-column problem. Both smaller pools prove that they need 16 buses; the earlier full pool supports 15 with matched inputs, settings and CG revision. All three LP objectives agree at 1,500,717.3733. More MIP time cannot fix either smaller pool.

Why did converged CG miss useful routes? Of the 15 routes in the known 15-bus solution, 12 trip patterns are absent from the core pool and 11 from the expanded pool. All 15 routes have positive reduced cost at each smaller pool’s final dual prices: route cost − sum of its trip duals \> 0\. They would not lower that LP objective, so pricing that adds only negative-reduced-cost routes has no reason to add them. This explains why LP convergence need not produce a good integer pool; it does not prove that every possible 15-bus solution needs these particular routes.

## **What the stricter tests show**

| Test | Result and limitation |
| ----- | ----- |
| 15% battery reserve; selected depot-speed and capacity variants | Eight of ten one-duty tests recover one bus. Duty 13405 returns two buses in both tested variants; this is a saved-pool result. All ten CGs converge. The battery is 236.44 kWh with a 35.466 kWh reserve. These tests omit a 65% ending-SOC target and nonlinear charging. |
| One duty, 13408: reserve \+ shared capacity \+ PARX at 60 kW | One bus recovered. CG takes 77 minutes, versus 2.2 minutes in its reserve-only baseline. This does not test competition between several buses. |
| Hard capacity-pricing calls, with the same input state and dual prices | The reference finishes in 3.58 and 3.19 hours. The cached version reaches its roughly four-hour deadline in both tests without finishing a call. No speedup is demonstrated on these two states. |

## **Overnight tests: results and remaining work**

**Queue in the 04:09–04:19 EDT collection:** 24 jobs running; 6 waiting for required inputs. No pending job in that check was blocked by an array throttle.

| Experiment | Current result or next answer |
| ----- | ----- |
| Chain continuation through k28 | Chain 1 now matches 26 and chain 2 matches 27\. Chain 3 at target 28 finds 29; its bound remains 28 and CG reached four hours without convergence. |
| Smaller starts at k20 and k25 | Nine MIPs are verified: seven target matches and two open gaps. Both starts match 25 on chain 1 and 20 on chains 4 and 6; chain 6’s expanded start also matches 25\. For target 20, chain 3 finds 24 buses from the core start and 23 from the expanded start. Both bounds remain 20 after three hours of fleet search, despite CG convergence. All 24 CGs have ended: 3 converged and 21 hit four hours. Another 15 MIPs await verification. |
| Which added columns repair a pool? — complete | All 13 paired tests still miss target by one bus: 9 for target 8, or 11 for target 10\. Adding routes used by a donor LP ties with adding the same number of unused donor routes. Neither rule restores a target fleet here. |
| Longer searches at k25 and k28 — running | Chain 5, k25: job 220545 searches the original pool after finding 26 with bound 25\. Chain 3, k28: new job 222757 searches its original pool after finding 29 with bound 28\. Each allows up to three hours for fleet and 3.5 hours total. Both start new search trees, with the same columns and model settings as their original runs. |

New extension submitted at 04:26 EDT: all six chains continue through k29 and k30, using the same frozen random order and model settings. All 12 graphs can run independently; CG keeps its true previous-k dependencies. All 12 graphs in array 224628 were running by 04:33 EDT. These launches occurred after the results collection above.

The monitor checks hourly, investigates failures or blocked dependencies, and verifies results before updating this page.

 It preserves real previous-k dependencies. If the queue thins, useful follow-ups must address a specific unresolved comparison with matched settings; repeating completed runs just to increase the job count is not the goal.

## **Figures and detailed evidence**

[Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) — existing comparison charts and editable explanations. Check each figure’s date and model settings.

[CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) — convergence curves and Gantt plots.

[Dated research log](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow) — earlier tables, assumptions, source links, and history. Earlier snapshots are not current results.

[Verified source report](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260915T080947Z/README.md) · [Chain and accumulated-time results](https://github.com/ndandnd/EVSP-DR/blob/54f0d8dc2ea4a968b7182239610f09da7f1e0b84/outputs/cumulative_budget_20260913/status_20260915T040735Z/README.md).
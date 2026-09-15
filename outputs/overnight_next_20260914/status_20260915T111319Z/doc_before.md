# **EVSP–DR: current results**

**Verified results through 15 September, 06:19 EDT.** Overnight results will replace these tables as they are checked. This page is the summary; the other tabs retain the figures and detailed evidence.

## **What we know**

**The six baseline chains now reach targets of 26–28 buses.** Reusing earlier columns helps substantially. The harder model with shared charging limits is still a small-instance research problem.

| Question | Evidence |
| ----- | ----- |
| Does a warm start help? | In the accumulated-time comparison, all 24 warm runs match their targets; 6 of 24 fresh runs do. All 24 fresh CG runs converge. Extra CG time alone therefore does not reproduce the warm results. |
| Why do integer solutions miss the target? | Two causes are now separated. Longer MIP searches recovered all 25 original k16–25 misses using unchanged columns: these pools already contained a target solution. Some smaller pools cannot form the target fleet, as their bounds now prove. Those need additional columns, not just more MIP time. |
| Are the stricter GIRO settings solved? | Some one-bus tests work. Two-bus tests with shared charging limits still have unfinished pricing and poor integer solutions. We cannot yet claim that the larger-chain results meet those stricter settings. |

## **How far the baseline chains reach**

**Target k** means the selected trips came from k GIRO buses. Each number below is the largest individual target matched, not a claim that every smaller case succeeded in the original time limit.

| Chain | Largest target matched with a one-hour MIP | Largest target matched including separate longer MIPs |
| ----- | ----- | ----- |
| **1** | **26 buses** | **26 buses** |
| **2** | **27 buses** | **27 buses** |
| **3** | **27 buses** | **27 buses** |
| **4** | **27 buses** | **27 buses** |
| **5** | **26 buses** | **26 buses** |
| **6** | **28 buses** | **28 buses** |

**Model:** set covering, inherited columns, 240 kWh batteries, 240 kW charging, and a fee of 5 per charging start. Shared charger capacity and minimum ending SOC are absent. Individual routes were replayed; removing duplicate trip assignments has not been separately validated.

**Proof scope:** “fleet proved in the saved pool” means Gurobi cannot use fewer buses from those columns. It does not establish the minimum over every feasible route. A CG certificate separately establishes LP convergence for the modeled pricing problem. In the original k16–25 batch, 45 of 60 CG runs have that certificate; 15 reached their time limits.

The LP is not forced to equal the GIRO target. For chain 5 at k27, the saved fractional solution covers all 660 trips with route weights summing to 26.0000 and weighted objective 2,601,198.9161. CG stopped at four hours, so this is a feasible fractional solution, not a certified full-model lower bound. Its one-hour MIP found 29 buses, with a saved-pool bound of 26\. That leaves open whether these columns support 26, 27 or 28 buses.

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

### **Larger tests: actual buses used**

Each cell is an integer bus count. Only 3 of the 24 CGs converged; 21 hit four hours. Each MIP allowed up to three hours for fleet search and 3.5 hours total.

| Chain | Target 20: core | Target 20: expanded | Target 25: core | Target 25: expanded |
| ----- | ----- | ----- | ----- | ----- |
| **1** | **21** | **21** | **25** | **25** |
| **2** | **22** | **21** | **31** | **26** |
| **3** | **24** | **23** | **26** | **25** |
| **4** | **20** | **20** | **26** | **26** |
| **5** | **21** | **21** | **26** | **26** |
| **6** | **20** | **20** | **26** | **25** |

Eight matches; eight proved target exclusions; eight open misses. For target 20, the target is ruled out in both chain 1 pools and the core pools of chains 2 and 5\. For target 25, it is ruled out in both pools of chains 4 and 5\. Every other above-target cell remains unresolved.

A bound can rule out the target without proving the exact fleet minimum. Chain 2’s core at target 20 found 22 buses with bound 21: twenty is impossible in that pool, but 21 versus 22 is still open.

## **What the stricter tests show**

| Test | Result and limitation |
| ----- | ----- |
| 15% battery reserve; selected depot-speed and capacity variants | Eight of ten one-duty tests recover one bus. Duty 13405 returns two buses in both tested variants; this is a saved-pool result. All ten CGs converge. The battery is 236.44 kWh with a 35.466 kWh reserve. These tests omit a 65% ending-SOC target and nonlinear charging. |
| One duty, 13408: reserve \+ shared capacity \+ PARX at 60 kW | One bus recovered. CG takes 77 minutes, versus 2.2 minutes in its reserve-only baseline. This does not test competition between several buses. |
| Hard capacity-pricing calls, with the same input state and dual prices | The reference finishes in 3.58 and 3.19 hours. The cached version reaches its roughly four-hour deadline in both tests without finishing a call. No speedup is demonstrated on these two states. |

## **Overnight tests: results and remaining work**

**Live queue at 06:40 EDT:** 30 jobs running; 29 waiting for required inputs. No pending job in that check was blocked by an array throttle.

| Experiment | Current result or next answer |
| ----- | ----- |
| Chain continuation through k28 | Chain 6 matches 28, with the saved-pool fleet minimum proved after 25.1 minutes of fleet search. Its CG hit four hours. Chain 3 at the same target still has 29 buses and bound 28; its longer search is running. |
| Which added columns repair a pool? — complete | All 13 paired tests still miss target by one bus: 9 for target 8, or 11 for target 10\. Adding routes used by a donor LP ties with adding the same number of unused donor routes. Neither rule restores a target fleet here. |
| Longer searches at k25 and k28 | Chain 5, k25 now finds and proves 25 in the original pool after 17.9 fleet-search minutes; charging still reaches the total time limit. All 25 original k16–25 misses are now recovered in separate searches. Chain 3, k28 is still running after its original 29-bus result. These use new search trees with unchanged columns and settings. The 17.9-minute fleet result is below the earlier 30-minute allowance, so extra time alone does not explain the improvement. |

Extension through k29–30: all 12 graph jobs in array 224628 are running, with the frozen random order and model settings unchanged. Each CG waits for its graph and the previous k’s columns; its MIP follows.

New comparison, launched 06:37 EDT: eight MIPs combine the core and expanded pools for the same case; four controls search unchanged expanded pools again. All 12 are running. This tests whether the two sets of columns complement each other. No new CG is run; the same native greedy initialization policy is used.

The monitor checks hourly, investigates failures or blocked dependencies, and verifies results before updating this page.

 Morning consolidation is planned around 09:00 EDT.

## **Figures and detailed evidence**

[Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) — existing comparison charts and editable explanations. Check each figure’s date and model settings.

[CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) — convergence curves and Gantt plots.

[Dated research log](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow) — earlier tables, assumptions, source links, and history. Earlier snapshots are not current results.

[Verified source report](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260915T101137Z/README.md) · [All original chain bus counts and CG stopping reasons](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260915T101137Z/CHAIN_TABLES.md) · [Chain and accumulated-time results](https://github.com/ndandnd/EVSP-DR/blob/54f0d8dc2ea4a968b7182239610f09da7f1e0b84/outputs/cumulative_budget_20260913/status_20260915T040735Z/README.md).
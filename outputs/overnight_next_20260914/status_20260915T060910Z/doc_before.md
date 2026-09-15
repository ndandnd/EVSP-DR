# **EVSP–DR: current results**

**Verified results through 15 September, 01:16 EDT.** Overnight results will replace these tables as they are checked. This page is the summary; the other tabs retain the figures and detailed evidence.

## **What we know**

**The baseline now reaches 25–27 buses on several chains.** Reusing earlier columns helps substantially. The harder model with shared charging limits is still a small-instance research problem.

| Question | Evidence |
| ----- | ----- |
| Does a warm start help? | In the accumulated-time comparison, all 24 warm runs match their targets; 6 of 24 fresh runs do. All 24 fresh CG runs converge. Extra CG time alone therefore does not reproduce the warm results. |
| Why do integer solutions miss the target? | There are at least two causes. Some saved column sets provably cannot form the target fleet. Other sets can: longer MIP reruns recovered 24 of the 25 misses in the k16–25 batch. The remaining miss is chain 5, k25. |
| Are the stricter GIRO settings solved? | Some one-bus tests work. Two-bus tests with shared charging limits still have unfinished pricing and poor integer solutions. We cannot yet claim that the larger-chain results meet those stricter settings. |

## **How far the baseline chains reach**

**Target k** means the selected trips came from k GIRO buses. Each number below is the largest individual target matched, not a claim that every smaller case succeeded in the original time limit.

| Chain | Largest target matched with a one-hour MIP | Largest target matched including separate longer MIPs |
| ----- | ----- | ----- |
| **1** | **25 buses** | **25 buses** |
| **2** | **26 buses** | **26 buses** |
| **3** | **27 buses** | **27 buses** |
| **4** | **23 buses** | **25 buses** |
| **5** | **24 buses** | **24 buses** |
| **6** | **27 buses** | **27 buses** |

**Model:** set covering, inherited columns, 240 kWh batteries, 240 kW charging, and a fee of 5 per charging start. Shared charger capacity and minimum ending SOC are absent. Individual routes were replayed; removing duplicate trip assignments has not been separately validated.

**Proof scope:** “fleet proved in the saved pool” means Gurobi cannot use fewer buses from those columns. It does not establish the minimum over every feasible route. A CG certificate separately establishes LP convergence for the modeled pricing problem. In the original k16–25 batch, 45 of 60 CG runs have that certificate; 15 reached their time limits.

## **Can a smaller inherited start work?**

**Core start:** retain earlier integer-solution routes and every route with positive weight in the earlier LP. **Expanded start:** fill that core to 512 distinct trip sequences. The number 512 is an experimental limit on starting sequences, not a fleet size or final column count.

| Target | Completed MIPs matching target | Still waiting for verified results |
| ----- | ----- | ----- |
| 8 buses | 12 of 12 | 0 |
| 10 buses | 12 of 12 | 0 |
| 15 buses | 6 of 7 | 5 |

Each target has six chains and two starting methods. At chain 3, k15, the 198-sequence core finds 17 buses after three hours of fleet search; 15 remains possible in that pool. The expanded start finds and proves 15 in 4.4 seconds. Both CG runs converge to the same weighted LP objective, 1,500,507.4203. This separates LP convergence from finding a good integer fleet. Chain 4’s core now also matches 15 buses: fleet proof takes 3.5 minutes, and its charging-cost search finishes with a saved-pool proof after 133.8 total MIP minutes.

## **What the stricter tests show**

| Test | Result and limitation |
| ----- | ----- |
| 15% battery reserve; selected depot-speed and capacity variants | Eight of ten one-duty tests recover one bus. Duty 13405 returns two buses in both tested variants; this is a saved-pool result. All ten CGs converge. The battery is 236.44 kWh with a 35.466 kWh reserve. These tests omit a 65% ending-SOC target and nonlinear charging. |
| One duty, 13408: reserve \+ shared capacity \+ PARX at 60 kW | One bus recovered. CG takes 77 minutes, versus 2.2 minutes in its reserve-only baseline. This does not test competition between several buses. |
| Hard capacity-pricing calls, with the same input state and dual prices | The reference finishes in 3.58 and 3.19 hours. The cached version reaches its roughly four-hour deadline in both tests without finishing a call. No speedup is demonstrated on these two states. |

## **What is running overnight, and why**

**Queue in the 01:08–01:16 EDT collection:** 59 jobs running; 36 waiting for required inputs. No pending job in that check was blocked by an array throttle.

| Work | Question it will answer |
| ----- | ----- |
| Chain continuation through k28 | How much farther does inherited-column solving reach? |
| Core versus expanded starts at k20 and k25 | Can smaller starting pools retain good integer solutions at larger sizes? |
| Controlled additions to previously inadequate pools | Which kinds of added columns restore a target fleet? |

The monitor checks hourly, investigates failures or blocked dependencies, and verifies results before updating this page. It preserves real previous-k dependencies. If the queue thins, useful follow-ups must address a specific unresolved comparison with matched settings; repeating completed runs just to increase the job count is not the goal.

## **Figures and detailed evidence**

[Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) — existing comparison charts and editable explanations. Check each figure’s date and model settings.

[CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) — convergence curves and Gantt plots.

[Dated research log](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow) — earlier tables, assumptions, source links, and history. Earlier snapshots are not current results.

[Verified source report](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260915T050815Z/README.md) · [Chain and accumulated-time results](https://github.com/ndandnd/EVSP-DR/blob/54f0d8dc2ea4a968b7182239610f09da7f1e0b84/outputs/cumulative_budget_20260913/status_20260915T040735Z/README.md).
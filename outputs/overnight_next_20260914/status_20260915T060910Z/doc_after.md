# **EVSP–DR: current results**

**Verified results through 15 September, 02:17 EDT.** Overnight results will replace these tables as they are checked. This page is the summary; the other tabs retain the figures and detailed evidence.

## **What we know**

**The baseline now reaches 25–27 buses on several chains.** Reusing earlier columns helps substantially. The harder model with shared charging limits is still a small-instance research problem.

| Question | Evidence |
| ----- | ----- |
| Does a warm start help? | In the accumulated-time comparison, all 24 warm runs match their targets; 6 of 24 fresh runs do. All 24 fresh CG runs converge. Extra CG time alone therefore does not reproduce the warm results. |
| Why do integer solutions miss the target? | Two causes are now separated. Some saved pools cannot form the target fleet: see chain 1 at k15 below. Others can: separate longer MIP searches recovered 24 of the 25 original k16–25 misses. A longer search for the last case, chain 5 at k25, is now running. |
| Are the stricter GIRO settings solved? | Some one-bus tests work. Two-bus tests with shared charging limits still have unfinished pricing and poor integer solutions. We cannot yet claim that the larger-chain results meet those stricter settings. |

## **How far the baseline chains reach**

**Target k** means the selected trips came from k GIRO buses. Each number below is the largest individual target matched, not a claim that every smaller case succeeded in the original time limit.

| Chain | Largest target matched with a one-hour MIP | Largest target matched including separate longer MIPs |
| ----- | ----- | ----- |
| **1** | **25 buses** | **25 buses** |
| **2** | **26 buses** | **26 buses** |
| **3** | **27 buses** | **27 buses** |
| **4** | **26 buses** | **26 buses** |
| **5** | **26 buses** | **26 buses** |
| **6** | **27 buses** | **27 buses** |

**Model:** set covering, inherited columns, 240 kWh batteries, 240 kW charging, and a fee of 5 per charging start. Shared charger capacity and minimum ending SOC are absent. Individual routes were replayed; removing duplicate trip assignments has not been separately validated.

**Proof scope:** “fleet proved in the saved pool” means Gurobi cannot use fewer buses from those columns. It does not establish the minimum over every feasible route. A CG certificate separately establishes LP convergence for the modeled pricing problem. In the original k16–25 batch, 45 of 60 CG runs have that certificate; 15 reached their time limits.

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

Chain 1 gives a firmer diagnosis. Both smaller pools require at least 16 buses, while the earlier full-pool run finds 15 with the same inputs, model settings and recorded CG revision. Their weighted LP objectives agree at 1,500,717.3733. More MIP time cannot produce 15 from either smaller pool: the necessary combination of columns is missing.

## **What the stricter tests show**

| Test | Result and limitation |
| ----- | ----- |
| 15% battery reserve; selected depot-speed and capacity variants | Eight of ten one-duty tests recover one bus. Duty 13405 returns two buses in both tested variants; this is a saved-pool result. All ten CGs converge. The battery is 236.44 kWh with a 35.466 kWh reserve. These tests omit a 65% ending-SOC target and nonlinear charging. |
| One duty, 13408: reserve \+ shared capacity \+ PARX at 60 kW | One bus recovered. CG takes 77 minutes, versus 2.2 minutes in its reserve-only baseline. This does not test competition between several buses. |
| Hard capacity-pricing calls, with the same input state and dual prices | The reference finishes in 3.58 and 3.19 hours. The cached version reaches its roughly four-hour deadline in both tests without finishing a call. No speedup is demonstrated on these two states. |

## **Overnight tests: results and remaining work**

**Queue in the 02:09–02:17 EDT collection:** 29 jobs running; 17 waiting for required inputs. No pending job in that check was blocked by an array throttle.

| Experiment | Current result or next answer |
| ----- | ----- |
| Chain continuation through k28 | Chains 4 and 5 now match 26\. Chain 3 at target 28 finds 29; its pool bound is 28 and CG reached four hours without convergence. |
| Smaller starts at k20 and k25 | First three MIPs match 25, 25 and 20, with fleet and charging-cost proofs in their saved pools. Their CGs hit four hours without convergence. The other 21 MIPs are not yet verified. |
| Which added columns repair a pool? — complete | All 13 paired tests still miss target by one bus: 9 for target 8, or 11 for target 10\. Adding routes used by a donor LP ties with adding the same number of unused donor routes. Neither rule restores a target fleet here. |
| Longer search for chain 5, k25 — running | Job 220545 searches the same 207,717-column pool that found 26 buses with bound 25\. It allows up to three hours to minimize fleet, then uses the remaining time for charging (3.5 hours total). |

The monitor checks hourly, investigates failures or blocked dependencies, and verifies results before updating this page. It preserves real previous-k dependencies. If the queue thins, useful follow-ups must address a specific unresolved comparison with matched settings; repeating completed runs just to increase the job count is not the goal.

## **Figures and detailed evidence**

[Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) — existing comparison charts and editable explanations. Check each figure’s date and model settings.

[CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) — convergence curves and Gantt plots.

[Dated research log](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow) — earlier tables, assumptions, source links, and history. Earlier snapshots are not current results.

[Verified source report](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260915T060910Z/README.md) · [Chain and accumulated-time results](https://github.com/ndandnd/EVSP-DR/blob/54f0d8dc2ea4a968b7182239610f09da7f1e0b84/outputs/cumulative_budget_20260913/status_20260915T040735Z/README.md).
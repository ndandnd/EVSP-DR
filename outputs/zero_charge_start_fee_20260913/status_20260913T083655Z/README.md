# Charge-start fee experiment complete — 13 September, 04:36 EDT

**All 36 MIPs finished. 35 match the target fleet.** All 36 selected solutions pass individual-route physical replay; 35 have a finite-pool fleet proof. All 36 CG runs finished: 34 certified and two reached the two-hour CG budget. No execution failure or confirmed preemption occurred in the final four allocations.

| Target buses | Fee 5 matches / 6 | Fee 0 matches / 6 | Exception |
|---|---:|---:|---|
| 5 | 6/6 | 6/6 | None |
| 10 | 6/6 | 6/6 | None |
| 15 | 6/6 | 5/6 | Chain 1 with fee 0: 16 buses, pool bound 15 |

## Removing the fee changes the search

Across the 17 input pairs with certified CG under both fees, observed CG time with fee 0 was **1.23–15.02 times** the fee-5 time, with a median ratio of **2.91**. These are paired observations on selected inputs, not repeated timing trials or a population-wide estimate. Runs executed concurrently, so host load also affects timings. CG time includes loading and inherited-route import, excludes original graph construction and the final MIP, and subtracts recorded telemetry overhead.

Charging starts increased in **all 18 pairs**, by factors of **2.00–4.86** (median 3.01). Among the 17 pairs retaining the same fleet, electricity expenditure fell in 16 and rose in one. Return energy differs and charging optima are not proved in every pool, so these are attained cost differences, not controlled savings at equal energy.

| Chain, target 15 | Buses, fee 5 → 0 | CG minutes, fee 5 → 0 | Charging starts, fee 5 → 0 |
|---|---|---|---|
| 1 | 15 → 16 | 119.1 → 119.2, both capped | 79 → 185 |
| 2 | 15 → 15 | 12.4 → 29.5 | 79 → 158 |
| 3 | 15 → 15 | 6.6 → 17.4 | 51 → 119 |
| 4 | 15 → 15 | 8.8 → 90.9 | 55 → 162 |
| 5 | 15 → 15 | 6.3 → 92.8 | 57 → 165 |
| 6 | 15 → 15 | 11.0 → 19.9 | 70 → 158 |

## Why chain 1 stopped at 16 buses

Both fee arms used the same frozen k=14 starting sequences. Each CG run exhausted its two-hour budget. Fee 0 ended after 441 iterations with restricted LP objective 1,500,558.572039, fractional route weight 15, and minimum reduced cost -1.086604. Fee 5 ended after 364 iterations with objective 1,500,717.627684, route weight 15, and minimum reduced cost -0.250562. Negative reduced costs remained, so neither objective is a certified full-model LP bound.

The fee-0 MIP's fleet stage found 16 buses with bound 15 after about 1,806 seconds: a 6.25% relative fleet gap. Its second stage minimized charging with fleet constrained to at most 16 and the remaining budget; the final solution still used 16. Both stages stopped at their time limits. Fee 5 proved a 15-bus fleet in about 189 seconds; its charging stage used the remaining time.

**Removing an objective fee does not change physical feasibility.** The physically replayed 15-bus fee-5 schedule remains a feasible schedule when its start charges are repriced at zero. Thus the 16-bus result is a search/pool outcome, not evidence that fee 0 physically requires another bus. This is a transfer argument from identical physical constraints; we have not established that all routes of that 15-bus schedule occur in the fee-0 saved pool. With bound 15 and incumbent 16, the fee-0 pool solve itself does not settle whether its own columns can attain 15.

The targeted follow-up would reprice and add the verified fee-5 incumbent to the fee-0 pool, then rerun only that MIP. It would measure the benefit of transferring an integer solution between objectives. This follow-up was **not launched by the monitor**. Preserve both original pools and the completed attempts.

## Scope and evidence

Both fees use covering, 240 kWh batteries, 240 kW charging, flat electricity prices, no shared charger capacity and no return-energy floor. The fee changes from 5 to 0 in CG and final MIP. Matched input/starting-sequence provenance, code pins and resource requests remain in the [launch record](../README.md). Chain 1 k=15 uses frozen k=14 sequences; the other 17 inputs use same-k sequences. Charging metrics concern selected routes before duplicate-trip removal; shared capacity is not established. The chain 1 selected solutions still have duplicate trip coverage, whose removal has not been independently validated.

[All 18 paired costs, starts, returning energy and proof fields](paired_results.csv), [compact paired table](paired_results.md), [validation and timing statistics](validation.json), and [source collection](collection.json). All source CG hashes match their MIP references and cost components reconcile. The [separate completed GIRO study](../status_20260913T053436Z/README.md) retains the comparison with the original GIRO charging and a common returning-energy floor.

Snapshot `outputs/post_meeting_20260910/monitor/20260913T083655Z.json`, SHA-256 `e861495814e301734f1847154dd9d4992b47d24607e02f1aef7f4b56191f7cbf`. Scheduler completion, time limits, pricing certificates, finite-pool proofs, physical replay and target attainment remain separate. Held historical tasks, V2G work and existing decomposition dependencies were untouched.

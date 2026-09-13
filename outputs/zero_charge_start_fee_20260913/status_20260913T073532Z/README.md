# Charge-start fee comparison — 13 September, 03:35 EDT

**32 of 36 MIPs have finished. All 32 match their target fleet**, with finite-pool fleet proofs and individual-route physical replay. All k=5 and k=10 cases are complete under both fees. Four k=15 MIPs remain running.

| Target buses | Fee 5 finished / 6 | Fee 0 finished / 6 | Buses in every finished case |
|---|---:|---:|---:|
| 5 | 6/6 | 6/6 | 5 |
| 10 | 6/6 | 6/6 | 10 |
| 15 | 5/6 | 3/6 | 15 |

The new results are chain 1 k=10, chain 5 k=10 and chain 2 k=15, all with fee 0. At k=15, fee 5 is complete for chains 2–6; fee 0 is complete for chains 2, 3 and 6. Remaining MIPs are chain 1 under both fees and chains 4 and 5 with fee 0. They retain their original budgets and paths; no retries were submitted.

**CG is finished for all 36 cases: 34 have pricing certificates, and two reached their time limits.** Both capped cases are chain 1 k=15. Negative reduced costs remained, so their restricted LP objectives are not certified full-model bounds. Their saved pools are still usable by the MIP.

| Chain 1, target 15 | CG iterations | Final restricted LP objective | Fractional route weight | Last minimum reduced cost | Stop |
|---|---:|---:|---:|---:|---|
| Fee 5 | 364 | 1,500,717.627684 | 15.000000 | -0.250562 | Two-hour CG budget |
| Fee 0 | 441 | 1,500,558.572039 | 15.000000 | -1.086604 | Two-hour CG budget |

The recorded fractional route weight of 15 does not certify pricing optimality. The negative reduced costs show that further weighted-objective improvement remains possible. Both processes exited successfully and saved their pools; these are time-capped scientific results, not execution failures.

Settings are unchanged: covering, 240 kWh batteries, 240 kW charging, flat electricity prices, no shared charger capacity and no return-energy floor. Paired fee runs start with identical frozen sequences. Chain 1 k=15 starts from a frozen k=14 pool; the other 17 inputs use same-k pools. Unequal return energy prevents interpreting electricity differences as savings at equal energy. Charging metrics concern selected routes before duplicate-trip removal; shared charger capacity is not established.

[Paired costs, charging starts, returning energy and proof fields](paired_results.csv) contain all 18 comparisons. [Validation](validation.json) checks target attainment, fleet proof, physical replay, cost reconciliation and source CG hashes for every finished MIP. [Collection](collection.json) retains source paths and hashes. The [separate completed GIRO comparison](../status_20260913T053436Z/README.md) retains the equal-return-energy cost result.

Source snapshot: `outputs/post_meeting_20260910/monitor/20260913T073532Z.json`, SHA-256 `4ec5da60ed0b4e77268687374bc4a07443d4dfc44d1132834956747f5e923761`. No new execution failures, confirmed preemptions or invalid dependencies appeared. Held historical jobs and V2G work were untouched.

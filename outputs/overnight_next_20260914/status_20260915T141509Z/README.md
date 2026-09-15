# Research update — 15 September, 10:34 EDT

**Combining the smaller pools did not improve any of the eight bus counts.** Four unchanged-pool controls also tied their combined-pool counterparts. Only C1 target20 rules out its target in the combined pool: its minimum is21. The other seven target gaps remain open. [Exact union/control results](compact_union_results.csv) · [Source and physical-replay checks](compact_union_validation.json).

The current Google Doc now has seven editable tables, with actual fleet counts, CG minutes and stopping reasons, model settings, and the completed pool comparison explained beside its table. Existing figures and history are preserved; Slides were not edited. [Saved dashboard](doc_after.md) · [Export checks](doc_verification.json).

| Newly verified original MIP | Buses found | Fleet bound in its pool | Target proved impossible? |
|---|---:|---:|---|
| C4, target28 |29|28|No; gap open|
| C5, target28 |34|27|No; gap open|

Both selected solutions pass individual-route physical replay. Both CG runs hit four hours without a pricing certificate. Neither the restricted LP objective nor the pool MIP bound is a full-model fleet proof. All six baseline chains still reach target25 when separate searches are included; C2/C6 reach28 originally, and C3 reaches28 in a separate unchanged-pool search. [Original chain table](CHAIN_TABLES.md).

The baseline uses covering, inherited columns,240kWh/240kW and route cost100000+electricity+5 per charging start. It omits reserve, shared charger capacity and an ending-SOC floor. Duplicate-trip removal has not been separately validated. The stricter GIRO model remains a separate small-instance study; large baseline matches are not evidence that all GIRO constraints are met.

## Useful work launched

| Campaign | Jobs and budgets | What it resolves |
|---|---|---|
| [Original pools C4/C5 target28](../../continuation_gaps3_20260915/README.md) |228579/228580;3h fleet,3.5h total | Whether longer integer search recovers28 without adding routes. Same frozen native solver/settings/source pools; new search trees. |
| [Six chains through31–32](../../chain_extension_31_32_20260915/README.md) |Graph228593_0–11;CG/MIP228594–228617;4hCG/1hMIP | Extend the same frozen random orders. All12 graphs run independently; CG keeps true prior-k and own-graph dependencies. Graph24h watchdog/25h allocation follows measured earlier12h exhaustion. |
| [Direct union target tests](../../union_target_feasibility_20260915/README.md) |228654–228660;1h solver each | Solve min0 with binary covering and fleet<=k on the seven open combined pools. Feasible proves existence; infeasible excludes only this pool; timeout without a solution leaves it open. No charging stage, new columns or supplied incumbent. |

All new production jobs use default_partition, requeue with private restart paths, and exclude scaglione-compute-01. MIPs request8CPUs/24GB; chain CG8CPUs/96GB; graphs2CPUs/64GB. Root verified source hashes, native validation and effective scheduler settings. Target-test fixtures cap119/cap1 validate the implementation; they are not scientific target recoveries. Frozen871d057 plus the separately hashed adapter identifies the new target test executable.

At14:34UTC, **36 jobs run and49 wait for required inputs**, excluding33held historical jobs. All24 graph builders for targets29–32 and all seven target tests run. No array throttle is blocking this batch. Held historical and EVSPV2G work remain untouched. [Timestamped queue](final_queue.json). SSH works; no new confirmed preemption or invalid dependency was observed. Hourly monitoring remains active.

## Record boundary

Main scientific snapshot20260915T141509Z ran14:15:09–14:24:27UTC,557.776s; SHA2563b438e528d6f2a1364951686cd45e36fc77b79bbb361492ea84b978002574da2. It contains14new MIP endpoints:12union/control and two original chains. Original extension totals77CG/77MIP across90cases. Register/workbook3308records/75groups, exact six historical supplements;321core checks,187evening,83pool and154original-chain endpoints pass. Preemption study960attempts.

New submissions followed the campaign scan, so their timestamped launch collections and receipts are linked separately. Do not change the raw snapshot or add a seventh supplement to make the timestamps appear simultaneous. Shared collector/normalizer support is prepared for the next hourly collection; direct target tests have their own proof scope and validation-only records. [New endpoints and hashes](new_endpoints.csv).

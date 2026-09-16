# Authorized F6 follow-up: k=5, reserve and minimum charging duration

**Submitted and running: jobs341682–341687, verified16September2026 at22:55UTC.** Exactly six solver jobs: fresh CG and fixed-duty charging under each of the original08:00/12:00/18:00 synthetic tariffs. Original GIRO repricing is read-only. No other submissions, retries, or partition changes are included.

Question: on the same audited five-duty,62-trip cohort, does changing trip assignments still save charging cost after requiring a15% SOC reserve and at least3-minute active charging periods?

| Held fixed from the prior F6 comparison | Setting |
|---|---|
| Battery / initial energy |240kWh /240kWh per bus|
| Maximum charging power |350kW; same simplified value as prior F6|
| Charging-start fee |0|
| Input |Same62 trips; duties13401,13403,13405,13408,13414|
| Tariffs |Exact original peak08/12/18 CSV hashes|
| Fleet |At most5 buses|
| Return energy |Aggregate minimum280.7833253kWh in both optimized arms|
| Graph |2.5kWh SOC grid;5min event grid|
| Shared charger capacity |Unconstrained|

The changes are a36kWh SOC floor and exclusion of grid charging windows shorter than3min. To ensure **actual positive charging** lasts at least3min after SOC rounding is removed, required energy is spread uniformly across each full retained window, at controllable power no greater than350kW. This continuous charging profile differs explicitly from earlier maximum-power prefix realization. Zero-energy visits are idle and reported separately. The graph remains a discretized approximation; its certificate does not prove optimality over every continuous charging schedule.

Fresh CG starts from singletons and phase-I artificial variables, with no saved, fixed-duty, or GIRO columns. Fixed-duty charging retains the five GIRO passenger-trip sequences and optimizes charging time/location using complete terminal-energy frontiers in the same graph. It does not fix original charging stations. Both arms enforce the same reserve, minimum charging duration and aggregate return-energy constraint.

Fresh-CG allowance:4h plus1h two-stage MIP. Fixed-duty frontier allowance:1h plus1h two-stage MIP. MIP stage1 receives30min for fleet, then remaining time minimizes charging with fleet<=incumbent. Previous same-cohort CG took15–22min plus about3.5min graph construction;4h leaves room for harder restrictions. All six jobs request8CPUs/16GB on default_partition, exclude scaglione-compute-01, with6h or3h allocation respectively. No artificial concurrency throttle. Automatic requeue is disabled for this bounded six-job authorization.

**Original GIRO check verified:** all five original modeled schedules pass the new36kWh reserve and350kW power/window validation. Their lowest model-replayed SOC is45.5146193kWh; shortest positive charging window is4min; aggregate ending energy is280.7833253kWh. These checks establish existence of a feasible power-bounded profile within recorded windows, not an observed power trace. Original invoice intervals are preserved because within-window charging power is unobserved.

Three code tests passed locally and natively, including a binding reserve test covering both pricing and fixed-duty frontiers. A native Gurobi two-trip smoke test certified CG, solved both one-bus MIPs with the same graph cost, and independently replayed their36kWh floor and>=3min active sessions. Execution commit: `a12ced439503bb3b70a0ca0418661ce5f8fb35a2`, isolated from the ongoing k15 experiment.

Every production endpoint gets independent timing/energy/power/reserve/duration replay and explicit duplicate-trip counts. Charging-cost ratios are allowed only for physically valid five-bus solutions. Fewer-bus outcomes are reported separately; incomplete/infeasible/no-incumbent cases are censored, not assigned an improvement ratio. Both achieved terminal energies are reported because a common minimum does not force identical ending energy.

Native root: `/home/nc437/ladder-lite/advisor_f6_k5_reserve_20260916`. `manifest.json` binds code, original input/tariff hashes, resources, original invoice source, and exact CLI settings. `original_revalidation.json`, `native_tests.log`, `native_smoke/passed.json`, and `validation.json` hold evidence. Prior outputs remain unchanged.

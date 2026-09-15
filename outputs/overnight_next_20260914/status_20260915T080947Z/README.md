# Research results — 15 September, 04:19 EDT

**All six baseline chains have matched a target of 26 or 27 buses.** New one-hour MIPs prove 26 buses in the saved pool for C1 k26 and 27 for C2 k27. Largest individual one-hour target matches, by chain1–6, are **26/27/27/26/26/27**. This does not mean every smaller case matched within its original hour.

The new results use set covering, inherited columns,240kWh/240kW and fee5 per charging start. Shared charger capacity, reserve and minimum ending SOC are absent. Individual routes pass recorded physical replay; duplicate removal remains unvalidated. Both new fleet minima are proved only within their saved pools; charging optimality is open and their CGs were time-capped. C1 has246376 columns and11.49minutes of fleet search; C2 has190532 columns and18.61minutes.

**A direct LP check answers the concern that the fractional fleet always equals the target.** C5 k27 covers all660 trips with fractional route weight26.0000000000 and weighted objective2,601,198.9161064. Independently summing its positive routes gives26.00000000000489 and2,601,198.916106943; minimum trip coverage0.9999999999995083. Native artificial weight is zero. This uses the final pool re-solve, not the slightly different last-iteration objective. CG reached four hours with negative reduced cost still reported (−0.97052568), so this is a feasible fractional solution, **not a certified full-model lower bound**. Its integer result was not yet published in this scientific collection. [Calculation and checks](c5_k27_fractional_audit.json).

**The C1 k15 diagnosis is more specific now.** Both compact pools prove16 buses, while a matched full-pool run has15. Inputs, model settings and recorded CG revision match, and weighted LP objectives agree within1e-6 at1,500,717.373326. More MIP time cannot repair either compact pool. [Matched-model and pool audit](../status_20260915T060910Z/c1_k15_pool_limitation_audit.json).

A new read-only audit compares the known15 selected routes against every compact-pool column. Twelve of their15 binary trip-coverage patterns are absent from the core pool;11 are absent from the expanded pool. All15 routes have **positive weighted reduced cost** at both compact pools’ final trip duals: c_r − Σᵢ aᵢᵣπᵢ >0. Ranges are0.095671–42.611723(core) and0.048995–42.693817(expanded). Pricing that seeks only negative-reduced-cost routes has no reason to add these particular integer-useful routes after convergence. This is evidence of a mechanism, not proof that every possible15-bus solution requires these routes or positive-reduced-cost columns. No new optimization or physical replay was performed. [Route-level table](c1_k15_donor_incidence.csv) · [Source hashes and audit](c1_k15_donor_incidence.json) · [Checks](c1_k15_donor_incidence_validation.json).

**Larger compact starts: nine verified MIPs, seven target matches and two open integer gaps.** All24 CGs have ended:3 certified and21 time-capped. Fifteen MIPs are not yet scientifically verified. Early completion selects this subset; seven of nine is not the overall success rate.

| Chain and target | Starting method | Integer buses | Saved-pool fleet bound | CG minutes and stop | Fleet-search minutes | Total MIP minutes |
|---|---|---:|---:|---|---:|---:|
| C1 k25 | Core | 25 | 25, proved | 239.1, time limit | 0.5 | 5.7 |
| C1 k25 | Expanded to 512 sequences | 25 | 25, proved | 239.2, time limit | 0.5 | 6.2 |
| C3 k20 | Core | 24 | 20, gap open | 110.7, converged | 180.0 | 210.1 |
| C3 k20 | Expanded to 512 sequences | 23 | 20, gap open | 102.7, converged | 180.0 | 210.1 |
| C4 k20 | Core | 20 | 20, proved | 239.1, time limit | 0.4 | 11.9 |
| C4 k20 | Expanded to 512 sequences | 20 | 20, proved | 239.1, time limit | 0.4 | 47.4 |
| C6 k20 | Core | 20 | 20, proved | 239.2, time limit | 0.1 | 11.2 |
| C6 k20 | Expanded to 512 sequences | 20 | 20, proved | 239.2, time limit | 0.1 | 5.4 |
| C6 k25 | Expanded to 512 sequences | 25 | 25, proved | 239.2, time limit | 1.1 | 30.8 |

All nine pass recorded individual-route replay. The seven target matches also prove the charging-related objective within their saved pools. C3 k20 core finds24/bound20 and expanded23/bound20 after three hours of fleet search, despite both CGs converging to the same weighted objective2,000,780.100492. These are **open search gaps**, unlike C1 k15’s proved pool limitation. The larger cohort uses a0e0 CG and the smaller cohort e091; do not treat their comparison as a pure size-only experiment. [All24 cases and source hashes](compact_large_results.csv).

**Newly launched at04:26 EDT: extend the six frozen chains through k29–30.** Array224628 prepares all12 graphs independently;12 CGs preserve previous-k dependencies and12 MIPs depend only on their own CG. No input resampling or algorithm change. Graphs have24h watchdog/25h allocation based on the observed12h native graph timeout; CG4h and MIP1h scientific budgets are unchanged. All default-partition CPU jobs exclude scaglione-compute-01. [Frozen design, native validation and launch verification](../../chain_extension_20260915/README.md). This launch is later than the scientific snapshot and will enter the next collection. At04:33EDT all12new graphs are running; the user queue excluding held537227 has35running/30pending and no invalid dependency.

Longer unchanged-pool searches220545(C5k25) and222757(C3k28) continue. Neither has a verified endpoint in this collection. [Status and proof checks](longer_gap_results.csv). No duplicate longer attempt, cancellation or requeue was launched here; held historical and V2G jobs were untouched.

Queue in the full collection:24 running and6 genuine input dependencies,33held historical tasks excluded. No new execution failure, confirmed preemption or invalid dependency. MIP207675 completed after its campaign scan; its native path/hash is [retained for next verification](late_scheduler_results.json), not counted in the nine MIPs above.

Snapshot20260915T080947Z began08:09:47UTC and finished08:18:55UTC after548.1seconds. SHA256 d8935da3e6760d8d1ad5073cb5168db634f00c5f824624ae40e7aab7377c537b. Preemption study929attempt records. Register/workbook3220records across72source groups, exact six supplements retained. Checks cover321core endpoints,175evening endpoints and68large/pool-diagnostic endpoints. The two longer-search campaigns are registered; results remain pending. [Six new endpoints](new_endpoints.csv) · [Pool checks](pool_experiments_validation.json) · [Document verification](doc_verification.json).

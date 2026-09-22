# Scoped operations — 22 September 2026, 07:57:42 UTC

One SSH collection succeeded after reading the remote resource policy. The preceding successful operations collection was 05:23 UTC, outside the 90-minute reuse window. No submissions, restarts, code changes, shared-plan/register edits or live-artifact edits were made. Completed salvage and capacity-pilot results were not recollected.

## Queue and baseline progression

**35 jobs run: 32 baseline graphs and three k33 CGs (C1/C3/C4).** Twelve of the 44 graph allocations have completed, versus five in the earlier snapshot. The 93 remaining baseline solver jobs wait on genuine registered dependencies; there is no broken active dependency. No utilization recovery is needed. The expanded user queue has 147 pending tasks, including preserved historical holds; these are not all ready work. Historical held537227 and V2G work remain untouched.

| Current CG | Job | Saved checkpoint | Certificate / endpoint |
|---|---:|---|---|
| C1 k33 |661618|Initializing; 0 saved iterations|Running, no endpoint or pricing certificate|
| C3 k33 |661668|274 iterations; route weight33; weighted RMP3,301,396.324078; min RC−0.012984|Running and uncertified|
| C4 k33 |661691|24 iterations; route weight32; weighted RMP3,201,445.894562; min RC−100,013.520739|Running and uncertified|

These are saved in-progress RMP values, not final full-model lower bounds or integer results. No baseline k33+ MIP endpoint exists yet. Completed graph task IDs are0,4,16,21,22,23,24,25,31,38,39,40 within array661616. Cache manifests and source identities are retained, without downloading the large graph payloads.

Accounting retains **10 cumulative preempted graph attempts totaling103,079 seconds (28h37m59s)**. All ten affected graph tasks have live replacement attempts; do not duplicate them. Four long discarded attempts each consumed about6h48m; the count is cumulative campaign evidence, not ten new preemptions in this check. Current graph jobs have no partial-build recovery. This reinforces the separate checkpoint/reuse backlog without authorizing changes to their pins.

## Strict k19 endpoint — limited by its unpriced initial pool

Parent prefix19 contains **331 trips from11 reference18E2duties**. MIP668434 completed0:0 in1h00m14s, with **65 buses / finite-pool lower bound64**, unproved and far above target11. Both optimizer stages reached TIME_LIMIT. This does not prove the full strict model needs64buses: the CG had spent16,292.771283s building its graph inside a nominal4hallowance and performed **zero pricing iterations**. The8,397-column initial pool contains8,343 inherited routes and54new singleton routes; the final65-route selection uses all54singletons plus11inherited routes.

| Quantity | Verified value |
|---|---:|
| Fleet incumbent / finite-pool bound |65 /64|
| Fleet optimizer runtime |1,800.086534s|
| Charging incumbent / bound |774.872 /380.665201|
| Charging optimizer runtime |1,800.134786s|
| Wrapper wall |3,609.954701s|
| Scheduler elapsed |3,614s|
| Batch MaxRSS |1,130,064KiB|
| Selected-route extra trip assignments |73 across61 trips|

Recorded covering validation confirms every trip is served. Individual exact-event route feasibility is by construction; this collection did not independently replay routes or scan the whole pool. The omitted shared-capacity audit **fails**:7880C has3 simultaneous connections against1 charger, JON_A has4 against1. Capacity was not enforced in this MIP, and duplicate service was not removed. There is no full-dispatch/GIRO-feasibility claim, no pricing certificate, no finite-pool fleet optimality proof, and no charging optimum proof.

Source commit35770aae2c08e7d5a356cc3b673e67608e5b1036 is recorded clean. Physics remains239.01kWh battery/initial SOC,35.8515kWh reserve,PARX60kW/other stations240kW,2.5kWh SOC grid,5-minute event block, reserve-only terminal constraint, covering master and unlimited shared capacity. MIP result hash `c0151c2ea6e27cd1f6af2bfa2feb310fb2ddbf52d40f363477df900cb7185e13` matches COMPLETE; CG status and declared pool hashes match the predecessor records. Exact two-stage Gurobi proof lines are indexed in `proof_lines.json`; full copied logs and results are under `strict/k19/`.

## Exact next-step readiness

[Graph reuse/export gap and preflight](graph_reuse_readiness.md) identifies the missing durable graph handoff. The completed k19 process retained a route-pool checkpoint, **not a graph cache**. Its runner rebuilds the graph even on `--resume`, after starting the CG clock. Existing packed serialization support makes a small isolated export/load change plausible, but no cached k19 graph can be assumed recoverable from this attempt. A one-time graph rebuild/export and verified same-input/physics reload would be required. Do not queue k20, reuse a graph across changed instances, relax source identity, or change the scientific allowance silently.

## Verification and automation

`python3 audit.py` passes **290 checks**:84 copied-file hashes, all44 current graph allocations, live recovery of preempted attempts, genuine registered predecessor sets, recorded CPU exclusions, strict source/pool/result identities, log endpoint agreement and explicit proof/physical limits. Native capacity/coverage diagnostics are preserved; these checks do not constitute a new physical simulation. Raw scheduler data, source paths/hashes and policy bytes are retained. No further cluster query was made.

The existing `unicorn-evsp-dr-research-progress` heartbeat was updated through the app tool to reflect the user's renewed automatic **both current Doc and current weekly Slides** instruction, superseding the prior Doc-only policy. Identity, active state, four-hour cadence, target task, notification behavior and every other persisted field were verified unchanged. It remains quiet on unchanged state. `heartbeat_before.json`, `heartbeat_after.json` and `heartbeat_verification.json` retain the receipt. Root owns live publication and shared plans/register integration.

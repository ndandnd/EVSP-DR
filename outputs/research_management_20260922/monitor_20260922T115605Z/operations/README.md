# Baseline operations — 22 September 2026, 11:57:12 UTC

One scoped SSH collection succeeded. **25 jobs run:18 graphs,5 CGs and2 MIPs;26/44 graphs have completed**, up14 since07:57. All85 pending baseline solver jobs retain genuine registered predecessor dependencies. No broken active dependency or new failed allocation requires recovery. Cumulative graph preemptions remain10, with current attempts preserved. Historical held537227 and V2G work were untouched. No submissions, repeated polls, shared-plan/register changes, automation changes or live Doc/Slides edits were made.

## Newly completed k33 CGs

All three scheduler allocations completed0:0 after their scientific wall limits; none has a pricing certificate. Zero artificials and numerically feasible saved LP solutions were independently reconstructed from their positive route weights. These are **restricted-master values**, not full-model lower bounds or integer target attainment.

| Case | Target duties / trips | Iterations | Fractional route weight | Weighted final RMP | Last pricing min RC | Stop |
|---|---|---:|---:|---:|---:|---|
| C1 k33,661618 |33 /785|179|33|3,301,350.565413|−0.3934678455|wall_limit; uncertified|
| C3 k33,661668 |33 /770|331|33|3,301,396.318977|−0.0067966028|wall_limit; uncertified|
| C4 k33,661691 |33 /785|259|32|3,201,424.123426|−0.2919154679|wall_limit; uncertified|

| Case | Separate graph build s | Cache I/O s | Reported CG wall s | CG wrapper wall s | Scheduler elapsed |
|---|---:|---:|---:|---:|---|
| C1 k33 |43,007.024902|98.357146|14,358.740488|14,482.315915|4h01m40s|
| C3 k33 |34,399.574317|121.608792|14,355.323379|14,440.366552|4h00m47s|
| C4 k33 |44,321.299031|56.988996|14,348.806535|14,416.384610|4h00m28s|

Graph construction was a separate allocation and each CG records a cache hit. Cache I/O is a reported component, not an extra quantity to add blindly to CG/wrapper wall. These times exclude earlier smaller-prefix ancestors. Current descendants C1/C3/C4k34 have correctly started from completed parents; C2/C6k33 also run. No replacement is indicated by a scientifically uncertified wall-limit endpoint alone.

## First completed k33 pool MIP: C3

MIP661669 returns **34 buses / finite-pool bound33**, so target33 is not attained and fleet optimality remains open. Fleet and charging stages both reach TIME_LIMIT. Charging-related incumbent2,299.24 / bound1,360.673619 applies under the unproved at-most34 fleet cap. Full proof lines are indexed in proof_lines.json and the complete Gurobi log is retained.

The native gate accepts all202,334 columns with zero repairs/rejections and no added GIRO routes. Native selected-route physical replay passes under historical240kWh/240kW, zero-reserve, no terminal-floor, flat-tariff, fee5, covering physics. Independently counting the saved selected routes confirms34 routes cover all770 trips, with **95 overcovered trips and118 extra trip assignments**. Duplicate removal and shared charger capacity remain unvalidated. No new independent physical simulation or complete journal scan was performed in this collection; no full-dispatch or GIRO-feasibility claim is made.

| Timing / resource | Value |
|---|---:|
| Gurobi stage1 optimize wall |1,800.556806s|
| Gurobi stage2 optimize wall |1,793.226625s|
| Total Gurobi optimize wall |3,593.783431s|
| Native stage1 runtime, including its overhead |1,807.340704s|
| Native reported runtime |3,611.469675s|
| Native physical-pool preparation |205.032804s|
| End-to-end before publication |3,818.842892s|
| Scheduler elapsed |3,832s|
| Slurm batch MaxRSS |9,868,996KiB|

The actual MIP executable is clean pinned871d057e1067411f09581e37d78f7c1ca43f68bb; the campaign wrapper/CG pin is a0e0bb7681c8451e3cbbbfa06aef390026d9af4b. Runtime CG provenance reports `git_dirty=true`; do not relabel it clean. Its recorded pin, registered source hashes, execution manifest and input identities are retained. The bounded collection did not diagnose that dirty flag. MIP source-result hash matches the copied final C3 CG byte-for-byte, and its recorded journal hash matches the execution receipt.

## Resources and deadlines

Registered allocation requests remain: graphs2CPU/96G/37h with36h scientific build watchdog; CG8CPU/96G/5h with4h scientific allowance; MIP8CPU/24G/2h with1h solver allowance. At the snapshot the longest running graph is17h16m32s into its current attempt, well below either limit. C2k33 CG is3h36m37s into its allocation, nearest the four-hour scientific allowance; its five-hour allocation has about1h23m headroom. Running C1/C4 MIPs are44m51s/42m40s into their two-hour allocations, approaching their separate one-hour solver windows. These are snapshot estimates, not later completion claims. No timeout or OOM occurred in the new terminal jobs; no resource change is justified automatically.

| Completed CG | Application peak, reported MB | Slurm batch MaxRSS, KiB | Requested memory |
|---|---:|---:|---|
| C1 k33 |43,884.960938|251,335,800|96G|
| C3 k33 |41,866.351563|224,255,904|96G|
| C4 k33 |44,011.417969|244,575,584|96G|

These metrics have different scope. Pinned exact_pricer_expanded.py:609 forks an inheritance Pool and the recorded configuration uses8workers. exact_cg_telemetry.py:29 uses `getrusage(RUSAGE_SELF)`, excluding worker RSS. This establishes a scope difference, **not a numerical reconciliation** of Slurm's214–240GiB or evidence of a particular shared-page accounting effect. Both raw measures remain recorded; reconcile scheduler accounting/process-tree memory before sizing new allocations. Source snippets/hash checks are in memory_scope_source_refs.json. Existing completed jobs are not OOM failures merely because reported MaxRSS exceeds the request.

For strict-graph preparation sizing only, the already-audited07:57snapshot records strict668433 batch MaxRSS6,301,824KiB at16Grequest, elapsed4h32m35s. It was not recollected. Its completed MIP and graph-export readiness remain in [the previous operations audit](../../monitor_20260922T075504Z/operations/README.md). Completed salvage, capacity and25-trial structure evidence were not recollected.

## Audit and next trigger

`python3 audit.py` passes234 checks, including110 copied-file hashes, predecessor membership, endpoint/source identities, saved-LP coverage/objective reconstruction, native physical gate counts, selected-route coverage and exact two-stage log endpoints. Editable cg_endpoints.csv and mip_endpoint.csv preserve the precise fields; source_hashes.json records remote paths and hashes. graph_attempts.csv preserves requeue history. Large graph/pool payloads were not downloaded; cache-payload hashes are recorded metadata, not a fresh whole-payload verification.

Next scoped check should collect genuinely new baseline CG/MIP endpoints or actionable failures, preserving k-to-k inheritance and current attempts. With25running jobs and no broken dependency, no utilization-driven submission is needed. Root owns current pointers and live publication.

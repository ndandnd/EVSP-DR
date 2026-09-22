# Baseline operations — 22 September 2026, 15:58:38 UTC

One scoped SSH collection succeeded. **20 jobs run:11 graphs,5 CGs and4 MIPs;33/44 graph preparations are complete**, up7 since11:57. All75 pending baseline solver jobs have genuine registered predecessor dependencies. No new failed allocation, broken active dependency or extra preemption requires recovery. Cumulative graph preemptions remain10. Historical held537227 and V2G work were untouched. No submissions, repeated polls, shared-plan/register/automation changes or live-artifact edits were made.

## Five new CG endpoints

All five scheduler allocations completed0:0 on their scientific wall limit, with zero artificials and **no pricing certificate**. These are restricted-master results, not full-model lower bounds or integer target attainment. Last-iterate pricing remains negative in every case.

| Case / job | Target duties / trips | Iterations | Route weight | Weighted recorded RMP | Last pricing min RC |
|---|---|---:|---:|---:|---:|
|w1_k34 /661620|34 /821|68|33|3,301,488.163687|-64.371061806|
|w2_k33 /661636|33 /787|138|32|3,201,358.363478|-2.007875578|
|w3_k34 /661670|34 /797|198|34|3,401,433.138986|-0.012311816|
|w4_k34 /661693|34 /816|197|33|3,301,451.696199|-0.204093717|
|w6_k33 /661724|33 /796|149|32|3,201,471.514998|-1.552538116|

| Case | Separate graph build s | Cache I/O s | Reported CG wall s | Wrapper wall s | Scheduler elapsed |
|---|---:|---:|---:|---:|---|
|w1_k34|60,440.023664|127.623698|14,399.086552|14,653.012942|04:04:30|
|w2_k33|48,761.879044|103.456836|14,394.130498|14,500.459962|04:01:47|
|w3_k34|57,497.855087|142.830680|14,393.847392|14,628.859277|04:04:04|
|w4_k34|35,256.326607|109.432172|14,394.984123|14,517.271369|04:02:06|
|w6_k33|54,351.127185|101.084379|14,394.553510|14,485.512506|04:01:35|

All five consumed previously built graph caches. Graph preparation is separately charged; cache I/O is a reported component, not an independent term to add blindly to CG/wrapper wall. Ancestor-prefix work is not included in these single-case times. Current C1/C3/C4k35, C2k34 and C6k34 CGs follow genuine previous-k dependencies. C5k33 still waits for its graph; its preempted build already has a running replacement attempt.

**C2k33 numerical disclosure:** saved positive-route coverage still covers every trip, but its positive-only route list reconstructs weighted objective **3,201,358.401015227**, which is0.037537633 above the recorded scalar3,201,358.363477594; route-weight difference is3.75160e−7. The saved LP reports max bound violation3.7515597e−7 under feasibility tolerance1e−6, consistent with omitted tiny negative route weights. This is not an exact objective reconstruction and is retained as one nonfatal audit discrepancy, without changing tolerances or relabeling the endpoint certified. The other four new saved LPs reconstruct within1e−5 objective units, and all five cover their trip sets within the recorded numerical tolerance.

## Three new k33 MIP endpoints

All three complete0:0 but both fleet and charging stages reach TIME_LIMIT. None reaches target33 or proves its fleet optimum. Bounds below are **finite-pool fleet bounds**; charging bounds apply only under each unproved incumbent fleet cap.

| Case / job | Trips | Accepted pool columns | Fleet / bound | Charging incumbent / bound | Overcovered trips / extra assignments |
|---|---:|---:|---|---|---|
|w1_k33 /661619|785|284,758|36 /33|2,442.544 /1,261.806169|199 /315|
|w2_k33 /661637|787|226,722|36 /32|2,273.640 /1,228.172333|159 /225|
|w4_k33 /661692|785|251,648|38 /32|2,529.592 /1,265.966329|236 /285|

The native admission gates accept all columns with **zero repairs/rejections and no added GIRO routes**. Native selected-route physical replay passes. Independently counting saved selected routes confirms every input trip is covered and reproduces the duplicate counts above. Duplicate removal and shared charger capacity remain unvalidated; no new independent physical simulator or whole-journal replay ran in this collection. No full-dispatch/GIRO-feasibility or continuous charging-optimum claim is made.

| Case | Gurobi fleet optimize s | Charging optimize s | Reported runtime s | Physical pool prep s | Before-publication wall s | Scheduler elapsed / MaxRSS KiB |
|---|---:|---:|---:|---:|---:|---|
|w1_k33|1,802.494191|1,776.637485|3,617.605486|614.225143|4,238.940707|01:11:30 /12588660|
|w2_k33|1,801.929195|1,784.594750|3,618.987655|431.908604|4,056.002967|01:08:10 /10102404|
|w4_k33|1,803.349305|1,787.754192|3,611.664370|337.590640|3,953.268952|01:06:26 /12969592|

Full Gurobi logs and result/execution files are retained; proof_lines.json identifies both optimizer endpoints. MIP source CG hashes match exact current C2 or previously audited C1/C4 files. The earlier finalized C3k33 MIP34/bound33 is referenced in [the11:57audit](../../monitor_20260922T115605Z/operations/README.md), not recollected here.

## Physics, source and memory scope

Original baseline physics are unchanged:240kWh battery/initial SOC,240kW constant charging, zero reserve, no terminal floor or shared capacity, flat tariff, fee5,2.5kWh SOC discretization,5-minute event block, covering master. CG minimizes100000times route weight plus charging-related cost; MIP uses fleet then charging under the validated incumbent fleet cap. Actual MIP pin871d057e1067411f09581e37d78f7c1ca43f68bb is recorded clean. Campaign/CG pina0e0bb7681c8451e3cbbbfa06aef390026d9af4b retains `git_dirty=true` in runtime provenance; do not relabel it clean. Registered source/input/execution hashes are retained, without a new diagnosis of the dirty flag.

| Completed CG | App RUSAGE_SELF peak, reported MB | Slurm batch MaxRSS, KiB | Request |
|---|---:|---:|---|
|w1_k34|48,189.660156|272141328|96G|
|w2_k33|44,528.648438|241963544|96G|
|w3_k34|45,005.531250|240367976|96G|
|w4_k34|47,671.257812|262891036|96G|
|w6_k33|44,950.824219|247936588|96G|

Application and scheduler memory values have different scopes. The previously hash-checked source at exact_pricer_expanded.py:609 forks eight inheritance workers, while exact_cg_telemetry.py:29 uses `getrusage(RUSAGE_SELF)`, excluding those workers. This explains a scope difference, not the numerical size of Slurm's aggregate or any quantified shared-page effect. Preserve both raw metrics; no OOM occurred, and no automatic resource change is proposed. See [pinned source references](../../monitor_20260922T115605Z/operations/memory_scope_source_refs.json).

Registered allocations remain graph2CPU/96G/37h (36h scientific watchdog),CG8CPU/96G/5h (4h scientific allowance),MIP8CPU/24G/2h (1h solver allowance). At this snapshot the longest current graph attempt is21h14m59s, still below its watchdog/allocation. C2k34 CG is3h36m12s into its allocation, nearest its scientific cap but with about1h24m allocation headroom. C3k34 MIP is1h01m50s into a two-hour allocation; physical preparation lies outside its one-hour optimizer allowance, so this is not evidence of a timeout. No queue mutation is needed.

## Validation and next trigger

The audit records **286 passing checks out of287**, with the single disclosed C2 positive-route objective-reconstruction discrepancy above; all required source, coverage, log, gate and dependency checks pass. It checks130 copied-file hashes and writes editable cg_endpoints.csv/mip_endpoints.csv plus verified_summary.json, proof_lines.json, source_hashes.json and graph_attempts.csv. Large graph/pool payloads were not downloaded; cache-payload and whole-journal hashes are recorded provenance, not new byte scans.

Next monitor should collect newly terminal baseline jobs or investigate genuine new allocation failures while preserving scientific settings and predecessor order. With20running jobs there is no low-utilization recovery trigger. Finalized structure trials, strict668434, capacity729675, supplemental recoveries and native741034 were not recollected. Root owns current pointers, live Doc/Slides and any reviewed recovery action; the separate graph-reuse agent owns its lineage gate.

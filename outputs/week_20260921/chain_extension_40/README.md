# Six baseline chains continued from k32 to k40 — 21 September

Prepared **48 new chain cases**, k33–40 on each of six chains. This continues the existing baseline experiment and its frozen random duty order. It does not switch the chains to the separate stricter-physics algorithm. **Launched:** graph array661616, 48 CG jobs and 48 MIP jobs, for140 production tasks. Native validation661175 passed9 checks, and an independent campaign audit found no blocking issue. At14:40:56EDT, five new graph tasks were running;39 waited on scheduler admission and96 CG/MIP jobs retained genuine input dependencies.

[Prepared execution manifest](manifest.json) · [Root independent input audit](parent_input_audit.json) · [Remote staged-input audit](input_validation.json) · [Verified k32 parent results](parent_endpoint_audit.json) · [Each added duty and graph owner](case_overview.json) · [Verified launch/dependencies](launch_verification.json) · [Every job ID and command](jobs.json) · [Case-to-job map](case_jobs.json) · [Immediate queue snapshot](immediate_queue.json)

## What we are extending

All six latest baseline CG pools are at k32, with zero artificials and usable terminal LPs; all six stopped at their four-hour budget without a pricing certificate. Original one-hour MIPs found C1–C6 fleets **35, 34, 33, 34, 37, 33**. Separate longer/seed searches recovered **32 on all six**. The 32-bus solution is fleet-proved within the pool for C1/C2/C3/C6; C4/C5 still have a pool bound of31. Individual-route physical replay passes, while duplicate-removal and shared-capacity validation are separate and not established by these results.

The new CGs inherit the **entire authenticated previous-k column journal**, not only its chosen integer routes. No GIRO solution columns are injected. Parent status bytes, journals and trip inputs are hash-checked. Every child retains its preceding CG dependency; its MIP waits only for its own CG. It does not wait for the previous MIP.

## Settings and resources

Scientific settings remain identical to k31–32: 240kWh battery/initialSOC, 240kW charging including PARX, zero reserve, no shared charger capacity or terminal-energy floor, flat tariff, set covering, weighted CG objective100000 per route plus electricity and5 per charge start. Event discretization remains2.5kWh/5minutes, 30 columns per iteration, reduced-cost threshold0.0001; full-pool inheritance with8 workers and no column/time truncation.

- CG source: `a0e0bb7681c8451e3cbbbfa06aef390026d9af4b`; four-hour CG scientific budget, 8CPUs/96G/five-hour allocation.
- MIP source: `871d057e1067411f09581e37d78f7c1ca43f68bb`; one-hour two-stage solve with30-minute fleet stage and second-stage fleet **<= validated incumbent**, 8CPUs/24G/two-hour allocation.
- Graph construction: unchanged solver and graph semantics, with **36-hour watchdog, 37-hour allocation and96G**, 2CPUs. The previous k32 jobs measured12.1–15.2hours and up to51.3GiB. Quadratic trip-count extrapolation to948trips projects roughly24.4hours and78GiB, so the prior24hour/64G setup has inadequate projected headroom. This is an operational resource change, not a new physical model or larger CG/MIP budget. [Measured accounting](parent_graph_accounting.txt).

All jobs use default_partition and exclude scaglione-compute-01. All **44 distinct graph tasks** are eligible concurrently (all independent cases, fewer than50). Slurm still controls resource admission. The graph cache shares immutable data only when input bytes, source commit and every cache-identity physical parameter agree. Separate chains retain separate inherited pools, CG trajectories and MIP results. Native validation additionally tests reusing a cache under a different filename with identical input bytes.

Automatic preemption requeue retains separate job/restart attempts. Existing complete identity-validated graphs are reused; interrupted CG resumes a copied checked journal/checkpoint; interrupted MIP starts a new search tree. Algorithmic failures and watchdog expiry are preserved rather than blindly retried. No held historical jobs are released or modified.

## Why “full40” has three input classes

The frozen source contains **42 duty labels representing40 numeric base duties**:13316 has `m`/`uwt` variants and13324 has `muw`/`t` variants. Each chain retains its original compatible variant choice rather than redrawing a service-day convention. Therefore full40 is the complete base-duty universe under each chain's established convention, not six identical data sets and not an unqualified43-duty comparison.

| Chains | Full40 trips | Variant pair | Input SHA256 prefix |
|---|---:|---|---|
|C1,C4|948|13316m +13324muw|904070ec8919|
|C2,C3,C6|947|13316uwt +13324t|3508a11f73d1|
|C5|946|13316uwt +13324muw|1c53d995701a|

All variants preserve the established prefix and suffix-compatibility rule. That rule does not identify a unique weekday in ambiguous chains. Equivalent graph classes save four full builds: C1/C4 k40, C2/C3 k39, and C2/C3/C6 k40. They do not merge the six ordered-column experiments.

## Timing expectation

Graph preparation is the first major bottleneck: k32 already took12–15hours. The larger graphs are projected around15–25hours, with36hours allowed. Once its k33 graph is ready, each chain has eight truly sequential CG stages, up to32hours of CG budgets. Consequently a complete k40 frontier is a roughly two-day task under favorable scheduling, potentially longer with resource waits or preemption. Early k33 results should precede k40; an overnight completion promise would be misleading.

Remote root: `/home/nc437/ladder-lite/chain_extension_33_40_20260921/`. Large case artifacts: `/share/scaglione/nc437/evsp-dr/chain_extension_33_40_20260921/`.

## Launch verification

At14:41EDT the post-launch audit verified all97 submission calls (44-array tasks plus96 single jobs), every resource request, default partition, reserved-node exclusion, graph-array concurrency44, shared graph-owner mapping, previous-k dependencies and per-case MIP dependencies. [Launch verification](launch_verification.json) records the exact scheduler output. All48 MIPs were added atomically to the existing preemption study registry, with no scheduler mutation; [registry receipt](mip_registry_receipt.json).

The prior held historical array537227 and held strict descendants were untouched. Strict packed recovery646675 is a separate experiment: at14:40:56EDT it was still running (3h34m50s) and its MIP646676 was waiting on that CG. Combined with the five newly running graph tasks, six EVSP jobs were running in that snapshot. Pending graph tasks reflect scheduler resource/priority admission, not broken scientific dependencies.

**Later snapshot, 14:46:43 EDT:** [14 new graph tasks are running](final_queue_snapshot.json), with 29 resource waits, one BeginTime wait and 96 true CG/MIP dependency waits. Strict CG646675 is also running, for 15 active jobs in this scope. The [BeginTime diagnostic](begin_time_diagnostic.txt) records one restart, exit code0:0 and eligibility at14:48:18 for array task661616_6; this is a scheduled requeue, without a demonstrated cause. Earlier launch snapshots remain preserved.

The publication branch applies only the new campaign-list entries to the existing collector and register builder. [Exact integration patch](collector_integration.patch) and [before/after hashes](collector_integration.json) document these small changes; unrelated dirty working-tree edits were not copied.

Native validation receipt: [9 passed checks](validation.json). Independent implementation/setting audit: [report](independent_campaign_audit.md), [machine-readable checks](independent_campaign_audit.json). The collector and register builder now recognize `chain_extension_33_40_20260921`, retain fixture outputs outside scientific endpoints and preserve all six treatment identities even when a graph cache is shared.

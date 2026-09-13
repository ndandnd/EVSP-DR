# Execution issues — 10 September 2026

This page distinguishes execution errors from optimization results. Read the register README for its current snapshot timestamp; original failures and later recoveries are retained separately.

| Campaign / jobs | Verified issue | What the evidence does and does not say |
|---|---|---|
| Capacity/speed CG array 772080 | Some shared-capacity cells reached the 90-minute Slurm wall limit. Earlier completed cells used about 0.43–3.55 GiB MaxRSS against a 24 GiB request. | The observed termination is TIMEOUT, not out of memory or mathematical infeasibility. Ten cells have completed CG and two-stage MIP artifacts in the current snapshot. Missing terminal CG output is not a pricing certificate. |
| Terminal-energy joint MIPs 778802, all three tasks | `KeyError: 'expanded_grid_terminal_soc_kwh'` during replay of saved columns, before optimization. | Older singleton records lack the newer terminal-energy metadata. Production Gurobi license preflight passed. This failure is unrelated to a solver fleet bound or charging optimum. |
| Terminal-energy fixed-duty frontiers 778801 | All three completed. | Their completion markers match the retained output hashes. The fixed-duty results remain valid; no frontier rerun is required for the metadata repair. |
| Warm-chain-3 k=10 job 772009 | Final publication collided with canceled duplicate 772031 after optimization. | The validated result was recovered without rerunning: fleet 10, pool fleet bound 10; charging objective 393.664, bound 337.0023334, gap 14.3934%. Scheduler FAILED does not invalidate these separately recovered and checked solver results. |

## Terminal-energy repair

**Recovery completed:** retry array **810459** completed all three tariff MIPs using immutable commit `5cdb8138c29faef9d5bf949175cb1e815a0b4220`, based on the original `2424369` execution. Original fixed-duty frontiers were reused with identical hashes. All three runs prove fleet five and charging-stage optimality within their replay-validated saved pools. Model charging objectives: 08:00 **261.5697635682**; 12:00 **332.0255245396**; 18:00 **217.5050887809**. The shared aggregate terminal-energy floor remains 280.7833253 kWh. These are finite-pool results, not a new full-model pricing certificate. See [recovery provenance](../parallel_research_20260911/README.md). The earlier local draft is historical, not the production retry source.

The bounded fix deterministically replays every saved route and carries the recomputed terminal energy and cost into the in-memory record. Where old metadata exists, it is checked against the replay. Missing energy is never replaced by a guessed SOC or zero.

The compatibility regression and real-pool smoke checks cover all three saved tariff pools: 3,464 records at peak08, 3,739 at peak12, and 3,328 at peak18, including 62 older singleton records per pool. The retry must preserve the original fixed-duty frontiers, saved CG pools, input hashes, physics and objective. Its immutable execution code must retain the original terminal-threshold dependencies. A successful smoke check is not a completed optimization result.

The original failed attempts remain in the register. Any repair submission and eventual solution require separate job IDs, code provenance and output validation. See [terminal-energy campaign](../post_meeting_20260910/terminal_energy/README.md) for the experimental model and original launch evidence.

## Evidence

- [Current collector snapshot](../post_meeting_20260910/monitor/20260910T225621Z.json)
- [Queue at register delivery](queue_at_register_delivery.json)
- [Cluster limits and memory observations](../post_meeting_20260910/capacity_speed/concurrency50_cluster_limits.json)
- [Verified warm-k10 recovery](../post_meeting_20260910/license_and_mip_recovery/STATUS.md)

Independent default-partition arrays now request concurrency 50. Increasing an array throttle does not increase an individual job's wall-time or memory allowance.

## Capacity timeout reruns

Six original cells saved no resumable pools: the driver kept columns in memory and published only at termination. Fresh matched reruns are now CG **811181**, dependent MIP **811182**, at `/home/nc437/ladder-lite/capacity_speed_pilot_20260910_timeout6_rerun_7d38ef`. The CG budget is extended to eight hours with nine-hour scheduler allocation. Code, inputs and arm physics remain fixed. A longer budget is an explicit experimental change; these are not resumed checkpoints. Original timeouts remain recorded.

## Fresh-covering downstream dependencies — 11 September, 04:31 EDT audit

Fresh CG array 810454 had 72 completed tasks and three running tasks, while 49 freeze tasks with completed predecessors still reported unfulfilled `aftercorr` dependencies. This was a scheduler gating issue, not a CG or MIP failure. The existing pending freeze and MIP jobs were repaired in place after checking per-case prerequisites; no scientific setting or solver budget changed. The repair audit preserves all passes, including an initial source-path validation error before freeze release and an intermediate array-wide update that was corrected with explicit `array_task` identifiers. Slurm can interpret the numeric array parent ID as the entire array: use the composite task identifier for every task-specific update and verify the resulting prerequisite.

Completed CG prerequisites were discharged after successful accounting exit and final-status/journal checks. Unfinished CG tasks retain individual `afterok` dependencies; every MIP retains its own freeze prerequisite. See [repair and verification](../parallel_research_20260911/results_20260911T0831Z/README.md). The hourly collector retains this audit under campaign workflow evidence.

## Nine-hour capacity rerun outcome — 11 September 08:36 EDT

CG811181 tasks5/9/11/13/15 reached TIMEOUT after9h02m, without final pool/status. Task7 completed after8h34m and its MIP completed: duty13406 with documented opportunity capacity and60kW PARX yields one bus, cost56.952, both stages optimal within a35-column pool; all14 trips covered once and station-capacity audit passes. CG itself is incomplete (`cg_wall_limit`), with no pricing certificate. This is not a preemption or a proof of model infeasibility.

The nested retry result schema was added to the collector/register; the successful retry is now visible. A Sol high subagent is implementing and testing bounded pricing deadlines and safe checkpointing; do not repeat the five timed-out jobs unchanged. See [evidence](../parallel_research_20260911/results_20260911T1236Z/README.md).

Recovery implementation passed23 tests locally and on Unicorn. Fixed CG commit `253588e9b22d68fcbc67cb56bc3eb30cbb0e16b6` adds cooperative pricing deadlines and atomic, identity-bound pool checkpoints. Five fresh retry tasks are array872397; individually dependent MIPs872398–872402 retain9bf3f75. Same8hCG/9hallocation and25minMIP/30minallocation. Successful old task7 is not repeated. These are launches, not recovered results. The collector includes `capacity_deadline5_retry`; exact job/resource/provenance records are in the linked retry evidence.

## Warm chain 1, k=7 — initialization timeout, job 810293

Verified 11 September at 20:14 UTC: Slurm TIMEOUT after 08:17:13 against 08:15:00 allocation. The event network loaded in 7.03 seconds (15,642 nodes, 74,597,352 arcs). The persisted status remains initializing, with zero iterations, no final LP, no pricing certificate and an empty column journal. The 7.66-second status wall time is the initial publication timestamp, not the completed runtime. The inherited-event-pool audit is null. Evidence places the timeout during initialization before the first recorded CG iteration; it does not identify a particular inherited route or establish infeasibility.

No usable child pool was saved, so its MIP cannot run and chain-1 k8–10 remain blocked by their true dependencies. The predecessor k6 pool and cached network remain available; this does not resume the lost child initialization work. No blind rerun was submitted. Recovery needs bounded, checkpointed initialization or a measured justification for a larger budget, tested against the same inputs and physics. Other chains and capacity retries remain active.

Evidence: outputs/parallel_research_20260911/warm_p1_k7_timeout_810293/evidence.json; collection monitor/20260911T201444Z.json.

## Capacity deadline recovery — all five saved pools and MIPs completed

Snapshot 20260911T211448Z: all five CG tasks 872397 and dependent MIPs872398–872402 completed at scheduler level. CG stopped at the cooperative pricing_deadline after 28,800 seconds; none has a pricing certificate. Atomic pool checkpoints are saved, unlike the earlier lost runs. All five MIPs prove fleet and charging optima only within their tiny saved pools and pass the shared-station capacity sweep; route feasibility is by exact-event construction, not an independently claimed full continuous replay.

| Case | Final RMP route weight | Saved columns | Integer buses | CG completed iterations |
|---|---:|---:|---:|---:|
| duty13406, capacity | 1 | 34 | 1 | 20 |
| k2, capacity / combined | 2.8 | 36 each | 3 each | 13 each |
| k3, capacity / combined | 16 | 38 each | 16 each | 3 each |

The k3 final weighted RMP objective is 1,600,127.456, not a certified full-model lower bound. In its third completed iteration, capacity-only pricing consumed 25,722.65 seconds (7.15 hours), while that iteration's LP took 0.00563 seconds. Combined-arm pricing took 24,797.47 seconds. The following pricing call hit the deadline. This is concrete evidence that pricing—not the tiny saved-pool MIP or LP solve—is the computational bottleneck in these pilot cases. It does not prove that capacity requires sixteen buses. The k3 selected solution has three duplicate-covered trips; removal is not separately validated.

All 16 original pilot cells now have MIP outcomes across original and recovery attempts. Ten original CG completions were certified; six recovered cases remain uncertified. No new campaign or unchanged retry was submitted. Next recovery decision: profile capacity-aware pricing before spending another eight hours on the same search; keep the identity-bound saved pools. Evidence and full hashes: outputs/parallel_research_20260911/capacity_deadline5_completed/.

## Warm chain 2, k=9 — import exhausts CG budget; freeze fails

Snapshot 20260911T221530Z: job810332 completed at scheduler level after08:09:19, but CG stop_reason is wall_limit and final/final_lp are null. Import reoptimized and accepted42,732 predecessor columns using8workers in29,324.84s (488.75min). Overall saved wall time29,337.41s (488.96min). There is no new LP endpoint or pricing certificate. Unlike P1k7, imported columns were saved and inherited by running child k10 job810333.

Freeze810974 failed after3s; its exact error is `no usable terminal source for k09_p2: continuation: source retains artificials for k09_p2; baseline: source retains artificials for k09_p2`. This rejection is not evidence of actual positive artificials: the source final LP fields are absent. MIP810975 is pending DependencyNeverSatisfied and has not optimized. Do not bypass validation or relabel absent LP fields as zero. Recovery needs an explicit validated terminal-RMP/pool export from the saved journal, or a tested continuation, with no fabricated pricing certificate. No blind retry was submitted.

This corroborates an inherited-column initialization bottleneck independently of the capacity-pricing issue. The mathematical feasibility of the full k9 model has not been disproved. Source path, input/provenance and predecessor hashes are in cg_record.json.

## Warm chain 4, k=10 — initialization timeout, job810344

Verified at 00:17 UTC on12September (20:17EDT11September): Slurm TIMEOUT after08:17:05 against08:15:00 allocation. Cached network loaded in8.1s (19,205nodes;109,583,108arcs). Persisted status remains initializing, zero iterations, final LP absent, inherited-pool audit null, and column journal empty. The status wall_s=9.18 is the initial publication time, not the completed job runtime. No LP certificate or new integer result exists.

This repeats the P1k7 initialization failure. The predecessor P4k9 pool survives but there is no usable child pool for the dependent MIP. No unchanged retry submitted; use the same bounded/checkpointed-initialization recovery requirement already recorded for P1. P2k10 remains running. Evidence: outputs/parallel_research_20260911/warm_p4_k10_timeout_810344/evidence.json.


## Warm full-pool control and bounded-import outcomes at 02:25 EDT on 12 September

Original chain 2 k10 CG job810333 completed at scheduler level after7:59:51. Its CG record has `wall_limit`, zero pricing iterations, no pricing certificate and no final iteration, but `final_lp_source=final_pool_resolve` now supplies an explicit terminal pool LP. Freeze810976 completed successfully and MIP810977 is running. This differs from the earlier k9 freeze rejection; do not mark the k10 MIP blocked or infer a pricing certificate from its final-pool solve.

The separately registered512-route import treatment now has certified CG and integer outcomes for the three earlier blocked cases: chain1 k7 uses8 buses (proved minimum in that saved pool), chain2 k9 uses10 with pool bound9 (unproved), and chain4 k10 uses11 (proved minimum in that saved pool). Their charging stages reached the time limit. These results do not repair or supersede the original full-pool controls as the same treatment. See [dated evidence](../overnight_extension_20260912/RESULTS_20260912T062451Z.md). All departed jobs checked in the supplemental accounting record completed with exit0:0; no new execution failure or preemption was found.


## Paired efficiency warm startup failures on 12 September

Jobs 964196, 964197, 964200 and 964201 failed after three scheduler seconds (exit1:0). Each cache-preparation subprocess exited2 with `--event-network-cache-only does not use --out`. No paired CG optimization or network build ran in these attempts. The exact preparation stderr, allocation records and v2 collector evidence are retained in [the launch status](efficiency_validation_20260912/README.md). This is a launcher argument failure, not model infeasibility or preemption. The implementation task owns a separate immutable recovery; the five other jobs remain untouched. No automatic requeue.


## Full-pool chain 2 k10 completed at 03:25 EDT

MIP810977 completed:71 buses from42,795 saved columns, fleet bound71 and stage1 OPTIMAL in6.24s; stage2 TIME_LIMIT. The source CG performed zero pricing iterations and had no pricing certificate after spending its budget on inheritance. The terminal pool resolve was sufficient to run MIP but did not turn the result into a converged CG outcome. This supersedes the earlier running-MIP status above. The bounded512-route counterpart has11 buses with unresolved pool bound10. See [dated evidence](../overnight_extension_20260912/RESULTS_20260912T072441Z.md). No job was retried or changed in this update.


## 12 September 05 25 EDT decomposition startup timeout

CG array949623_3 (concrete job949644), case d00_g3, reached its scheduler time limit after04:47:09. Only session_start telemetry exists; no cg.json, pricing iteration, certificate or journal was written. Exact preparation phase is unresolved. Dependent MIP949624_3 is DependencyNeverSatisfied. This is not preemption. Do not blind-requeue; preserve the lock and failure evidence. Source logs and hashes: monitor/20260912T092542Z_supplement.json; accounting:20260912T092542Z_sacct.txt.

Register reader fix: recombined decomposition schedules were incorrectly sent through the three-arm tariff parser, causing a manifest-input validation error. They now produce one decomposition_join record each, preserving manifest input/target and fleet sum without asserting a MIP proof. Full snapshot build passes with1767 rows.

Implementation-task follow-up (reported, pinned evidence pending): session_start follows successful preflight/build_problem/provenance; absence of completed graph telemetry narrows the timeout to graph construction. Constructor reportedly eagerly builds packed arcs even in lazy mode, with no internal deadline checks. Neighbor d00_g2 reportedly built94,577,843 arcs in8992s; failed d00_g3 MaxRSS~4.83GiB against32GiB request gives no OOM evidence. Authorized one separate15-minute diagnostic profile with unchanged scientific settings and external deadline; no long retry or dependent MIP authorized by this diagnostic. Cache preparation plus a new14400s CG budget would be an extended experiment, not an equal-budget recovery. Await immutable report and job record before treating these implementation observations as independently verified.

Diagnostic submitted by implementation task:982348, default partition,2CPU/32G,20-minute allocation,no requeue. Root `/home/nc437/ladder-lite/d00_g3_graph_diagnostic_20260912`; immutable instrumentation pin03d0823752e614f523ff9cbd49aaab30208cb1bf, manifest SHA1751e0d07cab8d3a452c0efaedc6bc4f258526f7640bb5f3557c0f7cde531119. Solver remains a299 source;900s watchdog then90s TERM grace/KILL. Expected diagnostic_budget_exhausted is a diagnostic limit, not a CG failure or certificate. Collect attempts/982348_r0 diagnostic_result.json,graph_progress.jsonl,graph_stacks.txt,graph_phases.jsonl,identity.json and run/execution.json after completion. Original experiment and dependent MIP unchanged.


## 12 September 06 26 EDT diagnostic crash

Diagnostic982348 ended after366.18s with subprocess returncode-11 (SIGSEGV), watchdog_triggered=false and state diagnostic_error. It did not reach the900s budget. Last progress:142/29396 trip/SOC sources completed,3,011,382 packed arcs at313.47s; RSS~276MiB. Two complete120s sampled stacks show JSON encoding under event_pricer_network._add and _charge_arcs; a third stack is truncated. These samples suggest a serialization hotspot but are not a time profile or complete graph-time estimate. Crash origin (solver versus instrumentation) remains unresolved; implementation task notified and no retry authorized until tested. All8 source artifacts and SHA256 hashes are retained in d00_g3_diagnostic_20260912/terminal_collection.json. No CG/MIP result or certificate.

Implementation terminal review pinned at57f663eb: [review](https://github.com/ndandnd/EVSP-DR/blob/57f663eb/outputs/d00_g3_startup_review_20260912/TERMINAL_REVIEW.md). A standalone local Python3.12.2 JSON loop succeeded without sampling but hung under a2ms repeated faulthandler timer; no solver involved. This implicates the sampling method but does not reproduce the cluster Python3.12.13 SIGSEGV. Retire asynchronous timer sampling. Two stack observations identify a candidate serialization hotspot only, not a measured CPU fraction. Future profiling should use tested synchronous progress/profiling; no extra run launched. Progress count142 includes depot plus141 trip/SOC sources.


## 12 September 07 27 EDT joined parent timeout

Join01 job949751 reached its3-hour scheduler limit (elapsed03:02:10) during the parent32 CG invocation. The saved35-bus decomposed construction predates this step and remains available; it is not a completed parent CG or joint improvement. No parent cg.json exists; telemetry records startup only. No pricing certificate or parent MIP result. This is a timeout, not preemption. Original artifacts retained; no retry. Logs/hashes and the late-arriving w2_k13 MIP are in monitor/20260912T112738Z_supplement.json. The explicitly reconciled snapshot retains the original collection hash and timestamp.


## 12 September 08 28 EDT repeated parent startup timeouts

Join02 (949753) and join03 (949755) each reached their3-hour scheduler limit, elapsed03:02:21. Both retain only session_start parent-CG telemetry and no CG result. Their existing35- and36-bus decomposed constructions remain separate usable baseline artifacts; neither is a completed parent optimization. With join01 this is three full-parent startup timeouts. No preemption or new retry. Source logs/hashes: monitor/20260912T122840Z_supplement.json; scheduler accounting:20260912T122840Z_sacct.txt.


## 12 September 09 30 EDT final joined-parent timeouts

Join08 job949766 and join09 job949768 reached scheduler time limits at03:02:28, retaining only session_start parent-CG telemetry. All five launched parent optimizations have now timed out during startup; the five preexisting decomposed constructions remain separate baseline outputs. No parent CG certificate or MIP result. No retry launched. Logs/hashes in monitor/20260912T133042Z_supplement.json.

All three capacity efficiency pairs completed normally at scheduler level, but all six solver arms terminated pricing_deadline after10800s with no pricing certificate. Each pair has equal restricted-LP endpoints; this does not establish a full-model lower bound or convergence speedup. Source rows/hashes retained in overnight_extension_20260912/status_20260912T133042Z/capacity_pair_results.json.


## Chain 2 k14 preparation timeout 12 September 11 32 EDT

Job949703 reached TIMEOUT after04:47:03. Network cache279,321,120 arcs loaded46.372s. Initial identity and header-only iteration CSV saved; columns journal empty. No first CG iteration, pricing certificate or integer result. MaxRSS44,051,800KiB versus96GiB request, no recorded OOM. Exact preparation stall unresolved; implementation task notified to inspect deadline coverage. No blind requeue. Evidence and hashes: `../post_meeting_20260910/monitor/20260912T153215Z_supplement.json`.


### Import deadline hang reproduced by implementation task

Implementation task `01a06e8e-c60a-7132-a588-d50e800d47d3` reports an isolated production-importer reproduction: parent run_cg installs a SIGTERM handler that only sets a flag, then fork workers inherit it. After the import deadline, pool.terminate()/join can wait indefinitely because workers do not exit or check the flag. With a 0.2-second deadline, default handling returned in0.208s, inherited handling exceeded the4s outer watchdog, and worker initializer reset returned in0.207s. This reproduces a control-flow bug; it is a strong explanation for w2_k14 but not an exact historical stack-trace proof. Pre-selection preparation is also outside the timer. Pinned evidence is pending; this paragraph records the task report, not an independently rerun test.

A bounded worker-shutdown/deadline fix and regressions are authorized. Retry only w2_k14 after tests and immutable provenance, preserving original artifacts and scientific parameters. No submission is claimed yet.


### Corrected w2k14 retry submitted 11 56 EDT

Job15687 submitted12September15:56:29UTC; initial PENDING Priority. Source68fce0093ec9768392442fe1b107a1b67ab0cb7b. Remote root `/home/nc437/ladder-lite/w2_k14_import_fix_20260912`; separate output `cases/w2_k14/cg.json`. Same8CPU96GiB default4h45, exclude scaglione-compute-01,14400s global budget,900s replay-only budget,512 routes,8 import workers. Parent949701 completed successfully but was purged from controller, so afterok submission was rejected without creating a job. Retry freezes verified exact parent artifacts and records fulfilled data dependency. Original failed artifacts and successors949704/949705/949706 unchanged. Future successors require new input paths as well as corrected dependencies, only after validated retry output. Ledger: `../w2_k14_import_timeout_review_20260912/retry/submission.json`; manifestSHA6a7fd77a5c6f779e0c32d19319f55262afa31c8fa1b8e37888bd69d512c56559. Collector root added. No result/certificate claimed.


### Import fix verified in cluster 12 September 12 32 EDT

Retry15687: replay900.000092s, shutdown0.162453s, total import904.208110s. CG checkpoint224iterations, telemetry232, continuing without certificate. Confirms bounded worker shutdown, not convergence. Snapshot20260912T163244Z. C4k15 independently stopped at its CG wall_limit with min_rc=-0.199828525 and no pricing certificate; saved pool MIP running. This is a solver time limit, not an execution crash.


### Targeted catch-up check 12 September 14 58 EDT

Retry15687 now reports certified_rc_optimal=true, stop_reason=certified,1484iterations,9572.358s,weighted LP1400558.3558959858,no artificials,min_rc=-9.09e-9. No longer in running queue; wrapper terminal/accounting and source hashes still to collect before downstream repair. Evidence `../post_meeting_20260910/monitor/20260912T1858_catchup_check.json`. Google Doc note updated; original66/87 campaign count remains separate from this isolated corrected retry. No MIP result or completed physical integer validation claimed.


### Chain 2 downstream recovery submitted

Only affected jobs replaced by distinct attempts37532(k14 MIP),37533(k15 CG),37534(k15 MIP after37533). Original blocked949704/5/6 retained. Predecessor15687 final source/journal hashes, certificate, zero artificials and completed0:0 verified. Real data path corrected; no result copied over failed original. Same science, resources and budgets as planned; no output/certificate claim for these three new jobs. Details outputs/w2_chain_recovery_20260912.


### Recovery wrapper failure and verified replacement

37532/37533 stopped in Gurobi preflight before optimization: PYTHON_BIN unset in new wrapper.37534 remained DependencyNeverSatisfied. This was a manager launcher error, unrelated to the recovered CG output or solver license. Fixed wrapper exports PYTHON_BIN; exact wrapper environment/preflight successfully executed before resubmission. New jobs37583(k14 MIP),37584(k15 CG),37585(k15 MIP after37584), root w2_chain_recovery_retry2_20260912. Original attempt outputs/jobs preserved. New Gurobi logs and result paths are isolated. Source science/budgets unchanged. First recovery root must remain: new attempts depend on its frozen inputs/cache and entrypoint.


## Queue recovery, 12 September

[Resolved queue blockers and exact replacement map](../queue_recovery_20260912/README.md): nine ready MIPs released after saved-parent validation;43 obsolete pendingentries cancelled;6 indexed full-poolstreams restored and extendedthroughk15; sharedgraphdecompositionretry2 started. Originalfailures preserved. Firstgraphcacheattempt rejected cache-only --out beforeconstruction; corrected fullcommands validated and retry2passedstartup. No invaliddependencies remain in16:48EDTsnapshot. Newjobs autoremovetheir impossible dependents; monitor must diagnoseandrecoverratherthan leavezombies.


## 13 September 05:14 EDT — shared parent graph timeout

Job42509 in graph_recovery_retry2_20260912 reached the12:30 scheduler allocation after12:32:22 while preparing the parent32 event-network cache (750 trips). The cache was not written; telemetry contains only session_start. No parent CG iteration, pricing certificate or MIP exists. Recorded maxRSS25.4GiB against128GiB requested; no OOM or preemption. Native Gurobi preflight passed. Slurm reports TIMEOUT at job level despite ExitCode0:0; batch status is CANCELLED0:15 and stderr explicitly identifies the time limit.

All ten parent CGs42512,42514,…,42530 and their ten MIPs42513,42515,…,42531 automatically cancelled before start because their true prerequisite failed. These are cancelled pending attempts, not MIP failures or preemptions during optimization. Existing component/recombined schedules remain preserved. No identical retry launched. The internal graph/cache preparation operation is unresolved and needs progress/profiling evidence before another full-size run. [Report, exact accounting, command and hashed logs](../queue_recovery_20260912/status_20260913T093758Z/README.md).

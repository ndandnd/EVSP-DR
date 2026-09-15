# Active overnight monitor — 15 September 2026

The user requests continued overnight monitoring, productive parallel use of Unicorn, and a clearer document after results arrive. The existing hourly heartbeat `unicorn-connection-and-research-progress` remains ACTIVE; it was updated rather than duplicated. Notify on meaningful findings, failure, access changes, completion or a needed decision. Quiet checks should finish without browser work or repeated briefings.

## Current state and priorities

Last full verified collection: `outputs/post_meeting_20260910/monitor/20260915T101137Z.json`, completed **06:19 EDT**. [Source report](../overnight_next_20260914/status_20260915T101137Z/README.md). Collection queue: **18 running / 29 true input dependencies**; no invalid dependency or array throttle. Scientific endpoints and later launches must retain separate timestamps.

| Work | Verified state | Next check |
|---|---|---|
| Original chain extension | Largest one-hour matches by chain: **26 / 27 / 27 / 27 / 26 / 28**. New C4 k27 matches27/proved in pool. C5 k27 uses29 with bound26, open. Its fractional LP route weight26 covers660trips; CG time-capped. | C1k27 MIP187971 started during collection; C1k28 graph recovery189917 and C2/C4/C5k28 CG continue. Consolidate with `summarize_chain_extensions.py`:90 submitted cases,73CG/73MIP endpoints. Do not promote scheduler-only results. |
| Original k16–25 misses | **All25 recovered** by separate unchanged-pool MIPs. Final C5k25 long220545 uses25/proved in pool after17.86fleet minutes, charging still time-limited. Ordered pool and initializer count verified. | Complete; do not repeat. Extra allocated time alone is not proved causal: this is a fresh search tree and fleet runtime is below the earlier30-minute allowance. Preserve original35/60matches separately. |
| C3k28 longer search | 222757 still running; original29/bound28. Same166052-column pool,12600total/10800fleet budget. | Verify published endpoint, ordered pool hash, initialization and proof separately. No duplicate. |
| Compact starts k8/k10/k15 | Complete36certifiedCG/36MIP/33targets. C1k15 both smaller pools prove16, earlier matched full pool supports15. C3k15core17/bound15 remains open. | Existing C1 donor audit:12/15patterns absentcore,11/15expanded, all15known witness routes positiveRC at compact final duals. Not a proof about every possible15-bus solution. |
| Compact starts k20/k25 | **Complete24MIPs:8target matches,8target-excluding pools,8open misses.** All24CG ended:3certified/21capped. Three certified cases C2k20expanded,C3k20core/expanded all have open integer gaps. | Do not rerun completed arms indiscriminately. New controlled pool-union pilot below. `target_excluded_in_saved_pool` uses bound>target+1e-5; C2k20core22/bound21 excludes20 without proving exact21vs22. |
| Existing pool additions |13paired LP-support versus equal-count zero-weight additions all tie above target.25/26prove above-target minimum; C2k8zero-weight9/bound8open. | Complete. Zero LP weight is not zero reduced cost. No new pricing certificates. |
| Stricter physics | Reserve screen complete:8one-bus/2two-bus; all10CGcertified. Duty13408reserve+capacity+PARX60works at1bus,77minCG. Hard fixed-dual reference3.58/3.19h; cache both unfinished at~4h. | Do not claim larger baseline chains satisfy these settings or a cache speedup. Reserve236.44kWh/15%; no65%endingfloor/nonlinear charging. |
| k29–30 extension | Graph array224628:12running. Each of12CG waits for its graph and prior-k columns; own MIP follows. Unchanged frozen random order/settings. | Preserve true dependencies. New24h graph watchdog/25h allocation follows observed old12h native watchdog exhaustion. |

**Bounded new implementation:** `/root/compact_union_pilot` is preparing `compact_pool_union_20260915`. Eight same-instance core+expanded unions: C1k15,C1k20,C2k20,C2k25,C3k20,C4k25,C5k20,C5k25. Four unchanged core512 controls: C2k20,C2k25,C3k20,C5k20. The other four pairs already exclude target in both pools, so existing proofs are sufficient controls. Frozen native871d057, native greedy policy, no supplied starts or new CG/GIRO columns.12600total/10800fleet,8CPU24GB,default,compute01excluded. Donor witnesses remain independent feasible upper bounds, not new-solver incumbents. Combined prior CG costs are recorded; this is not equal-computation performance evidence. Two real-pool build fixtures227214/227215 and three short native MIP fixtures227216–227218 are running/queued. Production still awaits fixture validation and root review. Agent may not modify shared records. Root has prepared collector/normalizer support; integration must be verified before publication.

A direct login without the standing SSHsocket was refused; the established control connection remained usable. No actual monitor access loss occurred. Always include the socket.

When the queue thins, inspect registered plans and artifacts first. Launch scientifically justified independent work and repair demonstrated execution issues without asking again. Do not remove true dependencies or launch arbitrary repeats to achieve a numerical job count. Protect held537227 and EVSPV2G work.

## Collection and recovery

- Read the cluster resource policy before any submission: `/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md`.
- SSH uses `/Users/nadan/.ssh/evsp-unicorn.sock`, `BatchMode=yes`, a short connect timeout and `nc437@unicorn-login-01.coecis.cornell.edu`. Notify promptly on access loss/restoration; collector failures are not automatically access failures.
- Preserve actual previous-k and own-CG dependencies. Never remove them to make a queued job run before its input exists. Validate artifact hashes and successful producer output before repairing dependency links.
- Independent default CG concurrency is50/all cases if fewer. Every CPU job excludes `scaglione-compute-01`. Held537227 and concurrent EVSPV2G experiments are protected.
- Routine MIPs use the default partition and documented preemption/requeue handling. Retain every attempt. Current compact and diagnostic campaigns intentionally have12600-second total /10800-second maximum fleet-stage budgets; do not overwrite those exceptions with default3600/1800 budgets.
- Collect with `outputs/meeting_20260910/collect_remote.py`, write a dated monitor snapshot, refresh `preemption_study/refresh.py`, then use `monitor_compact_delta.py`. Choose the preceding successful snapshot by embedded timestamp; compare canonical content for hashless CG rows.
- Rebuild the register/workbook only for material verified changes. Preserve the exact six active supplemental sources, compact JSON serialization and source-binding checks. Read relevant new runbook entries rather than the whole history. Use `summarize_longer_gap_results.py --snapshot ...` to verify the two new longer-search campaigns; it checks original ordered pools and initialization counts in addition to source/provenance bindings.
- Publish only a reviewed allowlist through the durable worktree named in `publication_workspace_path.txt`; push to `codex/parallel-research-20260911`, then verify the Unicorn mirror. Do not wholesale commit the dirty main checkout.

## Document: maintain one current dashboard

Google Doc: `1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY`.

| Tab | Role |
|---|---|
| `t.79m3d3x4h45m` — START HERE — Current results | First tab and the only current summary. Replace tables and dates in place. |
| `t.lumf8xm66fow` — Research log — through15Sep | Preserved dated history; no more hourly status paragraphs here. |
| `t.ts4vwph3s99i` — Figures with explanations | Preserve embedded figures and editable captions. |
| `t.h5h2ivyiprly` — CG curves and bus schedules | Preserve plots, Gantt schedules and captions. |

Do not edit Slides. Do not add another current tab every hour. The dashboard has editable tables, term definitions, explicit model/proof limits, and links to both figure tabs and source reports. Its source and verified export are in `overnight_monitor_20260915/`.

At completion of the active overnight comparisons, or in the morning check around09:00 America/New_York if work is still running, consolidate the dashboard. Include actual integer buses by chain and treatment, CG times and precise stopping reasons, clear units, and what remains unresolved. Preserve original versus longer MIP treatments and baseline versus stricter physics. Use short interpretations immediately beside the relevant tables/figures. Keep full provenance and historical chronology in linked reports, not repeated in the current summary. Update figure captions/dates only when their evidence changes; never imply a historical plot is current.

Before document edits, export the relevant tab. After edits, export again and check table values, links and preserved content. Snapshot scheduler counts separately from scientific verification cutoffs. Do not label an uncertified RMP objective a full-model lower bound; saved-pool fleet proof, charging proof, physical replay and shared-capacity validation remain distinct.

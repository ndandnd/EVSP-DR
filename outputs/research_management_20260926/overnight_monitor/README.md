# Overnight EVSP–DR monitor — 26 September 2026

## Final morning check — 08:09 EDT; monitoring paused

Recovery **520378 completed** at 05:08:07 EDT in **45m57s**, using **17.71 GiB peak RSS** against its unchanged 48 GiB / 8 CPU request. The saved 08:09:43 EDT snapshot has no remaining scoped EVSP–DR jobs; all scoped allocations are terminal. The overnight heartbeat is **PAUSED**, confirmed separately through the automation tool.

The expansion now has **80/80 exact-once five-bus cleanups: 55 direct fresh-CG and 25 GIRO-frontier fallbacks**. Charging is proved to solver tolerance within **61 finite cleanup pools**, with **19 time-limited**. The original **61/80 CG pricing certificates are unchanged**. Shared charger capacity remains unvalidated; these results do not prove a full-model charging optimum.

The recovered `mix1_two_price_split` covers all 111 trip indices exactly once, adds no fixed-duty fallback, and passes the **181-check saved-receipt and route-arithmetic audit**. [Final morning audit and provenance](morning/README.md) · [updated 80-cell table](morning/expansion_cells.csv) · [181-check verification](morning/verification.json). The failed original post and frozen 04:12 audit remain preserved. Final Doc and slide 55 publication passed [171 preservation/content checks](morning/publication/verification.json); the [automation receipt](morning/automation_pause.json) records the paused heartbeat.

## Historical overnight record — superseded by the final morning check

The monitoring instructions and incomplete counts below are retained as historical records. No further heartbeat or recovery is pending under this overnight request.

User authorized monitoring overnight. Reuse heartbeat `unicorn-evsp-dr-research-progress`, every four hours. End after the morning check at/after 08:00 America/New_York, or earlier if all scoped work is terminal and results/recovery decisions are recorded. Keep unchanged checks quiet.

## Historical check and remaining work — 04:30 EDT

The 80-case expansion's 240 original allocations are terminal. This is useful completed work, not an idle queue with broken dependencies. The frozen [endpoint audit](endpoint_summary/README.md) records 61/80 certified CG LPs; 79 exact-once five-bus cleanups (54 direct fresh, 25 using added GIRO fixed-duty frontier columns); charging optimality in 60 repair pools and time limits in 19. All 79 pass individual-route replay, but shared charger capacity is not modeled. The 25 CG scheduler failures are documented no-selection return-code-3 endpoints, followed by successful fallback/cleanup. Sixteen allocations restarted 17 times; stale attempt files are superseded. Five original-root recovery jobs also completed, separately recorded.

**One authorized same-setting recovery is now RUNNING: 520378**, submitted 04:22:05 EDT and started 04:22:10 on `luxlab-cpu-02`. It replaces failed post 498962, `mix1_two_price_split`. Failure was a computational safety guard, not a Gurobi error or OOM: duplicated-trip counts by selected route are 10,1,0,2,11. The exact repair enumerates at most 3,079 subsequences. A source-hash-pinned wrapper changes only guard10→11, preserving all enumeration, inputs, physics, objectives and solver settings. Seven tests and remote hash preflight pass. Frozen checkout unchanged. Resources remain 8 CPU/48G/5h, default_partition, requeue, required GPU-node exclusion, original completed dependencies. [Code, rationale and receipts](recovery/README.md).

### Next heartbeat instructions

1. Read `ROOT2/recovery_20260926_case11/submission.json` and current queue/owner records before any action. Do not submit a duplicate. Job520378 output uses `ROOT2/results/mix1_two_price_split/cleanup/job520378_r<R>/attempt.json` and `out/summary.json`; restart attempts remain separate. The original post record498962 remains failed by design; the replacement ledger supersedes it only after validation.
2. Check latest replacement attempt's scheduler state, source hash, exactly-once coverage, route replay, finite-pool fleet/charging statuses and costs separately. A successful replacement would make 80 cleanups, 55 direct and25 fallback; count it only after checking the actual receipt. Preserve the frozen 04:12 endpoint audit and raw snapshot; store new receipts separately.
3. Update the two current-work Doc paragraphs and weekly slide55 (`h5de1531e14aadef0_4_1`) only when this changes verified counts. Both are already published and checked: 171 preservation/content checks; prior54 slides, images and notes retained. The Doc's date/two status paragraphs are the only changed lines; figure/source footer retained. [Publication verification](publication/verification.json).
4. At the first wake at/after08:00 EDT, give the compact morning summary and **pause this heartbeat**, even if the replacement remains unfinished; report any remaining action explicitly. Pause earlier only if all scoped work is terminal and results/recovery records are complete. No new treatments or other-project changes.

The fallback summaries have overbroad static `proof_scope` prose. Use explicit certificate and solver fields: 16/25 fallbacks lack a CG certificate,7/25 fallback charging searches time out, and17/25 selected fallback covers are entirely GIRO trip sequences. This reporting issue is recorded for a later code correction; no source receipt was rewritten. No savings or example-ranking conclusion is claimed yet.

## Initial check, about 00:03 EDT

Unicorn reachable using `ssh nc437@unicorn-login-01.coecis.cornell.edu`. Noninteractive SSH needs `/usr/local/slurm/slurm-25.05.5/bin/squeue` (and corresponding sacct/scontrol). Local sandbox restrictions required approved network access; this was not a cluster outage.

EVSP–DR expansion: 21 running jobs (18 CG + 3 post), 18 post jobs pending afterany on their own running CGs. The running CG task indices of array498917 are 15,32,33,34,36,37,65,68,69,70,71,72,73,74,75,76,77,78. Running posts: 498960,498975,498976. Pending posts: 498934,498951,498952,498953,498955,498956,498984,498987,498988,498989,498990,498991,498992,498993,498994,498995,498996,498997. Representative scontrol checks confirm campaign WorkDir, requeue and scaglione-compute-01 exclusion. Array498917 task76 has one restart.

## Scope and sources

- Expansion ROOT2: `/home/nc437/ladder-lite/spatial_tariff_expansion_20260925`; job prefix sx_; CG arrays498916/498917, fixed array498918, post498919–498998, plus any recorded replacements.
- Read `outputs/independent_review_20260925_spatial_tariffs/expansion/README.md` and `expansion/from_cluster/jobs.tsv` for exact cell mapping, latest recovery receipts and collector commands. The expansion has 80 cells/240 original jobs. Execution commit4a8b497e668be9962bca5906a8b69116fa634882. CG concurrency is split20+30=50; do not change scientific settings or launch duplicates.
- Decisive post receipts: ROOT2/post_records/<cell>/job<J>_r<R>.json; attempts: ROOT2/results/<cell>/<stage>/job<J>_r<R>/attempt.json. Collect to dated destinations; preserve attempts and restart counts.
- Original ROOT `/home/nc437/ladder-lite/spatial_tariff_k5_20260925`, prefix st_, remains separate. Check latest recovery mapping before acting on previously noted cleanup462773 OOM or cells missing selections. Do not collide shared cell names across roots.
- Do not use yesterday's cluster collector as a ROOT2 collector; it targets completed full40/cap and the original root only.

## Actions and limits

One compact squeue/sacct check per wake. Check newly terminal attempts and real predecessor status. A post exit4/no_cg_output can be caused by preemption/requeue; exit5/no_fixed_source likewise requires source inspection. Recover only already-authorized same-setting work after checking for existing replacements and campaign-owner changes. Read remote SCAGLIONE_RESOURCE_POLICY.md before any submission; preserve dependencies, resource policy and CPU exclusion. Keep V2G/incentive jobs and held rvS/537227 untouched. No new experiment designs are authorized by this overnight monitoring request.

Keep scheduler completion, CG pricing certificate, finite-pool proof, exact-once dispatch and shared-capacity validation separate. Fallbacks that add fixed-duty frontier routes must stay labelled. Save verified endpoints and update only meaningfully changed current Doc/Slides content under standing instructions; preserve figures/history and avoid broad exports. Notify on meaningful completion/failure, needed decision or lost SSH access. Pause this heartbeat after the final morning check.

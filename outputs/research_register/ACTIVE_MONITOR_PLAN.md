# Active overnight monitor — 15 September 2026

The user requests continued overnight monitoring, productive parallel use of Unicorn, and a clearer document after results arrive. The existing hourly heartbeat `unicorn-connection-and-research-progress` remains ACTIVE; it was updated rather than duplicated. Notify on meaningful findings, failure, access changes, completion or a needed decision. Quiet checks should finish without browser work or repeated briefings.

## Current state and priorities

Last full verified collection: `outputs/post_meeting_20260910/monitor/20260915T060910Z.json`, completed at 02:17 EDT. [Source report](../overnight_next_20260914/status_20260915T060910Z/README.md). Queue in the collection: **29 running jobs / 17 true input dependencies**, no array throttle or invalid dependency. Scientific endpoints and separately dated launches have distinct timestamps.

| Registered work | Current result | What to check next |
|---|---|---|
| `chain_extension_20260914` | Largest one-hour target matches by chain: 25/26/27/26/26/27. C3 k28 finds 29 with bound 28; CG also time-capped. | Remaining genuine chain dependencies; consider a matched longer C3 k28 pool search if queue thins, after duplicate checks. An open gap is not a proved pool limitation. |
| `compact_seed_support_20260914` | Complete: 36 certified CGs, 36 MIPs, 33 targets. C1 k15 both smaller pools prove 16; matched earlier full pool supports 15. C3 core has 17/bound15, still open. | Do not relaunch completed arms. Any repair study must target a demonstrated pool deficit with matched sources. |
| `compact_large_seed_20260914` | 20 CG endpoints: 3 certified, 17 time-capped; first 3 MIPs match 25/25/20 with pool fleet/cost proofs. | Remaining 21 MIPs / 4 CG endpoints. Early completion is selected; do not report 3/3 as overall success. Keep a0e0 distinct from smaller e091 cohort. |
| `lp_support_pool_diagnostic_20260914` | Complete: 13 matched addition pairs all tie above target; 25/26 arms prove above-target minimum. C2 k8 zero-weight arm has 9/bound8. Nine support-only controls also above target. | No winner between these selection rules on this selected population. Zero LP weight is not zero reduced cost. Constructed pools have no new CG certificate. |
| `final_chain_gap_20260915` | Job220545 submitted at 02:13, running at 02:14 EDT; unchanged original C5 k25 pool with 207717 columns, previous26/bound25. | Full-size license passed. Collect next hour: campaign registration was added after current full collection began. Preserve 12600-second total /10800-second maximum fleet budget; new search tree, not checkpoint continuation. |

The reserve screen, nine previous remaining-gap MIPs and four corrected fixed-state pricing calls are complete. Do not relaunch them. Original k16–25: 60 original MIPs, 35 targets and 25 misses; separate longer searches recover 24 misses. The new C5 k25 job addresses the last unresolved miss. Keep original fixed-budget results separate. Source launch audit: `outputs/final_chain_gap_20260915/README.md`; worker remains byte-identical to its frozen source, so the preemption registry cohort remains `default_overnight_diagnostics_20260914` while root and case identify this campaign.

When the queue thins, inspect existing launch plans and pending artifacts first. Launch ready authorized work and repair demonstrated execution problems without a new approval request. A further controlled experiment should address a specific unresolved finding, have matched settings and provenance, and be checked against prior attempts before submission. Do not launch arbitrary repetitions simply to maintain a numerical job count. If no justified independent work remains, record that fact and report batch completion.

## Collection and recovery

- Read the cluster resource policy before any submission: `/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md`.
- SSH uses `/Users/nadan/.ssh/evsp-unicorn.sock`, `BatchMode=yes`, a short connect timeout and `nc437@unicorn-login-01.coecis.cornell.edu`. Notify promptly on access loss/restoration; collector failures are not automatically access failures.
- Preserve actual previous-k and own-CG dependencies. Never remove them to make a queued job run before its input exists. Validate artifact hashes and successful producer output before repairing dependency links.
- Independent default CG concurrency is50/all cases if fewer. Every CPU job excludes `scaglione-compute-01`. Held537227 and concurrent EVSPV2G experiments are protected.
- Routine MIPs use the default partition and documented preemption/requeue handling. Retain every attempt. Current compact and diagnostic campaigns intentionally have12600-second total /10800-second maximum fleet-stage budgets; do not overwrite those exceptions with default3600/1800 budgets.
- Collect with `outputs/meeting_20260910/collect_remote.py`, write a dated monitor snapshot, refresh `preemption_study/refresh.py`, then use `monitor_compact_delta.py`. Choose the preceding successful snapshot by embedded timestamp; compare canonical content for hashless CG rows.
- Rebuild the register/workbook only for material verified changes. Preserve the exact six active supplemental sources, compact JSON serialization and source-binding checks. Read relevant new runbook entries rather than the whole history.
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

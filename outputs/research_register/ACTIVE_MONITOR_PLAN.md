# Active overnight monitor — 15 September 2026

The user requests continued overnight monitoring, productive parallel use of Unicorn, and a clearer document after results arrive. The existing hourly heartbeat `unicorn-connection-and-research-progress` remains ACTIVE; it was updated rather than duplicated. Notify on meaningful findings, failure, access changes, completion or a needed decision. Quiet checks should finish without browser work or repeated briefings.

## Current state and priorities

Last full verified collection: `outputs/post_meeting_20260910/monitor/20260915T050815Z.json`, completed at01:16EDT. [Source report](../overnight_next_20260914/status_20260915T050815Z/README.md). Queue in the collection: **59 running jobs /36 true input dependencies**, no array throttle or invalid dependency. Scientific endpoints and late scheduler-only publications have separate timestamps in the report.

The running work already answers useful questions:

| Registered work | Question | What to check next |
|---|---|---|
| `chain_extension_20260914` | Can inherited-column solving reach k28? | Actual previous-k outputs and graph readiness; CG certificates and MIP targets separately. |
| `compact_seed_support_20260914` | Do the smaller starts work at k15? | Five unpublished MIPs at the last full collection. C3 core had 17 buses / bound15, so its pool limitation is not proved. |
| `compact_large_seed_20260914` | Do core and 512-sequence starts work at k20/k25? | 24 CG/MIP pairs. Three CG certificates: C3 k20 both methods and C2 k20 expanded; MIPs were pending. Keep a0e0 code distinct from the smaller e091 cohort. |
| `lp_support_pool_diagnostic_20260914` | Which added columns repair inadequate pools? | Finish 13 matched positive-LP/zero-LP-weight addition pairs. Zero weight does not mean zero reduced cost. |

The reserve screen, nine remaining-gap MIPs and four corrected fixed-state pricing calls are complete. Do not relaunch them or call them newly completed. The original k16–25 batch has 60 original MIPs:35 targets,25 misses; separate longer searches recover24 misses. C5 k25 remains unresolved. Broader recoveries do not alter the original fixed-budget control table.

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

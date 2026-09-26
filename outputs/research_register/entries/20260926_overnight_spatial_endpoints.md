# Overnight spatial-price expansion endpoints

## Final morning result — 26 September 2026, 08:09 EDT

The expansion is complete with **80/80 exact-once five-bus cleanups: 55 direct fresh-CG and 25 labelled GIRO-frontier fallbacks**. Charging is proved to solver tolerance in **61 finite cleanup pools**, while **19 remain time-limited**. Original CG pricing certificates remain **61/80**; no new CG certificate is claimed. Shared charger capacity remains unvalidated.

Recovery **520378** completed at 05:08:07 EDT after **45m57s**, with **17.71 GiB peak RSS**, 8 CPUs and an unchanged 48 GiB allocation. It closes the `mix1_two_price_split` cleanup with all 111 trips exactly once and no added fixed-duty fallback. The [morning audit](../../research_management_20260926/overnight_monitor/morning/README.md) passed [181 saved-receipt and route-arithmetic checks](../../research_management_20260926/overnight_monitor/morning/verification.json); [updated 80-cell table](../../research_management_20260926/overnight_monitor/morning/expansion_cells.csv).

At the saved 08:09:43 EDT snapshot, no scoped EVSP–DR jobs remain in the queue and all scoped jobs are terminal. The overnight heartbeat is **PAUSED**, confirmed separately through the automation tool. Final current-Doc and slide 55 publication passed [171 preservation/content checks](../../research_management_20260926/overnight_monitor/morning/publication/verification.json), separately from the scheduler and scientific audit. The [automation receipt](../../research_management_20260926/overnight_monitor/morning/automation_pause.json) records the pause.

## Frozen 04:12 endpoint and 04:30 recovery record — historical

The following sections preserve the earlier 79-cleanup state and running recovery. Their incomplete counts and next-heartbeat instructions are superseded by the final morning result above.

Snapshot: 26 September 2026, 04:12 EDT. [Entry point and remaining recovery](../../research_management_20260926/overnight_monitor/README.md). [80-cell audited table](../../research_management_20260926/overnight_monitor/endpoint_summary/expansion_cells.csv). Raw remote paths/hashes, attempts, scheduler receipts, executable reducer and 1,849 binding checks are retained beside it.

## Result and proof scopes

- Original CG: 61/80 pricing certificates; 17 CG time limits; 2 restricted LP stops at approximately the four-hour limit. No certificate is assigned to the latter 19.
- Original fresh-pool MIPs find a five-bus selection in 55/80 cases. The other 25 have no selection and use a labelled union fallback adding optimized GIRO trip-sequence columns; 17 fallback covers consist entirely of GIRO sequences.
- Exact-once cleanups succeed in 79/80: 54 direct, 25 fallback. All 79 have fleet 5 / bound 5 within their finite repair pools and pass individual-route replay. Charging is proved in 60 repair pools and remains time-limited in 19. Shared charger capacity is unmodeled.
- Sixteen jobs restarted 17 times. The 17 stale running attempt receipts have later terminal successors; these are not active queue entries or independent replicates.
- Original-root jobs 506710–506714 separately recover three cells. All three final cleanups use five buses; two prove their charging optimum within the repair pool and one remains time-limited.

## Execution and model

ROOT2 `/home/nc437/ladder-lite/spatial_tariff_expansion_20260925`; frozen commit 4a8b497e668be9962bca5906a8b69116fa634882. Original root `/home/nc437/ladder-lite/spatial_tariff_k5_20260925` is separate. Campaign inputs and resources are in `outputs/independent_review_20260925_spatial_tariffs/expansion/from_cluster/`; the per-cell audit records input/tariff hashes and source summaries, code commits, budgets, initialization/fallback origin and statuses.

Physics: 240 kWh / 350 kW, zero reserve, zero start fee, 2.5 kWh / 5 min graph, aggregate terminal-energy floor, no shared charger capacity, and energy-only deadhead pricing. CG uses covering; fixed-duty/cleanup uses exactly-once rows. MIX cohorts relax GIRO vehicle-family segregation. Correlated cohorts, synthetic prices and prespecified showcase selection remain declared limitations.

## Same-setting recovery

Job 498962 failed a duplicate-enumeration guard. Source selected hash `c771a2c97b8ba3a32af3201f0b7f3d898861e9c19b03d1f0ddb80df908ce5768`; source summary hash `684886c6dd791357cab107bc5eb951f951c91b21bf8b53a2171b3073d5551aac`. Route duplicate counts 10, 1, 0, 2, 11 imply at most 3,079 subsequences. The tested wrapper changes one byte in a separately executed copy: guard 10→11; all scientific code is otherwise byte-identical. Frozen checkout unchanged. No CG rerun or added GIRO columns.

Recovery 520378 started at 04:22:10 EDT. Original afterany predecessors 498917_43/498918_43 are terminal. Default partition, 8 CPUs, 48G, 5 h, requeue, `scaglione-compute-01` excluded. Failed attempt used 2.52 GiB; comparable successful posts used 14.38/17.40 GiB, supporting the unchanged 48G allocation. Source pins, code diff/tests, remote preflight, submit/start receipts and native output path are in [recovery](../../research_management_20260926/overnight_monitor/recovery/README.md). Completion is not claimed.

## Publication

Current Doc section 5 and weekly slide 55 carry the same scoped counts. Only the Doc date and two current-work paragraphs changed. Prior 54 slides, embedded images, notes, Doc figures/history/source links remain. [171-check publication verification](../../research_management_20260926/overnight_monitor/publication/verification.json). The next 08:00 EDT heartbeat checks recovery 520378, updates verified counts if warranted, gives a morning summary and pauses itself.

The raw fallback summaries' unconditional proof prose is overbroad; explicit certificate/status fields override it. No result artifacts were rewritten and no savings claim is inferred from stage completion.

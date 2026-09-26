# Final morning endpoint — 26 September, 08:09 EDT

The saved **12:09:43 UTC / 08:09:43 EDT** snapshot has no scoped `sx_`/`st_` jobs in the user queue and all scoped `sx_`/`st_` allocations terminal. Recovery **520378** completed at **05:08:07 EDT**, elapsed **45m57s**, with zero restarts (8 CPUs, 48 GiB request). Root confirmed the overnight heartbeat is **PAUSED** through its automation tool; that confirmation is separate from the local cluster snapshot.

## Recovery closes the last missing expansion cleanup

`mix1_two_price_split` now has **five buses and all 111 input trip indices exactly once**. Fleet and charging stages both report status 2, with finite cleanup-pool fleet bound 5 and **grid charging cost/bound 168.020**. Its continuously replayed charging cost is **159.7208272475**, a different accounting quantity, without a continuous-cost pricing certificate.

The five selected routes contain **39, 18, 12, 20 and 22** trips. Independent local checks count every index 0–110 once, recompute route/block hashes, sum route costs, and reconcile terminal energy. The summary's terminal energy **381.4390014 kWh** is the expanded-grid aggregate; the continuous aggregate is **385.7730016 kWh**. Both exceed the unchanged aggregate floor **379.7984451 kWh**. Saved native individual-route replay flags pass; shared charger capacity remains unvalidated. This local audit verifies saved route records and arithmetic, not a fresh physical simulation.

The recovery changes the enumeration guard **10 → 11** and retains the exhaustive subsequence method: per-route duplicated-trip counts **10/1/0/2/11** generate **3,079** sequences and **324,916** repair columns. It adds no fixed-duty fallback. The original failure **498962** and its failed post receipt are preserved. The original fresh-CG source remains SHA-256 `684886c6dd791357cab107bc5eb951f951c91b21bf8b53a2171b3073d5551aac`; its pricing certificate is unchanged.

## Final expansion counts

| Result | Count |
|---|---:|
| Original CG pricing certificates | 61/80, unchanged |
| Effective exact-once five-bus cleanups | 80/80 |
| Direct fresh-CG / GIRO-frontier fallback cleanups | 55 / 25 |
| Cleanup charging status 2 / status 9 | 61 / 19 |
| Individual-route replay reported passing | 80/80 |
| Shared charger capacity validated | 0/80 |

Status 2 means optimal to solver tolerance **within the specified finite pool**; status 9 remains time-limited. The 25 fallback cells retain their separate labels: 17 selected only GIRO sequences, 16 lack a CG certificate, and 7 fallback pre-cleanup charging MIPs timed out. Their unconditional `proof_scope` boilerplate must not override explicit statuses. The new cleanup's scope text correctly says finite frozen exact-once pool and no new CG certificate. It does not prove a full-model charging optimum or GIRO-operable MIX-family dispatch. Physics remain 240 kWh/350 kW, zero reserve, aggregate terminal floor, no shared capacity and energy-only deadhead pricing. No savings claim is made.

The original-root jobs **506710–506714** remain five separate recovered allocations covering three cells, as recorded in [the frozen audit](../endpoint_summary/README.md); they are not additional expansion cells.

## Files and lineage

- [Updated 80-cell CSV](expansion_cells.csv): only the missing cleanup is filled. Original `post_*` fields retain failed job 498962; `effective_cleanup_job_id=520378` and recovery fields identify its replacement. Other prior values remain unchanged.
- [Verification](verification.json): executed checks, route counts, canonical hashes, original/output source bindings and both energy/cost accounts.
- [Morning snapshot](snapshot.json): original source summary, failed post, recovery submission/attempt/design/summary/five routes and scheduler outputs.
- [Frozen 04:12 CSV](../endpoint_summary/expansion_cells.csv): preserved unchanged.
- Reproduce locally with `python3 outputs/research_management_20260926/overnight_monitor/morning/reduce_morning.py`.

Summary output SHA-256: `baa2b24af02bdba2385937e0799d1bf27012a03a38a8f4f5884bc3e1f4327a68`. Selected-routes SHA-256: `304994615e4e3077f319807bf055a398a05d2b4759221d5504a06156f655e481`. All seven embedded artifact hashes reproduce using their original JSON serialization; route/block canonical hashes are independently recomputed. The original selected-source and repair-routes payloads are not included in this snapshot, so their recorded hashes are retained but their bytes are not independently checked here.

## Publication and monitor completion

The current Doc section 5 and weekly slide 55 carry these final counts. [Publication verification](publication/verification.json) passes 171 checks: only two Doc paragraphs and slide 55 contents/notes changed; prior figures, slide notes, history and source links are preserved. Native before/after exports remain local with recorded hashes. [Automation receipt](automation_pause.json) confirms this overnight heartbeat is paused. No further submissions were made.

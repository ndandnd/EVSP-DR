# Captured endpoint audit — 26 September, 04:12 EDT

Read-only reduction of `../terminal_receipts_0400.json`, collected **2026-09-26 08:12:46 UTC**. This is the state of the captured jobs, not a new whole-cluster queue check. No SSH, solver, submission, or source mutation was performed.

## Scientific endpoints: 80 expansion cells

| Stage | Verified count | Scope |
|---|---:|---|
| Original fresh-CG pricing | 61 certified; 17 CG time limits; 2 restricted LP not optimal | Weighted expanded-event-graph LP only when explicitly certified |
| Original fresh-pool fleet MIP | 55 status 2; 14 status 9; 11 status 3 | 55 selections; 25 no-selection results. Status 3 concerns the finite pool under fleet cap 5, not full-model infeasibility |
| Original fresh-pool charging MIP | 53 status 2; 2 status 9; 25 absent | Finite original pool; these are pre-cleanup covers |
| Fixed GIRO duties | 80 fleet-5 proofs; charging 73 status 2 / 7 status 9 | Exact-once fixed-duty charging frontiers; stations/times reoptimized |
| Post branch | 55 direct fresh-CG cleanups; 25 fallback-plus-cleanup | Fallback adds GIRO fixed-duty frontier columns |
| Successful exact-once cleanup | 79/80: 54 direct, 25 fallback | All 79 fleet 5, status 2, finite-pool bound 5 |
| Cleanup charging | 60 status 2 / 19 status 9 / 1 absent | Optimal or time-limited within that finite generated cleanup pool |
| Cleanup validation | 79 exact-once + individual-route replay pass | Shared charger capacity unvalidated in all 79 |

**Status interpretation:** Gurobi 2 = optimal to the configured solver tolerance; 9 = time limit; 3 = infeasible for that specified finite capped problem. Missing is never counted as zero cost, a failed physical check, or a proof. A pricing certificate is distinct from a finite-pool fleet/charging proof and from replay. Grid charging costs/bounds and continuously replayed costs remain separate CSV fields; no price-saving conclusion is calculated.

**Fallback caveat:** 9/25 fallback cells retain an original fresh-CG pricing certificate; 16 do not and have no full-graph LP bound. Fallback charging has 18 status-2 and 7 status-9 endpoints. **17/25 fallback selections use only GIRO sequences**; these are not rerouting demonstrations. The saved fallback summaries contain unconditional `proof_scope` prose claiming MIP optimality/a valid CG bound even where explicit fields disagree. This audit trusts the explicit certificate and solver statuses. Any copied raw scope prose must be read conditionally; the union never creates a new pricing certificate.

**Unfinished cell:** `mix1_two_price_split`, post job **498962**, failed cleanup return code 1 at the duplicate-enumeration guard. Its original fresh CG has a selection; no cleanup summary exists in this snapshot. Duplicated-trip counts by selected route are **10, 1, 0, 2, 11**, giving **3,079** enumerated subsequences (`2^10 + 2^1 + 2^0 + 2^2 + 2^11`), with 12 distinct duplicated trips. The separately prepared same-enumeration guard recovery is outside this frozen endpoint result; it must not be counted complete here.

## Scheduler and restarts

| Expansion stage | Completed | Failed | Restarted jobs | Restart events |
|---|---:|---:|---:|---:|
| Fresh CG | 55 | 25 | 12 | 13 |
| Fixed duty | 80 | 0 | 4 | 4 |
| Post | 79 | 1 | 0 | 0 |
| Total | 214 | 26 | 16 | 17 |

The 25 failed CG allocations are exit-code-3 no-selection outcomes, each followed by successful fallback and cleanup. The 282 attempt receipts comprise 93 original-CG, 84 fixed-duty, 25 fallback and 80 cleanup attempts. Seventeen still say `running`—13 CG and 4 fixed—but every one lacks a summary and has a later terminal restart. They are stale attempt records, not running work. `loc_spatial_PARX_x2p00` reaches restart 2. Array scheduler IDs are mapped through array-job/task fields, rather than internal job-directory IDs. These accounting restarts establish repetition, not independent scientific replicates.

## Original-root recoveries remain separate

Jobs **506710–506714** are five completed allocations across three original-root cells; they are not part of the 80 expansion cells or 240 expansion jobs. Job 506710 yields a replayed exact-once five-bus cover, but charging remains status 9: grid cost **190.9599166431**, bound **190.8957202174**. The two fallback/cleanup pairs recover `loc_flat` and `loc_spatial_3127L_x0p50`; both final cleanups prove five buses and status-2 charging in their cleanup pools, using GIRO sequences. Shared capacity remains unvalidated. The receipt therefore contains 245 scheduler rows in total, including these five jobs.

## Model and provenance limits

The captured physics use **240 kWh, 350 kW, zero reserve, fee 0, 2.5-kWh/5-minute grid, aggregate terminal-energy targets, no shared charger capacity**, and energy-only deadhead pricing. Fresh CG covers trips at least once; fixed duties and successful cleanup enforce exactly once. MIX cross-family swaps are not operable under GIRO's vehicle-family restrictions. Overlapping duties make cohorts correlated. Preserve `(root, cell)` keys: repeated baseline names belong to distinct roots. Example selection still requires the preregistered eligible-population rank/median/range and `fig6_pond`; no selected example is called typical. The expansion README's `two_price_split` eligibility issue remains unresolved here.

- `expansion_cells.csv`: 80 auditable rows; stage status/cost/bound/replay, fallback origin split, source paths/hashes, commit and input identities.
- `original_root_recoveries.csv`: five separate recovery-stage rows.
- `scheduler_jobs.csv` and `attempt_receipts.csv`: all 245 accounting and 282 expansion-attempt rows.
- `summary.json`: reproducible counts and guard diagnosis. `provenance.json`: local snapshot hash, reducer hash and executed binding checks.
- Rebuild: `python3 outputs/research_management_20260926/overnight_monitor/endpoint_summary/reduce_receipts.py` from the repository root.

Remote original-file hashes are receipt-reported, not reconstructed from parsed JSON. Successful attempt output hashes and inter-stage source hashes are cross-checked. The 25 exit-code-3 CG attempts omit output hashes; each saved summary hash is instead bound by its fallback's `source_fresh_cg.summary_sha256`. Source experiment settings, resource requests and dependencies remain in `outputs/independent_review_20260925_spatial_tariffs/expansion/README.md` and `from_cluster/jobs.tsv`. This audit does not replace the frozen source receipts.

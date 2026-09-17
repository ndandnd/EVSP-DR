Six full single-factor arms are submitted: [experiment settings, job IDs and monitoring](action3_full_20260916/README.md). Replay array342540 covers every254,068sequence in each arm,50tasks at once; dependent CG/MIP jobs342668–342681 are queued. Other submission holds remain.

New post-hoc result: [mixed-group lambda in all102 LP endpoints](mixed_group_lp_20260916/README.md). F4 group mixing VERIFIED; F2 causal attribution UNRESOLVED. The9 k−1 cases average17.38% mixed weight versus8.28% in27 other k27–32 cases. No solver run.

Current Doc replacement and dated finish-time audit: [before/after verification](doc_rewrite_20260916/README.md). Six full single-factor action3 arms now authorized; older pilot-only restrictions below are superseded for those arms only.

# Independent review: execution and findings

**Latest sequencing:** [one pilot then five only on success; action2 unsubmitted; fullCG held pending recovery/memory; gated four-result report](advisor_sequence_20260916/README.md). This supersedes the earlier blanket hold only for the named actions.

**Advisor follow-up:** [six k5 F6 jobs launched; further plans held; checkpoint audit and corrected §1a counts](advisor_followup_20260916/README.md). No additional submissions or partition moves without user confirmation after current cluster load.

P0 was executed in the requested order: **3 → 2 → 4 → 1 → 5 → 6**. The table separates findings supported by evidence, claims contradicted by evidence, and hypotheses awaiting experiments. An unresolved hypothesis is not marked refuted merely because a run has not finished.

| Finding | Verified | Refuted or corrected | Evidence |
|---|---|---|---|
| **F4 — GIRO comparison** | Baseline PARX uses 240 kW and no 15% reserve. All 42 fixed trip sequences have feasible charging schedules after reoptimization at 240 kWh / 240 kW. | The 12/40 replay failures were measured at 350 kW. At 240 kW, 0/42 unchanged original schedules pass. This does **not** imply their trip sequences are infeasible. KEX is absent from this Partille model. | [Replay report and 42 new schedule witnesses](f4/README.md) |
| **F3 — lower bounds** | Corrected reduced-cost mathematics produces numerical integer-fleet lower bounds for all 52 uncertified endpoints. At k32: **32,32,32,31,31,32**. | K=integer incumbent fleet is not generally valid for the weighted objective; use U/100000. There are 52, not 54, uncertified rows. Pair each reduced cost with its own LP iteration. These are numerical event-model bounds, not exact-arithmetic or continuous-model proofs. | [Derivation and source checks](f3/README.md), [102-row augmented table](audited_chain_results.csv) |
| **F2 — near-GIRO reassembly** | Mean nearest-duty Jaccard is 0.644 when route weight=k, versus 0.462 when it is k−1. | Near-1 similarity is not observed across the first group. Longer MIPs match k in three of the nine k−1 cases. **The causal reassembly hypothesis remains unresolved.** | [Route-level overlap audit](f2/README.md), [figure](f2/jaccard_comparison.png) |
| **F1 — duplicate coverage** | All 128 schedules / 3,341 routes pass replay, including zero arrival-time grace. Assigning passengers once and retaining 14,450 duplicate traversals as empty driving preserves fleet and charging cost. | Duplicate coverage alone does not invalidate those dispatch costs. Conversion does not automatically create equality-master columns. Equal covering/partitioning optima require a route family closed under relabeling service legs as empty driving. | [Conversion and replay](f1/README.md) |
| **F6 — charging claims** | Original, fixed-duty and fresh-CG costs, charge counts and per-bus SOC are now audited. Both optimized arms contain sub-three-minute charges; some reach zero SOC. The result is one five-duty instance. | The 46-start count applies to peak08 only. Original costs are intervals because within-window power is unobserved. Larger-instance and stricter-rule savings remain unresolved. | [Three-arm table and per-bus SOC](f6/README.md) |
| **F9 — records** | Missing fee/tariff fields and stale subset labels were real. 308 case/telemetry records corrected; 3447 row IDs and existing scientific result fields preserved. | Workflow rows are not experiments. The old “largest 25” table covers a through-k25 subset; it now links the full k16–32 table. | [Register repair](f9/README.md) |
| **F5 — warm versus fresh** | All fresh CGs reach their certified weighted LP; the k15 integer misses are unresolved searches. Historical one-hour MIPs gave only 30 minutes to fleet search. | “Fresh CG cannot catch up” overstates what the experiment measures. Longer searches and seed repeats are needed to separate pool composition from search budget. | [P1 frozen-pool experiments](p1/README.md) |
| **F7 — scale and charging** | The large baseline ladder uses flat tariffs; the current fresh charging comparison is k5. | Generalization of the charging gain to k15 is unresolved. | [Matched k15 tariff experiment](p2/dr_mincharge/README.md) |
| **F8 — full/generalization inputs** | The large-chain experiments stop at k32. Frölunda raw data have 61 duties / 1,393 regular trips. | Partille's 987 raw regular rows contain 42 service-day variants. The frozen C1 full 40-duty selection has 948 trips, not 987. Frölunda k1/k2 are **VERIFIED**: 15/38 trips, 1/2 buses and exactly-once replay. For k2, overlapping mandatory trips independently prove that two buses are necessary. Larger results remain unresolved. | [Full-instance manifest](p3_full/manifest.json), [Frölunda source audit](p3_frolunda/inputs.json) |

## Runs requested by §4

All requested campaigns are submitted: **170 new Slurm submissions representing 183 tasks**, plus four reused seed-zero comparison cells. At 22:18 UTC the focused snapshot showed **91 running and 83 pending** registered jobs; some early stages had already completed. These are dated operational counts, not a scientific success rate. [Latest compact status](monitor/20260916T221824Z/SUMMARY.md).

| Item | Finding tested | Execution |
|---|---|---|
|7|F5|Six fresh k15 pools × three seeds × 3-hour fleet search;18 new jobs.|
|8|F2/F4|C5 k31 unchanged pool, 12-hour fleet search; one new job.|
|9|F2/F5|C1/C3/C4/C5 k32, three seeds; eight new jobs and four equivalent seed-zero runs retained.|
|10|F6/F7|Six k15 inputs × three synthetic tariffs plus one real SE3 day; fresh CG and fixed-duty arms,48 independent jobs submitted.|
|11|F4|C5 prefixes through 31, split into two vehicle groups; PARX 60 kW, 15% reserve and group-specific batteries;31 component inputs; 31 CG + 31 MIP jobs submitted.|
|12|F5|Random trip grouping with the same final 364 trips as C1 k15;14 sequential prefixes submitted (14 graphs + 14 CG + 14 MIP tasks). Intermediate stage numbers are not GIRO fleet targets.|
|13|F8|Fresh full 40-duty/948-trip Partille CG, 48 hours after separate graph preparation, followed by a saved-pool MIP; all three stages submitted.|
|14|F8|Frölunda pilot ladder from the unscreened seeded ordering; input/model audit passed; one 48-hour ladder job submitted.|

The P1 campaign has 27 accepted new jobs and four reused cells. Actual Gurobi Seed settings were checked in all 27 new logs. All CPU jobs exclude scaglione-compute-01 and use default_partition. Held jobs and other projects are untouched. All requested P2/P3 campaigns are now submitted. Submission receipts and the latest state are recorded in [ledger.json](ledger.json) and each campaign directory; a submitted or running job is not a completed scientific result.

P2's three-minute model allows constant controllable charging power up to 240 kW over the full chosen window. Zero-energy visits are idle, not charging starts. This is explicit modeling, not a claim to implement every GIRO charging rule. Its real-price arm is for internal analysis under the provider's data terms; raw or derived price outputs are excluded from public publication.

Original artifacts are preserved. The review itself is [REVIEW.md](../REVIEW.md); the neutral source map is [HANDOFF.md](../HANDOFF.md). All numeric audit tables retain source paths and hashes. The original Google Doc figure/history tabs and Slides are preserved; only the current Doc summary was updated.

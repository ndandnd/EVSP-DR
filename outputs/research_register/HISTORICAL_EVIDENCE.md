# Historical evidence inventory

Generated 2026-09-10 from the local checkout and retained workspace artifacts. This is a searchable index, not a claim that every Git object, Unicorn output, download, or historical run is present.

## How to read it

Each CSV row is one logical campaign, report, or evidence family. Repeated attempts, empty checkpoint directories, monitor snapshots, and PNG/PDF exports are grouped so they are not mistaken for independent replications. The `evidence_class` column uses:

- `historical evidence`: retained April/May raw logs, checkpoints, solutions, or legacy summaries.
- `results available`: current local result tables or auditable result bundles with explicit scope.
- `matched replay`: a same-input or source-bound replay/control, including charging and GIRO-witness comparisons.
- `audit`: provenance, implementation, physical-capacity, queue, plan, or interpretation work; it may contain no completed solver result.
- `figure-only/unreliable`: a visual artifact whose source binding is incomplete or mixed.

Use `result_scope` before quoting a number. In particular, `I_pool` is an integer optimum of one frozen generated column pool, `I_timed` is a censored incumbent, and a certified reduced-cost LP is scoped to its named discretized route graph. Neither automatically proves the unrestricted physical or shared-capacity problem.

## Inventory counts

| Measure | Count |
|---|---:|
| evidence class: audit | 15 |
| evidence class: figure-only/unreliable | 3 |
| evidence class: historical evidence | 9 |
| evidence class: matched replay | 9 |
| evidence class: results available | 3 |
| availability: local | 29 |
| availability: local audit + referenced remote raw | 1 |
| availability: local implementation + remote execution | 1 |
| availability: local plan + remote execution | 2 |
| availability: local plan only | 1 |
| availability: local report + remote replay artifacts | 1 |
| availability: local smoke + remote execution | 1 |
| availability: local submission + remote execution | 1 |
| availability: local summary + referenced-not-retained raw source | 1 |
| availability: partially classified | 1 |

## High-value anchors

| Evidence family | What the retained material supports | Main caveat |
|---|---|---|
| April/May legacy raw families | Finite historical pricing/MIP observations, including April 43 finite-model solves and May GIRO-seeded workflows. | Old grids, master semantics, seeds and route pools differ; no clean algorithm regression is present. |
| April selected 175-trip anchor | 12-bus timed incumbent in the retained legacy trace; later full-cache audit has a separate 10-route named-event-model witness. | Original run identity and pool are incomplete; do not call the 12-bus row a global optimum. |
| Current six nested chains | Exact row-level LP/MIP outcomes for 84 random nested cells, with event-grid hashes, CG stop/proof metadata and physical replay fields. | Six dependent chains are a sample, not a statistically significant estimate for all 40 duties; large cells are censored. |
| Full-cache witness audit | Target-fleet GIRO partitions physically replay in the current named event model for 12 above-target pool cases and the 175-trip extension. | This diagnoses missing pool columns; it is not RAW recovery and omits shared station capacity. |
| Warm p3 chain | A controlled inherited-column p3 comparison; k8 has the same LP objective but different pool integer outcome (fresh proves 9, warm proves 8). | Only p3 used inheritance; import time dominates k10; one case cannot establish a general warm-start speed claim. |
| Charging tariff study | Three-way original repriced/fixed-duty/joint costs at peaks 08/12/18 on the matched 62-trip cohort. | Original power trace and terminal policy are unknown; optimized schedules carry less terminal energy and lower modeled start fees. |
| GIRO source audit | Documented 15% floor, 65% recharge target, 60-kW depot, nonlinear opportunity charging and station/blocking constraints. | Current production master does not enforce all documented operational constraints. |

## Current nested-chain result snapshot

The detailed machine-readable source is [outputs/meeting_20260910/CHAIN_DIAGNOSIS_20260909.csv](</Users/nadan/Documents/projects/demandresponse/outputs/meeting_20260910/CHAIN_DIAGNOSIS_20260909.csv>). The following compact table is copied from the dated report; “fleet” is the finite-pool integer outcome, not a full route-space optimum:

| target k | chain 1 | chain 2 | chain 3 | chain 4 | chain 5 | chain 6 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 3 | 3 | 4 | 3 | 3 | 3 | 3 |
| 4 | 4 | 5 | 7 | 4 | 8 | 5 |
| 5 | 5 | 8 | 6 | 9 | 13 | 5 |
| 6 | 12* | 13 | 13 | 14* | 24* | 6 |
| 7 | 27* | 25* | 21* | 21* | 22* | 8 |

`*` marks a time-limited MIP status in the dated diagnosis. The CG terminal route-weight endpoint is approximately k for 82/84 cells because an independently computed simultaneous-trip overlap lower bound is k and the feasible fractional endpoint attains it; that does not imply the frozen pool contains a k-route integer combination.

## Git and retention boundary

- April/May anchors are tied to commits `8ef8049` (artifact retention), `7c564da` (GIRO initialization), `323c586` (continuation fixes), `22186da` (pricing stopping/escalation), and `58772c7` (full-SOC start/no artificial initial charging cost).
- The local `git log --all` has no tracked commits from 2026-05-22 through 2026-07-29. That is an evidence gap, not evidence that no work happened.
- July/August provenance includes the greedy reset (`0d7b48b`), event/pricing correctness (`ecfec4c`, `2f5935f`, `b96c046`, `e1b436b`) and durable exact-CG pools (`c74148c`).
- Current result packages bind to later commits listed in the CSV. The local checkout has many `codex/` and remote branches, but branch existence alone is not an experiment; only rows with retained outputs are indexed as evidence.

## Missing and unclassified material

Raw directories cited by historical reports but not retained locally include `analysis/scale_ladder/ll_20260820c`, `analysis/event_uniform_envelope_20260821`, `analysis/research_control_tower_20260830`, `analysis/legacy_exact_20260805`, and several Unicorn campaign roots under `/home/nc437`. The reports preserve some summaries, hashes and remote paths, so those entries are indexed as partial or referenced-not-retained rather than silently promoted to local results.

Within local `src/results`, many `ckpt_*`, `diag_*`, empty retries and trace-only files remain in the `unclassified_local_leftovers` row. They need an explicit source hash and run identity before being promoted into the register. This avoids inflating the sample with retries or treating a screenshot as a solver result.

## Reproduction and search

The CSV can be filtered by `evidence_class`, `campaign_or_family`, `period`, `availability`, or `comparability_warnings`. Primary paths are absolute for direct opening in the workspace. The generation script is [outputs/research_register/build_historical_inventory.py](</Users/nadan/Documents/projects/demandresponse/outputs/research_register/build_historical_inventory.py>).

Raw numeric claims remain anchored in the linked audit reports: [outputs/meeting_20260910/HISTORY_EVIDENCE.md](</Users/nadan/Documents/projects/demandresponse/outputs/meeting_20260910/HISTORY_EVIDENCE.md>), [outputs/meeting_20260910/HISTORICAL_175_TRIP_REGRESSION_AUDIT.md](</Users/nadan/Documents/projects/demandresponse/outputs/meeting_20260910/HISTORICAL_175_TRIP_REGRESSION_AUDIT.md>), [outputs/meeting_20260910/CHAIN_DIAGNOSIS_20260909.md](</Users/nadan/Documents/projects/demandresponse/outputs/meeting_20260910/CHAIN_DIAGNOSIS_20260909.md>), and [outputs/meeting_20260910/LP_K_EQUALS_AUDIT_20260910.md](</Users/nadan/Documents/projects/demandresponse/outputs/meeting_20260910/LP_K_EQUALS_AUDIT_20260910.md>).

# Run record: `ll_20260820c` (curator backfill, 2026-08-21)

## Why this directory does not contain the normalized CSV set

`scripts/ladder_lite/record_results.sh` is the sanctioned writer for
`records/RESULTS_LOG.csv` and for the normalized CSV set in this directory. It
requires `$LL_ROOT/normalized/{cg_run_summary,mip_run_summary,
cg_iteration_long,mip_checkpoint_long}.csv` plus the campaign's
`approved-plan.json` and `campaign.json`. **None of those files was ever
committed to any git ref** (verified across all refs on 2026-08-21); durable
copies exist only on the cluster under `~/ladder-lite`
(`STATUS_20260821.md` §8). Executed on this branch it fails closed:

```
$ LL_ROOT=/tmp/ll_empty LL_PYTHON=python3 bash scripts/ladder_lite/record_results.sh ll_20260820c
normalized summaries missing: /tmp/ll_empty/normalized
exit=1
```

## What was recorded instead

`scripts/curator/backfill_results_log_ll_20260820c.py` (committed, idempotent)
appended 53 rows to `records/RESULTS_LOG.csv` from the two committed, reviewed
campaign aggregates:

| source | rows | group |
|---|---|---|
| `analysis/scale_ladder/ll_20260820c/ladder_summary.csv` | 23 | `CG` (primary 15 kWh / 10 min) |
| `analysis/scale_ladder/ll_20260820c/resolution_matrix.csv` | 30 | `CG_SENSITIVITY` (non-primary grids only; the nine 15/10 matrix rows are the same runs as the `CG` rows and are not duplicated) |

Field-by-field provenance:

- `label=curator_backfill_from_committed_aggregates`; rows are derived, not
  emitted by the normalizer. `artifact_path`/`artifact_sha256` name the exact
  committed aggregate each row came from.
- `commit=339db0ab…` and campaign identity per `D0010` (local-only cluster
  commit; plan sha256 `063e413f…9220a277`).
- `route_weight_meaning` is copied verbatim from the aggregate
  (`combined-cost-master route weight`). Per `D0019`, certified rows may
  additionally be read as fleet LP lower bounds for the discretized model at
  the stated grid; the rows keep the weaker meaning the artifact states.
- `wall_s` is derived from `elapsed_h` (3-decimal hours in the ladder summary,
  ±1.8 s; 2-decimal hours in the matrix, ±18 s).
- `status=censored` carries the aggregate's own stop label
  (`pricing_censored_max_iters`, `near_converged`, `mid_flight`, or
  `uncertified_at_stop` for matrix rows with `certified=False`).
  Per `B0019`, `pricing_censored_max_iters` is an iteration-cap artifact, not
  a wall-budget statement.
- Fields the aggregates do not carry are left **empty**, not inferred:
  `arm`, `budget_s`, `master_s`, `pricing_s`, `max_rss_mb`, all `mip_*`
  fields, `first_incumbent_s`, and `cg_rep` for sensitivity rows.

## Schema note (2026-08-21, after this backfill)

`records/RESULTS_LOG.csv` was later migrated to the qualified-optimality
schema (nine appended columns; see
`records/runs/model_optimality_20260821/README.md`). The 53 rows of this run
are preserved byte-for-byte in their original columns; the new columns are
empty because CG rows make no integer-optimality claim.

## What is absent, and why (see `ARTIFACT_INVENTORY.csv`)

Raw per-cell status JSONs, iteration traces, MIP checkpoints/progress
directories, the normalized CSV set, the approved plan, and sacct snapshots
were never committed to git and exist only on the cluster. In particular there
are **no MIP rows** in this backfill: no MIP_RAW/MIP_KNOWN evidence for this
campaign is committed anywhere, so recording rows for them would fabricate
provenance. RAW and KNOWN arms therefore share no row here trivially; any
future MIP rows must come from `record_results.sh` run against the cluster's
normalized outputs.

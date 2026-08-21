# Run record: `model_optimality_20260821` (curator backfill, 2026-08-21)

27 qualified rows appended to `records/RESULTS_LOG.csv` by
`scripts/curator/backfill_model_optimality_20260821.py` (committed,
idempotent), after the schema migration by
`scripts/curator/migrate_results_log_qualified_optimality.py`. All values are
transcribed from committed artifacts on this branch (path + sha256 per row).
All results are at **historical 300/300 physics** (see STATUS `R7a`).

## Why the schema changed

An unqualified proven/optimal flag conflates two different claims. Gurobi's
`OPTIMAL` on a pool MIP means **optimal over the columns it received**,
nothing more — even when the source CG LP is certified, because an
integer-useful route can carry zero or positive reduced cost at the LP optimum
and never enter the pool. Every optimality statement now carries:
`source_cg_certified/stop_reason/iterations` (the pool's provenance),
`pool_fleet_proven`/`pool_mip_bound` (finite-pool claim),
`model_fleet_proven`/`model_optimality_method`/`optimality_scope`
(discrete-model claim: `sandwich | arcflow | branch_and_price`), and
`physical_witness_valid`.

## The sandwich rule (operator, 2026-08-21)

A certified fleet LP `L` is a valid model-wide lower bound and fleets are
integral, so `I_model >= ceil(L)`. Where a physically validated incumbent
equals `ceil(L)`, the discrete-model optimum is proven with **no extra
solve** (`model_optimality_method = sandwich`). Rows promoted under the
operator's blanket authorization (certified LP at target, pool MIP returned
target) whose incumbent replay is not recorded in committed evidence carry
`physical_witness_valid = ""` and say so in `notes`.

## What the 27 rows contain

| block | rows | scope |
|---|---:|---|
| arc-flow oracle, primary grid | 7 proven + 2 censored | `discrete_model` proofs for all k2 (3/3/3), all k3 (4/4/3), k05_s2 (5); k05_s1 (6–11) and k05_s3 (≥5) censored, unresolved |
| corrected branch-and-price | 2 | `k02_s2` = 3 (primary) and = 2 (1 kWh/5 min), both fully replayed |
| standalone sandwich | 1 | `k02_s1` @ 1/5 = 2: certified LP 2.0000 + replayed PRE_PHASE1 witness (that driver's bounds are invalid; its replayed incumbents remain valid upper bounds per the artifact itself) |
| primary-grid RAW pool MIPs | 8 | incumbents 4/4/7/5/10/4/11/6 from the arc-flow table; only `k02_s2` has a committed proven-for-pool record (D0023); **no discrete-model claim from any of these rows** |
| fine-grid RAW pool MIPs | 4 + 3 | D0017 cells promoted by sandwich (k02_s2, k02_s3 @ 1/5 = 2; k03_s1, k03_s3 @ 1/10 = 3); B0021's k05_s2 trio (8/10/13) stays `finite_pool` — certified LP 5.0 gives ceil 5 ≠ incumbent, the sandwich does not close |

`k02_s1` primary is the operator's worked example of a sandwich that does not
close: ceil(2.1818) = 3 but the pool returned 4 — resolved by arc-flow (= 3),
exposing exactly one bus of pool-composition excess.

## What is absent, and why

The **~100-cell cluster pool-MIP table** (cells like `k08_s3_g2_b10`) was
never committed to any git ref; it exists only on the cluster. Recording those
rows here would fabricate provenance. When the campaign's normalized outputs
reach the repository, the updated `scripts/ladder_lite/record_results.sh`
emits the qualified columns automatically (`optimality_scope = finite_pool` at
most), and sandwich promotion against the committed certified LPs can then be
applied by a script following `backfill_model_optimality_20260821.py`. The
operator's exemplar row — `k08_s3_g2_b10`: "63 buses, proven optimal over an
80-iteration finite pool; source CG truncated; no discrete-model optimality
claim" — is exactly what that emission will produce, but its underlying data
is not in git today.

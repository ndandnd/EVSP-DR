# Provisional findings — `cursor/ladder-lite-20260819-2969`

These labels are branch-local only. They are not authoritative bug or decision
IDs; a records curator may accept, rewrite, merge, or reject them.

## LOCAL-1 — diversification rounds scale and then saturate

- Claim: `--diversify-rounds` controls the executed loop count. In the
  controlled 3-versus-10 test, rounds 4–10 execute but add no new incidence or
  cheaper replacement, so equal endpoints are consistent with saturation.
- Evidence path:
  `tests/test_exact_initial_pools.py::ExactInitialPoolTests::test_diversification_round_count_scales_and_is_instrumented`
- Producing commit SHA:
  `425b71ed17be64c60cd1b46c2aa9dd5fb94562eb`

## LOCAL-2 — greedy warm pools support the frozen commensurate grid

- Claim: the prior greedy failure was an unsupported-grid restriction, not an
  algorithm regression. Warm-pool realization now supports the historical
  300/300, 15 kWh/10 min grid and commensurate grids including frozen 240/240,
  10 kWh/10 min physics. The published fixed-duty evidence producer remains
  byte-identical.
- Evidence paths:
  `tests/test_exact_initial_pools.py::ExactInitialPoolTests::test_all_initial_pool_modes_start_nonempty_on_k2_declared_physics`;
  cluster report `~/ladder-lite/big240/k30_r1_greedy.err`;
  local k30 smoke artifact
  `results/greedy_regression_probe/isolated_k30_g240.json`
- Producing commit SHA:
  `780a75a60e335967fbc20672286627fa95f1b078`

## LOCAL-3 — target-pool feasibility is unavailable at `6d02f91`

- Claim: `src/target_pool_feasibility.py` does not exist at commit `6d02f91`;
  it first appears at `eccd248`. Chained analyses bound to `6d02f91` cannot
  silently assume that capability.
- Evidence path: Git object
  `6d02f91:src/target_pool_feasibility.py` (absent), compared with
  `eccd248:src/target_pool_feasibility.py`.
- Producing commit SHA:
  `eccd24884ca9278a82e5319873bcf75394ef8bfd`

## LOCAL-4 — retain `columns_per_iter=30`

- Claim: at equal wall time on k20_s1, 30 columns per iteration reached route
  weight 20.12 at 2000 iterations, versus 20.73 at 1282 iterations for 100 and
  21.57 at 877 iterations for 300. The active plan therefore retains 30.
- Evidence path: `STATUS_20260820.md` section 5 identifies the `cpi` batch;
  endpoint values were supplied by Nathan, but no raw cluster artifact path was
  supplied to this branch.
- Producing commit SHA:
  `16209af3af400911f7f63ffadda24dd23078fe80`

## LOCAL-5 — cross-resolution k02_s2 union does not repair the primary pool

- Claim: the union reaches targets 3 and 2, but the target-3 witness uses only
  fine-grid routes and the fine pool reaches 2 alone. The primary finite pool
  remains infeasible at target 3, so this is a route-space change rather than
  recovery of missing primary-grid columns.
- Evidence path:
  `analysis/k02_s2_cross_resolution_union_20260820/summary.json`
- Producing commit SHA:
  `f44db98a48d266f52ae20c11a7908ce006c62ca9`

## LOCAL-6 — 18 duty-union instances produced; independent validation pending

- Claim: selections 4–6 for k2/k3/k5/k8/k13/k20 were produced from the
  original seed-20260803 duty-union families, with the legacy 22 rows preserved.
  This is a producer claim only. The branch does not claim independent
  validation; a separate curator/validator must verify the instances.
- Evidence paths:
  `data/scale_ladder/instances/duty_union_extension_seed20260803.json`;
  `data/scale_ladder/instances/SIX_SELECTION_DUTY_UNION_EXTENSION_20260821.md`
- Producing commit SHA:
  `72c7bf418ddd175de0cf18371c24b72ecd79ab68`

## LOCAL-7 — instance features built; RAW recovery association not estimable

- Claim: a 40-instance duty-union feature table now records trip/duty counts,
  direct-deadhead density and energy share, service-energy intensity, layover
  slack, time-feasible station-bridge reachability, and five-grid duty
  representability. No auditable normalized RAW integer-result rows are tracked
  in this checkout or available git refs, so no feature/recovery correlation or
  threshold claim is currently supportable.
- Evidence paths:
  `analysis/raw_recovery_feature_audit_20260821/instance_features.csv`;
  `analysis/raw_recovery_feature_audit_20260821/summary.json`;
  `analysis/raw_recovery_feature_audit_20260821/README.md`
- Producing commit SHA:
  `8f8907738c03aea01f051359dbc3158579bbbe60`

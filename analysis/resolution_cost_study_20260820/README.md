# Resolution-cost study

Date: 2026-08-20.

## Instrumentation

Every exact-CG status now records:

- `dag_nodes`, `dag_arcs`, `dag_build_wall_s`;
- every iteration's `master_wall_s`, `pricing_wall_s`, `columns_added`;
- `iterations_to_certificate`, `wall_to_certificate`, `certified`;
- `peak_rss_mb`, `pool_columns_final`;
- distinct `stop_reason` values including `max_iters`, `wall_limit`, and
  `memory`.

The append-only iteration CSV remains byte-compatible with historical runs.
The richer full iteration sequence lives in status JSON. Integral block sizes
retain their historical integer serialization; 2.5-minute blocks are now
supported through pricing, physical replay, and pool validation.

## Frozen full study

`resolution_cost_plan.json` contains 168 jobs:

- 18 instances: 6 scales (`k2`, `k3`, `k5`, `k8`, `k13`, `k20`) × 3
  selections;
- 7 grids: `(15,10)`, `(10,10)`, `(2.5,10)`, `(10,5)`, `(2.5,5)`,
  `(10,2.5)`, `(2.5,2.5)`;
- all 126 instance-grid cells at 240 kWh / 240 kW;
- 42 bridge cells for k2/k3 at 300 kWh / 300 kW.

`max_iters=1,000,000,000`, so practical censoring is wall or memory. Existing
scale budgets are retained: 2 hours for k2/k3/k5, 6 hours for k8/k13, and
12 hours for k20. Resource fields and estimated DAG-node upper bounds are part
of every job.

The commensurability premise has two explicit exceptions:

| profile | grid | charge/block | credited on SOC grid | loss |
|---|---|---:|---:|---:|
| 240/240 | 15/10 anchor | 40 kWh | 30 kWh | 10 kWh |
| 300/300 | 15/10 anchor | 50 kWh | 45 kWh | 5 kWh |
| 300/300 | 10/5 | 25 kWh | 20 kWh | 5 kWh |
| 300/300 | 10/2.5 | 12.5 kWh | 10 kWh | 2.5 kWh |

All other requested profile/grid combinations are commensurate. The
noncommensurate 15/10 cell is retained as the historical anchor, not relabelled.

The agent never submits cluster work. Nathan can regenerate the hash-bound plan
with a cluster artifact root and execute it with
`src/resolution_cost_study.py`.

## Local validation

Local validation executed all 84 k2/k3 cells with four workers and a
180-second allocation per cell (120 seconds usable before the serialization
margin):

| outcome | count |
|---|---:|
| status JSON produced | 84 / 84 |
| certified | 48 |
| wall-limit censored | 36 |
| memory / max-iteration / process failures | 0 |
| certified at 240/240 | 25 / 42 |
| certified at 300/300 | 23 / 42 |

Certification by grid:

| profile | 15/10 | 10/10 | 2.5/10 | 10/5 | 2.5/5 | 10/2.5 | 2.5/2.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 240/240 | 6 | 6 | 4 | 4 | 1 | 3 | 1 |
| 300/300 | 6 | 6 | 3 | 4 | 1 | 2 | 1 |

The long-form output has 168 rows. The unexecuted k5-k20 rows are explicit
`missing` rows rather than silent omissions. MIP columns are blank locally;
passing MIP result roots to the summarizer fills integer fleet and integer-gap
fields.

## Local scaling fits

The requested model is

`log(y) = c + a·log(trips) + b·log(1/soc_step) + d·log(1/block_min)`.

240/240 fits:

| response | rows | trips exponent | inverse SOC exponent | inverse block exponent | R² |
|---|---:|---:|---:|---:|---:|
| DAG nodes | 42 | 0.029 | 0.976 | 0.971 | 0.99996 |
| wall to certificate | 25 | 3.511 | 0.579 | 0.920 | 0.92279 |

300/300 bridge fits:

| response | rows | trips exponent | inverse SOC exponent | inverse block exponent | R² |
|---|---:|---:|---:|---:|---:|
| DAG nodes | 42 | 0.030 | 0.980 | 0.971 | 0.99997 |
| wall to certificate | 23 | 3.779 | 0.429 | 1.064 | 0.91360 |

The nearly identical DAG exponents across physics profiles are a useful sanity
check. Node count is almost linear in both grid refinements because charge-state
nodes dominate at k2/k3. The large wall-time trip exponent is directionally
consistent across profiles, but it is fitted only over 29–71 trips and only on
certified cells.

## k40 prediction

Target: 947 trips, 240 kWh / 240 kW, 1 kWh / 5 minutes.

| quantity | prediction |
|---|---:|
| multiplicative fitted DAG nodes | 497,780 |
| structural trip-plus-charge DAG upper count | 679,381 |
| fitted wall to certificate | 1,969 hours (82 days) |
| affordability threshold | 24 hours |
| affordable? | **No** |

The fitted point estimate is roughly 82 times the existing per-cell budget.
Uniform 1 kWh / 5 minute refinement is therefore not an affordable k40 path
under this sanity model.

This is not yet a cluster-calibrated forecast: 947 trips and 1 kWh are both
outside the local fit ranges (29–71 trips and 2.5–15 kWh), the multiplicative
node model approximates an actually additive trip-plus-charge graph, and the
wall fit excludes censored rows. The k5-k20 cluster grid is required to narrow
the forecast, but the local result already gives quantitative motivation for
the event-based pricer.

## Artifacts

- Internal plan SHA256:
  `0608f70d6ce79c7e33dae47a619350e1ad1eae2bf144b621dcbfe182537862e8`
- Plan file SHA256:
  `e9377c89cb3465eb610cb7f9d85e8fa49baa492200c145fa51e4d8adb172d72b`
- `local_k2_k3/resolution_cost_long.csv` SHA256:
  `e4de42759d72c470896f2d350f4728bd45ba837a9446c7ce9addc031866149b8`
- `local_k2_k3/resolution_cost_extrapolation.json` SHA256:
  `176d7a115a2aa4d09d521e6b7b1b130bfce7793eebcf2bd1dad0435fc08a35d3`

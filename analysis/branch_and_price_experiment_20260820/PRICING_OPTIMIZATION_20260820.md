# Signature-cache optimization and fine-grid k2 proof

Date: 2026-08-20.

## What was actually slow

The expanded SOC-by-time network was already built exactly once per instance.
Constrained pricing did not reconstruct it for each disjunct. The measured cost
was repeated exact DAG scans over the canonical required/forbidden signatures.

The safe optimization is:

- cache structural node masks by `(required trips, forbidden trips)`;
- remove signatures dominated by a less restrictive valid signature;
- cache a route result only by `(signature, objective, exact trip-dual vector)`.

Caching a route result by signature alone is invalid because reduced costs and
the minimizing path depend on the current node LP duals.

## Before/after primary-grid pricing

Same instances, root pools, physics, grid, limits, and 10x kill rule:

| cell | metric | before | after | change |
|---|---|---:|---:|---:|
| k02_s1 | pure pricing wall s | 38.069 | 25.750 | -32.4% |
| k02_s1 | total wall s | 45.087 | 32.753 | -27.4% |
| k02_s1 | exact signature solves | 1142 | 950 | -16.8% |
| k02_s1 | structural-mask hits | — | 854 | — |
| k02_s1 | dual-aware result hits | — | 16 | — |
| k02_s3 | pure pricing wall s | 126.963 | 102.330 | -19.4% |
| k02_s3 | total wall s | 149.330 | 124.938 | -16.3% |
| k02_s3 | exact signature solves | 2682 | 2682 | 0% |
| k02_s3 | structural-mask hits | — | 2586 | — |
| k02_s3 | dual-aware result hits | — | 0 | — |

Solved-node slowdown ratios, relative to each run's root Phase-II pricing call:

| depth | k02_s1 before | k02_s1 after | k02_s3 before | k02_s3 after |
|---:|---:|---:|---:|---:|
| 1 | 2.49x | 1.98x | 2.71x | 1.94x |
| 2 | 4.98x | 4.09x | 5.09x | 4.10x |
| 3 | 9.69x | 8.40x | 9.70x | 7.78x |
| 4 | 18.49x | 15.55x | 18.50x | 14.52x |
| 5 | 32.96x | 27.14x | 35.15x | 27.45x |
| 6 | 49.04x | 26.53x | 34.33x | 27.45x |

The optimization helped but did not satisfy the kill criterion. Exact
dual-aware result reuse was rare because dual vectors changed between master
iterations and sibling nodes. The default exact depth cap is therefore 6.
Deeper nodes remain an open frontier; pricing never becomes inexact.

## Highest-value run: k02_s2 at 1 kWh / 5 minutes

Executed locally at 300 kWh / 300 kW, reserve 0, flat tariff, RAW columns:

| root Phase-I artificial mass | root fleet LP | conservative fleet LB | integer fleet | proven? | nodes | pricing wall s | total wall s |
|---:|---:|---:|---:|---|---:|---:|---:|
| 0 | 2.000000000 | 1.999999971 | **2** | **yes** | 1 | 0.722 | 6.816 |

The 2-bus partition passed full physical replay. Integerizing the conservative
fleet bound gives `ceil(1.999999971) = 2`, so the root closes without branching.

This proves, for the 1 kWh / 5 minute discretized route space, that the model
minimum is **2 buses**. It is end-to-end RAW recovery of the industrial fleet by
our own exact pricing and master pipeline, with no injected routes.

The proof is fleet-only. The stored candidate cost is 200129.44; charging cost
was not separately optimized after fleet optimality.

## Validation and bindings

- Complete branch-and-price module: 16/16 tests pass.
- Randomized exhaustive multi-SOC oracle: pass for combined, Phase-I, and
  fleet-only objectives.
- Network build count: 1.
- Baseline source:
  `resolution_matrix.csv` SHA256
  `392cdba0002c907bf8b2ce2c0beb0e3f8de9a5bc0faee4ebfa397dc4daf984a7`.
- Code commit:
  `8a33b187a93a40572d02782f8ddac4ee56821dc8`.
- Fine-grid result SHA256:
  `229b7755b6ac8c43cc643e2d9b9e4b3d2a192762c342db8d6eb9f47d3f7760c1`.
- Fine-grid node-ledger SHA256:
  `5073570af04f549c8615ce4cf236f8bbaaa59e72eab24f70f8f0cab1986464c8`.

All runs were local. No cluster jobs were submitted.

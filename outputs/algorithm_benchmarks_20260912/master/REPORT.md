# Omitted incidence construction: measured local prototype

The benchmark omits the unused SciPy matrix while retaining the persistent Gurobi master, its full route validation/synchronization and exact same frozen pools. No production module was edited. These workloads are synthetic incidence structures, not physically validated bus duties.

## Measurements

Seven alternating repetitions on macOS ARM64, Python 3.12 and Gurobi 12.0.1; solver and BLAS threads set to one. Timed sections share an exclusive lock with the other local benchmarks. Medians are reported; raw samples and input hashes are in [results.json](results.json).

| Routes | Nonzeros | Original preparation | Metadata-only preparation | Local ratio | Absolute saved per call |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 8,958 | 1.957 ms | 0.030 ms | 66.23× | 1.927 ms |
| 10,000 | 112,126 | 19.922 ms | 0.296 ms | 67.32× | 19.626 ms |
| 50,000 | 571,999 | 99.196 ms | 1.515 ms | 65.48× | 97.681 ms |

This large ratio measures avoiding a data structure that the backend does not consume. It is not a solver-wide speedup. The replacement still scans route lengths for telemetry; incremental nonzero tracking was not tested.

## Paired persistent-master replay

Each replay starts a new model, adds an identical sequence of pools, solves after each change, and includes one cheaper same-incidence cost replacement. There are 80 trip rows, 1,040 final route columns and 14 solves per replay. No pricing or graph work occurs. Total timings include model construction, preparation, synchronization, optimization, extraction and independent numerical audits; model disposal is outside the timer.

| Master sense | Baseline median | Prototype median | Local replay ratio | Baseline range | Prototype range |
|---|---:|---:|---:|---:|---:|
| cover | 78.10 ms | 58.09 ms | 1.344× | 71.91–81.00 ms | 57.01–63.12 ms |
| partition | 86.17 ms | 66.96 ms | 1.287× | 81.26–90.14 ms | 65.34–71.87 ms |

All paired LP objectives, route weights, artificial totals, route solutions and trip duals matched exactly in these runs. Every solve passed independent route/artificial reduced-cost checks, covering dual-sign checks and a primal–dual objective-gap check, in addition to the adapter’s primal checks. Empty, repeated-trip and unknown-trip routes remain rejected by Gurobi synchronization. The cheaper column replacement remains active. These are restricted-master checks; no global pricing certificate or physical result is claimed.

## Interpretation and next step

This is a real local improvement in preparation and on the specified master replay. Production CG has additional work and a different pool trajectory; the measured master ratio cannot be applied to its full runtime. Historical per-case timing shares in [the profile analysis](../profile/REPORT.md) imply that removing all incidence time at zero replacement cost would improve total runtime by at most roughly 5–6% for the median cases at k5/10/15. That is a conditional ceiling, not an observed cluster improvement.

Next integration task: omit the matrix in ordinary, final and diversification Gurobi calls while preserving telemetry and SciPy behavior, then run paired full-CG cases. This harness does not modify those production call sites.

Reproduce with `python3 outputs/algorithm_benchmarks_20260912/master/benchmark_master.py`. It uses the existing local Gurobi license; no license contents or credentials are recorded. Source hashes match baseline `a29992196acb74d02b8c7891be4061718889999f`.

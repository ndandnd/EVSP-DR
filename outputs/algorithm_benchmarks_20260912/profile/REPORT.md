# Baseline profile and conditional ceilings

These are measured baseline phase shares and arithmetic scenarios, **not measured improvements from implemented changes**. Proposed changes were not implemented in this frozen baseline. Current local prototypes require their own paired measurements.

Source: `../../algorithm_review_20260912/runtime_evidence.json`, reconciled exactly against its frozen collector snapshot (98 records; 84 fresh cache-hit cases). The following rows use six fresh covering/singleton cases per k. Entries are **median [minimum, maximum] of per-case ratios**, never ratios of medians.

| k | Wall minutes | Incidence % | Pricing batch % | Enrichment only % | Master % | Graph load % |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 17.37 [8.43, 30.56] | 5.43 [4.12, 9.01] | 81.19 [73.46, 85.77] | 35.29 [22.56, 47.26] | 12.32 [8.48, 16.30] | 0.35 [0.22, 0.60] |
| 10 | 88.72 [48.94, 127.24] | 5.27 [4.64, 7.10] | 75.63 [65.82, 78.88] | 14.43 [9.92, 16.52] | 18.27 [15.68, 26.49] | 0.45 [0.17, 0.89] |
| 15 | 272.08 [100.40, 397.24] | 5.58 [4.72, 6.93] | 61.89 [53.77, 74.95] | 6.69 [6.60, 10.95] | 32.11 [19.50, 38.33] | 0.29 [0.14, 0.66] |

`pricing_extra_columns` is inclusive of shortest-path work. Exclusive enrichment is batch minus shortest path; it is already included in pricing and must not be added again. Graph timing includes cache loading on hits.

| k | Remove all incidence: ceiling × | Maximum minutes saved | Hypothetical entire pricing 2×: overall × | Hypothetical entire pricing 5×: overall × |
|---:|---:|---:|---:|---:|
| 5 | 1.057 [1.043, 1.099] | 0.93 [0.35, 2.36] | 1.683 [1.581, 1.751] | 2.853 [2.425, 3.186] |
| 10 | 1.056 [1.049, 1.076] | 4.68 [2.27, 9.03] | 1.608 [1.491, 1.651] | 2.532 [2.112, 2.710] |
| 15 | 1.059 [1.050, 1.075] | 16.98 [4.87, 21.41] | 1.448 [1.368, 1.599] | 1.981 [1.755, 2.497] |

Incidence ceiling: `T / (T − I)`, minutes saved `I / 60`. This assumes **all** incidence time disappears, zero replacement cost, identical iteration count/columns/solver trajectory, and unchanged other costs. Removing only redundant work cannot be credited with this entire budget without measurement.

Pricing scenarios: `1 / (1 − p + p/s)` where `p` is each case’s inclusive pricing share and `s` is 2 or 5. Neither scenario asserts that these phase accelerations are feasible. Improvements to one subroutine cannot automatically receive the entire pricing budget.

Validation: all 98 records have nonnegative exclusive enrichment and no overcount when summing graph + inclusive pricing + incidence + master + fsync. This is arithmetic consistency, not independent proof of instrumentation boundaries. Unaccounted time is retained in the denominator and is not credited as savings. Selected-case residual shares:

| k | Unaccounted % |
|---:|---:|
| 5 | 0.56 [0.47, 0.70] |
| 10 | 0.39 [0.34, 0.51] |
| 15 | 0.28 [0.23, 0.38] |

Cold graph construction belongs to a separate cohort: seven cold and seven cache-hit overnight records are excluded here. The review’s cold `d00_g0` example is not a paired cache speedup. Scheduler queue time is outside these process wall times. A separate seven-hour capacity call cannot be extrapolated into savings for these baseline cases without matching input, invocation count, timer scope, and solver trajectory.

The source records retain input hashes, execution commit, objective, physics and certificate scope. This analysis makes no new feasibility, finite-pool MIP, full-model lower-bound, or GIRO-attainment claim. It does not run production code, submit jobs, or alter experiment status.

Reproduce from the repository root: `python3 outputs/algorithm_benchmarks_20260912/profile/quantify_profile.py`. Full per-case ratios and provenance hashes are in `summary.json`; source raw records are referenced rather than duplicated.

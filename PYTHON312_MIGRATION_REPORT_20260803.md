# Python 3.12 migration validation (2026-08-03)

## Result

The maintained code and pinned dependency set run successfully under Python
3.12.13 on the Apple Silicon development machine. The previous Python 3.10.6
environment was retained for comparison and recovery; it was not deleted.

## Correctness gates

| Gate | Python 3.10.6 baseline | Python 3.12.13 |
|---|---:|---:|
| Source/test compilation | pass | pass |
| Unit tests | 124/124 pass | 124/124 pass |
| `Practice_1bus.csv` SciPy preflight | not applicable after the minimum-version change | pass |
| Gurobi import and one-variable solve | not installed | pass with gurobipy/Gurobi 12.0.3 |

The Python 3.12 environment used NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3,
Matplotlib 3.10.9, and gurobipy 12.0.3.

## Controlled pricing comparison

Two 30-second `NO_CHEAT` pricing calls per interpreter used the same Mac, Git
worktree, dependency versions, `Practice_1bus.csv`, flat prices, 10,000-label
cap, `start_fair_bound` queue, diversified output, and resource dominance.

| Runtime | Trial | Labels expanded | Completed routes | Eligible negative incidences | Maximum returned trips |
|---|---:|---:|---:|---:|---:|
| Python 3.10.6 | 1 | 6,491 | 6,183 | 1,075 | 34 |
| Python 3.10.6 | 2 | 6,787 | 6,462 | 1,080 | 35 |
| Python 3.12.13 | 1 | 7,969 | 7,618 | 1,116 | 35 |
| Python 3.12.13 | 2 | 8,016 | 7,665 | 1,116 | 35 |

Python 3.12 expanded an average of 7,992.5 labels versus 6,639 under Python
3.10, approximately 20.4% more in the same pricing time. Both runtimes returned
150 columns, eliminated artificial coverage, and ended with LP route weight 2.
Neither call was exhaustive.

This is evidence of a useful interpreter throughput improvement, not evidence
that Python 3.12 fixes the heuristic pricer's search incompleteness. Long-run
comparisons must continue to record interpreter and dependency provenance.

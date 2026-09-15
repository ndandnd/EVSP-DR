# C3 at target 28: what changed between the two MIP searches

| Fleet-search measurement | Original | Separate longer-budget run |
|---|---:|---:|
| Buses found | 29 | 28 |
| Saved-pool fleet bound | 28.00000000000002 | 28.0 |
| Allocated fleet minutes | 30.0 | 180.0 |
| Gurobi solve minutes | 30.01 | 27.35 |
| Reported work units | 2053.56 | 3685.14 |
| Search nodes | 206 | 673 |
| Root LP seconds | 4.81 | 2.55 |
| Root LP simplex iterations | 14238 | 14238 |
| Root LP work units | 5.26 | 5.26 |
| CPU | Xeon E5-2665 | Xeon Gold 6348 |
| Threads | 8 | 8 |

The same ordered pool, source CG/journal, input hashes, native code and initializer count are verified. Both fleet models contain 606 rows, 166,052 binary variables and 3,943,168 nonzeros.

The repeat reports 1.79 times as much solver work in less elapsed time; reported work per optimizer second is 1.97 times higher. Its root LP performs the same iterations and reported work, in 2.55 rather than 4.81 seconds. These are observed throughput differences.

This supports machine performance as a plausible contributor, not a proved causal explanation. Hardware and the allocated time limit both differ. The larger limit did not simply let the repeated fleet search run past 30 minutes: it completed in about27.4minutes. Other execution conditions and the search path are not controlled.

Gurobi defines Work as a solver effort metric, with deterministic behavior guaranteed for the same model, hardware and parameter/attribute settings. Our differing hardware does not satisfy those conditions; these work values should not be presented as a hardware-independent benchmark or an exact common search path. [Official Gurobi Work documentation](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/model.html#work).

For a future causal comparison, match hardware and settings and record both elapsed time and solver work. Keep the current original and repeated treatments separate.

[Exact log lines, paths and SHA-256 hashes](c3_k28_search_work.json). Log fleet summaries are at line103 of the original and line107 of the repeat. The underlying logs remain on Unicorn; this report does not duplicate them.

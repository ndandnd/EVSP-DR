# 03:30 EDT results — 11 September 2026

| Warm chain | Target | Integer fleet | CG time | MIP time |
|---|---:|---:|---:|---:|
| 6 | 9 | 9 | 88.54 min | 22.68 min |
| 4 | 7 | 7 | 113.59 min | 29.25 s |
| 1 | 5 | 5 | 89.74 min | 3.47 s |

Both MIP stages reached optimality within each saved column pool. Selected routes passed individual physical replay. Chains 6 and 4 each overcover three trips; chain 1 overcovers none. Duplicate removal and shared station capacity remain separate validation requirements. These are not full-model integer optimality claims.

The default-partition trial remains at 23 completed attempts with zero recorded preemptions. No new execution failure or broken dependency required recovery. Exact sources, hashes and proof details are in [evidence.json](evidence.json).

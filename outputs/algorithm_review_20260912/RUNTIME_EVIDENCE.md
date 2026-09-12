# Runtime evidence

Frozen collector snapshot: `20260912T052658Z.json`, SHA-256 `23d9fb05744b8a6f242c46785309cc1cc004ec4ab29c89f802e6fac638667c84`.

pricing_extra_columns starts before shortestpath, so it measures the entire batch; exclusive enrichment subtracts pricing_shortest_path. Never sum both raw counters. network_build includes cache read on a hit.

Fresh covering, singleton initialization, cached event graphs. Median seconds across the certified cases at each k; objective/physics remain as recorded in each source.

| k | Cases | Wall | Graph load | Pricing batch | Shortest path (inside batch) | Enrichment only | Incidence | Master | I/O fsync |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6 | 84.93 | 0.49 | 81.88 | 15.53 | 53.04 | 0.67 | 1.19 | 0.25 |
| 3 | 6 | 142.11 | 1.10 | 135.47 | 49.78 | 104.23 | 4.03 | 6.47 | 0.40 |
| 4 | 6 | 404.27 | 3.35 | 365.86 | 133.56 | 247.68 | 13.28 | 22.81 | 0.75 |
| 5 | 6 | 1042.34 | 3.57 | 848.05 | 421.68 | 352.18 | 55.88 | 120.49 | 1.49 |
| 6 | 6 | 1412.65 | 13.06 | 1074.65 | 697.52 | 401.23 | 79.59 | 221.89 | 1.63 |
| 7 | 6 | 1651.21 | 32.18 | 1278.68 | 909.74 | 503.74 | 88.82 | 240.79 | 1.43 |
| 8 | 6 | 2621.28 | 9.07 | 2002.95 | 1584.04 | 535.48 | 142.34 | 445.65 | 2.00 |
| 9 | 6 | 3084.92 | 39.78 | 2329.61 | 1821.26 | 452.15 | 173.01 | 520.17 | 2.25 |
| 10 | 6 | 5322.93 | 23.88 | 3910.25 | 3074.36 | 657.37 | 280.50 | 956.13 | 3.23 |
| 11 | 6 | 6494.15 | 104.81 | 4633.50 | 3803.40 | 753.19 | 347.25 | 1380.22 | 3.93 |
| 12 | 6 | 7847.32 | 92.92 | 5361.86 | 4582.11 | 814.06 | 473.96 | 1992.97 | 3.62 |
| 13 | 6 | 9090.20 | 72.73 | 5423.24 | 4289.40 | 938.96 | 610.69 | 2806.45 | 4.78 |
| 14 | 6 | 9238.40 | 75.41 | 5504.86 | 4690.61 | 842.99 | 594.44 | 3020.09 | 3.35 |
| 15 | 6 | 16324.94 | 37.75 | 9595.16 | 8413.25 | 1181.90 | 1019.06 | 5588.27 | 4.42 |

Medians of individual counters need not add to the median total. Scheduler waiting and cold graph construction are excluded from these cached-run wall times. Raw path, code/input provenance and case metrics are in `runtime_evidence.json`.

New overnight cold-build example:

`d00_g0`: wall 1662.77s; graph construction 1311.19s; pricing batch 331.91s; shortest path within batch 140.58s; LP 12.50s. This is a different cohort, illustrating cold-build cost rather than a paired speed comparison.

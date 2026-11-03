# What changed when the battery grid doubled from 2.5 to 5 kWh?

Three previously registered eight-bus chains used the same solver, budgets, station powers and minute-level plug constraints. Each run built its graph and route pool from scratch. The finer-grid controls were the authenticated October runs. Every selected schedule below was replayed, and all six weighted event LPs satisfied the registered pricing check. The LP fleet weight was eight throughout.

| Chain | Pricing graph, million arcs (2.5 → 5 kWh) | Graph build, min | Root CG, min | Buses in final cover | Extra trip services |
|---|---:|---:|---:|---:|---:|
| C2 | 71.33 → 18.03 | 132.8 → 33.2 | 141.1 → 79.6 | 9 → 8 | 44 → 19 |
| C3 | 55.49 → 14.03 | 107.1 → 22.1 | 91.0 → 43.3 | 8 → 8 | 5 → 7 |
| C6 | 32.00 → 8.06 | 28.0 → 6.4 | 91.1 → 45.4 | 8 → 8 | 3 → 0 |

A positive extra-service count means that at least one passenger trip is assigned to multiple buses. Zero extra services, independent replay and eight simultaneous passenger trips suffice to establish the modeled fleet minimum on an eight-bus selection.

| Chain | Weighted LP lower bound less the 800,000 fleet component (2.5 → 5 kWh) | Fleet MIP, seconds | Charging MIP, seconds | Total allocation, hours |
|---|---:|---:|---:|---:|
| C2 | 236.677 → 270.066 | 1800.3 → 1581.5 | 1799.3 → 2018.4 | 7.07 → 4.39 |
| C3 | 212.819 → 236.886 | 25.1 → 85.1 | 241.8 → 743.6 | 4.88 → 2.83 |
| C6 | 295.169 → 322.257 | 18.7 → 6.2 | 23.8 → 8.4 | 3.50 → 1.94 |

The cost figures in the second table are the nonfleet part of the certified weighted objective on each discrete graph. The five-unit charge-start fee is included. No optimistic continuous-model lower bound was run. The aggregate gives separate finite-pool MIP bounds, grid and continuous replay costs, CPU model and batch peak RSS. Fleet and repeated-service differences prevent a matched-dispatch cost comparison in these pairs. The two sets of runs used different hardware, so timings describe these executions.
The 5 kWh graph used 74.7–74.8% fewer arcs. Its root CG took 43.6–52.5% less time, and peak job memory fell 63.4–69.3%. Only coarse C6 furnished an exactly-once dispatch in this panel. C2's coarse charging MIP stopped at 317.504 with a finite-pool bound of 294.194 (7.34% gap). C3's charging-stage solve took longer on the coarse grid, so shrinking the graph does not guarantee a shorter final cost search.

[All six solver and replay records](aggregate.json) · [Machine-readable pairs](paired.csv) · [Validation and hardware details](README.md) · [Original registration](../registration/manifest.json)

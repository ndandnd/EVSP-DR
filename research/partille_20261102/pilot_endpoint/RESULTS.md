# Fresh C2 pilot and decision on larger experiments

The fresh integer-directed pricing pilot and the fresh simple dive both covered all 173 passenger trips with nine buses. Independent replay found 21 extra trip services in the repair arm and 11 in the simple arm. Each respected the minute-level plug limits. Neither produced the registered endpoint of eight buses with every passenger trip served exactly once. The known-duty diagnostic independently attained eight, as did an earlier saved-root trip-recombination study. Those witnesses establish feasibility under this model and were kept outside the blind arms.

| Final result on C2 | Integer-directed pricing | Simple dive |
|---|---:|---:|
| Buses in selected cover | 9 | 9 |
| Services beyond the required one per trip | 21 | 11 |
| Overloaded charger-minutes | 0 | 0 |
| Fleet-stage bound in the final column pool | 8 | 8 |
| Root column generation | Time limit, no pricing certificate | Certified weighted LP 800236.67706 |
| Columns at root | 26,085 | 28,253 |
| Final columns / selected columns added after root | 26,220 / 3 | 41,340 / 6 |
| Elapsed time from Slurm | 8 h 32 min | 6 h 42 min |

Integer-directed pricing made 89 pricing calls, added 135 distinct columns and used three of them in the final nine-bus selection. Both finite-pool fleet searches stopped at their time limit with a one-bus gap. This result cannot establish that nine buses are necessary in either final pool. The arms used identical nominal phase budgets and ran on different CPU models. The first 26,085 column records have different hashes between the two arms. The blind run stopped with a negative reduced cost still present (−0.02926), so its last restricted LP objective of 800236.67824 is uncertified. Consequently these end-to-end outcomes do not isolate the effect of switching the integer heuristic on an identical root.

The eight-bus and ten-bus release checks both recorded a scientific hold. Ten additional fresh k8 runs, twelve fresh k10 runs and the six k8 plus six k10 sparse-prefix controls were cancelled without starting. They have no observed outcomes or prefix compute. The successful fixed-duty diagnostic did not substitute for a blind eight-bus dispatch. No rerun or release is called for under the registered design.

[Aggregate, solver status and source hashes](pilot_endpoint.json) · [Audit and reproduction](README.md) · [Original release rule](../gate.py)

# LP-support decomposition campaign status

Recorded 2026-09-14T18:48:29.235319+00:00.

The frozen campaign manifest is `dd154f49757657e9d1a309f38a31177bf052a46c9d4363a78da1992fe2cd8630` and is matched to first-selector manifest `69cf4c1f7c3795ce17ce62adb4bef2dd7d34194a5b95a86684e4920ec27af4c0`. All nine support-preserving partition pools completed. Each contains 2,048 columns, preserves the source integer witness and every positive final child-RMP trip set, and was built without constructing the 750-trip parent graph.

The source audit found 3,413 positive child-RMP routes. The first selector retained 226 and omitted 3,187 across all 36 components. Integer witness plus LP support fits below the 512-column component cap in every real component; the maximum required count is 193. A synthetic forced-over-cap test also passed and retained all mandatory routes.

Native fixture job 190363 accepted all 2,048 pool columns, rejected and repaired none, replayed selected routes over all 750 parent trips, and proved the 34-bus fleet optimum. Its short charging-cost stage ended at its diagnostic time limit. Gurobi 12.0.3 used license 2674428.

Production jobs 190402–190447 were admitted as 46 independent allocations: nine same-partition controls, 36 pairs, and one all-nine union. They use 4 CPUs, 24 GB, the default partition, and exclude `scaglione-compute-01`. Standard cases have 2.5-hour allocations around 2-hour solves; all-nine has a 4.5-hour allocation around a 4-hour solve. Requeue is disabled.

These are finite-pool treatment comparisons on one 32-duty, 750-trip parent instance. The preserved child-RMP weights are feasible witnesses for the restricted child pools. They do not establish a parent CG certificate, parent LP bound, or independence across 46 instances.

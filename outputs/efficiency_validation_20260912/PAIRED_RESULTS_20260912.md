# Completed paired observations — 12 September 2026

This dated result supersedes the submission-time pending-status statements in README.md. It describes completed solver treatments, not new MIP fleet proofs. Source: research register snapshot20260912T153215Z; source hash and extracted evidence are in paired_results_20260912T153215Z.json.

| Case | Reference s | Optimized s | Reduction |
|---|---:|---:|---:|
|w1_k08|2465.19|2066.01|16.2%|
|w1_k08_repeat|1866.38|1447.44|22.4%|
|w4_k11|3529.48|2947.25|16.5%|
|w6_k12|4029.19|3278.56|18.6%|
|d00_g0|1339.12|1417.37|-5.8%|
|d00_g1|3129.17|3128.52|0.0%|

All six pairs certified the conservative expanded-grid LP; this is not continuous-cost optimality or an integer fleet proof. Warm replay order can change the CG trajectory. Shared cache preparation is excluded from these treatment runtimes; report time from scratch separately. Four warm comparisons across three cases, with only one reverse-order repeat, do not establish a broad statistical speedup.

All three capacity pairs reached the10800s pricing deadline without a CG certificate. No demonstrated convergence benefit or harder-instance breakthrough. Both capacity arms include corrected station-power accounting, so their comparison is not a measurement of that correctness fix.

Charts: show paired runtime and phase timings with treatment order; preserve certified endpoints and separate censored capacity results. Electricity tariffs are unchanged. The separate w2k14 importer shutdown fix/retry is a reliability experiment and is not part of these speedup numbers.

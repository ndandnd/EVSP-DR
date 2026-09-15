# Research results — 14 September, 23:14 EDT

**All nine longer MIP searches recovered their targets from unchanged saved pools.** Each fleet minimum is proved within its pool, and every selected route passes individual replay. All charging searches reached their time limits; their cost optima remain unproved. Total MIP time was about 210 minutes per run.

| Case | Original buses | Rerun buses | Minutes to fleet proof |
|---|---|---|---|
| w1_k20 | 21 | 20 | 40.8 |
| w1_k22 | 23 | 22 | 33.1 |
| w2_k23 | 25 | 23 | 7.5 |
| w2_k24 | 25 | 24 | 134.0 |
| w3_k25 | 26 | 25 | 52.6 |
| w4_k21 | 22 | 21 | 19.1 |
| w4_k24 | 25 | 24 | 111.0 |
| w4_k25 | 26 | 25 | 68.1 |
| w6_k25 | 26 | 25 | 36.8 |

The original and rerun pool hashes, solver revision, non-time settings and initializer summaries match. The fleet allowance increased from 30 minutes to three hours, and each rerun started a new search tree. Two proofs finished within 30 minutes, so the extra allowance alone does not explain every recovery; execution timing was not controlled. The supported conclusion is that these pools contain target-sized solutions. Global fleet optimality, removal of duplicate coverage and shared charger capacity remain separate. This baseline omits shared capacity and terminal energy requirements.

[Result and source table](remaining_gap_results.csv) · [Original publication/attempt equivalence](original_publication_binding.json) · [Matched settings audit](matched_mip_settings_validation.json). Original publication files and attempt files have different byte hashes but identical parsed JSON; both hashes were checked remotely.

**The reserve screen is complete: eight of ten tests recover one bus.** All ten CGs certified their weighted LP objectives; all fleet and charging-cost minima were proved within the resulting pools. Duty 13405 still needs two buses in both tested pools. Its fractional route weight is 1.090909; that is not a separately optimized fleet lower bound and does not alone prove global one-bus infeasibility.

| Duty | Treatment | Buses | CG minutes | Capacity enforced |
|---|---|---|---|---|
| 13405 | baseline | 2 | 6.0 | False |
| 13405 | parx60 | 2 | 8.6 | False |
| 13406 | baseline | 1 | 8.1 | False |
| 13406 | parx60 | 1 | 11.1 | False |
| 13407 | baseline | 1 | 15.3 | False |
| 13407 | parx60 | 1 | 20.4 | False |
| 13408 | baseline | 1 | 2.2 | False |
| 13408 | parx60 | 1 | 3.1 | False |
| 13408 | capacity | 1 | 66.4 | True |
| 13408 | combined | 1 | 77.0 | True |

All tests use 236.44 kWh batteries and a 35.466 kWh reserve. Baseline/PARX60 use 240/60 kW at PARX, respectively; other charging remains 240 kW. Capacity and combined enforce documented station counts, with PARX at 240 and 60 kW respectively. Charging power is constant, not the nonlinear GIRO curve. No assumed 65% terminal target is imposed. All ten selected schedules pass the station-count audit afterward; route feasibility follows from driver construction, without independent continuous replay.

On duty 13408, adding capacity rows raises CG time from 2.2 to 66.4 minutes at PARX 240 kW; combining capacity and PARX 60 takes 77.0 minutes. All four variants recover one bus and the same charging-related cost, 45.256. These one-bus tests do not establish performance under competition between buses. [Full reserve results and hashes](reserve_results.csv).

**One fixed-state reference pricing call took 11,490 seconds (3.19 hours).** Its associated RMP had 7,823 rows, 50 columns and 249 nonzeros; measured solve wall time was 0.024 seconds (solver-reported time 0.005 seconds). With two nonzero charger-capacity dual prices, exact pricing found a negative-cost route with reduced cost −799976.896. The raw dual hash used by pricing matches the frozen starting vector. This is a single pricing call, not CG convergence. The paired cached implementation has no endpoint, so a speed comparison is not yet available. [Source diagnostic, model sizes and dual hashes](pricing_calls.json).

**Compact starts:** all 36 CGs are now certified. The MIP count remains 28/36, all target matches; eight k15 MIPs are unpublished. The newly completed core CGs on chains 1, 2 and 4 took 168.4, 161.2 and 166.5 minutes. [Updated editable table and verification](../../overnight_evening_20260914/status_20260915T030644Z/README.md).

The first paired pool-addition diagnostic finished on C5 k8: both the positive-LP-weight additions and matched-count zero-LP-weight additions yield pools proved to require nine buses. This pair does not establish a winning addition strategy; twelve pairs remain unfinished. The nine earlier LP-support-only results are unchanged.

The original extension has 60 CG endpoints: 45 certificates and 15 time limits. New C1 k25 stops after 239.4 minutes with minimum reduced cost −0.020743; its MIP is unpublished. Continuation C2 k26 stops after 239.6 minutes with reduced cost −0.012524, also without a certificate. Those RMP objectives are not full-model lower bounds. Existing scheduled follow-ups continue.

The original 59 MIPs have 34 target matches and 25 misses. The earlier longer searches recovered 15; these nine additional recoveries leave C5 k25 unresolved among the completed original MIPs. Chain 1 k25 has no published MIP yet. The largest one-hour result remains C3 at 27 buses; longer-budget recoveries are kept separate.

**Queue:** 69 running / 44 true dependency waits, excluding 33 held historical tasks. No new confirmed preemption, execution failure or invalid dependency. The preemption study contains 903 attempt records. This monitor submitted, cancelled and requeued no jobs. Source snapshot: `20260915T030644Z`, SHA256 `3b4c8a55f2fefbbd622536e9a2bc7fc16af87757ec627064178941fd621a5fc2`.

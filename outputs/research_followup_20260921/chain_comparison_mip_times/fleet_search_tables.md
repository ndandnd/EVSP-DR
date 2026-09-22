# Fleet-search times and integer outcomes by chain

These use the same original one-hour, two-stage MIP campaign as the Chain 1 comparison. The fleet stage receives up to 30 minutes; unused time goes to charging optimization. Later repeats and integer-directed pricing are excluded.

Times below are measured fleet optimizer calls only. They exclude charging, CG, preparation and earlier sequential MIPs, and are not the time when the first target incumbent appeared. A listed time limit leaves fleet optimality unresolved. Every unmarked row ended with a finite-pool fleet proof. Original 240 kWh/240 kW, zero reserve, no shared capacity or terminal floor, flat tariff, start fee 5, set covering; historical code/hardware differ.

## Chain 1

| Target | Fresh fleet search | Sequential fleet search | Final buses: fresh / sequential |
|---:|---|---|---|
| 5 | 1.30 s | 1.16 s | 5 / 5 |
| 8 | 26.35 min | 27.15 s | 9 / 8 |
| 10 | 30 min — time limit | 152.29 s | 11 / 10 |
| 15 | 30 min — time limit | 105.75 s | 18 / 15 |

## Chain 2

| Target | Fresh fleet search | Sequential fleet search | Final buses: fresh / sequential |
|---:|---|---|---|
| 5 | 63.23 s | 5.26 s | 5 / 5 |
| 8 | 30 min — time limit | 8.27 s | 9 / 8 |
| 10 | 30 min — time limit | 12.63 s | 12 / 10 |
| 15 | 30 min — time limit | 405.96 s | 17 / 15 |

## Chain 3

| Target | Fresh fleet search | Sequential fleet search | Final buses: fresh / sequential |
|---:|---|---|---|
| 5 | 3.38 s | 2.16 s | 5 / 5 |
| 8 | 30 min — time limit | 0.94 s | 9 / 8 |
| 10 | 30 min — time limit | 3.20 s | 11 / 10 |
| 15 | 30 min — time limit | 223.71 s | 18 / 15 |

## Chain 4

| Target | Fresh fleet search | Sequential fleet search | Final buses: fresh / sequential |
|---:|---|---|---|
| 5 | 14.89 s | 0.95 s | 5 / 5 |
| 8 | 14.66 min | 1.94 s | 9 / 8 |
| 10 | 30 min — time limit | 2.45 s | 11 / 10 |
| 15 | 30 min — time limit | 178.60 s | 19 / 15 |

## Chain 5

| Target | Fresh fleet search | Sequential fleet search | Final buses: fresh / sequential |
|---:|---|---|---|
| 5 | 35.88 s | 3.02 s | 6 / 5 |
| 8 | 14.89 min | 2.13 s | 9 / 8 |
| 10 | 30 min — time limit | 8.02 s | 11 / 10 |
| 15 | 30 min — time limit | 18.93 s | 16 / 15 |

## Chain 6

| Target | Fresh fleet search | Sequential fleet search | Final buses: fresh / sequential |
|---:|---|---|---|
| 5 | 0.46 s | 0.42 s | 5 / 5 |
| 8 | 15.77 s | 5.36 s | 8 / 8 |
| 10 | 30 min — time limit | 1.35 s | 11 / 10 |
| 15 | 30 min — time limit | 443.97 s | 20 / 15 |

## Per-chain summary

Four targets per chain: k=5,8,10,15. Means and medians include time-limited observations and describe time spent, not time to optimality.

| Chain | Fresh mean | Sequential mean | Fresh median | Sequential median | Fresh targets matched | Sequential targets matched |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 21.59 min | 71.59 s | 28.18 min | 66.45 s | 1/4 | 4/4 |
| 2 | 22.77 min | 108.03 s | 30.00 min | 10.45 s | 1/4 | 4/4 |
| 3 | 22.52 min | 57.50 s | 30.00 min | 2.68 s | 1/4 | 4/4 |
| 4 | 18.73 min | 45.98 s | 22.33 min | 2.20 s | 1/4 | 4/4 |
| 5 | 18.87 min | 8.02 s | 22.44 min | 5.52 s | 0/4 | 4/4 |
| 6 | 15.07 min | 112.77 s | 15.13 min | 3.35 s | 2/4 | 4/4 |

## Combined summary for Chains 2–6

Twenty cases per method; one saved search per case.

| Metric | Fresh | Sequential |
|---|---:|---:|
| Mean fleet-search time spent | 19.59 min | 66.46 s |
| Median fleet-search time spent | 30.00 min | 4.23 s |
| Fleet-search time range | 0.46 s–30.01 min | 0.42 s–7.40 min |
| GIRO targets matched | 5/20 | 20/20 |
| Fleet optima proved within saved pools | 8/20 | 20/20 |
| Fleet stage stopped at time limit | 12/20 | 0/20 |
| Total fleet optimizer time over the 20 cases | 6.53 h | 22.15 min |

Fresh proves an above-target fleet in C4 k8 (nine), C5 k5 (six) and C5 k8 (nine). Those saved pools cannot match the target without changing the pool. The other twelve fresh misses remain unresolved at the fleet time limit. These statements do not establish full-model integer impossibility.

Sequential k≤10 fleet proofs in Chains 2–6 all finish within 12.64 seconds; k15 ranges from 18.93 to 443.97 seconds. These are final-MIP benefits after the extra cumulative sequential CG work. Total MIP time can remain close to one hour because charging optimization continues.

Sources: [48-row timings and source paths](mip_stage_times.csv), [exact derived summaries and checks](fleet_search_summary.json), [timing definitions and model scope](README.md). All 48 pinned source hashes and direct timer values were rechecked; an independent audit corroborated the 40 endpoints for Chains 2–6. No solver run, original data modification, Doc or Slides edit.

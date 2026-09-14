# Fresh strict-capacity k2 endpoint audit — 2026-09-14 16:59Z

Verified source: `/home/nc437/ladder-lite/strict_capacity_parallel_20260914`; manifest `bce15219e18ec2b3e2cb9fac6ff47223b1cded4d9eb8934510cc91cd14811b62`.

| case | CG stop | cert | iters | pool cols | RMP route weight | finite-pool fleet | MIP proof | capacity audit |
|---|---|---:|---:|---:|---:|---:|---|---|
| k2_flat_240r0_baseline | exact_nonnegative_reduced_cost | True | 121 | 143 | 2.0 | 2 | stage1=OPTIMAL; stage2=OPTIMAL | False |
| k2_flat_240r0_capacity | pricing_deadline | False | 4 | 27 | 3.0 | 3 | stage1=OPTIMAL; stage2=OPTIMAL | True |
| k2_flat_240r0_parx60 | exact_nonnegative_reduced_cost | True | 120 | 142 | 2.0 | 2 | stage1=OPTIMAL; stage2=OPTIMAL | False |
| k2_flat_240r0_combined | pricing_deadline | False | 4 | 27 | 3.0 | 3 | stage1=OPTIMAL; stage2=OPTIMAL | True |
| k2_flat_236p44r15_baseline | exact_nonnegative_reduced_cost | True | 115 | 137 | 2.0 | 2 | stage1=OPTIMAL; stage2=OPTIMAL | False |
| k2_flat_236p44r15_capacity | pricing_deadline | False | 3 | 26 | 7.0 | 12 | stage1=OPTIMAL; stage2=OPTIMAL | True |
| k2_flat_236p44r15_parx60 | exact_nonnegative_reduced_cost | True | 130 | 152 | 2.0 | 2 | stage1=OPTIMAL; stage2=OPTIMAL | False |
| k2_flat_236p44r15_combined | pricing_deadline | False | 3 | 26 | 7.0 | 12 | stage1=OPTIMAL; stage2=OPTIMAL | True |

All four capacity-enforced cells (`capacity` and `combined` at both battery/reserve settings) reached the 13,200-second pricing deadline. Their CG endpoints are **uncertified**: terminal exact minimum reduced cost is null. The saved pools were usable; their dedicated finite-pool MIPs completed both stages and passed the reported station-capacity audit. Those MIP proofs apply only to each saved pool.

The baseline and parx60 cells certified under 6,600-second CG limits. Because the capacity/combined cells received twice that CG budget, this is a feasibility and algorithm-boundary pilot rather than an isolated runtime causal comparison. The pilot MIP limits also differed (300 versus 600 seconds); use the separate uniform one-hour MIP follow-up for matched integer-search comparisons.

Artifacts:

- Compact verified audit: `strict_capacity_k2_audit_20260914T1659Z.json`
- Verifier output: `strict_capacity_fresh_collection_20260914T1659Z.json`

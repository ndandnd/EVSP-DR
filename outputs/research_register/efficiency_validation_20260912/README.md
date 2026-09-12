# Paired efficiency validation launched on 12 September

Status checked at 02:56 EDT: nine jobs submitted; five running and four warm jobs failed during cache preparation. No completed paired timing result or new MIP result is available.

Execution baseline `89c5ba3e8a66fd77ad8397e5ce63eae647a0a49a`; capacity `309d98d266ebaf6b7e99543a67f8f2be5736874a`. [Immutable launch evidence](https://github.com/ndandnd/EVSP-DR/tree/58edeb4b6274deb01023673e5f19153d555c25bf/outputs/efficiency_validation_20260912) includes manifest SHA-256 `27edc998921806c35889cb9e51ca2989e9544e5aa339c62395b848e8d622bbc9` and jobs SHA-256 `1f61a0ab24be8736a80c33801ce59fc5dc1a875bb3ed555fc17fc51198873734`.

| Case | Job | Arm order | CPUs / memory | Allocation | Observed status |
|---|---:|---|---|---|---|
| d00_g0 | 964194 | reference then optimized | 2 / 32G | 04:30:00 | Running |
| d00_g1 | 964195 | optimized then reference | 2 / 32G | 04:30:00 | Running |
| w1_k08 | 964196 | reference then optimized | 8 / 96G | 08:00:00 | Preparation failed |
| w4_k11 | 964197 | optimized then reference | 8 / 96G | 08:00:00 | Preparation failed |
| w6_k12 | 964200 | reference then optimized | 8 / 96G | 08:00:00 | Preparation failed |
| w1_k08_repeat | 964201 | optimized then reference | 8 / 96G | 08:00:00 | Preparation failed |
| cap_k1 | 964202 | reference then optimized | 1 / 24G | 06:30:00 | Running |
| cap_k2 | 964203 | optimized then reference | 1 / 24G | 06:30:00 | Running |
| cap_k1_combined_peak12 | 964204 | reference then optimized | 1 / 24G | 06:30:00 | Running |

The four warm attempts stopped after roughly three scheduler seconds with exit1:0. Their preparation process exited2 with exact error `--event-network-cache-only does not use --out`. This is a launcher argument error before graph building or CG, not a pricing failure, preemption, or scientific result. The implementation task owns a separate recovery root, immutable source/manifest and four new job IDs; no automatic requeue. The original five active jobs and four failed attempts remain retained.

The main hourly collector now embeds the campaign collector under `efficiency_validation`, schema `evsp-efficiency-collection-v2`. Initial collection raced allocation publication and showed no allocation records; the following snapshot retained all nine attempts and 18 arm placeholders. Placeholders are not completed runs. Collector schema and actual output were checked; scheduler accounting is separate. No new claim of speedup, fleet optimality, or physical feasibility is made.

## Storage dependencies

The two isolated clones borrow Git objects from `/home/nc437/ladder-lite/overnight_extension_20260912/code/.git/objects` and `/home/nc437/ladder-lite/capacity_speed_pilot_20260910_v2_7d38efd/code/.git/objects`. Their actual alternates files were read over SSH. Protect both source code repositories from deletion or garbage collection until the dependency is removed and verified.

## Retained manager evidence

`20260912T065512Z.json`, `20260912T065615Z.json`, `launch_sacct.txt`, and `warm_preparation_stderr.json` retain collection, accounting and exact failure evidence. Existing charts and completed research results remain unchanged.

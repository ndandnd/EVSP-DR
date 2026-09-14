# Parallel experiments — 14 September 2026

**Latest check, 11:33 EDT:** 47 jobs running, 55 true dependency waits. C4 now matches k23 and C6 matches k24; C1 k22 still needs 23 with an open bound 22. Noon-peak k1 tests recover one bus but both reach the 110-minute CG limit without certification. Four capacity-enforced k2 CGs remain active. [Current chain table](../cumulative_budget_20260913/status_20260914T152928Z/README.md), [charging results](../strict_capacity_parallel_20260914/status_20260914T152928Z/README.md). The launch snapshot below remains dated 09:14.

**54 EVSP–DR jobs were running at 09:14 EDT**, up from six at the initial check. Another 67 wait for actual predecessor data; 33 held historical tasks are unchanged. The initial default-partition reading showed more than 10,000 idle CPUs. [Live queue record](live_queue_check.json). All work below is submitted.

| Priority | Experiment | Work submitted | Scientific question |
|---|---|---:|---|
| 1 | Unresolved saved pools | 14 MIPs | Does more integer search recover a target already present in the columns? |
| 2 | Combine pools from different CG runs | 6 constructions, each followed by a MIP | Do different runs generate complementary routes that work better together? |
| 3 | Controlled charging constraints | 12 CG allocations, then 12 matched one-hour MIPs | What changes when charger capacity, depot power and battery reserve are varied separately? |
| 4 | Extend the existing six chains to k=26–28 | 18 graph preparations, then 18 CGs and 18 MIPs | Where does baseline scaling become difficult beyond k=25? |

## Verified progress behind this choice

At 08:53 EDT, C1 k21 also matches 21 buses in its original one-hour MIP, with a finite-pool fleet proof and individual replay. Its CG still has no pricing certificate. Largest individual original target matches by chain are now 21, 22, 24, 20, 21 and 22.

In the 08:32 EDT result collection, three additional reruns recover C1 k19, C3 k22 and C3 k23 using their unchanged saved pools. Each fleet is proved best in its pool and passes individual-route replay. Fleet proof took 22.3, 16.1 and 24.1 minutes: all within the original 30-minute fleet allowance. The larger allocated limit alone does not explain the earlier misses. Hardware and parallel search remain uncontrolled.

The original 47 extension MIPs match 27 targets. Longer-search reruns recover 12 of the remaining 20 misses. Eight targets remain unresolved: C1 k20; C2 k23–25; C3 k25; C4 k21–22; C6 k23. CG has 40 pricing certificates among 48 published endpoints; eight stopped at the time limit. MIP fleet proofs, full pricing certificates and charging-cost optimality remain separate.

The completed accumulated-budget comparison remains 6/24 fresh versus 24/24 inherited target matches. New MIP comparisons use existing columns; they do not add new CG computation or create pricing certificates.

## Execution and resource rules

All new CPU work uses the default partition and excludes scaglione-compute-01. Every independent case is eligible concurrently. An 18-task array therefore allows all 18; larger independent CG arrays retain the standing default of 50. Each later chain CG waits for its own graph and previous-k CG; its MIP waits only on its own CG. Held historical jobs and EVSPV2G work are untouched.

Graph budgets remain 12 hours, with a 12½-hour allocation; baseline CG has four hours and its MIP one hour. The 14 difficult-pool MIPs receive up to three hours for fleet search within 3½ hours total, matching earlier long-search diagnostics. Record actual time to target and proof, rather than interpreting the allocation as required solve time.

Strict charging cells allow 50, 110 or 220 minutes of CG, depending on the selected case. Their preliminary MIPs have unequal 5/10-minute budgets and are not the primary integer comparison. Each has a separate, matched 60-minute MIP on its unchanged saved pool, using the dedicated capacity-aware solver. Two k1 pairs have completed; ten CGs were still running at the queue check. Capacity arms receive 220 rather than 110 minutes of CG in the k2 pilot. The matched MIPs equalize only final integer-search time, not total CG effort; do not infer an isolated runtime effect of capacity. Reference-versus-cached-pricing pairs have matching CG settings. The 236.44-kWh/15% reserve treatment is a constant-power sensitivity, not the complete nonlinear GIRO vehicle model. No 65% terminal floor is inferred.

## Evidence

- [Chain k26–28 launch and job map](../chain_extension_20260914/README.md).
- [Saved-pool MIP design and immutable inputs](mip_design/DESIGN.md); jobs 188041–188054.
- [Union MIP launch receipt](union_design/launch_status.json); jobs 188099–188104 after completed builds 188093–188098.
- [Strict charging design](../strict_capacity_parallel_20260914/design_README.md); CG jobs 188123–188134.
- [Matched one-hour MIP manifest](../strict_capacity_mip1h_20260914/manifest.json); jobs 188143–188154, each waiting only on its own CG allocation.
- [Latest recovered targets and exact timings](../mip_repeatability_20260914/status_20260914T124947Z/README.md).
- [CG and original MIP results](../cumulative_budget_20260913/status_20260914T124947Z/README.md).

A union preparation command exited 255 without a traceback; the subsequent SSH check succeeded immediately. Its partial output is retained and is not a scientific result. All six union constructions subsequently completed on compute nodes, and all six MIPs are running; the cause of the interruption is unconfirmed. No existing job was cancelled or changed.

Dated launch receipts, source hashes, job IDs, dependencies and output locations are retained with each campaign. The collector and experiment register track the new roots. All 104 production tasks are submitted: 54 chain-extension tasks, 14 saved-pool MIPs, 6 union constructions plus 6 MIPs, and 12 strict allocations plus 12 matched MIPs. Validation jobs are recorded separately. Scheduler state changes after the dated check.

**Results through 00:16 EDT, 15 September:** [New chain matches, compact-start gap and paired pricing results](status_20260915T040735Z/README.md). The launch record below retains its original timestamp.

# Overnight additions — 14 September 2026

**Submitted and verified at 21:54 EDT: 89 EVSP–DR jobs running, 54 waiting for required inputs.** The three new campaigns add 69 independent allocations plus 24 own-CG MIPs. All ten reserve-screen jobs and all 24 larger CGs are running; nine of the 35 new pool MIPs have finished and 26 remain running. No array throttle, unsatisfiable dependency, or assignment to the reserved GPU node was observed. [Queue evidence](queue_verification.json).

The live queue at 21:36 EDT had 30 EVSP–DR allocations running and 32 waiting on real dependencies (22 MIPs and 10 CGs), excluding 33 held historical tasks. None reported an unsatisfiable dependency or an array concurrency limit. The default partition reported 8,820 idle CPUs in the earlier resource sample; scheduler availability changes. The sequential chains cannot be made independent without changing their initialization.

| Priority | New experiment | Independent jobs | Following jobs | Question |
|---|---|---:|---:|---|
| 1 | Compact inherited pools, six chains at k=20 and k=25 | 24 CGs | 24 MIPs, each after its own CG | Can a few hundred carefully retained routes replace the full inherited pool at larger sizes? |
| 2 | Controlled repair of 13 demonstrably insufficient small-instance pools | 26 MIPs | None | Do routes with positive weight in a richer final LP help more than the same number of other routes? |
| 2 | Integer solves using only the richer LP's positive-weight routes | 9 MIPs | None | Does that fractional solution's support already contain a good integer solution? |
| 3 | Stricter energy requirements on four individual duties | 10 CG/MIP allocations | None outside each allocation | Does the smaller usable battery prevent recovery before shared-charger contention matters? |

All 69 independent optimization allocations and 24 dependent MIPs are submitted. Native validation and data preparation passed before production launch. They do not wait for the active chains or introduce an artificial concurrency cap. Native preparation/smoke allocations are operational checks, not research outcomes.

## Why these experiments

The completed accumulated-time control already answered the immediate question about giving fresh CG the earlier chain's computation budget: all 24 fresh CGs certified, but their one-hour MIPs matched 6 targets versus 24 for the inherited pools. Historical CG revisions and machines varied, so this is a retrospective budget comparison. Repeating those identical controls would be less informative tonight.

The richer-seed campaign now matches all 24 k=8/k=10 cases across two arms and six chains. Thirteen earlier pools proved they could not match their targets despite virtually identical certified weighted LP objectives. This establishes an integer limitation of those saved pools. It does not establish that every unresolved MIP gap has the same cause.

The larger-size test uses completed k19 and k24 parents. Each child gets either the union of its parent's integer-selected routes and all positive-weight LP routes, or the same core filled to 512 distinct trip sets. New trips also receive singleton initialization. Both arms use the cache-producing a0e0 CG revision and the same input/cache/model/budget. Earlier smaller-seed experiments used e091; do not attribute differences across those cohorts solely to instance size or initialization. Parent computation is recorded, including the MIP used to obtain integer routes.

The pool-repair test freezes thirteen earlier proved misses and nine unique rich-core donor pools. For each miss it compares adding previously absent donor LP-positive routes with adding the same number of previously absent LP-zero routes selected by a fixed hash ranking. Neither selector uses the donor MIP's chosen routes. The donor LP solution must reconstruct correctly before it supplies a treatment. The nine LP-support-only MIPs are separate controls. This is a deliberately selected diagnostic panel, not a random estimate of success over all trip subsets. It tests information obtained by extra upstream computation; that cost is not free.

## Resources, budgets and proof scope

All jobs use default_partition and exclude scaglione-compute-01. CG requests 8 CPUs and 96 GiB for a four-hour native budget. MIPs request 8 CPUs and 24 GiB, with 12,600 seconds total and up to 10,800 seconds initially minimizing fleet, matching the seed diagnostic's budget. Stage two minimizes charging-related cost subject to fleet no greater than the first-stage validated incumbent. A timed incumbent is never described as proved. These are longer diagnostic MIPs, distinct from the one-hour main chain table.

The two pool/seed campaigns use the baseline covering model: 240 kWh, 240 kW, fee 5, no shared-station capacity or terminal-SOC floor. The ten reserve-screen cases are a separate physics cohort: 236.44 kWh battery, 35.466 kWh reserve, flat prices, and no inferred 65% terminal target. Eight compare baseline/PARX60 on duties 13405–13408; two test capacity/combined on previously tractable duty 13408. They use one CPU, 24 GiB, 13,200 seconds CG and 600 seconds MIP within a 4:15 allocation. This avoids repeating broad capacity runs on known slow duties. The model still uses constant charging rates, not the nonlinear GIRO vehicle curve.

Maintain true own-CG and previous-k dependencies. Preserve held historical and EVSP–V2G work. Every restart must retain its original attempt, input/source hashes and accounting. Scheduler success, CG certificate, pool fleet proof, individual-route replay and shared physical feasibility remain separate.

The workbook retains its 21:05–21:13 full-collection timestamp. The records below give actual new submissions and the nine early results collected at 21:54.

## Verified launch and early evidence

- [Larger compact pools: frozen cases, native checks and exact job IDs](../compact_large_seed_20260914/README.md).
- [Pool-repair MIPs: frozen cases, three-path native checks and exact job IDs](../lp_support_pool_diagnostic_20260914/README.md).
- [Reserve screen: model and implementation record](../reserve_feasibility_screen_20260914/README.md).
- [Nine completed support-only controls](FIRST_DIAGNOSTIC_RESULTS.md): all prove a fleet above target in the restricted pool, although each full donor pool matches target. Some zero-weight routes therefore matter for integer recovery. This is a result for these selected cases, not a general success-rate estimate.

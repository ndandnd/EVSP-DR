# Verified update — 14 September, 17:02 EDT

Two new completed pairs favor previous-k integer-route seeds. Each listed fleet minimum is proved within that treatment's saved pool:

| Chain / target | Integer-route seeds: buses | LP-weight seeds: buses | CG minutes: integer / LP |
|---|---:|---:|---:|
| C2 / 8 | 8 | 9 | 17.9 / 13.1 |
| C5 / 8 | 8 | 9 | 26.4 / 26.3 |
| C6 / 8 | 8 | 8 | 9.1 / 10.2 |
| C6 / 10 | 10 | 11 | 20.3 / 22.0 |

These four completed pairs have matching certified weighted LP objectives within numerical precision. The selected integer-route sequences cover more parent trips; this is a practical selection-method comparison, not an isolated test of integrality. Fourteen other pairs lack a complete pair of MIP results. Across all submitted seed cases, 33/36 CGs have certificates and 10 MIPs have results: six target matches and four proved pool limits above target. New C3 k8 LP-selected seeds require nine buses; its integer-route MIP retry remains running. Fleet proofs do not imply charging-cost proofs: the three new MIPs ended at the charging-stage time limit. [Exact seed results](README.md), [source-bound values](seed_results.csv).

## Decomposition pair searches completed

| Selection method | Individual pools completed | Pair unions completed | Pair fleets found | Pair fleet minima proved | Pairs improving their best component |
|---|---:|---:|---|---:|---:|
| Keep integer witness, then fill 512 routes per group | 9 | 36 | 34–36 | 36 | 0 |
| Also retain every positive-weight source LP route, same 512 limit | 9 | 36 | 34–36 | 1 | 0 |

All nine individual controls in each treatment have fleet proofs; fleets range 34–37. The second treatment's other 35 pair searches retain open gaps, with pool fleet bounds 33–34. **Every completed pairwise pool therefore excludes the 32-bus target.** This is a statement about those selected columns, not the full routing model. The two all-nine union searches remain running. All selected solutions pass individual route replay; shared charger capacity and terminal SOC floors are absent. Both treatments use the same 750-trip parent, so the 92 jobs are not independent datasets. [Every result, bound and source hash](decomposition_results.csv).

## Capacity-aware one-bus tests

Each cell shows reference / cached pricing. All seven completed MIPs find one bus, prove fleet and charging-related cost in their own pools, and pass the shared-capacity and duplicate-coverage checks.

| Duty | CG minutes | Completed CG iterations | CG certificate | Integer buses | Charging-related cost |
|---|---|---|---|---|---|
| 13405 | 220 / 220 | 18 / 16 | no / no | 1 / 1 | 94.720 / 94.720 |
| 13406 | 220 / 220 | 13 / 13 | no / no | 1 / 1 | 56.952 / 56.952 |
| 13407 | 220 / interrupted | 15 / no endpoint | no / none | 1 / no result | 81.584 / no result |
| 13408 | 2.93 / 1.75 | 33 / 33 | yes / yes | 1 / 1 | 36.536 / 36.536 |

The five 220-minute runs stopped during pricing without a convergence certificate. They nevertheless recovered one-bus schedules. The cached duty-13407 job 189169 was preempted after 8,410 seconds (2h20m10s), at 15:22:18 EDT. This check newly identified the interruption: its unchanged worker file still said running and returncode zero, but Slurm says PREEMPTED. No CG endpoint or MIP result exists; its partial pool remains preserved. No blind requeue or new retry was launched. This CG interruption is kept outside the MIP reliability cohort. [Scheduler and file evidence](capacity_preemption_189169.json), [complete capacity values](capacity_results.csv).

These tests use constant-rate 240-kWh/240-kW physics, zero reserve, flat prices and no terminal floor. Individual exact-event route feasibility is by construction; shared capacity is checked separately. One-bus success does not establish scalability under competition among several buses.

## Queue and reporting

The collection reports **52 running jobs: 7 CGs, 29 MIPs and 16 graph builds**. There are 43 solver dependencies, 16 conditional graph checks and 33 held historical tasks. No invalid dependency or new MIP preemption appeared. Chain 3 has begun CG at k=26 after graph preparation; two new graph builds have finished. No production settings, jobs, held tasks, V2G experiments or Slides were changed by this check.

The existing chain and cumulative-budget outcomes are unchanged from the previous collection. The register and workbook contain 2,937 artifact/stage rows across 63 source groups, retaining all six supplements. 312 core endpoints and 147 seed/decomposition/capacity endpoints were compared to source evidence; workbook formula checks passed. Current Google Doc tables were updated in place; reference content and figure tabs were preserved.

Source collection: 20260914T205702Z, completed at 17:02:39 EDT after 337 seconds; SHA-256 ca8f69e173d71679cb553881af165b0c758af572f79b038856b171d050ca57c2. Scheduler events arriving during collection remain distinct from published outputs.

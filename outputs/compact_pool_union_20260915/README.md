# Compact generated-pool complementarity — launched 15 September 2026

**Launched at 06:37 EDT; all 12 production MIPs running at 06:40.** Eight constructions completed. Jobs227315–227337 (exact mapping in `case_jobs.json`) use default_partition and exclude scaglione-compute-01. The full queue has30running/29true dependency waits, excluding held537227. Native fixture compatibility checks and resource/dependency checks passed; the three30-second fixture results are not production evidence. All12production MIPs are registered for per-restart preemption tracking; root has wired the collector and normalizer and tested all3native fixture publications. Scientific results will enter the next full register collection; this launch has its own timestamp.

The diagnostic asks whether independently generated compact core and core512 column pools complement each other for integer fleet. Each union uses every saved source record as input and applies exactly the native incidence deduplication rule: retain the cheapest cost by more than 1e-9, preserve the first record on ties, keep the complete winning raw record. There is no new column generation, no GIRO route import and no new pricing certificate.

| Pair | Core buses / pool bound | Core512 buses / pool bound | New control | Both donor CG hours | Shared upstream CG hours |
|---|---:|---:|---|---:|---:|
| c1_k15 | 16 / 16 | 16 / 16 | existing exclusions | 5.27 | 13.27 |
| c1_k20 | 21 / 21 | 21 / 21 | existing exclusions | 7.97 | 25.95 |
| c2_k20 | 22 / 21 | 21 / 20 | core512 | 6.84 | 22.18 |
| c2_k25 | 31 / 25 | 26 / 25 | core512 | 7.97 | 30.74 |
| c3_k20 | 24 / 20 | 23 / 20 | core512 | 3.56 | 20.09 |
| c4_k25 | 26 / 26 | 26 / 26 | existing exclusions | 7.97 | 37.66 |
| c5_k20 | 21 / 21 | 21 / 20 | core512 | 7.97 | 30.60 |
| c5_k25 | 26 / 26 | 26 / 26 | existing exclusions | 7.97 | 49.30 |

All 16 source MIPs have successful physical replay, zero rejected/repaired columns, identical source/result/journal bindings, frozen 871d057 MIP execution, 12600/10800-second budgets and 8 threads. Every source pair has the same input/tariff/reference/deadhead hashes and CG code within the pair. C1k15 uses e091; larger pairs use a0e0. These are not one uniform cross-size source-code cohort. Eight finite-pool fleet proofs establish the exclusions for C1k15,C1k20,C4k25,C5k25. C2k20 core also excludes 20 through its bound 21 although its incumbent 22 is not pool-optimal.

The four unchanged-pool controls all use core512. C5k20 is a fleet tie; core512 is selected because core already excludes 20 while core512 still permits20. C2k25 and C3k20 both donor pools still permit target; core512 supplies the better current fleet. No new control is needed where both donor pools already exclude target.

## Initialization decision and independent incumbent evidence

The initially proposed supplied-incumbent design is incompatible with the unchanged frozen solver. In native871d057, merge_validated_partition_start appends every validated route after the base pool, including existing incidences (reference/native_mip.py:1358–1381). pool_columns_reused is 0, every start variable is appended, and no subsequent deduplication is performed. Even a fully present witness changes column count and ordered pool hash. This was discovered in source inspection before any submission; it is a superseded preparation design, not an execution failure.

The parent explicitly revised the design to use the unchanged native greedy initialization algorithm in both union and control. No --initial-partition-routes, --verified-expanded-initial-partition or --extra-routes flag is allowed. The realized greedy start can differ with the pool and is recorded from native mip_start. A donor witness is not injected and is never relabeled as the new solver's incumbent. A new solver result may be worse than the independently known donor bound.

The constructor verifies each selected donor route against its exact original native winning record hash, expanded-grid cost and full instance coverage. For the union, every donor incidence must exist at no higher cost. A cheaper/tied already-admitted union route can replace the donor record of the same incidence; report exact retained routes and these substitutions separately. This establishes an independent feasible fleet/cost upper bound under covering. It does not assert that both donors' selected records survive cheapest-incidence dedup byte-for-byte, and it makes no claim of shared-charger validation.

## Execution and publication gates

Production plan: 8 independent construction jobs, then 8 union MIPs and 4 core512 controls. Each MIP depends only on its own pair's construction; a control also uses that pair's construction membership audit but reads its exact original source status/journal. Construction requests 2 CPU, 8 GB, 1 hour, 3300-second watchdog. MIPs request 8 CPU, 24 GB, 4.5 hours, 12600 total / 10800 fleet seconds, 15300-second watchdog. All use default_partition, exclude scaglione-compute-01, and have no arbitrary throttle. Production requeues receive unique job/restart attempt directories; old attempts remain intact and completed cases are not reoptimized. Submission intent is durably recorded before sbatch and ambiguous submission requires reconciliation.

Source statuses, journals, MIPs, completion markers, model/input files and execution scripts are hash-gated. Native code must remain a clean detached871d057 checkout. Each native MIP uses 240 kWh / 240 kW, event 2.5 SOC / 5 minute grid, cover, no reserve/shared capacity/terminal floor, 100000 bus coefficient and 5 charge-start fee. Stage 2 uses at most the validated fleet incumbent. A 2001-variable license solve checks the full Gurobi license inside every MIP allocation.

Publication requires successful native replay with zero rejection/repair, no added columns, identical base/augmented counts and ordered hashes, unchanged expanded-grid cost semantics, native greedy start, correct source hashes, exact budgets and clean native execution at completion. The control additionally must match its donor's exact native ordered pool hash. Construction and results publish through atomic links and completion JSON only after all gates pass. Scheduler completion, pool fleet proof, pool-bound target exclusion, physical replay and target attainment remain distinct. The independent donor upper bound remains separate from the current result incumbent.

All attempts and accounting stay in this campaign's private artifacts. No shared collector, experiment register, preemption registry, Google Doc, Slides, historical held jobs or V2G files are edited by this preparation agent. Integration belongs to the parent after review.

## Computation accounting and limits

prior_cg_accounting.json records both current-k donor CGs plus their shared inherited CG ancestry, counting upstream only once per pair. Across all eight pairs it deduplicates 121 verified status artifacts, totaling 219.27 native CG hours. This is source-generation expenditure, not added union computation. It excludes unreferenced failed attempts; scheduler CPU and graph-construction costs remain separately preserved in source_audit.json. It is not a same-computation performance claim.

The source audit rehashes all 16 status/MIP publications and completion-marker bindings, actual source/input/config files, and validates prior native journal-hash bindings. Full source journals are rehashed before and after compute-node construction and again by native MIP. Duplicate discovery inspects existing union/compact manifests and receipts, the current user queue and all immediate campaign manifests for exact donor status hashes. The prior parallel_pool_unions campaign uses different k8/k10 original/c200/complementary sources.

## Reviewable files

- pool_logic.py: disk-backed native union, set/cost hashes, exact donor membership and dominance.
- worker.py: immutable attempt execution, private records, strict native result gates.
- common.py: established hash/atomic-write/timeout helpers extracted from the earlier union worker; shared-registry code excluded.
- campaign.py: prepare, bounded fixture and explicit parent-authorized production submit; production refuses missing/stale native_validation.json.
- test_design.py: 5 meaningful gates tested locally and on Unicorn.
- source_audit.json,prior_cg_accounting.json,manifest.json and native_fixture/: immutable selection, prior cost and launch receipts.
- verify_fixture.py: independent output/license/effective-resource validation and private scheduler accounting.

Recorded production command after parent review:

```sh
/home/nc437/evsp_env/bin/python /home/nc437/ladder-lite/compact_pool_union_20260915/campaign.py submit --production-authorized
```

Parent reviewed the constructor, worker and frozen plan and confirmed its native incidence/cost deduplication policy on the no-capacity baseline. Native validation passed; the production manifest and solver tooling remain immutable.

Native validation completed successfully for jobs 227214–227218. C1 k15 union contains 100,861 columns; C2 k20 union contains 105,042 and its unchanged core512 control contains 54,090. Every native result has zero added/rejected/repaired columns, identical base/augmented hashes and successful physical replay, with the unchanged control matching its donor native ordered pool hash exactly. Full-size Gurobi license, 30/20-second fixture budgets, clean native execution and effective resources/dependencies all passed. Constructor wall/RAM and actual greedy start details are retained in native_validation.json. These short solves establish compatibility only.

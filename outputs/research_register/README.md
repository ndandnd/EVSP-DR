# EVSP–DR experiment register

**Current results, 14 September 10:35 EDT:** Chain 5 now matches **k=22 with 22 buses**, with a fleet proof in its saved pool and individual-route replay; its CG remains time-limited. Largest individual original matches across chains 1–6: **21, 22, 24, 20, 22, 22**. New pricing certificates: C1 k22 (149.7 minutes), C4 k23 (219.6), C6 k24 (171.6). [Current chain table and source records](../cumulative_budget_20260913/status_20260914T142853Z/README.md). Extension totals: 51 CG endpoints, 43 certificates and eight time limits; 48 original one-hour MIPs, 28 target matches and 20 misses. Twelve misses are recovered by separate longer searches, leaving eight unresolved targets.

**Charging controls:** Four settings on the same E1-short k2 input all recover two buses, including the 236.44-kWh/15% reserve and PARX-60-kW treatment. All four have CG certificates and matched-MIP fleet/charging proofs in their saved pools. **All four exceed the documented one-charger capacity at 2190L**, using two simultaneous connections; shared capacity was deliberately disabled in these controls. Capacity-enforced k2 tests remain running. [Editable settings/results and proof limits](../strict_capacity_parallel_20260914/status_20260914T142853Z/README.md). The k1 flat-price capacity controls both match one bus and pass shared-capacity checks. Dedicated-solver route feasibility is by construction, not independent continuous replay.

**Queue at collection: 52 running, 57 genuine dependency waits; 33 held historical tasks unchanged.** No new execution failure, confirmed preemption or invalid dependency. Six strict CG/MIP pairs are complete and six CGs remain active. All 104 previously authorized production tasks remain submitted; no new jobs were added in this check. Earlier launch/result notices below retain their original dates.

**New parallel work, 14 September:** [Experiment plan, questions and launch records](../parallel_followup_20260914/README.md). **54 jobs running at 09:14 EDT**, up from six. All 104 production tasks are submitted: chain extensions to k=26–28, fourteen unresolved-pool MIPs, six combined-pool MIPs, twelve charging-constraint CG tests and twelve matched one-hour MIP follow-ups. Genuine previous-k and own-CG dependencies remain intact; all independent cases are eligible concurrently.

**Verified results collected at 08:53 EDT:** C1 k21 now matches 21 buses in its original one-hour MIP, with a fleet proof in the saved pool and individual-route replay; its CG remains time-limited. Largest individual original matches by chain are 21, 22, 24, 20, 21 and 22. Separate reruns recovered C1 k19, C3 k22 and C3 k23, with fleet proofs in 22.3, 16.1 and 24.1 minutes. Extra allocated time alone does not explain their earlier misses. [Results and provenance](../mip_repeatability_20260914/status_20260914T124947Z/README.md). The 47 original extension MIPs give 27 target matches; longer-search reruns recover twelve other gaps, leaving eight unresolved targets. CG has 40 certificates among 48 endpoints. These baseline results omit shared charger capacity and terminal SOC requirements.

The dated 07:31 status below is superseded by the 08:53 collection and new launch records.

**Current results, 14 September 07:31 EDT:** [Chain table and accumulated-time comparison](../cumulative_budget_20260913/status_20260914T112740Z/README.md); [column-selection comparisons](../overnight_diagnostics_20260914/status_20260914T112740Z/README.md); [completed one-hour MIP repeats](../mip_repeatability_20260914/status_20260914T112740Z/README.md).

**All 27 one-hour repeats finished: 21 target matches, with agreement across all three repetitions of each case.** Seven selected inherited pools reach their target every time. C3 k19 and k21 use one extra bus in all three repeats, with open bounds at target; the earlier longer MIPs recovered both from the same pools. These are incomplete integer searches, not evidence of missing target routes. Ordered pools, inputs and initializer summaries match. Hardware and parallel-search timing remain uncontrolled; the repeats do not explain the original-versus-rerun difference for the seven successes.

**Chain 5’s largest match remains 21 buses**, with a finite-pool fleet proof and individual-route replay. Its CG still has no pricing certificate. Largest individual original matches by chain: **18, 22, 24, 20, 21, 22**. C2 k24 remains at 25 buses with bound 24, unproved. The 43 original MIPs give 26 target matches and 17 gaps. Separate longer MIPs recovered nine gaps, leaving eight unmatched targets across collected outcomes. Of 46 CG endpoints, 40 are certified and six reached the time limit. New certificates: C2 k25 in 102.8 minutes, C4 k22 in 140.0 minutes and C6 k23 in 213.8 minutes. Their MIPs are running; these are not new integer target matches.

New C1 k15 complementary selection finds 18 buses, the same as the original selection, with bound 15 and no fleet proof. Individual-route replay passes. The 200-column CG runs for C2, C4 and C6 at k15 newly certify after 435.0, 434.3 and 431.1 minutes; their MIPs are running. Diagnostics contain 37 CG certificates and 56 MIPs. All 18 selected inputs now have at least one treatment MIP. The completed cumulative-budget comparison remains 6/24 fresh versus 24/24 warm target matches; keep populations, budgets and proof scopes separate.

**Queue: 15 jobs running and 25 true dependency waits.** The 27 short repeats completed; three longer repeatability-campaign MIPs remain running. No new execution failure, confirmed preemption or invalid dependency. No new jobs submitted by this check. Baseline physics omit shared capacity and a terminal-SOC floor. Figures and Slides are preserved.

**Earlier results, 14 September 01:26 EDT (superseded):** [Chain table and cumulative-time comparison](../cumulative_budget_20260913/status_20260914T052404Z/README.md). Largest individual integer matches across chains 1–6: **18, 22, 18, 20, 18, 20**. New C2 k22 and C4 k20 matches have finite-pool fleet proofs and individual-route replay. Extension totals: 34 CG endpoints (31 certified), 33 MIPs (21 target matches, 12 open one-bus gaps). Original C1/C4/C5 k19 CG endpoints remain time-limited. A separate C4 k19 continuation now certifies after 267.0 cumulative minutes; it does not overwrite the original result.

**First column-selection comparison:** [C5 k5 results](../overnight_diagnostics_20260914/status_20260914T052404Z/README.md). Original 30-column selection and 200-column selection both yield pools proved to require 6 buses; selecting 30 complementary columns produces a pool supporting 5. All three have the same certified weighted LP objective to numerical precision. This selected case establishes that column composition matters; it does not establish a general success rate. These remain baseline models without shared charger capacity or a terminal-SOC floor.

**Live queue, 01:35 EDT:** 66 running EVSP–DR jobs (57 diagnostics, 9 chain extension); 65 pending on actual predecessor data. No array throttle or invalid dependency appeared. The 99-job overnight batch is already fully submitted; no duplicate work was launched. [Dated queue observation](../overnight_diagnostics_20260914/status_20260914T052404Z/live_queue_check.json).

**Overnight diagnostics fully submitted, 14 September00:33EDT:** [Experiment table, exact settings and jobs](../overnight_diagnostics_20260914/README.md). All99 jobs are submitted:61 independent jobs and38 MIPs dependent only on their own CG. The first59 independent jobs were running at00:09EDT. Native checkpoint check157899 passed and submitted the two k19 continuations and their MIPs automatically. The original six previous-k chains remain intact.

**Earlier results, 14 September00:25EDT (superseded):** [Current chain table and completed cumulative-budget comparison](../cumulative_budget_20260913/status_20260914T042303Z/README.md). New integer target matches: C2k21=21, C4k19=19, C6k20=20, each with finite-pool fleet proof and individual-route replay. Highest individual matches across chains1–6 are18,21,18,19,18,20; earlier gaps remain open. Extension totals:29MIPs,19target matches,10unresolved one-bus gaps;30pricing certificates among33CGendpoints. C1/C4/C5k19 hit CG time limits, so their restricted-master objectives are not certified full-model lower bounds. C4k19 nevertheless matches the integer target. Baseline physics still omit shared capacity and a terminal-SOC floor. The completed fresh-versus-warm comparison remains6/24 versus24/24 target matches. No new production execution failure, confirmed preemption or invalid dependency appeared; failed validation allocation155072 is a separately documented smoke check.

**Chain expansion, 13 September:** [Six chains continuing from k=16 to k=25](../chain_extension_20260913/README.md). The 60 new cases preserve the successful baseline settings, append one randomly chosen unused duty per k, and inherit the preceding saved pool. Graph preparation is independent (concurrency 50); CG dependencies stay within each chain; each final MIP waits only on its own CG. Inputs through k=40 are frozen for later continuation. Graph array **133908** had **50 running tasks** at10:25EDT; all120CG/MIP dependencies were verified. Exact job IDs and launch verification are in the campaign records. These are submissions, not new target matches. Astra only for this usage week.

**Model and terminology audit, 13 September:** [Start with the current model comparison and table definitions](../model_fairness_audit_20260913/README.md). The repaired baseline chains all reach15; the old C1/C2/C4 gaps are historical. The stricter actual-battery/15%-reserve/PARX60/nonlinear-charging/capacity diagnostics match2/4 k2 inputs and1/4 k3 inputs, with finite-pool proofs but no full pricing certificates. No experiment combines every documented GIRO constraint. The original65% entry is a recharge target, not a documented terminal floor. The live Google Doc now explains these scopes beside the tables, preserves the figure tabs, and includes editable assumptions/history/results comparisons. [Primary-source audit](../model_fairness_audit_20260913/giro_requirements_audit.md), [implemented constraints and algorithms](../model_fairness_audit_20260913/constraint_results_audit.md).

**Earlier queue, 13 September 05:38 EDT (superseded by the new chain expansion):** [The shared 32-duty graph preparation timed out](../queue_recovery_20260912/status_20260913T093758Z/README.md) after12h32m, before a cache or parent CG result was saved. All20 dependent jobs auto-cancelled before start. No active EVSP–DR jobs remain; held historical work is untouched. Completed chain, algorithm-comparison and fee results below are unchanged. This was a scheduler time limit, not preemption or out-of-memory.


**Queue fixed, 12 September 16:57 EDT:** [Running work, dependency explanations and launch map](../queue_recovery_20260912/README.md).

**Start here:** [All six chains reach k=15, 13 September 02:35 EDT](../queue_recovery_20260912/status_20260913T063519Z/README.md). The [earlier catch-up](CATCH_UP_20260912.md) preserves the state before the queue recovery.

**Controlled comparisons complete:** [24 paired allocations on three frozen cases](../controlled_comparison_20260913/status_20260913T063519Z/README.md). Indexed replay reduced total CG time by 12.4–18.1%; omitted LP setup reduced it by 9.3–14.6%, with matched inherited pool hashes, iteration counts and certified objectives. Full inheritance reduced time by 40.6–64.5% and improved MIP fleets from 9 to 8, 11 to 10, and 17 to 15. Six original full-scan arms hit the CG budget during import; their MIPs were intentionally skipped.

**Charging-start fee experiment complete, 13 September 04:36 EDT:** [36 chain runs plus 12 GIRO jobs](../zero_charge_start_fee_20260913/README.md) compare fees 5 and 0. All 36 chain MIPs finished; 35 match their target with finite-pool fleet proofs. The exception is C1k15 fee 0: 16 buses, bound 15, unproved. All 36 selected solutions pass individual-route replay. CG has 34 certificates; both C1k15 arms hit the two-hour budget. Across 17 certified pairs, fee-0 CG took a median 2.91 times as long. Charging starts increased in all 18 pairs. [Completed tables, costs and proof limits](../zero_charge_start_fee_20260913/status_20260913T083655Z/README.md). The physically feasible fee-5 15-bus schedule remains feasible when its fee is repriced to zero, but its inclusion in the fee-0 saved pool is not established. The separate GIRO study is complete: electricity savings remain 43–57% without a start fee while returning at least GIRO’s aggregate energy. Joint pooling adds no gain beyond fixed-duty charging optimization in those zero-fee cases. [GIRO cost tables](../zero_charge_start_fee_20260913/status_20260913T053436Z/README.md).

This is the common index for experiment settings, execution records, results and source evidence. The compact [Week of 14 September status tab](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow) and [Figures with explanations tab](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) are the live document view. The [CG curves and bus schedules library](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) restores the earlier visual evidence with dated captions. Preserve all figures when updating or simplifying text. Superseded material is also kept in the separate [Historical archive document](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit).

The [current Slides deck](https://docs.google.com/presentation/d/11bJ-4B5khXtSPwv1sNlvGgme8JCB65jVIT-RSWu3x9E/edit) is the nine-slide editable presentation view; the [historical Slides copy](https://docs.google.com/presentation/d/1RAzaiZSh7DRf_By32mQXPCcwT1PvPDsk0S2xzxMOnDQ/edit) is kept separate.

Current build: **2,658 artifact/stage records across 57 campaign/source groups**, from the collection completed at 10:35 EDT on 14 September. Counts are not independent experimental sample sizes.

| To find | Open |
|---|---|
| Filter experiments and compare exact results | [Experiment register workbook](../01a07ecc-9b77-79b3-9782-e4308a80ba07/EVSP_DR_Experiment_Register.xlsx) |
| Which campaigns exist and what has been collected | [Campaign index](REGISTER.md) |
| Every normalized result and its original source | [Results](RESULTS.md), [CSV](register.csv), [JSON](register.json) |
| April/May and summer evidence outside the live collector | [Historical evidence](HISTORICAL_EVIDENCE.md), [inventory](historical_inventory.csv) |
| Failed runs, precise errors and recoverability | [Execution issues](EXECUTION_ISSUES_20260910.md) |
| Field meanings and proof limits | [Data dictionary](DATA_DICTIONARY.md) |
| Concurrency, memory and reserved-node policy | [Resource policy](RESOURCE_POLICY.md) |

Latest verified baseline results: **all six chains attain k=15**. All 37 full-pool CG cases are certified; all 37completed MIPs match their targets, with finite-pool fleet proofs and individual-route replay. These use covering, 240 kWh / 240 kW, no shared-station capacity or terminal-SOC floor. Duplicate-coverage removal has not been validated. [Dated six-chain table](../queue_recovery_20260912/status_20260913T063519Z/README.md).

All 24 controlled comparison allocations finished.42 CG arms certified and42 MIPs completed; six original full-scan arms were capped during import. No new execution failure, confirmed preemption or invalid dependency appeared. Scientific outcomes, time caps and scheduler completion remain distinct. Held historical tasks and V2G work were untouched.

## How to read a result

A case is identified by its input and settings, not just by k. Keep trip selection, battery and charging power, SOC/time grid, tariff, master sense, initialization, objective and code revision together. A restart or repeated snapshot is not a new independent sample.

- **Weighted LP objective** includes the bus coefficient and route costs. **Fractional route count** is the sum of LP route weights; it is a different quantity.
- **CG certified** means pricing reached its stated reduced-cost tolerance within its documented graph/model. A running or time-limited RMP objective is not automatically a full-model lower bound.
- **Pool fleet proof** concerns the supplied route columns. It does not establish full-model integer optimality.
- **Physical replay** and **shared-station capacity** are separate checks. Unknown means the source does not establish the claim.
- **Scheduler completion**, **solver termination**, **validated result**, and **target matched** have separate fields. Failed and superseded attempts remain visible.

## Updating the record

Keep raw output files and hashes. Collect a dated cluster snapshot, rebuild the normalized register from that snapshot, then refresh the workbook. Record new campaigns in the collector before launch. Preserve manual interpretation separately from generated tables. The hourly monitor refreshes registered evidence when a meaningful change occurs; its timestamp identifies the age of the result view.

Historical files are indexed without silently treating old figures or unaudited summaries as validated experimental results. See the inventory boundary and missing-data notes before claiming coverage of the entire project history.

From the project root:

```sh
python3 outputs/research_register/build_register.py \
  --snapshot outputs/post_meeting_20260910/monitor/20260913T093758Z.json \
  --out-dir outputs/research_register --reports-root outputs
/Users/nadan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  outputs/research_register/build_workbook.mjs
```

Replace the snapshot with the newly collected dated file. Preserve all supplements listed in register.json (currently six) by passing each with --supplement, including the graph timeout audit. The workbook builder requires the bundled `@oai/artifact-tool` package; `node_modules` here is a local dependency symlink, not experiment data. `validation.json` checks source identity and result scopes; the workbook's `workbook_checks.json` checks record counts and formula errors. All seven workbook tabs were rendered; updated result and charging fields were visually checked.

A copy of this index, dated snapshots and workbook is stored on Unicorn at `/home/nc437/ladder-lite/research-register/`. Local documentation paths refer to the Mac project; solver source paths refer to Unicorn. See `cluster_mirror.json` for the copied version and hashes.

## Parallel expansion — 11 September 2026 UTC

[Launch records](../parallel_research_20260911/README.md): 75 independent fresh covering CG cases (array 810454, concurrency 50) complete the six-chain k=2–15 grid without repeating the nine existing covering cases. Four additional inherited-column chains P1/P2/P4/P6 have 36 CG jobs with dependencies only within each chain. These are submissions, not results or certificates.

## Default-partition MIP reliability

[Preemption study](preemption_study/README.md) tracks per-attempt scheduler outcomes, priority samples, queue wait, runtime exposure and preemption cost for overnight default MIPs. Scientific results and scheduler outcomes are separate; pending/running jobs do not count as completed attempts. The dated collector snapshots retain accounting history.

## Storage and restoration

The [12 September cleanup](https://github.com/ndandnd/EVSP-DR/tree/b1129bf4/outputs/storage_cleanup_20260912) completed lossless archival of 156 cold `phys240kw` and `cg_acceleration_20260903` journals/cache files: **133,737,953,364 bytes (133.74 GB/124.55 GiB) reclaimed**, with 15,583,969,540 archive bytes retained. All four corrected array 950555 workers completed successfully; hashes, ledger reconciliation and a real-file restore passed. Initial array 950484 removed nothing and remains recorded separately. Active/pending/held inputs and V2G work stayed protected. Exact restoration manifests and code are pushed on `codex/storage-cleanup-20260912`; the prior September 11 savings below are separate.

[Storage cleanup audit](../storage_cleanup_20260911/README.md) records lossless archival of cold historical column journals. Compressed pools retain original hashes and require restoration before legacy scripts expecting the original uncompressed paths are run. Current and held-job pools remain protected. GitHub stores code and selected evidence; it is not a complete backup of untracked cluster artifacts.

## Proposed algorithm improvements

**13 September correction to the historical notes below:** the dedicated capacity event path's station-specific tariff reconstruction issue was fixed in `550bc795` and independently regression-tested before the `309d98d2` capacity comparison. That accounting blocker is resolved for this path. Generic pool-MIP/seed readers do not yet establish heterogeneous-power support. Capacity acceleration is implemented and locally tested, but three real-data comparison pairs still hit their three-hour limits without pricing certificates. Indexed inheritance, omitted LP setup and full inheritance have since completed controlled cluster benchmarks. Use the [current status audit](../model_fairness_audit_20260913/constraint_results_audit.md); the dated preliminary claims below are preserved as history.

The [independent code and literature review](https://github.com/ndandnd/EVSP-DR/blob/206315dddd99f44a49141c04bb786f2b38f459dd/outputs/algorithm_review_20260912/REVIEW.md) and its work orders describe proposed optimizations. That review did not implement accelerations or claim measured solver speedups. Coordinate any subsequent benchmarks with that task and preserve existing runs as controls.


**12 September, 02:12 EDT — capacity acceleration awaits an accounting fix.** The separate algorithm-review task reports a pre-existing route-cost reconstruction issue in source `253588e9b22d68fcbc67cb56bc3eb30cbb0e16b6`: with time-varying electricity prices and a 60-kW station, the saved route cost uses the global 240-kW rate to allocate charging across tariff periods. Pricing uses the station's 60-kW rate. Its eight synthetic checks contain two cost discrepancies (0.12 and 4.8); the independent reduced-cost check rejects both. The six flat-price or 240-kW controls pass. The register maintainer inspected these recorded outcomes, but has not independently rerun the reproducer or verified the reported source diagnosis.

The review task owns the minimal reproduction, source diagnosis and fix validation. Do not integrate its capacity acceleration or deploy variable-tariff, heterogeneous-power runs until that check passes. This observation does not establish that the registered flat-price/240-kW capacity pilot is invalid; no campaign result or certificate is changed here. Local evidence: `outputs/algorithm_benchmarks_20260912/capacity/record_mismatch.json` (SHA-256 `f06cc939619fb677c82e162d6aeefebf6d143e22675198c6a674a44d16297fca`) and its adjacent `reproduce_record_mismatch.py`; the review task will provide the pinned final report separately.


**Completed local benchmarks, 12 September.** The [pinned prototype report](https://github.com/ndandnd/EVSP-DR/blob/b142f360f8a9018c2f50e3419f0dddcaf87a489d/outputs/algorithm_benchmarks_20260912/README.md) records replay of 210 generated sequences on a 48-trip real-data subset at 0.558 → 0.372 s (1.50×; graph already built). The latter combines median replay with one measured cold index setup, rather than timing a complete import job. A synthetic 14-solve covering-master benchmark measured 78.10 → 58.09 ms after omitting unused incidence construction (1.34×). A prepared-index replay test reaches 9.75×, excluding its setup. Historical profiles imply only about 5–6% whole-run time savings even if incidence construction were eliminated entirely; these are conditional ceilings, not measured solver gains. Capacity selector timings depend strongly on repeated queries and remain subject to the accounting fix described above. Complete CG runs and the actual bounded warm importer have not been benchmarked; no cluster speedup is measured. The synthetic fixtures do not establish whether a registered production run encountered the accounting defect. Production code, cluster jobs, fleet results and certificates are unchanged by this benchmark package. Source-only reproductions, raw samples and validation records are pinned in commit `b142f360f8a9018c2f50e3419f0dddcaf87a489d`.


**Paired validation launched:** [nine paired efficiency allocations](efficiency_validation_20260912/README.md) were submitted at 02:53 EDT. Four warm attempts stopped before optimization with a launcher argument error; all four replacements (966398–966401) are running as of 03:02 EDT. The five other jobs remain unchanged. Hourly collection retains both original and recovery v2 records. No completed paired speedup result is available. The [pre-launch record](EFFICIENCY_VALIDATION_PLANNED_20260912.md) is historical.


**05:25 EDT paired efficiency update:** Two fresh pairs show 5.8% slower and essentially unchanged runtime. One reverse-order warm pair uses 22.4% less runtime excluding common cache preparation, with import checking 366.08 → 5.62 s and matching certified LP objectives but different CG iterations. These are descriptive observations, not a general speedup claim. [Dated evidence](../overnight_extension_20260912/status_20260912T092542Z/paired_results.json).

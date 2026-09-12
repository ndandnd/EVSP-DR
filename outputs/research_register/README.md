# EVSP–DR experiment register

This is the common index for experiment settings, execution records, results and source evidence. The compact [Week of 14 September status tab](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow) and [Figures with explanations tab](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) are the live document view. The [CG curves and bus schedules library](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) restores the earlier visual evidence with dated captions. Preserve all figures when updating or simplifying text. Superseded material is also kept in the separate [Historical archive document](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit).

The [current Slides deck](https://docs.google.com/presentation/d/11bJ-4B5khXtSPwv1sNlvGgme8JCB65jVIT-RSWu3x9E/edit) is the nine-slide editable presentation view; the [historical Slides copy](https://docs.google.com/presentation/d/1RAzaiZSh7DRf_By32mQXPCcwT1PvPDsk0S2xzxMOnDQ/edit) is kept separate.

Current build: **1757 artifact/stage records across 32 campaign/source groups**, from snapshot 2026-09-12T05:26:58+00:00. The historical inventory covers **39 evidence families**. These counts are not independent experimental sample sizes.

| To find | Open |
|---|---|
| Filter experiments and compare exact results | [Experiment register workbook](../01a07ecc-9b77-79b3-9782-e4308a80ba07/EVSP_DR_Experiment_Register.xlsx) |
| Which campaigns exist and what has been collected | [Campaign index](REGISTER.md) |
| Every normalized result and its original source | [Results](RESULTS.md), [CSV](register.csv), [JSON](register.json) |
| April/May and summer evidence outside the live collector | [Historical evidence](HISTORICAL_EVIDENCE.md), [inventory](historical_inventory.csv) |
| Failed runs, precise errors and recoverability | [Execution issues](EXECUTION_ISSUES_20260910.md) |
| Field meanings and proof limits | [Data dictionary](DATA_DICTIONARY.md) |
| Concurrency, memory and reserved-node policy | [Resource policy](RESOURCE_POLICY.md) |

Latest launch: [Overnight chains through k=15 and 32-duty decomposition](../overnight_extension_20260912/README.md). 87 CG cases and 87 dependent default-partition MIPs. At 01:27 EDT, 14 new CG cases were pricing-certified and two decomposition-group MIPs each proved eight buses within their saved pools. Warm-chain integer extensions and a combined 32-duty result are not completed in this snapshot. [Verified update](../overnight_extension_20260912/RESULTS_20260912T052658Z.md).

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
  --snapshot outputs/post_meeting_20260910/monitor/20260912T052658Z.json \
  --out-dir outputs/research_register --reports-root outputs
/Users/nadan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  outputs/research_register/build_workbook.mjs
```

Replace the snapshot with the newly collected dated file. The workbook builder requires the bundled `@oai/artifact-tool` package; `node_modules` here is a local dependency symlink, not experiment data. `validation.json` checks source identity and result scopes; the workbook's `workbook_checks.json` checks record counts and formula errors. All seven workbook tabs were rendered and visually checked for this build.

A copy of this index, dated snapshots and workbook is stored on Unicorn at `/home/nc437/ladder-lite/research-register/`. Local documentation paths refer to the Mac project; solver source paths refer to Unicorn. See `cluster_mirror.json` for the copied version and hashes.

## Parallel expansion — 11 September 2026 UTC

[Launch records](../parallel_research_20260911/README.md): 75 independent fresh covering CG cases (array 810454, concurrency 50) complete the six-chain k=2–15 grid without repeating the nine existing covering cases. Four additional inherited-column chains P1/P2/P4/P6 have 36 CG jobs with dependencies only within each chain. These are submissions, not results or certificates.

## Default-partition MIP reliability

[Preemption study](preemption_study/README.md) tracks per-attempt scheduler outcomes, priority samples, queue wait, runtime exposure and preemption cost for overnight default MIPs. Scientific results and scheduler outcomes are separate; pending/running jobs do not count as completed attempts. The dated collector snapshots retain accounting history.

## Storage and restoration

The [12 September cleanup](https://github.com/ndandnd/EVSP-DR/tree/b1129bf4/outputs/storage_cleanup_20260912) completed lossless archival of 156 cold `phys240kw` and `cg_acceleration_20260903` journals/cache files: **133,737,953,364 bytes (133.74 GB/124.55 GiB) reclaimed**, with 15,583,969,540 archive bytes retained. All four corrected array 950555 workers completed successfully; hashes, ledger reconciliation and a real-file restore passed. Initial array 950484 removed nothing and remains recorded separately. Active/pending/held inputs and V2G work stayed protected. Exact restoration manifests and code are pushed on `codex/storage-cleanup-20260912`; the prior September 11 savings below are separate.

[Storage cleanup audit](../storage_cleanup_20260911/README.md) records lossless archival of cold historical column journals. Compressed pools retain original hashes and require restoration before legacy scripts expecting the original uncompressed paths are run. Current and held-job pools remain protected. GitHub stores code and selected evidence; it is not a complete backup of untracked cluster artifacts.

## Proposed algorithm improvements

The [independent code and literature review](https://github.com/ndandnd/EVSP-DR/blob/206315dddd99f44a49141c04bb786f2b38f459dd/outputs/algorithm_review_20260912/REVIEW.md) and its work orders describe proposed optimizations. That review did not implement accelerations or claim measured solver speedups. Coordinate any subsequent benchmarks with that task and preserve existing runs as controls.

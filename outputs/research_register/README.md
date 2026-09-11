# EVSP–DR experiment register

This is the common index for experiment settings, execution records, results and source evidence. The [Google Doc register tab](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.icnyriwovxgf) explains the index; the other tabs contain interpretation and figures.

Current build: **1740 artifact/stage records across 30 campaign/source groups**, from the 11 September 2026, 01:27 EDT cluster snapshot. The historical inventory covers **39 evidence families**. These counts are not independent experimental sample sizes.

| To find | Open |
|---|---|
| Filter experiments and compare exact results | [Experiment register workbook](../01a07ecc-9b77-79b3-9782-e4308a80ba07/EVSP_DR_Experiment_Register.xlsx) |
| Which campaigns exist and what has been collected | [Campaign index](REGISTER.md) |
| Every normalized result and its original source | [Results](RESULTS.md), [CSV](register.csv), [JSON](register.json) |
| April/May and summer evidence outside the live collector | [Historical evidence](HISTORICAL_EVIDENCE.md), [inventory](historical_inventory.csv) |
| Failed runs, precise errors and recoverability | [Execution issues](EXECUTION_ISSUES_20260910.md) |
| Field meanings and proof limits | [Data dictionary](DATA_DICTIONARY.md) |
| Concurrency, memory and reserved-node policy | [Resource policy](RESOURCE_POLICY.md) |

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
  --snapshot outputs/post_meeting_20260910/monitor/20260911T020803Z.json \
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

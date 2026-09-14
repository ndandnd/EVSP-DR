# Overnight experiments: distinguish missing columns from insufficient MIP time

**Latest results, 14 September 05:29 EDT:** [Sixteen completed-case comparisons](status_20260914T092543Z/README.md). Complementary C4/C5/C6k15 find19/17/19, all bounds15/unproved, versus original19/16/20. C1k10c200 finds11/bound10. Diagnostic32CGcertificates/54MIPs. All23longer-MIP results unchanged. No new jobs or source changes.

**Latest results, 14 September 04:28 EDT:** All 23 longer MIPs are complete: inherited pools 9/9 target matches; fresh pools 0/14, with three proved pool limits and eleven open gaps. All pool/input hashes agree with originals. Seven inherited reruns prove the fleet within 30 minutes, so additional elapsed time alone is not the explanation. [Detailed evidence](status_20260914T082509Z/LONGER_MIP_RESULTS.md). There are 31 certified diagnostic CG endpoints and 50 MIPs overall. [All treatments](status_20260914T082509Z/README.md). A separate [30-job repeatability/gap batch](../mip_repeatability_20260914/README.md) is now running; this frozen 99-case campaign is unchanged.

**Results, 14 September 03:26 EDT:** [All completed comparisons and continuations](status_20260914T072439Z/README.md).27certified CGendpoints and23MIPs. C5k19continuation recovers19/proved in pool/replay after314.0cumulative CGminutes. C2k10c200 improves12→11/bound10; C3k15both17/bound15 versusoriginal18. C1k8complementary proves9, so improvements remain case dependent. No longer unchanged-pool MIP endpoints yet. No new jobs or algorithm changes.

**Results, 14 September 01:26 EDT:** [First C5 k5 comparison and C4 k19 continuation](status_20260914T052404Z/README.md). Fifteen CG endpoints are certified; two final MIPs are available. Complementary selection recovers five buses where original and 200-column pools require six. **Live queue at 01:35 EDT:** 57 diagnostic jobs plus nine chain-extension jobs running; no arbitrary array throttle. All 99 diagnostic jobs remain submitted.

Prepared on the night of 13–14 September. This is a separately registered diagnostic batch; it does not replace the six ongoing k=16–25 chains or the completed cumulative-budget comparison.

**All99 jobs submitted, verified00:33EDT.** The first59 independent jobs were all running at00:09EDT. Native checkpoint validation job157899 passed and automatically submitted the final two CG continuations (160347/160348) and their MIPs (160349/160350). The complete batch has61 independent jobs and38 MIPs that depend only on their own CG. See [scheduler verification](scheduler_verification.json) and [job map](case_jobs.json). Submission status is separate from scientific results.

## What we will learn

| Experiment | Independent jobs | Following jobs | Question |
|---|---:|---:|---|
| Longer MIPs on unchanged saved pools | 23 MIPs | — | Can a longer integer search find the target, or prove that this pool cannot? |
| Fresh CG adding up to 200 columns per iteration | 18 CGs | 18 MIPs | Does adding more improving routes produce a better integer pool? |
| Fresh CG selecting 30 less-overlapping columns per iteration | 18 CGs | 18 MIPs | Does a more complementary selection produce a better integer pool? |
| Continue two time-limited k=19 CG runs | 2 CGs | 2 MIPs | Does another four hours reach a pricing certificate or improve the integer solution? |
| **Total** | **61 jobs can start without another new job finishing** | **38 MIPs wait only for their own CG** | **99 jobs** |

The existing warm chains must wait for the preceding k because they import its columns. That leaves roughly six CGs active at a time after graph preparation finishes. The new comparisons are independent across cases and treatments, so they fill this parallelism gap without breaking those dependencies.

## Time allowances and fair comparisons

- **Longer MIPs:** 3½ hours total, with up to three hours for fleet search and the remaining time for charging-cost optimization (at least 30 minutes if stage 1 uses its full allowance). The second stage constrains the fleet to be no greater than the first-stage incumbent. These start new Gurobi search trees on the identical saved pools; they do not resume a saved search tree. The original one-hour results remain the controls.
- **Fresh CG comparisons:** use each case's original cumulative CG allowance, from 1.83 to 23.14 hours for the selected cases. CG stops early if pricing certifies convergence. The final MIP keeps the original one-hour total allowance, with at most 30 minutes initially assigned to fleet search. Longer CG allowances are not predictions of runtime.
- **k=19 continuations:** chains 4 and 5 resume private copies of their saved checkpoints. The CG budget rises from four to eight hours **including time already spent**. Native resume also rechecks the parent routes; this overhead is charged. Each receives the usual one-hour MIP. Their production successors continue using the original pools.

These remain baseline cases: set covering, 240 kWh batteries, 240 kW charging, the original tariffs and charging-start fee, with no shared charger-capacity or terminal-SOC constraint. This batch changes column selection or computational allowance, not the physical model. The 200-column arm retains reduced-cost selection; the complementary arm keeps 30 columns and uses diversity weight 0.5 with candidate multiplier 4. These are two separate treatments, not a combined change.

## Why these cases

Selection is frozen from snapshot `20260914T024847Z`, whose SHA-256 is recorded in [selection.json](selection.json). All 18 fresh cases that missed their targets receive both CG treatments. The longer MIPs cover all 14 unresolved fresh-pool gaps and nine unresolved extension gaps in that snapshot. Four fresh pools already proved to require an extra bus are excluded from longer MIP searches: more time cannot fix a proved limitation of the unchanged pool.

These are deliberately selected difficult cases, not an unbiased sample for estimating success rates over all possible trip subsets. A changed column-selection rule can generate a different pool even when the original LP has a pricing certificate: that certificate concerns the LP objective, not the existence of an integer target-fleet solution in its saved columns.

## Execution and evidence

All jobs use `default_partition` and exclude `scaglione-compute-01`. CG requests 8 CPUs/96 GiB; MIP requests 8 CPUs/24 GiB. There is no additional concurrency throttle: all 61 independent jobs are eligible, subject to Slurm's admission and priority decisions. Existing jobs, true previous-k dependencies, held historical work and V2G work are unchanged.

The frozen `manifest.json` records each input hash, source commit, source pool hash, graph-cache identity, exact command, model settings, resource request and comparison. `jobs.json` records submission commands and scheduler IDs; `case_jobs.json` maps cases to jobs. `scheduler_verification.json` records the post-submission check. Native validation uses a separate smoke directory and is excluded from scientific results.

The top-level settings block records the baseline control; each case's arguments and `changed_factor` specify its treatment. The two short resume checks, jobs155072/155894, allowed only30/600 additional seconds. Both stopped during inherited-route replay before a new CG iteration. Their retained prior LP was not promoted to a new usable endpoint. Job157899 allows2400 seconds for setup/replay and one new iteration. This is a validation-only allowance; production continuation budgets remain eight cumulative hours. The initial explanation that max-iters equalling the saved iteration caused zero new iterations was incorrect: native max-iters counts new iterations after resume.

Outputs live under `/share/scaglione/nc437/evsp-dr/overnight_diagnostics_20260914/cases/`, linked from `/home/nc437/ladder-lite/overnight_diagnostics_20260914/cases/`. Every scheduler attempt has a private directory. A `completion.json` marker binds each published result and column journal to their hashes. Failed/preempted attempts remain visible. MIP attempts also enter the existing preemption study. Requeued MIPs restart their search; their lost work is not hidden.

Report these separately: CG stopping reason and pricing certificate; weighted LP objective and fractional route count; integer buses and finite-pool fleet bound/proof; individual-route physical replay; shared capacity; target attainment; wall time/CPU use; scheduler failure or preemption. A time-limited restricted-master objective is not automatically a full-model lower bound.

## Reproduce and collect

The campaign scripts are [campaign.py](campaign.py), [worker.py](worker.py), and [worker.sub](worker.sub). Preparation refuses duplicate manifests; submission is locked and skips already recorded jobs. `validate.py` retains the completed native checks for both CG treatments and dependent two-stage MIPs, then checks a copied same-k checkpoint. Worker tests cover failure, publication, source binding and attempt isolation.

The main collector `outputs/meeting_20260910/collect_remote.py` includes this campaign and only promotes results with matching completion hashes. The experiment register resolves identities from the manifest. The hourly monitor should collect this batch alongside existing chains, preserve this frozen design, and report material results or actionable failures. Do not launch a new treatment simply because one finishes.

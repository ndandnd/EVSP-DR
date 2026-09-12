# Independent queue classification

Snapshot cutoff: `2026-09-12T20:11:33.242475Z`, from `before.json` (SHA-256 `3819d33133cd6dd7ead1126230d1c1e0497e8188e55510fc18e69f8b5f85e6d5`).  The audit covers the 45 pending EVSP–DR jobs in that snapshot and excludes held array `537227`/`EVSPV2G` by instruction.  Jobs submitted or cancelled after the cutoff are not used to change this classification.

The categories mean:

* **A**: an exact same-treatment replacement attempt was already present in the snapshot, so the old queued job is obsolete.  A replacement job being present does not mean that it has produced a result.
* **B**: the requested work is still live and its direct parent was running at the cutoff.
* **C**: no exact replacement result/attempt was present at the cutoff and the case still lacks the requested research result; the dependency is failed, transitively failed, or is a failed array gate.

## A — superseded queued attempts (4)

| queued job | case / stage | queue dependency and state | exact replacement present at cutoff | evidence |
|---|---|---|---|---|
| `949704` | `w2_k14`, bounded-extension MIP | `DependencyNeverSatisfied`, `afterok:949703(failed)` | `37583` (`rec2b14M`), **RUNNING** | `w2_chain_recovery_retry2_20260912/README.md` states that corrected jobs `37583–37585` replace original blocked `949704–949706`; `before.json` shows both paths and resources. |
| `949705` | `w2_k15`, bounded-extension CG | `DependencyNeverSatisfied`, `afterok:949703(failed)` | `37584` (`rec2b15CG`), **RUNNING** | Same recovery README; corrected retry uses isolated output and the same bounded-extension science and budgets. |
| `949706` | `w2_k15`, bounded-extension MIP | `Dependency`, `afterok:949705(unfulfilled)` | `37585` (`rec2b15M`), **PENDING** after `37584` | Same recovery README and `before.json`. |
| `37534` | `w2_k15`, first recovery MIP (`rec2k15M`) | `DependencyNeverSatisfied`, `afterok:37533(failed)` | `37585` (`rec2b15M`), **PENDING** after `37584` | `EXECUTION_ISSUES_20260910.md` records `37532/37533` as launcher failures (`PYTHON_BIN` omitted) and the corrected `37583–37585` retry. |

These are the only pre-cutoff queued jobs for which an exact same-treatment replacement was already recorded.  In particular, the bounded outcomes for `w1_k07…k10`, `w2_k09`, and `w4_k10` are not replacements for the older full-pool chains described below.

## B — genuinely waiting on a running parent (2)

| queued job | case / stage | direct dependency | parent evidence |
|---|---|---|---|
| `949692` | `w1_k15`, bounded-extension MIP | `afterok:949691(unfulfilled)` | `949691` (`w1_k15`) was **RUNNING** in `before.json`; no failure is in its dependency chain. |
| `37585` | `w2_k15`, corrected-recovery MIP | `afterok:37584(unfulfilled)` | `37584` (`rec2b15CG`) was **RUNNING** in `before.json`; this is the valid downstream of the corrected retry. |

## C — blocked cases still lacking a result (39)

### Overnight decomposition MIPs: 10 jobs

All ten jobs show `DependencyNeverSatisfied` with `aftercorr:949623_*(failed)` in `before.json`.  The array parent statuses in the same snapshot/accounting are the important distinction: only task `949623_3` failed (`TIMEOUT`); the other nine corresponding CG tasks completed.  Thus the nine completed pools need a fresh, task-specific MIP dependency/input; they do not have an existing MIP result.  The d00_g3 task needs CG recovery first.

| queued MIP | case | corresponding CG | evidence at cutoff |
|---|---|---|---|
| `949624_3` | `d00_g3` | `949623_3` **TIMEOUT** | No `cg.json`; only `cg_949644_r0_start.json`, lock and phase telemetry. `EXECUTION_ISSUES_20260910.md` identifies the 04:47:09 startup/graph-construction timeout and no pricing certificate. |
| `949624_17` | `d04_g1` | `949623_17` **COMPLETED** | Certified `cases/d04_g1/cg.json` exists (`stop_reason=certified`, 1,083 iterations, zero artificials); no MIP result was recorded at cutoff. |
| `949624_18` | `d04_g2` | `949623_18` **COMPLETED** | Certified `cases/d04_g2/cg.json` exists (712 iterations, zero artificials); no MIP result at cutoff. |
| `949624_21` | `d05_g1` | `949623_21` **COMPLETED** | Certified `cases/d05_g1/cg.json` exists (911 iterations, zero artificials); no MIP result at cutoff. |
| `949624_23` | `d05_g3` | `949623_23` **COMPLETED** | Certified `cases/d05_g3/cg.json` exists (768 iterations, zero artificials); no MIP result at cutoff. |
| `949624_25` | `d06_g1` | `949623_25` **COMPLETED** | Certified `cases/d06_g1/cg.json` exists (1,310 iterations, zero artificials); no MIP result at cutoff. |
| `949624_26` | `d06_g2` | `949623_26` **COMPLETED** | Certified `cases/d06_g2/cg.json` exists (547 iterations, zero artificials); no MIP result at cutoff. |
| `949624_27` | `d06_g3` | `949623_27` **COMPLETED** | Certified `cases/d06_g3/cg.json` exists (1,777 iterations, zero artificials); no MIP result at cutoff. |
| `949624_28` | `d07_g0` | `949623_28` **COMPLETED** | Certified `cases/d07_g0/cg.json` exists (1,447 iterations, zero artificials); no MIP result at cutoff. |
| `949624_29` | `d07_g1` | `949623_29` **COMPLETED** | Certified `cases/d07_g1/cg.json` exists (1,165 iterations, zero artificials); no MIP result at cutoff. |

The array-level `aftercorr` gate is therefore not evidence that the nine corresponding CG computations failed.  It is a scheduler/dependency failure affecting otherwise completed pools.  `overnight_extension_20260912/manifest.json` and `jobs.json` bind these indices to the cases above.

### Recombined decomposition chain: 15 jobs

The five queued `join` jobs have no parent result yet, and the ten queued `Mjoin` jobs either wait on one of them or point directly at a timed-out parent.  `before.json`/accounting gives the following exact graph.

| queued job | case | direct dependency | failed/transitive root and evidence |
|---|---|---|---|
| `949749` | `join00` | `afterok:949624_0:949624_1:949624_2:949624_3` (task `_3` pending) | `949623_3` (`d00_g3`) TIMEOUT; no component MIP for `_3`. |
| `949757` | `join04` | `afterok:949624_16:949624_17:949624_18:949624_19` (`_17`, `_18` pending) | The two queued component MIPs are blocked by the failed array gate; their CG parents completed. |
| `949759` | `join05` | `afterok:949624_20:949624_21:949624_22:949624_23` (`_21`, `_23` pending) | The two queued component MIPs are blocked by the failed array gate; their CG parents completed. |
| `949761` | `join06` | `afterok:949624_24:949624_25:949624_26:949624_27` (`_25`–`_27` pending) | The three queued component MIPs are blocked by the failed array gate; their CG parents completed. |
| `949764` | `join07` | `afterok:949624_28:949624_29:949624_30:949624_31` (`_28`, `_29` pending) | The two queued component MIPs are blocked by the failed array gate; their CG parents completed. |
| `949750` | `join00` MIP | `afterok:949749(unfulfilled)` | Transitive `join00` → component MIP `_3` → `949623_3` TIMEOUT. |
| `949752` | `join01` MIP | `afterok:949751(failed)` | `949751` TIMEOUT; its `decomposed_solution.json` is only a 35-bus component-sum baseline, not a parent CG/MIP result. |
| `949754` | `join02` MIP | `afterok:949753(failed)` | `949753` TIMEOUT; its 35-bus decomposed construction is not a parent result. |
| `949756` | `join03` MIP | `afterok:949755(failed)` | `949755` TIMEOUT; its 36-bus decomposed construction is not a parent result. |
| `949758` | `join04` MIP | `afterok:949757(unfulfilled)` | Transitive component MIPs `949624_17/18` remain blocked. |
| `949760` | `join05` MIP | `afterok:949759(unfulfilled)` | Transitive component MIPs `949624_21/23` remain blocked. |
| `949763` | `join06` MIP | `afterok:949761(unfulfilled)` | Transitive component MIPs `949624_25/26/27` remain blocked. |
| `949765` | `join07` MIP | `afterok:949764(unfulfilled)` | Transitive component MIPs `949624_28/29` remain blocked. |
| `949767` | `join08` MIP | `afterok:949766(failed)` | `949766` TIMEOUT; its 37-bus decomposed construction is not a parent result. |
| `949769` | `join09` MIP | `afterok:949768(failed)` | `949768` TIMEOUT; its 35-bus decomposed construction is not a parent result. |

The five failed parent jobs (`949751`, `949753`, `949755`, `949766`, `949768`) retained startup telemetry but no parent `cg.json`, pricing certificate, or parent MIP result.  This is why the existing decomposed constructions do not qualify as category A replacements.

### Older full-pool warm chain P1: 11 jobs

These are the full predecessor-pool treatment in campaign `warm_multichain_p1246`, not the later bounded 512-route treatment.  Root `810293` (`k07_p1` CG) timed out after 08:17:13 against an 08:15:00 allocation with zero iterations/final LP and no usable child pool.  The pending descendants are:

| queued job | case / stage | direct dependency | transitive root |
|---|---|---|---|
| `810296` | `k08_p1` CG | `afterok:810293(failed)` | `810293` TIMEOUT |
| `810319` | `k09_p1` CG | `afterok:810296(unfulfilled)` | `810319` → `810296` → `810293` |
| `810320` | `k10_p1` CG | `afterok:810319(unfulfilled)` | `810320` → `810319` → `810296` → `810293` |
| `810952` | `k07_p1` freeze | `afterok:810293(failed)` | `810293` TIMEOUT |
| `810953` | `k07_p1` MIP | `afterok:810952(unfulfilled)` | `810952` → `810293` |
| `810954` | `k08_p1` freeze | `afterok:810296(unfulfilled)` | `810296` → `810293` |
| `810955` | `k08_p1` MIP | `afterok:810954(unfulfilled)` | `810954` → `810296` → `810293` |
| `810956` | `k09_p1` freeze | `afterok:810319(unfulfilled)` | `810319` → `810296` → `810293` |
| `810957` | `k09_p1` MIP | `afterok:810956(unfulfilled)` | `810956` → `810319` → `810296` → `810293` |
| `810958` | `k10_p1` freeze | `afterok:810320(unfulfilled)` | `810320` → `810319` → `810296` → `810293` |
| `810959` | `k10_p1` MIP | `afterok:810958(unfulfilled)` | `810958` → `810320` → `810319` → `810296` → `810293` |

The completed bounded cases `w1_k07…w1_k10` in the overnight manifest have `max_columns=512` and a 900-second replay limit.  The full-pool campaign manifest at remote path `/home/nc437/ladder-lite/nested_warm_multichain_p1246_k2_10_20260910_ecb60c1/batch_manifest.json` records `inheritance=full predecessor event-column pool plus real child singletons`.  Their matching input hashes do not make them the same treatment; no exact full-pool result exists for these pending cases.

### Older full-pool warm chain P2/P4: 3 jobs

| queued job | case / stage | direct dependency | transitive root and evidence |
|---|---|---|---|
| `810975` | `k09_p2` MIP | `afterok:810974(failed)` | Freeze `810974` failed because there was no usable terminal source; its CG parent `810332` timed out with no final LP/pricing certificate. `EXECUTION_ISSUES_20260910.md` explicitly says the MIP did not optimize. |
| `810994` | `k10_p4` freeze | `afterok:810344(failed)` | `810344` (`k10_p4` CG) TIMEOUT during initialization; no usable child pool. |
| `810995` | `k10_p4` MIP | `afterok:810994(unfulfilled)` | Transitive `810995` → `810994` → `810344` TIMEOUT. |

The overnight `w2_k09` and `w4_k10` bounded results are separate 512-route treatments and do not replace these full-pool cases.  The same distinction applies to the full-pool P1 descendants above.

## Evidence index

* Queue, direct dependencies, and accounting: `outputs/queue_recovery_20260912/before.json`.
* Overnight case/index mapping and bounded inheritance: `outputs/overnight_extension_20260912/manifest.json`, `jobs.json`, and `README.md`.
* Exact recovery replacement provenance: `outputs/w2_chain_recovery_retry2_20260912/README.md`, `outputs/w2_chain_recovery_20260912/README.md`.
* Timeout, failed-freeze, and joined-baseline interpretations: `outputs/research_register/EXECUTION_ISSUES_20260910.md`.
* Latest pre-cutoff research count: `outputs/overnight_extension_20260912/RESULTS_20260912T193423Z.md` (64 original extension MIPs; no completed result claimed for the ten pending decomposition MIPs).

## Full-pool indexed recovery pre-launch correction

The bounded review initially compared the MIP arguments with the e091 source checkout.  The worker actually invokes the separately pinned MIP checkout `871d057e1067411f09581e37d78f7c1ca43f68bb`; that runner accepts `--stage1-timelimit 1800` and implements the requested 3,600-second two-stage solve.  The relative case CSV paths also resolve correctly because `exact_pricer_expanded.py` anchors them under `code/data`.

Remote verification found all 37 target caches attested to e091 and their case input hashes, six usable initial parent LPs (`artificials=0`, positive iterations), staged worker/test files, and `30 passed, 1 skipped` combined tests.  The six root CG jobs were running with the chain-local afterok graph and the required node exclusion.  The consumer network hash assertion was added to close the remaining compatibility-audit gap.  No launch blocker remained at the end of this review; no cluster action was performed by this audit.

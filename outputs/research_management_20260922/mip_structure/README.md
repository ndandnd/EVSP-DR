# Frozen-pool MIP structure and solver pilot — 22 September

**Final result update, 22 September07:56UTC:** all25 trials completed and738 audit checks pass. Fifteen prove the finite-pool fleet optimum; ten terminate normally at the30-minute allowance. All k8 arms prove9; all sequential C1 k15 arms prove15. Fresh C1 k15 ends18 except PreSparsify1=19; fresh C3 k15 ends18 except PreSparsify1/saved start=17, all bound15/open. Solver-setting effects are mixed; saved-incumbent acquisition and loading revisions remain excluded from optimizer-time comparisons. [Final25-cell tables, full logs and audit](../monitor_20260922T075504Z/mip_structure/README.md). No recovery is needed. The earlier partial snapshots below remain historical evidence.


**Recovery v3 is accepted with the scientific model unchanged.** All five preparations have authenticated matrices. C3 v3 preparation **730241 completed**, replacing preserved failures 729457 and 729775. Its five replacement trials are **730242–730246**. Sequential C1 k15 loader-only attempts 729464–729468 were stopped before any Gurobi log existed and replaced by **730247–730251**, reusing preparation 729463. All 15 healthy original trials were left untouched. See [latest recovery pointer](latest_recovery.json), [v3 receipt](recovery_v3.json), and [immutable v3 manifest](manifest_v3.json).

At the first v3 snapshot, eight trial endpoints are complete: all five C4 k8 arms prove fleet 9; C1 k8 MIPFocus2 and PreSparsify1 prove fleet 9; sequential C1 k15 offline strong-start proves fleet 15 in **11.150 optimizer seconds** (1.362 seconds loading, 4.104 seconds building). This is a partial pilot, not a completed arm comparison.

**Original launch inventory (historical):** C3 and sequential trial IDs below are superseded; active replacement IDs are in [latest_recovery.json](latest_recovery.json). The original jobs and manifests remain preserved.

| Frozen pool | Rows × columns | Nonzeros | Original fleet / target | Preparation | Five trials: default, focus1, focus2, presparsify1, strong start |
|---|---:|---:|---:|---:|---|
| C1 k8 fresh | 194 × 39,940 | 1,092,654 | 9 / 8 (pool proof 9) |729439|729440–729444|
| C4 k8 fresh | 188 × 39,668 | 1,057,136 | 9 / 8 (pool proof 9) |729445|729446–729450|
| C1 k15 fresh | 364 × 79,611 | 2,322,459 | 18 / 15 (open) |729451|729452–729456|
| C3 k15 fresh | 302 × 38,130 | 896,915 | 18 / 15 (open) |729457|729458–729462|
| C1 k15 sequential | 364 × 130,468 | 3,541,122 | 15 / 15 (pool proof 15) |729463|729464–729468|

## What the tests isolate

Each preparation authenticates the original endpoint, CG status, journal, instance, tariff, reference and deadhead hashes, replays the pinned native physical gate once, and requires the resulting ordered pool hash and matrix dimensions to match the original endpoint. Its compact incidence/cost/order artifact is then shared by that case's five trials. No original source or solver is modified. Whole journals remain on Unicorn.

The structure diagnostic measures remaining identical incidence, redundant covering rows, cost-respecting safe column dominance and connected components. It uses bitset indexes and a bounded ten-minute column scan; any incomplete scan is explicitly marked. It never changes a matrix used by the parameter trials. Dominance preserves every negative-cost victim column and only removes a nonnegative-cost victim in favor of a no-more-expensive superset; it applies to binary covering, optionally with an at-most fleet cap; it is not asserted for arbitrary capacity side constraints.

The two LP diagnostics use the **same fleet objective** and, separately, the **same charging-related objective with the original validated stage-one fleet cap**. Each has a 300-second allowance. Candidate duals are sign-corrected and quantized; the certificate includes `sum min(0,c−Aᵀy−μ)` for binary-variable upper bounds. Integer arithmetic with downward objective rounding and upward incumbent rounding makes the reported bound conservative even if solver duals have numerical residuals. A strict forced-variable bound test reports columns that cannot match or improve the verified same-pool incumbent. No screening is applied to the 25 trials, and no new charging MIP runs.

All fleet trials use the **same pool, order, binary covering rows, unit fleet objective, 8 threads, Seed 0 and 1,800-second allowance**. Four arms retain the original deterministic greedy pool start: default, MIPFocus=1, MIPFocus=2, and PreSparsify=1. The fifth supplies a validated saved stage-one incumbent from the exact same ordered pool. This is explicitly an **offline diagnostic**: its original acquisition time is excluded and it is not a new timed start algorithm. Unmatched saved identities block that arm rather than injecting external routes.

Model building, artifact loading, optimize time, first observed target incumbent, bounds, proof flags, all effective relevant parameters and native logs are recorded separately. Fixed Seed 0 and a single run per cell are a pilot, not a seed-robust performance claim; hardware placement also remains a source of variation.

## Physics, proof and resources

Original baseline physics stay fixed: 240 kWh initial/battery capacity; constant 240 kW; zero reserve; no terminal floor or shared station capacity; original flat tariff and start fee 5; set covering. Physical admission is rerun once per case, not once per arm. Finite-pool fleet proof, original pricing certificate, target attainment and dispatch validity remain separate. Duplicate passenger assignments are allowed by this master. No new route generation or full-model integer proof is claimed.

Every preparation/trial allocation uses `default_partition`, 8 CPUs, 32 GiB, two hours, and excludes `scaglione-compute-01`. Historical peak process MaxRSS is 2.60–5.98 GiB, so 32 GiB leaves over five times the measured maximum for preparation/indexing and alternate solver behavior. All 25 trials are independently eligible after their own preparation, with no arbitrary throttle or all-case barrier. Requeue uses job/restart-specific output directories; Gurobi trees are not resumed. Held historical work and V2G jobs were untouched. The remote resource policy was read before submission.

## Tests, source and monitoring

- [Manifest and immutable source pins](manifest.json), [all job commands/IDs](jobs.json), [effective scheduler/resource/dependency verification](submission_verification.json).
- [Local tests](local_tests.log) and [native tests](native_tests.log): **11 pass**, including randomized exhaustive dominance and bounds-aware certificates, retained optimum ties, per-case launch dependencies, repeat-launch idempotence and uncertain-submit refusal. Native-free NPZ identity/start smoke also passed; all five compute jobs passed tiny licensed fleet/charging LP certificate smokes.
- Isolated code branch `codex/mip-structure-20260922`, worktree `/tmp/evsp-mip-structure-20260922`, commit `94eabb25b5f1da9b24708320eaaec15a760c5f0e`. [Core](code/core.py), [runner](code/runner.py), [launch](code/launch.py), [tests](code/test_core.py). Native physical-gate code remains pinned at `871d057e1067411f09581e37d78f7c1ca43f68bb`; Gurobi 12.0.3 is required.
- Remote root: `/home/nc437/ladder-lite/mip_structure_20260922`. Gate `prepared/<case>.json` authenticates each preparation's `COMPLETE.json`, which binds the compact matrix/metadata files. Attempts live under `results/<case>/prepare/<job>_r<restart>` and `results/<case>/<arm>/<job>_r<restart>`.
- [Scoped snapshot](snapshot.py) checks the original jobs and every explicit recovery/diagnostic ID, with accounting fallback for completed jobs purged from scontrol. [Collector](collect.py) fetches completed small receipts, complete Gurobi logs and compressed dual certificates with SHA-256 checks; it excludes whole route journals and compact matrix arrays. Preserve every attempt and investigate failed preparation before changing dependencies. Do not release historical holds.

The launcher was submitted only after parent review and explicit campaign approval. No Google Doc or Slides changes were made by this task.

## Immediate verified preparation results

Four preparations have completed and reproduced their original matrices/ordered route identities. All scans finished, all four have zero identical incidence columns, zero redundant covering rows and one connected component. These are same-pool diagnostics; no rows or columns were removed from the running parameter trials. This historical four-case table is preserved; the completed C3 v3 result appears below.

| Pool | Safe dominated columns | Fleet certified LB / witness UB | Fleet safe zero fixes | Charging certified LB / witness UB | Charging safe zero fixes |
|---|---:|---|---:|---|---:|
| c1_k08_fresh | 2,575 / 39,940 (6.447%) | 8.00000000 / 9 | 0 | 361.69328130 / 426.82400006 | 425 |
| c1_k15_fresh | 1,298 / 79,611 (1.630%) | 14.99999991 / 18 | 0 | 670.75003505 / 1163.62400006 | 0 |
| c1_k15_sequential | 1,442 / 130,468 (1.105%) | 14.99999989 / 15 | 8,458 | 717.37332424 / 1063.80800007 | 0 |
| c4_k08_fresh | 2,748 / 39,668 (6.927%) | 8.00000000 / 9 | 0 | 335.82061950 / 411.24000005 | 18 |

The certificate includes bound contributions from all 0≤x≤1 variables. Its rational lower bound is conservative; UB is an upward-rounded cost of a validated saved same-pool incumbent. Charging caps are 9, 18, 15 and 9 respectively in the table order. A column fixed to zero cannot appear in any solution matching or improving the validated incumbent; all such solutions are preserved. these diagnostics are not new MIP solves or full-routing-model pricing certificates. Fleet screening finds no removable column in the three fresh pools with open LP-to-incumbent gaps, and 8,458 (6.483%) in the sequential pool whose saved fleet is already 15. Charging screening excludes 425 C1k8 and 18 C4k8 columns, none in the two k15 pools.

[Editable diagnostic table](preparation_results.csv) · [Exact fields and matrix identities](preparation_results.json) · [Authenticated output/log/certificate collection](collections/20260922T053806Z/receipt.json).

Historical first-preparation state: C3 preparation729457 failed the initial nonnegative guard. Diagnostic729731 subsequently completed; its exact findings and isolated recoveries are recorded below.

## Historical C3 signed-cost diagnosis and v2 recovery (superseded by v3)

Diagnostic job **729731 completed** after scanning the authenticated source journal on a compute node: 38,130 raw/unique columns, with **32 negative charging coefficients** ranging from **−2.9103830456733704e−11 to −1.4551915228366852e−11**. Representative expanded-grid route totals are 99999.99999999997 and 99999.99999999999; their charging-stop lists are empty and continuous-realized total cost is 100000.0. This is one/two-ULP floating residue in the saved totals, not meaningful negative electricity or start fees. All original signed coefficients are retained exactly, without clamping.

The first failed preparation729457 and all initial source paths remain immutable. New `code_v2/` and [manifest_v2.json](manifest_v2.json), commit **41b4383c5e4d32ba462a1ff7c82a75af9647d6c4**, permit signed costs and retain every negative-cost victim in the safe-dominance diagnostic. Eleven v2 tests pass locally and natively, including exhaustive negative-cost objective preservation. Existing dual certificates already include variable-bound terms and supported negative costs.

Replacement preparation **729775** was accepted with the original8CPU/32G/2h settings and reserved-node exclusion. Only pending trials **729458–729462** were changed to depend on `afterok:729775`; no running solver or other dependency changed. Initial replacement state is a normal Priority wait. The original 25 trial IDs and trial source code stay unchanged. The pending C3 result is not included in the four-case diagnostic table above.

[Exact signed-cost diagnosis](cost_diagnosis.json) · [Recovery commands and before/after dependencies](c3_recovery.json) · [Native v2 tests](native_tests_v2.log) · [Recovery implementation](code_v2/core.py). These v2 receipts are historical; use latest_recovery.json for the active v3 preparation and trial IDs.


## Final implementation recovery and audit trail

Original and v2 code/manifests remain immutable. Latest code is **74906efe7d8787c82795d6dd80a640a56a594beb**, on isolated branch `codex/mip-structure-20260922`; [v3 source](code_v3/runner.py), [local 13-test receipt](local_tests_v3.log), [native 13-test receipt](native_tests_v3.log), and [upload hash verification](upload_receipt_v3.json) are preserved. New tests count one fetch per NPZ member and validate the exact integral witness cost independently of a tolerance-fractional solver objective.

C3 v2 preparation729775 preserved all signed coefficients but failed a second, unrelated guard: raw Gurobi stage2 ObjVal is 676.2159718853597, whereas independently summing its selected integral routes gives 676.2160000000149, agreeing with the original `variable_route_cost`. The 0.0000281146552 discrepancy is recorded rather than used as a physical incumbent. V3 compares against that exact selected-route witness; the conservative certificate upper bound is still rounded upward from actual selected coefficients. No coefficients, routes, physical checks or solver tolerances change.

The old trial loader repeatedly accessed `NpzFile['indices']` within the column loop, decompressing the full array for each column. V3 materializes indices, offsets and costs once. [Compute benchmark730235](loader_benchmark_v3.json) decoded the actual largest matrix (130,468 columns / 3,541,122 nonzeros) in **1.258 seconds**, with one fetch per member, 197,360 KiB process MaxRSS, and exact matrix identity match. New phase receipts distinguish loading, model building and optimization.

The guarded [stop receipt](loader_failure_stop.json) records that only five running sequential loading attempts and five pending C3 attempts were canceled. Every trial with a Gurobi log was protected. Canceled loading attempts used 1,126–1,127 elapsed seconds each, totaling **45,056 allocated CPU-seconds** (12.516 allocated CPU-hours); this is allocation accounting, not measured process CPU time. Pending C3 attempts used no allocation. Failed preparation, canceled execution and Slurm logs are archived with [hashes](preserved_failed_loading_attempts/hashes.json). Healthy original trials retain their original loading overhead in results; optimizer-only timing can be compared, while total lifecycle timing must explicitly account for the loader revision.

[Recovery launcher](recover_v3.py) records intents and job receipts and refuses uncertain retries. It authenticates existing preparation gates for the sequential replacements and preserves the real `afterok:730241` dependency for all five C3 trials. [Collector](collect.py) scans every attempt directory for completed artifacts, while failed-attempt logs remain in the preserved archive; [snapshot](snapshot.py) includes the recovery IDs and keeps timestamped evidence. Historical receipts are not overwritten.

## Eight completed endpoints at the partial snapshot

[Editable CSV with matrix/result/log hashes](endpoint_results_partial.csv), [JSON](endpoint_results_partial.json), and [source collection](collections/20260922T055929Z/receipt.json). All eight have finite-pool fleet proof. Optimize time excludes artifact loading/building and any original saved-start acquisition.

| Pool | Arm | Fleet / bound | Optimizer seconds | Loading seconds |
|---|---|---:|---:|---:|
|c1_k08_fresh|focus2|9 / 9|849.588|226.403|
|c1_k08_fresh|presparsify1|9 / 9|984.828|224.657|
|c1_k15_sequential|strong_start|15 / 15|11.150|1.362|
|c4_k08_fresh|default|9 / 9|461.957|215.013|
|c4_k08_fresh|focus1|9 / 9|700.653|205.771|
|c4_k08_fresh|focus2|9 / 9|511.809|205.843|
|c4_k08_fresh|presparsify1|9 / 9|756.059|218.278|
|c4_k08_fresh|strong_start|9 / 9|601.274|194.841|


## Latest handoff snapshot: 06:01:55 UTC

All five preparations completed; all five C3 v3 trials are RUNNING, and the snapshot verifies their correct afterok dependency and unchanged resources/settings. Eleven trial endpoints completed and fourteen remain active at this snapshot. [Latest eleven-endpoint table](endpoint_results_latest.csv), [all five preparation diagnostics](preparation_results_latest.csv), [hashed collection](collections/20260922T060155Z/receipt.json) and [scheduler/phase receipt](snapshots/20260922T060155Z.json) are the handoff evidence. The earlier eight-endpoint receipt is preserved.

Sequential C1 k15 now also proves fleet15 using the ordinary 87-bus greedy start: default **104.168s**, MIPFocus1 **105.088s**, and MIPFocus2 **80.752s**. Its saved15-bus offline start proves in11.150s. PreSparsify1 is still active. This single-seed, different-node pilot does not establish a robust speedup.

C3 v3 authenticated the original302×38130 matrix with896915nonzeros, preserved32 signed residue coefficients, and validated its same-pool strong start. Its complete structural scan finds0identical columns,0redundant rows,997safe dominated columns and1component. Fleet certified bound14.99999992 versus18-bus witness and charging bound447.02888632 versus upward-rounded integral witness676.21600005 certify **zero safe zero fixes** for each objective. The charging fleet cap is18. These diagnostics do not modify any trial matrix.

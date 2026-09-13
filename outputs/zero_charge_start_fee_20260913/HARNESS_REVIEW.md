# Launch-harness review against `666cd839`

Reviewed `prepare.py`, `launch.py`, `worker.sh`, `tooling/campaign.py`, and `tooling/test_fee_campaign.py` without changing them.

## Verdict

The harness is compatible with unified CG/MIP commit `666cd839ab3923071b1a571ef349b70f61a82fa9` and is ready for the planned native cache/CG/MIP smoke. The 36-run construction is 18 matched cases times two single-arm allocations (`fee0`, `fee5`). Every allocation is independent, requests 8 CPUs/96 GiB/4 hours, excludes `scaglione-compute-01`, disables requeue, and has no array throttle below the policy default; all 36 may run concurrently.

Launch should remain conditional on the native smoke confirming an actual large cache loads and reprices under fee 0 without changing its bytes. This is the remaining platform/cache-format validation, not a harness defect.

## Fee and command identity

Both CG and MIP receive `--charge-start-cost` directly from the same manifest arm. Zero is converted with `str(0.0)` and is never selected through a truthiness fallback. The command-difference test confirms fee 0 and fee 5 commands differ only at this value. The source code records the fee in CG status/provenance/journal records and MIP physics/provenance; the MIP refuses an explicit fee that differs from its source CG status and refuses fee-0 journals with missing legacy-fee metadata.

The two-stage MIP invocation is the pinned 871 implementation integrated into `666cd839`: `--two-stage --cover --timelimit 3600 --stage1-timelimit 1800`. Its stage-two constraint remains `sum(a) <= stage1_buses`, including the validated stage-one fallback. There is no stale separate MIP checkout or commit in the manifest: `code` and `mip_code`, and `cg_commit` and `mip_commit`, are intentionally identical.

## Cache and inherited-pool integrity

Preparation freezes the cache pickle hash, cache-manifest hash, and the manifest identity's 40-character source commit. The worker authenticates both files before execution. CG passes `--event-network-cache-mode require` and `--event-network-cache-source-commit` for every case. In `666cd839`, this permits a historical cache only when the supplied source commit matches the cache manifest and `git_commit` is the sole identity difference. Instance, tariff, reference/deadhead, SOC/time/charger physics, arc mode, pickle hash, object type, and network metrics must still match. Fee-only repricing is then in memory and records the previous/current fee and repriced-arc count.

`prepare.py` does not itself compare every cache-manifest physics field with the case before freezing it. This is acceptable because the production cache loader performs the complete comparison before any pool work and fails closed. The smoke should check `network_metrics.cache_hit == true`, `network_metrics.charge_start_cost == 0`, `cache_charge_start_reprice.current == 0`, a positive `repriced_arcs`, and unchanged pre/post pickle and manifest hashes.

Parent status, copied descriptor, frozen journal, parent CSV, and their hashes are authenticated before CG. Preparation requires a certified parent with a positive final iteration and zero artificials. Same-k inheritance is supported: the exact pricer translates ordered stable trip IDs and calls target-graph fixed-sequence replay, preserving the coverage pattern while recomputing charging timing, activity count, energy, and cost under the destination fee. C1 k15's documented k14 fallback is the sole possible previous-k exception. No parent costs, duals, bases, or certificates are reused as destination-fee results.

## Process bounds and result integrity

CG has an internal cumulative 7200-second budget and a 7380-second process watchdog. MIP gets 3600 solver seconds and a 4200-second watchdog, leaving 600 seconds for source hashing, pool replay, and final physical validation. The 4-hour allocation leaves 2820 seconds beyond both watchdog maxima for authentication, Gurobi preflight, and completion hashing. A scheduler-end guard requires the full 11,580-second CG+MIP watchdog budget before the only arm begins.

Every process directory and initial `execution.json` is created exclusively; final execution metadata records timing, return code, watchdog state, resources, environment, and child resource usage. Attempts are exclusive by job/restart token, and `--no-requeue` is set. CG may feed MIP only after process success, a readable final status with positive iteration and zero artificials, and a present journal. The completed status and journal hashes are passed through `EVSP_MIP_EXPECTED_RESULT_SHA256` and `EVSP_MIP_EXPECTED_JOURNAL_SHA256`; MIP rechecks both before and after loading. `EVSP_EXPECTED_COMMIT` and detached-check enforcement bind the MIP to unified commit `666cd839`.

An uncertified but valid finite CG pool may proceed to MIP, with its certificate state recorded separately. Cache/import censoring that reaches the CG wall limit before an iteration or with artificials is classified as a measured skipped-MIP outcome; other process or gate failures make the allocation fail. A MIP process failure is not mislabeled as a successful measurement: the arm may say `finished`, but `arm_measurement_completed` returns false and the pair becomes `finished_with_failures`. Collector `process_success` remains separate from pricing, finite-pool proof, physical validation, and GIRO attainment.

## Charging decomposition

The collector derives charging starts from stops, not tariff blocks. Its two-block fixture correctly charges one activity fee. It separately reports expanded-grid and continuous electricity costs, start-fee totals, energy, terminal energy, and total charging costs. Zero remains zero. It also reports reconstruction error and `cost_components_reconcile`; missing route detail yields `available=false` instead of a fabricated zero.

The metric scope is accurately labeled as selected routes before duplicate-trip removal and does not infer shared charger capacity. For a covering MIP, consumers must retain `overcovered_trips`, `duplicate_trip_removal_validated`, and physical replay fields when interpreting these totals. The native smoke should require `cost_components_reconcile=true` and confirm the emitted CG/MIP fee fields equal the manifest treatment before the 36-run launch.

## Local checks

- Five harness unit tests passed, including fee-zero preservation and per-activity rather than per-block fees.
- All four Python harness files compile.
- `worker.sh` passes `bash -n`.

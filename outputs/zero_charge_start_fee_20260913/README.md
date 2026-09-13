# Charging-start fee: zero versus five

**Current results, 13 September, 02:35 EDT:** [29 of 36MIPs complete, all target matches](status_20260913T063519Z/README.md). The [completed GIRO comparison](status_20260913T053436Z/README.md) still shows electricity savings with no start fee and equal-or-greater returning energy at all three peaks.

**Launched 13 September at 01:08 EDT: 36 baseline CG→MIP jobs and 12 GIRO comparison jobs, all on default.** The user requested an isolated branch and new runs removing the five-unit cost per charging activity. Existing running experiments remain unchanged. The current submission ledgers are the authority for launched jobs.

The earlier five-duty charging comparison counted **52 GIRO charging starts**, versus **15–23 in the joint optimized schedules** across the three tariffs. Charging-start penalties therefore contributed 260 to the GIRO total and 75–115 to the joint totals. Those were modeled penalties, not documented operator invoices. That earlier comparison also returned different amounts of battery energy, so it does not establish savings at equal terminal inventory. [Original audited table](../meeting_20260910/presentation_manager/charging_audit/charging_three_baselines.csv).

## Chain experiment

| Setting | Control | New treatment |
|---|---|---|
| Cost per charging start | 5 | 0 |
| CG route objective | 100,000 + electricity + 5 × starts | 100,000 + electricity |
| Inputs | Six chains at k=5,10,15 | Identical 18 inputs |
| Initial sequences | Frozen completed target pools; C1k15 uses the frozen k14 pool | Identical saved sequences, replayed with the new fee |
| Final MIP | Minimize buses, then minimize charging with fleet≤validated first-stage incumbent | Same |
| Budgets | 2h CG +1h MIP, including≤30min first MIP stage | Same |

This is 36 independent CG→MIP runs, all eligible together on default. Each requests 8 CPUs and 96 GiB for a four-hour allocation, excludes `scaglione-compute-01`, and writes to a unique job/restart directory. No earlier k must be solved first. Required saved inputs are frozen and hashed before launch; original graph caches and route journals remain immutable.

Both treatments use the same new code revision with explicit fee support. Other settings stay at covering, 240 kWh batteries,240 kW charging,2.5 kWh / 5-minute event graph, 30 columns per iteration, reduced-cost tolerance 1e-4, flat prices, indexed unlimited inheritance, and the existing LP setup. These baseline chains do not impose shared-station capacity or a terminal-SOC floor. They are separate from the fair-terminal GIRO cost experiment.

The tests must establish that zero reaches pricing, saved-route replay, cache handling, cost reconstruction and both MIP stages. Changing only the final MIP coefficient would not test the requested algorithm. Saved fee 5 costs are not treated as fee 0 costs, and a certificate from fee 5 is not transferred to fee 0.

## GIRO charging comparison

A separate matched cohort retains the existing five duties / 62 trips, 240 kWh batteries,350 kW charging scenario and peaks08/12/18. All optimized schedules must return at least the original GIRO aggregate 280.7833253 kWh, checked in both expanded-grid and continuous replay. This retains the existing physical scenario to isolate the fee change; it is not the 240 kW six-chain experiment.

For each peak and each fee, generate a fresh fixed-duty charging frontier. Then use a common union of both frontiers and the same saved CG pool, repriced under the destination fee, for the joint MIP. Each joint solve waits for both fee frontiers at its peak. Fixed-duty optimization preserves each bus's ordered passenger trips; joint optimization can change their grouping within the common available pool. Joint cost proofs concern that finite pool. The old CG certificate does not certify pricing under a new terminal-energy dual or fee.

The three reported schedules are:

1. Original GIRO charging retained and repriced under the tariff.
2. Original GIRO trip sequences with charging optimized.
3. Joint routes and charging selected from the common augmented pool.

## What the result table will show

- Buses and whether the target was attained; finite-pool fleet bound/proof.
- Electricity cost alone, charge-start count and its modeled fee subtotal.
- Energy charged and terminal energy, including expanded-grid and continuous replay values.
- CG import, pricing and LP times; weighted objective and exact stopping reason/certificate.
- MIP charging objective, gap, both-stage runtimes and physical validation.
- Source input/pool/cache/output hashes, code revision, Slurm resources and dependencies.

The GIRO electricity cost remains an interval where its within-window power trace was not observed. Fee0 means we assign no penalty to connecting to charge; it does not mean electricity is free. Removing the fee may allow more fragmented charging and more equivalent-cost paths, so runtime could improve or worsen.

These are 18 selected matched chain inputs, not 36 independent random samples. Initial pools were produced with fee 5; both arms deliberately share them. This measures reoptimization of existing solutions, not performance from a fresh singleton-only start. Preserve that distinction when reporting results.

## Implementation and audit

Unified CG/MIP execution commit: `06b5cb86d6c24df0ec0a5ca7189fa9552f527dd0`, pushed on branch `codex/zero-charge-start-fee-20260913`. See [code audit](CODE_AUDIT.md) and [independent launch review](HARNESS_REVIEW.md). The native cache check passed: 211,179,084 charge-entry arcs were repriced from fee 5 to fee 0, with the source pickle and manifest unchanged. Both short CG runs produced usable pools. The first native MIP attempts exposed a progress-observer API mismatch before optimization; the observer was restored from the pinned MIP revision. The source amendment and original manifest are preserved. The corrected MIP validation passed using those saved smoke pools.

The original GIRO jobs `113589`–`113600` passed scheduler-setting checks but their frontiers then failed at launcher startup. This attempt is superseded by `114100`–`114111`, as explained below. [Historical launch check](giro/launch_verified.json).

Native validation passed under commit `06b5cb8`: both fees completed MIP optimization, individual-route physical replay passed, and electricity plus the charge-start fees independently reconciled to the replayed route costs. These short, truncated validation runs are software checks, not research outcomes. [Validation evidence](native_validation_pass.json).

Baseline job IDs are `114033`–`114068`. All 36 are independent and were submitted together, with no previous-k dependencies. Each has a two-hour CG budget and a one-hour MIP budget. The initial MIP observer failure is retained under validation job `110952`; corrected MIP-only validation job `114000` reused the successful CG smoke pools. [Submission ledger](jobs.json), [frozen experiment settings](manifest.json).

The first six GIRO frontiers exited before Python started because Slurm stages worker scripts under its spool directory. Their six dependent MIPs were cancelled before starting. The external replacement worker uses an explicit, unchanged solver-code path; original plan/data/physics/fee settings are preserved. Current GIRO jobs are `114100`–`114111`, with six frontiers running and six MIPs waiting for both matching-tariff frontiers as of 01:14 EDT. Original jobs and logs remain in [initial submission](giro/jobs.initial_failed.json); [current submission](giro/jobs.json) and [launcher diagnosis](giro/worker_path_audit.md) record the repair.

## Early results — 13 September 01:12 EDT

Eight campaign MIPs have completed, each attaining five buses with a finite-pool fleet proof and individual-route physical replay. Two inputs have completed both fees:

| Chain, target k=5 | Buses: fee 5 → fee 0 | Charging starts: fee 5 → fee 0 | Electricity cost: fee 5 → fee 0 | Returning energy kWh: fee 5 → fee 0 |
|---|---:|---:|---:|---:|
| 3 | 5 → 5 | 7 → 27 | 91.565 → 89.106 | 41.953 → 17.166 |
| 6 | 5 → 5 | 7 → 34 | 100.932 → 98.140 | 36.808 → 10.863 |

Removing the fee changes charging fragmentation substantially. The lower electricity totals are not evidence of savings at equal returning energy: these baseline runs have no terminal floor. The separately launched GIRO study supplies that control. Cost components independently reconcile in all eight completed MIPs. [Complete paired table](early_pairs.csv), [raw early results](early_results.json). Software-smoke validation results are excluded.

The current Google Doc status was updated with editable tables and this scope distinction. The prior material below its historical marker is byte-for-byte unchanged in the export; figure tabs and Slides were not edited. [Document verification](doc_verification.json).

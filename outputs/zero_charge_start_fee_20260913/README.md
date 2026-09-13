# Charging-start fee: zero versus five

**Preparing; not yet submitted.** The user requested an isolated branch and new runs removing the five-unit cost per charging activity. Existing running experiments remain unchanged. Submission records, once written, are the authority for launched jobs.

The earlier five-duty charging comparison counted **52 GIRO charging starts**, versus **15–23 in the joint optimized schedules** across the three tariffs. Charging-start penalties therefore contributed 260 to the GIRO total and 75–115 to the joint totals. Those were modeled penalties, not documented operator invoices. That earlier comparison also returned different amounts of battery energy, so it does not establish savings at equal terminal inventory. [Original audited table](../meeting_20260910/presentation_manager/charging_audit/charging_three_baselines.csv).

## Chain experiment

| Setting | Control | New treatment |
|---|---|---|
| Cost per charging start | 5 | 0 |
| CG route objective | 100,000 + electricity + 5 × starts | 100,000 + electricity |
| Inputs | Six chains at k=5,10,15 | Identical 18 inputs |
| Initial sequences | Frozen completed target pools; C1k15 may use k14 if still running | Identical saved sequences, replayed with the new fee |
| Final MIP | Minimize buses, then minimize charging with fleet≤validated first-stage incumbent | Same |
| Budgets | 2h CG +1h MIP, including≤30min first MIP stage | Same |

This is 36 independent CG→MIP runs, all eligible together on default. Each requests 8 CPUs and 96 GiB for a four-hour allocation, excludes `scaglione-compute-01`, and writes to a unique job/restart directory. No earlier k must be solved first. Required saved inputs are frozen and hashed before launch; original graph caches and route journals remain immutable.

Both treatments use the same new code revision with explicit fee support. Other settings stay at covering, 240 kWh batteries,240 kW charging,2.5 kWh / 5-minute event graph, 30 columns per iteration, reduced-cost tolerance 1e-4, flat prices, indexed unlimited inheritance, and the existing LP setup. These baseline chains do not impose shared-station capacity or a terminal-SOC floor. They are separate from the fair-terminal GIRO cost experiment.

The tests must establish that zero reaches pricing, saved-route replay, cache handling, cost reconstruction and both MIP stages. Changing only the final MIP coefficient would not test the requested algorithm. Saved fee5 costs are not treated as fee0 costs, and a certificate from fee5 is not transferred to fee0.

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

These are 18 selected matched chain inputs, not 36 independent random samples. Initial pools were produced with fee5; both arms deliberately share them. This measures reoptimization of existing solutions, not performance from a fresh singleton-only start. Preserve that distinction when reporting results.

## Implementation and audit

Unified CG/MIP execution commit: `666cd839ab3923071b1a571ef349b70f61a82fa9`, pushed on branch `codex/zero-charge-start-fee-20260913`. See [code audit](CODE_AUDIT.md) and [independent launch review](HARNESS_REVIEW.md). Native cluster validation will be recorded separately before submission.

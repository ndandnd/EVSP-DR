# Advisor follow-up — 16 September 2026

Only action 1 was submitted. Actions 2–4 create no new cluster work. The hourly monitor and runbooks now require user confirmation after a cluster-load report before any additional submission or partition move.

| Action / finding | Result and evidence |
|---|---|
| 1 — F6 | Six k5 jobs submitted and observed running: peak08 fresh341682/fixed341683; peak12 fresh341684/fixed341685; peak18 fresh341686/fixed341687. Original schedules are repriced without another solve. [Campaign and validation](f6_k5/README.md). Stricter-physics savings remain **unresolved** until results pass replay. |
| 2 — F6/F7 | Prepared72 MIP-only comparisons:24 saved fresh pools ×3 seeds,3h fleet search, hardfleetcap=k, with remaining30min minimum for charging. Charging comparisons require physically validated fleet exactly k. These fresh models already have fleet<=k; the follow-up changes search effort and seed. **Not submitted.** [Plan](mip_followup/README.md). |
| 3 — F4 | Six single-factor arms prepared: baseline, PARX60only, reserve15%only, battery236.44only, battery239.01only, groupsegregationonly. Each reoptimizes charging of the original C5k31 pool's ordered trip sequences and imports verified survivors. Timeout is not infeasibility. **Not submitted.** A bounded replay-cost pilot must precede full254,068-column processing. [Design and tests](single_factor/README.md). |
| 4 — F8 / reliability | Native atomic persistence and ordinary resume are **verified** by bounded tests. An unconditional recovery guarantee is **refuted**: latest-corrupt-status fallback, shutdown final LP, and lost uncheckpointed runtime accounting need attention. Current50h allocation is48h scientificCG; no live change made. [Checkpoint audit](checkpoints/README.md). |
| 5 — F1/F3 | Counts independently recomputed and §1a updated.93/102 have numerical integer-fleet bound=k;9have k−1. Original51/102 match the bound, leaving51open overall. With26audited longer searches:70targetmatches,67boundmatches,35open. The interpretation “only9open overall” is **refuted**. [Reproduction script and hashes](proof_counts/counts.json), [§1a draft](section_1a.html). |

## What the k5 test holds fixed

The same62-trip cohort and peak08/12/18 tariffs as the prior F6 study;240kWh battery,350kW maximum charging, fee0, fleet<=5 and aggregate ending energy>=280.7833253kWh. The new constraints are36kWh minimumSOC and3minutes of positive charging. Controllable power spreads required energy across the retained charging window; the physical interpretation is explicit in the campaign record. Shared charger capacity is still absent. This isolates the proposed stricter test from the unrelated240kW baseline ladder; it is not a full-GIRO-compliance claim.

All five original modeled schedules pass the new reserve and duration requirements: minimumSOC45.5146193kWh and shortest positive charging window4minutes. Original invoice intervals remain necessary because observed within-window power is unknown. Three-arm saving claims wait for completed optimized schedules with matching fleet and physical replay.

## Cluster load and placement

At22:46:51UTC, before the six new jobs, this account had88running and77pending default-partition jobs. The five allowed Scaglione CPU nodes had52idleCPU slots in total; the GPU node scaglione-compute-01 remains excluded. Empty scaglione-partition squeue output does not imply those nodes are idle: default-partition jobs can occupy them.

The full-CG job requests128G (131072MiB), while the allowed Scaglione CPU nodes advertise128350MB each. It therefore cannot move unchanged to those nodes. Do not reduce its memory request without measured justification. No partition move or new follow-up was submitted. [Load snapshot](cluster_load.json).

## Documentation

The current Google Doc §1a now distinguishes the original search budget from longer searches, numerical event-model certificates from exact proofs, and the nine below-GIRO-bound cases from all open gaps. Figure tabs and Slides are preserved. The source and export were checked; see [verification](doc_verification.json).

These counts concern the frozen102 original cases and26 audited longer searches, not the newly running seed campaigns. Baseline physical assumptions still omit reserve, station capacity and return-SOC requirements. Preserve those qualifications when presenting optimality.

## First results, 23:00 UTC

F6 fixed-duty peak08 and peak18 finished with five physically validated buses and no repeated trips. Their continuous modeled costs are157.9646 and107.1887, minimumSOC37.0 and37.5kWh, minimum positive charging duration3minutes, and ending energy281.17 and282.90kWh. The three freshCG results remain pending. The peak12 fixed job is scheduler-running but worker startup was not yet observed. Joint-versus-fixed savings remain unresolved. [Seven-campaign collection](../monitor/20260916T230004Z/snapshot.json).

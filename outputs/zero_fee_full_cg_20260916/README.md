# Full CG with zero start fee and matched return energy

**Launched 15 September, late evening.** Array **275432**, three independent cases, all observed RUNNING. Native validation275221 passed. [Current Google Doc, question3](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.79m3d3x4h45m).

The earlier fixed-duty/joint ties answer only a saved-pool question. Those columns were generated under different conditions. They do not answer whether full CG under zero fee and matched return energy discovers better routes. Keep the earlier values, but never describe their tie as evidence of full joint/GIRO optimality.

| New experiment | Setting |
|---|---|
| Input | Same62 trips / five GIRO duties as the saved-pool comparison |
| Tariffs | Frozen08:00,12:00,18:00 peaks, one independent run each |
| Energy/power | 240kWh initial and capacity;350kW; no reserve/shared-capacity row |
| Ending energy | Sum across selected buses >=280.7833253kWh, conservative post-return energy |
| Fleet/trips | At most5 buses; set covering |
| CG objective | 100000 per bus + electricity; zero start fee |
| Start | Fresh direct singletons plus explicit Phase I; no inherited or GIRO columns |
| CG budget | Four hours after graph construction; stop earlier on a valid pricing certificate |
| MIP | Up to30min fleet minimization, remaining one-hour budget electricity minimization with fleet <= first-stage incumbent |
| Resources | default_partition,8CPU,48GB,8h allocation including graph build; all3 parallel; compute-01 excluded |
| Retry | New attempt directory; prior columns/results retained. No Gurobi tree recovery |

## What changed in the algorithm

The return-energy row now supplies a dual value to pricing:

`reduced cost = route cost - trip duals - terminal dual * return energy - fleet dual`.

The graph retains cost/return-energy alternatives before collapsing terminal arcs. The pool retains distinct energy alternatives even when trips are identical. Every admitted route is physically replayed, and its return-energy coefficient is checked against pricing. Phase-I artificial variables are excluded from the MIP.

The enumeration tests also found a lazy-pricing bug: combined-cost pricing with a nonzero fleet dual incorrectly removed the100000 bus cost on source arcs. This branch fixes that path. These tests do not establish that earlier chain runs exercised it.

Execution: `c210187bb50a2eac2022178ea21aa39d7ff6b9b1`, branch `codex/zero-fee-terminal-cg`, pushed to GitHub. Baselinea0e0bb. Production baseline files are unchanged except the isolated branch's one-line lazy-pricing fix. New experiment driver uses the same event graph internally; no old graph cache or old pricing certificate is imported.

Validation: explicit and lazy pricing compared with enumeration of all paths for four objectives and four terminal-dual values; independent physical energy checks; full enumerated LP and integer optimum compared with CG/MIP; all12 existing event-pricing tests. Passed locally and on Unicorn. The full62-trip runs are experimental results pending physical/output checks, not already verified successes.

## Interpretation and records

A weighted-CG certificate concerns the declared conservative event graph and objective. It does not prove the final integer charging solution globally optimal. The MIP bounds prove only statements about the generated finite pool. Continuous replay costs and grid costs remain separate. Shared charger capacity and65% return SOC are not modeled here; this deliberately matches the prior charging-comparison cohort.

Manifest.json freezes input/reference/tariff hashes, execution commit, physics, objectives and resources. launch.json records jobs and initial observed scheduler state. Source logs and attempt artifacts live under `/home/nc437/ladder-lite/zero_fee_full_cg_20260916/`; final columns have hashes in summary.json. The main collector now includes this campaign separately. Do not classify a still-building graph as a failed CG, or a finite-pool proof as full-model optimality.

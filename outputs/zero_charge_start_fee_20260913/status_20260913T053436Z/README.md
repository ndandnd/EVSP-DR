# Charging-start-fee update — 13 September, 01:41 EDT

Chain results were collected at 01:34 EDT. A targeted GIRO follow-up at 01:41 caught results released during collection; the original snapshot is preserved separately from the reconciled view.

All six chains match five buses at k=5 under both fees. Zero fee also matches ten buses in chain 6 at k=10; its control was pending in the snapshot. Thirteen MIPs are complete, all with target attainment, pool fleet proofs and individual-route replay. Twenty-nine of 36 CG runs have pricing certificates.

[All 18 paired chain inputs](paired_results.csv). The lower chain electricity totals have unequal returning energy, so they are not savings at equal terminal inventory.

## GIRO: electricity-only cost, fee zero

| Price peak | Original charging | Fixed duties, charging optimized | Joint route pool |
|---|---:|---:|---:|
| 08:00 | 230.287–230.981 | 128.293 | 128.293 |
| 12:00 | 289.594–290.597 | 164.231 | 164.231 |
| 18:00 | 223.447–223.723 | 95.285 | 95.285 |

All schedules use five buses. This cohort has 62 trips, 240 kWh batteries and 350 kW charging, without shared-station capacity constraints. Optimized returning energy is 281.17, 281.17 and 282.90 kWh, versus GIRO's 280.7833 kWh. Optimized charging starts are 46, 43 and 42; original GIRO has 52. The original cost interval reflects its unknown within-window power trace.

The 43–57% electricity-cost improvement survives removing the start fee and respecting the original aggregate returning energy. In these finite pools, joint routing adds no cost improvement beyond charging optimization on fixed duties. This is an attained-schedule comparison, not a continuous-model optimum proof. Joint grid-objective gaps are at most 0.0088%; individual-route replay, all 62 trips, both energy checks and completion hashes pass.

[Both fees and all three baselines](giro_costs.csv), [independent verification](giro_validation.json), [GIRO source collection](giro_followup.json), [scheduler completion](giro_sacct.txt). All 12 GIRO allocations completed with exit 0:0. No new execution failure or confirmed preemption was found in the full collection.

Twenty of 24 controlled algorithm comparisons were finished at 01:34; analysis is pending. Those results remain separate from the fee experiment.

The Google Doc now contains editable tables above the preserved chain/history sections. Historical content below its previous marker is unchanged; both figure tabs and Slides were untouched. See doc_verification.json.

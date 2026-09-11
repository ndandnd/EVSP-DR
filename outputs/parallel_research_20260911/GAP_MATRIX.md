# Missing-experiment matrix and next independent CG grid

Generated 2026-09-11T03:08:23.416581+00:00. This is a bounded inventory of the saved local register and current collector snapshots; it is not a complete Git-history census.

## Recommended launch now

Launch the nine **fresh set-covering CG** cells below on the default partition. The six-chain partitioning campaign is already complete, and covering currently exists only for chains 1, 3, and 5. This grid completes the formulation comparison on all six random chains at k=5, 8, and 10. Keep the downstream MIPs dependent on each frozen pool; do not submit a second independent MIP before its CG pool is frozen.

Configuration held fixed: current event implementation commit `21fbecba826824c44f897feef038fcf51c532582`, 240 kWh / 240 kW, zero reserve, 2.5 kWh / 5-minute event grid, flat tariff, singleton RAW initialization, 30 reduced-cost columns per iteration, `rc_eps=1e-4`, eight-hour CG wall limit, default partition. The changed factor is the master row sense (`>=1` covering versus equality partitioning). Because the duals change, the route pool changes too; this is a full workflow comparison, not a same-pool MIP ablation.

| Cell | Trips | Input SHA-256 (prefix) | Existing partition CG | Existing partition MIP |
|---|---:|---|---:|---|
| `k05_p2` | 109 | `286ef1471874…` | 1027 it / 25.3min / 29,166 cols | 8 buses, bound 8, OPTIMAL |
| `k05_p4` | 104 | `d51b0dab0f1f…` | 771 it / 17.3min / 21,902 cols | 9 buses, bound 9, OPTIMAL |
| `k05_p6` | 79 | `607cba23d981…` | 573 it / 13.8min / 13,983 cols | 5 buses, bound 5, OPTIMAL |
| `k08_p2` | 173 | `c1dbe1dd15d2…` | 1104 it / 67.6min / 32,493 cols | 30 buses, bound 8, TIME_LIMIT |
| `k08_p4` | 188 | `b0c454201f81…` | 1921 it / 138.5min / 55,293 cols | 31 buses, bound 9, TIME_LIMIT |
| `k08_p6` | 128 | `749b356e355c…` | 741 it / 11.7min / 20,186 cols | 12 buses, bound 9, TIME_LIMIT |
| `k10_p2` | 236 | `969e49b79b45…` | 1690 it / 80.0min / 49,441 cols | 46 buses, bound 10, TIME_LIMIT |
| `k10_p4` | 220 | `c186ded0e8b4…` | 1895 it / 76.9min / 55,730 cols | 30 buses, bound 10, TIME_LIMIT |
| `k10_p6` | 192 | `b6b8abf3c83e…` | 1371 it / 40.8min / 41,255 cols | 36 buses, bound 10, TIME_LIMIT |

Full source paths, hashes, and duty-set hashes are in [`gap_matrix.json`](gap_matrix.json), copied from [`input_selection_manifest.csv`](../nested_integer_diagnosis_20260908/launch/input_selection_manifest.csv).

The scientific question is whether allowing overcoverage produces a better compatible integer pool at the same target k. The MIP report must retain: actual fleet, independent overlap lower bound, fractional route weight, CG certificate, pool bound/incumbent, number of overcovered trip incidences, duplicate-removal/physical replay status, and Stage 2 cost status.

## What is already running or complete

| Work item | Status | What remains |
|---|---|---|
| Covering rerun9, chains 1/3/5 at k=5/8/10 | CG complete; MIPs/physical audit in progress | Harvest; no resubmission. Existing k5 p1/p3 hit 5 buses proved; p5 hit 6 with 12 overcovered incidences. At k8, current saved fleet proofs are 9 for p1/p3/p5; at k10 p1/p5 are 11 proved and p3 has 11 incumbent with pool bound 10, time-limited. |
| Warm previous-k chains P3 and P5 | In progress | Wait for k7–k10 and validate inherited-pool hashes. Do not launch a third warm chain yet. |
| 42-cell random replication p7–p20 | Submitted | Harvest descriptive replication; preserve chain blocking and no optional stopping. |
| Equal-terminal-energy charging control | Submitted / partial | Harvest frontier and dependent MIP. This is the fairness control for the three-peak charging bars. |
| Capacity/depot-power pilot | Submitted / partial | Harvest all four arms, especially duty 13406; only 5/16 cells were complete at the latest snapshot. |
| Easy and heavy deterministic ladders | Submitted / partial | Harvest existing partition arrays and censored outcomes; do not relaunch equivalent partition CG. |
| Tariff peak 12/18 | Submitted / partial | Harvest existing jobs; no new tariff CG needed. |
| April regression control | Partial | Current event 240/300 sensitivity is launched; an exact old-code/grid replay remains a later, separate control. |

The existing work therefore gives a clear launch boundary: the missing independent formulation arm is covering on chains 2, 4, and 6. New warm chains, new random chains, another easy/heavy partition run, and another tariff run would duplicate questions already in flight.

## Missing-experiment matrix

The machine-readable version has the full fields and exact input IDs. The short decision list is:

- **P0 launch:** nine-cell fresh covering complement above.
- **P0 harvest:** existing covering MIPs; warm P3/P5; retain proof scope and overcoverage fields.
- **P1 harvest:** p7–p20 replication, equal-terminal-energy pilot, capacity/depot-power pilot, easy/heavy scale ladders.
- **P2 later:** easy covering control after the random covering complement; exact April-code/grid replay only if the current 240/300 sensitivity leaves the regression unresolved.

## Figures that can be made without another solver run

1. **Paired formulation matrix:** for each chain and k=5/8/10, show target, fractional route weight, pool bound/incumbent, proof status, and overcoverage for partition, fresh covering, and warm covering. Leave the nine missing covering complement cells blank until they finish.
2. **Proof-scope waterfall:** show independent overlap lower bound → certified fractional CG endpoint → frozen-pool fleet bound → incumbent → Stage 2 cost status. This separates a global event-model lower bound from a finite-pool MIP result.
3. **Scaling/censoring panel:** plot trip count against CG time and MIP status for easy, heavy, and random cells; mark time limits as censored. A fourth optional panel can add the charging bars with terminal energy shown separately.

## Interpretation guardrails

- A certified CG endpoint means no implemented event-grid route has reduced cost below tolerance for the final duals. It does not by itself certify the integer solution over routes never generated.
- A Gurobi `OPTIMAL` result on the frozen pool proves that pool MIP only. It is not a proof that the full route universe needs that many buses.
- Covering can remove duplicate coverage after the MIP, but the report must show whether the selected solution already has no duplicates and whether cross-route station capacity was checked.
- The random chains are nested within chain, so they are useful for descriptive variability and diagnosis, not 84 independent Bernoulli trials.

See [`gap_matrix.json`](gap_matrix.json) for the exact nine input IDs, SHA-256 values, status matrix, source list, and figure specifications.

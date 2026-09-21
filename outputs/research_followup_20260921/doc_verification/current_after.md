# **EVSP DR research journal**

Week of 21 September 2026\. The strongest result is that a good fractional solution can leave an inadequate integer route pool. In 24 matched cases, fresh and sequential CG both certify their LP endpoints, but their one-hour MIPs match 6 versus 24 GIRO targets. New experiments test whether integer-directed pricing closes that gap without known sequential routes.  
[Paper results preview: five figures, editable tables and source checks](https://github.com/ndandnd/EVSP-DR/blob/e07c877f0adc23bb8d64b1db1c3d104b0ceefc19/outputs/research_management_20260921/paper_results/RESULTS_PREVIEW.md). The preview counts accumulated sequential work and separates LP certificates, pool integer proofs and physical validation. [Today’s experiments and monitoring plan](https://github.com/ndandnd/EVSP-DR/blob/e07c877f0adc23bb8d64b1db1c3d104b0ceefc19/outputs/research_management_20260921/MANAGER_PLAN.md).  
 

## **21 September LP agreement and battery audit**

**Fresh versus sequential:** all 24 paired LP endpoints (six chains, k=5,8,10,15) agree within 0.0000010617 cost units. Both certify the event-grid LP at reduced-cost tolerance 0.0001. Their integer pools differ. [New per-chain figures](https://github.com/ndandnd/EVSP-DR/blob/3f2228605e0c9e1cc46140a3dbd03fd6c1f6e3ce/outputs/research_followup_20260921/chain_comparison/README.md) show accumulated sequential CG time, fresh time, integer buses above target, and magnified LP costs.

**3.93 GiB means computer memory.** Battery energy is measured in kWh. Reducing capacity and initial charge alone makes some saved schedules infeasible:

| Capacity-only scenario | Failed route occurrences | Affected saved fleets |
| ----- | ----- | ----- |
| 240 kWh control | 0 / 487 | 0 / 48 |
| 239.01 kWh throughout | 40 / 487 | 29 / 48 |
| 236.44 kWh throughout | 195 / 487 | 46 / 48 |

These are frozen-schedule sensitivity tests, not actual vehicle assignments. Deficits are at most 0.99 and 3.56 kWh; charging repair was not tested. Optimized schedules often approach zero SOC, so small rounding is not automatically harmless. [Replay, source hashes and limits](https://github.com/ndandnd/EVSP-DR/blob/3f2228605e0c9e1cc46140a3dbd03fd6c1f6e3ce/outputs/research_followup_20260921/battery_rounding/README.md).

**Duty 13309, whole day:** the same C6 k5 input provides a fee0 route sharing all 22 original trips and a fee5 route sharing 11\. Displayed charging starts are 4 / 7 / 1\. Both counterparts pass the capacity-only 239.01 kWh check; their historical zero-reserve charging physics still differ from GIRO. [Full-day figures and exact itineraries](https://github.com/ndandnd/EVSP-DR/blob/3f2228605e0c9e1cc46140a3dbd03fd6c1f6e3ce/outputs/research_followup_20260921/duty13309/README.md).

## **21 September Extending all six chains to 40**

All six baseline chains have found 32-bus schedules after longer MIPs and seed repeats. The original one-hour searches reached largest targets 26, 28, 31, 29, 26 and 28\. These are largest matches, not a claim that every smaller target matched. At k32 all six CGs stopped at four hours without a pricing certificate.

Now submitted: **48 cases, k33–40 on each chain**, using frozen duty additions and the entire preceding column pool. Graph array **661616** has 44 independent builds; identical inputs share four builds. Each CG keeps its previous-k dependency, followed by its own MIP. Budgets remain 4 h CG and 1 h two-stage MIP. Graphs receive 96 GB and up to 36 h; k32 builds took 12–15 h. At 15:57 EDT on 21 September, all 44 graph builds were running; six preempted attempts had restarted automatically. Reaching k40 is roughly a two-day pipeline under favorable scheduling, longer if preempted or queued.

This extension retains the baseline 240 kWh / 240 kW, flat tariff, 5-unit start fee, and no reserve, shared charger limit or terminal-energy floor. It is separate from the stricter k5 comparison. At full40, identical input groups are C1/C4 (948 trips), C2/C3/C6 (947) and C5 alone (946), because service-day variants were preserved. Within an identical group, fully certified LPs must have the same optimal weighted objective, though their fractional routes can differ. Time-limited LP endpoints and finite-pool integer results may differ with the inherited pool. The true full-model integer optimum is also independent of the path taken.

New: [geographic maps and editable travel table](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i), with charger/depot locations, travel minutes and energy. [Sources and remaining movement approximations](https://github.com/ndandnd/EVSP-DR/blob/bc5391af975db7de61a1df804a57883b821c2868/outputs/week_20260921/geography_map/README.md).

## 

## **21 September Eight jobs completed**

| Job | Test | Outcome |
| ----- | ----- | ----- |
| 628428 | k1 capacity, shortcut off | CG 4 h limit; pool optimum 1 bus |
| 628429 | k1 capacity, shortcut on | CG certified in 26.9 min; pool optimum 1 |
| 628436 | k2 capacity, shortcut off | CG 4 h limit; pool optimum 3 |
| 628437 | k2 capacity, shortcut on | CG 4 h limit; pool optimum 2 |
| 628438 | k3 capacity, shortcut off | CG 4 h limit; pool optimum 21 |
| 628439 | k3 capacity, shortcut on | CG 4 h limit; pool optimum 5 |
| 628314 | Strict 18E2 k15 | Graph build 4 h 17 min; zero pricing iterations |
| 628441 | C1 incumbent transfer | 8 buses, bound 8; charging optimum 447.44 in pool |

All six dependent capacity MIPs also finished. “Pool optimum” means the best fleet using only saved routes; it does not prove the full model's optimum. Only one capacity CG certified its LP. These capacity tests use 240 kWh, 240 kW, zero reserve and flat prices. Charger-count checks pass, but k2-off and k3-on retain 1 and 6 extra trip assignments. [Settings, results and logs](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/capacity_strict/README.md).

## **21 September Why the fresh integer pools are worse**

**Four fresh k8 pools require nine buses: C1, C3, C4 and C5.** Adding eight known sequential routes to each restores a proved eight-bus cover. This is a diagnostic using known solutions, separate from the new pricing pilot.

The LP spreads eight units of route weight across 80–101 fractional routes. Whole buses need a compatible combination. Of 40 sequential witness routes across five audited cases, 39 trip sets are absent from the fresh pools, 37 have positive reduced cost, and 34 were inherited. A route can help complete an integer cover without improving the fractional objective.

For C1, the eight-route witness costs 800479.960, versus LP 800383.688. The difference is **96.272 \= 72.142 in reduced costs \+ 24.130 in dual-valued extra coverage**. The LP prefers its cheaper fractional mixture, even though the witness avoids an extra integer bus. This explains the mechanism; it does not identify a unique combinatorial obstruction. [Derivation and reproducible checks](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/evidence/README.md).

**Read the Gurobi proof:** [C1 control, lines 155–156](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/evidence/k8_witness/c1_k08/control/391804_r0/gurobi.log#L155): objective 9, bound 9, gap zero. Stage 1 uses unit bus costs. [After eight added routes, line 112](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/evidence/k8_witness/c1_k08/augmented/391805_r0/gurobi.log#L112): objective 8, bound 8\. The later charging-stage time limit does not undo the fleet proof. These are numerical proofs for the specified finite pools.

## **21 September Longer MIPs and new pricing**

All twelve 12-hour fleet searches **and their charging stages are finished**. For C1–C6, plain search found **18, 17, 17, 19, 16, 18**; stronger MIP heuristics found **18, 17, 16, 17, 16, 17**. Every fleet bound is 15\. This favors generating better columns, but does not prove that a 15-bus solution is absent. [All outcomes and full log paths](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/evidence/k15_12h_summary.csv).

The pricing pilot recovered eight buses in **3/4 final MIPs**, versus nine in controls. C1's dive had already found eight, but its final MIP did not receive that incumbent. Follow-up 628441 transferred it and recovered eight in 29.6 additional minutes, with the same distinct routes. The original benchmark remains 3/4. The corrected replication has 16 allocations: four k8 cases × two seeds × control/treatment. At 16:24 EDT, four treatments had finished: C3 and C5 at both seeds recover eight buses with a pool proof, in 14.1–18.8 minutes including overhead. Twelve allocations remain running; the paired comparison is incomplete. No known sequential routes were added. Physical route replay passes; duplicate cleanup and shared capacity are not established. Six paired k15 follow-ups will launch only after the native validation gate passes. [Design, tests, completed results and full logs](https://github.com/ndandnd/EVSP-DR/blob/09ca22ae28d7e92026dcb36f402b87589f507d12/outputs/research_management_20260921/integer_columns/README.md). [Follow-up log](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/evidence/c1_followup/628441_r0/gurobi.log).

## **21 September Harder physics and duplicate service**

The capacity shortcut completed 134 versus 6 pricing iterations at k2, and 51 versus 2 at k3. The strict k15 run instead spent its budget constructing 260.7 million arcs; its 37-bus initial-pool result tells us little about the model. k16 timed out at 211.9 GiB peak memory.

The packed recovery is complete. Its label k16 is the parent prefix: this subgroup contains 256 trips from nine GIRO duties. Graph construction took 115.4 minutes within the four-hour budget; CG then completed 2,453 iterations at 3.93 GiB peak memory. Fractional route weight is 9, but pricing did not certify convergence. The saved pool provably needs 10 integer buses. Its selected routes cover every trip, with 57 extra assignments, and exceed shared charger limits at two stations. Capacity was omitted from this arm, so this is an important remaining model constraint, not a solver violation. Successors with 10 and 11 subgroup duties and two larger representation benchmarks are now submitted. [Final results, physical checks and full logs](https://github.com/ndandnd/EVSP-DR/blob/b7efdd948a8d0f20ca730fbb4da4a8e40e9d83a5/outputs/research_management_20260921/operations/README.md).

[Completed graph benchmark and replay checks](https://github.com/ndandnd/EVSP-DR/blob/4c8e4c005ec92259d520b1baa990b39ed4a851bf/outputs/week_20260921/capacity_strict/recovery/benchmark/646674_r0/result.json) — five fixed-dual checks on 26 trips: build 121.11 to 41.81 seconds, peak memory 2.06 to 0.147 GiB. This is a same-node benchmark; the larger recovery used different stopping behavior and hardware.

 [Code and experiment receipts](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/capacity_strict/README.md).

Duplicate cleanup now chooses one bus to serve each trip, checks shorter connections and reoptimizes charging. It cannot simply wait at the wrong stop. On the historical k5 solution, it preserves five buses and 62 exactly-once trips, lowers charged energy by 80.87 kWh and electricity cost by 2.58. This test retains its historical 240/350, zero-reserve physics. The optional pipeline step saves an independently checked dispatch result without overwriting the source. [Implementation, tests and validation](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/cleanup_physics/README.md).

## **21 September Corrected k5 charging comparison**

The replacement figure uses **236.44 kWh, 15% reserve, PARX 60 kW, the documented opportunity-charging taper and a three-minute minimum**. All three fleets have five buses, 62 trips and equal total ending energy. New schedules pass exactly-once, SOC and modeled charger-count checks for this five-bus cohort.

| Schedule | Electricity | Starts | Total with its fee |
| ----- | ----- | ----- | ----- |
| GIRO, fee 0 | 230.64 | 52 | 230.64 |
| Saved sequences, fee 0 | 158.71 | 42 | 158.71 |
| GIRO, fee 5 | 230.64 | 52 | 490.64 |
| Saved sequences, fee 5 | 183.71 | 30 | 333.71 |

Costs are synthetic tariff units. The two saved solutions have different trip sequences and station paths, so 42 versus 30 starts is not a controlled fee-only effect. Fee 5 stops after 120 seconds with a 0.6504% gap. This is **charging reoptimization of saved trip sequences, not fresh CG**. It restricts each gap's charging to one tariff hour. Deadheads use a static reference; platform blocking, FIFO and crew rules remain unchecked. The other 35 buses are excluded. Optimizing only the original charging windows gives no material saving; that is a narrower fixed-duty benchmark. [Figure, solver logs and exact assumptions](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/cleanup_physics/README.md).

## 

## **21 September Fee-only comparison completed**

Holding each assignment, station path and physics fixed, all nine fee pairs give 8–22 fewer starts when the fee changes from 0 to 5\. Electricity cost rises, while total cost at a common fee of 5 falls by 24.2–73.3 synthetic units. All 18 witnesses pass independent validation; one requires a separately recorded 6 ms charge extension. Six searches retain 0.75–3.83% gaps. This is restricted fixed-path charging optimization, not fresh CG. [Nine-row table, bounds and logs](https://github.com/ndandnd/EVSP-DR/blob/d2a25746d160e0c181eca3c9ed09069f23fbeede/outputs/research_management_20260921/charging_fee_factorial/RESULTS.md); the figure is in the Figures tab.

## **Historical context and sources**

**16–17 September:** the baseline audit recorded 67 closed and 35 open event-model fleet gaps among 102 LP endpoints. Sequential searches reached k32 in all six chains after longer searches and repeats; this was not full GIRO physics. **18–20 September:** witness-route proofs, open k15 bounds and the pricing pilot. Today's entries supersede their stale running statuses; source artifacts remain intact.

Local evidence root: /Users/nadan/Documents/projects/demandresponse/outputs/week\_20260921/evidence/

Unicorn C1 proof: /home/nc437/ladder-lite/review\_witness\_columns\_20260917/results/c1\_k08/control/391804\_r0/gurobi.log

[Log walkthrough](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/evidence/LOG_EXCERPTS.md) · [Source hashes](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/evidence/source_manifest.json) · [This week's slides](https://docs.google.com/presentation/d/1F6udjgkiPH51vMUcku7ZT3PhZPAxsQwUCi3TK9vFRDM/edit)

[F1–F9 verdicts and execution ledger](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/README.md) · [Chain results with numerical lower bounds](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/audited_chain_results.csv) · [Independent review](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/REVIEW.md)

[Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) · [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) · [Historical research log](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow)

## 

## **Figures for 21 September**

**Capacity shortcut comparison.** Left: best integer fleet in each saved pool. Right: pricing iterations completed. Four-hour budgets; only k1 with the shortcut certified CG, in 26.9 minutes. All other CG runs timed out. These capacity tests use 240 kWh / 240 kW and zero reserve.

![][image1]

**One bus from the matched k5 comparison.** Blue bars are service trips; orange segments are charging. The lower panels show battery energy, with the 15% reserve dashed. The trip IDs are source IDs, sorted by departure time, not newly assigned route-order labels. The original / fee 0 / fee 5 example buses have 9 / 9 / 6 charging starts. All five buses were jointly checked for the modeled charger counts; the picture displays one representative bus from each solution.

![][image2]

The corrected schedules use the matched charging physics listed above. They reoptimize charging on saved trip sequences; they are not fresh full-CG solutions under all GIRO constraints. [Open full-resolution figure](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/cleanup_physics/one_bus_k5_joint_matched.png).
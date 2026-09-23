[22 Sep: fleet timings and sparsity](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lt33xg84cn65) · [Route columns, capacity and tests](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ikbgt85cdszz) · [22 Sep completed tests and queue evidence](https://github.com/ndandnd/EVSP-DR/tree/codex/week-evidence-20260921/outputs/research_management_20260922/monitor_20260922T075504Z)

# **EVSP DR research journal**

Week of 21 September 2026\. The strongest result is that a good fractional solution can leave an inadequate integer route pool. In 24 matched cases, both methods certify the event-grid LP. Their one-hour MIPs reach the minimum baseline fleet in 6/24 fresh cases versus 24/24 sequential cases. New exact time-and-travel certificates establish these fleet minima; charging optimality and full GIRO compliance do not follow. Sequential growth adds complete GIRO duties. New experiments test whether integer-directed pricing closes that gap without known sequential routes.  
[Paper results preview: five figures, editable tables and source checks](https://github.com/ndandnd/EVSP-DR/blob/e07c877f0adc23bb8d64b1db1c3d104b0ceefc19/outputs/research_management_20260921/paper_results/RESULTS_PREVIEW.md). The preview counts accumulated sequential work and separates LP certificates, pool integer proofs and physical validation. [Today’s experiments and monitoring plan](https://github.com/ndandnd/EVSP-DR/blob/e07c877f0adc23bb8d64b1db1c3d104b0ceefc19/outputs/research_management_20260921/MANAGER_PLAN.md).  
 

## **21 September LP agreement and battery audit**

**Fresh versus sequential:** all 24 paired LP endpoints (six chains, k=5,8,10,15) agree within 0.0000010617 cost units. Both certify the event-grid LP at reduced-cost tolerance 0.0001. Their integer pools differ. [New per-chain figures](https://github.com/ndandnd/EVSP-DR/blob/3f2228605e0c9e1cc46140a3dbd03fd6c1f6e3ce/outputs/research_followup_20260921/chain_comparison/README.md) show accumulated sequential CG time, fresh time, integer buses above target, and magnified LP costs.

**3.93 GiB means computer memory.** Battery energy is measured in kWh. Reducing capacity and initial charge alone makes some saved schedules infeasible:

| Capacity-only scenario | Failed route occurrences | Affected saved fleets |
| ----- | ----- | ----- |
| 240 kWh control | 0 / 487 | 0 / 48 |
| 239.01 kWh throughout | 40 / 487 | 29 / 48 |
| 236.44 kWh throughout | 195 / 487 | 46 / 48 |

These frozen-schedule tests are not actual vehicle assignments. Follow-up: all 195 failures at 236.44 kWh were repaired with unchanged trips and bus counts. 194 fit the original intervals; one needs 0.150 seconds more. All 487 occurrences pass independent replay. This is a continuous capacity-only repair under 240 kW and zero reserve, not a new event-grid optimality proof. See the “21 Sep — chains and pricing” tab for all six figures, actual MIP times, repair results and the 7/8 versus 0/8 pricing experiment. [Replay, source hashes and limits](https://github.com/ndandnd/EVSP-DR/blob/3f2228605e0c9e1cc46140a3dbd03fd6c1f6e3ce/outputs/research_followup_20260921/battery_rounding/README.md).

**Duty 13309, whole day:** the same C6 k5 input provides a fee0 route sharing all 22 original trips and a fee5 route sharing 11\. Displayed charging starts are 4 / 7 / 1\. Both counterparts pass the capacity-only 239.01 kWh check; their historical zero-reserve charging physics still differ from GIRO. [Full-day figures and exact itineraries](https://github.com/ndandnd/EVSP-DR/blob/3f2228605e0c9e1cc46140a3dbd03fd6c1f6e3ce/outputs/research_followup_20260921/duty13309/README.md).

## **21 September Extending all six chains to 40**

All six baseline chains have found 32-bus schedules after longer MIPs and seed repeats. The original one-hour searches reached largest targets 26, 28, 31, 29, 26 and 28\. These are largest matches, not a claim that every smaller target matched. At k32 all six CGs stopped at four hours without a pricing certificate.

Now submitted: **48 cases, k33–40 on each chain**, with the full preceding column pool, 4 h CG and 1 h two-stage MIP per case. The 44 graph builds are shared only for identical inputs.

**22 September, 20:01 EDT:** 42/44 graphs are ready. Thirteen jobs run: two baseline graphs, six baseline CGs, four baseline MIPs, and the strict cached CG. All 52 pending baseline dependencies are genuine; no failed allocation or broken dependency needs recovery.

[Figure guide:](https://github.com/ndandnd/EVSP-DR/blob/codex/week-evidence-20260921/outputs/research_management_20260922/figure_explanations/README.md) pool MIP selects saved routes; integer-directed pricing creates complementary routes. Heuristic fleet solutions are upper bounds. The guide also explains the 512-route limit and packed graphs.

[Independent Opus review, checked:](https://github.com/ndandnd/EVSP-DR/blob/codex/week-evidence-20260921/outputs/independent_review_20260922_opus55/ASSESSMENT.md) direct trip connections are capped at 57-minute gaps, and station bridges require positive charging. The [follow-up and exact certificates](https://github.com/ndandnd/EVSP-DR/blob/codex/week-evidence-20260921/outputs/independent_review_20260922_opus55/followup_response/README.md) now establish all 24 benchmark fleet minima. C1/C3 k15 still require 15 buses without the 57-minute cap. Charging optimality and broader operating constraints remain separate.

| Chain, k=33 | Trips | CG minutes | Fractional buses\* | Integer buses / pool bound |
| :---- | :---- | :---- | :---- | :---- |
| C1 | 785 | 239.3 | 33 | 36 / 33 — open |
| C2 | 787 | 239.9 | 32 | 36 / 32 — open |
| C3 | 770 | 239.3 | 33 | 34 / 33 — open |
| C4 | 785 | 239.1 | 32 | 38 / 32 — open |
| C5 | 768 | 239.7 | 32 | 39 / 32 — open |
| C6 | 796 | 239.9 | 32 | 38 / 32 — open |

\* Fractional buses is the final restricted-master route weight. All six CGs hit the four-hour limit without a pricing certificate; these are not certified LP lower bounds. Separate graph builds took 9.56–15.10 h. CG minutes exclude earlier smaller instances.

All six completed MIPs missed 33 and reached both time limits. Native route replay passes baseline physics, but duplicate removal and shared capacity remain unvalidated. Extra trip assignments: C1 315, C2 225, C3 118, C4 285, C5 397, C6 182\.

Latest larger endpoints: at k35, C1/C2/C3/C4 found 42/37/36/42 buses, with pool bounds 34/34/35/34. C6 k34 remains 40 buses with bound 33; its k35 MIP is running. New C1/C3/C4 k36 CGs stopped at four hours with fractional route weights 35/36/35, without pricing certificates; their MIPs are running. C2/C6 k35 CGs also stopped without certificates, each with route weight 34\. All completed MIPs remain open after both stage limits. C2 k33’s saved positive-only LP support differs from its scalar objective by 0.038 cost units; the audit retains this numerical discrepancy.

[Exact LP values, time definitions, Gurobi logs, source hashes and audit](https://github.com/ndandnd/EVSP-DR/tree/codex/week-evidence-20260921/outputs/research_management_20260922/monitor_20260922T155736Z/operations); [earlier C3 result](https://github.com/ndandnd/EVSP-DR/tree/codex/week-evidence-20260921/outputs/research_management_20260922/monitor_20260922T115605Z/operations).

[22 September, 17:31 EDT: new endpoints, full Gurobi logs and 423 passing audit checks](https://github.com/ndandnd/EVSP-DR/tree/codex/week-evidence-20260921/outputs/research_management_20260922/monitor_20260922T195842Z/operations)

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

The pricing pilot recovered eight buses in **3/4 final MIPs**, versus nine in controls. C1's dive had already found eight, but its final MIP did not receive that incumbent. Follow-up 628441 transferred it and recovered eight in 29.6 additional minutes, with the same distinct routes. The original benchmark remains 3/4. **The corrected replication is complete: 7/8 treatment runs recover eight buses, versus 0/8 controls.** These are four selected cases with two seeds each. C1, C3 and C5 succeed at both seeds; C4 succeeds once and retains nine with bound eight in the other run. Every hit proves eight within its augmented pool and accepts its own generated incumbent; no known sequential routes were imported. Hits take 14.1–35.4 minutes end-to-end. One validates exactly-once coverage; six still contain 1–7 extra assignments. Shared charger capacity is omitted. The nominal shared allowance is 3,600 seconds, with setup/replay overhead separate and observed solver overshoot at most 4.71 seconds. [Complete paired table, independent checks and Gurobi proof lines](https://github.com/ndandnd/EVSP-DR/blob/1c0266cf17cdcda3db9ff98e2b247f1dd72a515c/outputs/research_management_20260921/monitor_20260921T235438Z/integer_audit/README.md). The automatic k15 gate failed at a Slurm accounting query before validation. That validation passed and launched the six k15 jobs, which are now terminal.

k15 follow-up, checked 21 September at 23:43 EDT: C3’s new-pricing treatment found 15 buses and proved 15 optimal within its augmented pool; its unchanged-pool control ended at 17, with bound 15\. Charging optimization remains open at a 10.93% gap. C1/C5 controls ended at 19/16, both with bound 15\. Their treatments stopped before the final MIP: an LP time limit in C1 and a numerical feasibility check in C5. Update, 22 September 01:22 EDT: supplemental MIPs 728184/728185 completed. Both passed native preparation with zero rejected or repaired columns. C1 finished at 18 buses versus control 19; C5 at 17 versus control 16\. Both bounds remain 15 and neither reaches the target. These use the remaining original solver allowance and are supplemental recoveries, not replacements for the failed paired trials. The original software stops remain separate from these recoveries. C3 passes route replay but retains 12 extra trip assignments; shared capacity is unvalidated under the historical 240/240, zero-reserve model.

[k15 endpoints, failure diagnostics and full Gurobi logs](https://github.com/ndandnd/EVSP-DR/blob/codex/week-evidence-20260921/outputs/research_management_20260921/operations/finalcheck_20260922/README.md)

 [Recovery and submission records](https://github.com/ndandnd/EVSP-DR/blob/1c0266cf17cdcda3db9ff98e2b247f1dd72a515c/outputs/research_management_20260921/monitor_20260921T235438Z/k15_recovery/README.md). [Design, tests, completed results and full logs](https://github.com/ndandnd/EVSP-DR/blob/09ca22ae28d7e92026dcb36f402b87589f507d12/outputs/research_management_20260921/integer_columns/README.md). [Follow-up log](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/evidence/c1_followup/628441_r0/gurobi.log).

## **21 September Harder physics and duplicate service**

The capacity shortcut completed 134 versus 6 pricing iterations at k2, and 51 versus 2 at k3. The strict k15 run instead spent its budget constructing 260.7 million arcs; its 37-bus initial-pool result tells us little about the model. k16 timed out at 211.9 GiB peak memory.

The packed recovery is complete. Its label k16 is the parent prefix: this subgroup contains 256 trips from nine GIRO duties. Graph construction took 115.4 minutes within the four-hour budget; CG then completed 2,453 iterations at 3.93 GiB peak memory. Fractional route weight is 9, but pricing did not certify convergence. The saved pool provably needs 10 integer buses. Its selected routes cover every trip, with 57 extra assignments, and exceed shared charger limits at two stations. Capacity was omitted from this arm, so this is an important remaining model constraint, not a solver violation. The next strict CG has now published its endpoint: 277 trips from ten subgroup duties, 5,236 iterations and 8,343 columns. Its fractional route weight is ten; weighted RMP objective is 1,000,391.890434. The four-hour deadline stopped CG without a pricing certificate, so this is not a certified full-model lower bound. Graph construction took 73.62 minutes within that budget. Its one-hour MIP is now complete: 11 buses with bound 10, so the ten-duty target remains unmatched and the fleet optimum is unresolved. Both MIP stages hit their time limits. The selected cover has 73 extra trip assignments and exceeds the omitted capacity limits at 7880C (three simultaneous buses versus one charger) and JON\_A (four versus one). The eleven-duty continuation (parent prefix k19, 331 trips) spent 271.55 minutes building its graph, exhausting the included four-hour CG allowance before any pricing iteration. Its inherited-plus-singleton RMP has route weight 64 and no pricing certificate; its MIP finished with 65 buses and finite-pool bound 64, both stages at their time limits. It selects all 54 new singletons plus 11 inherited routes. The cover has 73 extra assignments and fails the omitted charger-capacity checks. This is an unpriced initial-pool result, not a lower bound or infeasibility claim for the full model. The graph-save/reload fix passes 13 local and 13 Unicorn tests. A separate native check now reproduces all 8,343 inherited records and validates all 54 new singleton routes: job 768638 completed in 35 seconds. This validates the saved initial pool, not a new CG result. The full 331-trip graph save/reload test now passes: 8,397 initial routes reproduce exactly, two diagnostic pricing queries agree, and native reload takes 15.96 seconds. Job 772820 took 3 h 10 min, with 6.08 GiB scheduler peak memory. This validates graph reuse, not CG optimality. Graph preparation is accounted for separately from the next four-hour CG run. 

[22 September, 20:01 EDT: new CG endpoints, C2 k35 MIP and cached-CG startup evidence](https://github.com/ndandnd/EVSP-DR/tree/codex/week-evidence-20260921/outputs/research_management_20260922/monitor_20260922T235958Z/operations)

Recovery job 824877 was submitted at 17:43 EDT with 8 CPUs, 16 GiB and a five-hour allocation. At 20:01 EDT it was running, restart 0, after the native source/input/cache checks and Gurobi license probe passed. Restricted-master solves are progressing; there is no final CG result or pricing certificate yet. Full-run memory remains unmeasured. [Full graph validation and cached-CG recovery](https://github.com/ndandnd/EVSP-DR/tree/codex/week-evidence-20260921/outputs/research_management_20260922/monitor_20260922T195842Z/strict_review). [Complete pool check and native logs](https://github.com/ndandnd/EVSP-DR/tree/codex/week-evidence-20260921/outputs/research_management_20260922/strict_graph_reuse/production_gate). [Code, native tests and next gates](https://github.com/ndandnd/EVSP-DR/tree/codex/week-evidence-20260921/outputs/research_management_20260922/strict_graph_reuse/native_preflight). [Audited MIP endpoint and full logs](https://github.com/ndandnd/EVSP-DR/blob/codex/week-evidence-20260921/outputs/research_management_20260921/monitor_20260922T035403Z/strict_k17/README.md).  [Verified endpoint, pool hash and stopping reason](https://github.com/ndandnd/EVSP-DR/blob/4132ba69db4ca8f058df0d9db8094363a90734bb/outputs/research_management_20260921/operations/heartbeat_20260921T235509Z/k17_endpoint_summary.json). [Final results, physical checks and full logs](https://github.com/ndandnd/EVSP-DR/blob/b7efdd948a8d0f20ca730fbb4da4a8e40e9d83a5/outputs/research_management_20260921/operations/README.md).

[Completed graph benchmark and replay checks](https://github.com/ndandnd/EVSP-DR/blob/4c8e4c005ec92259d520b1baa990b39ed4a851bf/outputs/week_20260921/capacity_strict/recovery/benchmark/646674_r0/result.json) — five fixed-dual checks on 26 trips: build 121.11 to 41.81 seconds, peak memory 2.06 to 0.147 GiB. This is a same-node benchmark; the larger recovery used different stopping behavior and hardware. **New: the 53- and 90-trip benchmarks also pass.** Packed versus original explicit storage cuts graph-build time by factors 2.84 and 2.58, and process peak memory from 9.12 to 0.31 GiB and 26.19 to 0.64 GiB. All five tested reduced costs agree; all fifteen generated routes per case pass replay. Mean fixed-dual pricing falls from 261/520 seconds to 0.267/0.246 seconds. These are same-node implementation tests, not end-to-end CG speedups or capacity-aware pricing results; shared capacity is omitted. [Exact benchmark table, source hashes and limits](https://github.com/ndandnd/EVSP-DR/blob/1c0266cf17cdcda3db9ff98e2b247f1dd72a515c/outputs/research_management_20260921/operations/heartbeat_20260921T235509Z/README.md).

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
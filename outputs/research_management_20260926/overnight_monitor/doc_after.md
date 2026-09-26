# **EVSP DR current results**

## Updated 26 September 2026\. Sequential growth reliably recovers the baseline fleet on the 24-case benchmark. The extension to target 40 is now complete and remains above target. Matching all GIRO operating constraints at scale is still open.

## **1 Fleet recovery and computation**

## Six chains, four targets per chain. Fresh starts generate routes from scratch; sequential starts retain routes generated for smaller trip sets.

| Benchmark result | Fresh | Sequential |
| :---- | :---- | :---- |
| **Certified event-grid LPs** | **24 / 24** | **24 / 24** |
| **Target 5 matched** | **5 / 6** | **6 / 6** |
| **Target 8 matched** | **1 / 6** | **6 / 6** |
| **Target 10 matched** | **0 / 6** | **6 / 6** |
| **Target 15 matched** | **0 / 6** | **6 / 6** |
| **Median final fleet-search time** | **30.0 min** | **5.31 s** |

## The paired weighted LP objectives agree within 0.0000011 cost units. Pricing certifies the event-grid LP at tolerance 0.0001. Exact time-and-travel lower bounds establish the minimum baseline covering fleets. These statements do not certify charging optimality or exactly-once dispatch.

## **Time definitions.** Fleet search is the final integer bus-count optimization, not pricing. Its allowance is 30 minutes; charging optimization uses the remaining one-hour MIP budget, with fleet capped by the first-stage incumbent. Sequential CG time includes all smaller-instance CG runs. Graph construction, earlier MIPs and queue waits are separate. See [Fleet times and matrix](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lt33xg84cn65) for all six chains.

## **Extension through target 40\.** All 48 cases at targets 33–40 finished. None matched its target; none certified the CG LP or closed the finite-pool fleet gap. The final target-40 results are:

| Chain | Trips | Integer buses | Final LP route weight | Pool fleet bound |
| :---- | :---- | :---- | :---- | :---- |
| **C1** | **948** | **43** | **39.000** | **39.000** |
| **C2** | **947** | **44** | **39.000** | **39.000** |
| **C3** | **947** | **44** | **39.000** | **39.000** |
| **C4** | **948** | **47** | **39.000** | **39.000** |
| **C5** | **946** | **44** | **39.000** | **39.000** |
| **C6** | **947** | **45** | **39.000** | **39.000** |

## Values are rounded. Each case had four hours of CG and a one-hour MIP, with graph building separate. LP route weight is the sum of fractional route variables; here it is **not a certified full-model lower bound**. The pool bound applies only to saved routes. Individual routes pass replay, but duplicate removal and shared charging capacity are unvalidated. Earlier longer and repeated MIPs matched target 32 on all six chains. The target-40 inputs form three service-day variants, not one identical set. [All 48 endpoints and sources](https://github.com/ndandnd/EVSP-DR/blob/codex/research-maintenance-20260925/outputs/research_management_20260925/cluster/README.md).

## **2 Why better columns matter**

## Four fresh target-8 pools provably require nine buses. Adding eight known sequential routes to each restores eight. Most audited witness routes have positive reduced cost: they help assemble whole buses without improving the fractional objective.

## Integer-directed pricing temporarily commits to routes and prices the remaining coverage. An earlier four-case, two-seed comparison reached eight buses in **7/8 treatments versus 0/8 controls**. It used a supplied fleet cap of eight and passed its own generated incumbent to the MIP; it did not import the sequential solution. This is a heuristic, not complete branch-and-price.

## **New result:** all four C1 cap/stopping variants finished. Their 15-minute dives found no integer incumbent; each subsequent 45-minute MIP ended at nine buses with bound eight. Incumbent-transfer comparisons were skipped because there was no incumbent. These different settings do not overturn the earlier result, but do show that success is not yet reliable. At target 15, all twelve older 12-hour fleet searches also missed, ending at 16–19 buses with bound 15\. [Pilot results](https://github.com/ndandnd/EVSP-DR/blob/codex/research-maintenance-20260925/outputs/research_management_20260925/cluster/README.md).

## **3 Charging cost on matched assignments**

## Reoptimize charging on original GIRO trip assignments and on saved CG-derived assignments. Compare the complete five-bus, 62-trip fleet, not just a pictured bus.

| Price peak | Start fee | GIRO assignments | CG assignments | CG cost change |
| :---- | :---- | :---- | :---- | :---- |
| **08:00** | **0** | **174.52** | **158.71** | **−9.1%** |
| **08:00** | **5** | **369.64** | **333.71** | **−9.7%** |
| **12:00** | **0** | **243.83** | **234.29** | **−3.9%** |
| **12:00** | **5** | **438.19** | **397.47** | **−9.3%** |
| **18:00** | **0** | **159.38** | **163.15** | **\+2.4%** |
| **18:00** | **5** | **349.57** | **313.06** | **−10.4%** |

## Costs are electricity plus start fees in synthetic price units. Physics are matched: 236.44 kWh, 15% reserve, PARX 60 kW, nonlinear charging, minimum three-minute sessions, modeled charger counts and matched return-energy floors. All 18 factorial witnesses pass whole-fleet checks.

## These are charging MIPs on fixed station paths, **not fresh CG under all those constraints**; some searches retain gaps. The selected assignments differ between fees. The evening zero-fee result is more expensive, so we cannot claim universal improvement. Price-overlay examples retain recorded GIRO alongside both reoptimized schedules in [Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i).

## **4 What remains difficult**

## **Baseline versus stricter physics.** The chain benchmark uses 240 kWh / 240 kW, zero reserve, no return-energy floor or shared charger limit, flat prices and a start fee of five. Its fleet success does not establish full GIRO compliance.

## **Capacity and memory.** The capacity-pricing shortcut improves the tested target-1/2/3 saved-pool fleets from 1/3/21 to 1/2/5; only one CG run certifies convergence. Packed graph storage and cache reuse substantially improve measured memory/startup costs, but those benchmarks omit shared capacity. The latest cached 331-trip CG still stops without a pricing certificate. Larger capacity-constrained scheduling remains open.

## **5 Current work and next decisions**

## 26 September, 04:25 EDT: all 240 original jobs in the 80-case spatial-price expansion have ended. CG certified 61/80 event-grid LPs. Fresh pools produced five-bus selections in 55/80 cases; the other 25 used a labelled fallback that adds optimized GIRO trip-sequence columns. Cleanup produced 79 five-bus schedules with exactly-once trip coverage and individual-route replay checks: 54 from fresh selections and 25 from fallbacks. Recovery job 520378 is running for the remaining cleanup; it changes only the source-pinned duplicate limit from ten to eleven.

## These spatial-price tests use 240 kWh batteries, 350 kW charging, zero reserve, an aggregate return-energy floor and no shared charger limits. The 79 cleanup fleet proofs apply only to their finite repair pools; charging cost is proved within 60 pools and remains open in 19\. MIX cohorts allow vehicle-group mixing forbidden by GIRO. Next: recover the remaining cleanup, then compare costs and select examples under the recorded selection rules. Earlier original-campaign recoveries also finished; their results remain separate.

## **Figures and examples to keep**

## [Six-chain timing and matrix tables](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lt33xg84cn65) · [Price overlays and route graphs](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) · [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly).

## Duty 13414 illustrates the matched charging comparison. Duty 13323 gives the more active midday route and node graph; it is a historical example with unequal return energy and unchecked fleet capacity, not evidence for matched cost savings. Original figures and captions remain below and in their existing tabs.

## **History and source links**

## 16–17 September: 67 closed and 35 open event-model fleet gaps among 102 audited LP endpoints. 18–23 September: missing-column proofs, pricing trials and matched charging comparisons. 25 September: extension to target 40 completed and current summary consolidated. [Preserved earlier summary](https://github.com/ndandnd/EVSP-DR/blob/codex/research-maintenance-20260925/outputs/research_management_20260925/document/current_before.docx) · [Dated cluster evidence](https://github.com/ndandnd/EVSP-DR/blob/codex/research-maintenance-20260925/outputs/research_management_20260925/cluster/README.md).

## [Log walkthrough](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/evidence/LOG_EXCERPTS.md) · [Source hashes](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/evidence/source_manifest.json) · [This week’s slides](https://docs.google.com/presentation/d/1F6udjgkiPH51vMUcku7ZT3PhZPAxsQwUCi3TK9vFRDM/edit)

## [F1–F9 verdicts and execution ledger](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/README.md) · [Chain results with numerical lower bounds](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/audited_chain_results.csv) · [Independent review](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/REVIEW.md) · [Historical research log](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow)

## 

## **Figures for 21 September**

**Capacity shortcut comparison.** Left: best integer fleet in each saved pool. Right: pricing iterations completed. Four-hour budgets; only k1 with the shortcut certified CG, in 26.9 minutes. All other CG runs timed out. These capacity tests use 240 kWh / 240 kW and zero reserve.

![][image1]

**One bus from the matched k5 comparison.** Blue bars are service trips; orange segments are charging. The lower panels show battery energy, with the 15% reserve dashed. The trip IDs are source IDs, sorted by departure time, not newly assigned route-order labels. The original / fee 0 / fee 5 example buses have 9 / 9 / 6 charging starts. All five buses were jointly checked for the modeled charger counts; the picture displays one representative bus from each solution.

![][image2]

The corrected schedules use the matched charging physics listed above. They reoptimize charging on saved trip sequences; they are not fresh full-CG solutions under all GIRO constraints. [Open full-resolution figure](https://github.com/ndandnd/EVSP-DR/blob/ca1dda82247db995ffd0d524e045430ea197ec13/outputs/week_20260921/cleanup_physics/one_bus_k5_joint_matched.png).

## 

## **Completed extension through target 40**

Each cell shows the final integer bus count; darker cells mean more buses above target. All 48 CG runs reached their time limit without a pricing certificate, and all finite-pool fleet gaps remain open. These are baseline covering results, not validated full-GIRO dispatch schedules.

[Editable results and source hashes](https://github.com/ndandnd/EVSP-DR/blob/codex/research-maintenance-20260925/outputs/research_management_20260925/figures/full40_fleet_results.csv)

![][image3]
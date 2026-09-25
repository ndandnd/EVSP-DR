# EVSP–DR: four research questions

Prepared 25 September from saved evidence. Live endpoints are supplied separately.

## 1. Does sequential growth improve the integer solution?

Yes, in the **24 matched cases** (six chains, k=5/8/10/15): fresh and sequential CG certify the conservative event-grid LP at reduced-cost tolerance 0.0001; weighted objectives differ by at most 0.0000010617. Their one-hour two-stage MIPs reach the minimum baseline covering fleet in **6/24 versus 24/24**. Exact time-and-travel certificates establish those fleet minima; charging optimality and exactly-once dispatch do not follow. Sequential growth adds complete GIRO duties, making the grouping reference-informed, without importing GIRO route columns.

Median final fleet-search time is **30.001 minutes fresh versus 5.31 seconds sequential**; median total MIP time remains about one hour because charging optimization continues. Sequential CG time includes every ancestor CG; earlier MIPs, graph construction and queue waits are separate. Historical hardware/code differ, so timings are descriptive. Full40 preserves three input classes—C1/C4, C2/C3/C6, C5—not one identical input across all chains.

Baseline: 240 kWh/240 kW, zero reserve, flat tariff, start fee 5, 2.5 kWh/5-minute grid, no terminal floor/shared capacity. Covering means each trip receives **at least one** assignment, not exactly one. Weighted objective = 100,000 × fractional route weight + charging-related cost; these quantities and fleet-only bounds are distinct.

## 2. Why generate columns directed toward an integer cover?

An LP can spread eight units of route weight across 80–101 fractional routes without supplying a compatible eight-route selection. Across five audited k8 cases, 39/40 sequential witness trip sets are absent from fresh pools and 37/40 have positive final reduced cost. Improving the LP therefore need not discover the routes that complete a whole-bus cover.

The corrected replication recovers eight buses in **7/8 treatments versus 0/8 controls**, across four selected cases and two seeds; successful end-to-end times are 14.1–35.4 minutes. Each hit proves eight within its augmented pool and matches the exact baseline fleet floor. Treatment combines new columns with its own dive incumbent; no known sequential/GIRO cover is imported. Historical `fleet_cap=8` was explicitly supplied, and first-feasible stopping depends on that cap. This is bounded heuristic fixing/pricing, not complete branch-and-price; cap, stopping and incumbent contributions require the separate pilot. Six successful covers retain duplicate service; shared capacity is omitted.

## 3. Do saved CG assignments improve matched charging costs?

Charging is reoptimized for the same five buses/62 exactly-once trips, on original GIRO assignments (A) or saved CG-derived assignments (B).

| Tariff peak | Fee | A cost | B cost | B change |
|---|---:|---:|---:|---:|
| 08:00 | 0 | 174.52 | 158.71 | −9.1% |
| 08:00 | 5 | 369.64 | 333.71 | −9.7% |
| 12:00 | 0 | 243.83 | 234.29 | −3.9% |
| 12:00 | 5 | 438.19 | 397.47 | −9.3% |
| 18:00 | 0 | 159.38 | 163.15 | +2.4% |
| 18:00 | 5 | 349.57 | 313.06 | −10.4% |

Costs include electricity/start fees in synthetic units. Matched physics: 236.44 kWh, 15% reserve, PARX 60 kW, nonlinear charging, ≥3-minute sessions, shared charger counts and matched return-energy floors. All 18 factorial witnesses pass independent whole-fleet checks. These are fixed-station-path charging MIPs; some time out. B differs between fee 0 and fee 5, so these selected rows do not isolate fee effects. Evening fee 0 is dearer; universal improvement and fresh-CG optimality are unestablished.

## 4. What still blocks stricter-physics conclusions?

Fresh routing/charging under the matched physics, complete dispatch validation and scalable capacity-aware pricing remain open. The capacity shortcut improves tested saved-pool fleets from 1/3/21 to 1/2/5 under four-hour CG budgets; only k1-on certifies pricing. The strict cached 331-trip endpoint saves 10,620 columns, route weight 11 and weighted RMP 1,100,465.822, but stops at its pricing deadline: no full-model bound, pool-MIP proof or whole-fleet physical certificate follows. Packed no-capacity benchmarks cannot establish capacity-aware performance.

## Retained figures and editable captions

1. [Chain 1: CG, LP and final-MIP clocks](/Users/nadan/Documents/projects/demandresponse/outputs/research_followup_20260921/chain_comparison_mip_times/chain1_charts.png). **Caption:** Accumulated sequential CG and final-target MIP are different clocks. LPs coincide; pool bounds are not error bars. Companion C2–C6 charts remain retained.
2. [Capacity shortcut comparison](/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/capacity_strict/capacity_on_off_plot.png). **Caption:** More pricing iterations yield better finite pools. All six MIPs prove their pool fleets; only k1-on certifies CG. Shared-capacity checks pass, but k2-off/k3-on retain 1/6 extra assignments.
3. [08:00 price overlay, both fees](/Users/nadan/Documents/projects/demandresponse/outputs/coauthor_followup_20260923/price_overlay_revision/comparison/duty13414_peak08_fee0_fee5_combined_six_panel.png). **Caption:** Recorded GIRO and reoptimized GIRO/CG assignments, with the exact hourly tariff. Duty 13414 illustrates schedules; costs above concern all five buses.
4. [18:00 price overlay, fee 0](/Users/nadan/Documents/projects/demandresponse/outputs/coauthor_followup_20260923/price_overlay_revision/comparison/duty13414_peak18_fee0_price_overlay.png). **Caption:** The saved CG-assignment fleet costs 2.4% more here. Fixed station paths and matched physical validation do not establish fresh-CG optimality.
5. [Duty 13323 activity graphs](/Users/nadan/Documents/projects/demandresponse/outputs/coauthor_followup_20260923/price_overlay_revision/duty13323_graph/duty13323_side_by_side.png). **Caption:** Saved route retains eight trips and adds nine, including 91 midday service minutes. Historical physics differ: return energy 0.977 versus 107.902 kWh; parent cover has 17 overcovered trips, capacity unchecked. Schematics and reconstructed depot links are not road/dispatch evidence or causal charging comparisons.

## Primary evidence

- [24-pair LP data](/Users/nadan/Documents/projects/demandresponse/outputs/research_followup_20260921/chain_comparison/README.md), [actual MIP clocks](/Users/nadan/Documents/projects/demandresponse/outputs/research_followup_20260921/chain_comparison_mip_times/README.md), [exact fleet certificates](/Users/nadan/Documents/projects/demandresponse/outputs/independent_review_20260922_opus55/followup_response/README.md).
- [Pricing mechanism and cap provenance](/Users/nadan/Documents/projects/demandresponse/outputs/research_followup_20260921/integer_pricing_explanation/README.md), [eight paired proof logs](/Users/nadan/Documents/projects/demandresponse/outputs/research_followup_20260921/integer_pricing_explanation/proof_links.md).
- [18-cell charging audit, validation paths and hashes](/Users/nadan/Documents/projects/demandresponse/outputs/coauthor_followup_20260923/AB_AUDIT.md), [capacity results](/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/capacity_strict/README.md), [cached endpoint/hash](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260923/dive_cap_pilot/MEETING_EDIT_NOTES.md).
- [Price-overlay provenance](/Users/nadan/Documents/projects/demandresponse/outputs/coauthor_followup_20260923/price_overlay_revision/comparison/figure_manifest.json), [13323 provenance, itinerary and unequal-physics scope](/Users/nadan/Documents/projects/demandresponse/outputs/coauthor_followup_20260923/price_overlay_revision/duty13323_graph/README.md).

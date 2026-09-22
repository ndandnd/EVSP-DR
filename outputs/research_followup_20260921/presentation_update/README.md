# Doc and Slides publication — 21 September 2026 follow-up

The user's request to add these results to both artifacts authorizes this bounded update. No solver, new cluster allocation, shared-link permission change or historical-tab deletion is part of this publication.

## Live results

- [Doc: 21 Sep — chains and pricing](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.fycml6be3fks): a new six-page tab with three native editable tables and all six chain figures.
- [Weekly Slides: new comparison section](https://docs.google.com/presentation/d/1F6udjgkiPH51vMUcku7ZT3PhZPAxsQwUCi3TK9vFRDM/edit?slide=id.g3f8ecbc5488_6_160#slide=id.g3f8ecbc5488_6_160): slides 15–28, with an updated summary on slide 2. Text, mathematics, captions and the paired-results table remain native editable elements; chart axes and legends remain in the scientific figures.

| Slides | Content |
|---|---|
| 15 | Computation accounting and LP agreement |
| 16–21 | Chains 1–6: cumulative CG, integer fleet, weighted LP, and measured fleet/charging/total MIP times |
| 22 | Actual Chain 1 MIP times and all-case medians |
| 23 | Capacity-only repair of all 195 failures at 236.44 kWh |
| 24 | Integer-directed pricing: algorithm and intuition |
| 25–26 | Covering-LP gap identity and conditional pricing mathematics |
| 27 | Native editable table: 7/8 treatment hits versus 0/8 controls |
| 28 | Experiment scope, time accounting and evidence links |

The existing Current Research tab changed in exactly one paragraph: “charging repair was not tested” is superseded by the completed repair result and a link-by-tab-name to this follow-up. Its frozen-schedule failure table remains as the before-repair record. Existing images and source footer are preserved. Figure, CG-curve and history tabs were not edited. Slides 1–14 retain their content except the intentional slide 2 summary update; earlier pilots remain labeled as historical context.

## Evidence and interpretation

The data, code and source manifests were published before the presentation at commit `6fcc672b16db3255ec552294ff0f34a1b5c9290e` on `codex/week-evidence-20260921`. All new slide notes and the Doc source links point to those pinned artifacts.

- [Battery repairs](../battery_repair/README.md): all 487 route occurrences pass continuous replay; 194 failing occurrences repair inside their original intervals and one needs 0.1499625 seconds more at the same station. All 48 selected fleets retain their trips and bus counts. This changes battery/initial energy only under historical 240 kW, zero-reserve, no-shared-capacity physics. It is not a new event-grid or full-GIRO certificate.
- [Timing and all six figures](../chain_comparison_mip_times/README.md): sequential cumulative CG includes ancestors and excludes earlier MIPs. MIP charts show the final displayed k only. All 24 LP pairs agree within 1.0617e-6 weighted cost units. Median fleet-search time is 30.001 minutes fresh versus 5.31 seconds sequential, but both median total MIP times remain near an hour because charging search continues. First-target timestamps were not retained; historical hardware/code differ.
- [Integer pricing and proof logs](../integer_pricing_explanation/README.md): the treatment adds conditional pricing and its own discovered incumbent start. It reaches eight buses in seven of eight trials across four selected instances and two seeds; unchanged-pool controls reach eight in none. It is a bounded heuristic, not exhaustive branch-and-price, and no columns-only ablation is claimed. Shared capacity is absent and duplicate service remains in six successful covers.

## Before/after verification

The `verification/` folder preserves before/after Slides exports, before/after Current Research exports and final follow-up exports. `verify_exports.py` and `verification.json` record structural checks, source-figure matching, preservation checks and source hashes. Native Google exports downsample the chart images; normalized pixel comparison checks that each image is the correct chain rather than requiring identical binary hashes. Final PDFs were visually reviewed after rendering. The final terminology correction spells out “positive reduced-cost” on slides 25–26.

Only final exports and verification records are published. Intermediate drafting screenshots are excluded. No experiment claim is inferred from scheduler state, and the newer k15 continuation is not folded into the completed k8 paired result.

# **EVSP DR Current Research**

Updated 12 September 2026, 00:20 EDT. Results below are completed observations; overnight extensions are still running.

**Current finding.** With set covering and inherited columns, all six chains match the GIRO fleet at k=3–6. Chains 3, 5 and 6 also reach ten buses at k=10. Larger warm-chain cases remain unfinished.

## **Integer fleet with inherited columns**

Each cell is the number of integer bus routes found. Green matches the target. Red requires extra buses. Grey means no completed result.

| Target buses | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 2 | 2 | 2 | 2 | 3 | 2 |
| 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| 7 | — | 7 | 7 | 7 | 7 | 7 |
| 8 | — | 8 | 8 | 8 | 8 | 8 |
| 9 | — | — | 9 | 9 | 9 | 9 |
| 10 | — | — | 10 | — | 10 | 10 |

**k=11–15:** no completed warm results yet. The overnight extension is queued through k=15 for all six chains.

k=2 starts from single-trip routes; later sizes reuse validated previous-k columns. Chain 3 k=10 also includes routes from a fresh solver solution. Displayed warm fleet optima are proved within their saved pools. Individual route replay passed; duplicate-trip removal and shared charger capacity are not established by this table.

## 

## **Fresh starts on the same chains**

Each size starts independently. Both tables use covering and 240-kWh batteries with 240-kW charging. Fresh fleet proof status varies by cell.

| Target buses | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 5 | 5 | 5 | 5 | 5 | 6 | 5 |
| 8 | 9 | 9 | 9 | 9 | 9 | 8 |
| 10 | 11 | 11 | 11 | 11 | 11 | 11 |
| 15 | 18 | 17 | 18 | 19 | 16 | 19 |

Inherited columns improve the available integer pool: chain 3 at k=8 improves from a proved nine-bus pool optimum to eight; chain 5 at k=5 improves from a proved six-bus pool optimum to five.

## **What is limiting progress**

**Baseline warm runs:** replaying every inherited route exhausted initialization budgets in chain 1 k=7, chain 2 k=9 and chain 4 k=10. The new treatment replays at most 512 sequences for 15 minutes, then starts ordinary CG. Compare it separately from full-pool inheritance.

**Capacity pilot:** pricing dominates. One k=3 pricing call took 7.15 hours; its LP took 0.006 seconds. CG stopped without a pricing certificate.

| Case | Baseline | PARX 60 kW | Station capacity | Both |
| ----- | ----- | ----- | ----- | ----- |
| k=1, two duties tested | 1 each | 1 each | 1 each | 1 each |
| k=2, 23 trips | 2 | 2 | 3 | 3 |
| k=3, 35 trips | 3 | 3 | 16 | 16 |

Sixteen is the result in a small generated pool, not a proved physical requirement. Capacity-constrained schedules passed station sweeps. This pilot has no 65% return-SOC floor and is separate from the six baseline chains.

## 

## **Running overnight**

* 37 warm CG cases continue six chains to k=15.

* 40 independent CG cases split one 32-duty, 750-trip parent into four groups of eight, using ten different partitions.

* 10 recombination cases retain the component solution and try routes across groups.

Each CG case has a dependent one-hour, two-stage MIP on the default partition. At the check, 46 CG jobs were running. No completed result from this new campaign is shown above.

## **Methods and evidence**

Each CG route costs 100,000 \+ electricity \+ 5 per charging start.  MIP stage 1 minimizes buses. Stage 2 constrains buses ≤ the validated stage-1 incumbent and minimizes electricity plus start fees.

A pricing certificate establishes the LP result in the represented graph at its stated tolerance. An unfinished restricted LP objective is not a full-model lower bound. A MIP proof concerns its saved columns. Fleet target attainment, physical replay and shared-capacity validation are separate checks.

[Experiment register and exact source records](https://github.com/ndandnd/EVSP-DR/tree/codex/parallel-research-20260911/outputs/research_register) · [Overnight experiment settings](https://github.com/ndandnd/EVSP-DR/tree/codex/parallel-research-20260911/outputs/overnight_extension_20260912) · [Historical document and figures](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit)

**Storage maintenance, 12 September:** 156 cold files were archived losslessly, freeing 133.74 GB. Active research and V2G data were preserved. [Verification and restore instructions](https://github.com/ndandnd/EVSP-DR/tree/b1129bf4/outputs/storage_cleanup_20260912).
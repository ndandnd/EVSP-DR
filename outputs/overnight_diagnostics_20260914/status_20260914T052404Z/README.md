# Overnight diagnostic results — 14 September, 01:26 EDT

## First completed comparison: chain 5, target five buses

| Column selection | CG minutes | Certified weighted LP objective | Columns in MIP pool | Integer buses | Fleet proved in pool? |
|---|---:|---:|---:|---:|---|
| Original: 30 by reduced cost | 8.0 | 500,276.192 | 14,175 | 6 | yes |
| 200 by reduced cost | 26.1 | 500,276.192 | 45,105 | 6 | yes |
| 30 complementary columns | 21.0 | 500,276.192 | 12,029 | 5 | yes |

All three runs reach the same weighted LP objective to numerical precision, with fractional route count five. Their integer pools differ. In these runs, the complementary selection supports five buses while the other two pools provably require six. These are different column sets, not nested pools: 45,105 columns need not contain the useful routes in the 12,029-column pool.

This is one selected difficult case. It demonstrates that an LP pricing certificate and a large column count do not guarantee an integer target solution in the saved pool. It does not establish a general success rate or full-model integer optimality. Both new treatments took longer CG time than the original in this case; hardware variation limits direct timing attribution.

The input, baseline physics, objective, execution code, cumulative CG allowance and final one-hour MIP allowance are held fixed for these treatment comparisons. The methods use covering, 240 kWh / 240 kW, flat prices and a charging-start fee of five. Shared charger capacity and a terminal-SOC floor are absent. Individual-route replay passes; other physical checks remain separate.

[Exact results and source hashes](chain5_k5_comparison.csv).

## Chain 4 k=19 continuation

The separate continuation reached its pricing certificate after 267.0 cumulative CG minutes: 27.9 beyond the original run. Its weighted LP objective improved by only 0.000332. The original four-hour endpoint remains uncertified; this later certificate belongs to a distinct longer-budget treatment. Its final MIP is pending. This is a certificate at the stated reduced-cost tolerance within the tested graph.

Collected diagnostic endpoints: 15 CG runs (15 certified) and 2 MIPs. Other cases remain pending or running; missing results are not failures.

[All collected endpoints](endpoints.csv).

# Overnight diagnostic results — 14 September, 14:59 EDT

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

## All cases with a completed treatment MIP

| Chain / target | Original buses | 200-column buses | Complementary buses |
|---|---:|---:|---:|
| C1 / 8 | 9 | 9 | 9 |
| C1 / 10 | 11 | 11 | 12 |
| C1 / 15 | 18 | 18 | 18 |
| C2 / 8 | 9 | 9 | 9 |
| C2 / 10 | 12 | 11 | 12 |
| C2 / 15 | 17 | 19 | 19 |
| C3 / 8 | 9 | 8 | 8 |
| C3 / 10 | 11 | 11 | 11 |
| C3 / 15 | 18 | 17 | 17 |
| C4 / 8 | 9 | 9 | 9 |
| C4 / 10 | 11 | 12 | 11 |
| C4 / 15 | 19 | 20 | 19 |
| C5 / 5 | 6 | 6 | 5 |
| C5 / 8 | 9 | 9 | 9 |
| C5 / 10 | 11 | 11 | 11 |
| C5 / 15 | 16 | 17 | 17 |
| C6 / 10 | 11 | 11 | 12 |
| C6 / 15 | 20 | 19 | 19 |

Numbers are integer incumbents, not all optimal fleets. Pending means no published MIP result in this collection. See the CSV for each fleet bound, proof, timing and source hash. Results arriving early are not a random sample of the batch. A worse incumbent does not prove that the new pool lacks the earlier fleet.

[Complete comparison with proof scopes](column_selection_comparison.csv).

## Chain 4 k=19 continuation

The separate continuation reached its pricing certificate after 267.0 cumulative CG minutes: 27.9 beyond the original run. Its weighted LP objective improved by only 0.000331. The original four-hour endpoint remains uncertified; this later certificate belongs to a distinct longer-budget treatment. Its final MIP is collected separately. This is a certificate at the stated reduced-cost tolerance within the tested graph.

The continuation MIP found 20 buses with pool fleet bound 19; fleet proved: False. The original pool's separate MIP found 19 and proved it. The new MIP used its ordinary greedy_pool_partition initializer with 181 buses, accepted by Gurobi; it did not inherit the previous 19-bus integer incumbent. This new timed incumbent is not evidence that the continued pool cannot support 19. Membership of the earlier selected routes in the continued pool has not been audited. No full-model integer conclusion follows.

## Chain 5 k=19 continuation

Certified after 314.0 cumulative CG minutes, 74.7 additional minutes. The weighted LP objective improved by 0.032497. Its final MIP is collected separately. The original four-hour endpoint remains uncertified.

The continuation MIP found 19 buses with pool fleet bound 19; fleet proved within this pool: True; individual-route replay: True. The original four-hour treatment had found 20/bound19. This is a result for the separate longer-CG treatment, not a revision of the original attempt.

## Longer MIPs on unchanged pools

All nine selected inherited pools recover their target; none of the fourteen selected fresh pools does. Three fresh pools now prove that an extra bus is necessary. Seven inherited-pool reruns finish their fleet proof within 30 minutes, so longer elapsed search alone does not explain all recoveries. All ordered MIP pools and input hashes match the original runs. [Full comparison and limits of interpretation](LONGER_MIP_RESULTS.md).

Collected diagnostic endpoints: 38 CG runs (38 certified) and 61 MIPs. Other cases remain pending or running; missing results are not failures.

[All collected endpoints](endpoints.csv).

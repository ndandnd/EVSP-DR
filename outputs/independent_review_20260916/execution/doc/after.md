# **EVSP–DR: current research**

Independent-review checks: 16 September 2026\. Results below distinguish feasible schedules, pricing certificates, and proofs within saved route pools. New experiments are running; their outcomes remain open.

## **1a. How far does sequential CG \+ MIP recover GIRO’s fleet?**

**Target 32 is matched in chains 2, 4, 5 and 6\.** Chains 1 and 3 have matched 31\. Each step adds one GIRO duty’s trips and retains previous columns, so this experiment uses known duty grouping.

| Target / result | C1 | C2 | C3 | C4 | C5 | C6 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 31 / original one-hour MIP | 34 | 32 | 31 | 40 | 36 | 32 |
| 31 / best with completed longer searches | 31 | 31 | 31 | 37 | 31 | 31 |
| 32 / original one-hour MIP | 35 | 34 | 33 | 34 | 37 | 33 |
| 32 / best with completed longer searches | 35 | 32 | 33 | 32 | 32 | 32 |
| 32 / numerical event-model fleet lower bound | 32 | 32 | 32 | 31 | 31 | 32 |

Original one-hour MIPs allocated **30 minutes to fleet search**, then the remaining time to charging. Longer runs allocated three hours to fleet search. Separately generated pools and limited searches need not produce monotone results.

**F1 — verified:** all 128 audited schedules pass physical replay. Each passenger trip is assigned once; other copies remain as empty driving, with unchanged timing, energy and cost. This validates those dispatch schedules under the baseline assumptions; it does not insert new exactly-once columns into the saved MIP.

**F3 — verified with a corrected derivation:** numerical lower bounds were recovered for all 52 uncertified CG endpoints. The target-32 lower bounds above match the best fleets in C2 and C6. They concern the event-route model, rely on floating-point solver results, and do not prove the continuous problem. [Math and qualifications](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/f3/README.md).

**Baseline:** 240 kWh battery; 240 kW at every modeled charger, including PARX; fee 5 per charging start; flat tariff; set covering; no reserve, shared charger limit or ending-SOC floor. These are not all GIRO constraints.

## **1b. Does a fresh start catch up with the accumulated CG time?**

| Target | Warm matches | Fresh matches |
| :---- | :---- | :---- |
| 5 | 6/6 | 5/6 |
| 8 | 6/6 | 1/6 |
| 10 | 6/6 | 0/6 |
| 15 | 6/6 | 0/6 |

**F5 — verified distinction:** all 24 fresh CG runs converged to the same weighted LP values as their warm comparisons. Their integer recovery differed under the original 30-minute fleet searches. Four small fresh pools exclude the target; the other 14 misses are unresolved searches. This does not establish that fresh CG cannot catch up.

**Now testing:** six fresh k15 pools × three MIP seeds × three hours of fleet search; repeated warm k32 searches; and a sequential chain whose intermediate groups are randomized trips, rather than GIRO duties. Report every seed, including failures to match.

**F2:** selected routes are more similar to GIRO duties when fractional route weight equals k: mean nearest-duty Jaccard 0.644 versus 0.462 when it equals k−1. That association is verified; the proposed explanation that inheritance merely reassembles GIRO is unresolved. [Overlap results and figure](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/f2/README.md).

## **2\. Which algorithm changes helped?**

Summer work replaced coarse/capped pricing with event-based time/SOC graphs and exact shortest-path pricing on that graph. Recent selected-input tests support indexed route replay, avoiding unused LP-matrix construction, and retaining the full inherited pool. In the three inheritance tests, final fleets improved 9→8, 11→10 and 17→15. These are limited controlled comparisons, not a historical or universal speedup claim. Capacity-pricing tests remain time-capped without pricing certificates. [Algorithm evidence](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/controlled_comparison_20260913/status_20260913T063519Z/README.md).

## **3\. Without a start fee, does changing duties improve charging cost?**

**F6 — verified on one selected five-duty instance:** all three arms below were audited. Costs are tariff cost units. Original GIRO prices are intervals because power within each recorded charging window is unknown.

| Tariff peak | GIRO as-is | Same trips, optimized charging | Fresh CG \+ trip assignment |
| :---- | :---- | :---- | :---- |
| 08:00 | 230.29–230.98 | 128.29 | 124.69 |
| 12:00 | 289.59–290.60 | 164.23 | 160.61 |
| 18:00 | 223.45–223.72 | 95.29 | 88.30 |

All use five buses. Charging starts (original / fixed trips / CG) are 52/46/44, 52/43/46 and 52/42/44. Both optimized arms contain charges shorter than three minutes, and some buses approach zero SOC. Their total ending energies match each other; GIRO’s original total is slightly lower. Settings here are 240 kWh / 350 kW, fee zero, no reserve or shared capacity. Do not merge this with the 240 kW scaling experiment or claim global charging optimality. [Three-arm costs, start counts and per-bus SOC](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/f6/README.md).

**F6/F7 — now testing:** all six k15 instances, three synthetic peaks and one real-price day, using fresh tariff-aware CG and fixed-duty charging under matched 240 kW settings with a three-minute minimum and no start fee. Real-price analysis remains internal under the data provider’s terms. Larger-instance charging gains are not yet established.

## **4\. Do these results survive stricter physics and larger inputs?**

**F4 — corrected:** zero of 42 unchanged GIRO charging schedules pass the 240 kW replay; after optimizing only charging, all 42 fixed trip sequences are feasible in the continuous baseline. The earlier “12 of 40 fail” screen used 350 kW. A feasible continuous schedule is not a certificate that the event grid represents it.

New tests cover chain 5 through target 31 with 60 kW PARX charging, 15% reserve and separate vehicle groups; a fresh full Partille selection of 40 duties / 948 trips; and a Frölunda ladder. The strict chain still omits some GIRO rules, and changes several assumptions together. The full Partille graph is prepared separately before its 48-hour CG allowance. **First Frölunda results (F8):** k1/k2 use 1/2 buses for 15/38 trips; CG certified in 2.85/51.64 seconds. Both dispatches pass exactly-once physical replay. For k2, overlapping mandatory trips independently prove that two buses are necessary. These are initial cases, not a larger-instance generalization claim.

**Next decision:** distinguish MIP search limits from inadequate column pools; measure charging gains on larger inputs; and check which gains survive the stated physical constraints. New jobs are experiments, not conclusions.

[F1–F9 verdicts and execution ledger](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/README.md) · [Chain results with numerical lower bounds](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/audited_chain_results.csv) · [Independent review](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/REVIEW.md)

[Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) · [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) · [Historical research log](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow)
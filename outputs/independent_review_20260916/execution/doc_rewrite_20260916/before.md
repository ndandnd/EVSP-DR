# **EVSP–DR: current research**

Independent-review checks: 16 September 2026\. Results below distinguish feasible schedules, pricing certificates, and proofs within saved route pools. New experiments are running; their outcomes remain open.

## **1a. How far does sequential CG \+ MIP recover GIRO’s fleet?**

**Across six chains and targets 16–32 (102 cases), the numerical event-model fleet lower bound equals GIRO’s count in 93 cases; it is one bus lower in nine.** The original searches match that lower bound in 51 cases. The other 51 fleet-optimality questions remain open—not only nine.

| Search evidence | Cases matching GIRO’s fleet | Cases matching the numerical fleet lower bound | Cases with an open fleet gap |
| :---- | :---- | :---- | :---- |
| Original one-hour MIPs | 51 / 102 | 51 / 102 | 51 / 102 |
| Including 26 audited longer searches | 70 / 102 | 67 / 102 | 35 / 102 |

The original one-hour MIPs gave fleet search 30 minutes; the remaining time optimized charging. Longer searches gave fleet search three hours. The second row uses the best audited result for each case, not an equal-budget comparison. It excludes the new seed repeats still running.

**What “optimal” means here:** a physically replayed integer schedule meets a numerical lower bound for the discretized event-route model. These are floating-point certificates, not exact-arithmetic proofs or proofs for all continuous charging schedules. The number is an **integer-fleet lower bound**; the weighted LP objective includes electricity and start fees. Fractional route weight is a separate quantity, even though it also equals k in the same 93 cases.

**The nine special cases:** their lower bound is k−1. Three now match GIRO’s k buses, but none has reached k−1; optimality remains open in all nine. Among the other 93 cases, 26 remain open after the audited longer searches.

**Largest matched targets:** 31, 32, 31, 32, 32 and 32 for chains 1–6. These are largest observed matches, not guarantees that every smaller target matched. Each step adds one GIRO duty’s trips and retains previous columns.

**F1 / F3 evidence:** all 128 original and longer-run dispatches pass replay after assigning each passenger trip once and retaining other traversals as empty driving. Numerical bounds were reconstructed for the 52 uncertified CG endpoints. [Recomputed counts and source hashes](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/advisor_followup_20260916/proof_counts/counts.json) · [Bound derivation and qualifications](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/f3/README.md).

**Baseline:** 240 kWh battery; 240 kW at all modeled chargers, including PARX; fee 5 per charging start; flat tariff; set covering; no reserve, shared charger limit or ending-SOC floor. These are not all GIRO constraints.

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

**New F6 reserve test, 16 September 23:00 UTC:** with a 36 kWh reserve and three-minute minimum charging, fixed-duty schedules have finished at five buses for peaks 08:00 and 18:00, costing 157.96 and 107.19. Both pass physical replay. The three fresh-CG arms are still running, so no new joint-versus-fixed saving is established. These runs retain the earlier 350 kW setting and common ending-energy minimum; they are a separate treatment from the table above.

## **4\. Do these results survive stricter physics and larger inputs?**

**Single-factor timing pilot (16 September):** among the same 20 sampled pool trip sequences, all 20 remain feasible with PARX charging at 60 kW and with either smaller battery. A 15% reserve retains 19; separating vehicle groups retains 17 and excludes three mixed-group sequences. There were no unknown outcomes. These are fixed-sequence event-model results, **not fleet-size tests or full-pool success rates**. Each follow-up took 43–52 seconds. Full replay remains unsubmitted. [Pilot table and validation](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/advisor_sequence_20260916/single_factor_pilot/RESULTS.md).

**F4 — corrected:** zero of 42 unchanged GIRO charging schedules pass the 240 kW replay; after optimizing only charging, all 42 fixed trip sequences are feasible in the continuous baseline. The earlier “12 of 40 fail” screen used 350 kW. A feasible continuous schedule is not a certificate that the event grid represents it.

New tests cover chain 5 through target 31 with 60 kW PARX charging, 15% reserve and separate vehicle groups; a fresh full Partille selection of 40 duties / 948 trips; and a Frölunda ladder. The strict chain still omits some GIRO rules, and changes several assumptions together. The full Partille graph is prepared separately before its 48-hour CG allowance. **First Frölunda results (F8):** k1/k2 use 1/2 buses for 15/38 trips; CG certified in 2.85/51.64 seconds. Both dispatches pass exactly-once physical replay. For k2, overlapping mandatory trips independently prove that two buses are necessary. These are initial cases, not a larger-instance generalization claim.

**Next decision:** distinguish MIP search limits from inadequate column pools; measure charging gains on larger inputs; and check which gains survive the stated physical constraints. New jobs are experiments, not conclusions.

[F1–F9 verdicts and execution ledger](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/README.md) · [Chain results with numerical lower bounds](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/audited_chain_results.csv) · [Independent review](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/REVIEW.md)

[Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) · [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) · [Historical research log](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow)
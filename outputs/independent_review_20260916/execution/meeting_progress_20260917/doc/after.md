# **EVSP–DR: where the research stands**

*Updated from the 16 September 2026, 23:44 EDT collection. The 128/102/67/35 audit counts remain frozen; new seed results are reported separately.*

*New proof: [Time-only fleet bounds for all 102 instances](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/time_only_vsp_20260916/README.md) · [Full-Partille job and memory rationale](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/full40_12h_scaglione_20260916/README.md).*

## **In one paragraph**

We optimize electric-bus schedules and charging for Transdev's Partille network (GIRO data) with column generation (CG) and a MIP, under time-varying electricity prices. **Fleet size:** in 93 of 102 test instances, our lower bound equals GIRO's own bus count, so GIRO's fleet is optimal under the continuous baseline there; we have matched that bound with a valid schedule in 67 of 102 while 35 event-grid fleet gaps remain open. In 9 instances the bound is one bus *below* GIRO — the only place our method could beat GIRO on fleet — and none has been closed yet. **Charging cost:** on one 5-duty instance, re-optimizing charging alone cuts GIRO's electricity bill by \~44%; also re-optimizing which bus does which trip saves a further 2–7%. Both numbers were obtained without GIRO's 15% reserve and 3-minute charging rules, and a rerun with those rules is in progress. **Next 48 hours** decide three things: whether the 1-bus savings are real, whether the charging gain survives realistic rules, and whether the warm-start method matters or the MIP just needed more time.

## **Question 1 — Can we recover GIRO's fleet, and can we beat it?**

**Latest sequential reach: all six chains now have a 32-bus result at target 32, including selected longer searches and seed repeats. Original one-hour MIPs reached largest targets 26/28/31/29/26/28. These are largest observed matches, not success at every smaller k. New seed results pass individual-route replay; their separate dispatch-conversion audit is pending. The 128/102/67/35 table below is the frozen earlier audit, not a refreshed total. Mixed-model one-bus savings remain unproven.**

|  | Count |
| ----- | ----- |
| Instances (6 chains × k \= 16…32) | 102 |
| Lower bound \= GIRO's k | 93 |
| Lower bound \= k − 1 (model could beat GIRO) | 9 |
| Bound matched by a validated schedule (including 3 h searches) | 67 |
| GIRO fleet count matched (a different test) | 70 |
| Open gap to the lower bound | 35 |
| Open: bound k, best schedule k+1 or worse | 26 |
| Open: bound k−1, best schedule k or worse | 9 |

Every one of the 128 schedules passes physical replay. Duplicate trip copies are driven empty; this leaves fleet and cost unchanged. The bounds are numerical certificates on the discretized model (2.5 kWh / 5 min), not exact-arithmetic proofs.

Important context: all 42 GIRO duties are feasible in our model once their charging is re-planned (0 of 42 are feasible with GIRO's own charging plan, under the baseline 240 kW replay). This establishes a continuous-charging k-bus upper bound. It does not itself establish representability on our event grid; that requires a separate feasible event-grid schedule.

Vehicle-group result: time-only scheduling with deadheading needs exactly GIRO’s fleet in all 102 instances when groups stay separate, for each group individually. With mixing allowed, the time-only minimum is k in 93 cases and k−1 in the same nine as our LP results. Thus group separation rules out those one-bus savings, independently of energy constraints. The electric-bus integer saving with mixing is still open. [All102 saved LP endpoints](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/mixed_group_lp_20260916/README.md).

**Open:** can an integer electric-bus schedule achieve the one-bus saving when vehicle groups may mix? The six single-factor arms also test the effects of depot power, reserve and battery size.

## **Question 2 — Does warm-starting from the previous instance matter?**

**Updated 16 September, 23:44 EDT: sequential construction remains much stronger in the tested comparisons. All 18 fresh-k15 repeats missed the target after 3 hours of fleet search; their integer gaps remain open.**

In the original accumulated-CG-time comparison, warm runs matched 24/24 targets and fresh runs 6/24: 5/6 at k=5, 1/6 at k=8, none at k=10 or 15\. Four fresh pools provably exclude their target; other gaps remain open. All 24 fresh CGs reached their pricing stopping certificate. The completed k15 repeats use 16–19 buses, with a pool bound near 15, across three seeds per chain. This is evidence for the tested sequential method, not proof that fresh pricing can never succeed. Also running: a chain built from *random* trip groups rather than GIRO's duties, to test whether warm-starting helps without knowing GIRO's grouping.

## **Question 3 — What is demand response worth?**

**On one 5-duty instance, charging reoptimization produces most of the savings; fresh-CG-derived duties add 2–7% under relaxed rules. A preliminary morning comparison retains a 1.93% gain with reserve and minimum-charge rules. Full GIRO constraints and larger-size gains remain unestablished.**

| Tariff peak | GIRO as-is | Same duties, optimized charging | Optimized duties \+ charging |
| ----- | ----- | ----- | ----- |
| 08:00 | 230.3–231.0 | 128.3 | 124.7 |
| 12:00 | 289.6–290.6 | 164.2 | 160.6 |
| 18:00 | 223.4–223.7 | 95.3 | 88.3 |

Five buses in every arm; ending energy matched between the two optimized arms. **Caveats that matter:** both optimized arms run batteries to \~0% (GIRO never goes below 19%) and use charges as short as 26 seconds (GIRO's minimum is 3 min). Part of the 44% is spending GIRO's safety margin. Settings: 240 kWh / 350 kW / no start fee / flat depot rate.

**Running:** Stricter k5 check (15% reserve, 3-minute minimum, still 350 kW): morning costs are 157.965 fixed-duty versus 154.921 fresh CG, 1.93% lower. Both serve trips exactly once; fresh CG returns with more total energy. Evening shows a 2.89% lower cost but retains one duplicate trip pending its dispatch check. The noon fixed-duty job timed out before its first worker record. Larger k15 comparisons remain under review.

## **Question 4 — Which algorithm changes helped?**

Event-based time/SOC graphs with exact pricing (summer) give certified LP bounds where the old capped search could not. Indexed route replay: −12 to −18% CG time. Skipping unused LP-matrix construction: −9 to −15%. Inheriting the full column pool instead of 512 routes: −41 to −65% CG time and better fleets (9→8, 11→10, 17→15) at the same LP value. Three inputs, two orders each — directional, not statistical. Capacity-pricing caching: no measurable effect (both arms time out).

## **Our model vs GIRO's reality**

|  | Our baseline | GIRO / Transdev | Status |
| ----- | ----- | ----- | ----- |
| Battery | 240 kWh, one type | 236.4 / 239.0 kWh, two vehicle groups | Negligible |
| Terminal charging | 240 kW constant | 371→120 kW, SOC-dependent | Ours is slower on average |
| Depot charging | 240 kW | **60 kW** | Ours is 4× faster — testing (item 11\) |
| Minimum SOC | none | **15% hard** | Testing (k5 rerun, item 11\) |
| Min charge duration | none | **3 min \+ 45 s setup** | Testing (k5 rerun, item 10\) |
| Vehicle groups | mixed | route-21 blocks separate | Testing (item 11\) |
| Shared charger capacity | unlimited | 1–2 per station, FIFO at 4808 | Not modeled |
| Tariff in scaling runs | flat | — | k=15 tariff runs in progress |

## **Running now**

Times are EDT. For running jobs, Expected gives the Slurm allocation deadline (start \+ wall limit). Sequential projections use the remaining dependency chain at full wall limits, with zero queue delay; they are not scheduled finish times.

| Experiment | Answers | Jobs | Expected |
| ----- | ----- | ----- | ----- |
| Fresh k=15 pools, 3 h fleet search × 3 seeds | Q2: is the 6/24 an artifact of MIP time? | 18 | Complete: 0/18 target matches; 16–19 buses found. |
| Chain 5 k=31, 12 h search on existing pool | Q1: can we find 30 buses where GIRO has 31? | 1 | 17 Sep, 07:22 |
| k=32 repeats × 3 seeds, chains 1/3/4/5 | Q1: how much is seed luck? | 12 | Complete: 12 seed results; new C1/C3 seeds reach 32\. |
| k=5 three-arm with 15% reserve \+ 3-min | Q3: does the DR gain survive real rules? | 6 | 5/6 results; noon fixed-duty startup timed out. |
| k=15 × (3 synthetic \+ 1 real tariff), CG vs fixed-duty | Q3: does the gain hold at scale? | 48 | 17 Sep, 03:04 fixed / 07:04 fresh CG |
| Chain 5 strict physics (60 kW depot, 15%, groups) to k=31 | Q1: are the 9 cases an artifact? | 62 sequential | 21 Sep, 13:24 — full-budget dependency projection |
| Random-trip-group chain to 364 trips | Q2: does warm start need GIRO's grouping? | 42 sequential | 17 Sep, 22:37 — full-budget dependency projection |
| Full Partille, 40 duties / 948 trips, 12 h allocation | Scale | CG 343119: Scaglione, 120 GiB; 11 h 45 min CG; checkpoint fallback tested | Waiting for graph 341404\_0; no scheduled start |
| Frölunda ladder | Generalization | 1 (48 h) | 18 Sep, 18:12 |
| C5 k31 single-factor arms, full-pool initialization | Q1: isolate depot power, reserve, battery size and vehicle groups | 6 arms / 750 replay tasks | Replay started 16 Sep, 20:22; full-pipeline finish not scheduled |

## **What is still needed before writing up**

**Must have**

1\. Q1: group separation now rules out k−1 in all nine cases. Still open: a validated electric-bus schedule attaining k−1 when groups may mix, and sensitivity to the tighter physics. (Items 8, 11.)

2\. Q3 under GIRO's rules: the k=5 three-arm with 15% reserve and 3-min minimum. If the gain collapses, the DR story changes. (Running.)

3\. Q3 at scale: at least k=15, at *equal fleet*, on ≥ 3 chains. Needs the k=15 tariff CGs plus a follow-up MIP with fleet cap \= k. (Running \+ planned.)

4\. Q2: the 3-hour fresh-k15 searches are complete (0/18 matches). Next distinguish missing useful integer routes from unfinished MIP search; all 18 fleet gaps remain open.

5\. One large instance: full Partille certified or bounded. (CG queued after graph preparation.)

**Should have**

6\. Shared charger capacity at ≥ 1 station in the k=15 runs — the one GIRO constraint we have never modeled.

7\. Frölunda beyond k=2.

8\. A real price series in the headline (currently internal-only under license; need a publishable alternative such as Nord Pool day-ahead if terms allow).

**Decisions pending**

\- Cluster: six full single-factor arms are now launched. Other new submissions remain on hold until the fresh-k15, C5 k31, constrained-k5 and k32 seed results are complete.

\- Full-Partille CG 343119 is queued on Scaglione with 120 GiB and a 12 h allocation (11 h 45 min CG plus shutdown allowance), after the existing graph job. Memory sizing uses the fresh single-process run, not the older eight-worker inheritance peaks. The downstream MIP remains held.

## **Sources**

Chain table with bounds: `outputs/independent_review_20260916/execution/audited_chain_results.csv`. Verdicts and evidence: `execution/README.md`. Independent review: `REVIEW.md`. Charging three-arm: `execution/f6/`. GIRO replay: `execution/f4/`. All hashes and job IDs in `execution/ledger.json`.

[F1–F9 verdicts and execution ledger](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/README.md) · [Chain results with numerical lower bounds](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/audited_chain_results.csv) · [Independent review](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/REVIEW.md)

[Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) · [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly) · [Historical research log](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow)
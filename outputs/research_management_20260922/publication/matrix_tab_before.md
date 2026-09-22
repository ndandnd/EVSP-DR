# Fleet times and MIP size

22 September 2026\. Original fresh-versus-sequential comparison: six chains, k \= 5, 8, 10 and 15\. These tables exclude later repeat searches and integer-directed pricing.

## What “fleet-search time” means

CG first builds a pool of feasible routes. Pricing is the search for new routes during CG. After CG stops, the final MIP chooses whole routes from that saved pool.  
Stage 1 — fleet search: minimize the number of buses, up to 30 minutes. The reported time is the optimizer call until it stops, not the first time it finds the target. Stage 2 — charging: constrain buses ≤ the best validated stage-one fleet and minimize electricity plus start fees with the remaining one-hour optimizer budget. For example, an 11-bus incumbent gives a cap of 11, even if 11 is not proved optimal.  
Fleet-search times exclude pricing, master LP solves, setup, charging optimization and earlier sequential MIPs. A short fleet search can still lead to a nearly one-hour total MIP because stage two continues.  
Table key: \* means the fleet stage reached its 30-minute limit; every unstarred result proves its fleet optimum within that saved pool. A pool proof above the target does not prove that the full routing model needs extra buses.

## Chain 1

| Target k | Fresh fleet search | Sequential fleet search | Buses: fresh / sequential |
| :---- | :---- | :---- | :---- |
| 5 | 1.30 s | 1.16 s | 5 / 5 |
| 8 | 26.35 min | 27.15 s | 9 / 8 |
| 10 | 30 min\* | 152.29 s | 11 / 10 |
| 15 | 30 min\* | 105.75 s | 18 / 15 |

## Chain 2

| Target k | Fresh fleet search | Sequential fleet search | Buses: fresh / sequential |
| :---- | :---- | :---- | :---- |
| 5 | 63.23 s | 5.26 s | 5 / 5 |
| 8 | 30 min\* | 8.27 s | 9 / 8 |
| 10 | 30 min\* | 12.63 s | 12 / 10 |
| 15 | 30 min\* | 405.96 s | 17 / 15 |

## Chain 3

| Target k | Fresh fleet search | Sequential fleet search | Buses: fresh / sequential |
| :---- | :---- | :---- | :---- |
| 5 | 3.38 s | 2.16 s | 5 / 5 |
| 8 | 30 min\* | 0.94 s | 9 / 8 |
| 10 | 30 min\* | 3.20 s | 11 / 10 |
| 15 | 30 min\* | 223.71 s | 18 / 15 |

## Chain 4

| Target k | Fresh fleet search | Sequential fleet search | Buses: fresh / sequential |
| :---- | :---- | :---- | :---- |
| 5 | 14.89 s | 0.95 s | 5 / 5 |
| 8 | 14.66 min | 1.94 s | 9 / 8 |
| 10 | 30 min\* | 2.45 s | 11 / 10 |
| 15 | 30 min\* | 178.60 s | 19 / 15 |

## Chain 5

| Target k | Fresh fleet search | Sequential fleet search | Buses: fresh / sequential |
| :---- | :---- | :---- | :---- |
| 5 | 35.88 s | 3.02 s | 6 / 5 |
| 8 | 14.89 min | 2.13 s | 9 / 8 |
| 10 | 30 min\* | 8.02 s | 11 / 10 |
| 15 | 30 min\* | 18.93 s | 16 / 15 |

## Chain 6

| Target k | Fresh fleet search | Sequential fleet search | Buses: fresh / sequential |
| :---- | :---- | :---- | :---- |
| 5 | 0.46 s | 0.42 s | 5 / 5 |
| 8 | 15.77 s | 5.36 s | 8 / 8 |
| 10 | 30 min\* | 1.35 s | 11 / 10 |
| 15 | 30 min\* | 443.97 s | 20 / 15 |

## 

## Summary across four targets per chain

Means and medians include time limits: they measure time spent, not time to optimality. F \= fresh; S \= sequential. Target hits are shown F → S.

| Chain | F mean (min) | S mean (s) | F median (min) | S median (s) | Target hits F → S |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | 21.59 | 71.59 | 28.18 | 66.45 | 1/4 → 4/4 |
| 2 | 22.77 | 108.03 | 30.00 | 10.45 | 1/4 → 4/4 |
| 3 | 22.52 | 57.50 | 30.00 | 2.68 | 1/4 → 4/4 |
| 4 | 18.73 | 45.98 | 22.33 | 2.20 | 1/4 → 4/4 |
| 5 | 18.87 | 8.02 | 22.44 | 5.52 | 0/4 → 4/4 |
| 6 | 15.07 | 112.77 | 15.13 | 3.35 | 2/4 → 4/4 |

## Combined Chains 2–6

| Metric (20 cases per method) | Fresh | Sequential |
| :---- | :---- | :---- |
| Mean fleet-search time spent | 19.59 min | 66.46 s |
| Median fleet-search time spent | 30.00 min | 4.23 s |
| Range | 0.46 s–30.01 min | 0.42 s–7.40 min |
| Targets matched | 5/20 | 20/20 |
| Fleet optima proved within saved pools | 8/20 | 20/20 |
| Fleet time limits | 12/20 | 0/20 |
| Total fleet-search time | 6.53 h | 22.15 min |

Sequential pays extra cumulative CG time before these fast integer solves. Across all six chains, target hits are 6/24 fresh and 24/24 sequential; both median total MIP times are about one hour because charging optimization continues.  
Fresh pool proofs above target: C1 k8 \= 9, C4 k8 \= 9, C5 k5 \= 6, C5 k8 \= 9\. The time-limited misses remain unresolved.  
Baseline: 240 kWh, constant 240 kW, zero reserve, no terminal floor or shared station capacity, flat tariff, start fee 5, covering. Historical code and hardware differ. These tables are descriptive, not a controlled hardware-speedup experiment.  
[Timing tables, exact summaries, source paths and hashes](https://github.com/ndandnd/EVSP-DR/blob/86162955693b08084b6a6e652409c48e1b93bbaa/outputs/research_followup_20260921/chain_comparison_mip_times/fleet_search_tables.md)

## 

## What is the final MIP matrix?

[Full 48-model matrix audit, native Gurobi logs and recommended tests](https://github.com/ndandnd/EVSP-DR/tree/9ea683dc28c2afa85ed30e195ae5adc97c51679c/outputs/research_followup_20260921/mip_matrix_audit)

A has one row per trip and one column per saved route. Aᵢᵣ \= 1 if route r serves trip i, otherwise 0\. Choose xᵣ ∈ {0,1}. Stage one is min ∑ᵣ xᵣ subject to Ax ≥ 1\. In an all-≤ convention this is −Ax ≤ −1.  
Battery and timing feasibility are encoded in admitted routes; this baseline has no shared station-capacity rows. The pricing network’s millions of arcs are not MIP variables. Binary bounds are variable bounds, not extra trip rows.

### Chain 1 — original matrix before Gurobi presolve

| Target / pool | Trip rows | Route columns | Nonzeros | Nonzero % |
| :---- | :---- | :---- | :---- | :---- |
| 5 / fresh | 104 | 28,549 | 596,747 | 20.10% |
| 5 / sequential | 104 | 27,340 | 623,626 | 21.93% |
| 8 / fresh | 194 | 39,940 | 1,092,654 | 14.10% |
| 8 / sequential | 194 | 78,053 | 2,107,909 | 13.92% |
| 10 / fresh | 225 | 50,585 | 1,329,898 | 11.68% |
| 10 / sequential | 225 | 84,209 | 2,237,466 | 11.81% |
| 15 / fresh | 364 | 79,611 | 2,322,459 | 8.01% |
| 15 / sequential | 364 | 130,468 | 3,541,122 | 7.46% |

Density \= nonzeros ÷ (rows × columns). Across all 48 models, 78.1%–93.6% of entries are zero; an average column covers 13.8–29.2 trips. Stage two adds one fleet-cap row and one nonzero per column. For C1 k15 sequential: 365 × 130,468, with 3,671,590 nonzeros.  
Sequential has more columns in 20/24 pairs, yet a faster fleet stage in all 24\. Size alone does not explain performance; the pool’s ability to form good integer covers matters.

## How can we exploit this structure?

Already used: sparse incidence lists, cheapest route per identical trip set, validated MIP starts, Gurobi presolve and automatic sifting for wide LPs. C3 k15 fresh: the root LP takes 1.42 seconds, but fleet search reaches 30 minutes. Sparse linear algebra alone does not resolve that integer search.  
Next tests: measure remaining dominated columns and separable components; compare stronger starts within the identical pool; test MIPFocus settings according to whether finding a fleet or proving it is slow. Keep the promising integer-directed pricing work separate: it changes which columns are available.  
Reduced-cost fixing requires a valid bound and incumbent for the same objective and constraints. Weighted CG reduced costs are not fleet reduced costs. For the simple fleet-cover dual, reduced costs lie between 0 and 1, so the usual test cannot remove columns when the fleet gap is at least one bus. Stage-two fixing must include the fleet-cap dual.  
Sparse 0–1 A is a set-cover matrix, not automatically a network matrix with integral LP solutions. Safe dominance rules must respect charging cost and every side constraint; equal trip sets may cease to be interchangeable after station-capacity rows are added.  
[Gurobi parameter guidance and automatic sifting](https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters/guidelines.html) · [Presolve](https://support.gurobi.com/hc/en-us/articles/360024738352-How-does-presolve-work) · [MIP starts and reduced costs](https://docs.gurobi.com/projects/optimizer/en/12.0/reference/attributes/variable.html)  
Evidence: outputs/research\_followup\_20260921/mip\_matrix\_audit/ — matrix\_dimensions.csv, full Gurobi logs in sources/, exact execution code and 577 passing consistency checks. All 48 endpoints checked; no solver was rerun. Saved solver: Gurobi 12.0.3; current online parameter documentation describes 13.0.
# Chains and integer pricing

21 September 2026\. Sequential CG spends more cumulative computation on the LP, builds better integer pools and reaches every target in these 24 comparisons. Fresh reaches 6/24. Their LP objectives agree to numerical precision.

## Measured computation

We compare k \= 5, 8, 10 and 15 in six chains. Sequential CG time sums every ancestor through k; fresh covers one solve at k. Graph construction, queue time and all MIPs are excluded. MIP times below cover only the final k and split fleet search from charging. They are actual optimizer times, not time limits.  
All 48 CG endpoints have event-grid pricing certificates at tolerance 0.0001. The maximum paired LP difference is 0.0000010617 cost units. The chart subtracts 100,000 × k to expose charging cost; this is neither a violation nor a reduced cost.

| Chain 1 target | Fresh buses / pool bound | Sequential buses / pool bound | Fleet seconds fresh / sequential | Total MIP minutes fresh / sequential |
| :---- | :---- | :---- | :---- | :---- |
| 5 | 5 / 5 | 5 / 5 | 1.30 / 1.16 | 0.058 / 0.049 |
| 8 | 9 / 9 | 8 / 8 | 1580.95 / 27.15 | 59.98 / 59.96 |
| 10 | 11 / 10 | 10 / 10 | 1800.08 / 152.29 | 60.05 / 59.96 |
| 15 | 18 / 15 | 15 / 15 | 1800.22 / 105.75 | 59.96 / 59.93 |

Equality between buses and pool bound proves the fleet within that pool. Across 24 cases, median fleet-search time is 30 minutes fresh and 5.31 seconds sequential. Both median total times are about an hour: sequential often uses the saved fleet-search time on charging. All six fresh k=5 MIPs finish within three minutes. First-target timestamps were not retained.  
Comparison physics: 240 kWh, constant 240 kW, zero reserve, no terminal floor or shared capacity, flat tariff, start fee 5, covering, 2.5 kWh / 5-minute grid. Historical code and hardware differ; timings are descriptive.

## 

## 236.44 kWh repair

| Result | Route occurrences |
| :---- | :---- |
| Already feasible at smaller capacity | 292 |
| Repaired inside original charge intervals | 194 |
| Repaired with a 0.150-second extension | 1 |
| Unresolved | 0 |

All 487 occurrences (463 distinct schedules) pass replay at 236.44 kWh. The 195 unchanged schedules failed, but their trip sequences are repairable. Bus counts, trips, charging stops and start counts stay fixed. Median net added energy per repair is 2.02 kWh; the maximum fleet cost increase is 1.851 synthetic units. The test took 0.47 seconds locally, with no cluster job or CG rerun.  
Conservative grid charging left spare time in the original intervals. We adjust continuous charging amounts within that slack. This capacity-only test retains 240 kW, zero reserve and no shared capacity. It provides feasible schedules, not new grid-model optimality certificates.

## Integer-directed pricing

Starting from the fresh pool with cap K \= 8, fix one fractional route at xᵣ \= 1 and reprice under the new duals. Continue or try alternatives. The final MIP receives generated columns and any self-discovered cover, without known sequential or GIRO routes.  
For covering min cᵀx subject to Ax ≥ 1, x ≥ 0, dual prices satisfy Aᵀπ ≤ c and π ≥ 0\. With r \= c − Aᵀπ, any integer cover y satisfies:  
cᵀy − zLP \= rᵀy \+ πᵀ(Ay − 1).  
In C1, an eight-route diagnostic witness costs 96.272 above the LP: 72.142 of positive reduced costs plus 24.130 of duplicate-coverage dual value. Those routes help form a better integer cover despite their positive reduced costs.  
After fixing F, remaining service is 1 − AF1 and remaining fleet is K − |F|. Pricing uses cᵣ − aᵣᵀπ − μ, where μ ≤ 0 prices the fleet cap. Changed duals can make previously unattractive routes useful. We have not classified every generated route against the original duals.

| k=8 case | New pricing hits | Unchanged-pool hits |
| :---- | :---- | :---- |
| C1 | 2 / 2 | 0 / 2 |
| C3 | 2 / 2 | 0 / 2 |
| C4 | 1 / 2 | 0 / 2 |
| C5 | 2 / 2 | 0 / 2 |

All seven hits finish at eight buses with pool bound eight, in 14.1–35.4 elapsed minutes. Five controls prove nine; the treatment miss remains nine/bound eight. Nominal budget is 3,600 seconds of dive wall time plus final MIP solver time, with MIP setup additional.  
Scope: four selected cases, two seeds each. The treatment combines new columns and its own incumbent start. This bounded heuristic is not exhaustive branch-and-price. Six hits duplicate service; shared capacity remains absent under the historical 240/240 model.

## Figure guide and sources

Integer-panel vertical segments span the pool bound and incumbent.  
[Figures, exact timing tables and source paths](https://github.com/ndandnd/EVSP-DR/tree/6fcc672b16db3255ec552294ff0f34a1b5c9290e/outputs/research_followup_20260921/chain_comparison_mip_times)  
[Battery repairs and independent validation](https://github.com/ndandnd/EVSP-DR/tree/6fcc672b16db3255ec552294ff0f34a1b5c9290e/outputs/research_followup_20260921/battery_repair)  
[Pricing code, mathematics and Gurobi proof logs](https://github.com/ndandnd/EVSP-DR/tree/6fcc672b16db3255ec552294ff0f34a1b5c9290e/outputs/research_followup_20260921/integer_pricing_explanation)

## Chain 1

![][image1]

## Chain 2

![][image2]

## Chain 3

![][image3]

## Chain 4

![][image4]

## Chain 5

![][image5]

## Chain 6

![][image6]